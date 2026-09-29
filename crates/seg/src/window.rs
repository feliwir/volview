//! nnU-Net sliding-window inference with Gaussian tile blending.
//!
//! Tiles are visited in z order and logits are accumulated in a buffer one tile
//! deep; slices no later tile can touch are reduced to their argmax and emitted,
//! so memory stays at `classes * tile_depth * ny * nx` instead of the whole volume.

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;

/// Tile start offsets along one axis (`compute_steps_for_sliding_window`).
pub fn steps(image: usize, tile: usize, step_size: f64) -> Vec<usize> {
    let n = ((image - tile) as f64 / (tile as f64 * step_size)).ceil() as usize + 1;
    if n == 1 {
        return vec![0];
    }
    let actual = (image - tile) as f64 / (n - 1) as f64;
    (0..n)
        .map(|i| (actual * i as f64).round_ties_even() as usize)
        .collect()
}

/// 1D factor of nnU-Net's Gaussian importance map (a separable, unnormalised Gaussian).
fn gaussian(n: usize, sigma_scale: f64) -> Vec<f32> {
    let (c, s) = ((n / 2) as f64, n as f64 * sigma_scale);
    (0..n)
        .map(|i| (-(i as f64 - c).powi(2) / (2.0 * s * s)).exp() as f32)
        .collect()
}

pub struct Params {
    /// Tile size along x, y, z.
    pub tile: [usize; 3],
    pub step_size: f64,
    pub sigma_scale: f64,
    /// Input value for padding voxels (the HU that normalises to 0).
    pub pad_value: f32,
    pub num_classes: usize,
}

/// Per voxel: the most and second most likely class, and how many logits the
/// first leads by (blended over tiles like the logits, so it varies smoothly and
/// is zero on the predicted boundary between the two).
#[derive(Debug, Clone, Copy, Default)]
pub struct Prediction {
    pub best: u8,
    pub second: u8,
    pub margin: f32,
}

/// Runs `session` over `img` (dims x, y, z) and calls `emit(z, predictions)` once
/// per z-slice, in increasing z. `on_tile(done, total)` reports progress after
/// each tile.
pub fn predict(
    session: &mut Session,
    img: &[f32],
    dims: [usize; 3],
    p: &Params,
    mut emit: impl FnMut(usize, &[Prediction]),
    on_tile: &mut dyn FnMut(usize, usize),
) -> Result<()> {
    let [tx, ty, tz] = p.tile;
    let padded = [0, 1, 2].map(|a| dims[a].max(p.tile[a]));
    let lo = [0, 1, 2].map(|a| (padded[a] - dims[a]) / 2);
    let [px, py, pz] = padded;
    let [sx, sy, sz] = [0, 1, 2].map(|a| steps(padded[a], p.tile[a], p.step_size));
    let [gx, gy, gz] = [tx, ty, tz].map(|n| gaussian(n, p.sigma_scale));

    let nc = p.num_classes;
    let slab = px * py;
    let class_len = tz * slab;
    let mut acc = vec![0f32; nc * class_len];
    // Sum of tile weights per voxel, to turn blended logits back into logits.
    let mut weights = vec![0f32; class_len];
    let mut base = 0usize;
    let tiles = sx.len() * sy.len() * sz.len();
    let mut done = 0;

    let finalize = |acc: &[f32],
                    weights: &[f32],
                    base: usize,
                    z_end: usize,
                    emit: &mut dyn FnMut(usize, &[Prediction])| {
        let mut out = vec![Prediction::default(); dims[0] * dims[1]];
        for z in base..z_end {
            let Some(uz) = z.checked_sub(lo[2]).filter(|&u| u < dims[2]) else {
                continue;
            };
            let zl = z - base;
            out.par_chunks_mut(dims[0])
                .enumerate()
                .for_each(|(uy, row)| {
                    let off = zl * slab + (uy + lo[1]) * px + lo[0];
                    for (ux, o) in row.iter_mut().enumerate() {
                        let (mut v1, mut v2) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
                        let (mut c1, mut c2) = (0u8, 0u8);
                        for c in 0..nc {
                            let v = acc[c * class_len + off + ux];
                            if v > v1 {
                                (v2, c2) = (v1, c1);
                                (v1, c1) = (v, c as u8);
                            } else if v > v2 {
                                (v2, c2) = (v, c as u8);
                            }
                        }
                        let w = weights[off + ux].max(1e-12);
                        *o = Prediction {
                            best: c1,
                            second: c2,
                            margin: (v1 - v2) / w,
                        };
                    }
                });
            emit(uz, &out);
        }
    };

    for &z0 in &sz {
        if z0 > base {
            finalize(&acc, &weights, base, z0, &mut emit);
            let shift = (z0 - base) * slab;
            acc.par_chunks_mut(class_len)
                .chain(weights.par_chunks_mut(class_len))
                .for_each(|a| {
                    a.copy_within(shift.min(class_len).., 0);
                    a[class_len - shift.min(class_len)..].fill(0.0);
                });
            base = z0;
        }
        for &y0 in &sy {
            for &x0 in &sx {
                let mut input = vec![p.pad_value; tx * ty * tz];
                input
                    .par_chunks_mut(tx * ty)
                    .enumerate()
                    .for_each(|(dz, plane)| {
                        let Some(uz) = (z0 + dz).checked_sub(lo[2]).filter(|&u| u < dims[2]) else {
                            return;
                        };
                        for dy in 0..ty {
                            let Some(uy) = (y0 + dy).checked_sub(lo[1]).filter(|&u| u < dims[1])
                            else {
                                continue;
                            };
                            for dx in 0..tx {
                                if let Some(ux) =
                                    (x0 + dx).checked_sub(lo[0]).filter(|&u| u < dims[0])
                                {
                                    plane[dx + tx * dy] = img[ux + dims[0] * (uy + dims[1] * uz)];
                                }
                            }
                        }
                    });
                let tensor = Tensor::from_array(([1usize, 1, tz, ty, tx], input))?;
                let outputs = session.run(ort::inputs!["hu" => tensor])?;
                let (_, logits) = outputs["logits"]
                    .try_extract_tensor::<f32>()
                    .context("reading logits")?;
                anyhow::ensure!(
                    logits.len() == nc * tx * ty * tz,
                    "unexpected logits size {}",
                    logits.len()
                );

                let tile_len = tx * ty * tz;
                acc.par_chunks_mut(class_len)
                    .enumerate()
                    .for_each(|(c, a)| {
                        let l = &logits[c * tile_len..(c + 1) * tile_len];
                        for dz in 0..tz {
                            for dy in 0..ty {
                                let w_zy = gz[dz] * gy[dy];
                                let dst = (z0 + dz - base) * slab + (y0 + dy) * px + x0;
                                let src = tx * (dy + ty * dz);
                                for dx in 0..tx {
                                    a[dst + dx] += w_zy * gx[dx] * l[src + dx];
                                }
                            }
                        }
                    });
                for dz in 0..tz {
                    for dy in 0..ty {
                        let w_zy = gz[dz] * gy[dy];
                        let dst = (z0 + dz - base) * slab + (y0 + dy) * px + x0;
                        for dx in 0..tx {
                            weights[dst + dx] += w_zy * gx[dx];
                        }
                    }
                }
                done += 1;
                on_tile(done, tiles);
            }
        }
    }
    finalize(&acc, &weights, base, pz, &mut emit);
    Ok(())
}
