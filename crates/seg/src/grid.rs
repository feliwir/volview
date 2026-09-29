//! Grid operations mirroring TotalSegmentator's preprocessing: nibabel's
//! `as_closest_canonical` and `scipy.ndimage.zoom` (order 0/1, `grid_mode=False`).
//!
//! Arrays are stored with axis 0 fastest: `data[x + nx * (y + ny * z)]`.

use rayon::prelude::*;
use volio::Volume;

/// Axis permutation and flips that take a volume into RAS-canonical order.
#[derive(Debug, Clone)]
pub struct Canonical {
    /// Source axis feeding canonical axis `a` (0 = R, 1 = A, 2 = S).
    pub src_axis: [usize; 3],
    /// Whether canonical axis `a` runs opposite to its source axis.
    pub flip: [bool; 3],
    src_dims: [usize; 3],
    pub dims: [usize; 3],
    pub spacing: [f64; 3],
}

impl Canonical {
    pub fn new(vol: &Volume) -> Self {
        let src_dims = [
            vol.dims.x as usize,
            vol.dims.y as usize,
            vol.dims.z as usize,
        ];
        let src_spacing = vol.spacing.to_array();
        // Direction columns are LPS; RAS flips the first two components.
        let dirs: [[f64; 3]; 3] = [0, 1, 2].map(|s| {
            let d = vol.direction.col(s);
            [-d.x, -d.y, d.z]
        });

        // Greedily pair source axes with world axes by largest |cosine|.
        let mut pairs: Vec<(usize, usize)> =
            (0..3).flat_map(|s| (0..3).map(move |w| (s, w))).collect();
        pairs.sort_by(|a, b| dirs[b.0][b.1].abs().total_cmp(&dirs[a.0][a.1].abs()));
        let (mut src_axis, mut flip) = ([usize::MAX; 3], [false; 3]);
        let mut used = [false; 3];
        for (s, w) in pairs {
            if !used[s] && src_axis[w] == usize::MAX {
                used[s] = true;
                src_axis[w] = s;
                flip[w] = dirs[s][w] < 0.0;
            }
        }
        Self {
            src_axis,
            flip,
            src_dims,
            dims: src_axis.map(|s| src_dims[s]),
            spacing: src_axis.map(|s| src_spacing[s]),
        }
    }

    fn src_index(&self, c: [usize; 3]) -> usize {
        let mut s = [0usize; 3];
        for a in 0..3 {
            s[self.src_axis[a]] = if self.flip[a] {
                self.dims[a] - 1 - c[a]
            } else {
                c[a]
            };
        }
        s[0] + self.src_dims[0] * (s[1] + self.src_dims[1] * s[2])
    }

    pub fn to_canonical<T: Copy + Send + Sync + Default>(&self, src: &[T]) -> Vec<T> {
        let [nx, ny, _] = self.dims;
        let mut out = vec![T::default(); src.len()];
        out.par_chunks_mut(nx * ny)
            .enumerate()
            .for_each(|(z, slab)| {
                for y in 0..ny {
                    for x in 0..nx {
                        slab[x + nx * y] = src[self.src_index([x, y, z])];
                    }
                }
            });
        out
    }

    fn canonical_index(&self, s: [usize; 3]) -> usize {
        let c = [0, 1, 2].map(|a| {
            let v = s[self.src_axis[a]];
            if self.flip[a] {
                self.dims[a] - 1 - v
            } else {
                v
            }
        });
        c[0] + self.dims[0] * (c[1] + self.dims[1] * c[2])
    }

    pub fn from_canonical<T: Copy + Send + Sync + Default>(&self, can: &[T]) -> Vec<T> {
        let [sx, sy, _] = self.src_dims;
        let mut out = vec![T::default(); can.len()];
        out.par_chunks_mut(sx * sy)
            .enumerate()
            .for_each(|(z, slab)| {
                for y in 0..sy {
                    for x in 0..sx {
                        slab[x + sx * y] = can[self.canonical_index([x, y, z])];
                    }
                }
            });
        out
    }
}

/// Output length used by `scipy.ndimage.zoom` for a spacing change (`np.round`, ties to even).
pub fn zoomed_len(n: usize, from_mm: f64, to_mm: f64) -> usize {
    ((n as f64 * (from_mm / to_mm)).round_ties_even() as usize).max(1)
}

/// Input coordinate of output sample `o` with `grid_mode=False` (corner voxels aligned).
fn source_coord(o: usize, n_in: usize, n_out: usize) -> f64 {
    if n_out > 1 {
        o as f64 * ((n_in - 1) as f64 / (n_out - 1) as f64)
    } else {
        0.0
    }
}

/// Per-axis linear taps: (lower index, upper index, upper weight), clamped like `mode="nearest"`.
fn linear_taps(n_in: usize, n_out: usize) -> Vec<(usize, usize, f64)> {
    (0..n_out)
        .map(|o| {
            let c = source_coord(o, n_in, n_out).clamp(0.0, (n_in - 1) as f64);
            let i0 = c.floor() as usize;
            (i0, (i0 + 1).min(n_in - 1), c - i0 as f64)
        })
        .collect()
}

/// Trilinear resample to `out_dims`, then truncate toward zero like `.astype(np.int32)`.
pub fn zoom_linear_trunc(src: &[i16], dims: [usize; 3], out_dims: [usize; 3]) -> Vec<f32> {
    let [tx, ty, tz] = [0, 1, 2].map(|a| linear_taps(dims[a], out_dims[a]));
    let (nx, ny) = (dims[0], dims[1]);
    let at = |x: usize, y: usize, z: usize| src[x + nx * (y + ny * z)] as f64;
    let mut out = vec![0f32; out_dims.iter().product()];
    out.par_chunks_mut(out_dims[0] * out_dims[1])
        .enumerate()
        .for_each(|(oz, slab)| {
            let (z0, z1, wz) = tz[oz];
            for (oy, &(y0, y1, wy)) in ty.iter().enumerate() {
                for (ox, &(x0, x1, wx)) in tx.iter().enumerate() {
                    let lerp = |a: f64, b: f64, t: f64| a * (1.0 - t) + b * t;
                    let plane = |z| {
                        lerp(
                            lerp(at(x0, y0, z), at(x1, y0, z), wx),
                            lerp(at(x0, y1, z), at(x1, y1, z), wx),
                            wy,
                        )
                    };
                    slab[ox + out_dims[0] * oy] = lerp(plane(z0), plane(z1), wz).trunc() as f32;
                }
            }
        });
    out
}

/// Resamples labels by trilinear interpolation of each label's one-hot mask,
/// keeping the label with the largest weight (TotalSegmentator's
/// "higher-order" resampling). Unlike nearest-neighbour it does not turn the
/// coarse prediction grid into visible blocks on the finer output grid.
pub fn zoom_labels(src: &[u8], dims: [usize; 3], out_dims: [usize; 3]) -> Vec<u8> {
    let [tx, ty, tz] = [0, 1, 2].map(|a| linear_taps(dims[a], out_dims[a]));
    let (nx, ny) = (dims[0], dims[1]);
    let mut out = vec![0u8; out_dims.iter().product()];
    out.par_chunks_mut(out_dims[0] * out_dims[1])
        .enumerate()
        .for_each(|(oz, slab)| {
            let (z0, z1, wz) = tz[oz];
            for (oy, &(y0, y1, wy)) in ty.iter().enumerate() {
                for (ox, &(x0, x1, wx)) in tx.iter().enumerate() {
                    // At most 8 distinct labels meet in a cell.
                    let mut labels = [0u8; 8];
                    let mut weights = [0f64; 8];
                    let mut n = 0;
                    for (z, fz) in [(z0, 1.0 - wz), (z1, wz)] {
                        for (y, fy) in [(y0, 1.0 - wy), (y1, wy)] {
                            for (x, fx) in [(x0, 1.0 - wx), (x1, wx)] {
                                let l = src[x + nx * (y + ny * z)];
                                let w = fx * fy * fz;
                                match labels[..n].iter().position(|&k| k == l) {
                                    Some(i) => weights[i] += w,
                                    None => {
                                        labels[n] = l;
                                        weights[n] = w;
                                        n += 1;
                                    }
                                }
                            }
                        }
                    }
                    let best = (0..n).fold(0, |b, i| if weights[i] > weights[b] { i } else { b });
                    slab[ox + out_dims[0] * oy] = labels[best];
                }
            }
        });
    out
}
