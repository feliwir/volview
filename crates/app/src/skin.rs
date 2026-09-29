//! Skin: a thin shell of the body surface, which the segmentation does not
//! label, added as its own label so it can be drawn as translucent context.

use rayon::prelude::*;
use volio::Volume;

/// Label value of the skin shell (TotalSegmentator's labels stay far below).
pub const SKIN: u8 = 255;
/// Air just outside the body, labelled until the skin's statistics are taken
/// (see [`add_skin`]), then merged into [`SKIN`].
pub const SKIN_OUTER: u8 = 254;
/// Dense material outside the body: table, head holder, tubes, cables.
pub const DEVICES: u8 = 253;
/// Dense material this far outside the soft-tissue body is a device (mm).
const DEVICE_MARGIN_MM: f32 = 3.0;
/// How far the skin label reaches into the air (voxels).
const OUTER_VOXELS: usize = 2;
/// Tissue denser than this (HU) belongs to the body, air and lung lighter.
const TISSUE_HU: i16 = -400;
/// Soft tissue (skin, fat, muscle) stays below this (HU); plastics, bone and
/// contrast-filled vessels are denser.
const SOFT_MAX_HU: i16 = 120;

/// Labels unlabelled voxels within `thickness_mm` of the air around the body as
/// [`SKIN`], and the air within two voxels outside it as [`SKIN_OUTER`]: the
/// label then covers the whole air boundary and the CT density places the
/// surface (hybrid mode), not the voxel steps of the label. Returns the number
/// of skin voxels.
///
/// The body is the largest connected component of soft tissue (see
/// [`body_mask`]); outside air is what connects to the border of its axial
/// slice, so the lungs and bowel gas stay inside. Material of any tissue
/// density outside the body (with bone and contrast counted in) is labelled
/// [`DEVICES`].
pub fn add_skin(labels: &mut [u8], vol: &Volume, thickness_mm: f32) -> usize {
    let t = std::time::Instant::now();
    let [nx, ny, nz] = [vol.dims.x, vol.dims.y, vol.dims.z].map(|d| d as usize);
    let dims = [nx, ny, nz];

    // Dense material outside the body (any tissue density): devices.
    let body = body_mask(vol, |hu| hu > TISSUE_HU);
    labels
        .par_iter_mut()
        .zip(body.par_iter())
        .zip(vol.hu.par_iter())
        .for_each(|((l, &b), &hu)| {
            if *l == 0 && !b && hu > TISSUE_HU {
                *l = DEVICES;
            }
        });
    drop(body);
    tracing::debug!(elapsed = ?t.elapsed(), "skin: devices");

    // The skin's body is connected soft tissue only: plastic head holders and
    // tables are denser and would otherwise join where they touch the patient.
    let soft = |hu: i16| hu > TISSUE_HU && hu < SOFT_MAX_HU;
    let body = body_mask(vol, soft);
    let outside = outside_of(&body, dims);
    drop(body);
    tracing::debug!(elapsed = ?t.elapsed(), "skin: outside");

    // Near outside: separable box dilation by the thickness along each axis.
    let spacing = vol.spacing.to_array();
    let radius = [0, 1, 2].map(|a| (thickness_mm as f64 / spacing[a]).round().max(1.0) as usize);
    let outside_u8: Vec<u8> = outside.par_iter().map(|&o| o as u8).collect();
    let near = dilate(&outside_u8, [nx, ny, nz], radius);
    drop(outside_u8);

    let inside: Vec<u8> = outside.par_iter().map(|&o| !o as u8).collect();
    let near_body = dilate(&inside, [nx, ny, nz], [OUTER_VOXELS; 3]);
    // Devices touching the patient (a head holder the head rests on) join the
    // all-tissue body above; beyond a margin around the soft-tissue body they
    // are devices too. The margin keeps teeth in an open mouth.
    let margin =
        [0, 1, 2].map(|a| (DEVICE_MARGIN_MM as f64 / spacing[a]).round().max(1.0) as usize);
    let near_soft = dilate(&inside, [nx, ny, nz], margin);
    drop(inside);
    labels
        .par_iter_mut()
        .zip(outside.par_iter().zip(near_soft.par_iter()))
        .zip(vol.hu.par_iter())
        .for_each(|((l, (&o, &n)), &hu)| {
            if *l == 0 && o && n == 0 && hu > TISSUE_HU {
                *l = DEVICES;
            }
        });
    drop(near_soft);
    tracing::debug!(elapsed = ?t.elapsed(), "skin: dilated");
    let count: usize = labels
        .par_iter_mut()
        .zip(
            near.par_iter()
                .zip(outside.par_iter())
                .zip(near_body.par_iter()),
        )
        .zip(vol.hu.par_iter())
        .map(|((l, ((&n, &o), &b)), &hu)| {
            let skin = *l == 0 && n != 0 && !o && soft(hu);
            if skin {
                *l = SKIN;
            } else if *l == 0 && o && b != 0 && hu <= TISSUE_HU {
                *l = SKIN_OUTER;
            }
            skin as usize
        })
        .sum();
    tracing::info!(voxels = count, elapsed = ?t.elapsed(), "skin shell");
    count
}

/// The body: the largest connected component of voxels whose HU passes
/// `tissue`, found on a grid of 2^3 voxel cells that each must pass throughout,
/// so the table and thin films (the body bag) fall away; at full resolution,
/// passing voxels in or next to a body cell (cells lose up to a voxel at the
/// surface).
fn body_mask(vol: &Volume, tissue: impl Fn(i16) -> bool + Sync) -> Vec<bool> {
    let [nx, ny, nz] = [vol.dims.x, vol.dims.y, vol.dims.z].map(|d| d as usize);
    let tissue = |i: usize| tissue(vol.hu[i]);
    let [cx, cy, cz] = [nx, ny, nz].map(|n| n.div_ceil(2));
    let coarse: Vec<bool> = (0..cx * cy * cz)
        .into_par_iter()
        .map(|c| {
            let (x, y, z) = (2 * (c % cx), 2 * (c / cx % cy), 2 * (c / (cx * cy)));
            (0..8).all(|k| {
                let (x, y, z) = (
                    (x + (k & 1)).min(nx - 1),
                    (y + (k >> 1 & 1)).min(ny - 1),
                    (z + (k >> 2)).min(nz - 1),
                );
                tissue(x + nx * (y + ny * z))
            })
        })
        .collect();
    let body_coarse = largest_component(&coarse, [cx, cy, cz]);
    (0..nx * ny * nz)
        .into_par_iter()
        .map(|i| {
            let (x, y, z) = (i % nx, i / nx % ny, i / (nx * ny));
            let (x, y, z) = ((x / 2) as isize, (y / 2) as isize, (z / 2) as isize);
            tissue(i)
                && (0..27).any(|k| {
                    let c = |v: isize, d: isize, n: usize| {
                        (v + d - 1).clamp(0, n as isize - 1) as usize
                    };
                    let (a, b, e) = (c(x, k % 3, cx), c(y, k / 3 % 3, cy), c(z, k / 9, cz));
                    body_coarse[a + cx * (b + cy * e)]
                })
        })
        .collect()
}

/// Voxels connected to the border of their axial slice through non-body voxels.
fn outside_of(body: &[bool], dims: [usize; 3]) -> Vec<bool> {
    let [nx, ny, _] = dims;
    let mut outside = vec![false; body.len()];
    outside
        .par_chunks_mut(nx * ny)
        .zip(body.par_chunks(nx * ny))
        .for_each(|(out, body)| {
            let mut stack: Vec<usize> = (0..nx)
                .flat_map(|x| [x, x + nx * (ny - 1)])
                .chain((0..ny).flat_map(|y| [nx * y, nx - 1 + nx * y]))
                .filter(|&i| !body[i])
                .collect();
            for &i in &stack {
                out[i] = true;
            }
            while let Some(i) = stack.pop() {
                let (x, y) = (i % nx, i / nx);
                let mut visit = |j: usize| {
                    if !out[j] && !body[j] {
                        out[j] = true;
                        stack.push(j);
                    }
                };
                if x > 0 {
                    visit(i - 1);
                }
                if x + 1 < nx {
                    visit(i + 1);
                }
                if y > 0 {
                    visit(i - nx);
                }
                if y + 1 < ny {
                    visit(i + nx);
                }
            }
        });
    outside
}

/// The largest 6-connected component of `mask`.
fn largest_component(mask: &[bool], dims: [usize; 3]) -> Vec<bool> {
    let [nx, ny, nz] = dims;
    let mut component = vec![u32::MAX; mask.len()];
    let (mut best, mut best_size) = (u32::MAX, 0usize);
    let mut stack = Vec::new();
    let mut id = 0u32;
    for start in 0..mask.len() {
        if !mask[start] || component[start] != u32::MAX {
            continue;
        }
        component[start] = id;
        stack.push(start);
        let mut size = 0;
        while let Some(i) = stack.pop() {
            size += 1;
            let (x, y, z) = (i % nx, i / nx % ny, i / (nx * ny));
            let neighbours = [
                (x > 0).then(|| i - 1),
                (x + 1 < nx).then(|| i + 1),
                (y > 0).then(|| i - nx),
                (y + 1 < ny).then(|| i + nx),
                (z > 0).then(|| i - nx * ny),
                (z + 1 < nz).then(|| i + nx * ny),
            ];
            for j in neighbours.into_iter().flatten() {
                if mask[j] && component[j] == u32::MAX {
                    component[j] = id;
                    stack.push(j);
                }
            }
        }
        if size > best_size {
            (best, best_size) = (id, size);
        }
        id += 1;
    }
    component.into_par_iter().map(|c| c == best).collect()
}

/// Box dilation (any non-zero within `radius` voxels along each axis): a
/// sliding count along rows, then ORs of whole neighbouring rows and slices,
/// which stay contiguous in memory.
pub fn dilate(v: &[u8], dims: [usize; 3], radius: [usize; 3]) -> Vec<u8> {
    let [nx, ny, nz] = dims;
    let [rx, ry, rz] = radius;
    let mut a = vec![0u8; v.len()];
    a.par_chunks_mut(nx)
        .zip(v.par_chunks(nx))
        .for_each(|(out, row)| {
            let mut count: usize = row[..=rx.min(nx - 1)].iter().map(|&b| b as usize).sum();
            for (k, o) in out.iter_mut().enumerate() {
                *o = (count > 0) as u8;
                if k + rx + 1 < nx {
                    count += row[k + rx + 1] as usize;
                }
                if k >= rx {
                    count -= row[k - rx] as usize;
                }
            }
        });
    let slice = nx * ny;
    let mut b = vec![0u8; v.len()];
    b.par_chunks_mut(slice)
        .zip(a.par_chunks(slice))
        .for_each(|(out, src)| {
            for y in 0..ny {
                let row = &mut out[y * nx..(y + 1) * nx];
                for yy in y.saturating_sub(ry)..=(y + ry).min(ny - 1) {
                    for (o, &s) in row.iter_mut().zip(&src[yy * nx..(yy + 1) * nx]) {
                        *o |= s;
                    }
                }
            }
        });
    a.par_chunks_mut(slice).enumerate().for_each(|(z, out)| {
        out.fill(0);
        for zz in z.saturating_sub(rz)..=(z + rz).min(nz - 1) {
            for (o, &s) in out.iter_mut().zip(&b[zz * slice..(zz + 1) * slice]) {
                *o |= s;
            }
        }
    });
    a
}

/// Box blur of `v` (values 0-255) with `radius` voxels along each axis: sliding
/// sums along rows, then over neighbouring rows and slices.
pub fn box_blur(v: &[u8], dims: [usize; 3], radius: [usize; 3]) -> Vec<u8> {
    let [nx, ny, nz] = dims;
    let [rx, ry, rz] = radius;
    let mut a = vec![0u8; v.len()];
    a.par_chunks_mut(nx)
        .zip(v.par_chunks(nx))
        .for_each(|(out, row)| {
            let mut sum: u32 = row[..rx.min(nx - 1) + 1].iter().map(|&b| b as u32).sum();
            let mut n = (rx.min(nx - 1) + 1) as u32;
            for (k, o) in out.iter_mut().enumerate() {
                *o = (sum / n) as u8;
                if k + rx + 1 < nx {
                    sum += row[k + rx + 1] as u32;
                    n += 1;
                }
                if k >= rx {
                    sum -= row[k - rx] as u32;
                    n -= 1;
                }
            }
        });
    let slice = nx * ny;
    let mut b = vec![0u8; v.len()];
    b.par_chunks_mut(slice)
        .zip(a.par_chunks(slice))
        .for_each(|(out, src)| {
            let mut sum = vec![0u32; nx];
            for y in 0..ny {
                sum.fill(0);
                let range = y.saturating_sub(ry)..=(y + ry).min(ny - 1);
                let n = range.clone().count() as u32;
                for yy in range {
                    for (s, &p) in sum.iter_mut().zip(&src[yy * nx..(yy + 1) * nx]) {
                        *s += p as u32;
                    }
                }
                for (o, s) in out[y * nx..(y + 1) * nx].iter_mut().zip(&sum) {
                    *o = (s / n) as u8;
                }
            }
        });
    a.par_chunks_mut(slice).enumerate().for_each(|(z, out)| {
        let mut sum = vec![0u32; slice];
        let range = z.saturating_sub(rz)..=(z + rz).min(nz - 1);
        let n = range.clone().count() as u32;
        for zz in range {
            for (s, &p) in sum.iter_mut().zip(&b[zz * slice..(zz + 1) * slice]) {
                *s += p as u32;
            }
        }
        for (o, s) in out.iter_mut().zip(&sum) {
            *o = (s / n) as u8;
        }
    });
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dilate_is_a_box() {
        let dims = [7, 6, 5];
        let mut v = vec![0u8; 7 * 6 * 5];
        let at = |x: usize, y: usize, z: usize| x + 7 * (y + 6 * z);
        v[at(3, 2, 2)] = 1;
        let d = dilate(&v, dims, [2, 1, 1]);
        for z in 0..5usize {
            for y in 0..6usize {
                for x in 0..7usize {
                    let inside = x.abs_diff(3) <= 2 && y.abs_diff(2) <= 1 && z.abs_diff(2) <= 1;
                    assert_eq!(d[at(x, y, z)], inside as u8, "at {x} {y} {z}");
                }
            }
        }
    }
}
