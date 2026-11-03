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

/// Per axis `[start, end)`: an index box.
pub type BBox = [[usize; 2]; 3];

/// Bounding box of the voxels carrying any of `labels`, widened by `margin` voxels
/// per axis and clamped to the volume (TotalSegmentator's `get_bbox_from_mask`).
/// `None` if there are none.
pub fn bounding_box(
    src: &[u8],
    dims: [usize; 3],
    labels: &[u8],
    margin: [usize; 3],
) -> Option<BBox> {
    let wanted: [bool; 256] = std::array::from_fn(|v| labels.contains(&(v as u8)));
    let (nx, ny) = (dims[0], dims[1]);
    let found = src
        .par_chunks(nx * ny)
        .enumerate()
        .filter_map(|(z, slab)| {
            let mut b = [[usize::MAX, 0], [usize::MAX, 0], [z, z + 1]];
            for (i, &l) in slab.iter().enumerate() {
                if wanted[l as usize] {
                    let (x, y) = (i % nx, i / nx);
                    b[0] = [b[0][0].min(x), b[0][1].max(x + 1)];
                    b[1] = [b[1][0].min(y), b[1][1].max(y + 1)];
                }
            }
            (b[0][0] != usize::MAX).then_some(b)
        })
        .reduce_with(|a, b| {
            std::array::from_fn(|i| [a[i][0].min(b[i][0]), a[i][1].max(b[i][1])])
        })?;
    Some(std::array::from_fn(|a| {
        [
            found[a][0].saturating_sub(margin[a]),
            (found[a][1] + margin[a]).min(dims[a]),
        ]
    }))
}

/// The voxels of `src` inside `bbox`, x fastest.
pub fn crop<T: Copy + Send + Sync>(src: &[T], dims: [usize; 3], bbox: BBox) -> Vec<T> {
    let [[x0, x1], [y0, y1], [z0, z1]] = bbox;
    let (nx, ny) = (dims[0], dims[1]);
    (z0..z1)
        .into_par_iter()
        .flat_map_iter(|z| {
            (y0..y1).flat_map(move |y| {
                let row = nx * (y + ny * z);
                src[row + x0..row + x1].iter().copied()
            })
        })
        .collect()
}

/// Calls `f(dst voxel, src value)` for each voxel of `src` (the contents of `bbox`,
/// x fastest), in order.
pub fn paste<T: Copy>(
    dst: &mut [T],
    dims: [usize; 3],
    bbox: BBox,
    src: &[T],
    mut f: impl FnMut(&mut T, T),
) {
    let [[x0, x1], [y0, y1], [z0, z1]] = bbox;
    let (nx, ny) = (dims[0], dims[1]);
    let mut values = src.iter().copied();
    for z in z0..z1 {
        for y in y0..y1 {
            let row = nx * (y + ny * z);
            for (o, v) in dst[row + x0..row + x1].iter_mut().zip(&mut values) {
                f(o, v);
            }
        }
    }
}

/// Sets `nodes[n]` (a grid of `node_dims` spanning the volume corner to corner, as
/// [`zoom_labels`] samples it) for the node nearest each voxel flagged in `voxels`
/// (the contents of `bbox`, x fastest).
pub fn mark_nodes(
    nodes: &mut [bool],
    node_dims: [usize; 3],
    dims: [usize; 3],
    bbox: BBox,
    voxels: &[bool],
) {
    let nearest = |a: usize, v: usize| {
        let scale = (node_dims[a] - 1) as f64 / (dims[a] - 1).max(1) as f64;
        ((v as f64 * scale).round() as usize).min(node_dims[a] - 1)
    };
    let [[x0, x1], [y0, y1], [z0, _]] = bbox;
    let (w, h) = (x1 - x0, y1 - y0);
    for (i, _) in voxels.iter().enumerate().filter(|(_, c)| **c) {
        let (x, y, z) = (x0 + i % w, y0 + (i / w) % h, z0 + i / (w * h));
        let n = [nearest(0, x), nearest(1, y), nearest(2, z)];
        nodes[n[0] + node_dims[0] * (n[1] + node_dims[1] * n[2])] = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounding_box_widens_and_clamps() {
        let dims = [6, 5, 4];
        let mut labels = vec![0u8; 6 * 5 * 4];
        labels[2 + 6 * (1 + 5 * 1)] = 7; // (2, 1, 1)
        labels[3 + 6 * (3 + 5 * 2)] = 7; // (3, 3, 2)
        labels[5 + 6 * (4 + 5 * 3)] = 9; // other label
        assert_eq!(
            bounding_box(&labels, dims, &[7], [0; 3]),
            Some([[2, 4], [1, 4], [1, 3]])
        );
        assert_eq!(
            bounding_box(&labels, dims, &[7], [1, 2, 3]),
            Some([[1, 5], [0, 5], [0, 4]])
        );
        assert_eq!(bounding_box(&labels, dims, &[3], [0; 3]), None);
    }

    #[test]
    fn crop_and_paste_round_trip() {
        let dims = [4, 3, 2];
        let src: Vec<u8> = (0..24).collect();
        let bbox = [[1, 3], [1, 3], [0, 2]];
        let cropped = crop(&src, dims, bbox);
        assert_eq!(cropped, [5, 6, 9, 10, 17, 18, 21, 22]);
        let mut dst = vec![0u8; 24];
        paste(&mut dst, dims, bbox, &cropped, |o, v| *o = v);
        for (i, (&d, &s)) in dst.iter().zip(&src).enumerate() {
            let (x, y) = (i % 4, (i / 4) % 3);
            assert_eq!(
                d,
                if (1..3).contains(&x) && (1..3).contains(&y) {
                    s
                } else {
                    0
                }
            );
        }
    }

    #[test]
    fn mark_nodes_picks_nearest() {
        // 9 voxels per axis onto 5 nodes: voxel v lies at node v / 2, rounded.
        let mut nodes = vec![false; 125];
        let bbox = [[4, 6], [0, 1], [8, 9]];
        mark_nodes(&mut nodes, [5; 3], [9; 3], bbox, &[false, true]);
        let marked: Vec<usize> = (0..125).filter(|&i| nodes[i]).collect();
        // Only voxel (5, 0, 8) is flagged: node (3, 0, 4).
        assert_eq!(marked, [3 + 25 * 4]);
    }
}
