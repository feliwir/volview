//! Context tissue: the fat and connective tissue that packs the space between
//! organs (mesentery, omentum, mediastinal fat), which the segmentation does
//! not label. Without it, organs float in empty space.

use rayon::prelude::*;
use volio::Volume;

use crate::skin::{box_blur, dilate};

/// Unlabelled fat near organs.
pub const FAT: u8 = 252;
/// Unlabelled soft tissue near organs: fascia, unsegmented muscle and vessels.
pub const SOFT_TISSUE: u8 = 251;

/// Share of organ (of 255) in the blurred organ mask above which gaps are filled.
const FILL_THRESHOLD: u8 = 90;
/// Fat HU range (the usual fat window).
const FAT_HU: (i16, i16) = (-190, -30);
/// Soft tissue HU range, from fat up to unenhanced blood and muscle.
const SOFT_HU: (i16, i16) = (-30, 150);

/// Labels unlabelled fat and soft tissue in the gaps between organs (voxels
/// whose label `anchor` accepts): where the organ mask, blurred over about
/// `gap_mm`, is more than [`FILL_THRESHOLD`] organ. That fills gaps between
/// nearby organs (up to about 0.8 `gap_mm`) but adds only a thin film on their
/// convex outer surfaces, like a morphological closing with a smooth rather
/// than box-shaped boundary. Not within `wall_mm` of a `surface` voxel (the
/// skin): the body wall would cover the organs. Voxels are classified by
/// their 3^3 neighbourhood's mean HU, so CT noise does not speckle fat and
/// soft tissue. Returns the voxel counts (fat, soft tissue).
pub fn add_context(
    labels: &mut [u8],
    vol: &Volume,
    anchor: &[bool; 256],
    gap_mm: f32,
    surface: &[u8],
    wall_mm: f32,
) -> (usize, usize) {
    let t = std::time::Instant::now();
    let dims = [vol.dims.x, vol.dims.y, vol.dims.z].map(|d| d as usize);
    let spacing = vol.spacing.to_array();
    let radius = [0, 1, 2].map(|a| (gap_mm as f64 / spacing[a]).round().max(1.0) as usize);
    // Two box blurs of half the radius approximate a Gaussian.
    let half = radius.map(|r| r.div_ceil(2));
    let anchors: Vec<u8> = labels
        .par_iter()
        .map(|&l| if anchor[l as usize] { 255 } else { 0 })
        .collect();
    let blurred = box_blur(&box_blur(&anchors, dims, half), dims, half);
    drop(anchors);
    let near: Vec<u8> = blurred
        .into_par_iter()
        .map(|b| (b > FILL_THRESHOLD) as u8)
        .collect();
    let [nx, ny, nz] = dims;
    let mean_hu = |i: usize| -> i32 {
        let (x, y, z) = (i % nx, i / nx % ny, i / (nx * ny));
        let (mut sum, mut n) = (0i32, 0i32);
        for zz in z.saturating_sub(1)..=(z + 1).min(nz - 1) {
            for yy in y.saturating_sub(1)..=(y + 1).min(ny - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(nx - 1) {
                    sum += vol.hu[xx + nx * (yy + ny * zz)] as i32;
                    n += 1;
                }
            }
        }
        sum / n
    };
    let wall_radius = [0, 1, 2].map(|a| (wall_mm as f64 / spacing[a]).round().max(1.0) as usize);
    let skin: Vec<u8> = labels
        .par_iter()
        .map(|l| surface.contains(l) as u8)
        .collect();
    let wall = dilate(&skin, dims, wall_radius);
    drop(skin);
    let (fat, soft) = labels
        .par_iter_mut()
        .zip(near.par_iter().zip(wall.par_iter()))
        .enumerate()
        .map(|(i, (l, (&n, &w)))| {
            if *l != 0 || n == 0 || w != 0 {
                return (0, 0);
            }
            let hu = mean_hu(i) as i16;
            if (FAT_HU.0..FAT_HU.1).contains(&hu) {
                *l = FAT;
                (1, 0)
            } else if (SOFT_HU.0..SOFT_HU.1).contains(&hu) {
                *l = SOFT_TISSUE;
                (0, 1)
            } else {
                (0, 0)
            }
        })
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    tracing::info!(fat, soft, elapsed = ?t.elapsed(), "context tissue");
    (fat, soft)
}
