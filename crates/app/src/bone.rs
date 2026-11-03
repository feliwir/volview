//! Bone the segmentation misses. The model labels the skull at its own coarse
//! resolution and drops thin plates, such as the nasal bones, the orbital walls
//! and the sinus walls, so label modes would show holes where the CT has bone.

use rayon::prelude::*;
use volio::Volume;

/// Unlabelled bone connected to labelled bone.
pub const BONE: u8 = 250;

/// Voxels at least this dense (HU) can be bone: thin plates only reach a few
/// hundred HU through partial volume. Contrast-filled vessels are as dense, so
/// how far bone grows is limited too.
const BONE_HU: i16 = 200;

/// Labels unlabelled voxels of at least [`BONE_HU`] as [`BONE`] that connect
/// (6-neighbourhood, through such voxels) to a voxel whose label `seed` accepts
/// within a path of `reach_mm`. Returns the number of voxels labelled.
pub fn add_bone(labels: &mut [u8], vol: &Volume, seed: &[bool; 256], reach_mm: f32) -> usize {
    let t = std::time::Instant::now();
    let [nx, ny, nz] = [vol.dims.x, vol.dims.y, vol.dims.z].map(|d| d as usize);
    let steps = (reach_mm as f64 / vol.spacing.min_element()).round() as usize;
    let candidate = |labels: &[u8], i: usize| labels[i] == 0 && vol.hu[i] >= BONE_HU;
    let neighbours = |i: usize| {
        let (x, y, z) = (i % nx, i / nx % ny, i / (nx * ny));
        [
            (x > 0).then(|| i - 1),
            (x + 1 < nx).then(|| i + 1),
            (y > 0).then(|| i - nx),
            (y + 1 < ny).then(|| i + nx),
            (z > 0).then(|| i - nx * ny),
            (z + 1 < nz).then(|| i + nx * ny),
        ]
        .into_iter()
        .flatten()
    };

    // Breadth-first from the labelled bone, one voxel step per round.
    let mut frontier: Vec<usize> = (0..labels.len())
        .into_par_iter()
        .filter(|&i| candidate(labels, i) && neighbours(i).any(|j| seed[labels[j] as usize]))
        .collect();
    let mut count = 0;
    for _ in 0..steps {
        if frontier.is_empty() {
            break;
        }
        for &i in &frontier {
            labels[i] = BONE;
        }
        count += frontier.len();
        let mut next: Vec<usize> = frontier
            .iter()
            .flat_map(|&i| neighbours(i))
            .filter(|&j| candidate(labels, j))
            .collect();
        next.sort_unstable();
        next.dedup();
        frontier = next;
    }
    tracing::info!(voxels = count, elapsed = ?t.elapsed(), "unlabelled bone");
    count
}
