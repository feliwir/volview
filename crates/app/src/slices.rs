use std::path::Path;

use anyhow::Result;
use volio::Volume;

/// Writes the three orthogonal mid-slices through the voxel grid, resampled to
/// square pixels at the finest in-plane spacing.
pub fn dump_slices(v: &Volume, out: &Path, center: f32, width: f32) -> Result<()> {
    std::fs::create_dir_all(out)?;
    let d = v.dims;
    let s = v.spacing;
    let mid = d / 2;
    let window = |hu: i16| -> u8 {
        (((hu as f32 - (center - width / 2.0)) / width).clamp(0.0, 1.0) * 255.0).round() as u8
    };
    // (name, horizontal axis, vertical axis, fixed axis)
    let planes = [
        ("axial", 0, 1, 2),
        ("coronal", 0, 2, 1),
        ("sagittal", 1, 2, 0),
    ];
    for (name, u, w, fixed) in planes {
        let px = s[u].min(s[w]);
        let width_px = ((d[u] as f64 * s[u]) / px).round() as u32;
        let height_px = ((d[w] as f64 * s[w]) / px).round() as u32;
        let img = image::GrayImage::from_fn(width_px, height_px, |x, y| {
            let mut idx = [0u32; 3];
            idx[u] = (((x as f64 + 0.5) * px / s[u]) as u32).min(d[u] - 1);
            // Slice index increases upwards (superior for a standard axial stack).
            let yy = if w == 2 { height_px - 1 - y } else { y };
            idx[w] = (((yy as f64 + 0.5) * px / s[w]) as u32).min(d[w] - 1);
            idx[fixed] = mid[fixed];
            image::Luma([window(v.get(idx[0], idx[1], idx[2]))])
        });
        let path = out.join(format!("{name}.png"));
        img.save(&path)?;
        tracing::info!("wrote {}", path.display());
    }
    Ok(())
}
