//! Soft segmentation: the model's decision margins on its own grid.
//!
//! Hard labels are piecewise constant on the 1.5 mm prediction grid, so any
//! surface drawn from them is a staircase. The margin between the best and the
//! second best class varies smoothly and is zero on the predicted boundary, so
//! interpolating it gives boundaries with sub-voxel precision.

use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result, ensure};

/// Margins beyond this many logits are stored as this (the boundary is far).
pub const MAX_MARGIN: f32 = 15.9;
/// Margins are stored in steps of 1/16 logit.
pub const MARGIN_SCALE: f32 = 16.0;
const MAGIC: &[u8; 8] = b"VVSOFT1\0";

pub struct SoftLabels {
    /// Grid size along the RAS-canonical axes.
    pub dims: [usize; 3],
    /// Volume axis that each canonical axis runs along, and whether it is reversed.
    pub src_axis: [usize; 3],
    pub flip: [bool; 3],
    /// Per grid node (x fastest): best label, runner-up label, margin × 16, unused.
    pub data: Vec<[u8; 4]>,
}

impl SoftLabels {
    pub fn new(
        dims: [usize; 3],
        src_axis: [usize; 3],
        flip: [bool; 3],
        best: &[u8],
        second: &[u8],
        margin: &[f32],
    ) -> Self {
        let data = best
            .iter()
            .zip(second)
            .zip(margin)
            .map(|((&b, &s), &m)| {
                [
                    b,
                    s,
                    (m.clamp(0.0, MAX_MARGIN) * MARGIN_SCALE).round() as u8,
                    0,
                ]
            })
            .collect();
        Self {
            dims,
            src_axis,
            flip,
            data,
        }
    }
}

/// Writes gzip-compressed soft labels (a volview cache format).
pub fn write_soft(path: &Path, soft: &SoftLabels) -> Result<()> {
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut gz = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
    gz.write_all(MAGIC)?;
    for d in soft.dims {
        gz.write_all(&(d as u32).to_le_bytes())?;
    }
    gz.write_all(&soft.src_axis.map(|a| a as u8))?;
    gz.write_all(&soft.flip.map(u8::from))?;
    gz.write_all(soft.data.as_flattened())?;
    gz.finish()?.flush()?;
    Ok(())
}

pub fn read_soft(path: &Path) -> Result<SoftLabels> {
    let mut raw = Vec::new();
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    flate2::read::GzDecoder::new(file).read_to_end(&mut raw)?;
    ensure!(
        raw.len() >= 26 && &raw[..8] == MAGIC,
        "not a soft label file"
    );
    let dim =
        |i: usize| u32::from_le_bytes(raw[8 + 4 * i..12 + 4 * i].try_into().unwrap()) as usize;
    let dims = [dim(0), dim(1), dim(2)];
    let src_axis = [raw[20], raw[21], raw[22]].map(usize::from);
    let flip = [raw[23], raw[24], raw[25]].map(|f| f != 0);
    let n = dims.iter().product::<usize>();
    let body = &raw[26..];
    ensure!(body.len() == 4 * n, "soft label file is truncated");
    let data = body
        .chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect();
    Ok(SoftLabels {
        dims,
        src_axis,
        flip,
        data,
    })
}
