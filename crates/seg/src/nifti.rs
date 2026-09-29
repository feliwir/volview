//! Minimal NIfTI-1 reading and writing for label volumes, plus a JSON sidecar
//! (`<name>.json` next to `<name>.nii[.gz]`) holding the label names.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use glam::{DMat4, DVec4};
use volio::Volume;

/// Tolerance for matching a label map's voxel-to-world transform, in mm.
const AFFINE_TOLERANCE_MM: f64 = 0.01;

/// Voxel-index to NIfTI world (RAS) transform of `vol`; volio's world space is LPS.
fn ras_affine(vol: &Volume) -> DMat4 {
    DMat4::from_diagonal(DVec4::new(-1.0, -1.0, 1.0, 1.0)) * vol.index_to_world()
}

fn sidecar_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let stem = name.strip_suffix(".gz").unwrap_or(&name);
    let stem = stem.strip_suffix(".nii").unwrap_or(stem);
    path.with_file_name(format!("{stem}.json"))
}

/// Writes `labels` (on `vol`'s grid) as a uint8 NIfTI, gzip-compressed if the path
/// ends in `.gz`, and the label names to the JSON sidecar.
pub fn write_labels(
    path: &Path,
    vol: &Volume,
    labels: &[u8],
    names: &BTreeMap<u8, String>,
) -> Result<()> {
    let ras = ras_affine(vol);
    let mut h = vec![0u8; 348];
    let put_i16 =
        |h: &mut [u8], off: usize, v: i16| h[off..off + 2].copy_from_slice(&v.to_le_bytes());
    let put_f32 =
        |h: &mut [u8], off: usize, v: f32| h[off..off + 4].copy_from_slice(&v.to_le_bytes());
    h[0..4].copy_from_slice(&348i32.to_le_bytes());
    for (i, d) in [
        3,
        vol.dims.x as i16,
        vol.dims.y as i16,
        vol.dims.z as i16,
        1,
        1,
        1,
        1,
    ]
    .into_iter()
    .enumerate()
    {
        put_i16(&mut h, 40 + 2 * i, d);
    }
    put_i16(&mut h, 70, 2); // datatype: uint8
    put_i16(&mut h, 72, 8); // bitpix
    let pixdim = [
        1.0,
        vol.spacing.x,
        vol.spacing.y,
        vol.spacing.z,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    for (i, v) in pixdim.into_iter().enumerate() {
        put_f32(&mut h, 76 + 4 * i, v as f32);
    }
    put_f32(&mut h, 108, 352.0); // vox_offset
    put_f32(&mut h, 112, 1.0); // scl_slope
    h[123] = 10; // xyzt_units: mm, s
    put_i16(&mut h, 254, 1); // sform_code: scanner
    for (r, off) in [(0, 280), (1, 296), (2, 312)] {
        for (i, v) in ras.row(r).to_array().into_iter().enumerate() {
            put_f32(&mut h, off + 4 * i, v as f32);
        }
    }
    h[344..348].copy_from_slice(b"n+1\0");

    let write = |w: &mut dyn Write| -> std::io::Result<()> {
        w.write_all(&h)?;
        w.write_all(&[0u8; 4])?; // no extensions
        w.write_all(labels)
    };
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    if path.extension().is_some_and(|e| e == "gz") {
        let mut gz = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        write(&mut gz)?;
        gz.finish()?.flush()?;
    } else {
        write(&mut file)?;
        file.flush()?;
    }
    let sidecar = serde_json::json!({ "labels": names });
    std::fs::write(sidecar_path(path), serde_json::to_string_pretty(&sidecar)?)?;
    Ok(())
}

/// Reads a label map that must lie on `vol`'s voxel grid. Returns the labels and the
/// names from the JSON sidecar (empty if there is none).
pub fn read_labels(path: &Path, vol: &Volume) -> Result<(Vec<u8>, BTreeMap<u8, String>)> {
    let mut raw = Vec::new();
    std::fs::File::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .read_to_end(&mut raw)?;
    if raw.starts_with(&[0x1f, 0x8b]) {
        let mut out = Vec::new();
        flate2::read::MultiGzDecoder::new(&raw[..]).read_to_end(&mut out)?;
        raw = out;
    }
    ensure!(raw.len() >= 352, "file too small for a NIfTI-1 header");
    ensure!(
        i32::from_le_bytes(raw[0..4].try_into()?) == 348,
        "not a little-endian NIfTI-1 file"
    );
    let i16_at = |off: usize| i16::from_le_bytes([raw[off], raw[off + 1]]);
    let f32_at = |off: usize| f32::from_le_bytes(raw[off..off + 4].try_into().unwrap());

    let dims = [i16_at(42), i16_at(44), i16_at(46)].map(|d| d as u32);
    ensure!(
        i16_at(40) >= 3 && dims == vol.dims.to_array(),
        "label map dims {dims:?} do not match the volume {:?}",
        vol.dims
    );
    if i16_at(254) > 0 {
        let expected = ras_affine(vol);
        for (r, off) in [(0, 280), (1, 296), (2, 312)] {
            for c in 0..4 {
                let got = f32_at(off + 4 * c) as f64;
                let want = expected.row(r)[c];
                // Offsets are in mm; direction entries are scaled by spacing, also mm.
                if (got - want).abs() > AFFINE_TOLERANCE_MM {
                    bail!(
                        "label map is on a different grid than the volume (sform row {r} col {c}: {got} vs {want})"
                    );
                }
            }
        }
    } else {
        tracing::warn!("label map has no sform; assuming it matches the volume grid");
    }

    let n = vol.voxel_count();
    let offset = (f32_at(108) as usize).max(352);
    let data = &raw[offset..];
    let labels: Vec<u8> = match i16_at(70) {
        2 => data.get(..n).context("truncated voxel data")?.to_vec(),
        4 | 512 => {
            let bytes = data.get(..2 * n).context("truncated voxel data")?;
            bytes
                .chunks_exact(2)
                .map(|b| u8::try_from(i16::from_le_bytes([b[0], b[1]]).max(0)))
                .collect::<Result<_, _>>()
                .context("label values above 255")?
        }
        t => bail!("unsupported NIfTI datatype {t} for labels"),
    };

    let sidecar = sidecar_path(path);
    let names = match std::fs::read_to_string(&sidecar) {
        Ok(text) => {
            #[derive(serde::Deserialize)]
            struct Sidecar {
                labels: BTreeMap<u8, String>,
            }
            serde_json::from_str::<Sidecar>(&text)
                .with_context(|| format!("parsing {}", sidecar.display()))?
                .labels
        }
        Err(_) => BTreeMap::new(),
    };
    Ok((labels, names))
}
