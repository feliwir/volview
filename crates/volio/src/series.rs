use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use dicom_dictionary_std::tags;
use dicom_object::{DefaultDicomObject, OpenFileOptions};
use dicom_pixeldata::PixelDecoder;
use glam::{DMat3, DVec3, UVec3};
use rayon::prelude::*;
use walkdir::WalkDir;

use crate::{Error, Result, Volume};

/// Relative tolerance for slice spacing irregularities.
const SPACING_TOLERANCE: f64 = 0.01;
/// Minimum cosine between the slice stacking direction and the slice normal.
const TILT_COS_TOLERANCE: f64 = 0.9999;

#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// Load this `SeriesInstanceUID`; otherwise the CT series with the most slices.
    pub series_uid: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SeriesInfo {
    pub uid: String,
    pub modality: String,
    pub description: String,
    pub slices: usize,
}

#[derive(Debug, Clone)]
struct SliceHeader {
    path: PathBuf,
    sop_uid: String,
    series_uid: String,
    modality: String,
    description: String,
    position: DVec3,
    row_dir: DVec3,
    col_dir: DVec3,
    rows: u32,
    cols: u32,
    /// (between rows, between columns) in mm, as in `PixelSpacing`.
    pixel_spacing: (f64, f64),
    frames: u32,
}

impl SliceHeader {
    fn normal(&self) -> DVec3 {
        self.row_dir.cross(self.col_dir).normalize()
    }
}

/// Lists the image series found in `dir`.
pub fn list_series(dir: &Path) -> Result<Vec<SeriesInfo>> {
    let headers = scan_headers(dir)?;
    let mut series: Vec<SeriesInfo> = group_by_series(headers)
        .into_values()
        .map(|slices| SeriesInfo {
            uid: slices[0].series_uid.clone(),
            modality: slices[0].modality.clone(),
            description: slices[0].description.clone(),
            slices: slices.len(),
        })
        .collect();
    series.sort_by(|a, b| b.slices.cmp(&a.slices));
    Ok(series)
}

/// Loads a DICOM series from a folder (searched recursively) into a HU volume.
pub fn load_series(dir: &Path, options: &LoadOptions) -> Result<Volume> {
    let headers = scan_headers(dir)?;
    let mut groups = group_by_series(headers);

    let uid = match &options.series_uid {
        Some(uid) => uid.clone(),
        None => groups
            .iter()
            .filter(|(_, s)| s[0].modality == "CT")
            .max_by_key(|(_, s)| s.len())
            .or_else(|| groups.iter().max_by_key(|(_, s)| s.len()))
            .map(|(uid, _)| uid.clone())
            .ok_or_else(|| Error::NoImages(dir.to_owned()))?,
    };
    let slices = groups
        .remove(&uid)
        .ok_or_else(|| Error::SeriesNotFound(uid.clone()))?;
    if slices[0].modality != "CT" {
        tracing::warn!(modality = %slices[0].modality, "series is not CT; HU presets may not apply");
    }

    let (slices, geometry) = sort_and_validate(slices)?;
    tracing::info!(
        series = %uid,
        dims = ?geometry.dims,
        spacing = ?geometry.spacing,
        "decoding series"
    );

    let slice_len = (geometry.dims.x * geometry.dims.y) as usize;
    let mut hu = vec![0i16; slice_len * slices.len()];
    hu.par_chunks_mut(slice_len)
        .zip(slices.par_iter())
        .try_for_each(|(dst, slice)| decode_slice(&slice.path, dst))?;

    Ok(Volume {
        dims: geometry.dims,
        spacing: geometry.spacing,
        origin: geometry.origin,
        direction: geometry.direction,
        hu,
        series_uid: uid,
    })
}

fn scan_headers(dir: &Path) -> Result<Vec<SliceHeader>> {
    let paths: Vec<PathBuf> = WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .collect();

    let headers: Vec<SliceHeader> = paths
        .par_iter()
        .filter_map(|path| {
            let obj = OpenFileOptions::new()
                .read_until(tags::PIXEL_DATA)
                .open_file(path)
                .ok()?;
            match read_header(path, &obj) {
                Ok(h) => Some(h),
                Err(e) => {
                    tracing::debug!("skipping {}: {e}", path.display());
                    None
                }
            }
        })
        .collect();

    if headers.is_empty() {
        return Err(Error::NoImages(dir.to_owned()));
    }
    Ok(headers)
}

fn read_header(path: &Path, obj: &DefaultDicomObject) -> Result<SliceHeader> {
    let err = |msg: String| Error::File {
        path: path.to_owned(),
        msg,
    };
    let floats = |tag, name: &str, n: usize| -> Result<Vec<f64>> {
        let v = obj
            .element(tag)
            .map_err(|_| err(format!("missing {name}")))?
            .to_multi_float64()
            .map_err(|e| err(format!("invalid {name}: {e}")))?;
        if v.len() < n {
            return Err(err(format!("{name} has {} values, expected {n}", v.len())));
        }
        Ok(v)
    };
    let string = |tag| -> String {
        obj.element_opt(tag)
            .ok()
            .flatten()
            .and_then(|e| e.to_str().ok())
            .map(|s| s.trim().to_owned())
            .unwrap_or_default()
    };
    let int = |tag, name: &str| -> Result<u32> {
        obj.element(tag)
            .map_err(|_| err(format!("missing {name}")))?
            .to_int::<u32>()
            .map_err(|e| err(format!("invalid {name}: {e}")))
    };

    let series_uid = string(tags::SERIES_INSTANCE_UID);
    if series_uid.is_empty() {
        return Err(err("missing SeriesInstanceUID".into()));
    }
    let ipp = floats(tags::IMAGE_POSITION_PATIENT, "ImagePositionPatient", 3)?;
    let iop = floats(
        tags::IMAGE_ORIENTATION_PATIENT,
        "ImageOrientationPatient",
        6,
    )?;
    let ps = floats(tags::PIXEL_SPACING, "PixelSpacing", 2)?;
    let frames = obj
        .element_opt(tags::NUMBER_OF_FRAMES)
        .ok()
        .flatten()
        .and_then(|e| e.to_int::<u32>().ok())
        .unwrap_or(1);

    Ok(SliceHeader {
        path: path.to_owned(),
        sop_uid: string(tags::SOP_INSTANCE_UID),
        series_uid,
        modality: string(tags::MODALITY),
        description: string(tags::SERIES_DESCRIPTION),
        position: DVec3::new(ipp[0], ipp[1], ipp[2]),
        row_dir: DVec3::new(iop[0], iop[1], iop[2]).normalize(),
        col_dir: DVec3::new(iop[3], iop[4], iop[5]).normalize(),
        rows: int(tags::ROWS, "Rows")?,
        cols: int(tags::COLUMNS, "Columns")?,
        pixel_spacing: (ps[0], ps[1]),
        frames,
    })
}

/// Groups slices by series, dropping duplicate copies of the same instance
/// (common when a study was copied into several folders).
fn group_by_series(mut headers: Vec<SliceHeader>) -> HashMap<String, Vec<SliceHeader>> {
    // Sort by path so the kept copy does not depend on directory walk order.
    headers.sort_by(|a, b| a.path.cmp(&b.path));
    let mut seen = HashSet::new();
    let mut groups: HashMap<String, Vec<SliceHeader>> = HashMap::new();
    for h in headers {
        if !h.sop_uid.is_empty() && !seen.insert(h.sop_uid.clone()) {
            continue;
        }
        groups.entry(h.series_uid.clone()).or_default().push(h);
    }
    groups
}

struct Geometry {
    dims: UVec3,
    spacing: DVec3,
    origin: DVec3,
    direction: DMat3,
}

fn sort_and_validate(mut slices: Vec<SliceHeader>) -> Result<(Vec<SliceHeader>, Geometry)> {
    let first = slices[0].clone();
    if slices.iter().any(|s| s.frames != 1) {
        return Err(Error::Unsupported("multi-frame images".into()));
    }
    for s in &slices {
        if s.rows != first.rows || s.cols != first.cols {
            return Err(Error::Inconsistent(
                "slices have different image sizes".into(),
            ));
        }
        if s.row_dir.dot(first.row_dir) < TILT_COS_TOLERANCE
            || s.col_dir.dot(first.col_dir) < TILT_COS_TOLERANCE
        {
            return Err(Error::Inconsistent(
                "slices have different orientations".into(),
            ));
        }
        if (s.pixel_spacing.0 - first.pixel_spacing.0).abs() > 1e-4
            || (s.pixel_spacing.1 - first.pixel_spacing.1).abs() > 1e-4
        {
            return Err(Error::Inconsistent(
                "slices have different pixel spacing".into(),
            ));
        }
    }

    let normal = first.normal();
    slices.sort_by(|a, b| a.position.dot(normal).total_cmp(&b.position.dot(normal)));

    // Duplicate positions usually mean several acquisitions share a series UID.
    let offsets: Vec<f64> = slices.iter().map(|s| s.position.dot(normal)).collect();
    if offsets.windows(2).any(|w| (w[1] - w[0]).abs() < 1e-3) {
        return Err(Error::Inconsistent(
            "multiple slices at the same position; use --series or split the folder".into(),
        ));
    }

    let slice_spacing = if slices.len() > 1 {
        let span = offsets[offsets.len() - 1] - offsets[0];
        let mean = span / (slices.len() - 1) as f64;
        for w in offsets.windows(2) {
            let d = w[1] - w[0];
            if (d - mean).abs() > SPACING_TOLERANCE * mean + 1e-3 {
                return Err(Error::Inconsistent(format!(
                    "non-uniform slice spacing ({d:.3} mm vs mean {mean:.3} mm); missing slices?"
                )));
            }
        }
        let stack = (slices[slices.len() - 1].position - slices[0].position).normalize();
        if stack.dot(normal) < TILT_COS_TOLERANCE {
            return Err(Error::Unsupported("gantry tilt / sheared stack".into()));
        }
        mean
    } else {
        1.0
    };

    let geometry = Geometry {
        dims: UVec3::new(first.cols, first.rows, slices.len() as u32),
        // PixelSpacing is (row spacing, column spacing) = (j, i).
        spacing: DVec3::new(first.pixel_spacing.1, first.pixel_spacing.0, slice_spacing),
        origin: slices[0].position,
        direction: DMat3::from_cols(first.row_dir, first.col_dir, normal),
    };
    Ok((slices, geometry))
}

fn decode_slice(path: &Path, dst: &mut [i16]) -> Result<()> {
    let err = |msg: String| Error::File {
        path: path.to_owned(),
        msg,
    };
    let obj = dicom_object::open_file(path).map_err(|e| err(e.to_string()))?;
    let pixels = obj.decode_pixel_data().map_err(|e| err(e.to_string()))?;
    if pixels.samples_per_pixel() != 1 {
        return Err(err("expected monochrome pixel data".into()));
    }
    // Applies the modality LUT (RescaleSlope / RescaleIntercept) -> HU.
    let values: Vec<f32> = pixels.to_vec_frame(0).map_err(|e| err(e.to_string()))?;
    if values.len() != dst.len() {
        return Err(err(format!(
            "expected {} pixels, got {}",
            dst.len(),
            values.len()
        )));
    }
    for (d, v) in dst.iter_mut().zip(values) {
        *d = v.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
