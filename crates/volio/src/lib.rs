//! DICOM CT series loading.
//!
//! Loads a folder of single-frame DICOM slices into a [`Volume`] of Hounsfield
//! units with its patient-space (LPS, millimetres) geometry.

mod series;
mod volume;

pub use series::{LoadOptions, SeriesInfo, list_series, load_series};
pub use volume::Volume;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no DICOM image files found in {0}")]
    NoImages(std::path::PathBuf),
    #[error("series {0} not found")]
    SeriesNotFound(String),
    #[error("inconsistent series: {0}")]
    Inconsistent(String),
    #[error("unsupported series: {0}")]
    Unsupported(String),
    #[error("{path}: {msg}")]
    File {
        path: std::path::PathBuf,
        msg: String,
    },
}

pub type Result<T> = std::result::Result<T, Error>;
