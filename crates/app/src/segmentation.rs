//! Automatic TotalSegmentator labels: cached per series, computed in the background.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use volio::Volume;

use crate::labels::LabelSet;

/// Repository root at build time; the default model and CUDA library locations live under `data/`.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `$VOLVIEW_MODELS`, else `data/models/totalseg` in the repository.
pub fn default_models_dir() -> PathBuf {
    std::env::var_os("VOLVIEW_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root().join("data/models/totalseg"))
}

pub fn models_available(dir: &Path) -> bool {
    dir.join("totalseg.json").is_file()
}

/// Makes the CUDA, cuDNN and TensorRT libraries under `$VOLVIEW_CUDA_DIR` (default
/// `data/cuda13`) findable by re-executing with `LD_LIBRARY_PATH` extended. The
/// dynamic loader only reads that variable at startup, so it cannot be set later.
pub fn ensure_cuda_library_path() {
    const MARK: &str = "VOLVIEW_LIBRARY_PATH_SET";
    if std::env::var_os(MARK).is_some() {
        return;
    }
    let root = std::env::var_os("VOLVIEW_CUDA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root().join("data/cuda13"));
    let mut dirs: Vec<PathBuf> = ["nvidia/cu13/lib", "nvidia/cudnn/lib", "tensorrt_libs"]
        .iter()
        .map(|d| root.join(d))
        .filter(|d| d.is_dir())
        .collect();
    // OpenImageDenoise for the path tracer (see .cargo/config.toml).
    let oidn = workspace_root().join("data/oidn/lib");
    if oidn.is_dir() {
        dirs.push(oidn);
    }
    if dirs.is_empty() {
        return;
    }
    let mut paths = dirs;
    if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH") {
        paths.extend(std::env::split_paths(&existing));
    }
    let Ok(joined) = std::env::join_paths(paths) else {
        return;
    };
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    use std::os::unix::process::CommandExt;
    let err = std::process::Command::new(exe)
        .args(std::env::args_os().skip(1))
        .env("LD_LIBRARY_PATH", joined)
        .env(MARK, "1")
        .exec();
    tracing::warn!("could not re-execute with CUDA libraries on LD_LIBRARY_PATH: {err}");
}

/// Cached labels (NIfTI) and soft labels for a series.
fn cache_paths(vol: &Volume) -> Option<(PathBuf, PathBuf)> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    // v3: with the model's decision margins (soft labels).
    let dir = base.join("volview/labels");
    Some((
        dir.join(format!("{}.v3.nii.gz", vol.series_uid)),
        dir.join(format!("{}.v3.soft.gz", vol.series_uid)),
    ))
}

/// Labels computed earlier for this series, if cached.
pub fn load_cached(vol: &Volume) -> Option<LabelSet> {
    let (path, soft_path) = cache_paths(vol)?;
    if !path.is_file() || !soft_path.is_file() {
        return None;
    }
    match seg::read_labels(&path, vol)
        .and_then(|(data, names)| Ok((data, names, seg::read_soft(&soft_path)?)))
    {
        Ok((data, names, soft)) => {
            tracing::info!("loaded cached labels from {}", path.display());
            let mut set = LabelSet::new(data, &names, vol);
            set.soft = Some(soft);
            Some(set)
        }
        Err(e) => {
            tracing::warn!("ignoring cached labels {}: {e:#}", path.display());
            None
        }
    }
}

/// Segments `vol` and stores the result in the cache.
pub fn compute(
    vol: &Volume,
    models: &Path,
    use_gpu: bool,
    progress: &mut dyn FnMut(f32),
) -> Result<LabelSet> {
    let segmenter = seg::Segmenter::open(models, use_gpu)?;
    let t = std::time::Instant::now();
    let result = segmenter.segment_with_progress(vol, progress)?;
    tracing::info!(elapsed = ?t.elapsed(), "segmented");
    if let Some((path, soft_path)) = cache_paths(vol) {
        let written = path
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .map_err(anyhow::Error::from)
            .and_then(|()| seg::write_labels(&path, vol, &result.labels, segmenter.labels()))
            .and_then(|()| seg::write_soft(&soft_path, &result.soft));
        match written {
            Ok(()) => tracing::info!("cached labels in {}", path.display()),
            Err(e) => tracing::warn!("could not cache labels: {e:#}"),
        }
    }
    let mut set = LabelSet::new(result.labels, segmenter.labels(), vol);
    set.soft = Some(result.soft);
    Ok(set)
}

/// A segmentation running on a background thread.
pub struct Job {
    progress: Arc<Mutex<f32>>,
    result: mpsc::Receiver<Result<LabelSet>>,
}

impl Job {
    pub fn spawn(vol: Arc<Volume>, models: PathBuf, use_gpu: bool) -> Result<Self> {
        let progress = Arc::new(Mutex::new(0.0));
        let (tx, result) = mpsc::channel();
        let shared = Arc::clone(&progress);
        std::thread::Builder::new()
            .name("segmentation".into())
            .spawn(move || {
                let mut report = |f: f32| *shared.lock().unwrap() = f;
                let _ = tx.send(compute(&vol, &models, use_gpu, &mut report));
            })
            .context("starting segmentation thread")?;
        Ok(Self { progress, result })
    }

    pub fn progress(&self) -> f32 {
        *self.progress.lock().unwrap()
    }

    /// The result once the thread has finished.
    pub fn try_take(&self) -> Option<Result<LabelSet>> {
        match self.result.try_recv() {
            Ok(r) => Some(r),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                Some(Err(anyhow::anyhow!("segmentation thread panicked")))
            }
        }
    }
}
