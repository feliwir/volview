//! Whole-body CT segmentation with TotalSegmentator's "total" nnU-Net models,
//! exported to ONNX by `tools/totalseg/export_onnx.py`.
//!
//! The pipeline reproduces TotalSegmentator: reorient to RAS, resample to
//! 1.5 mm, run each part model with sliding windows, merge the part labels
//! (later parts win), and resample back to the input grid.
//!
//! Then each subtask (finer models for single regions, e.g. the head muscles)
//! runs the same way on the bounding box of the labels it is cropped to, at its
//! own spacing, and its labels are pasted over the result (later subtasks win).

mod grid;
mod nifti;
mod soft;
mod window;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use ort::session::Session;
use serde::Deserialize;
use volio::Volume;

pub use nifti::{read_labels, write_labels};
pub use soft::{SoftLabels, read_soft, write_soft};

/// Result of [`Segmenter::segment`].
pub struct Segmentation {
    /// One label per voxel on the volume's own grid.
    pub labels: Vec<u8>,
    /// The model's scores on its own grid, for smooth boundaries.
    pub soft: SoftLabels,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    spacing_mm: f64,
    step_size: f64,
    gaussian_sigma_scale: f64,
    labels: BTreeMap<u8, String>,
    models: Vec<ModelInfo>,
    #[serde(default)]
    subtasks: Vec<Subtask>,
}

#[derive(Debug, Deserialize)]
struct Subtask {
    name: String,
    /// Whether its labels go into the result; if not it only serves as a crop source.
    merge: bool,
    step_size: f64,
    crop: Option<Crop>,
    /// Along the canonical axes R, A, S.
    spacing_mm: [f64; 3],
    /// Labels of the "total" models that this task takes over inside its crop:
    /// voxels it does not give them lose them.
    #[serde(default)]
    replaces: Vec<u8>,
    /// Model class index -> label (combined labels if `merge`, else the task's own).
    models: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct Crop {
    /// "total" or an earlier subtask.
    source: String,
    labels: Vec<u8>,
    margin_mm: f64,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    task: u32,
    file: String,
    /// nnU-Net order: z, y, x.
    patch_size: [usize; 3],
    num_classes: usize,
    pad_hu: f32,
    /// Model class index -> combined label (0 = none).
    label_map: Vec<u8>,
}

pub struct Segmenter {
    dir: PathBuf,
    manifest: Manifest,
    use_gpu: bool,
}

impl Segmenter {
    /// Opens a model folder containing `totalseg.json` and the ONNX files it lists.
    pub fn open(dir: &Path, use_gpu: bool) -> Result<Self> {
        let path = dir.join("totalseg.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let manifest: Manifest =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        Ok(Self {
            dir: dir.to_owned(),
            manifest,
            use_gpu,
        })
    }

    /// Names of the combined labels, keyed by label value.
    pub fn labels(&self) -> &BTreeMap<u8, String> {
        &self.manifest.labels
    }

    /// Starts a session for `model` and runs it over `img` (canonical, at the
    /// model's spacing), calling `emit` per z-slice.
    fn predict(
        &self,
        model: &ModelInfo,
        step_size: f64,
        img: &[f32],
        dims: [usize; 3],
        emit: impl FnMut(usize, &[window::Prediction]),
        progress: &mut dyn FnMut(usize, usize),
    ) -> Result<()> {
        let t = std::time::Instant::now();
        let mut session = self.session(model)?;
        let [pz, py, px] = model.patch_size;
        let params = window::Params {
            tile: [px, py, pz],
            step_size,
            sigma_scale: self.manifest.gaussian_sigma_scale,
            pad_value: model.pad_hu,
            num_classes: model.num_classes,
        };
        window::predict(&mut session, img, dims, &params, emit, progress)
            .with_context(|| format!("running model {}", model.task))?;
        tracing::info!(task = model.task, ?dims, elapsed = ?t.elapsed(), "model done");
        Ok(())
    }

    fn session(&self, model: &ModelInfo) -> Result<Session> {
        let path = self.dir.join(&model.file);
        let mut builder = Session::builder().map_err(|e| anyhow!("{e}"))?;
        if self.use_gpu {
            #[cfg(feature = "cuda")]
            {
                // ORT's prebuilt CUDA kernels lack Blackwell (sm_120) code, so the whole
                // graph goes through TensorRT. Engines are cached next to the models; the
                // first run per GPU builds them, which takes a few minutes.
                let [z, y, x] = model.patch_size;
                let shape = format!("hu:1x1x{z}x{y}x{x}");
                let cache = self.dir.join("trt_cache");
                std::fs::create_dir_all(&cache)?;
                let trt = ort::ep::TensorRT::default()
                    .with_fp16(true)
                    .with_engine_cache(true)
                    .with_engine_cache_path(cache.display())
                    .with_timing_cache(true)
                    .with_timing_cache_path(cache.display())
                    .with_profile_min_shapes(&shape)
                    .with_profile_opt_shapes(&shape)
                    .with_profile_max_shapes(&shape)
                    .build()
                    .error_on_failure();
                builder = builder
                    .with_execution_providers([
                        trt,
                        ort::ep::CUDA::default().build().error_on_failure(),
                    ])
                    .map_err(|e| {
                        anyhow!("enabling TensorRT/CUDA (use --cpu to run without it): {e}")
                    })?;
            }
            #[cfg(not(feature = "cuda"))]
            anyhow::bail!("built without the `cuda` feature");
        }
        builder
            .commit_from_file(&path)
            .map_err(|e| anyhow!("loading {}: {e}", path.display()))
    }

    /// Segments `vol` and returns one label per voxel on the volume's own grid.
    pub fn segment(&self, vol: &Volume) -> Result<Segmentation> {
        self.segment_with_progress(vol, &mut |_| {})
    }

    /// Like [`Segmenter::segment`], calling `progress` with the completed fraction.
    pub fn segment_with_progress(
        &self,
        vol: &Volume,
        progress: &mut dyn FnMut(f32),
    ) -> Result<Segmentation> {
        let m = &self.manifest;
        let can = grid::Canonical::new(vol);
        let hu = can.to_canonical(&vol.hu);
        let rs_dims =
            [0, 1, 2].map(|a| grid::zoomed_len(can.dims[a], can.spacing[a], m.spacing_mm));
        let t = std::time::Instant::now();
        let img = grid::zoom_linear_trunc(&hu, can.dims, rs_dims);
        tracing::info!(from = ?can.dims, to = ?rs_dims, elapsed = ?t.elapsed(), "resampled to {} mm", m.spacing_mm);

        let slice_len = rs_dims[0] * rs_dims[1];
        let mut labels = vec![0u8; img.len()];
        // Per node: runner-up label and how many logits the winner leads by.
        let mut runner_up = vec![0u8; img.len()];
        let mut margin = vec![soft::MAX_MARGIN; img.len()];
        // Progress counts models; subtasks run on crops, so this is rough.
        let n_models =
            (m.models.len() + m.subtasks.iter().map(|s| s.models.len()).sum::<usize>()) as f32;
        let mut models_done = 0;
        for model in &m.models {
            self.predict(
                model,
                m.step_size,
                &img,
                rs_dims,
                |z, predictions| {
                    let range = z * slice_len..(z + 1) * slice_len;
                    let (best, second, lead) = (
                        &mut labels[range.clone()],
                        &mut runner_up[range.clone()],
                        &mut margin[range],
                    );
                    for (i, p) in predictions.iter().enumerate() {
                        let l1 = model.label_map[p.best as usize];
                        let l2 = model.label_map[p.second as usize];
                        if l1 != 0 {
                            // Later parts win, as in TotalSegmentator. The competitor
                            // is this part's runner-up, or what was there before if
                            // that runner-up is background.
                            second[i] = if l2 != 0 { l2 } else { best[i] };
                            best[i] = l1;
                            lead[i] = p.margin;
                        } else if best[i] == 0 && l2 != 0 && p.margin < lead[i] {
                            // Background here, but close to one of this part's labels.
                            second[i] = l2;
                            lead[i] = p.margin;
                        }
                    }
                },
                &mut |done, tiles| {
                    progress((models_done as f32 + done as f32 / tiles as f32) / n_models)
                },
            )?;
            models_done += 1;
        }
        drop(img);

        let mut merged = grid::zoom_labels(&labels, rs_dims, can.dims);
        // Soft-grid nodes whose labels the subtasks changed; the model margins there
        // describe the "total" labels only.
        let mut overridden = vec![false; labels.len()];
        if !m.subtasks.is_empty() {
            let total = merged.clone();
            let mut outputs: BTreeMap<&str, Vec<u8>> = BTreeMap::new();
            for task in &m.subtasks {
                let bbox = match &task.crop {
                    None => Some([[0, can.dims[0]], [0, can.dims[1]], [0, can.dims[2]]]),
                    Some(crop) => {
                        let source = if crop.source == "total" {
                            &total
                        } else {
                            outputs.get(crop.source.as_str()).with_context(|| {
                                format!(
                                    "subtask {} is cropped to unknown task {}",
                                    task.name, crop.source
                                )
                            })?
                        };
                        let margin = can.spacing.map(|s| (crop.margin_mm / s) as usize);
                        grid::bounding_box(source, can.dims, &crop.labels, margin)
                    }
                };
                let Some(bbox) = bbox else {
                    tracing::info!(task = task.name, "nothing to crop to; skipped");
                    models_done += task.models.len();
                    continue;
                };
                let t = std::time::Instant::now();
                let crop_dims = bbox.map(|[a, b]| b - a);
                let rs = [0, 1, 2]
                    .map(|a| grid::zoomed_len(crop_dims[a], can.spacing[a], task.spacing_mm[a]));
                let img = grid::zoom_linear_trunc(&grid::crop(&hu, can.dims, bbox), crop_dims, rs);
                tracing::info!(task = task.name, ?bbox, to = ?rs, elapsed = ?t.elapsed(), "cropped and resampled");

                let slice_len = rs[0] * rs[1];
                let mut fine = vec![0u8; img.len()];
                for model in &task.models {
                    self.predict(
                        model,
                        task.step_size,
                        &img,
                        rs,
                        |z, predictions| {
                            let out = &mut fine[z * slice_len..(z + 1) * slice_len];
                            for (o, p) in out.iter_mut().zip(predictions) {
                                let l = model.label_map[p.best as usize];
                                if l != 0 {
                                    *o = l;
                                }
                            }
                        },
                        &mut |done, tiles| {
                            progress((models_done as f32 + done as f32 / tiles as f32) / n_models)
                        },
                    )?;
                    models_done += 1;
                }
                drop(img);
                let fine = grid::zoom_labels(&fine, rs, crop_dims);

                if !task.merge {
                    let mut out = vec![0u8; merged.len()];
                    grid::paste(&mut out, can.dims, bbox, &fine, |o, v| *o = v);
                    outputs.insert(&task.name, out);
                    continue;
                }
                let replaced: [bool; 256] =
                    std::array::from_fn(|v| task.replaces.contains(&(v as u8)));
                let mut changed = vec![false; fine.len()];
                {
                    let mut i = 0;
                    grid::paste(&mut merged, can.dims, bbox, &fine, |o, v| {
                        let new = if v != 0 {
                            v
                        } else if replaced[*o as usize] {
                            0
                        } else {
                            *o
                        };
                        changed[i] = new != *o;
                        *o = new;
                        i += 1;
                    });
                }
                grid::mark_nodes(&mut overridden, rs_dims, can.dims, bbox, &changed);
            }
        }
        drop(hu);

        let soft = SoftLabels::new(
            rs_dims,
            can.src_axis,
            can.flip,
            &labels,
            &runner_up,
            &margin,
            &overridden,
        );
        let labels = can.from_canonical(&merged);
        Ok(Segmentation { labels, soft })
    }
}
