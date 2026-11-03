mod bone;
mod context;
mod labels;
mod scene;
mod segmentation;
mod skin;
mod slices;
mod viewer;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use render::TransferFunction;
use volio::LoadOptions;

/// Photorealistic rendering of CT data.
#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// Folder containing a DICOM series (searched recursively) [default: the --scene's].
    #[arg(required_unless_present = "scene")]
    dicom_dir: Option<PathBuf>,
    /// Open a scene saved in the viewer: its dataset (unless DICOM_DIR is given),
    /// settings, camera and label looks, which replace the view options below.
    /// With --screenshot, renders it as seen in the viewer.
    #[arg(long, value_name = "JSON")]
    scene: Option<PathBuf>,
    /// SeriesInstanceUID to load (default: largest CT series).
    #[arg(long)]
    series: Option<String>,
    /// List the series found in the folder and exit.
    #[arg(long)]
    list_series: bool,
    /// Write axial/coronal/sagittal mid-slice PNGs to this folder and exit.
    #[arg(long, value_name = "DIR")]
    dump_slices: Option<PathBuf>,
    /// Window centre/width in HU for --dump-slices.
    #[arg(long, num_args = 2, value_names = ["CENTER", "WIDTH"], default_values_t = [40.0, 400.0])]
    window: Vec<f32>,
    /// Render one frame headless to this PNG and exit.
    #[arg(long, value_name = "PNG")]
    screenshot: Option<PathBuf>,
    /// Output size for --screenshot, or initial window size [default: 1280 960,
    /// or for --screenshot the --scene's image size].
    #[arg(long, num_args = 2, value_names = ["W", "H"])]
    size: Option<Vec<u32>>,
    /// Start the window fullscreen, showing only the render (F11 toggles).
    #[arg(long)]
    fullscreen: bool,
    /// Transfer-function preset (bone, angio, soft tissue).
    #[arg(long, default_value = "bone")]
    preset: String,
    /// Initial camera yaw/pitch in degrees (0 0 = anterior view).
    #[arg(long, num_args = 2, value_names = ["YAW", "PITCH"], default_values_t = [0.0, 0.0], allow_negative_numbers = true)]
    view: Vec<f32>,
    /// Initial zoom: divides the camera distance (2 = twice as close).
    #[arg(long, default_value_t = 1.0)]
    zoom: f32,
    /// Initial pan in fractions of the view height (right, up).
    #[arg(long, num_args = 2, value_names = ["DX", "DY"], default_values_t = [0.0, 0.0], allow_negative_numbers = true)]
    pan: Vec<f32>,
    /// TotalSegmentator ONNX models (see tools/totalseg/export_onnx.py) used to
    /// segment the volume at startup [default: $VOLVIEW_MODELS or data/models/totalseg].
    #[arg(long, value_name = "DIR")]
    models: Option<PathBuf>,
    /// Do not segment the volume at startup.
    #[arg(long, conflicts_with = "labels")]
    no_segment: bool,
    /// Segment, write the label map here (NIfTI, .nii or .nii.gz, plus a .json
    /// with the label names) and exit.
    #[arg(long, value_name = "NII", conflicts_with_all = ["labels", "no_segment"])]
    seg_out: Option<PathBuf>,
    /// Segment on the CPU instead of CUDA/TensorRT.
    #[arg(long)]
    cpu: bool,
    /// Load this label map (on the volume's grid) instead of segmenting.
    #[arg(long, value_name = "NII")]
    labels: Option<PathBuf>,
    /// Save the first fully rendered window frame, UI included, to this PNG and exit.
    #[arg(long, value_name = "PNG")]
    capture_ui: Option<PathBuf>,
    /// Ray step as a multiple of the finest voxel spacing.
    #[arg(long, value_name = "SCALE", default_value_t = 0.5)]
    step: f32,
    /// Pre-integrate the transfer function over each ray segment and shade opaque
    /// segments at their hit depth. Pays off with coarse steps or transfer
    /// functions with narrow opacity peaks; at the default step it only costs time.
    #[arg(long)]
    preintegrate: bool,
    /// Disable empty-space skipping (and delta tracking with brick bounds).
    #[arg(long)]
    no_skip: bool,
    /// Start with shadows off.
    #[arg(long)]
    no_shadows: bool,
    /// Start with ambient occlusion off.
    #[arg(long)]
    no_ao: bool,
    /// How labels are shown initially.
    #[arg(long, value_enum, default_value_t = LabelModeArg::Hybrid)]
    label_mode: LabelModeArg,
    /// Label groups to draw as translucent context (default: skeleton).
    #[arg(long, value_delimiter = ',', value_parser = labels::Group::parse, value_name = "GROUPS")]
    ghost: Option<Vec<labels::Group>>,
    /// Do not ghost anything (overrides the skeleton default).
    #[arg(long, conflicts_with = "ghost")]
    no_ghost: bool,
    /// How the skin (a shell at the body surface, derived from the CT) is drawn;
    /// --ghost and --hide override it.
    #[arg(long, value_enum, default_value_t = SkinArg::Ghost)]
    skin: SkinArg,
    /// Tint mode: how much label colours replace the transfer function's (0-1).
    #[arg(long, default_value_t = 0.7, value_name = "AMOUNT")]
    tint: f32,
    /// Label groups to hide.
    #[arg(long, value_delimiter = ',', value_parser = labels::Group::parse, value_name = "GROUPS")]
    hide: Vec<labels::Group>,
    /// Anatomical cutaway, repeatable: PLANE:SIDE[:OFFSET_MM], e.g. coronal:anterior
    /// or axial:superior:-40 (offset from the volume centre towards the removed side).
    #[arg(long, value_parser = viewer::ClipSpec::parse, value_name = "CUT")]
    clip: Vec<viewer::ClipSpec>,
    /// Remove only the region beyond every --clip plane (corner cutaway) instead
    /// of everything beyond any of them.
    #[arg(long)]
    clip_all: bool,
    /// Tint mode: also cut unlabelled tissue within this distance (mm) of hidden or
    /// ghosted labels, where segmentations stop short of the real boundary.
    #[arg(long, default_value_t = 2.0, value_name = "MM")]
    cut_margin: f32,
    /// Blur radius (voxels, 0-3) of label boundaries: smoother surfaces, but
    /// structures thinner than about twice the radius can vanish.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(0..=8))]
    label_smoothing: u32,
    /// Strength of subsurface scattering, scaling every material (0 = off).
    #[arg(long, default_value_t = 1.0, value_name = "STRENGTH")]
    subsurface: f32,
    /// Strength of the organs' procedural surface texture (colour variation and
    /// relief; 0 = off).
    #[arg(long, default_value_t = 1.0, value_name = "STRENGTH")]
    surface_detail: f32,
    /// Path trace instead of ray marching; for --screenshot, render this many
    /// paths per pixel [screenshots default: 512].
    #[arg(long, value_name = "SAMPLES", conflicts_with = "realtime")]
    path_trace: Option<u32>,
    /// Take --screenshot with the real-time ray marcher instead of the path tracer.
    #[arg(long)]
    realtime: bool,
    /// Default label colours and materials.
    #[arg(long, value_enum, default_value_t = PaletteArg::Anatomical)]
    palette: PaletteArg,
    /// Scales every label's wetness (clear coat); 0 is dry, 1 the defaults.
    #[arg(long, default_value_t = 1.0, value_name = "SCALE")]
    wetness: f32,
    /// Roughness of tissue without a label (transfer-function colours).
    #[arg(long, default_value_t = 0.5)]
    roughness: f32,
    /// Metalness of tissue without a label (for stylised renders).
    #[arg(long, default_value_t = 0.0)]
    metalness: f32,
    /// Surroundings that light the scene; `map` needs a light map and falls
    /// back to `studio` without one.
    #[arg(long, value_enum, default_value_t = EnvironmentArg::Map)]
    environment: EnvironmentArg,
    /// HDR light map (equirectangular Radiance .hdr) for `--environment map`
    /// [default: data/hdri/studio_small_09_2k.hdr if present].
    #[arg(long, value_name = "HDR")]
    env_map: Option<PathBuf>,
    /// Filter the density trilinearly instead of with a cubic B-spline (faster,
    /// but voxel and slice terraces show on surfaces).
    #[arg(long)]
    linear_sampling: bool,
    /// Forward scattering (Henyey-Greenstein g) of tissue in the path tracer.
    #[arg(long, default_value_t = 0.6, value_name = "G")]
    anisotropy: f32,
    /// Red translucency of tissue in the path tracer (0 = grey extinction).
    #[arg(long, default_value_t = 1.0, value_name = "STRENGTH")]
    chromatic: f32,
    /// Glow around bright areas (share of light spread by the lens, 0-0.3).
    #[arg(long, default_value_t = 0.04, value_name = "AMOUNT")]
    bloom: f32,
    /// Darkening towards the frame corners (0-1).
    #[arg(long, default_value_t = 0.25, value_name = "AMOUNT")]
    vignette: f32,
    /// Film grain amplitude in display values (0-0.08).
    #[arg(long, default_value_t = 0.012, value_name = "AMOUNT")]
    grain: f32,
    /// Keep the manual exposure instead of metering the drawn anatomy.
    #[arg(long)]
    no_auto_exposure: bool,
    /// Lens radius (mm) for depth of field in the path tracer (0 = all sharp).
    #[arg(long, default_value_t = 0.0, value_name = "MM")]
    aperture: f32,
    /// Focus on what is drawn at this image position (fractions from the top
    /// left), as Ctrl+clicking does [default: the orbit centre].
    #[arg(long, num_args = 2, value_names = ["U", "V"])]
    focus_at: Option<Vec<f32>>,
    /// Denoiser for the path-traced image.
    #[arg(long, value_enum, default_value_t = DenoiserArg::Oidn)]
    denoiser: DenoiserArg,
}

/// Paths per pixel of a --screenshot without --path-trace.
const DEFAULT_SCREENSHOT_PATHS: u32 = 512;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PaletteArg {
    Anatomical,
    Distinct,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SkinArg {
    Ghost,
    Shown,
    Hidden,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum EnvironmentArg {
    Map,
    Studio,
    Dome,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DenoiserArg {
    Oidn,
    Atrous,
    Off,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum LabelModeArg {
    Off,
    Tint,
    Isolate,
    Hybrid,
}

impl From<LabelModeArg> for render::LabelMode {
    fn from(m: LabelModeArg) -> Self {
        match m {
            LabelModeArg::Off => Self::Off,
            LabelModeArg::Tint => Self::Tint,
            LabelModeArg::Isolate => Self::Isolate,
            LabelModeArg::Hybrid => Self::Hybrid,
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ort=warn".into()),
        )
        .init();
    let args = Args::parse();
    let scene = args.scene.as_deref().map(scene::Scene::load).transpose()?;
    // The scene's dataset, unless a folder is given.
    let saved_dataset = scene
        .as_ref()
        .filter(|_| args.dicom_dir.is_none())
        .map(|s| &s.dataset);
    let dicom_dir = match (&args.dicom_dir, saved_dataset) {
        (Some(dir), _) => dir.clone(),
        (None, Some(d)) => d.dicom_dir.clone(),
        (None, None) => unreachable!("clap requires DICOM_DIR or --scene"),
    };
    let series = args
        .series
        .clone()
        .or_else(|| saved_dataset.and_then(|d| d.series.clone()));
    let labels_path = args
        .labels
        .clone()
        .or_else(|| saved_dataset.and_then(|d| d.labels.clone()));
    if !args.no_segment {
        segmentation::ensure_cuda_library_path();
    }

    if args.list_series {
        for s in volio::list_series(&dicom_dir)? {
            println!(
                "{}  {:>4} slices  {:<3} {}",
                s.uid, s.slices, s.modality, s.description
            );
        }
        return Ok(());
    }

    let presets = TransferFunction::presets();
    let preset = match &scene {
        Some(scene) => scene.settings.preset(),
        None => presets
            .iter()
            .position(|p| p.name.eq_ignore_ascii_case(&args.preset))
            .with_context(|| {
                let names: Vec<_> = presets.iter().map(|p| p.name).collect();
                format!("unknown preset {:?}; available: {names:?}", args.preset)
            })?,
    };

    let t = std::time::Instant::now();
    let volume = volio::load_series(&dicom_dir, &LoadOptions { series_uid: series })
        .with_context(|| format!("loading {}", dicom_dir.display()))?;
    let (lo, hi) = volume.hu_range();
    tracing::info!(
        dims = ?volume.dims,
        spacing_mm = ?volume.spacing,
        origin_mm = ?volume.origin,
        hu = ?(lo, hi),
        elapsed = ?t.elapsed(),
        "loaded volume"
    );

    let models = args
        .models
        .clone()
        .unwrap_or_else(segmentation::default_models_dir);
    if let Some(out) = &args.seg_out {
        let segmenter = seg::Segmenter::open(&models, !args.cpu)?;
        let result = segmenter.segment(&volume)?;
        return seg::write_labels(out, &volume, &result.labels, segmenter.labels())
            .with_context(|| format!("writing {}", out.display()));
    }

    // Labels: an explicit file, else the per-series cache, else segment (in the
    // background for the viewer, so the window opens right away).
    let mut labels = None;
    let mut segment = false;
    if let Some(path) = &labels_path {
        let (data, names) = seg::read_labels(path, &volume)
            .with_context(|| format!("loading labels {}", path.display()))?;
        labels = Some(labels::LabelSet::new(data, &names, &volume));
    } else if !args.no_segment && args.dump_slices.is_none() {
        labels = segmentation::load_cached(&volume);
        if labels.is_none() {
            segment = segmentation::models_available(&models);
            if !segment {
                tracing::warn!(
                    "no segmentation models in {}; pass --models or run tools/totalseg/export_onnx.py",
                    models.display()
                );
            }
        }
    }

    if let Some(out) = &args.dump_slices {
        return slices::dump_slices(&volume, out, args.window[0], args.window[1]);
    }

    let size = match (&args.size, &scene) {
        (Some(size), _) => (size[0], size[1]),
        (
            None,
            Some(scene::Scene {
                image_size: Some([w, h]),
                ..
            }),
        ) if args.screenshot.is_some() => (*w, *h),
        _ => (1280, 960),
    };
    let dataset = scene::Dataset {
        dicom_dir: std::fs::canonicalize(&dicom_dir).unwrap_or(dicom_dir),
        series: Some(volume.series_uid.clone()),
        labels: labels_path.map(|p| std::fs::canonicalize(&p).unwrap_or(p)),
    };
    // Screenshots are path traced unless asked otherwise: with as many paths
    // as the scene's image, if it was path traced.
    let screenshot_paths = || {
        scene
            .as_ref()
            .and_then(|s| s.settings.path_samples())
            .unwrap_or(DEFAULT_SCREENSHOT_PATHS)
    };
    let path_samples = args
        .path_trace
        .or((args.screenshot.is_some() && !args.realtime).then(screenshot_paths));
    let view = viewer::ViewOptions {
        size,
        dataset,
        scene_path: args.scene.clone().unwrap_or_else(|| "scene.json".into()),
        realtime: args.realtime,
        fullscreen: args.fullscreen,
        preset,
        yaw_pitch_deg: (args.view[0], args.view[1]),
        zoom: args.zoom,
        tint: args.tint.clamp(0.0, 1.0),
        env_map: args
            .env_map
            .clone()
            .or_else(|| scene.as_ref().and_then(|s| s.env_map.clone()))
            .or_else(|| {
                let default = PathBuf::from("data/hdri/studio_small_09_2k.hdr");
                default.exists().then_some(default)
            }),
        smooth_sampling: !args.linear_sampling,
        finish: viewer::Finish {
            auto_exposure: !args.no_auto_exposure,
            bloom: args.bloom.clamp(0.0, 0.3),
            vignette: args.vignette.clamp(0.0, 1.0),
            grain: args.grain.clamp(0.0, 0.08),
        },
        aperture: args.aperture.max(0.0),
        focus_at: args.focus_at.as_ref().map(|v| (v[0], v[1])),
        anisotropy: args.anisotropy.clamp(-0.95, 0.95),
        chromatic: args.chromatic,
        environment: match args.environment {
            EnvironmentArg::Map => render::Environment::Map,
            EnvironmentArg::Studio => render::Environment::Studio,
            EnvironmentArg::Dome => render::Environment::Dome,
        },
        pan: (args.pan[0], args.pan[1]),
        label_mode: args.label_mode.into(),
        capture: args.capture_ui.clone(),
        step_scale: args.step,
        preintegrate: args.preintegrate,
        skip_empty_space: !args.no_skip,
        clips: args.clip.clone(),
        clip_all: args.clip_all,
        cut_margin: args.cut_margin,
        label_smoothing: args.label_smoothing,
        subsurface: args.subsurface,
        surface_detail: args.surface_detail.max(0.0),
        path_samples,
        palette: match args.palette {
            PaletteArg::Anatomical => labels::Palette::Anatomical,
            PaletteArg::Distinct => labels::Palette::Distinct,
        },
        wetness: args.wetness.max(0.0),
        roughness: args.roughness.clamp(0.02, 1.0),
        metalness: args.metalness.clamp(0.0, 1.0),
        denoiser: match args.denoiser {
            DenoiserArg::Oidn => render::Denoiser::Oidn,
            DenoiserArg::Atrous => render::Denoiser::Atrous,
            DenoiserArg::Off => render::Denoiser::Off,
        },
        group_styles: {
            use labels::{Group, Style};
            let ghost = match (&args.ghost, args.no_ghost) {
                (_, true) => vec![],
                (Some(groups), false) => groups.clone(),
                (None, false) => vec![Group::Skeleton],
            };
            // Undo the skeleton default when it is not requested.
            let skin = match args.skin {
                SkinArg::Ghost => Style::Ghost,
                SkinArg::Shown => Style::Shown,
                SkinArg::Hidden => Style::Hidden,
            };
            let mut styles = vec![(Group::Skeleton, Style::Shown), (Group::Skin, skin)];
            styles.extend(ghost.into_iter().map(|g| (g, Style::Ghost)));
            styles.extend(args.hide.iter().map(|&g| (g, Style::Hidden)));
            styles
        },
        shadows: !args.no_shadows,
        ambient_occlusion: !args.no_ao,
        scene,
    };
    if let Some(labels) = labels.as_mut() {
        view.style_labels(labels);
    }
    match &args.screenshot {
        Some(path) => {
            if segment {
                let mut set = segmentation::compute(&volume, &models, !args.cpu, &mut |_| {})?;
                view.style_labels(&mut set);
                labels = Some(set);
            }
            viewer::screenshot(&volume, labels.as_ref(), &view, path)
        }
        None => {
            let volume = Arc::new(volume);
            let job = segment
                .then(|| segmentation::Job::spawn(Arc::clone(&volume), models, !args.cpu))
                .transpose()?;
            viewer::run(volume, labels, job, view)
        }
    }
}
