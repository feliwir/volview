use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use ash::vk;
use egui_ash_renderer::{DynamicRendering, Options, RenderMode};
use render::{
    AnatomicalPlane, Clipping, Context, Denoiser, Environment, FrameParams, LabelMode, Lighting,
    OrbitCamera, PathTracing, Presenter, Renderer, ToneMapper, TransferFunction,
};
use volio::Volume;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::raw_window_handle::HasDisplayHandle;
use winit::window::{Window, WindowId};

use crate::labels::{Group, LabelSet, Palette, Style};
use crate::segmentation::Job;

type UiRenderer = egui_ash_renderer::Renderer<egui_ash_renderer::allocator::GpuAllocator>;

#[derive(Debug, Clone)]
pub struct ViewOptions {
    pub size: (u32, u32),
    pub preset: usize,
    pub yaw_pitch_deg: (f32, f32),
    pub environment: Environment,
    /// HDR light map (equirectangular .hdr) for [`Environment::Map`].
    pub env_map: Option<std::path::PathBuf>,
    /// Cubic B-spline density filtering for surfaces and normals.
    pub smooth_sampling: bool,
    /// Bloom, vignette, grain and auto exposure.
    pub finish: Finish,
    /// Lens radius (mm) of the path tracer's camera.
    pub aperture: f32,
    /// Image position (0-1 from the top left) to focus on.
    pub focus_at: Option<(f32, f32)>,
    pub anisotropy: f32,
    pub chromatic: f32,
    pub palette: Palette,
    /// Scales every label's clear coat.
    pub wetness: f32,
    /// Material of tissue without a label.
    pub roughness: f32,
    pub metalness: f32,
    /// Tint mode: 0 keeps transfer-function colours, 1 uses label colours.
    pub tint: f32,
    /// Divides the default camera distance.
    pub zoom: f32,
    /// Moves the view right/up by fractions of the (zoomed) view height.
    pub pan: (f32, f32),
    pub label_mode: LabelMode,
    /// Save the first fully rendered window frame (UI included) here and exit.
    pub capture: Option<std::path::PathBuf>,
    /// Ray step as a fraction of the finest voxel spacing.
    pub step_scale: f32,
    pub preintegrate: bool,
    pub skip_empty_space: bool,
    pub shadows: bool,
    pub ambient_occlusion: bool,
    /// Initial label styles per group, applied in order.
    pub group_styles: Vec<(Group, Style)>,
    /// Initial cutaways.
    pub clips: Vec<ClipSpec>,
    /// Cut only where all clip planes cut (corner cutaway).
    pub clip_all: bool,
    /// Tint mode: extra cut around hidden/ghosted labels, mm.
    pub cut_margin: f32,
    /// Blur radius of the label masks, voxels (0-3).
    pub label_smoothing: u32,
    /// Scales all subsurface amounts (0 = off).
    pub subsurface: f32,
    /// Path trace with this many paths per pixel instead of ray marching.
    pub path_samples: Option<u32>,
    /// Denoise the path-traced image.
    pub denoiser: Denoiser,
}

/// A cutaway requested on the command line: `PLANE:SIDE[:OFFSET_MM]`, e.g.
/// `coronal:anterior` or `axial:superior:-40`. The offset is relative to the
/// volume centre, towards the removed side.
#[derive(Debug, Clone, Copy)]
pub struct ClipSpec {
    plane: AnatomicalPlane,
    cut_positive: bool,
    offset: f32,
}

impl ClipSpec {
    pub fn parse(s: &str) -> Result<Self, String> {
        let mut parts = s.split(':');
        let (plane, side) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
        let offset = match parts.next() {
            Some(o) => o.parse().map_err(|_| format!("bad clip offset {o:?}"))?,
            None => 0.0,
        };
        let (plane, cut_positive) = match (plane, side) {
            ("sagittal", "left") => (AnatomicalPlane::Sagittal, true),
            ("sagittal", "right") => (AnatomicalPlane::Sagittal, false),
            ("coronal", "posterior") => (AnatomicalPlane::Coronal, true),
            ("coronal", "anterior") => (AnatomicalPlane::Coronal, false),
            ("axial" | "transverse", "superior") => (AnatomicalPlane::Axial, true),
            ("axial" | "transverse", "inferior") => (AnatomicalPlane::Axial, false),
            _ => {
                return Err(format!(
                    "bad clip {s:?}: use sagittal:left|right, coronal:anterior|posterior or \
                     axial:superior|inferior, optionally followed by :OFFSET_MM"
                ));
            }
        };
        Ok(Self {
            plane,
            cut_positive,
            offset,
        })
    }
}

/// Names of the two sides of a plane, towards -LPS then +LPS.
fn plane_sides(plane: AnatomicalPlane) -> (&'static str, [&'static str; 2]) {
    match plane {
        AnatomicalPlane::Sagittal => ("Sagittal", ["right", "left"]),
        AnatomicalPlane::Coronal => ("Coronal", ["anterior", "posterior"]),
        AnatomicalPlane::Axial => ("Axial", ["inferior", "superior"]),
    }
}

impl ViewOptions {
    /// Applies the palette, wetness and group styles to a new label set.
    pub fn style_labels(&self, labels: &mut LabelSet) {
        if self.palette != Palette::Anatomical {
            labels.apply_palette(self.palette);
        }
        for e in &mut labels.entries {
            e.coat = (e.coat * self.wetness).min(1.0);
        }
        for &(group, style) in &self.group_styles {
            labels.set_group_style(group, style);
        }
    }
}

/// Reads an equirectangular HDR light map.
pub fn load_env_map(path: &std::path::Path) -> Result<render::EnvironmentMap> {
    let t = std::time::Instant::now();
    let img = image::open(path)
        .with_context(|| format!("reading light map {}", path.display()))?
        .into_rgb32f();
    let (width, height) = (img.width() as usize, img.height() as usize);
    let rgb = img.pixels().map(|p| p.0).collect();
    tracing::info!(path = %path.display(), width, height, elapsed = ?t.elapsed(), "loaded light map");
    Ok(render::EnvironmentMap { width, height, rgb })
}

fn set_env_map(ctx: &Context, renderer: &mut Renderer, view: &ViewOptions) -> Result<()> {
    if let Some(path) = &view.env_map {
        renderer.set_environment_map(ctx, Some(&load_env_map(path)?))?;
    }
    Ok(())
}

/// Camera finishing settings (see [`Lighting`]).
#[derive(Debug, Clone, Copy)]
pub struct Finish {
    pub auto_exposure: bool,
    pub bloom: f32,
    pub vignette: f32,
    pub grain: f32,
}

fn initial_camera(renderer: &Renderer, view: &ViewOptions) -> OrbitCamera {
    let mut camera = renderer.default_camera();
    camera.yaw = view.yaw_pitch_deg.0.to_radians();
    camera.pitch = view.yaw_pitch_deg.1.to_radians();
    camera.zoom(1.0 / view.zoom.max(1e-3));
    camera.pan(-view.pan.0, view.pan.1);
    camera
}

/// Renders a single frame without a window and saves it as PNG.
pub fn screenshot(
    volume: &Volume,
    labels: Option<&LabelSet>,
    view: &ViewOptions,
    path: &Path,
) -> Result<()> {
    let ctx = Context::new(None)?;
    let presets = TransferFunction::presets();
    let mut renderer = Renderer::new(&ctx, volume, &presets[view.preset])?;
    let mut settings = Settings::new(&renderer, view);
    set_env_map(&ctx, &mut renderer, view)?;
    if let Some(labels) = labels {
        renderer.set_labels(&ctx, &labels.data, labels.soft_grid())?;
    }
    if let Some((x, y)) = view.focus_at {
        let (w, h) = view.size;
        settings.path_tracing.focus = pick(
            volume,
            labels,
            &settings,
            egui::vec2(x, y),
            w as f32 / h as f32,
        );
        tracing::info!(focus = ?settings.path_tracing.focus, "focus");
    }
    let params = settings.frame_params(&renderer, labels);
    let (w, h) = view.size;
    let t = std::time::Instant::now();
    let frames = view.path_samples.map_or(1, |n| {
        n.div_ceil(settings.path_tracing.samples_per_frame.max(1))
    });
    let pixels = render::render_to_rgba(&ctx, &mut renderer, &params, w, h, frames)?;
    tracing::info!(elapsed = ?t.elapsed(), "rendered frame");
    renderer.destroy(&ctx);
    image::RgbaImage::from_raw(w, h, pixels)
        .expect("pixel buffer size")
        .save(path)?;
    tracing::info!("wrote {}", path.display());
    Ok(())
}

pub fn run(
    volume: Arc<Volume>,
    labels: Option<LabelSet>,
    job: Option<Job>,
    view: ViewOptions,
) -> Result<()> {
    let event_loop = EventLoop::new()?;
    let labels = LabelState {
        set: labels,
        job,
        error: None,
    };
    let mut app = App {
        volume,
        labels,
        view,
        gpu: None,
        error: None,
    };
    event_loop.run_app(&mut app)?;
    match app.error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// User-adjustable render settings shown in the debug panel.
struct Settings {
    camera: OrbitCamera,
    preset: usize,
    /// Ray step as a fraction of the finest voxel spacing.
    step_scale: f32,
    /// Volume render resolution relative to the viewport's pixels.
    resolution_scale: f32,
    /// Re-render every frame (for timing) instead of only on change.
    continuous: bool,
    label_mode: LabelMode,
    tint: f32,
    extinction: f32,
    label_filter: String,
    palette: Palette,
    lighting: Lighting,
    preintegrate: bool,
    skip_empty_space: bool,
    smooth_sampling: bool,
    ghost_opacity: f32,
    /// Tint mode: extra cut around hidden/ghosted labels, mm.
    cut_margin: f32,
    label_smoothing: u32,
    clipping: Clipping,
    /// Path trace instead of ray marching in real time.
    path_trace: bool,
    path_tracing: PathTracing,
    /// Path tracing stops refining after this many paths per pixel.
    max_samples: u32,
}

impl Settings {
    fn new(renderer: &Renderer, view: &ViewOptions) -> Self {
        Self {
            camera: initial_camera(renderer, view),
            preset: view.preset,
            step_scale: view.step_scale,
            resolution_scale: 1.0,
            continuous: false,
            label_mode: view.label_mode,
            tint: view.tint,
            extinction: 0.5,
            label_filter: String::new(),
            palette: view.palette,
            lighting: Lighting {
                shadows: view.shadows,
                subsurface_strength: view.subsurface,
                ambient_occlusion: view.ambient_occlusion,
                environment: view.environment,
                roughness: view.roughness,
                metalness: view.metalness,
                anisotropy: view.anisotropy,
                auto_exposure: view.finish.auto_exposure,
                bloom: view.finish.bloom,
                vignette: view.finish.vignette,
                grain: view.finish.grain,
                chromatic: view.chromatic,
                ..Lighting::default()
            },
            preintegrate: view.preintegrate,
            skip_empty_space: view.skip_empty_space,
            smooth_sampling: view.smooth_sampling,
            ghost_opacity: 0.08,
            cut_margin: view.cut_margin,
            label_smoothing: view.label_smoothing,
            clipping: initial_clipping(renderer, view),
            path_trace: view.path_samples.is_some(),
            path_tracing: PathTracing {
                denoiser: view.denoiser,
                aperture: view.aperture,
                ..PathTracing::default()
            },
            max_samples: view.path_samples.unwrap_or(1024),
        }
    }

    fn frame_params(&self, renderer: &Renderer, labels: Option<&LabelSet>) -> FrameParams {
        FrameParams {
            camera: self.camera,
            step_mm: self.step_scale * renderer.min_spacing,
            srgb_encode: true,
            labels: labels
                .map(|l| {
                    l.params(
                        self.label_mode,
                        self.tint,
                        self.extinction,
                        self.ghost_opacity,
                        self.cut_margin,
                        self.label_smoothing,
                    )
                })
                .unwrap_or_default(),
            lighting: self.lighting,
            preintegrate: self.preintegrate,
            skip_empty_space: self.skip_empty_space,
            clipping: self.clipping,
            path_tracing: self.path_trace.then_some(self.path_tracing),
            smooth_sampling: self.smooth_sampling,
        }
    }
}

/// The volume render target as an egui texture.
struct VolumeTexture {
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    sampler: vk::Sampler,
    id: egui::TextureId,
}

impl VolumeTexture {
    fn new(ctx: &Context, ui: &mut UiRenderer) -> Result<Self> {
        let device = &ctx.device;
        let layout = egui_ash_renderer::vulkan::create_vulkan_descriptor_set_layout(device)?;
        let sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
        }];
        unsafe {
            let pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )?;
            let set = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&[layout]),
            )?[0];
            let sampler = device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?;
            Ok(Self {
                layout,
                pool,
                set,
                sampler,
                id: ui.add_user_texture(set),
            })
        }
    }

    /// Points the texture at a new image view. The GPU must not be using the set.
    fn update(&self, ctx: &Context, view: vk::ImageView) {
        let info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&info);
        unsafe { ctx.device.update_descriptor_sets(&[write], &[]) };
    }

    fn destroy(&self, ctx: &Context) {
        unsafe {
            ctx.device.destroy_sampler(self.sampler, None);
            ctx.device.destroy_descriptor_pool(self.pool, None);
            ctx.device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

struct Gpu {
    window: Arc<Window>,
    renderer: Renderer,
    presenter: Presenter,
    volume_tex: VolumeTexture,
    egui_ctx: egui::Context,
    egui_winit: egui_winit::State,
    // Dropped after `Drop::drop` destroys the objects above; the UI renderer
    // frees through the shared allocator, so it must go before the context.
    ui: UiRenderer,
    ctx: Context,
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
        }
        self.presenter.destroy(&self.ctx);
        self.volume_tex.destroy(&self.ctx);
        self.renderer.destroy(&self.ctx);
    }
}

/// Facts about the scene shown in the info section.
struct Info {
    device: String,
    dims: [u32; 3],
    spacing: [f64; 3],
    voxel_ml: f64,
}

/// Labels shown in the viewer and the segmentation that may still produce them.
struct LabelState {
    set: Option<LabelSet>,
    job: Option<Job>,
    error: Option<String>,
}

struct App {
    volume: Arc<Volume>,
    labels: LabelState,
    view: ViewOptions,
    gpu: Option<(Gpu, Viewer)>,
    error: Option<anyhow::Error>,
}

/// Per-window viewer state that is not GPU-owned.
struct Viewer {
    settings: Settings,
    volume: Arc<Volume>,
    /// Image position (0-1) Ctrl+clicked to focus on, handled in `redraw`.
    focus_request: Option<egui::Vec2>,
    view: ViewOptions,
    capture: Option<std::path::PathBuf>,
    /// Camera that frames the whole volume.
    home: OrbitCamera,
    /// Patient-space bounds of the volume (mm), for the clip-plane sliders.
    bounds: (glam::Vec3, glam::Vec3),
    info: Info,
    /// Render target size in pixels.
    target: (u32, u32),
    /// Parameters of the image currently in the render target.
    rendered: Option<FrameParams>,
    quit: bool,
    /// Paths per pixel of the current path-traced image.
    samples: u32,
    /// Frames presented since the volume was last (re)rendered.
    settled_frames: u32,
    /// `VOLVIEW_BENCH`: orbit every frame and log the mean GPU time of this many frames.
    bench: Option<(u32, Vec<f32>)>,
}

impl App {
    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title("volview")
                    .with_inner_size(LogicalSize::new(self.view.size.0, self.view.size.1)),
            )?,
        );
        let ctx = Context::new(Some(event_loop.display_handle()?.as_raw()))?;
        let presets = TransferFunction::presets();
        let mut renderer = Renderer::new(&ctx, &self.volume, &presets[self.view.preset])?;
        set_env_map(&ctx, &mut renderer, &self.view)?;
        if let Some(labels) = &self.labels.set {
            renderer.set_labels(&ctx, &labels.data, labels.soft_grid())?;
        }
        let size = window.inner_size();
        let presenter = Presenter::new(&ctx, window.as_ref(), size.width, size.height)?;
        let mut ui = egui_ash_renderer::Renderer::with_gpu_allocator(
            Arc::clone(&ctx.allocator),
            ctx.device.clone(),
            RenderMode::DynamicRendering(DynamicRendering {
                color_attachment_format: presenter.format(),
                depth_attachment_format: None,
                stencil_attachment_format: None,
            }),
            Options {
                in_flight_frames: render::FRAMES_IN_FLIGHT,
                srgb_framebuffer: presenter.is_srgb(),
                ..Default::default()
            },
        )?;
        let volume_tex = VolumeTexture::new(&ctx, &mut ui)?;

        let egui_ctx = egui::Context::default();
        let max_side = unsafe {
            ctx.instance
                .get_physical_device_properties(ctx.physical_device)
        }
        .limits
        .max_image_dimension2_d;
        let egui_winit = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(max_side as usize),
        );

        let v = &self.volume;
        let info = Info {
            device: ctx.device_name.clone(),
            dims: v.dims.to_array(),
            spacing: v.spacing.to_array(),
            voxel_ml: v.spacing.x * v.spacing.y * v.spacing.z / 1000.0,
        };
        let viewer = Viewer {
            settings: Settings::new(&renderer, &self.view),
            volume: Arc::clone(&self.volume),
            focus_request: None,
            view: self.view.clone(),
            capture: self.view.capture.clone(),
            home: renderer.default_camera(),
            bounds: renderer.bounds,
            info,
            target: (0, 0),
            rendered: None,
            quit: false,
            settled_frames: 0,
            samples: 0,
            bench: std::env::var("VOLVIEW_BENCH")
                .ok()
                .map(|n| (n.parse().unwrap_or(200), Vec::new())),
        };
        self.gpu = Some((
            Gpu {
                window,
                renderer,
                presenter,
                volume_tex,
                egui_ctx,
                egui_winit,
                ui,
                ctx,
            },
            viewer,
        ));
        Ok(())
    }

    fn handle(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) -> Result<()> {
        let Some((gpu, viewer)) = self.gpu.as_mut() else {
            return Ok(());
        };
        let response = gpu.egui_winit.on_window_event(&gpu.window, &event);
        if response.repaint {
            gpu.window.request_redraw();
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                gpu.presenter.recreate(&gpu.ctx, size.width, size.height)?;
                gpu.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                redraw(gpu, viewer, &mut self.labels)?;
                if viewer.quit {
                    event_loop.exit();
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Runs the UI, re-renders the volume if anything changed, and presents.
fn redraw(gpu: &mut Gpu, viewer: &mut Viewer, labels: &mut LabelState) -> Result<()> {
    if let Some(result) = labels.job.as_ref().and_then(Job::try_take) {
        labels.job = None;
        match result {
            Ok(mut set) => {
                viewer.view.style_labels(&mut set);
                gpu.renderer
                    .set_labels(&gpu.ctx, &set.data, set.soft_grid())?;
                tracing::info!(labels = set.entries.len(), "labels ready");
                labels.set = Some(set);
                viewer.rendered = None;
            }
            Err(e) => {
                tracing::error!("segmentation failed: {e:#}");
                labels.error = Some(format!("{e:#}"));
            }
        }
    }

    let compute_ms = gpu.presenter.compute_ms();
    viewer.samples = gpu.renderer.accumulated_samples();
    if let Some((frames, times)) = &mut viewer.bench {
        viewer.settings.camera.orbit(0.01, 0.0);
        times.extend(compute_ms);
        if times.len() as u32 >= *frames {
            let mean = times.iter().sum::<f32>() / times.len() as f32;
            tracing::info!(frames = times.len(), mean_ms = mean, target = ?viewer.target, "benchmark (orbiting)");
            viewer.quit = true;
        }
    }
    let raw_input = gpu.egui_winit.take_egui_input(&gpu.window);
    let mut preset_changed = false;
    let egui_ctx = gpu.egui_ctx.clone();
    let mut output = egui_ctx.run_ui(raw_input, |ui| {
        preset_changed = panel(ui, viewer, labels, compute_ms);
        viewport(ui, viewer, gpu.volume_tex.id);
    });
    if let Some(uv) = viewer.focus_request.take() {
        let aspect = viewer.target.0 as f32 / viewer.target.1.max(1) as f32;
        let hit = pick(
            &viewer.volume,
            labels.set.as_ref(),
            &viewer.settings,
            uv,
            aspect,
        );
        tracing::info!(?hit, "focus");
        if hit.is_some() {
            viewer.settings.path_tracing.focus = hit;
        }
    }
    gpu.egui_winit
        .handle_platform_output(&gpu.window, output.platform_output);

    let ctx = &gpu.ctx;
    if preset_changed {
        let presets = TransferFunction::presets();
        gpu.renderer
            .set_transfer_function(ctx, &presets[viewer.settings.preset])?;
        viewer.rendered = None;
    }
    let (w, h) = viewer.target;
    let current = gpu
        .renderer
        .output()
        .map(|o| (o.extent.width, o.extent.height));
    if w > 0 && h > 0 && current != Some((w, h)) {
        gpu.renderer.resize(ctx, w, h)?;
        gpu.volume_tex.update(ctx, gpu.renderer.output_srgb_view());
        viewer.rendered = None;
    }

    for (id, deltas) in output.textures_delta.set.drain() {
        for delta in &deltas {
            gpu.ui
                .set_texture(ctx.queue, ctx.command_pool(), id, delta)?;
        }
    }
    let ppp = output.pixels_per_point;
    let primitives = gpu.egui_ctx.tessellate(output.shapes, ppp);
    let params = viewer
        .settings
        .frame_params(&gpu.renderer, labels.set.as_ref());
    // A path-traced image keeps refining until it has enough samples.
    let refining = viewer.settings.path_trace
        && gpu.renderer.accumulated_samples() < viewer.settings.max_samples;
    let render = gpu.renderer.output().is_some()
        && (viewer.settings.continuous || refining || viewer.rendered != Some(params));
    let (renderer, ui) = (&mut gpu.renderer, &mut gpu.ui);
    let mut draw_result = Ok(());
    let presented = gpu.presenter.draw(
        ctx,
        |cmd, frame| {
            if render {
                renderer.record(
                    ctx,
                    cmd,
                    frame,
                    &params,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
            }
            render
        },
        |cmd, extent| draw_result = ui.cmd_draw(cmd, extent, ppp, &primitives),
    )?;
    draw_result?;
    // OIDN runs after the frame that added samples: for each of the first few, then
    // at powers of two and when done. In between the last result stays on screen.
    if render
        && viewer.settings.path_trace
        && viewer.settings.path_tracing.denoiser == Denoiser::Oidn
    {
        let n = gpu.renderer.accumulated_samples();
        if n <= 8 || n.is_power_of_two() || n >= viewer.settings.max_samples {
            if let Err(e) = gpu
                .renderer
                .denoise_oidn(ctx, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            {
                tracing::warn!("OIDN unavailable, using the à-trous filter: {e:#}");
                viewer.settings.path_tracing.denoiser = Denoiser::Atrous;
            }
        }
    }
    if render {
        viewer.rendered = Some(params);
        viewer.settled_frames = 0;
    } else {
        viewer.settled_frames += 1;
    }
    if let Some((w, h, pixels)) = gpu.presenter.take_capture() {
        let path = viewer.capture.take().expect("capture requested");
        image::RgbaImage::from_raw(w, h, pixels)
            .expect("capture size")
            .save(&path)?;
        tracing::info!("wrote {}", path.display());
        viewer.quit = true;
    } else if viewer.capture.is_some() && viewer.settled_frames >= 3 && labels.job.is_none() {
        // Let egui's layout settle for a few frames before capturing.
        gpu.presenter.capture_next_frame();
    }
    if !presented {
        let size = gpu.window.inner_size();
        gpu.presenter.recreate(ctx, size.width, size.height)?;
        gpu.window.request_redraw();
    }
    if !output.textures_delta.free.is_empty() {
        unsafe { ctx.device.device_wait_idle()? };
        for id in output.textures_delta.free.drain() {
            gpu.ui.free_texture(id)?;
        }
    }

    let repaint_now = output
        .viewport_output
        .get(&egui::ViewportId::ROOT)
        .is_some_and(|v| v.repaint_delay.is_zero());
    if repaint_now
        || viewer.settings.continuous
        || refining
        || viewer.rendered.is_none()
        || viewer.capture.is_some()
        || viewer.bench.is_some()
    {
        gpu.window.request_redraw();
    }
    Ok(())
}

/// Debug controls. Returns whether the transfer-function preset changed.
fn panel(
    ui: &mut egui::Ui,
    viewer: &mut Viewer,
    labels: &mut LabelState,
    compute_ms: Option<f32>,
) -> bool {
    let bounds = viewer.bounds;
    let has_env_map = viewer.view.env_map.is_some();
    let s = &mut viewer.settings;
    let mut preset_changed = false;
    let presets = TransferFunction::presets();

    // Keyboard shortcuts, unless a text field has focus.
    if !ui.ctx().egui_wants_keyboard_input() {
        ui.input(|i| {
            if i.key_pressed(egui::Key::Escape) {
                viewer.quit = true;
            }
            if i.key_pressed(egui::Key::R) {
                s.camera = viewer.home;
            }
            let digits = [
                egui::Key::Num1,
                egui::Key::Num2,
                egui::Key::Num3,
                egui::Key::Num4,
            ];
            for (i_preset, key) in digits.iter().enumerate().take(presets.len()) {
                if i.key_pressed(*key) && s.preset != i_preset {
                    s.preset = i_preset;
                    preset_changed = true;
                }
            }
        });
    }

    egui::Panel::left("controls")
        .resizable(true)
        .default_size(300.0)
        .show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("volview");
                let info = &viewer.info;
                egui::Grid::new("info").num_columns(2).show(ui, |ui| {
                    ui.label("GPU");
                    ui.label(&info.device);
                    ui.end_row();
                    ui.label("Volume");
                    ui.label(format!(
                        "{} × {} × {}",
                        info.dims[0], info.dims[1], info.dims[2]
                    ));
                    ui.end_row();
                    ui.label("Spacing");
                    ui.label(format!(
                        "{:.3} × {:.3} × {:.3} mm",
                        info.spacing[0], info.spacing[1], info.spacing[2]
                    ));
                    ui.end_row();
                    ui.label("Render");
                    ui.label(format!("{} × {}", viewer.target.0, viewer.target.1));
                    ui.end_row();
                    ui.label("GPU time");
                    ui.label(compute_ms.map_or("–".into(), |ms| format!("{ms:.2} ms")));
                    ui.end_row();
                    if s.path_trace {
                        ui.label("Samples");
                        ui.label(format!("{} / {}", viewer.samples, s.max_samples));
                        ui.end_row();
                    }
                });

                ui.separator();
                egui::CollapsingHeader::new("Rendering")
                    .default_open(true)
                    .show(ui, |ui| {
                        egui::ComboBox::from_label("Preset")
                            .selected_text(presets[s.preset].name)
                            .show_ui(ui, |ui| {
                                for (i, p) in presets.iter().enumerate() {
                                    preset_changed |=
                                        ui.selectable_value(&mut s.preset, i, p.name).changed();
                                }
                            });
                        ui.add(
                            egui::Slider::new(&mut s.step_scale, 0.1..=2.0)
                                .text("Step (× spacing)")
                                .logarithmic(true),
                        );
                        ui.add(
                            egui::Slider::new(&mut s.resolution_scale, 0.25..=1.0)
                                .text("Resolution"),
                        );
                        ui.checkbox(&mut s.preintegrate, "Pre-integrated transfer function");
                        ui.checkbox(&mut s.skip_empty_space, "Skip empty space");
                        ui.checkbox(&mut s.smooth_sampling, "Smooth (cubic) sampling")
                            .on_hover_text(
                                "B-spline filtered density for surfaces and normals: \
                                 no voxel or slice terraces, but slower",
                            );
                        ui.checkbox(&mut s.continuous, "Render every frame");
                        ui.horizontal(|ui| {
                            ui.selectable_value(&mut s.path_trace, false, "Real-time");
                            ui.selectable_value(&mut s.path_trace, true, "Path traced");
                        });
                        if s.path_trace {
                            let pt = &mut s.path_tracing;
                            ui.add(egui::Slider::new(&mut pt.bounces, 1..=8).text("Bounces"));
                            ui.add(
                                egui::Slider::new(&mut pt.samples_per_frame, 1..=16)
                                    .text("Paths / frame"),
                            );
                            ui.add(
                                egui::Slider::new(&mut s.max_samples, 16..=8192)
                                    .text("Max paths / pixel")
                                    .logarithmic(true),
                            );
                            ui.add(
                                egui::Slider::new(&mut s.path_tracing.aperture, 0.0..=20.0)
                                    .text("Aperture (mm)"),
                            )
                            .on_hover_text("Lens radius: depth of field. Ctrl+click to focus.");
                            if s.path_tracing.aperture > 0.0 {
                                ui.horizontal(|ui| {
                                    ui.label(if s.path_tracing.focus.is_some() {
                                        "Focus: picked point"
                                    } else {
                                        "Focus: orbit centre"
                                    });
                                    if s.path_tracing.focus.is_some()
                                        && ui.button("Reset").clicked()
                                    {
                                        s.path_tracing.focus = None;
                                    }
                                });
                            }
                            ui.horizontal(|ui| {
                                ui.label("Denoise");
                                let d = &mut s.path_tracing.denoiser;
                                ui.selectable_value(d, Denoiser::Oidn, "OIDN")
                                    .on_hover_text(
                                        "Intel Open Image Denoise (neural, CUDA or CPU)",
                                    );
                                ui.selectable_value(d, Denoiser::Atrous, "À-trous")
                                    .on_hover_text(
                                        "Edge-aware GPU filter; fades out as samples accumulate",
                                    );
                                ui.selectable_value(d, Denoiser::Off, "Off");
                            });
                        }
                    })
                    .header_response
                    .on_hover_text("1–4: presets");

                egui::CollapsingHeader::new("Lighting")
                    .default_open(true)
                    .show(ui, |ui| lighting_controls(ui, &mut s.lighting, has_env_map));

                egui::CollapsingHeader::new("Clipping")
                    .default_open(true)
                    .show(ui, |ui| clipping_controls(ui, &mut s.clipping, bounds));

                egui::CollapsingHeader::new("Camera")
                    .default_open(true)
                    .show(ui, |ui| {
                        // Sliders clamp their value every frame by default, so they
                        // only write back when edited: orbiting keeps the yaw in
                        // 0-360 degrees, and a clamped copy would flip the view.
                        let mut yaw = s.camera.yaw.to_degrees();
                        if yaw > 180.0 {
                            yaw -= 360.0;
                        }
                        let mut pitch = s.camera.pitch.to_degrees();
                        if ui
                            .add(egui::Slider::new(&mut yaw, -180.0..=180.0).text("Yaw °"))
                            .changed()
                        {
                            s.camera.yaw = yaw.to_radians().rem_euclid(std::f32::consts::TAU);
                        }
                        if ui
                            .add(egui::Slider::new(&mut pitch, -89.0..=89.0).text("Pitch °"))
                            .changed()
                        {
                            s.camera.pitch = pitch.to_radians();
                        }
                        ui.add(
                            egui::Slider::new(&mut s.camera.distance, 10.0..=5000.0)
                                .text("Distance mm")
                                .logarithmic(true)
                                .clamping(egui::SliderClamping::Edits),
                        );
                        if ui.button("Reset view (R)").clicked() {
                            s.camera = viewer.home;
                        }
                        ui.label("Drag: orbit · Shift/middle drag: pan · Right drag/wheel: zoom · Double-click: reset view");
                    });

                if let Some(job) = &labels.job {
                    ui.separator();
                    ui.label("Segmenting with TotalSegmentator…");
                    ui.add(egui::ProgressBar::new(job.progress()).show_percentage());
                }
                if let Some(err) = &labels.error {
                    ui.separator();
                    ui.colored_label(
                        ui.visuals().error_fg_color,
                        format!("Segmentation failed: {err}"),
                    );
                }
                if let Some(labels) = labels.set.as_mut() {
                    egui::CollapsingHeader::new("Labels")
                        .default_open(true)
                        .show(ui, |ui| label_controls(ui, s, labels, viewer.info.voxel_ml));
                }
            });
        });
    preset_changed
}

/// Planes start at the volume centre; command-line cuts are enabled.
fn initial_clipping(renderer: &Renderer, view: &ViewOptions) -> Clipping {
    let center = (renderer.bounds.0 + renderer.bounds.1) * 0.5;
    let mut clipping = Clipping {
        all: view.clip_all,
        ..Clipping::default()
    };
    for p in &mut clipping.planes {
        p.position = center[p.plane.axis()];
    }
    for spec in &view.clips {
        let p = &mut clipping.planes[spec.plane.axis()];
        p.enabled = true;
        p.cut_positive = spec.cut_positive;
        // Offsets point towards the removed side.
        p.position += if spec.cut_positive {
            spec.offset
        } else {
            -spec.offset
        };
    }
    clipping
}

fn clipping_controls(ui: &mut egui::Ui, c: &mut Clipping, bounds: (glam::Vec3, glam::Vec3)) {
    for p in &mut c.planes {
        let (name, sides) = plane_sides(p.plane);
        ui.horizontal(|ui| {
            ui.checkbox(&mut p.enabled, name);
            ui.label("remove");
            ui.selectable_value(&mut p.cut_positive, false, sides[0]);
            ui.selectable_value(&mut p.cut_positive, true, sides[1]);
        });
        let a = p.plane.axis();
        ui.add_enabled(
            p.enabled,
            egui::Slider::new(&mut p.position, bounds.0[a]..=bounds.1[a])
                .text(format!("{} to {} (mm)", sides[0], sides[1])),
        );
    }
    ui.horizontal(|ui| {
        ui.label("Cut where");
        ui.selectable_value(&mut c.all, false, "any plane cuts");
        ui.selectable_value(&mut c.all, true, "all cut (corner)");
    })
    .response
    .on_hover_text(
        "Any: halves and slabs. All: only the region beyond every plane, e.g. one octant.",
    );
}

fn lighting_controls(ui: &mut egui::Ui, l: &mut Lighting, has_env_map: bool) {
    let mut azimuth = l.azimuth.to_degrees();
    let mut elevation = l.elevation.to_degrees();
    ui.add(egui::Slider::new(&mut azimuth, -180.0..=180.0).text("Light azimuth °"));
    ui.add(egui::Slider::new(&mut elevation, -89.0..=89.0).text("Light elevation °"));
    l.azimuth = azimuth.to_radians();
    l.elevation = elevation.to_radians();
    ui.add(egui::Slider::new(&mut l.intensity, 0.0..=20.0).text("Key light"));
    ui.add(egui::Slider::new(&mut l.ambient, 0.0..=2.0).text("Ambient"));
    ui.horizontal(|ui| {
        ui.label("Environment");
        if has_env_map {
            ui.selectable_value(&mut l.environment, Environment::Map, "Light map")
                .on_hover_text("HDR photo of a studio (--env-map)");
        }
        ui.selectable_value(&mut l.environment, Environment::Studio, "Studio")
            .on_hover_text("Soft fill and rim light in dark surroundings, black background");
        ui.selectable_value(&mut l.environment, Environment::Dome, "Dome")
            .on_hover_text("Evenly lit from all sides, brighter above");
    });
    let mut size = l.light_size.to_degrees();
    ui.add(egui::Slider::new(&mut size, 0.0..=30.0).text("Key light size (°)"))
        .on_hover_text("Larger lights cast softer shadows (path tracer only)");
    l.light_size = size.to_radians();
    if has_env_map && l.environment == Environment::Map {
        let mut rotation = l.env_rotation.to_degrees();
        ui.add(egui::Slider::new(&mut rotation, -180.0..=180.0).text("Light map rotation °"))
            .on_hover_text("0: the map's brightest light on the key light's side");
        l.env_rotation = rotation.to_radians();
    }
    ui.add(egui::Slider::new(&mut l.anisotropy, 0.0..=0.95).text("Forward scattering"))
        .on_hover_text("Phase function anisotropy g of tissue (path tracer only)");
    ui.add(egui::Slider::new(&mut l.chromatic, 0.0..=1.0).text("Red translucency"))
        .on_hover_text("Red light travels further through tissue than blue (path tracer only)");
    ui.horizontal(|ui| {
        ui.checkbox(&mut l.shadows, "Shadows");
        ui.checkbox(&mut l.ambient_occlusion, "Ambient occlusion");
    });
    ui.add(egui::Slider::new(&mut l.exposure, -4.0..=4.0).text("Exposure (EV)"));
    ui.checkbox(&mut l.auto_exposure, "Auto exposure")
        .on_hover_text("Meter the drawn anatomy (not the background), like a camera");
    ui.add(egui::Slider::new(&mut l.bloom, 0.0..=0.3).text("Bloom"))
        .on_hover_text("Glow around bright areas, as scattered in a real lens");
    ui.add(egui::Slider::new(&mut l.vignette, 0.0..=1.0).text("Vignette"));
    ui.add(egui::Slider::new(&mut l.grain, 0.0..=0.08).text("Film grain"));
    egui::ComboBox::from_label("Tone mapping")
        .selected_text(format!("{:?}", l.tone_mapper))
        .show_ui(ui, |ui| {
            for op in [ToneMapper::Aces, ToneMapper::Agx, ToneMapper::None] {
                ui.selectable_value(&mut l.tone_mapper, op, format!("{op:?}"));
            }
        });
    ui.label("Unlabelled tissue");
    ui.add(egui::Slider::new(&mut l.roughness, 0.02..=1.0).text("Roughness"));
    ui.add(egui::Slider::new(&mut l.metalness, 0.0..=1.0).text("Metalness"));
    ui.add(egui::Slider::new(&mut l.subsurface, 0.0..=1.0).text("Subsurface"));
    ui.add(
        egui::Slider::new(&mut l.scattering_distance, 1.0..=50.0)
            .text("Scattering distance (mm)")
            .logarithmic(true),
    )
    .on_hover_text("How far light diffuses through tissue (all materials)");
    ui.add(egui::Slider::new(&mut l.subsurface_strength, 0.0..=1.0).text("Subsurface strength"))
        .on_hover_text("Scales every material's subsurface amount; 0 turns it off");
}

fn label_controls(ui: &mut egui::Ui, s: &mut Settings, labels: &mut LabelSet, voxel_ml: f64) {
    ui.horizontal(|ui| {
        ui.selectable_value(&mut s.label_mode, LabelMode::Off, "Off");
        ui.selectable_value(&mut s.label_mode, LabelMode::Tint, "Tint");
        ui.selectable_value(&mut s.label_mode, LabelMode::Isolate, "Isolate");
        ui.selectable_value(&mut s.label_mode, LabelMode::Hybrid, "Hybrid")
            .on_hover_text("Only labelled structures, with surfaces and texture from the CT");
    });
    ui.horizontal(|ui| {
        ui.label("Palette");
        for palette in [Palette::Anatomical, Palette::Distinct] {
            if ui
                .selectable_value(&mut s.palette, palette, format!("{palette:?}"))
                .clicked()
            {
                labels.apply_palette(palette);
            }
        }
    })
    .response
    .on_hover_text("Resets all label colours and materials");
    match s.label_mode {
        LabelMode::Tint => {
            ui.add(egui::Slider::new(&mut s.tint, 0.0..=1.0).text("Tint"));
        }
        LabelMode::Isolate | LabelMode::Hybrid => {
            ui.add(
                egui::Slider::new(&mut s.extinction, 0.01..=10.0)
                    .text("Opacity (1/mm)")
                    .logarithmic(true),
            );
        }
        LabelMode::Off => {}
    }
    ui.add(
        egui::Slider::new(&mut s.ghost_opacity, 0.0..=0.5)
            .text("Ghost opacity")
            .clamping(egui::SliderClamping::Always),
    )
    .on_hover_text(
        "Opacity of ghosted structures where they face the viewer; silhouettes stay opaque",
    );
    ui.add(egui::Slider::new(&mut s.label_smoothing, 0..=8).text("Label smoothing"))
        .on_hover_text(
            "Blur radius (voxels) of label boundaries. Higher is smoother but can \
             remove structures thinner than about twice the radius.",
        );
    if s.label_mode == LabelMode::Tint {
        ui.add(egui::Slider::new(&mut s.cut_margin, 0.0..=3.0).text("Cut margin (mm)"))
            .on_hover_text(
                "Also removes unlabelled tissue this close to hidden or ghosted labels. \
                 Segmentations end slightly inside the real boundary, so without a margin \
                 hidden bones leave their outer rim behind.",
            );
    }

    // Whole anatomical systems at once.
    egui::Grid::new("label groups")
        .num_columns(4)
        .show(ui, |ui| {
            for group in Group::ALL {
                // The group's style if all its labels share one.
                let mut styles = labels
                    .entries
                    .iter()
                    .filter(|e| e.group == group)
                    .map(|e| e.style);
                let Some(first) = styles.next() else { continue };
                let common = styles.all(|st| st == first).then_some(first);
                ui.label(group.name());
                for style in [Style::Shown, Style::Ghost, Style::Hidden] {
                    if ui
                        .selectable_label(common == Some(style), style.name())
                        .clicked()
                    {
                        labels
                            .entries
                            .iter_mut()
                            .filter(|e| e.group == group)
                            .for_each(|e| e.style = style);
                    }
                }
                ui.end_row();
            }
        });

    ui.horizontal(|ui| {
        ui.label("Filter");
        ui.text_edit_singleline(&mut s.label_filter);
    });
    let filter = s.label_filter.to_lowercase();
    let matches = |name: &str| filter.is_empty() || name.to_lowercase().contains(&filter);
    ui.horizontal(|ui| {
        for style in [Style::Shown, Style::Ghost, Style::Hidden] {
            if ui.button(style.name()).clicked() {
                for e in labels.entries.iter_mut().filter(|e| matches(&e.name)) {
                    e.style = style;
                }
            }
        }
        ui.label("(filtered)");
    });
    egui::Grid::new("labels")
        .num_columns(3)
        .striped(true)
        .show(ui, |ui| {
            for e in labels.entries.iter_mut().filter(|e| matches(&e.name)) {
                let button = egui::Button::new(e.style.name()).min_size(egui::vec2(52.0, 0.0));
                if ui
                    .add(button)
                    .on_hover_text("Click to cycle: shown, ghost, hidden")
                    .clicked()
                {
                    e.style = e.style.next();
                }
                ui.horizontal(|ui| {
                    egui::color_picker::color_edit_button_rgb(ui, &mut e.color);
                    ui.menu_button("…", |ui| {
                        ui.add(egui::Slider::new(&mut e.roughness, 0.02..=1.0).text("Roughness"));
                        ui.add(egui::Slider::new(&mut e.metalness, 0.0..=1.0).text("Metalness"));
                        ui.add(egui::Slider::new(&mut e.subsurface, 0.0..=1.0).text("Subsurface"));
                        ui.add(egui::Slider::new(&mut e.coat, 0.0..=1.0).text("Wetness"))
                            .on_hover_text("Glossy film on top (clear coat)");
                    })
                    .response
                    .on_hover_text("Material");
                });
                ui.label(&e.name).on_hover_text(format!(
                    "value {} · {} voxels · {:.1} ml",
                    e.value,
                    e.voxels,
                    e.voxels as f64 * voxel_ml
                ));
                ui.end_row();
            }
        });
}

/// The volume image, filling the space next to the panel, with camera controls.
fn viewport(ui: &mut egui::Ui, viewer: &mut Viewer, texture: egui::TextureId) {
    egui::CentralPanel::no_frame().show(ui, |ui| {
        let rect = ui.max_rect();
        let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
        let ppp = ui.ctx().pixels_per_point();
        let scale = viewer.settings.resolution_scale;
        viewer.target = (
            (rect.width() * ppp * scale).round().max(1.0) as u32,
            (rect.height() * ppp * scale).round().max(1.0) as u32,
        );

        let camera = &mut viewer.settings.camera;
        let delta = response.drag_delta() / rect.height().max(1.0);
        let shift = ui.input(|i| i.modifiers.shift);
        if response.dragged_by(egui::PointerButton::Middle)
            || (response.dragged_by(egui::PointerButton::Primary) && shift)
        {
            camera.pan(delta.x, delta.y);
        } else if response.dragged_by(egui::PointerButton::Primary) {
            camera.orbit(-delta.x * 3.0, delta.y * 3.0);
        } else if response.dragged_by(egui::PointerButton::Secondary) {
            camera.zoom((delta.y * 2.0).exp());
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                camera.zoom((-scroll / 40.0 * 0.1).exp());
            }
        }

        let ctrl = ui.input(|i| i.modifiers.command);
        if response.double_clicked() {
            viewer.settings.camera = viewer.home;
        } else if response.clicked()
            && ctrl
            && let Some(pos) = response.interact_pointer_pos()
        {
            viewer.focus_request = Some((pos - rect.min) / rect.size());
        }

        if viewer.rendered.is_some() {
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            ui.painter().image(texture, rect, uv, egui::Color32::WHITE);
        }
    });
}

/// The first point drawn along the camera ray through image position `uv`
/// (0-1, from the top left): a shown label in the isolating label modes, else
/// tissue the transfer function makes fairly opaque (and not a hidden label).
/// Clip planes are ignored.
fn pick(
    volume: &Volume,
    labels: Option<&LabelSet>,
    s: &Settings,
    uv: egui::Vec2,
    aspect: f32,
) -> Option<glam::Vec3> {
    let cam = &s.camera;
    let (right, up) = cam.basis();
    let tan = (cam.fov_y / 2.0).tan();
    let (x, y) = (uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0);
    let dir = (cam.forward() + right * (x * tan * aspect) - up * (y * tan)).normalize();
    let eye = cam.eye();
    let to_index = volume.index_to_world().inverse().as_mat4();
    let dims = volume.dims.as_ivec3();

    let mut style = [None; 256];
    if let Some(l) = labels {
        for e in &l.entries {
            style[e.value as usize] = Some(e.style);
        }
    }
    let isolate =
        labels.is_some() && matches!(s.label_mode, LabelMode::Isolate | LabelMode::Hybrid);
    let tint = labels.is_some() && s.label_mode == LabelMode::Tint;
    let tf = &TransferFunction::presets()[s.preset];
    let step = 0.5 * volume.spacing.min_element() as f32;
    let far = cam.distance + 2.0 * (volume.dims.as_vec3() * volume.spacing.as_vec3()).length();
    let mut t = 0.0;
    while t < far {
        let p = eye + dir * t;
        let i = to_index.transform_point3(p).round().as_ivec3();
        t += step;
        if i.cmplt(glam::IVec3::ZERO).any() || i.cmpge(dims).any() {
            continue;
        }
        let v = (i.x + dims.x * (i.y + dims.y * i.z)) as usize;
        let label = labels.map_or(0, |l| l.data[v]);
        let label_style = style[label as usize];
        let hit = if isolate {
            label_style == Some(Style::Shown)
        } else {
            label_style != Some(Style::Hidden)
                && !(tint && label_style == Some(Style::Ghost))
                && tf.evaluate(volume.hu[v] as f32)[3] > 0.3
        };
        if hit {
            return Some(p);
        }
    }
    None
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_none() {
            if let Err(e) = self.init(event_loop) {
                self.error = Some(e);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let Err(e) = self.handle(event_loop, event) {
            self.error = Some(e);
            event_loop.exit();
        }
    }

    fn new_events(&mut self, _event_loop: &ActiveEventLoop, cause: StartCause) {
        if matches!(cause, StartCause::ResumeTimeReached { .. }) {
            if let Some((gpu, _)) = &self.gpu {
                gpu.window.request_redraw();
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Poll a running segmentation for progress and completion.
        event_loop.set_control_flow(if self.labels.job.is_some() {
            ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(100))
        } else {
            ControlFlow::Wait
        });
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Tear down Vulkan objects while the window still exists.
        self.gpu = None;
    }
}
