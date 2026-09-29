use anyhow::{Result, bail};
use ash::vk;
use glam::{Mat3, Mat4, UVec3, UVec4, Vec3, Vec4};
use gpu_allocator::MemoryLocation;
use rayon::prelude::*;
use volio::Volume;

use crate::environment::{DIFFUSE_WIDTH, EnvironmentMap};
use crate::resources::{Buffer, Image, transition};
use crate::transfer::{LUT_SIZE, TransferFunction, hu_to_unorm16};
use crate::{Context, OrbitCamera};

pub const FRAMES_IN_FLIGHT: usize = 2;
const WORKGROUP: u32 = 8;

const DVR_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dvr.comp.spv"));
const TONEMAP_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tonemap.comp.spv"));
const NORMALS_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/normals.comp.spv"));
const SHADOW_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shadow.comp.spv"));
const AO_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ao.comp.spv"));
const MASK_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mask.comp.spv"));
const PT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pt.comp.spv"));
const DENOISE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/denoise.comp.spv"));
const MAJORANT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/majorant.comp.spv"));
const BLOOM_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bloom.comp.spv"));
const OIDN_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/oidn.comp.spv"));
/// Passes of bloom.comp, in order (1 is auto exposure).
const BLOOM_PASSES: i32 = 7;
/// Edge length of the majorant grid's bricks in voxels; must match BRICK_SIZE in common.glsl.
const BRICK_SIZE: u32 = 8;
/// À-trous levels of the path-tracing denoiser (steps 1, 2, 4, 8, 16 pixels).
const DENOISE_LEVELS: i32 = 5;

/// Shadow volume resolution relative to the density volume.
const SHADOW_DOWNSAMPLE: u32 = 2;
/// Sideways spread of diffused light per shadow-volume slice, in shadow voxels.
const SUBSURFACE_SPREAD: f32 = 0.7;
/// Ambient-occlusion volume resolution relative to the density volume.
const AO_DOWNSAMPLE: u32 = 4;

/// Must match `Uniforms` in shaders/common.glsl (std140; all members are 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    world_to_index: Mat4,
    eye: Vec4,
    forward: Vec4,
    right: Vec4,
    up: Vec4,
    dims: Vec4,
    flags: UVec4,
    labels: Vec4,
    light_dir: Vec4,
    ambient: Vec4,
    material: Vec4,
    lighting: UVec4,
    light_cells: Vec4,
    clip: UVec4,
    clip_planes: [Vec4; 3],
    clip_normals: [Vec4; 3],
    subsurface: Vec4,
    path: UVec4,
    env: Vec4,
    lens: Vec4,
    finish: Vec4,
    tf_bucket_max: [Vec4; 16],
    label_colors: [Vec4; 256],
    label_materials: [Vec4; 256],
    label_density: [Vec4; 256],
}

/// Must match `Push` in shadow.comp and ao.comp.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Push {
    target_dims: UVec4,
    sweep: [i32; 4],
    /// Mask pass: soft-label grid orientation, see [`Renderer::set_labels`].
    soft: [i32; 4],
}

/// The segmentation model's decision margins on its own grid (see seg::SoftLabels).
pub struct SoftLabelGrid<'a> {
    /// Grid size along its axes.
    pub dims: [usize; 3],
    /// Volume axis each grid axis runs along, and whether it runs reversed.
    pub src_axis: [usize; 3],
    pub flip: [bool; 3],
    /// Per node, grid axis 0 fastest: best label, runner-up label, margin × 16, unused.
    pub data: &'a [[u8; 4]],
}

/// How the label volume affects rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LabelMode {
    Off,
    /// Transfer-function opacity; visible labels tint the albedo, hidden labels are cut away.
    Tint,
    /// Only visible labels are rendered, with their own colour and opacity.
    Isolate,
    /// Only visible labels are rendered, like [`LabelMode::Isolate`], but their
    /// surfaces follow CT intensity edges and their albedo carries CT texture.
    Hybrid,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabelParams {
    pub mode: LabelMode,
    /// Tint mode: 0 = transfer-function colour, 1 = label colour.
    pub tint: f32,
    /// Isolate mode: extinction of labelled voxels, 1/mm.
    pub extinction: f32,
    /// Per label value: linear RGB albedo and visibility (alpha 0 = hidden).
    pub colors: [[f32; 4]; 256],
    /// Per label value: roughness, metalness, subsurface amount and clear coat
    /// (a wet film's strength, 0-1).
    pub materials: [[f32; 4]; 256],
    /// Per label value: drawn translucently, keeping only its silhouettes.
    pub ghost: [bool; 256],
    /// Opacity of ghosted structures where they face the viewer (0..1).
    pub ghost_opacity: f32,
    /// Tint mode: unlabelled tissue this close to a hidden or ghosted label is cut
    /// too, since segmentations end slightly inside the real boundary (mm).
    pub cut_margin_mm: f32,
    /// Blur radius of the label masks in voxels (0-3): larger gives smoother
    /// boundaries but can remove structures thinner than about twice the radius.
    pub smoothing: u32,
    /// Per label value, hybrid mode: normalised-density opacity ramp (start, end),
    /// median and interquartile range of the label's voxels. A negative IQR marks
    /// hollow (air-containing) structures, drawn by their segmentation alone.
    pub density: [[f32; 4]; 256],
}

impl Default for LabelParams {
    fn default() -> Self {
        Self {
            mode: LabelMode::Off,
            tint: 0.7,
            extinction: 0.5,
            colors: [[1.0; 4]; 256],
            materials: [[0.5, 0.0, 0.3, 0.0]; 256],
            ghost: [false; 256],
            ghost_opacity: 0.08,
            cut_margin_mm: 2.0,
            smoothing: 1,
            density: [[0.0, 1e-3, 0.5, 1.0]; 256],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToneMapper {
    /// Clamp only.
    None,
    Aces,
    Agx,
}

/// What surrounds the volume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Environment {
    /// Evenly lit dome, brighter above, over a dark grey background.
    Dome = 0,
    /// Photo studio: a soft fill opposite the key light and a rim light behind the
    /// subject in otherwise dark surroundings, over a black background.
    Studio = 1,
    /// The HDR light map set with [`Renderer::set_environment_map`] (the studio
    /// until one is set), over a black background.
    Map = 2,
}

/// Lighting, material defaults and display transform.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lighting {
    /// Key light direction relative to the camera, radians: azimuth to the right
    /// of the view axis and elevation above it.
    pub azimuth: f32,
    pub elevation: f32,
    /// Key light irradiance.
    pub intensity: f32,
    /// Uniform ambient radiance.
    pub ambient: f32,
    pub environment: Environment,
    /// Angular radius of the key light (radians); larger lights cast softer
    /// shadows. Only the path tracer uses it.
    pub light_size: f32,
    /// Turns the light map about the camera's up axis (radians). At 0 its
    /// brightest region lies on the key light's side.
    pub env_rotation: f32,
    /// Henyey-Greenstein anisotropy of volume scattering in the path tracer:
    /// 0 isotropic, towards 1 forward scattering as in real tissue.
    pub anisotropy: f32,
    /// How much redder light travels further than blue through tissue that
    /// scatters below its surface, in the path tracer (0 = grey, 1 = like skin).
    pub chromatic: f32,
    pub shadows: bool,
    pub ambient_occlusion: bool,
    /// Exposure in stops (on top of auto exposure, if on).
    pub exposure: f32,
    /// Meter the drawn part of the image (not the background) and expose for
    /// it, adapting over frames like a camera.
    pub auto_exposure: bool,
    /// Share of light spread into a glow around bright areas (lens bloom).
    pub bloom: f32,
    /// Darkening towards the corners of the frame (0-1).
    pub vignette: f32,
    /// Film grain amplitude in display values (0-0.1).
    pub grain: f32,
    pub tone_mapper: ToneMapper,
    /// Material of tissue that is not covered by a label.
    pub roughness: f32,
    pub metalness: f32,
    /// Share of diffuse light scattered below the surface of unlabelled tissue.
    pub subsurface: f32,
    /// How far light diffuses through tissue before being absorbed, roughly the
    /// red mean free path at the default label opacity (mm).
    pub scattering_distance: f32,
    /// Scales every material's subsurface amount (0 disables subsurface scattering).
    pub subsurface_strength: f32,
}

impl Default for Lighting {
    fn default() -> Self {
        Self {
            azimuth: -60f32.to_radians(),
            elevation: 40f32.to_radians(),
            intensity: 3.0,
            ambient: 0.3,
            environment: Environment::Studio,
            light_size: 12f32.to_radians(),
            env_rotation: 0.0,
            anisotropy: 0.6,
            chromatic: 1.0,
            shadows: true,
            ambient_occlusion: true,
            exposure: 0.0,
            auto_exposure: true,
            bloom: 0.04,
            vignette: 0.25,
            grain: 0.012,
            tone_mapper: ToneMapper::Aces,
            roughness: 0.5,
            metalness: 0.0,
            subsurface: 0.3,
            scattering_distance: 10.0,
            subsurface_strength: 1.0,
        }
    }
}

impl Lighting {
    /// World-space unit vector towards the key light for `camera`.
    pub fn light_direction(&self, camera: &OrbitCamera) -> Vec3 {
        let (right, up) = camera.basis();
        let (se, ce) = self.elevation.sin_cos();
        let (sa, ca) = self.azimuth.sin_cos();
        (-camera.forward() * ce * ca + right * ce * sa + up * se).normalize()
    }
}

/// Per-frame render settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameParams {
    pub camera: OrbitCamera,
    /// Ray-marching step in mm.
    pub step_mm: f32,
    /// Encode the output as sRGB in the shader (for UNORM targets).
    pub srgb_encode: bool,
    pub labels: LabelParams,
    pub lighting: Lighting,
    /// Integrate the transfer function over each ray segment instead of point-sampling it.
    pub preintegrate: bool,
    /// Skip bricks with nothing to draw (and, when path tracing, use their
    /// extinction bounds for delta tracking).
    pub skip_empty_space: bool,
    pub clipping: Clipping,
    /// Path trace progressively instead of ray marching in real time.
    pub path_tracing: Option<PathTracing>,
    /// Filter the density with a cubic B-spline where surfaces are found and
    /// shaded, instead of trilinearly: smooth surfaces and normals from coarse
    /// or anisotropic voxels (no slice terraces), at more texture fetches.
    pub smooth_sampling: bool,
}

/// Settings of the progressive path tracer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathTracing {
    /// Scattering events per path before it is cut off.
    pub bounces: u32,
    /// Paths per pixel added each frame.
    pub samples_per_frame: u32,
    /// How the accumulated image is filtered for display; changing it does not
    /// restart accumulation.
    pub denoiser: Denoiser,
    /// Lens radius (mm) for depth of field; 0 is a pinhole (all sharp).
    pub aperture: f32,
    /// Patient-space point in focus; `None` focuses at the orbit target.
    pub focus: Option<Vec3>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Denoiser {
    Off,
    /// Edge-avoiding à-trous filter on the GPU (denoise.comp).
    Atrous,
    /// Intel Open Image Denoise (see [`Renderer::denoise_oidn`]).
    Oidn,
}

impl Default for PathTracing {
    fn default() -> Self {
        Self {
            bounces: 4,
            samples_per_frame: 1,
            denoiser: if cfg!(feature = "oidn") {
                Denoiser::Oidn
            } else {
                Denoiser::Atrous
            },
            aperture: 0.0,
            focus: None,
        }
    }
}

/// The anatomical planes, as normals in patient (LPS) space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnatomicalPlane {
    /// Normal left-right (x).
    Sagittal,
    /// Normal anterior-posterior (y).
    Coronal,
    /// Normal inferior-superior (z); also called transverse.
    Axial,
}

impl AnatomicalPlane {
    pub const ALL: [AnatomicalPlane; 3] = [Self::Sagittal, Self::Coronal, Self::Axial];

    pub fn axis(self) -> usize {
        self as usize
    }
}

/// A cut along an anatomical plane: everything on one side is removed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipPlane {
    pub enabled: bool,
    pub plane: AnatomicalPlane,
    /// Remove the side towards +LPS (left, posterior, superior) rather than
    /// towards -LPS (right, anterior, inferior).
    pub cut_positive: bool,
    /// Position along the plane's axis, patient coordinates in mm.
    pub position: f32,
}

/// Anatomical cutaways.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clipping {
    /// One per anatomical plane, in [`AnatomicalPlane::ALL`] order.
    pub planes: [ClipPlane; 3],
    /// Cut only where every enabled plane cuts (corner cutaways) instead of
    /// wherever any plane does (halves and slabs).
    pub all: bool,
}

impl Default for Clipping {
    fn default() -> Self {
        Self {
            planes: AnatomicalPlane::ALL.map(|plane| ClipPlane {
                enabled: false,
                plane,
                cut_positive: false,
                position: 0.0,
            }),
            all: false,
        }
    }
}

/// The clip planes as uploaded, for noticing when lighting must be recomputed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ClipKey([u32; 14]);

/// Everything that decides which voxels are opaque, i.e. when the shadow and
/// ambient-occlusion volumes must be recomputed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OpacityKey {
    /// Bumped when the transfer function or label volume changes.
    generation: u64,
    mode: LabelMode,
    extinction_bits: u32,
    visible: [u64; 4],
    ghost: [u64; 4],
    ghost_opacity_bits: u32,
    /// Cut margin in voxels.
    margin: i32,
    smoothing: u32,
}

/// Renders a CT volume into an RGBA8 image that can be displayed or read back.
pub struct Renderer {
    density: Image,
    tf: Image,
    /// Largest transfer-function extinction in each 1/64 of the density range.
    tf_bucket_max: [Vec4; 16],
    /// Extinction bound per brick (`R32F`).
    majorant: Image,
    majorant_pipeline: vk::Pipeline,
    bloom_pipeline: vk::Pipeline,
    /// Bloom at quarter resolution (A, blur partner B) and eighth resolution
    /// (C, partner D), `RGBA16F`; see bloom.comp.
    bloom_images: Vec<Image>,
    /// Auto exposure, adapted over frames (`R32F`, 1 texel; 0 = not yet metered).
    exposure: Image,
    majorant_key: Option<OpacityKey>,
    /// Cumulative transfer-function integrals for pre-integration (`RGBA32F`).
    preint: Image,
    /// `R8_UINT` label per voxel; a 1-voxel placeholder until [`Renderer::set_labels`].
    labels: Image,
    /// Soft labels (`RGBA8_UINT`); a 1-node placeholder without them.
    soft: Image,
    /// Soft-label grid axes (xyz), flip bits and 8 = present (w), for the mask pass.
    soft_push: [i32; 4],
    has_labels: bool,
    /// Light map radiance (`RGBA16F`), importance-sampling table (`RG32F`) and
    /// diffuse convolution (`RGBA16F`); 1-texel placeholders without a map.
    env_radiance: Image,
    env_table: Image,
    env_diffuse: Image,
    /// Horizontal image coordinate of the light map's brightest region, if set.
    env_peak_u: Option<f32>,
    /// World normal and surfaceness per voxel, computed once.
    normals: Image,
    /// Transmittance towards the key light.
    shadow: Image,
    /// Ambient visibility.
    ao: Image,
    /// Smoothed masks of solid (r) and ghosted (g) voxels in label modes.
    mask: Image,
    /// Ping-pong partner of `mask` for its separable blur.
    mask_b: Image,
    sampler: vk::Sampler,
    nearest_sampler: vk::Sampler,
    /// Linear filtering with an opaque white border: full transmittance outside the volume.
    border_sampler: vk::Sampler,
    output: Option<Image>,
    /// sRGB view of `output`: sampling it decodes the shader's sRGB-encoded values.
    output_srgb_view: vk::ImageView,
    /// Linear HDR radiance and transmittance, input of the tone-mapping pass.
    hdr: Option<Image>,
    /// Path tracing: running sum of radiance (rgb) and background (a), `RGBA32F`.
    /// Path tracing, `RGBA32F`, in binding order 17-21: accumulated radiance and
    /// background; accumulated first-hit albedo and luminance²; accumulated
    /// first-hit normal and depth; two denoiser ping-pong images.
    pt_images: Vec<Image>,
    /// OIDN, created on first use (an error is kept so it is not retried).
    #[cfg(feature = "oidn")]
    oidn: Option<std::result::Result<crate::oidn::Oidn, String>>,
    /// OIDN host buffers: read-back of accumulation, albedo and normal images,
    /// then the upload of the denoised HDR image. Allocated on first use.
    oidn_buffers: Vec<Buffer>,
    /// OIDN on the GPU without host copies: colour, albedo, normal and result
    /// as Vulkan memory shared with OIDN (see oidn.comp). Created on first use.
    #[cfg(feature = "oidn")]
    oidn_shared: Vec<SharedBuffer>,
    /// Sharing failed or is unsupported: use the host copies.
    oidn_shared_failed: bool,
    oidn_pipeline: vk::Pipeline,

    /// Uniform slot of the last recorded frame (for passes run outside `record`).
    last_slot: usize,
    /// Path tracing: paths per pixel in `accum`, and the settings they were traced with.
    accumulated: u32,
    accumulated_params: Option<FrameParams>,
    uniforms: Vec<Buffer>,
    descriptor_pool: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
    sets: Vec<vk::DescriptorSet>,
    pipeline_layout: vk::PipelineLayout,
    dvr: vk::Pipeline,
    tonemap: vk::Pipeline,
    shadow_pipeline: vk::Pipeline,
    ao_pipeline: vk::Pipeline,
    mask_pipeline: vk::Pipeline,
    pt_pipeline: vk::Pipeline,
    denoise_pipeline: vk::Pipeline,
    generation: u64,
    mask_key: Option<OpacityKey>,
    shadow_key: Option<(OpacityKey, ClipKey, [u32; 3], u64)>,
    ao_key: Option<(OpacityKey, ClipKey)>,
    world_to_index: Mat4,
    index_to_world: Mat4,
    dims: Vec3,
    /// Axis-aligned patient-space bounds of the volume (mm).
    pub bounds: (Vec3, Vec3),
    /// Finest voxel spacing, mm.
    pub min_spacing: f32,
    max_spacing: f32,
    pub center: Vec3,
    pub diagonal: f32,
}

impl Renderer {
    pub fn new(ctx: &Context, volume: &Volume, tf: &TransferFunction) -> Result<Self> {
        let limits = unsafe {
            ctx.instance
                .get_physical_device_properties(ctx.physical_device)
        }
        .limits;
        if volume.dims.max_element() > limits.max_image_dimension3_d {
            bail!(
                "volume {:?} exceeds the GPU's 3D texture limit of {}",
                volume.dims,
                limits.max_image_dimension3_d
            );
        }

        let density_data: Vec<u16> = volume.hu.par_iter().map(|&hu| hu_to_unorm16(hu)).collect();
        let extent = |d: UVec3| vk::Extent3D {
            width: d.x,
            height: d.y,
            depth: d.z,
        };
        let density = Image::with_data(
            ctx,
            vk::ImageType::TYPE_3D,
            vk::Format::R16_UNORM,
            extent(volume.dims),
            bytemuck::cast_slice(&density_data),
            "density",
        )?;
        let tf_bucket_max = bucket_max(tf);
        let (tf, preint) = upload_tf(ctx, tf)?;
        let labels = upload_labels(ctx, &[0], UVec3::ONE)?;
        let soft = upload_soft(ctx, &[[0; 4]], UVec3::ONE)?;
        let [env_radiance, env_table, env_diffuse] = upload_environment(ctx, None)?.0;
        let derived = |format, dims: UVec3, name| {
            Image::new(
                ctx,
                vk::ImageType::TYPE_3D,
                format,
                extent(dims),
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                name,
            )
        };
        let normals = derived(vk::Format::R8G8B8A8_SNORM, volume.dims, "normals")?;
        // rgb: diffused light for subsurface scattering, a: direct transmittance.
        let shadow = derived(
            vk::Format::R16G16B16A16_SFLOAT,
            downsampled(volume.dims, SHADOW_DOWNSAMPLE),
            "shadow",
        )?;
        let ao = derived(
            vk::Format::R16_SFLOAT,
            downsampled(volume.dims, AO_DOWNSAMPLE),
            "ambient occlusion",
        )?;
        let mask = derived(vk::Format::R16G16_UNORM, volume.dims, "label masks")?;
        let majorant = derived(
            vk::Format::R32_SFLOAT,
            downsampled(volume.dims, BRICK_SIZE),
            "extinction bounds",
        )?;
        let mask_b = derived(vk::Format::R16G16_UNORM, volume.dims, "label masks (blur)")?;

        let sampler_info = |filter, address| {
            vk::SamplerCreateInfo::default()
                .mag_filter(filter)
                .min_filter(filter)
                .address_mode_u(address)
                .address_mode_v(address)
                .address_mode_w(address)
                .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
        };
        let (sampler, nearest_sampler, border_sampler) = unsafe {
            (
                ctx.device.create_sampler(
                    &sampler_info(vk::Filter::LINEAR, vk::SamplerAddressMode::CLAMP_TO_EDGE),
                    None,
                )?,
                ctx.device.create_sampler(
                    &sampler_info(vk::Filter::NEAREST, vk::SamplerAddressMode::CLAMP_TO_EDGE),
                    None,
                )?,
                ctx.device.create_sampler(
                    &sampler_info(vk::Filter::LINEAR, vk::SamplerAddressMode::CLAMP_TO_BORDER),
                    None,
                )?,
            )
        };

        let uniforms = (0..FRAMES_IN_FLIGHT)
            .map(|_| {
                Buffer::new(
                    ctx,
                    size_of::<Uniforms>() as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    MemoryLocation::CpuToGpu,
                    "uniforms",
                )
            })
            .collect::<Result<Vec<_>>>()?;

        // One layout for all passes; see shaders/common.glsl.
        use vk::DescriptorType as T;
        let binding_types = [
            T::UNIFORM_BUFFER,         // 0 uniforms
            T::COMBINED_IMAGE_SAMPLER, // 1 density
            T::COMBINED_IMAGE_SAMPLER, // 2 transfer function
            T::STORAGE_IMAGE,          // 3 display output
            T::COMBINED_IMAGE_SAMPLER, // 4 labels
            T::COMBINED_IMAGE_SAMPLER, // 5 normals
            T::COMBINED_IMAGE_SAMPLER, // 6 shadow
            T::COMBINED_IMAGE_SAMPLER, // 7 ambient occlusion
            T::STORAGE_IMAGE,          // 8 HDR radiance
            T::STORAGE_IMAGE,          // 9 normals (write)
            T::STORAGE_IMAGE,          // 10 shadow (write)
            T::STORAGE_IMAGE,          // 11 ambient occlusion (write)
            T::COMBINED_IMAGE_SAMPLER, // 12 pre-integration table
            T::COMBINED_IMAGE_SAMPLER, // 13 label mask
            T::STORAGE_IMAGE,          // 14 label mask (write)
            T::STORAGE_IMAGE,          // 15 label mask blur partner (write)
            T::COMBINED_IMAGE_SAMPLER, // 16 label mask blur partner
            T::STORAGE_IMAGE,          // 17 path-tracing accumulation
            T::STORAGE_IMAGE,          // 18 path-tracing guide: albedo
            T::STORAGE_IMAGE,          // 19 path-tracing guide: normal and depth
            T::STORAGE_IMAGE,          // 20 denoiser ping
            T::STORAGE_IMAGE,          // 21 denoiser pong
            T::COMBINED_IMAGE_SAMPLER, // 22 extinction bounds
            T::STORAGE_IMAGE,          // 23 extinction bounds (write)
            T::COMBINED_IMAGE_SAMPLER, // 24 soft labels
            T::COMBINED_IMAGE_SAMPLER, // 25 light map
            T::COMBINED_IMAGE_SAMPLER, // 26 light map sampling table
            T::COMBINED_IMAGE_SAMPLER, // 27 light map, diffuse
            T::STORAGE_IMAGE,          // 28 bloom A (quarter resolution)
            T::STORAGE_IMAGE,          // 29 bloom B (A's blur partner)
            T::STORAGE_IMAGE,          // 30 bloom C (eighth resolution)
            T::STORAGE_IMAGE,          // 31 bloom D (C's blur partner)
            T::COMBINED_IMAGE_SAMPLER, // 32 bloom A
            T::COMBINED_IMAGE_SAMPLER, // 33 bloom C
            T::STORAGE_IMAGE,          // 34 auto exposure
            T::STORAGE_BUFFER,         // 35 OIDN colour
            T::STORAGE_BUFFER,         // 36 OIDN albedo
            T::STORAGE_BUFFER,         // 37 OIDN normal
            T::STORAGE_BUFFER,         // 38 OIDN result
        ];
        let bindings: Vec<_> = binding_types
            .iter()
            .enumerate()
            .map(|(i, &ty)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as u32)
                    .descriptor_type(ty)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_layout = unsafe {
            ctx.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?
        };
        let n = FRAMES_IN_FLIGHT as u32;
        let count = |ty| binding_types.iter().filter(|&&t| t == ty).count() as u32 * n;
        let pool_sizes = [
            T::UNIFORM_BUFFER,
            T::COMBINED_IMAGE_SAMPLER,
            T::STORAGE_IMAGE,
            T::STORAGE_BUFFER,
        ]
        .map(|ty| vk::DescriptorPoolSize {
            ty,
            descriptor_count: count(ty),
        });
        let descriptor_pool = unsafe {
            ctx.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(n)
                    .pool_sizes(&pool_sizes),
                None,
            )?
        };
        let layouts = vec![set_layout; FRAMES_IN_FLIGHT];
        let sets = unsafe {
            ctx.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts),
            )?
        };

        let set_layouts = [set_layout];
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(size_of::<Push>() as u32)];
        let pipeline_layout = unsafe {
            ctx.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push_ranges),
                None,
            )?
        };
        let pipeline = |spirv| create_compute_pipeline(ctx, pipeline_layout, spirv);
        let (
            dvr,
            tonemap,
            normals_pipeline,
            shadow_pipeline,
            ao_pipeline,
            mask_pipeline,
            pt_pipeline,
            denoise_pipeline,
            majorant_pipeline,
            bloom_pipeline,
            oidn_pipeline,
        ) = (
            pipeline(DVR_SPV)?,
            pipeline(TONEMAP_SPV)?,
            pipeline(NORMALS_SPV)?,
            pipeline(SHADOW_SPV)?,
            pipeline(AO_SPV)?,
            pipeline(MASK_SPV)?,
            pipeline(PT_SPV)?,
            pipeline(DENOISE_SPV)?,
            pipeline(MAJORANT_SPV)?,
            pipeline(BLOOM_SPV)?,
            pipeline(OIDN_SPV)?,
        );
        let exposure = Image::new(
            ctx,
            vk::ImageType::TYPE_2D,
            vk::Format::R32_SFLOAT,
            vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
            "exposure",
        )?;
        ctx.one_shot(|cmd| unsafe {
            transition(
                &ctx.device,
                cmd,
                exposure.handle,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );
            ctx.device.cmd_clear_color_image(
                cmd,
                exposure.handle,
                vk::ImageLayout::GENERAL,
                &vk::ClearColorValue::default(),
                &[crate::resources::COLOR_RANGE],
            );
        })?;

        let world_to_index = volume.world_to_index().as_mat4();
        let index_to_world = volume.index_to_world();
        let center = index_to_world
            .transform_point3((volume.dims.as_dvec3() - 1.0) * 0.5)
            .as_vec3();
        let corners = (0..8).map(|i| {
            let hi = volume.dims.as_dvec3() - 0.5;
            let c = glam::DVec3::new(
                if i & 1 == 0 { -0.5 } else { hi.x },
                if i & 2 == 0 { -0.5 } else { hi.y },
                if i & 4 == 0 { -0.5 } else { hi.z },
            );
            index_to_world.transform_point3(c).as_vec3()
        });
        let bounds = corners.fold((Vec3::MAX, Vec3::MIN), |(lo, hi), c| (lo.min(c), hi.max(c)));

        let mut renderer = Renderer {
            density,
            tf,
            tf_bucket_max,
            majorant,
            majorant_pipeline,
            bloom_pipeline,
            bloom_images: Vec::new(),
            exposure,
            majorant_key: None,
            preint,
            labels,
            soft,
            soft_push: [0; 4],
            env_radiance,
            env_table,
            env_diffuse,
            env_peak_u: None,
            has_labels: false,
            normals,
            shadow,
            ao,
            mask,
            mask_b,
            sampler,
            nearest_sampler,
            border_sampler,
            output: None,
            output_srgb_view: vk::ImageView::null(),
            hdr: None,
            pt_images: Vec::new(),
            #[cfg(feature = "oidn")]
            oidn: None,
            oidn_buffers: Vec::new(),
            #[cfg(feature = "oidn")]
            oidn_shared: Vec::new(),
            oidn_shared_failed: false,
            oidn_pipeline,
            last_slot: 0,
            accumulated: 0,
            accumulated_params: None,
            uniforms,
            descriptor_pool,
            set_layout,
            sets,
            pipeline_layout,
            dvr,
            tonemap,
            shadow_pipeline,
            ao_pipeline,
            mask_pipeline,
            pt_pipeline,
            denoise_pipeline,
            generation: 0,
            mask_key: None,
            shadow_key: None,
            ao_key: None,
            world_to_index,
            index_to_world: index_to_world.as_mat4(),
            dims: volume.dims.as_vec3(),
            bounds,
            min_spacing: volume.spacing.min_element() as f32,
            max_spacing: volume.spacing.max_element() as f32,
            center,
            diagonal: volume.extent_mm().length() as f32,
        };
        renderer.write_descriptors(ctx);
        renderer.compute_normals(ctx, normals_pipeline)?;
        unsafe { ctx.device.destroy_pipeline(normals_pipeline, None) };
        Ok(renderer)
    }

    /// Precomputes the normal volume and moves the derived volumes to `GENERAL`,
    /// where they stay (they are both written and sampled).
    fn compute_normals(&mut self, ctx: &Context, pipeline: vk::Pipeline) -> Result<()> {
        let t = std::time::Instant::now();
        self.write_uniforms(0, &Uniforms::zeroed_with(self.world_to_index, self.dims));
        let dims = self.dims.as_uvec3();
        ctx.one_shot(|cmd| unsafe {
            for image in [
                &self.normals,
                &self.shadow,
                &self.ao,
                &self.mask,
                &self.mask_b,
                &self.majorant,
            ] {
                transition(
                    &ctx.device,
                    cmd,
                    image.handle,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                );
            }
            self.bind(ctx, cmd, pipeline, 0);
            ctx.device.cmd_dispatch(
                cmd,
                dims.x.div_ceil(4),
                dims.y.div_ceil(4),
                dims.z.div_ceil(4),
            );
            compute_barrier(&ctx.device, cmd);
        })?;
        tracing::info!(elapsed = ?t.elapsed(), "computed normals");
        Ok(())
    }

    pub fn default_camera(&self) -> OrbitCamera {
        OrbitCamera::framing(self.center, self.diagonal)
    }

    pub fn output(&self) -> Option<&Image> {
        self.output.as_ref()
    }

    /// View of the output that samples as linear colour (for UI compositing).
    /// Paths per pixel accumulated so far when path tracing.
    /// Whether OIDN is (or, not yet tried, may be) usable.
    pub fn oidn_available(&self) -> bool {
        #[cfg(feature = "oidn")]
        return !matches!(self.oidn, Some(Err(_)));
        #[cfg(not(feature = "oidn"))]
        false
    }

    /// Denoises the path tracer's accumulated image with OIDN and tone-maps it into
    /// the output, leaving that in `final_layout`. Runs outside `record`, after
    /// the frame that accumulated the samples; waits for the GPU.
    pub fn denoise_oidn(&mut self, ctx: &Context, final_layout: vk::ImageLayout) -> Result<()> {
        #[cfg(not(feature = "oidn"))]
        {
            let _ = (ctx, final_layout);
            bail!("built without OIDN support");
        }
        #[cfg(feature = "oidn")]
        {
            let Some(output) = self.output.as_ref() else {
                return Ok(());
            };
            if self.accumulated == 0 || self.pt_images.len() < 3 {
                return Ok(());
            }
            let opened = self.oidn.get_or_insert_with(|| {
                crate::oidn::Oidn::new(ctx.device_uuid).map_err(|e| format!("{e:#}"))
            });
            if let Err(e) = opened {
                bail!("{e}");
            }
            let (w, h) = (output.extent.width, output.extent.height);
            let (output_handle, extent) = (output.handle, output.extent);
            let pixels = (w * h) as usize;
            let started = std::time::Instant::now();
            // VOLVIEW_OIDN_HOST=1 forces the copies through host memory (for comparison).
            if !self.oidn_shared_failed && std::env::var_os("VOLVIEW_OIDN_HOST").is_none() {
                match self.denoise_oidn_shared(ctx, final_layout) {
                    Ok(()) => {
                        tracing::debug!(elapsed = ?started.elapsed(), "denoise total (shared)");
                        return Ok(());
                    }
                    Err(e) => {
                        tracing::warn!("OIDN without host copies unavailable, copying: {e:#}");
                        self.oidn_shared_failed = true;
                        self.oidn_shared.drain(..).for_each(|b| b.destroy(ctx));
                        self.write_descriptors(ctx);
                    }
                }
            }
            let oidn = match self.oidn.as_mut() {
                Some(Ok(oidn)) => oidn,
                _ => unreachable!("OIDN was opened above"),
            };
            if self.oidn_buffers.is_empty() {
                for (size, usage, location, name) in [
                    (
                        pixels * 16,
                        vk::BufferUsageFlags::TRANSFER_DST,
                        MemoryLocation::GpuToCpu,
                        "oidn colour",
                    ),
                    (
                        pixels * 16,
                        vk::BufferUsageFlags::TRANSFER_DST,
                        MemoryLocation::GpuToCpu,
                        "oidn albedo",
                    ),
                    (
                        pixels * 16,
                        vk::BufferUsageFlags::TRANSFER_DST,
                        MemoryLocation::GpuToCpu,
                        "oidn normal",
                    ),
                    (
                        pixels * 8,
                        vk::BufferUsageFlags::TRANSFER_SRC,
                        MemoryLocation::CpuToGpu,
                        "oidn result",
                    ),
                ] {
                    self.oidn_buffers
                        .push(Buffer::new(ctx, size as u64, usage, location, name)?);
                }
            }

            unsafe { ctx.device.device_wait_idle()? };
            let region = [vk::BufferImageCopy::default()
                .image_subresource(crate::resources::COLOR_LAYERS)
                .image_extent(extent)];
            ctx.one_shot(|cmd| unsafe {
                for (image, buffer) in self.pt_images[..3].iter().zip(&self.oidn_buffers) {
                    ctx.device.cmd_copy_image_to_buffer(
                        cmd,
                        image.handle,
                        vk::ImageLayout::GENERAL,
                        buffer.handle,
                        &region,
                    );
                }

            })?;

            // Means of the accumulated sums, as tightly packed RGB for OIDN.
            let n = self.accumulated as f32;
            let read = |i: usize| -> Vec<f32> {
                bytemuck::cast_slice::<u8, f32>(self.oidn_buffers[i].mapped_ref().unwrap())
                    [..pixels * 4]
                    .to_vec()
            };
            let (accum, albedo_sum, normal_sum) = (read(0), read(1), read(2));
            let rgb = |v: &[f32], f: &(dyn Fn([f32; 3]) -> [f32; 3] + Sync)| -> Vec<f32> {
                v.par_chunks(4)
                    .flat_map_iter(|p| f([p[0], p[1], p[2]]))
                    .collect()
            };
            let color = rgb(&accum, &|c| c.map(|x| x / n));
            let albedo = rgb(&albedo_sum, &|c| c.map(|x| (x / n).clamp(0.0, 1.0)));
            let normal = rgb(&normal_sum, &|c| {
                let len = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                if len > 1e-6 {
                    c.map(|x| x / len)
                } else {
                    [0.0; 3]
                }
            });
            let mut denoised = vec![0f32; pixels * 3];
            let t = std::time::Instant::now();
            oidn.denoise(
                w as usize,
                h as usize,
                &color,
                &albedo,
                &normal,
                &mut denoised,
            )?;
            tracing::debug!(elapsed = ?t.elapsed(), samples = self.accumulated, "OIDN");

            // Denoised radiance plus the background fraction, as the HDR image.
            let upload: Vec<u16> = denoised
                .par_chunks(3)
                .zip(accum.par_chunks(4))
                .flat_map_iter(|(c, a)| {
                    [c[0], c[1], c[2], a[3] / n].map(|x| half::f16::from_f32(x).to_bits())
                })
                .collect();
            self.oidn_buffers[3].write(&upload);

            let hdr = self.hdr.as_ref().unwrap().handle;
            let slot = self.last_slot;
            ctx.one_shot(|cmd| unsafe {
                let device = &ctx.device;
                transition(
                    device,
                    cmd,
                    hdr,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
                device.cmd_copy_buffer_to_image(
                    cmd,
                    self.oidn_buffers[3].handle,
                    hdr,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &region,
                );
                transition(
                    device,
                    cmd,
                    hdr,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::GENERAL,
                );
                transition(
                    device,
                    cmd,
                    output_handle,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                );
                self.record_post(ctx, cmd, slot, w, h);
                transition(
                    device,
                    cmd,
                    output_handle,
                    vk::ImageLayout::GENERAL,
                    final_layout,
                );
            })?;
            tracing::debug!(elapsed = ?started.elapsed(), "denoise total (host copies)");
            Ok(())
        }
    }

    /// [`Renderer::denoise_oidn`] on the GPU: packs the accumulation into
    /// buffers shared with OIDN (oidn.comp), denoises them there and unpacks
    /// the result, without copies through host memory.
    #[cfg(feature = "oidn")]
    fn denoise_oidn_shared(&mut self, ctx: &Context, final_layout: vk::ImageLayout) -> Result<()> {
        let output = self.output.as_ref().unwrap();
        let (w, h) = (output.extent.width, output.extent.height);
        let output_handle = output.handle;
        let Some(Ok(oidn)) = self.oidn.as_mut() else {
            bail!("OIDN not open");
        };
        if self.oidn_shared.is_empty() {
            if !oidn.imports_fd() {
                bail!("the OIDN device cannot import Vulkan memory");
            }
            let bytes = w as u64 * h as u64 * 12;
            let mut shared = Vec::new();
            for _ in 0..4 {
                shared.push(SharedBuffer::new(ctx, oidn, bytes)?);
            }
            self.oidn_shared = shared;
            unsafe { ctx.device.device_wait_idle()? };
            self.write_descriptors(ctx);
        }
        let slot = self.last_slot;
        let n = self.accumulated as i32;
        let pass = |this: &Self, cmd, pass: i32| unsafe {
            this.bind(ctx, cmd, this.oidn_pipeline, slot);
            this.push(
                ctx,
                cmd,
                Push {
                    target_dims: UVec4::new(w, h, 1, 0),
                    sweep: [pass, n, 0, 0],
                    soft: [0; 4],
                },
            );
            ctx.device
                .cmd_dispatch(cmd, w.div_ceil(WORKGROUP), h.div_ceil(WORKGROUP), 1);
        };
        // Vulkan work finishes before OIDN reads (one_shot waits), and OIDN
        // before Vulkan reads its result (it syncs its device).
        ctx.one_shot(|cmd| pass(self, cmd, 0))?;
        let t = std::time::Instant::now();
        let [color, albedo, normal, result] = [0, 1, 2, 3].map(|i| &self.oidn_shared[i].oidn);
        let Some(Ok(oidn)) = self.oidn.as_mut() else {
            unreachable!()
        };
        oidn.denoise_buffers(w as usize, h as usize, color, albedo, normal, result)?;
        tracing::debug!(elapsed = ?t.elapsed(), samples = self.accumulated, "OIDN (shared memory)");
        let hdr = self.hdr.as_ref().unwrap().handle;
        ctx.one_shot(|cmd| unsafe {
            let device = &ctx.device;
            transition(device, cmd, hdr, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);
            pass(self, cmd, 1);
            compute_barrier(device, cmd);
            transition(
                device,
                cmd,
                output_handle,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );
            self.record_post(ctx, cmd, slot, w, h);
            transition(device, cmd, output_handle, vk::ImageLayout::GENERAL, final_layout);
        })?;
        Ok(())
    }

    pub fn accumulated_samples(&self) -> u32 {
        self.accumulated
    }

    pub fn output_srgb_view(&self) -> vk::ImageView {
        self.output_srgb_view
    }

    /// Replaces the transfer function. Waits for the GPU to go idle.
    pub fn set_transfer_function(&mut self, ctx: &Context, tf: &TransferFunction) -> Result<()> {
        unsafe { ctx.device.device_wait_idle()? };
        self.tf.destroy(ctx);
        self.preint.destroy(ctx);
        (self.tf, self.preint) = upload_tf(ctx, tf)?;
        self.tf_bucket_max = bucket_max(tf);
        self.generation += 1;
        self.write_descriptors(ctx);
        Ok(())
    }

    /// Uploads one label per voxel (same layout as the volume's HU values).
    /// Waits for the GPU to go idle.
    /// With `soft`, label boundaries are placed between CT voxels using the model's
    /// margins, guided by the CT itself (see mask.comp); without, from `labels` alone.
    pub fn set_labels(
        &mut self,
        ctx: &Context,
        labels: &[u8],
        soft: Option<SoftLabelGrid<'_>>,
    ) -> Result<()> {
        let dims = self.dims.as_uvec3();
        if labels.len() != (dims.x * dims.y * dims.z) as usize {
            bail!(
                "label volume has {} voxels, expected {:?}",
                labels.len(),
                dims
            );
        }
        unsafe { ctx.device.device_wait_idle()? };
        self.labels.destroy(ctx);
        self.labels = upload_labels(ctx, labels, dims)?;
        self.soft.destroy(ctx);
        (self.soft, self.soft_push) = match soft {
            Some(g) => {
                let [x, y, z] = g.dims.map(|d| d as u32);
                anyhow::ensure!(
                    g.data.len() == (x * y * z) as usize,
                    "soft label grid size mismatch"
                );
                let flips = (0..3).map(|a| (g.flip[a] as i32) << a).sum::<i32>();
                let [a0, a1, a2] = g.src_axis.map(|a| a as i32);
                (
                    upload_soft(ctx, g.data, UVec3::new(x, y, z))?,
                    [a0, a1, a2, flips | 8],
                )
            }
            None => (upload_soft(ctx, &[[0; 4]], UVec3::ONE)?, [0; 4]),
        };
        self.has_labels = true;
        self.generation += 1;
        self.write_descriptors(ctx);
        Ok(())
    }

    /// Sets (or with `None` removes) the HDR light map of [`Environment::Map`].
    /// Waits for the GPU to go idle.
    pub fn set_environment_map(
        &mut self,
        ctx: &Context,
        map: Option<&EnvironmentMap>,
    ) -> Result<()> {
        unsafe { ctx.device.device_wait_idle()? };
        let (images, peak_u) = upload_environment(ctx, map)?;
        for image in [
            &mut self.env_radiance,
            &mut self.env_table,
            &mut self.env_diffuse,
        ] {
            image.destroy(ctx);
        }
        [self.env_radiance, self.env_table, self.env_diffuse] = images;
        self.env_peak_u = peak_u;
        self.accumulated_params = None;
        self.write_descriptors(ctx);
        Ok(())
    }

    pub fn has_environment_map(&self) -> bool {
        self.env_peak_u.is_some()
    }

    pub fn has_labels(&self) -> bool {
        self.has_labels
    }

    /// (Re)creates the output images. Waits for the GPU to go idle.
    pub fn resize(&mut self, ctx: &Context, width: u32, height: u32) -> Result<()> {
        unsafe { ctx.device.device_wait_idle()? };
        self.destroy_output(ctx);
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        let output = Image::with_flags(
            ctx,
            vk::ImageType::TYPE_2D,
            vk::Format::R8G8B8A8_UNORM,
            extent,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::SAMPLED,
            vk::ImageCreateFlags::MUTABLE_FORMAT,
            "output",
        )?;
        self.output_srgb_view = unsafe {
            ctx.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(output.handle)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_SRGB)
                    .subresource_range(crate::resources::COLOR_RANGE)
                    .push_next(
                        &mut vk::ImageViewUsageCreateInfo::default()
                            .usage(vk::ImageUsageFlags::SAMPLED),
                    ),
                None,
            )?
        };
        let hdr = Image::new(
            ctx,
            vk::ImageType::TYPE_2D,
            vk::Format::R16G16B16A16_SFLOAT,
            extent,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
            "hdr",
        )?;
        let pt_images = [
            "accumulation",
            "albedo guide",
            "normal guide",
            "denoise ping",
            "denoise pong",
        ]
        .into_iter()
        .map(|name| {
            Image::new(
                ctx,
                vk::ImageType::TYPE_2D,
                vk::Format::R32G32B32A32_SFLOAT,
                extent,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                name,
            )
        })
        .collect::<Result<Vec<_>>>()?;
        let quarter = (width.div_ceil(4), height.div_ceil(4));
        let eighth = (quarter.0.div_ceil(2), quarter.1.div_ceil(2));
        let bloom_images = [
            ("bloom A", quarter),
            ("bloom B", quarter),
            ("bloom C", eighth),
            ("bloom D", eighth),
        ]
        .into_iter()
        .map(|(name, (w, h))| {
            Image::new(
                ctx,
                vk::ImageType::TYPE_2D,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::Extent3D {
                    width: w,
                    height: h,
                    depth: 1,
                },
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                name,
            )
        })
        .collect::<Result<Vec<_>>>()?;
        self.output = Some(output);
        self.hdr = Some(hdr);
        self.pt_images = pt_images;
        self.bloom_images = bloom_images;

        self.oidn_buffers.drain(..).for_each(|mut b| b.destroy(ctx));
        #[cfg(feature = "oidn")]
        self.oidn_shared.drain(..).for_each(|b| b.destroy(ctx));
        self.accumulated_params = None;
        self.write_descriptors(ctx);
        Ok(())
    }

    fn write_uniforms(&mut self, slot: usize, uniforms: &Uniforms) {
        self.uniforms[slot].write(std::slice::from_ref(uniforms));
    }

    unsafe fn bind(
        &self,
        ctx: &Context,
        cmd: vk::CommandBuffer,
        pipeline: vk::Pipeline,
        slot: usize,
    ) {
        unsafe {
            ctx.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            ctx.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.sets[slot]],
                &[],
            );
        }
    }

    unsafe fn push(&self, ctx: &Context, cmd: vk::CommandBuffer, push: Push) {
        unsafe {
            ctx.device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            )
        };
    }

    /// Enabled clip planes in voxel-index space, scaled so that `dot(n, idx) - d`
    /// is the signed distance in mm into the cut side.
    fn clip_uniforms(&self, clipping: &Clipping) -> (UVec4, [Vec4; 3], [Vec4; 3]) {
        let a = Mat3::from_mat4(self.index_to_world);
        let b = self.index_to_world.w_axis.truncate();
        let (mut planes, mut normals, mut n) = ([Vec4::ZERO; 3], [Vec4::ZERO; 3], 0);
        for p in clipping.planes.iter().filter(|p| p.enabled) {
            let sign = if p.cut_positive { 1.0 } else { -1.0 };
            let normal = Vec3::AXES[p.plane.axis()] * sign;
            // World point = a * idx + b, so dot(normal, world) - d = dot(aᵀ normal, idx) - (d - normal·b).
            planes[n] = (a.transpose() * normal).extend(sign * p.position - normal.dot(b));
            normals[n] = normal.extend(0.0);
            n += 1;
        }
        (
            UVec4::new(n as u32, clipping.all as u32, 0, 0),
            planes,
            normals,
        )
    }

    fn opacity_key(&self, labels: &LabelParams) -> OpacityKey {
        let (mut visible, mut ghost) = ([0u64; 4], [0u64; 4]);
        for i in 0..256 {
            if labels.colors[i][3] > 0.5 {
                visible[i / 64] |= 1 << (i % 64);
            }
            if labels.ghost[i] {
                ghost[i / 64] |= 1 << (i % 64);
            }
        }
        OpacityKey {
            generation: self.generation,
            mode: if self.has_labels {
                labels.mode
            } else {
                LabelMode::Off
            },
            extinction_bits: labels.extinction.to_bits(),
            visible,
            ghost,
            ghost_opacity_bits: labels.ghost_opacity.to_bits(),
            // The mask pass costs (2 * margin + 1)^3 fetches per unlabelled voxel.
            margin: (labels.cut_margin_mm / self.min_spacing)
                .round()
                .clamp(0.0, 3.0) as i32,
            smoothing: labels.smoothing.min(8),
        }
    }

    /// Records the render passes for `frame`, leaving the output image in `final_layout`.
    /// Shadow and ambient-occlusion volumes are recomputed first when their inputs changed.
    pub fn record(
        &mut self,
        ctx: &Context,
        cmd: vk::CommandBuffer,
        frame: u64,
        params: &FrameParams,
        final_layout: vk::ImageLayout,
    ) {
        let output = self
            .output
            .as_ref()
            .expect("resize() must be called before record()");
        let (width, height) = (output.extent.width, output.extent.height);
        let (output_handle, hdr_handle) = (output.handle, self.hdr.as_ref().unwrap().handle);
        let slot = (frame % FRAMES_IN_FLIGHT as u64) as usize;
        self.last_slot = slot;

        let cam = &params.camera;
        let (right, up) = cam.basis();
        let tan_half = (cam.fov_y / 2.0).tan();
        let aspect = width as f32 / height as f32;
        let light = &params.lighting;
        let light_dir = light.light_direction(cam);
        let labels = &params.labels;
        let (clip, clip_planes, clip_normals) = self.clip_uniforms(&params.clipping);
        let environment = match (light.environment, self.env_peak_u) {
            (Environment::Map, None) => Environment::Studio,
            (e, _) => e,
        };
        // Shaders map azimuth phi (from the view axis towards the right) to the
        // light map's column u = 0.5 + (phi + rotation) / 2pi; turn the map's
        // brightest column to the key light's azimuth.
        let env_rotation = self.env_peak_u.map_or(0.0, |peak_u| {
            let key_phi = light.azimuth.sin().atan2(-light.azimuth.cos());
            (peak_u - 0.5) * std::f32::consts::TAU - key_phi + light.env_rotation
        });
        // Path tracing accumulates until anything that affects the image changes.
        // Denoising only changes what is displayed.
        let mut accumulation_key = *params;
        if let Some(pt) = &mut accumulation_key.path_tracing {
            pt.denoiser = Denoiser::Off;
        }
        if params.path_tracing.is_some() && self.accumulated_params != Some(accumulation_key) {
            self.accumulated = 0;
            self.accumulated_params = Some(accumulation_key);
        }
        let skip = params.skip_empty_space as u32;
        let path = params.path_tracing.map_or(UVec4::new(0, 0, 0, skip), |pt| {
            UVec4::new(
                self.accumulated,
                pt.bounces,
                pt.samples_per_frame.max(1),
                skip,
            )
        });
        self.write_uniforms(
            slot,
            &Uniforms {
                world_to_index: self.world_to_index,
                eye: cam.eye().extend(1.0),
                forward: cam.forward().extend(0.0),
                right: (right * tan_half * aspect).extend(0.0),
                up: (up * tan_half).extend(0.0),
                dims: self.dims.extend(params.step_mm),
                flags: UVec4::new(frame as u32, params.srgb_encode as u32, width, height),
                labels: Vec4::new(
                    match labels.mode {
                        _ if !self.has_labels => 0.0,
                        LabelMode::Off => 0.0,
                        LabelMode::Tint => 1.0,
                        LabelMode::Isolate => 2.0,
                        LabelMode::Hybrid => 3.0,
                    },
                    labels.tint,
                    labels.extinction,
                    labels.ghost_opacity,
                ),
                light_dir: light_dir.extend(light.intensity),
                // The studio is dark but for its fill and rim lights: on average
                // half as bright as the dome, which the ray marcher's uniform
                // ambient stands in for.
                ambient: Vec3::splat(match environment {
                    Environment::Dome | Environment::Map => light.ambient,
                    Environment::Studio => 0.5 * light.ambient,
                })
                .extend(light.exposure.exp2()),
                material: Vec4::new(
                    light.roughness,
                    light.metalness,
                    match light.tone_mapper {
                        ToneMapper::None => 0.0,
                        ToneMapper::Aces => 1.0,
                        ToneMapper::Agx => 2.0,
                    },
                    light.light_size,
                ),
                lighting: UVec4::new(
                    light.shadows as u32,
                    light.ambient_occlusion as u32,
                    params.preintegrate as u32,
                    environment as u32,
                ),
                light_cells: Vec4::new(
                    SHADOW_DOWNSAMPLE as f32 * self.max_spacing,
                    AO_DOWNSAMPLE as f32 * self.max_spacing,
                    0.0,
                    0.0,
                ),
                clip,
                clip_planes,
                clip_normals,
                path,
                env: Vec4::new(
                    env_rotation,
                    params.smooth_sampling as u32 as f32,
                    light.anisotropy,
                    light.chromatic,
                ),
                lens: params.path_tracing.map_or(Vec4::ZERO, |pt| {
                    let focus = pt
                        .focus
                        .map_or(cam.distance, |p| (p - cam.eye()).dot(cam.forward()));
                    Vec4::new(pt.aperture, focus.max(1.0), 0.0, 0.0)
                }),
                finish: Vec4::new(
                    light.bloom,
                    light.vignette,
                    light.grain,
                    light.auto_exposure as u32 as f32,
                ),
                tf_bucket_max: self.tf_bucket_max,
                subsurface: Vec4::new(
                    light.subsurface * light.subsurface_strength,
                    10.0 / light.scattering_distance.max(0.1),
                    SUBSURFACE_SPREAD,
                    SHADOW_DOWNSAMPLE as f32 * self.max_spacing,
                ),
                // Alpha: 0 hidden, 1 shown, 2 ghosted.
                label_colors: std::array::from_fn(|i| {
                    let c = Vec4::from_array(labels.colors[i]);
                    if c.w > 0.5 && labels.ghost[i] {
                        c.with_w(2.0)
                    } else {
                        c
                    }
                }),
                label_density: labels.density.map(Vec4::from_array),
                label_materials: std::array::from_fn(|i| {
                    let [r, m, sss, coat] = labels.materials[i];
                    Vec4::new(r, m, coat, sss * light.subsurface_strength)
                }),
            },
        );

        let opacity = self.opacity_key(labels);
        let clip_key = ClipKey(std::array::from_fn(|i| match i {
            0 => clip.x,
            1 => clip.y,
            _ => clip_planes[(i - 2) / 4][(i - 2) % 4].to_bits(),
        }));
        let device = &ctx.device;
        unsafe {
            // The mask feeds every pass that decides opacity, so it goes first.
            if opacity.mode != LabelMode::Off && self.mask_key != Some(opacity) {
                let d = self.mask.extent;
                self.bind(ctx, cmd, self.mask_pipeline, slot);
                // Binary masks, then the blur along x, y and z (see mask.comp).
                for pass in 0..4 {
                    self.push(
                        ctx,
                        cmd,
                        Push {
                            target_dims: UVec4::new(d.width, d.height, d.depth, 0),
                            sweep: [opacity.margin, pass, opacity.smoothing as i32, 0],
                            soft: self.soft_push,
                        },
                    );
                    device.cmd_dispatch(
                        cmd,
                        d.width.div_ceil(4),
                        d.height.div_ceil(4),
                        d.depth.div_ceil(4),
                    );
                    compute_barrier(device, cmd);
                }
                self.mask_key = Some(opacity);
            }
            if self.majorant_key != Some(opacity) {
                let d = self.majorant.extent;
                self.bind(ctx, cmd, self.majorant_pipeline, slot);
                device.cmd_dispatch(
                    cmd,
                    d.width.div_ceil(4),
                    d.height.div_ceil(4),
                    d.depth.div_ceil(4),
                );
                compute_barrier(device, cmd);
                self.majorant_key = Some(opacity);
            }
            // The path tracer computes its own lighting.
            let ray_march = params.path_tracing.is_none();
            if ray_march && light.ambient_occlusion && self.ao_key != Some((opacity, clip_key)) {
                let d = self.ao.extent;
                self.bind(ctx, cmd, self.ao_pipeline, slot);
                self.push(
                    ctx,
                    cmd,
                    Push {
                        target_dims: UVec4::new(d.width, d.height, d.depth, 0),
                        sweep: [0; 4],
                        soft: [0; 4],
                    },
                );
                device.cmd_dispatch(
                    cmd,
                    d.width.div_ceil(4),
                    d.height.div_ceil(4),
                    d.depth.div_ceil(4),
                );
                compute_barrier(device, cmd);
                self.ao_key = Some((opacity, clip_key));
            }
            let light_bits = light_dir.to_array().map(f32::to_bits);
            // The diffused light also depends on material hues and the scattering distance.
            let tint_key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                labels
                    .colors
                    .iter()
                    .flatten()
                    .for_each(|c| c.to_bits().hash(&mut h));
                light.scattering_distance.to_bits().hash(&mut h);
                h.finish()
            };
            if ray_march
                && light.shadows
                && self.shadow_key != Some((opacity, clip_key, light_bits, tint_key))
            {
                self.record_shadow_sweep(ctx, cmd, slot, light_dir);
                self.shadow_key = Some((opacity, clip_key, light_bits, tint_key));
            }

            let path_tracing = params.path_tracing.filter(|_| !self.pt_images.is_empty());
            // With OIDN the display is produced by denoise_oidn after the frame, so
            // the output keeps showing the last denoised image until then.
            let oidn = path_tracing.is_some_and(|pt| pt.denoiser == Denoiser::Oidn);
            transition(
                device,
                cmd,
                hdr_handle,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );
            if !oidn {
                transition(
                    device,
                    cmd,
                    output_handle,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                );
            }
            if path_tracing.is_some() {
                // Accumulation and guides keep their contents between frames of one
                // image; the denoiser's scratch images do not.
                let restart = self.accumulated == 0;
                for (i, pt_image) in self.pt_images.iter().enumerate() {
                    let old = if restart || i >= 3 {
                        vk::ImageLayout::UNDEFINED
                    } else {
                        vk::ImageLayout::GENERAL
                    };
                    transition(device, cmd, pt_image.handle, old, vk::ImageLayout::GENERAL);
                }
                self.bind(ctx, cmd, self.pt_pipeline, slot);
            } else {
                self.bind(ctx, cmd, self.dvr, slot);
            }
            device.cmd_dispatch(
                cmd,
                width.div_ceil(WORKGROUP),
                height.div_ceil(WORKGROUP),
                1,
            );
            compute_barrier(device, cmd);
            if let Some(pt) = path_tracing {
                if pt.denoiser == Denoiser::Atrous {
                    self.record_denoise(ctx, cmd, slot, width, height);
                }
                self.accumulated += pt.samples_per_frame.max(1);
            }
            if oidn {
                return;
            }
            self.record_post(ctx, cmd, slot, width, height);
            transition(
                device,
                cmd,
                output_handle,
                vk::ImageLayout::GENERAL,
                final_layout,
            );
        }
    }

    /// Prepare, filter at growing steps (ping-ponging) and finish into the HDR image.
    /// Camera finishing (bloom.comp: bloom and auto exposure) and tone mapping
    /// of the HDR image into the output (left in `GENERAL`).
    unsafe fn record_post(
        &self,
        ctx: &Context,
        cmd: vk::CommandBuffer,
        slot: usize,
        width: u32,
        height: u32,
    ) {
        let device = &ctx.device;
        unsafe {
            for image in &self.bloom_images {
                transition(
                    device,
                    cmd,
                    image.handle,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                );
            }
            compute_barrier(device, cmd);
            self.bind(ctx, cmd, self.bloom_pipeline, slot);
            for pass in 0..BLOOM_PASSES {
                let e = if pass >= 4 { 1 } else { 0 };
                let extent = self.bloom_images[2 * e].extent;
                let (w, h) = (extent.width, extent.height);
                self.push(
                    ctx,
                    cmd,
                    Push {
                        target_dims: UVec4::new(w, h, 1, 0),
                        sweep: [pass, 0, 0, 0],
                        soft: [0; 4],
                    },
                );
                if pass == 1 {
                    device.cmd_dispatch(cmd, 1, 1, 1);
                } else {
                    device.cmd_dispatch(cmd, w.div_ceil(WORKGROUP), h.div_ceil(WORKGROUP), 1);
                }
                compute_barrier(device, cmd);
            }
            self.bind(ctx, cmd, self.tonemap, slot);
            device.cmd_dispatch(
                cmd,
                width.div_ceil(WORKGROUP),
                height.div_ceil(WORKGROUP),
                1,
            );
        }
    }

    unsafe fn record_denoise(
        &self,
        ctx: &Context,
        cmd: vk::CommandBuffer,
        slot: usize,
        width: u32,
        height: u32,
    ) {
        let target_dims = UVec4::new(width, height, 1, 0);
        let mut passes = vec![[0, 0, 0, 0]];
        for level in 0..DENOISE_LEVELS {
            // Level 0 reads ping (written by the preparation), then alternate.
            passes.push([1 + level, 1 << level, level % 2, 0]);
        }
        passes.push([-1, 0, DENOISE_LEVELS % 2, 0]);
        unsafe {
            self.bind(ctx, cmd, self.denoise_pipeline, slot);
            for sweep in passes {
                self.push(
                    ctx,
                    cmd,
                    Push {
                        target_dims,
                        sweep,
                        soft: [0; 4],
                    },
                );
                ctx.device.cmd_dispatch(
                    cmd,
                    width.div_ceil(WORKGROUP),
                    height.div_ceil(WORKGROUP),
                    1,
                );
                compute_barrier(&ctx.device, cmd);
            }
        }
    }

    /// One dispatch per slice along the light's dominant axis, starting at the lit side.
    unsafe fn record_shadow_sweep(
        &self,
        ctx: &Context,
        cmd: vk::CommandBuffer,
        slot: usize,
        light_dir: Vec3,
    ) {
        let e = self.shadow.extent;
        let sdims = UVec3::new(e.width, e.height, e.depth);
        // Must match `l_idx` in shadow.comp.
        let l_idx =
            Mat3::from_mat4(self.world_to_index) * light_dir * (sdims.as_vec3() / self.dims);
        let abs = l_idx.abs();
        let axis = if abs.x >= abs.y && abs.x >= abs.z {
            0
        } else if abs.y >= abs.z {
            1
        } else {
            2
        };
        let sign = if l_idx[axis] >= 0.0 { 1 } else { -1 };
        let (u_axis, v_axis) = ((axis + 1) % 3, (axis + 2) % 3);
        let n = sdims[axis] as i32;
        unsafe {
            self.bind(ctx, cmd, self.shadow_pipeline, slot);
            for i in 0..n {
                let slice = if sign > 0 { n - 1 - i } else { i };
                self.push(
                    ctx,
                    cmd,
                    Push {
                        target_dims: sdims.extend(0),
                        sweep: [axis as i32, slice, sign, 0],
                        soft: [0; 4],
                    },
                );
                ctx.device.cmd_dispatch(
                    cmd,
                    sdims[u_axis].div_ceil(WORKGROUP),
                    sdims[v_axis].div_ceil(WORKGROUP),
                    1,
                );
                compute_barrier(&ctx.device, cmd);
            }
        }
    }

    fn destroy_output(&mut self, ctx: &Context) {
        if let Some(mut old) = self.output.take() {
            unsafe { ctx.device.destroy_image_view(self.output_srgb_view, None) };
            self.output_srgb_view = vk::ImageView::null();
            old.destroy(ctx);
        }
        for mut image in self.pt_images.drain(..).chain(self.bloom_images.drain(..)) {
            image.destroy(ctx);
        }
        if let Some(mut hdr) = self.hdr.take() {
            hdr.destroy(ctx);
        }
    }

    fn write_descriptors(&self, ctx: &Context) {
        let image = |sampler, view, layout| {
            [vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(view)
                .image_layout(layout)]
        };
        let ro = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let general = vk::ImageLayout::GENERAL;
        let null = vk::Sampler::null();
        let sampled = [
            (1, image(self.sampler, self.density.view, ro)),
            (2, image(self.sampler, self.tf.view, ro)),
            (4, image(self.nearest_sampler, self.labels.view, ro)),
            (5, image(self.sampler, self.normals.view, general)),
            (6, image(self.border_sampler, self.shadow.view, general)),
            (7, image(self.border_sampler, self.ao.view, general)),
            (12, image(self.nearest_sampler, self.preint.view, ro)),
            (13, image(self.sampler, self.mask.view, general)),
            (16, image(self.nearest_sampler, self.mask_b.view, general)),
            (22, image(self.nearest_sampler, self.majorant.view, general)),
            (24, image(self.nearest_sampler, self.soft.view, ro)),
            (25, image(self.sampler, self.env_radiance.view, ro)),
            (26, image(self.nearest_sampler, self.env_table.view, ro)),
            (27, image(self.sampler, self.env_diffuse.view, ro)),
        ];
        let mut storage = vec![
            (9, image(null, self.normals.view, general)),
            (10, image(null, self.shadow.view, general)),
            (11, image(null, self.ao.view, general)),
            (14, image(null, self.mask.view, general)),
            (15, image(null, self.mask_b.view, general)),
            (23, image(null, self.majorant.view, general)),
        ];
        if let (Some(output), Some(hdr)) = (&self.output, &self.hdr) {
            storage.push((3, image(null, output.view, general)));
            storage.push((8, image(null, hdr.view, general)));
        }
        for (i, pt_image) in self.pt_images.iter().enumerate() {
            storage.push((17 + i as u32, image(null, pt_image.view, general)));
        }
        storage.push((34, image(null, self.exposure.view, general)));

        let mut sampled = sampled.to_vec();
        if let [a, _, c, _] = self.bloom_images.as_slice() {
            for (i, bloom) in self.bloom_images.iter().enumerate() {
                storage.push((28 + i as u32, image(null, bloom.view, general)));
            }
            sampled.push((32, image(self.sampler, a.view, general)));
            sampled.push((33, image(self.sampler, c.view, general)));
        }
        let buffer_infos: Vec<_> = self
            .uniforms
            .iter()
            .map(|b| {
                [vk::DescriptorBufferInfo::default()
                    .buffer(b.handle)
                    .range(b.size)]
            })
            .collect();

        let mut writes = Vec::new();
        for (set, buffer_info) in self.sets.iter().zip(&buffer_infos) {
            let write = |binding: u32, ty| {
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(binding)
                    .descriptor_type(ty)
            };
            writes.push(write(0, vk::DescriptorType::UNIFORM_BUFFER).buffer_info(buffer_info));
            for (binding, info) in &sampled {
                writes.push(
                    write(*binding, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(info),
                );
            }
            for (binding, info) in &storage {
                writes.push(write(*binding, vk::DescriptorType::STORAGE_IMAGE).image_info(info));
            }
        }
        #[cfg(feature = "oidn")]
        let shared_infos: Vec<[vk::DescriptorBufferInfo; 1]> = self
            .oidn_shared
            .iter()
            .map(|b| {
                [vk::DescriptorBufferInfo::default()
                    .buffer(b.buffer)
                    .range(vk::WHOLE_SIZE)]
            })
            .collect();
        #[cfg(feature = "oidn")]
        for set in &self.sets {
            for (i, info) in shared_infos.iter().enumerate() {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(35 + i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info),
                );
            }
        }
        unsafe { ctx.device.update_descriptor_sets(&writes, &[]) };
    }

    pub fn destroy(&mut self, ctx: &Context) {
        self.oidn_buffers.drain(..).for_each(|mut b| b.destroy(ctx));
        #[cfg(feature = "oidn")]
        self.oidn_shared.drain(..).for_each(|b| b.destroy(ctx));
        unsafe {
            let _ = ctx.device.device_wait_idle();
            for p in [
                self.dvr,
                self.tonemap,
                self.shadow_pipeline,
                self.ao_pipeline,
                self.mask_pipeline,
                self.pt_pipeline,
                self.denoise_pipeline,
                self.majorant_pipeline,
                self.bloom_pipeline,
                self.oidn_pipeline,
            ] {
                ctx.device.destroy_pipeline(p, None);
            }
            ctx.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            ctx.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            ctx.device
                .destroy_descriptor_set_layout(self.set_layout, None);
            for s in [self.sampler, self.nearest_sampler, self.border_sampler] {
                ctx.device.destroy_sampler(s, None);
            }
        }
        for b in &mut self.uniforms {
            b.destroy(ctx);
        }
        self.destroy_output(ctx);
        for image in [
            &mut self.tf,
            &mut self.preint,
            &mut self.labels,
            &mut self.soft,
            &mut self.env_radiance,
            &mut self.env_table,
            &mut self.env_diffuse,
            &mut self.density,
            &mut self.normals,
            &mut self.shadow,
            &mut self.ao,
            &mut self.mask,
            &mut self.mask_b,
            &mut self.majorant,
            &mut self.exposure,
        ] {
            image.destroy(ctx);
        }
    }
}

impl Uniforms {
    /// Just the geometry, for passes that need nothing else.
    fn zeroed_with(world_to_index: Mat4, dims: Vec3) -> Self {
        Self {
            world_to_index,
            dims: dims.extend(0.0),
            ..bytemuck::Zeroable::zeroed()
        }
    }
}

fn downsampled(dims: UVec3, factor: u32) -> UVec3 {
    UVec3::new(
        dims.x.div_ceil(factor),
        dims.y.div_ceil(factor),
        dims.z.div_ceil(factor),
    )
}

/// Makes compute writes visible to subsequent compute reads and writes.
fn compute_barrier(device: &ash::Device, cmd: vk::CommandBuffer) {
    let barrier = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)];
    unsafe {
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().memory_barriers(&barrier),
        )
    };
}

/// Largest extinction of the LUT in each of 64 density buckets, including the
/// first entry of the next bucket (densities are interpolated between entries).
fn bucket_max(tf: &TransferFunction) -> [Vec4; 16] {
    let lut = tf.to_lut();
    let per = LUT_SIZE / 64;
    let m: Vec<f32> = (0..64)
        .map(|b| {
            let end = ((b + 1) * per + 1).min(LUT_SIZE);
            lut[b * per..end].iter().map(|e| e[3]).fold(0.0, f32::max)
        })
        .collect();
    std::array::from_fn(|i| Vec4::new(m[4 * i], m[4 * i + 1], m[4 * i + 2], m[4 * i + 3]))
}

/// Uploads the transfer-function LUT and its pre-integration table.
fn upload_tf(ctx: &Context, tf: &TransferFunction) -> Result<(Image, Image)> {
    let lut: Vec<u16> = tf
        .to_lut()
        .iter()
        .flat_map(|rgba| rgba.map(|v| half::f16::from_f32(v).to_bits()))
        .collect();
    let extent = vk::Extent3D {
        width: LUT_SIZE as u32,
        height: 1,
        depth: 1,
    };
    let lut = Image::with_data(
        ctx,
        vk::ImageType::TYPE_1D,
        vk::Format::R16G16B16A16_SFLOAT,
        extent,
        bytemuck::cast_slice(&lut),
        "transfer function",
    )?;
    let preint = Image::with_data(
        ctx,
        vk::ImageType::TYPE_1D,
        vk::Format::R32G32B32A32_SFLOAT,
        extent,
        bytemuck::cast_slice(&tf.cumulative_integrals()),
        "pre-integration",
    )?;
    Ok((lut, preint))
}

fn upload_soft(ctx: &Context, data: &[[u8; 4]], dims: UVec3) -> Result<Image> {
    Image::with_data(
        ctx,
        vk::ImageType::TYPE_3D,
        vk::Format::R8G8B8A8_UINT,
        vk::Extent3D {
            width: dims.x,
            height: dims.y,
            depth: dims.z,
        },
        data.as_flattened(),
        "soft labels",
    )
}

/// Light map images (radiance, sampling table, diffuse) and the brightest
/// column; 1-texel placeholders for `None`.
fn upload_environment(
    ctx: &Context,
    map: Option<&EnvironmentMap>,
) -> Result<([Image; 3], Option<f32>)> {
    let prepared = map.map(|m| (m.prepare(), m.width as u32, m.height as u32));
    let (radiance, table, diffuse, (w, h), peak_u) = match &prepared {
        Some((p, w, h)) => (
            p.radiance.as_slice(),
            p.table.as_slice(),
            p.diffuse.as_slice(),
            (*w, *h),
            Some(p.peak_u),
        ),
        None => (
            &[[0.0; 4]][..],
            &[[1.0, 1.0]][..],
            &[[0.0; 4]][..],
            (1, 1),
            None,
        ),
    };
    let extent = |width, height| vk::Extent3D {
        width,
        height,
        depth: 1,
    };
    let half = |data: &[[f32; 4]]| -> Vec<u16> {
        data.iter()
            .flat_map(|c| c.map(|v| half::f16::from_f32(v.min(60000.0)).to_bits()))
            .collect()
    };
    let (dw, dh) = if peak_u.is_some() {
        (DIFFUSE_WIDTH as u32, DIFFUSE_WIDTH as u32 / 2)
    } else {
        (1, 1)
    };
    let table_w = if peak_u.is_some() { w + 1 } else { 1 };
    Ok((
        [
            Image::with_data(
                ctx,
                vk::ImageType::TYPE_2D,
                vk::Format::R16G16B16A16_SFLOAT,
                extent(w, h),
                bytemuck::cast_slice(&half(radiance)),
                "light map",
            )?,
            Image::with_data(
                ctx,
                vk::ImageType::TYPE_2D,
                vk::Format::R32G32_SFLOAT,
                extent(table_w, h),
                bytemuck::cast_slice(table),
                "light map sampling",
            )?,
            Image::with_data(
                ctx,
                vk::ImageType::TYPE_2D,
                vk::Format::R16G16B16A16_SFLOAT,
                extent(dw, dh),
                bytemuck::cast_slice(&half(diffuse)),
                "light map diffuse",
            )?,
        ],
        peak_u,
    ))
}

fn upload_labels(ctx: &Context, labels: &[u8], dims: UVec3) -> Result<Image> {
    Image::with_data(
        ctx,
        vk::ImageType::TYPE_3D,
        vk::Format::R8_UINT,
        vk::Extent3D {
            width: dims.x,
            height: dims.y,
            depth: dims.z,
        },
        labels,
        "labels",
    )
}

pub(crate) fn create_compute_pipeline(
    ctx: &Context,
    layout: vk::PipelineLayout,
    spirv: &[u8],
) -> Result<vk::Pipeline> {
    let code = ash::util::read_spv(&mut std::io::Cursor::new(spirv))?;
    let module = unsafe {
        ctx.device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?
    };
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(c"main");
    let pipeline = unsafe {
        ctx.device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)],
            None,
        )
    };
    unsafe { ctx.device.destroy_shader_module(module, None) };
    pipeline.map(|p| p[0]).map_err(|(_, e)| e.into())
}

/// A device-local Vulkan buffer whose memory is also imported into OIDN.
#[cfg(feature = "oidn")]
struct SharedBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    /// OIDN's view of the memory; released before the memory is freed.
    oidn: oidn::Buffer,
}

#[cfg(feature = "oidn")]
impl SharedBuffer {
    /// A storage buffer of `size` bytes in its own exportable allocation,
    /// imported into `oidn` through a file descriptor.
    fn new(ctx: &Context, oidn: &crate::oidn::Oidn, size: u64) -> Result<Self> {
        let Some(fd_fn) = &ctx.external_memory_fd else {
            bail!("VK_KHR_external_memory_fd is not supported");
        };
        let handle = vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;
        let device = &ctx.device;
        unsafe {
            let buffer = device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                    .push_next(&mut vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle)),
                None,
            )?;
            let requirements = device.get_buffer_memory_requirements(buffer);
            let properties = ctx
                .instance
                .get_physical_device_memory_properties(ctx.physical_device);
            let Some(memory_type) = (0..properties.memory_type_count).find(|&i| {
                requirements.memory_type_bits & (1 << i) != 0
                    && properties.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            }) else {
                device.destroy_buffer(buffer, None);
                bail!("no device-local memory type for a shared buffer");
            };
            let memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type)
                    .push_next(&mut vk::ExportMemoryAllocateInfo::default().handle_types(handle))
                    .push_next(&mut vk::MemoryDedicatedAllocateInfo::default().buffer(buffer)),
                None,
            );
            let memory = match memory {
                Ok(m) => m,
                Err(e) => {
                    device.destroy_buffer(buffer, None);
                    return Err(e.into());
                }
            };
            let cleanup = |e: anyhow::Error| {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
                e
            };
            device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|e| cleanup(e.into()))?;
            let fd = fd_fn
                .get_memory_fd(
                    &vk::MemoryGetFdInfoKHR::default()
                        .memory(memory)
                        .handle_type(handle),
                )
                .map_err(|e| cleanup(e.into()))?;
            // OIDN checks the buffer holds exactly width x height x 3 floats;
            // the allocation may be larger.
            let oidn = oidn.import(fd, size as usize).map_err(cleanup)?;
            Ok(Self {
                buffer,
                memory,
                oidn,
            })
        }
    }

    fn destroy(self, ctx: &Context) {
        drop(self.oidn);
        unsafe {
            ctx.device.destroy_buffer(self.buffer, None);
            ctx.device.free_memory(self.memory, None);
        }
    }
}
