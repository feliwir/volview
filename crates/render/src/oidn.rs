//! Intel Open Image Denoise for the path tracer's accumulated image.

use anyhow::{Result, anyhow};

/// An OIDN device (CUDA when available, otherwise the default device, e.g. the
/// CPU) and a ray-tracing filter configured for HDR colour with albedo and
/// normal guides.
pub struct Oidn {
    device: &'static oidn::Device,
    filter: oidn::RayTracing<'static>,
    size: (usize, usize),
}

impl Oidn {
    /// Opens OIDN on the GPU with `uuid` (the Vulkan device's, so memory can be
    /// shared), else any CUDA device, else the default device (e.g. the CPU).
    pub fn new(uuid: [u8; 16]) -> Result<Self> {
        let (device, device_name) = match oidn::Device::by_uuid(&uuid) {
            Ok(device) => (device, "same GPU"),
            Err(_) => match oidn::Device::cuda() {
                Ok(device) => (device, "CUDA"),
                Err(_) => (
                    oidn::Device::new().map_err(|e| anyhow!("creating OIDN device: {e:?}"))?,
                    "default (CPU)",
                ),
            },
        };
        // The device lives for the rest of the program; leaking it lets the
        // filter, which borrows it, be kept and reused between frames.
        let device: &'static oidn::Device = Box::leak(Box::new(device));
        let mut filter = oidn::RayTracing::try_new(device)
            .map_err(|e| anyhow!("creating OIDN filter: {e:?}"))?;
        // The guides come from the same noisy paths as the colour.
        filter.hdr(true).clean_aux(false);
        tracing::info!(device = device_name, "OIDN ready");
        Ok(Self {
            device,
            filter,
            size: (0, 0),
        })
    }

    /// Whether memory exported as POSIX file descriptors (e.g. by Vulkan) can
    /// be imported, for denoising without copies through the host.
    pub fn imports_fd(&self) -> bool {
        self.device
            .external_memory_types()
            .contains(oidn::ExternalMemoryTypeFlags::OPAQUE_FD)
    }

    /// Imports `byte_size` bytes of a dedicated allocation exported as `fd`
    /// (ownership of `fd` passes to OIDN).
    ///
    /// # Safety
    /// `fd` must be an opaque-fd export of a dedicated allocation of at least
    /// `byte_size` bytes on the same physical GPU.
    pub unsafe fn import(&self, fd: std::os::fd::RawFd, byte_size: usize) -> Result<oidn::Buffer> {
        unsafe {
            self.device.create_shared_buffer_from_raw_fd(
                oidn::ExternalMemoryTypeFlags::OPAQUE_FD | oidn::ExternalMemoryTypeFlags::DEDICATED,
                fd,
                byte_size,
            )
        }
        .map_err(|e| anyhow!("importing shared memory into OIDN: {e:?}"))
    }

    /// Like [`Oidn::denoise`], on buffers in device memory (tightly packed RGB
    /// floats). Returns once the result is in `output`.
    pub fn denoise_buffers(
        &mut self,
        width: usize,
        height: usize,
        color: &oidn::Buffer,
        albedo: &oidn::Buffer,
        normal: &oidn::Buffer,
        output: &oidn::Buffer,
    ) -> Result<()> {
        if self.size != (width, height) {
            self.filter.image_dimensions(width, height);
            self.size = (width, height);
        }
        self.filter
            .albedo_normal_buffer(albedo, normal)
            .map_err(|e| anyhow!("OIDN guides: {e:?}"))?;
        self.filter
            .filter_buffer(color, output)
            .map_err(|e| anyhow!("OIDN denoising failed: {e:?}"))?;
        self.device.sync();
        Ok(())
    }

    /// Denoises `color` (linear HDR RGB, `width * height * 3` floats) into `output`,
    /// guided by first-hit `albedo` (RGB in [0, 1]) and `normal` (unit or zero).
    pub fn denoise(
        &mut self,
        width: usize,
        height: usize,
        color: &[f32],
        albedo: &[f32],
        normal: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        if self.size != (width, height) {
            self.filter.image_dimensions(width, height);
            self.size = (width, height);
        }
        self.filter.albedo_normal(albedo, normal);
        self.filter
            .filter(color, output)
            .map_err(|e| anyhow!("OIDN denoising failed: {e:?}"))
    }
}
