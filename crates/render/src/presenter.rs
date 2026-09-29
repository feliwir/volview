use anyhow::{Result, bail};
use ash::{khr, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::renderer::FRAMES_IN_FLIGHT;
use gpu_allocator::MemoryLocation;

use crate::Context;
use crate::resources::{Buffer, COLOR_LAYERS, COLOR_RANGE, transition};

/// Background behind the draw phase, written as-is into the swapchain image.
const CLEAR_COLOR: [f32; 4] = [0.1, 0.1, 0.11, 1.0];

/// Owns the window surface and swapchain. Each frame runs a compute phase
/// (e.g. volume rendering) followed by a draw phase into the swapchain image
/// (e.g. the UI, which shows the volume image as a texture).
pub struct Presenter {
    surface: vk::SurfaceKHR,
    swapchain_fn: khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    pub extent: vk::Extent2D,
    format: vk::SurfaceFormatKHR,
    commands: Vec<vk::CommandBuffer>,
    image_available: Vec<vk::Semaphore>,
    in_flight: Vec<vk::Fence>,
    /// One per swapchain image: it may only be reused once that image is re-acquired.
    render_finished: Vec<vk::Semaphore>,
    frame: u64,
    timestamps: vk::QueryPool,
    timestamp_period_ns: f32,
    /// Per slot: whether the compute phase recorded work that was timed.
    timed: [bool; FRAMES_IN_FLIGHT],
    compute_ms: Option<f32>,
    capture_requested: bool,
    /// RGBA8 pixels of the last captured frame.
    captured: Option<(u32, u32, Vec<u8>)>,
}

impl Presenter {
    pub fn new(
        ctx: &Context,
        window: &(impl HasWindowHandle + HasDisplayHandle),
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let surface_fn = ctx
            .surface_fn
            .as_ref()
            .expect("context created without a display");
        let surface = unsafe {
            ash_window::create_surface(
                &ctx.entry,
                &ctx.instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )?
        };
        let supported = unsafe {
            surface_fn.get_physical_device_surface_support(
                ctx.physical_device,
                ctx.queue_family,
                surface,
            )?
        };
        if !supported {
            bail!("the selected queue family cannot present to this window");
        }

        let formats = unsafe {
            surface_fn.get_physical_device_surface_formats(ctx.physical_device, surface)?
        };
        let format = [vk::Format::B8G8R8A8_UNORM, vk::Format::R8G8B8A8_UNORM]
            .iter()
            .find_map(|&f| {
                formats.iter().copied().find(|sf| {
                    sf.format == f && sf.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
            })
            .unwrap_or(formats[0]);

        let n = FRAMES_IN_FLIGHT;
        let device = &ctx.device;
        let timestamps = unsafe {
            device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2 * n as u32),
                None,
            )?
        };
        let timestamp_period_ns = unsafe {
            ctx.instance
                .get_physical_device_properties(ctx.physical_device)
        }
        .limits
        .timestamp_period;
        let mut presenter = Presenter {
            surface,
            swapchain_fn: khr::swapchain::Device::new(&ctx.instance, device),
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            views: Vec::new(),
            extent: vk::Extent2D { width, height },
            format,
            commands: ctx.allocate_command_buffers(n as u32)?,
            image_available: (0..n)
                .map(|_| unsafe { device.create_semaphore(&Default::default(), None) })
                .collect::<Result<_, _>>()?,
            in_flight: (0..n)
                .map(|_| unsafe {
                    device.create_fence(
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )
                })
                .collect::<Result<_, _>>()?,
            render_finished: Vec::new(),
            frame: 0,
            timestamps,
            timestamp_period_ns,
            timed: [false; FRAMES_IN_FLIGHT],
            compute_ms: None,
            capture_requested: false,
            captured: None,
        };
        presenter.recreate(ctx, width, height)?;
        Ok(presenter)
    }

    /// Swapchain image format, for creating pipelines that draw into it.
    pub fn format(&self) -> vk::Format {
        self.format.format
    }

    /// Whether the swapchain encodes to sRGB itself (shaders should output linear colour).
    pub fn is_srgb(&self) -> bool {
        matches!(
            self.format.format,
            vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB
        )
    }

    /// GPU time of the most recent timed compute phase.
    pub fn compute_ms(&self) -> Option<f32> {
        self.compute_ms
    }

    /// Copies the next presented frame to host memory; see [`Presenter::take_capture`].
    pub fn capture_next_frame(&mut self) {
        self.capture_requested = true;
    }

    /// The captured frame as `(width, height, RGBA8 pixels)`, once available.
    pub fn take_capture(&mut self) -> Option<(u32, u32, Vec<u8>)> {
        self.captured.take()
    }

    /// Recreates the swapchain for a new window size. Waits for the GPU to go idle.
    pub fn recreate(&mut self, ctx: &Context, width: u32, height: u32) -> Result<()> {
        unsafe { ctx.device.device_wait_idle()? };
        let surface_fn = ctx.surface_fn.as_ref().unwrap();
        let caps = unsafe {
            surface_fn
                .get_physical_device_surface_capabilities(ctx.physical_device, self.surface)?
        };
        if !caps
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        {
            bail!("swapchain images cannot be colour attachments on this surface");
        }
        self.extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };
        if self.extent.width == 0 || self.extent.height == 0 {
            return Ok(()); // minimised; keep the old swapchain until we get a real size
        }
        let modes = unsafe {
            surface_fn
                .get_physical_device_surface_present_modes(ctx.physical_device, self.surface)?
        };
        let present_mode = if modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        };
        let mut image_count = caps.min_image_count + 1;
        if caps.max_image_count > 0 {
            image_count = image_count.min(caps.max_image_count);
        }

        let old = self.swapchain;
        self.swapchain = unsafe {
            self.swapchain_fn.create_swapchain(
                &vk::SwapchainCreateInfoKHR::default()
                    .surface(self.surface)
                    .min_image_count(image_count)
                    .image_format(self.format.format)
                    .image_color_space(self.format.color_space)
                    .image_extent(self.extent)
                    .image_array_layers(1)
                    .image_usage(
                        vk::ImageUsageFlags::COLOR_ATTACHMENT
                            | (caps.supported_usage_flags & vk::ImageUsageFlags::TRANSFER_SRC),
                    )
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .pre_transform(caps.current_transform)
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    .present_mode(present_mode)
                    .clipped(true)
                    .old_swapchain(old),
                None,
            )?
        };
        unsafe {
            if old != vk::SwapchainKHR::null() {
                self.swapchain_fn.destroy_swapchain(old, None);
            }
            for s in self.render_finished.drain(..) {
                ctx.device.destroy_semaphore(s, None);
            }
            for v in self.views.drain(..) {
                ctx.device.destroy_image_view(v, None);
            }
        }
        self.images = unsafe { self.swapchain_fn.get_swapchain_images(self.swapchain)? };
        self.views = self
            .images
            .iter()
            .map(|&image| unsafe {
                ctx.device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(self.format.format)
                        .subresource_range(COLOR_RANGE),
                    None,
                )
            })
            .collect::<Result<_, _>>()?;
        self.render_finished = self
            .images
            .iter()
            .map(|_| unsafe { ctx.device.create_semaphore(&Default::default(), None) })
            .collect::<Result<_, _>>()?;
        tracing::debug!(extent = ?self.extent, format = ?self.format, ?present_mode, "swapchain");
        Ok(())
    }

    /// Records and presents one frame. `compute(cmd, frame)` runs outside any
    /// rendering scope and returns whether it recorded work (which is timed);
    /// `draw(cmd, extent)` runs inside a dynamic-rendering scope on the cleared
    /// swapchain image. Returns `false` when the swapchain must be recreated.
    pub fn draw(
        &mut self,
        ctx: &Context,
        compute: impl FnOnce(vk::CommandBuffer, u64) -> bool,
        draw: impl FnOnce(vk::CommandBuffer, vk::Extent2D),
    ) -> Result<bool> {
        if self.extent.width == 0 || self.extent.height == 0 {
            return Ok(true);
        }
        let slot = (self.frame % FRAMES_IN_FLIGHT as u64) as usize;
        let device = &ctx.device;
        unsafe { device.wait_for_fences(&[self.in_flight[slot]], true, u64::MAX)? };
        if self.timed[slot] {
            let mut ticks = [0u64; 2];
            unsafe {
                device.get_query_pool_results(
                    self.timestamps,
                    2 * slot as u32,
                    &mut ticks,
                    vk::QueryResultFlags::TYPE_64,
                )?
            };
            let ns = ticks[1].wrapping_sub(ticks[0]) as f64 * self.timestamp_period_ns as f64;
            self.compute_ms = Some((ns * 1e-6) as f32);
            self.timed[slot] = false;
        }

        let index = match unsafe {
            self.swapchain_fn.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available[slot],
                vk::Fence::null(),
            )
        } {
            Ok((index, _suboptimal)) => index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(false),
            Err(e) => return Err(e.into()),
        };
        unsafe { device.reset_fences(&[self.in_flight[slot]])? };

        let cmd = self.commands[slot];
        let image = self.images[index as usize];
        let capture = if std::mem::take(&mut self.capture_requested) {
            let size = self.extent.width as u64 * self.extent.height as u64 * 4;
            Some(Buffer::new(
                ctx,
                size,
                vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
                "capture",
            )?)
        } else {
            None
        };
        let q0 = 2 * slot as u32;
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            device.cmd_reset_query_pool(cmd, self.timestamps, q0, 2);
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                self.timestamps,
                q0,
            );
            let timed = compute(cmd, self.frame);
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                self.timestamps,
                q0 + 1,
            );
            self.timed[slot] = timed;

            transition(
                device,
                cmd,
                image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
            let attachment = [vk::RenderingAttachmentInfo::default()
                .image_view(self.views[index as usize])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: CLEAR_COLOR,
                    },
                })];
            device.cmd_begin_rendering(
                cmd,
                &vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent: self.extent,
                    })
                    .layer_count(1)
                    .color_attachments(&attachment),
            );
            draw(cmd, self.extent);
            device.cmd_end_rendering(cmd);
            let mut layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
            if let Some(buffer) = &capture {
                transition(
                    device,
                    cmd,
                    image,
                    layout,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                );
                layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                device.cmd_copy_image_to_buffer(
                    cmd,
                    image,
                    layout,
                    buffer.handle,
                    &[vk::BufferImageCopy::default()
                        .image_subresource(COLOR_LAYERS)
                        .image_extent(vk::Extent3D {
                            width: self.extent.width,
                            height: self.extent.height,
                            depth: 1,
                        })],
                );
            }
            transition(device, cmd, image, layout, vk::ImageLayout::PRESENT_SRC_KHR);
            device.end_command_buffer(cmd)?;

            let wait = [vk::SemaphoreSubmitInfo::default()
                .semaphore(self.image_available[slot])
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let signal = [vk::SemaphoreSubmitInfo::default()
                .semaphore(self.render_finished[index as usize])
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            device.queue_submit2(
                ctx.queue,
                &[vk::SubmitInfo2::default()
                    .wait_semaphore_infos(&wait)
                    .command_buffer_infos(&cmds)
                    .signal_semaphore_infos(&signal)],
                self.in_flight[slot],
            )?;
        }
        self.frame += 1;
        if let Some(mut buffer) = capture {
            unsafe { device.wait_for_fences(&[self.in_flight[slot]], true, u64::MAX)? };
            let mut pixels = buffer.mapped().unwrap().to_vec();
            buffer.destroy(ctx);
            if matches!(
                self.format.format,
                vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB
            ) {
                pixels.chunks_exact_mut(4).for_each(|p| p.swap(0, 2));
            }
            self.captured = Some((self.extent.width, self.extent.height, pixels));
        }

        let wait = [self.render_finished[index as usize]];
        let swapchains = [self.swapchain];
        let indices = [index];
        match unsafe {
            self.swapchain_fn.queue_present(
                ctx.queue,
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&wait)
                    .swapchains(&swapchains)
                    .image_indices(&indices),
            )
        } {
            Ok(false) => Ok(true),
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    pub fn destroy(&mut self, ctx: &Context) {
        unsafe {
            let _ = ctx.device.device_wait_idle();
            for &s in self.image_available.iter().chain(&self.render_finished) {
                ctx.device.destroy_semaphore(s, None);
            }
            for &f in &self.in_flight {
                ctx.device.destroy_fence(f, None);
            }
            for &v in &self.views {
                ctx.device.destroy_image_view(v, None);
            }
            ctx.device.destroy_query_pool(self.timestamps, None);
            self.swapchain_fn.destroy_swapchain(self.swapchain, None);
            ctx.surface_fn
                .as_ref()
                .unwrap()
                .destroy_surface(self.surface, None);
        }
    }
}
