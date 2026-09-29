use anyhow::Result;
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};

use crate::Context;

/// A buffer with its memory. Must be released with [`Buffer::destroy`].
pub struct Buffer {
    pub handle: vk::Buffer,
    pub size: u64,
    allocation: Option<Allocation>,
}

impl Buffer {
    pub fn new(
        ctx: &Context,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> Result<Self> {
        let handle = unsafe {
            ctx.device.create_buffer(
                &vk::BufferCreateInfo::default().size(size).usage(usage),
                None,
            )?
        };
        let requirements = unsafe { ctx.device.get_buffer_memory_requirements(handle) };
        let allocation = ctx
            .allocator
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;
        unsafe {
            ctx.device
                .bind_buffer_memory(handle, allocation.memory(), allocation.offset())?
        };
        Ok(Buffer {
            handle,
            size,
            allocation: Some(allocation),
        })
    }

    /// Host-visible contents; `None` for device-local memory.
    pub fn mapped(&mut self) -> Option<&mut [u8]> {
        self.allocation.as_mut()?.mapped_slice_mut()
    }

    /// Host-visible contents for reading; `None` for device-local memory.
    pub fn mapped_ref(&self) -> Option<&[u8]> {
        self.allocation.as_ref()?.mapped_slice()
    }

    pub fn write<T: bytemuck::Pod>(&mut self, data: &[T]) {
        let bytes = bytemuck::cast_slice(data);
        self.mapped().expect("buffer is not host visible")[..bytes.len()].copy_from_slice(bytes);
    }

    pub fn destroy(&mut self, ctx: &Context) {
        if let Some(allocation) = self.allocation.take() {
            let _ = ctx.allocator.lock().unwrap().free(allocation);
            unsafe { ctx.device.destroy_buffer(self.handle, None) };
        }
    }
}

/// An image, its view and memory. Must be released with [`Image::destroy`].
pub struct Image {
    pub handle: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    allocation: Option<Allocation>,
}

impl Image {
    pub fn new(
        ctx: &Context,
        image_type: vk::ImageType,
        format: vk::Format,
        extent: vk::Extent3D,
        usage: vk::ImageUsageFlags,
        name: &str,
    ) -> Result<Self> {
        Self::with_flags(
            ctx,
            image_type,
            format,
            extent,
            usage,
            vk::ImageCreateFlags::empty(),
            name,
        )
    }

    pub fn with_flags(
        ctx: &Context,
        image_type: vk::ImageType,
        format: vk::Format,
        extent: vk::Extent3D,
        usage: vk::ImageUsageFlags,
        flags: vk::ImageCreateFlags,
        name: &str,
    ) -> Result<Self> {
        let handle = unsafe {
            ctx.device.create_image(
                &vk::ImageCreateInfo::default()
                    .flags(flags)
                    .image_type(image_type)
                    .format(format)
                    .extent(extent)
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?
        };
        let requirements = unsafe { ctx.device.get_image_memory_requirements(handle) };
        let allocation = ctx
            .allocator
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;
        unsafe {
            ctx.device
                .bind_image_memory(handle, allocation.memory(), allocation.offset())?
        };
        let view_type = match image_type {
            vk::ImageType::TYPE_1D => vk::ImageViewType::TYPE_1D,
            vk::ImageType::TYPE_3D => vk::ImageViewType::TYPE_3D,
            _ => vk::ImageViewType::TYPE_2D,
        };
        let view = unsafe {
            ctx.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(handle)
                    .view_type(view_type)
                    .format(format)
                    .subresource_range(COLOR_RANGE),
                None,
            )?
        };
        Ok(Image {
            handle,
            view,
            format,
            extent,
            allocation: Some(allocation),
        })
    }

    /// Creates a sampled image and fills it from `data` via a staging buffer.
    pub fn with_data(
        ctx: &Context,
        image_type: vk::ImageType,
        format: vk::Format,
        extent: vk::Extent3D,
        data: &[u8],
        name: &str,
    ) -> Result<Self> {
        let image = Image::new(
            ctx,
            image_type,
            format,
            extent,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            name,
        )?;
        let mut staging = Buffer::new(
            ctx,
            data.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "staging",
        )?;
        staging.mapped().unwrap()[..data.len()].copy_from_slice(data);
        ctx.one_shot(|cmd| unsafe {
            transition(
                &ctx.device,
                cmd,
                image.handle,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            ctx.device.cmd_copy_buffer_to_image(
                cmd,
                staging.handle,
                image.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_subresource(COLOR_LAYERS)
                    .image_extent(extent)],
            );
            transition(
                &ctx.device,
                cmd,
                image.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
        })?;
        staging.destroy(ctx);
        Ok(image)
    }

    pub fn destroy(&mut self, ctx: &Context) {
        if let Some(allocation) = self.allocation.take() {
            unsafe {
                ctx.device.destroy_image_view(self.view, None);
                ctx.device.destroy_image(self.handle, None);
            }
            let _ = ctx.allocator.lock().unwrap().free(allocation);
        }
    }
}

pub const COLOR_RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

pub const COLOR_LAYERS: vk::ImageSubresourceLayers = vk::ImageSubresourceLayers {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    mip_level: 0,
    base_array_layer: 0,
    layer_count: 1,
};

/// Full-pipeline image layout transition. Coarse, but the renderer only issues
/// a handful per frame.
pub fn transition(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
) {
    let barrier = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .old_layout(old)
        .new_layout(new)
        .image(image)
        .subresource_range(COLOR_RANGE)];
    unsafe {
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&barrier),
        )
    };
}
