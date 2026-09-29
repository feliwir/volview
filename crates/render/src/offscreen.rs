use anyhow::Result;
use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::renderer::{Denoiser, FrameParams};
use crate::resources::{Buffer, COLOR_LAYERS};
use crate::{Context, Renderer};

/// Renders `frames` frames at `width` x `height` (more than one only matters when
/// path tracing, which accumulates) and returns tightly packed RGBA8 pixels.
pub fn render_to_rgba(
    ctx: &Context,
    renderer: &mut Renderer,
    params: &FrameParams,
    width: u32,
    height: u32,
    frames: u32,
) -> Result<Vec<u8>> {
    renderer.resize(ctx, width, height)?;
    let mut readback = Buffer::new(
        ctx,
        width as u64 * height as u64 * 4,
        vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuToCpu,
        "readback",
    )?;
    // OIDN denoises after the last frame instead of inside it.
    let oidn = params
        .path_tracing
        .is_some_and(|pt| pt.denoiser == Denoiser::Oidn);
    let last = if oidn { frames + 1 } else { frames };
    for frame in 1..last as u64 {
        ctx.one_shot(|cmd| {
            renderer.record(ctx, cmd, frame, params, vk::ImageLayout::GENERAL);
        })?;
    }
    // Without OIDN (e.g. its device is out of memory) the last frame is traced
    // with the à-trous filter instead, as the interactive viewer does.
    let mut last_params = *params;
    let mut denoised = false;
    if oidn {
        match renderer.denoise_oidn(ctx, vk::ImageLayout::TRANSFER_SRC_OPTIMAL) {
            Ok(()) => denoised = true,
            Err(e) => {
                tracing::warn!("OIDN unavailable, using the à-trous filter: {e:#}");
                if let Some(pt) = &mut last_params.path_tracing {
                    pt.denoiser = Denoiser::Atrous;
                }
            }
        }
    }
    ctx.one_shot(|cmd| {
        if !denoised {
            renderer.record(
                ctx,
                cmd,
                frames as u64,
                &last_params,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
        }
        let output = renderer.output().unwrap();
        unsafe {
            ctx.device.cmd_copy_image_to_buffer(
                cmd,
                output.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback.handle,
                &[vk::BufferImageCopy::default()
                    .image_subresource(COLOR_LAYERS)
                    .image_extent(output.extent)],
            );
        }
    })?;
    let pixels = readback.mapped().unwrap().to_vec();
    readback.destroy(ctx);
    Ok(pixels)
}
