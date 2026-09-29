use std::ffi::{CStr, c_void};
use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result, anyhow};
use ash::{ext, khr, vk};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use raw_window_handle::RawDisplayHandle;

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// Vulkan instance, device and allocator shared by all rendering code.
pub struct Context {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    debug: Option<(ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    pub surface_fn: Option<khr::surface::Instance>,
    pub physical_device: vk::PhysicalDevice,
    pub device_name: String,
    pub device: ash::Device,
    pub queue_family: u32,
    pub queue: vk::Queue,
    /// Memory can be exported as POSIX file descriptors (VK_KHR_external_memory_fd),
    /// e.g. to share images with OIDN on the same GPU without copies.
    pub external_memory_fd: Option<khr::external_memory_fd::Device>,
    /// The GPU's UUID, to open other APIs (OIDN, CUDA) on the same device.
    pub device_uuid: [u8; 16],
    /// Shared with other Vulkan users (e.g. the UI renderer); every clone must be
    /// dropped before the context.
    pub allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,
    command_pool: vk::CommandPool,
}

impl Context {
    /// Creates a context. With `display`, the surface and swapchain extensions
    /// are enabled for presenting to a window; without, it runs headless.
    pub fn new(display: Option<RawDisplayHandle>) -> Result<Self> {
        let entry = unsafe { ash::Entry::load() }.context("loading the Vulkan library")?;

        let want_validation =
            cfg!(debug_assertions) || std::env::var_os("VOLVIEW_VALIDATION").is_some();
        let has_validation = unsafe { entry.enumerate_instance_layer_properties()? }
            .iter()
            .any(|l| l.layer_name_as_c_str() == Ok(VALIDATION_LAYER));
        let validation = want_validation && has_validation;
        if want_validation && !has_validation {
            tracing::warn!("Vulkan validation layer not installed; running without it");
        }

        let mut extensions: Vec<*const i8> = Vec::new();
        if let Some(display) = display {
            extensions.extend_from_slice(ash_window::enumerate_required_extensions(display)?);
        }
        if validation {
            extensions.push(ext::debug_utils::NAME.as_ptr());
        }
        let layers = if validation {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            vec![]
        };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"volview")
            .api_version(vk::API_VERSION_1_3);
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app_info)
                    .enabled_layer_names(&layers)
                    .enabled_extension_names(&extensions),
                None,
            )
        }
        .context("creating Vulkan instance")?;

        let debug = if validation {
            let debug_fn = ext::debug_utils::Instance::new(&entry, &instance);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback));
            let messenger = unsafe { debug_fn.create_debug_utils_messenger(&info, None)? };
            Some((debug_fn, messenger))
        } else {
            None
        };
        let surface_fn = display.map(|_| khr::surface::Instance::new(&entry, &instance));

        let (physical_device, queue_family) = pick_device(&instance)?;
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_name = props
            .device_name_as_c_str()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        tracing::info!(device = %device_name, "using GPU");

        let mut vk12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .scalar_block_layout(true);
        let mut vk13 = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .dynamic_rendering(true);
        let mut device_extensions = if display.is_some() {
            vec![khr::swapchain::NAME.as_ptr()]
        } else {
            vec![]
        };
        let available = unsafe { instance.enumerate_device_extension_properties(physical_device)? };
        let has_external_fd = available.iter().any(|e| {
            e.extension_name_as_c_str() == Ok(khr::external_memory_fd::NAME)
        });
        if has_external_fd {
            device_extensions.push(khr::external_memory_fd::NAME.as_ptr());
        }
        let mut id_props = vk::PhysicalDeviceIDProperties::default();
        unsafe {
            instance.get_physical_device_properties2(
                physical_device,
                &mut vk::PhysicalDeviceProperties2::default().push_next(&mut id_props),
            )
        };
        let device_uuid = id_props.device_uuid;
        // r16f/rg16f storage images for the lighting volumes.
        let features =
            vk::PhysicalDeviceFeatures::default().shader_storage_image_extended_formats(true);
        let priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let device = unsafe {
            instance.create_device(
                physical_device,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_info)
                    .enabled_extension_names(&device_extensions)
                    .enabled_features(&features)
                    .push_next(&mut vk12)
                    .push_next(&mut vk13),
                None,
            )
        }
        .context("creating Vulkan device")?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let external_memory_fd =
            has_external_fd.then(|| khr::external_memory_fd::Device::new(&instance, &device));

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;

        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };

        Ok(Context {
            entry,
            instance,
            debug,
            surface_fn,
            physical_device,
            device_name,
            device,
            queue_family,
            queue,
            external_memory_fd,
            device_uuid,
            allocator: ManuallyDrop::new(Arc::new(Mutex::new(allocator))),
            command_pool,
        })
    }

    /// Resettable command pool on the context's queue family (single-threaded use).
    pub fn command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }

    pub fn allocate_command_buffers(&self, count: u32) -> Result<Vec<vk::CommandBuffer>> {
        Ok(unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(count),
            )?
        })
    }

    /// Records commands with `f`, submits them and waits for completion.
    pub fn one_shot<R>(&self, f: impl FnOnce(vk::CommandBuffer) -> R) -> Result<R> {
        let cmd = self.allocate_command_buffers(1)?[0];
        let device = &self.device;
        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            let result = f(cmd);
            device.end_command_buffer(cmd)?;
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cmd_info = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            device.queue_submit2(
                self.queue,
                &[vk::SubmitInfo2::default().command_buffer_infos(&cmd_info)],
                fence,
            )?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            device.destroy_fence(fence, None);
            device.free_command_buffers(self.command_pool, &[cmd]);
            Ok(result)
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.command_pool, None);
            debug_assert_eq!(
                Arc::strong_count(&self.allocator),
                1,
                "allocator still shared"
            );
            ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            if let Some((debug_fn, messenger)) = self.debug.take() {
                debug_fn.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

/// Prefers a discrete GPU with a queue family supporting graphics + compute.
/// On the common desktop GPUs that family can also present.
fn pick_device(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, u32)> {
    let devices = unsafe { instance.enumerate_physical_devices()? };
    let mut best = None;
    for pd in devices {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        if props.api_version < vk::API_VERSION_1_3 {
            continue;
        }
        let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
        let Some(family) = families.iter().position(|f| {
            f.queue_flags
                .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
        }) else {
            continue;
        };
        let score = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 3,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
            _ => 0,
        };
        if best.is_none_or(|(s, _, _)| score > s) {
            best = Some((score, pd, family as u32));
        }
    }
    best.map(|(_, pd, f)| (pd, f))
        .ok_or_else(|| anyhow!("no Vulkan 1.3 GPU with a graphics+compute queue found"))
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe { (*data).message_as_c_str() }
        .map(|m| m.to_string_lossy().into_owned())
        .unwrap_or_default();
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        tracing::error!(target: "vulkan", "{message}");
    } else {
        tracing::warn!(target: "vulkan", "{message}");
    }
    vk::FALSE
}
