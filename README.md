# volview

Photorealistic rendering of CT data: loads a DICOM series, renders it with a
Vulkan ray marcher or path tracer (denoised with Intel Open Image Denoise) and
optionally labels the anatomy with TotalSegmentator models run through ONNX
Runtime.

```
cargo run --release -- <DICOM_DIR>
cargo run --release -- --help
```

## Setting up a Windows machine

Builds with the MSVC toolchain on Windows 10/11 (x64).

### Required

1. **Visual Studio 2022 Build Tools** (or Visual Studio) with the
   *Desktop development with C++* workload, for the MSVC linker and C/C++
   compiler used by some dependencies.
2. **Rust** via [rustup](https://rustup.rs), with the default
   `x86_64-pc-windows-msvc` toolchain (Rust 1.88 or newer).
3. **Vulkan SDK** from [LunarG](https://vulkan.lunarg.com/sdk/home#windows).
   The build compiles `shaders/*.comp` with `glslangValidator`, which the SDK
   installer puts on `PATH` (otherwise point `GLSLANG_VALIDATOR` at it).
   The SDK also brings the validation layer that debug builds enable; without
   it they run with a warning. A GPU driver with Vulkan 1.3 is required.
4. **Open Image Denoise 2.5.1**: download
   [`oidn-2.5.1.x64.windows.zip`](https://github.com/RenderKit/oidn/releases/tag/v2.5.1)
   and unpack it so that `data/oidn/lib/OpenImageDenoise.lib` and
   `data/oidn/bin/OpenImageDenoise.dll` exist. `.cargo/config.toml` points
   the `oidn` crate at this directory (its own download is broken in 2.5.1),
   and the build copies the DLLs next to `volview.exe`, because Windows has no
   rpath. In PowerShell, from the repository root:

   ```powershell
   curl.exe -L -o oidn.zip https://github.com/RenderKit/oidn/releases/download/v2.5.1/oidn-2.5.1.x64.windows.zip
   Expand-Archive oidn.zip data
   Rename-Item data\oidn-2.5.1.x64.windows oidn
   Remove-Item oidn.zip
   ```

   Without it, linking fails with `LNK1181: cannot open input file
   "OpenImageDenoise.lib"`.

Then `cargo build --release` and `cargo test --workspace` should pass.

### Optional

- **Segmentation models**: TotalSegmentator ONNX models in
  `data/models/totalseg` (or `$VOLVIEW_MODELS`), exported with
  `tools/totalseg/export_onnx.py`. Without them, pass `--no-segment`.
- **GPU segmentation**: ONNX Runtime itself is downloaded by the `ort` crate
  during the build. For CUDA/TensorRT, the CUDA 13, cuDNN and TensorRT DLLs are
  looked for under `data/cuda13` (or `$VOLVIEW_CUDA_DIR`) in
  `nvidia/cu13/bin`, `nvidia/cudnn/bin` and `tensorrt_libs`, the layout of
  NVIDIA's pip wheels; volview restarts itself with those folders on `PATH`.
  This path is not yet tested on Windows; `--cpu` segments on the CPU.
- **Label cache**: computed labels are cached under `%XDG_CACHE_HOME%` or
  `%HOME%\.cache`. Windows sets neither by default, so set one of them to
  keep labels between runs.
- **Light map**: an equirectangular `.hdr` at
  `data/hdri/studio_small_09_2k.hdr` (e.g. from
  [Poly Haven](https://polyhaven.com/a/studio_small_09)) for
  `--environment map`; otherwise the studio environment is used.

`data/` is ignored by git, so all of the above stays local.

## Sample data

Public CT series from [The Cancer Imaging Archive](https://www.cancerimagingarchive.net)
can be downloaded without an account through its API, as a zip of DICOM files:

```powershell
curl.exe -L -o series.zip "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID=<UID>"
Expand-Archive series.zip data\samples\<name>
```

Series that render well (all CC BY; credit TCIA and the collection when
publishing images):

| Collection | Content | Slices | SeriesInstanceUID |
|---|---|---|---|
| LIDC-IDRI | Chest, 3 mm | 100 | `1.3.6.1.4.1.14519.5.2.1.6279.6001.309707676674808510671394142910` |
| FDG-PET-CT-Lesions | Shoulders to thighs, 2.5 mm | 391 | `1.3.6.1.4.1.14519.5.2.1.34775519577537201970528940353322258449` |
| C4KC-KiTS | Contrast abdomen, 1 mm (try `--preset angio`) | 417 | `1.3.6.1.4.1.14519.5.2.1.6919.4624.237361109707836265327019544590` |
| CT Lymph Nodes | Contrast chest to pelvis, 1 mm (try `--preset "soft tissue"`) | 645 | `61.7.14370462047469955976541236852004977244` |

For example:

```powershell
cargo run --release -- data\samples\kits_abdomen --no-segment --preset angio
cargo run --release -- data\samples\kits_abdomen --no-segment --screenshot kits.png
```
