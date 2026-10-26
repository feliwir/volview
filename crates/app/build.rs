//! Lets the binary find the OpenImageDenoise libraries it links against
//! (OIDN_DIR, see .cargo/config.toml) without LD_LIBRARY_PATH.

fn main() {
    println!("cargo:rerun-if-env-changed=OIDN_DIR");
    // Windows has no rpath; DLLs are found through PATH instead (see
    // segmentation::ensure_cuda_library_path).
    if std::env::var_os("CARGO_CFG_WINDOWS").is_some() {
        return;
    }
    if let Ok(dir) = std::env::var("OIDN_DIR") {
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{dir}/lib");
    }
}
