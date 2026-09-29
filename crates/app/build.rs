//! Lets the binary find the OpenImageDenoise libraries it links against
//! (OIDN_DIR, see .cargo/config.toml) without LD_LIBRARY_PATH.

fn main() {
    println!("cargo:rerun-if-env-changed=OIDN_DIR");
    if let Ok(dir) = std::env::var("OIDN_DIR") {
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{dir}/lib");
    }
}
