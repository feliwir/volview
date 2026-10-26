//! Lets the binary find the OpenImageDenoise libraries it links against
//! (OIDN_DIR, see .cargo/config.toml) without LD_LIBRARY_PATH or PATH.

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-env-changed=OIDN_DIR");
    let Ok(dir) = std::env::var("OIDN_DIR") else {
        return;
    };
    if std::env::var_os("CARGO_CFG_WINDOWS").is_some() {
        copy_dlls(&Path::new(&dir).join("bin"));
    } else {
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{dir}/lib");
    }
}

/// Windows has no rpath, but searches the executable's directory first: copies
/// the DLLs there. OIDN loads its device modules (CPU, CUDA, ...) from its own
/// directory, so all of them go along.
fn copy_dlls(bin: &Path) {
    println!("cargo:rerun-if-changed={}", bin.display());
    // OUT_DIR is <target>/<profile>/build/<package>-<hash>/out.
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let Some(profile_dir) = Path::new(&out_dir).ancestors().nth(3) else {
        return;
    };
    let Ok(entries) = std::fs::read_dir(bin) else {
        println!("cargo:warning=no OIDN DLLs in {}", bin.display());
        return;
    };
    for entry in entries.flatten() {
        let src = entry.path();
        if src.extension().is_none_or(|e| !e.eq_ignore_ascii_case("dll")) {
            continue;
        }
        let dst = profile_dir.join(entry.file_name());
        let same_size = |a: &Path, b: &Path| match (a.metadata(), b.metadata()) {
            (Ok(a), Ok(b)) => a.len() == b.len(),
            _ => false,
        };
        if same_size(&src, &dst) {
            continue;
        }
        if let Err(e) = std::fs::copy(&src, &dst) {
            println!("cargo:warning=copying {}: {e}", src.display());
        }
    }
}
