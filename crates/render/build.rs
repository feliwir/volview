//! Compiles `shaders/*.comp` (GLSL) to SPIR-V with glslangValidator.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let shader_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../shaders");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let glslang = std::env::var("GLSLANG_VALIDATOR").unwrap_or_else(|_| "glslangValidator".into());
    println!("cargo:rerun-if-env-changed=GLSLANG_VALIDATOR");
    println!("cargo:rerun-if-changed={}", shader_dir.display());

    for entry in std::fs::read_dir(&shader_dir).expect("shaders/ directory") {
        let path = entry.unwrap().path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default();
        // Headers (*.glsl) are only compiled as part of the stages that include them.
        if ext == "glsl" {
            println!("cargo:rerun-if-changed={}", path.display());
            continue;
        }
        if !matches!(ext, "comp" | "vert" | "frag") {
            continue;
        }
        println!("cargo:rerun-if-changed={}", path.display());
        let out = out_dir.join(format!(
            "{}.spv",
            path.file_name().unwrap().to_str().unwrap()
        ));
        let debug = std::env::var("PROFILE").as_deref() == Ok("debug");
        let output = Command::new(&glslang)
            .args(["-V", "--target-env", "vulkan1.3"])
            .args(if debug { &["-g"][..] } else { &[][..] })
            .arg("-I".to_owned() + shader_dir.to_str().unwrap())
            .arg("-o")
            .arg(&out)
            .arg(&path)
            .output()
            .unwrap_or_else(|e| panic!("failed to run {glslang}: {e}"));
        if !output.status.success() {
            panic!(
                "shader compilation failed for {}:\n{}{}",
                path.display(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}
