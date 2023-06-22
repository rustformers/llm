//! Helper tool to generate the bindings for the ggml crate.
//!
//! Assumed to be run from the root of the workspace.

use std::fs;
use std::path::PathBuf;

fn main() {
    let sys_path = PathBuf::from("crates").join("ggml").join("sys");
    let ggml_path = sys_path.join("llama-cpp");
    let include_path = ggml_path.to_str().unwrap().to_string();
    let src_path = sys_path.join("src");

    let bindings = bindgen::Builder::default()
        .header(ggml_path.join("k_quants.h").to_str().unwrap().to_string())
        .allowlist_file(r".*k_quants.h")
        // Suppress some warnings
        .raw_line("#![allow(non_upper_case_globals)]")
        .raw_line("#![allow(non_camel_case_types)]")
        .raw_line("#![allow(non_snake_case)]")
        .raw_line("#![allow(unused)]")
        .raw_line(r#"#[cfg(feature = "cublas")]"#)
        .raw_line("pub mod cuda;")
        .raw_line(r#"#[cfg(feature = "metal")]"#)
        .raw_line("pub mod metal;")
        .raw_line(r#"#[cfg(feature = "clblast")]"#)
        .raw_line("pub mod opencl;")
        // Only generate code if it's from GGML
        .allowlist_file("crates/ggml/.*")
        .generate()
        .expect("Unable to generate bindings");

    bindgen::Builder::default()
        .header(ggml_path.join("ggml-cuda.h").to_str().unwrap().to_string())
        .allowlist_file(r".*ggml-cuda\.h")
        .allowlist_recursively(false)
        .clang_arg("-I")
        .clang_arg(&include_path)
        .raw_line("use super::ggml_compute_params;")
        .raw_line("use super::ggml_tensor;")
        .generate()
        .expect("Unable to generate cuda bindings")
        .write_to_file(src_path.join("cuda.rs"))
        .expect("Couldn't write cuda bindings");

    bindgen::Builder::default()
        .header(
            ggml_path
                .join("ggml-opencl.h")
                .to_str()
                .unwrap()
                .to_string(),
        )
        .allowlist_file(r".*ggml-opencl\.h")
        .allowlist_recursively(false)
        .clang_arg("-I")
        .clang_arg(&include_path)
        .raw_line("use super::ggml_tensor;")
        .generate()
        .expect("Unable to generate opencl bindings")
        .write_to_file(src_path.join("opencl.rs"))
        .expect("Couldn't write opencl bindings");

    bindgen::Builder::default()
        .header(ggml_path.join("ggml-metal.h").to_str().unwrap().to_string())
        .allowlist_file(r".*ggml-metal\.h")
        .allowlist_recursively(false)
        .clang_arg("-I")
        .clang_arg(&include_path)
        .generate()
        .expect("Unable to generate metal bindings")
        .write_to_file(src_path.join("metal.rs"))
        .expect("Couldn't write metal bindings");

    let mut generated_bindings = bindings.to_string();

    if cfg!(windows) {
        // windows generates all ::std::os::raw::c_* enum types as i32.
        // We need to replace some of them with c_uint as the rust bindings expect them to be unsigned.
        // Temporary hack until bindgen supports defining the enum types manually. See https://github.com/rust-lang/rust-bindgen/issues/1907
        for name in &[
            "type",
            "backend",
            "op",
            "linesearch",
            "opt_type",
            "task_type",
        ] {
            generated_bindings = generated_bindings.replace(
                &format!("ggml_{name} = ::std::os::raw::c_int;"),
                &format!("ggml_{name} = ::std::os::raw::c_uint;"),
            );
        }
    }

    fs::write(src_path.join("lib.rs"), generated_bindings).expect("Couldn't write bindings");

    println!("Successfully updated bindings");
}
