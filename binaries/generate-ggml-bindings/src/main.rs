//! Helper tool to generate the bindings for the ggml crate.
//!
//! Assumed to be run from the root of the workspace.

use std::{
    fs,
    path::{Path, PathBuf},
};

fn main() {
    let sys_path = PathBuf::from("crates").join("ggml").join("sys");
    let ggml_path = sys_path.join("llama-cpp");
    let src_path = sys_path.join("src");

    generate_main(&ggml_path, &src_path);
    generate_cuda(&ggml_path, &src_path);
    generate_opencl(&ggml_path, &src_path);
    generate_metal(&ggml_path, &src_path);
    generate_llama(&ggml_path, &src_path);

    println!("Successfully updated bindings");
}

fn generate_main(ggml_path: &Path, src_path: &Path) {
    let bindings = bindgen::Builder::default()
        .header(ggml_path.join("ggml.h").to_str().unwrap().to_string())
        .allowlist_file(r".*ggml.h")
        .header(ggml_path.join("k_quants.h").to_string_lossy())
        .allowlist_file(r".*k_quants.h")
        // Suppress some warnings
        .raw_line("#![allow(non_upper_case_globals)]")
        .raw_line("#![allow(non_camel_case_types)]")
        .raw_line("#![allow(non_snake_case)]")
        .raw_line("#![allow(unused)]")
        .raw_line("pub mod llama;")
        .raw_line("")
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
}

fn generate_cuda(ggml_path: &Path, src_path: &Path) {
    generate_extra("cuda", ggml_path, src_path, |b| {
        b.header(ggml_path.join("ggml-cuda.h").to_string_lossy())
            .allowlist_file(r".*ggml-cuda\.h")
            .raw_line("use super::ggml_compute_params;")
            .raw_line("use super::ggml_tensor;")
    })
}

fn generate_opencl(ggml_path: &Path, src_path: &Path) {
    generate_extra("opencl", ggml_path, src_path, |b| {
        b.header(ggml_path.join("ggml-opencl.h").to_string_lossy())
            .allowlist_file(r".*ggml-opencl\.h")
            .raw_line("use super::ggml_tensor;")
    })
}

fn generate_metal(ggml_path: &Path, src_path: &Path) {
    generate_extra("metal", ggml_path, src_path, |b| {
        b.header(ggml_path.join("ggml-metal.h").to_string_lossy())
            .allowlist_file(r".*ggml-metal\.h")
    });
}

fn generate_llama(ggml_path: &Path, src_path: &Path) {
    // We do not use `llama.cpp` for its implementation at all;
    // we only use it for its header file and its associated constants.
    generate_extra("llama", ggml_path, src_path, |b| {
        b.header(ggml_path.join("llama.h").to_string_lossy())
            .allowlist_type("llama_ftype")
            .allowlist_var("LLAMA_.*")
            .prepend_enum_name(false)
            .ignore_functions()
    });
}

fn generate_extra(
    name: &str,
    ggml_path: &Path,
    src_path: &Path,
    mut callback: impl FnMut(bindgen::Builder) -> bindgen::Builder,
) {
    let builder = callback(
        bindgen::Builder::default()
            .allowlist_recursively(false)
            .clang_arg("-I")
            .clang_arg(ggml_path.to_string_lossy()),
    );

    builder
        .generate()
        .unwrap_or_else(|_| panic!("Unable to generate {name} bindings"))
        .write_to_file(src_path.join(format!("{name}.rs")))
        .unwrap_or_else(|_| panic!("Couldn't write {name} bindings"));
}
