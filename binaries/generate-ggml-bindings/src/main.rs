//! Helper tool to generate the bindings for the ggml crate.
//!
//! Assumed to be run from the root of the workspace.

use std::io::Write;
use std::path::PathBuf;

fn main() {
    let bindings = bindgen::Builder::default()
        .header("crates/ggml/sys/ggml/include/ggml/ggml.h")
        // Suppress some warnings
        .raw_line("#![allow(non_upper_case_globals)]")
        .raw_line("#![allow(non_camel_case_types)]")
        .raw_line("#![allow(non_snake_case)]")
        .raw_line("#![allow(unused)]")
        // Only generate code if it's from GGML
        .allowlist_file("crates/ggml/.*")
        .generate()
        .expect("Unable to generate bindings");

    let cuda_bindings = bindgen::Builder::default()
        .header("crates/ggml/sys/ggml/src/ggml-cuda.h")
        .allowlist_file("crates/ggml/sys/ggml/src/ggml-cuda.h")
        .allowlist_recursively(false)
        .clang_arg("-I")
        .clang_arg("crates/ggml/sys/ggml/include/ggml")
        .generate()
        .expect("Unable to generate cuda bindings");

    let opencl_bindings = bindgen::Builder::default()
        .header("crates/ggml/sys/ggml/src/ggml-opencl.h")
        .allowlist_file("crates/ggml/sys/ggml/src/ggml-opencl.h")
        .allowlist_recursively(false)
        .clang_arg("-I")
        .clang_arg("crates/ggml/sys/ggml/include/ggml")
        .generate()
        .expect("Unable to generate opencl bindings");

    let out_dir = PathBuf::from("crates").join("ggml").join("sys").join("src");

    cuda_bindings
        .write_to_file(out_dir.join("lib_cuda.rs"))
        .expect("Couldn't write cuda bindings");

    opencl_bindings
        .write_to_file(out_dir.join("lib_opencl.rs"))
        .expect("Couldn't write opencl bindings");

    bindings
        .write_to_file(out_dir.join("lib.rs"))
        .expect("Couldn't write bindings");

    //Reopen the file and add the missing imports
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(out_dir.join("lib.rs"))
        .expect("Couldn't open bindings file");

    writeln!(file, "#[cfg(feature = \"cublas\")]").expect("Couldn't write to bindings file");
    writeln!(file, "include!(\"lib_cuda.rs\");").expect("Couldn't write to bindings file");

    writeln!(file, "#[cfg(feature = \"clblast\")]").expect("Couldn't write to bindings file");
    writeln!(file, "include!(\"lib_opencl.rs\");").expect("Couldn't write to bindings file");

    println!("Successfully updated bindings");
}
