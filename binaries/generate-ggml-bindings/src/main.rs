//! Helper tool to generate the bindings for the ggml crate.
//!
//! Assumed to be run from the root of the workspace.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let bindings = bindgen::Builder::default()
        .header("crates/ggml/sys/llama-cpp/ggml.h")
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

    let out_path = PathBuf::from("crates")
        .join("ggml")
        .join("sys")
        .join("src")
        .join("lib.rs");

    let mut generated_bindings = bindings.to_string();

    if cfg!(windows) {
        // windows generates all ::std::os::raw::c_* enum types as i32.
        // We need to replace some of them with c_uint as the rust bindings expect them to be unsigned.
        // Temporary hack until bindgen supports defining the enum types manually. See https://github.com/rust-lang/rust-bindgen/issues/1907
        generated_bindings = generated_bindings.replace(
            "ggml_type = ::std::os::raw::c_int;",
            "ggml_type = ::std::os::raw::c_uint;",
        );
        generated_bindings = generated_bindings.replace(
            "ggml_backend = ::std::os::raw::c_int;",
            "ggml_backend = ::std::os::raw::c_uint;",
        );
        generated_bindings = generated_bindings.replace(
            "ggml_op = ::std::os::raw::c_int;",
            "ggml_op = ::std::os::raw::c_uint;",
        );
    }

    fs::write(out_path, generated_bindings).expect("Couldn't write bindings");

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
