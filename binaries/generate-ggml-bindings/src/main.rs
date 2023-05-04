//! Helper tool to generate the bindings for the ggml crate.
//!
//! Assumed to be run from the root of the workspace.

use std::path::PathBuf;

fn main() {
    const HEADER_PATH: &str = "crates/ggml/sys/ggml/include/ggml/ggml.h";

    let bindings = bindgen::Builder::default()
        .header(HEADER_PATH)
        // Suppress some warnings
        .raw_line("#![allow(non_upper_case_globals)]")
        .raw_line("#![allow(non_camel_case_types)]")
        .raw_line("#![allow(non_snake_case)]")
        .raw_line("#![allow(unused)]")
        // Do not generate code for ggml's includes (stdlib)
        .allowlist_file(HEADER_PATH)
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from("crates")
        .join("ggml")
        .join("sys")
        .join("src")
        .join("lib.rs");
    bindings
        .write_to_file(out_path)
        .expect("Couldn't write bindings");

    println!("Successfully updated bindings");
}
