use std::{env, path::PathBuf};

fn main() {
    // Parse arguments
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        println!("Usage: {} <path_to_ggml_crate>", args[0]);
        return;
    }

    let ggml_crate_path = &args[1];

    let header_path = format!("{ggml_crate_path}/ggml/ggml.h");

    let bindings = bindgen::Builder::default()
        .header(&header_path)
        // Suppress some warnings
        .raw_line("#![allow(non_upper_case_globals)]")
        .raw_line("#![allow(non_camel_case_types)]")
        .raw_line("#![allow(non_snake_case)]")
        .raw_line("#![allow(unused)]")
        // Do not generate code for ggml's includes (stdlib)
        .allowlist_file(&header_path)
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(ggml_crate_path).join("src").join("lib.rs");
    bindings
        .write_to_file(out_path)
        .expect("Couldn't write bindings");

    println!("Successfully updated bindings in src/lib.rs");
}
