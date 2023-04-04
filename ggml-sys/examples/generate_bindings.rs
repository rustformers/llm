use std::env;

fn main() {
    let bindings = bindgen::Builder::default()
        .header("ggml/ggml.h")
        // Suppress some warnings
        .raw_line("#![allow(non_upper_case_globals)]")
        .raw_line("#![allow(non_camel_case_types)]")
        .raw_line("#![allow(non_snake_case)]")
        .raw_line("#![allow(unused)]")
        // Do not generate code for ggml's includes (stdlib)
        .allowlist_file("ggml/ggml.h")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = env::current_dir().unwrap().join("src");
    bindings
        .write_to_file(out_path.join("lib.rs"))
        .expect("Couldn't write bindings");

    println!("Successfully updated bindings in src/lib.rs");
}
