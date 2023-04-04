use std::env;

fn main() {
    let bindings = bindgen::Builder::default()
        .header("ggml/ggml.h")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = env::current_dir().unwrap().join("src");
    bindings
        .write_to_file(out_path.join("lib.rs"))
        .expect("Couldn't write bindings");

    println!("Successfully updated bindings in src/lib.rs");
}
