use std::collections::HashSet;
use std::env;
use std::path::PathBuf;

fn main() {
    let ggml_src = ["ggml/ggml.c"];

    let mut builder = cc::Build::new();

    let build = builder.files(ggml_src.iter()).include("include");

    // This is a very basic heuristic for applying compile flags.
    // Feel free to update this to fit your operating system.
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    let is_release = std::env::var("PROFILE").unwrap() == "release";

    let supported_features: HashSet<_> = std::env::var("CARGO_CFG_TARGET_FEATURE")
        .unwrap()
        .split(',')
        .map(|s| s.to_string())
        .collect();

    match target_arch.as_str() {
        "x86" | "x86_64" => {
            let supports_fma = supported_features.contains("fma");
            let supports_avx = supported_features.contains("avx");
            let supports_avx2 = supported_features.contains("avx2");
            let supports_f16c = supported_features.contains("f16c");
            let supports_sse3 = supported_features.contains("sse3");

            match target_os.as_str() {
                "freebsd" | "haiku" | "ios" | "macos" | "linux" => {
                    build.flag("-pthread");

                    if supports_avx {
                        build.flag("-mavx");
                    }
                    if supports_avx2 {
                        build.flag("-mavx2");
                    }
                    if supports_fma {
                        build.flag("-mfma");
                    }
                    if supports_f16c {
                        build.flag("-mf16c");
                    }
                    if supports_sse3 {
                        build.flag("-msse3");
                    }
                }
                "windows" => match (supports_avx2, supports_avx) {
                    (true, _) => {
                        build.flag("/arch:AVX2");
                    }
                    (_, true) => {
                        build.flag("/arch:AVX");
                    }
                    _ => {}
                },
                _ => {}
            }
        }
        _ => {}
    }
    if is_release {
        build.define("NDEBUG", None);
    }
    build.compile("ggml");

    println!("cargo:rerun-if-changed=ggml/ggml.h");

    let bindings = bindgen::Builder::default()
        .header("ggml/ggml.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .allowlist_var("ggml_.*")
        .allowlist_file("ggml_.*")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
