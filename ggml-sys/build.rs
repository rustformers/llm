use std::env;

fn main() {
    // By default, this crate will attempt to compile ggml with the features of your host system if
    // the host and target are the same. If they are not, it will turn off auto-feature-detection,
    // and you will need to manually specify target features through target-features.

    println!("cargo:rerun-if-changed=ggml");

    let ggml_src = ["ggml/ggml.c"];

    let mut builder = cc::Build::new();

    let build = builder.files(ggml_src.iter()).include("include");

    // This is a very basic heuristic for applying compile flags.
    // Feel free to update this to fit your operating system.
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let is_release = env::var("PROFILE").unwrap() == "release";
    let compiler = build.get_compiler();

    match target_arch.as_str() {
        "x86" | "x86_64" => {
            let features = x86::Features::get();

            if compiler.is_like_clang() || compiler.is_like_gnu() {
                build.flag("-pthread");

                if features.avx {
                    build.flag("-mavx");
                }
                if features.avx2 {
                    build.flag("-mavx2");
                }
                if features.fma {
                    build.flag("-mfma");
                }
                if features.f16c {
                    build.flag("-mf16c");
                }
                if features.sse3 {
                    build.flag("-msse3");
                }
            } else if compiler.is_like_msvc() {
                match (features.avx2, features.avx) {
                    (true, _) => {
                        build.flag("/arch:AVX2");
                    }
                    (_, true) => {
                        build.flag("/arch:AVX");
                    }
                    _ => {}
                }
            }
        }
        "aarch64" => {
            if compiler.is_like_clang() || compiler.is_like_gnu() {
                if std::env::var("HOST") == std::env::var("TARGET") {
                    build.flag("-mcpu=native");
                } else {
                    #[allow(clippy::single_match)]
                    match target_os.as_str() {
                        "macos" => {
                            build.flag("-mcpu=apple-m1");
                            build.flag("-mfpu=neon");
                        }
                        _ => {}
                    }
                }
                build.flag("-pthread");
            }
        }
        _ => {}
    }

    #[allow(clippy::single_match)]
    match target_os.as_str() {
        "macos" => {
            build.define("GGML_USE_ACCELERATE", None);
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }
        _ => {}
    }

    if is_release {
        build.define("NDEBUG", None);
    }
    build.warnings(false);
    build.compile("ggml");
}

fn get_supported_target_features() -> std::collections::HashSet<String> {
    env::var("CARGO_CFG_TARGET_FEATURE")
        .unwrap()
        .split(',')
        .map(ToString::to_string)
        .collect()
}

mod x86 {
    #[allow(clippy::struct_excessive_bools)]
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct Features {
        pub fma: bool,
        pub avx: bool,
        pub avx2: bool,
        pub f16c: bool,
        pub sse3: bool,
    }
    impl Features {
        pub fn get() -> Self {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if std::env::var("HOST") == std::env::var("TARGET") {
                return Self::get_host();
            }

            Self::get_target()
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        pub fn get_host() -> Self {
            Self {
                fma: std::is_x86_feature_detected!("fma"),
                avx: std::is_x86_feature_detected!("avx"),
                avx2: std::is_x86_feature_detected!("avx2"),
                f16c: std::is_x86_feature_detected!("f16c"),
                sse3: std::is_x86_feature_detected!("sse3"),
            }
        }

        pub fn get_target() -> Self {
            let features = crate::get_supported_target_features();
            Self {
                fma: features.contains("fma"),
                avx: features.contains("avx"),
                avx2: features.contains("avx2"),
                f16c: features.contains("f16c"),
                sse3: features.contains("sse3"),
            }
        }
    }
}
