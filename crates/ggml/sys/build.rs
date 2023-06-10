use std::env;

// By default, this crate will attempt to compile ggml with the features of your host system if
// the host and target are the same. If they are not, it will turn off auto-feature-detection,
// and you will need to manually specify target features through target-features.
fn main() {
    println!("cargo:rerun-if-changed=ggml");

    let mut builder = cc::Build::new();

    let build = builder.files(["llama-cpp/ggml.c"]).includes(["llama-cpp"]);

    // This is a very basic heuristic for applying compile flags.
    // Feel free to update this to fit your operating system.
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let is_release = env::var("PROFILE").unwrap() == "release";
    let compiler = build.get_compiler();

    #[cfg(feature = "cublas")]
    enable_cublas(build);

    #[cfg(feature = "clblast")]
    enable_clblast(build);

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

#[cfg(feature = "clblast")]
fn enable_clblast(build: &mut cc::Build) {
    println!("cargo:rustc-link-lib=clblast");
    println!("cargo:rustc-link-lib=OpenCL");
    println!("cargo:rustc-link-lib=openblas");

    build.file("ggml/src/ggml-opencl.c");
    build.flag("-DGGML_USE_CLBLAST");
}

#[cfg(feature = "cublas")]
fn enable_cublas(build: &mut cc::Build) {
    let out_dir = env::var("OUT_DIR").unwrap();
    let object_file = format!(r"{}\ggml\src\ggml-cuda.o", &out_dir);

    let path = std::path::Path::new(&object_file);
    let parent_dir = path.parent().unwrap();

    std::fs::create_dir_all(parent_dir).unwrap();

    if cfg!(windows) {
        let targets_include = concat!(env!("CUDA_PATH"), r"\include");
        let targets_lib = concat!(env!("CUDA_PATH"), r"\lib\x64");

        std::process::Command::new("nvcc")
            .arg("-ccbin")
            .arg(
                cc::Build::new()
                    .get_compiler()
                    .path()
                    .parent()
                    .unwrap()
                    .join("cl.exe"),
            )
            .arg("-I")
            .arg(targets_include)
            .arg("-o")
            .arg(&object_file)
            .arg("-x")
            .arg("cu")
            .arg("-maxrregcount=0")
            .arg("--machine")
            .arg("64")
            .arg("--compile")
            .arg("-cudart")
            .arg("static")
            .arg("-D_WINDOWS")
            .arg("-DNDEBUG")
            .arg("-DGGML_USE_CUBLAS")
            .arg("-D_CRT_SECURE_NO_WARNINGS")
            .arg("-D_MBCS")
            .arg("-DWIN32")
            .arg(r"-Iggml\include\ggml")
            .arg(r"ggml\src\ggml-cuda.cu")
            .status()
            .unwrap();

        println!("cargo:rustc-link-search=native={}", targets_lib);
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublasLt");

        build.object(object_file);
        build.flag("-DGGML_USE_CUBLAS");
        build.include(targets_include);
    } else {
        let targets_include = concat!(env!("CUDA_PATH"), "/targets/x86_64-linux/include");
        let targets_lib = concat!(env!("CUDA_PATH"), "/targets/x86_64-linux/lib");

        std::process::Command::new("nvcc")
            .arg("--forward-unknown-to-host-compiler")
            .arg("-O3")
            .arg("-std=c++11")
            .arg("-fPIC")
            .arg("-Iggml/include/ggml")
            .arg("-mtune=native")
            .arg("-pthread")
            .arg("-DGGML_USE_CUBLAS")
            .arg("-I/usr/local/cuda/include")
            .arg("-I/opt/cuda/include")
            .arg("-I")
            .arg(targets_include)
            .arg("-c")
            .arg("ggml/src/ggml-cuda.cu")
            .arg("-o")
            .arg(&object_file)
            .status()
            .unwrap();

        println!("cargo:rustc-link-search=native={}", targets_lib);
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=culibos");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublasLt");
        println!("cargo:rustc-link-lib=dylib=stdc++");

        build.object(object_file);
        build.flag("-DGGML_USE_CUBLAS");
        build.include("/usr/local/cuda/include");
        build.include("/opt/cuda/include");
        build.include(targets_include);
    }
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
