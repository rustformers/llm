use std::env;
use std::path::{Path, PathBuf};

// By default, this crate will attempt to compile ggml with the features of your host system if
// the host and target are the same. If they are not, it will turn off auto-feature-detection,
// and you will need to manually specify target features through target-features.
fn main() {
    verify_state();

    println!("cargo:rerun-if-changed=llama-cpp");

    let mut builder = cc::Build::new();

    let build = builder
        .files(["llama-cpp/ggml.c", "llama-cpp/k_quants.c"])
        .define("GGML_USE_K_QUANTS", None)
        .includes(["llama-cpp"]);

    // This is a very basic heuristic for applying compile flags.
    // Feel free to update this to fit your operating system.
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let is_release = env::var("PROFILE").unwrap() == "release";
    let compiler = build.get_compiler();

    // Enable accelerators
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is not defined"));
    if cfg_cublas() && !cfg!(target_os = "macos") {
        enable_cublas(build, &out_dir);
    } else if cfg_clblast() {
        enable_clblast(build);
    } else if cfg!(target_os = "macos") {
        if cfg_metal() {
            enable_metal(build, &out_dir);
        } else {
            println!("cargo:rustc-link-lib=framework=Accelerate");

            build.define("GGML_USE_ACCELERATE", None);
        }
    }

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

    if is_release {
        build.define("NDEBUG", None);
    }

    build.warnings(false);

    if let Err(error) = build.try_compile("ggml") {
        eprintln!("{} {error}", get_error_message());

        std::process::exit(1);
    }
}

/// Verify the state of the repo to catch common newbie mistakes.
fn verify_state() {
    assert!(
        Path::new("llama-cpp/ggml.c").exists(),
        "Could not find llama-cpp/ggml.c. Try running `git submodule update --init`"
    );
}

fn cfg_cublas() -> bool {
    !cfg!(target_os = "macos") && cfg!(feature = "cublas")
}

fn cfg_clblast() -> bool {
    !cfg!(target_os = "macos") && cfg!(feature = "clblast")
}

fn cfg_metal() -> bool {
    cfg!(feature = "metal")
}

fn get_error_message() -> String {
    if cfg_cublas() {
        "Please make sure nvcc is executable and the paths are defined using CUDA_PATH, CUDA_INCLUDE_PATH and/or CUDA_LIB_PATH"
    }else if cfg_clblast() {
        "Please make sure the paths are defined using CLBLAST_PATH, CLBLAST_INCLUDE_PATH, CLBLAST_LIB_PATH, OPENCL_PATH, OPENCL_INCLUDE_PATH, and/or OPENCL_LIB_PATH"
    } else {
        "Please read the llm documentation"
    }.to_string()
}

fn include_path(prefix: &str) -> String {
    if let Ok(path) = env::var(format!("{prefix}_PATH")) {
        PathBuf::from(path).join("include")
    } else if let Ok(include_path) = env::var(format!("{prefix}_INCLUDE_PATH")) {
        PathBuf::from(include_path)
    } else {
        PathBuf::from("/usr/include")
    }
    .to_str()
    .unwrap_or_else(|| panic!("Could not build {prefix} include path"))
    .to_string()
}

fn lib_path(prefix: &str) -> String {
    if let Ok(path) = env::var(format!("{prefix}_PATH")) {
        PathBuf::from(path).join("lib")
    } else if let Ok(lib_path) = env::var(format!("{prefix}_LIB_PATH")) {
        PathBuf::from(lib_path)
    } else {
        PathBuf::from("/usr/lib")
    }
    .to_str()
    .unwrap_or_else(|| panic!("Could not build {prefix} lib path"))
    .to_string()
}

fn enable_clblast(build: &mut cc::Build) {
    println!("cargo:rustc-link-lib=clblast");
    println!("cargo:rustc-link-lib=OpenCL");

    if cfg!(target_os = "linux") {
        //enable dynamic linking against stdc++
        println!(r"cargo:rustc-link-lib=dylib=stdc++");
    }

    build.file("llama-cpp/ggml-opencl.cpp");
    build.flag("-DGGML_USE_CLBLAST");

    let clblast_include_path = include_path("CLBLAST");
    let opencl_include_path = include_path("OPENCL");
    let clblast_lib_path = lib_path("CLBLAST");
    let opencl_lib_path = lib_path("OPENCL");

    if cfg!(windows) {
        build.flag("/MT");
    }

    println!(r"cargo:rustc-link-search=native={clblast_lib_path}");
    println!(r"cargo:rustc-link-search=native={opencl_lib_path}");

    build.flag(&format!(r"-I{clblast_include_path}"));
    build.flag(&format!(r"-I{opencl_include_path}"));
}

fn enable_metal(build: &mut cc::Build, out_dir: &Path) {
    const GGML_METAL_METAL_PATH: &str = "llama-cpp/ggml-metal.metal";
    const GGML_METAL_PATH: &str = "llama-cpp/ggml-metal.m";

    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalKit");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    println!("cargo:rerun-if-changed={GGML_METAL_METAL_PATH}");
    println!("cargo:rerun-if-changed={GGML_METAL_PATH}");

    // HACK: patch ggml-metal.m so that it includes ggml-metal.metal, so that
    // a runtime dependency is not necessary
    let ggml_metal_path = {
        let ggml_metal_metal = std::fs::read_to_string(GGML_METAL_METAL_PATH)
            .expect("Could not read ggml-metal.metal")
            .replace('\\', "\\\\")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\"', "\\\"");

        let ggml_metal =
            std::fs::read_to_string(GGML_METAL_PATH).expect("Could not read ggml-metal.m");

        let needle = r#"NSString * src  = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];"#;
        if !ggml_metal.contains(needle) {
            panic!("ggml-metal.m does not contain the needle to be replaced; the patching logic needs to be reinvestigated. Contact a `llm` developer!");
        }

        // Replace the runtime read of the file with a compile-time string
        let ggml_metal = ggml_metal.replace(
            needle,
            &format!(r#"NSString * src  = @"{ggml_metal_metal}";"#),
        );

        // Replace the judicious use of `fprintf` with the already-existing `metal_printf`,
        // backing up the definition of `metal_printf` first
        let ggml_metal = ggml_metal
            .replace(
                r#"#define metal_printf(...) fprintf(stderr, __VA_ARGS__)"#,
                "METAL_PRINTF_DEFINITION",
            )
            .replace("fprintf(stderr,", "metal_printf(")
            .replace(
                "METAL_PRINTF_DEFINITION",
                r#"#define metal_printf(...) fprintf(stderr, __VA_ARGS__)"#,
            );

        let patched_ggml_metal_path = out_dir.join("ggml-metal.m");
        std::fs::write(&patched_ggml_metal_path, ggml_metal)
            .expect("Could not write temporary patched ggml-metal.m");

        patched_ggml_metal_path
    };

    build.file(ggml_metal_path);
    build.flag("-DGGML_USE_METAL");

    #[cfg(not(debug_assertions))]
    build.flag("-DGGML_METAL_NDEBUG");
}

fn cuda_include_path() -> String {
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let cuda_path = PathBuf::from(cuda_path);

        if cfg!(windows) {
            cuda_path
        } else {
            cuda_path.join("targets").join("x86_64-linux")
        }
        .join("include")
    } else if let Ok(cuda_include_path) = env::var("CUDA_INCLUDE_PATH") {
        PathBuf::from(cuda_include_path)
    } else {
        PathBuf::from("/usr/include")
    }
    .to_str()
    .expect("Could not build CUDA include path")
    .to_string()
}

fn cuda_lib_path() -> String {
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let cuda_path = PathBuf::from(cuda_path);

        if cfg!(windows) {
            cuda_path.join("lib").join("x64")
        } else {
            cuda_path.join("targets").join("lib")
        }
    } else if let Ok(cuda_lib_path) = env::var("CUDA_LIB_PATH") {
        PathBuf::from(cuda_lib_path)
    } else {
        PathBuf::from("/usr/lib/x86_64-linux-gnu")
    }
    .to_str()
    .expect("Could not build CUDA lib path")
    .to_string()
}

fn enable_cublas(build: &mut cc::Build, out_dir: &Path) {
    let object_file = out_dir
        .join("llama-cpp")
        .join("ggml-cuda.o")
        .to_str()
        .expect("Could not build ggml-cuda.o filename")
        .to_string();

    let path = std::path::Path::new(&object_file);
    let parent_dir = path.parent().unwrap();

    std::fs::create_dir_all(parent_dir).unwrap();

    let include_path = cuda_include_path();
    let lib_path = cuda_lib_path();

    if cfg!(windows) {
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
            .arg(&include_path)
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
            .arg("--generate-code=arch=compute_52,code=[compute_52,sm_52]")
            .arg("--generate-code=arch=compute_61,code=[compute_61,sm_61]")
            .arg("-D_WINDOWS")
            .arg("-DNDEBUG")
            .arg("-DGGML_USE_CUBLAS")
            .arg("-D_CRT_SECURE_NO_WARNINGS")
            .arg("-D_MBCS")
            .arg("-DWIN32")
            .arg(r"-Illama-cpp\include\ggml")
            .arg(r"-Illama-cpp\include\ggml")
            .arg(r"llama-cpp\ggml-cuda.cu")
            .status()
            .unwrap_or_else(|_| panic!("{}", get_error_message()));

        println!("cargo:rustc-link-search=native={}", lib_path);
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublasLt");

        build.object(object_file);
        build.flag("-DGGML_USE_CUBLAS");
        build.include(&include_path);
    } else {
        std::process::Command::new("nvcc")
            .arg("--forward-unknown-to-host-compiler")
            .arg("-O3")
            .arg("-std=c++11")
            .arg("-fPIC")
            .arg("-Illama-cpp/include/ggml")
            .arg("-mtune=native")
            .arg("-pthread")
            .arg("--generate-code=arch=compute_52,code=[compute_52,sm_52]")
            .arg("--generate-code=arch=compute_61,code=[compute_61,sm_61]")
            .arg("-DGGML_USE_CUBLAS")
            .arg("-I/usr/local/cuda/include")
            .arg("-I/opt/cuda/include")
            .arg("-I")
            .arg(&include_path)
            .arg("-c")
            .arg("llama-cpp/ggml-cuda.cu")
            .arg("-o")
            .arg(&object_file)
            .status()
            .unwrap_or_else(|_| panic!("{}", get_error_message()));

        println!("cargo:rustc-link-search=native={}", lib_path);
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
        build.include(include_path);
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
