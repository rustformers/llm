# Acceleration Support

The `llm` ecosystem of crates, including `llm`, `llm-base` and `ggml` support various acceleration backends, selectable via `--features` flags. The availability of supported backends varies by platform, and these crates can only be built with a single active acceleration backend at a time. If CuBLAS and CLBlast are both specified, CuBLAS is prioritized and CLBlast is ignored.

| Platform/OS | `cublas`           | `clblast`          | `metal`            |
| ----------- | ------------------ | ------------------ | ------------------ |
| Windows     | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| Linux       | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| MacOS       | :x:                | :x:                | :heavy_check_mark: |

## Utilizing GPU Support

To activate GPU support (assuming that you have enabled one of the features above), set the `use_gpu` attribute of the `ModelParameters` to `true`.

- **CLI Users**: You can enable GPU support by adding the `--use-gpu` flag.

- **Backend Consideration**: For users leveraging the `cublas` or `clblast` backends, you can specify the number of layers you wish to offload to your GPU with the `gpu_layers` parameter in the `ModelParameters`. By default, all layers are offloaded.

  However, if your model size exceeds your GPU's VRAM, you can specify a limit, like `20`, to offload only the first 20 layers. For CLI users, this can be achieved using the `--gpu-layers` parameter.

**Example**: To run a `llama` model with CUDA acceleration and offload all its layers, your CLI command might resemble:

```bash
cargo run --release --features cublas -- infer -a llama -m [path/to/model.bin] --use-gpu -p "Help a llama is standing in my garden!"
```

💡 **Protip**: For those with ample VRAM using `cublas` or `clblast`, you can significantly reduce your prompt's feed time by increasing the batch size; for example, you can use `256` or `512` (default is `8`).

- Programmatic users of `llm` can adjust this by setting the `n_batch` parameter in the `InferenceSessionConfig` when initializing a session.

- CLI users can utilize the `--batch-size` parameter to achieve this.

## Supported Accelerated Models

While specific accelerators only support certain model architectures, some unmarked architectures may function, but their performance is not guaranteed—it hinges on the operations used by the model's architecture. The table below lists models with confirmed compatibility for each accelerator:

| Model/accelerator | `cublas` | `clblast` | `metal` |
| ----------------- | -------- | --------- | ------- |
| LLaMA             | ✅       | ✅        | ✅      |
| MPT               | ❌       | ❌        | ❌      |
| Falcon            | ❌       | ❌        | ❌      |
| GPT-NeoX          | ❌       | ❌        | ❌      |
| GPT-J             | ❌       | ❌        | ❌      |
| GPT-2             | ❌       | ❌        | ❌      |
| BLOOM             | ❌       | ❌        | ❌      |

## Pre-requisites for Building with Accelerated Support

To build with acceleration support, certain dependencies must be installed. These dependencies are contingent upon your chosen platform and the specific acceleration backend you're working with.

For developers aiming to distribute packages equipped with acceleration capabilities, our [CI/CD setup](../.github/workflows/rust.yml) serves as an exemplary foundation.

### Windows

#### CuBLAS

CUDA must be installed. You can download CUDA from the official [Nvidia site](https://developer.nvidia.com/cuda-downloads).

#### CLBlast

CLBlast can be installed via [vcpkg](https://vcpkg.io/en/getting-started.html) using the command `vcpkg install clblast`. After installation, the `OPENCL_PATH` and `CLBLAST_PATH` environment variables should be set to the `opencl_x64-windows` and `clblast_x64-windows` directories respectively.

Here's an example of the required commands:

```
git clone https://github.com/Microsoft/vcpkg.git
.\vcpkg\bootstrap-vcpkg.bat
.\vcpkg\vcpkg install clblast
set OPENCL_PATH=....\vcpkg\packages\opencl_x64-windows
set CLBLAST_PATH=....\vcpkg\packages\clblast_x64-windows
```

⚠️ When working with MSVC in a Windows environment, it is essential to set the `-Ctarget-feature=+crt-static` Rust flag. This flag is critical as it enables the static linking of the C runtime, which can be paramount for certain deployment scenarios or specific runtime environments.

To set this flag, you can modify the .cargo\config file in your project directory. Please add the following configuration snippet:

```
[target.x86_64-pc-windows-msvc]
rustflags = ["-Ctarget-feature=+crt-static"]
```

This will ensure the Rust flag is appropriately set for your compilation process.

For a comprehensive guide on the usage of Rust flags, including other possible ways to set them, please refer to this detailed [StackOverflow discussion](https://stackoverflow.com/questions/38040327/how-to-pass-rustc-flags-to-cargo). Make sure to choose an option that best fits your project requirements and development environment.

⚠️ For `llm` to function properly, it requires the `clblast.dll` and `OpenCL.dll` files. These files can be found within the `bin` subdirectory of their respective vcpkg packages. There are two options to ensure `llm` can access these files:

1. Amend your `PATH` environment variable to include the `bin` directories of each respective package.

2. Manually copy the `clblast.dll` and `OpenCL.dll` files into the `./target/release` or `./target/debug` directories. The destination directory will depend on the profile that was active during the compilation process.

Please choose the option that best suits your needs and environment configuration.

### Linux

#### CuBLAS

You need to have CUDA installed on your system. CUDA can be downloaded and installed from the official [Nvidia site](https://developer.nvidia.com/cuda-downloads). On Linux distributions that do not have `CUDA_PATH` set, the environment variables `CUDA_INCLUDE_PATH` and `CUDA_LIB_PATH` can be set to their corresponding paths.

#### CLBlast

CLBlast can be installed on Linux through various package managers. For example, using `apt` you can install it via `sudo apt install clblast`. After installation, make sure that the `OPENCL_PATH` and `CLBLAST_PATH` environment variables are correctly set. Additionally the environment variables `OPENCL_INCLUDE_PATH`/`OPENCL_LIB_PATH` & `CBLAST_INCLUDE_PATH`/`CLBLAST_LIB_PATH` can be used to specify the location of the files. All environment variables are supported by all listed operating systems.

### MacOS

#### Metal

Xcode and the associated command-line tools should be installed on your system, and you should be running a version of MacOS that supports Metal. For more detailed information, please consult the [official Metal documentation](https://developer.apple.com/metal/).

To enable Metal using the CLI, ensure it was built successfully using `--features=metal` and then pass the `--use-gpu` flag.

The current underlying implementation of Metal in GGML is still in flux and has some limitations:

- Evaluating a model with more than one token at a time is not currently supported in GGML's Metal implementation. An `llm` inference session will fall back to the CPU implementation (typically during the 'feed prompt' phase) but will automatically use the GPU once a single token is passed per evaluation (typically after prompt feeding).
- Not all model architectures will be equally stable when used with Metal due to ongoing work in the underlying implementation. Expect `llama` models to work fine though.
- With Metal, it is possible but not required to use `mmap`. As buffers do not need to be copied to VRAM on M1, `mmap` is the most efficient however.
- Debug messages may be logged by the underlying GGML Metal implementation. This will likely go away in the future for release builds of `llm`.
