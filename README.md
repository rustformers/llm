# LLaMA-rs

This project is a Rust port of
[llama.cpp](https://github.com/ggerganov/llama.cpp) 🦙🦀🚀

Just like its C++ counterpart, it is powered by the
[`ggml`](https://github.com/ggerganov/ggml) tensor library, which allows running
inference for Facebook's [LLaMA](https://github.com/facebookresearch/llama)
model on a CPU with good performance using full precision, f16 or 4-bit
quantized versions of the model.

[![Latest version](https://img.shields.io/crates/v/llama-rs.svg)](https://crates.io/crates/llama_rs)
![MIT/Apache2](https://shields.io/badge/license-MIT%2FApache--2.0-blue)
[![Discord](https://img.shields.io/discord/1085885067601137734)](https://discord.gg/YB9WaXYAWU)

![A llama riding a crab, AI-generated](./doc/resources/logo2.png)

> _Image by [@darthdeus](https://github.com/darthdeus/), using Stable Diffusion_

## Getting started

Make sure you have a Rust 1.65.0 or above and C toolchain[^1] set up.

`llm-base`, `gpt2`, and `llama` are Rust libraries, while `llm-cli` is a CLI
applications that wraps `gpt2` and `llama` and offer basic inference
capabilities.

The following instructions explain how to build CLI applications.

**NOTE**: For best results, make sure to build and run in release mode.
Debug builds are going to be very slow.

### Building using `cargo`

Run

```shell
cargo install --git https://github.com/rustformers/llama-rs llm-cli
```

to install `llm-cli` to your Cargo `bin` directory, which `rustup` is likely to
have added to your `PATH`.

The CLI application can then be run through `llm-cli`.

![Gif showcasing language generation using llama-rs](./doc/resources/llama_gif.gif)

### Building from repository

Clone the repository and then build it with

```shell
git clone --recurse-submodules git@github.com:rustformers/llama-rs.git
cargo build --release
```

The resulting binary will be at `target/release/llm-cli[.exe]`.

It can also be run directly through Cargo, using

```shell
cargo run --release --bin llm-cli -- <ARGS>
```

This is useful for development.

### Getting LLaMA weights

In order to run the inference code in `llama-rs`, a copy of the model's weights
are required.

#### From Hugging Face

Compatible weights - not necessarily the original LLaMA weights - can be found
on [Hugging Face by searching for GGML](https://huggingface.co/models?search=ggml).
At present, LLaMA-architecture models are supported.

#### LLaMA original weights

Currently, the only legal source to get the original weights is [this
repository](https://github.com/facebookresearch/llama/blob/main/README.md#llama).
Note that the choice of words also may or may not hint at the existence of other
kinds of sources.

After acquiring the weights, it is necessary to convert them into a format that
is compatible with ggml. To achieve this, follow the steps outlined below:

> **Warning**
>
> To run the Python scripts, a Python version of 3.9 or 3.10 is required. 3.11
> is unsupported at the time of writing.

```shell
# Convert the model to f16 ggml format
python3 scripts/convert-pth-to-ggml.py /path/to/your/models/7B/ 1

# Quantize the model to 4-bit ggml format
cargo run -p llama-cli quantize /path/to/your/models/7B/ggml-model-f16.bin /path/to/your/models/7B/ggml-model-q4_0.bin q4_0
```

> **Note**
>
> The [llama.cpp repository](https://github.com/ggerganov/llama.cpp) has
> additional information on how to obtain and run specific models.

### GPT2

OpenAI's [GPT-2](https://jalammar.github.io/illustrated-gpt2/) architecture is
also supported. The open-source family of
[Cerebras](https://www.cerebras.net/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models/)
models is built on this architecture.

_Support for other open source models is currently planned. For models where
weights can be legally distributed, this section will be updated with scripts to
make the install process as user-friendly as possible. Due to the model's legal
requirements, this is currently not possible with LLaMA itself and a more
lengthy setup is required._

### Running

For example, try the following prompt:

```shell
llama-cli infer -m <path>/ggml-model-q4_0.bin -p "Tell me how cool the Rust programming language is:"
```

Some additional things to try:

- Use `--help` to see a list of available options.
- If you have the [alpaca-lora](https://github.com/tloen/alpaca-lora) weights,
  try `repl` mode!

  ```shell
  llama-cli repl -m <path>/ggml-alpaca-7b-q4.bin -f examples/alpaca_prompt.txt
  ```

  ![Gif showcasing alpaca repl mode](./doc/resources/alpaca_repl_screencap.gif)

- Sessions can be loaded (`--load-session`) or saved (`--save-session`) to file.
  To automatically load and save the same session, use `--persist-session`.
  This can be used to cache prompts to reduce load time, too:

  ![Gif showcasing prompt caching](./doc/resources/prompt_caching_screencap.gif)

  (This GIF shows an older version of the flags, but the mechanics are still the same.)

[^1]:
    A modern-ish C toolchain is required to compile `ggml`. A C++ toolchain
    should not be necessary.

### Docker

```shell
# To build (This will take some time, go grab some coffee):
docker build -t llama-rs .

# To run with prompt:
docker run --rm --name llama-rs -it -v ${PWD}/data:/data -v ${PWD}/examples:/examples llama-rs infer -m data/gpt4all-lora-quantized-ggml.bin -p "Tell me how cool the Rust programming language is:"

# To run with prompt file and repl (will wait for user input):
docker run --rm --name llama-rs -it -v ${PWD}/data:/data -v ${PWD}/examples:/examples llama-rs repl -m data/gpt4all-lora-quantized-ggml.bin -f examples/alpaca_prompt.txt
```

## Q&A

### Why did you do this?

It was not my choice. Ferris appeared to me in my dreams and asked me
to rewrite this in the name of the Holy crab.

### Seriously now.

Come on! I don't want to get into a flame war. You know how it goes,
_something something_ memory _something something_ cargo is nice, don't make
me say it, everybody knows this already.

### I insist.

_Sheesh! Okaaay_. After seeing the huge potential for **llama.cpp**,
the first thing I did was to see how hard would it be to turn it into a
library to embed in my projects. I started digging into the code, and realized
the heavy lifting is done by `ggml` (a C library, easy to bind to Rust) and
the whole project was just around ~2k lines of C++ code (not so easy to bind).
After a couple of (failed) attempts to build an HTTP server into the tool, I
realized I'd be much more productive if I just ported the code to Rust, where
I'm more comfortable.

### Is this the real reason?

Haha. Of course _not_. I just like collecting imaginary internet
points, in the form of little stars, that people seem to give to me whenever I
embark on pointless quests for _rewriting X thing, but in Rust_.

### How is this different from `llama.cpp`?

This is a reimplementation of `llama.cpp` that does not share any code with it
outside of `ggml`. This was done for a variety of reasons:

- `llama.cpp` requires a C++ compiler, which can cause problems for
  cross-compilation to more esoteric platforms. An example of such a platform
  is WebAssembly, which can require a non-standard compiler SDK.
- Rust is easier to work with from a development and open-source perspective;
  it offers better tooling for writing "code in the large" with many other
  authors. Additionally, we can benefit from the larger Rust ecosystem with
  ease.
- We would like to make `ggml` an optional backend
  (see [this issue](https://github.com/rustformers/llama-rs/issues/31)).

In general, we hope to build a solution for model inferencing that is as easy
to use and deploy as any other Rust crate.
