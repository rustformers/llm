# llm

![A llama riding a crab, AI-generated](./doc/resources/logo2.png)

> _Image by [@darthdeus](https://github.com/darthdeus/), using Stable Diffusion_

[![Latest version](https://img.shields.io/crates/v/llm.svg)](https://crates.io/crates/llm)
![MIT/Apache2](https://shields.io/badge/license-MIT%2FApache--2.0-blue)
[![Discord](https://img.shields.io/discord/1085885067601137734)](https://discord.gg/YB9WaXYAWU)

`llm` is a Rust ecosystem of libraries for running inference on large language
models, inspired by [llama.cpp](https://github.com/ggerganov/llama.cpp).

The primary crate is the `llm` crate, which wraps `llm-base` and supported model
crates. This is used by `llm-cli` to provide inference for all supported models.

It is powered by the [`ggml`](https://github.com/ggerganov/ggml) tensor library,
and aims to bring the robustness and ease of use of Rust to the world of large
language models.

## Getting started

Make sure you have a Rust 1.65.0 or above and C toolchain[^1] set up.

`llm` is a Rust library that re-exports `llm-base` and the model crates (e.g.
`bloom`, `gpt2` `llama`).

`llm-cli` (binary name `llm`) is a basic application that provides a CLI
interface to the library.

**NOTE**: For best results, make sure to build and run in release mode. Debug
builds are going to be very slow.

### Building using `cargo`

Run

```shell
cargo install --git https://github.com/rustformers/llm llm-cli
```

to install `llm` to your Cargo `bin` directory, which `rustup` is likely to have
added to your `PATH`.

The CLI application can then be run through `llm`.

![Gif showcasing language generation using llm](./doc/resources/llama_gif.gif)

### Building from repository

Clone the repository and then build it with

```shell
git clone --recurse-submodules git@github.com:rustformers/llm.git
cargo build --release
```

The resulting binary will be at `target/release/llm[.exe]`.

It can also be run directly through Cargo, using

```shell
cargo run --release -- <ARGS>
```

This is useful for development.

### Getting models

GGML files are easy to acquire. Currently, the following models are supported:

- [GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [GPT-J](https://huggingface.co/docs/transformers/model_doc/gptj)
- [LLaMA](https://huggingface.co/docs/transformers/model_doc/llama): LLaMA, Alpaca, Vicuna, Koala, GPT4All v1, GPT4-X, Wizard
- [GPT-NeoX](https://huggingface.co/docs/transformers/model_doc/gpt_neox): GPT-NeoX, StableLM, Dolly v2 (partial, not the same tensor names?)
- [BLOOM](https://huggingface.co/docs/transformers/model_doc/bloom): BLOOMZ

For a list of models that have been tested, see the [known-good models](./known-good-models.md).

Certain older GGML formats are not supported by this project, but the goal is to
maintain feature parity with the upstream GGML project. For problems relating to
loading models, or requesting support for
[supported GGML model types](https://github.com/ggerganov/ggml#roadmap), please
[open an Issue](https://github.com/rustformers/llm/issues/new).

#### From Hugging Face

Hugging Face 🤗 is a leader in open-source machine learning and hosts hundreds
of GGML models.
[Search for GGML models on Hugging Face 🤗](https://huggingface.co/models?search=ggml).

#### r/LocalLLaMA

This Reddit community maintains
[a wiki](https://www.reddit.com/r/LocalLLaMA/wiki/index/) related to GGML
models, including well organized lists of links for acquiring
[GGML models](https://www.reddit.com/r/LocalLLaMA/wiki/models/) (mostly from
Hugging Face 🤗).

#### LLaMA original weights

Currently, the only legal source to get the original weights is
[this repository](https://github.com/facebookresearch/llama/blob/main/README.md#llama).

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
cargo run --release llama quantize /path/to/your/models/7B/ggml-model-f16.bin /path/to/your/models/7B/ggml-model-q4_0.bin q4_0
```

> **Note**
>
> The [llama.cpp repository](https://github.com/ggerganov/llama.cpp) has
> additional information on how to obtain and run specific models.

### Running

For example, try the following prompt:

```shell
llm llama infer -m <path>/ggml-model-q4_0.bin -p "Tell me how cool the Rust programming language is:"
```

Some additional things to try:

- Use `--help` to see a list of available options.
- If you have the [alpaca-lora](https://github.com/tloen/alpaca-lora) weights,
  try `repl` mode!

  ```shell
  llm llama repl -m <path>/ggml-alpaca-7b-q4.bin -f examples/alpaca_prompt.txt
  ```

  ![Gif showcasing alpaca repl mode](./doc/resources/alpaca_repl_screencap.gif)

- Sessions can be loaded (`--load-session`) or saved (`--save-session`) to file.
  To automatically load and save the same session, use `--persist-session`. This
  can be used to cache prompts to reduce load time, too:

  ![Gif showcasing prompt caching](./doc/resources/prompt_caching_screencap.gif)

  (This GIF shows an older version of the flags, but the mechanics are still the
  same.)

[^1]:
    A modern-ish C toolchain is required to compile `ggml`. A C++ toolchain
    should not be necessary.

### Docker

```shell
# To build (This will take some time, go grab some coffee):
docker build -t llm .

# To run with prompt:
docker run --rm --name llm -it -v ${PWD}/data:/data -v ${PWD}/examples:/examples llm llama infer -m data/gpt4all-lora-quantized-ggml.bin -p "Tell me how cool the Rust programming language is:"

# To run with prompt file and repl (will wait for user input):
docker run --rm --name llm -it -v ${PWD}/data:/data -v ${PWD}/examples:/examples llm llama repl -m data/gpt4all-lora-quantized-ggml.bin -f examples/alpaca_prompt.txt
```

## Q&A

### Why did you do this?

It was not my choice. Ferris appeared to me in my dreams and asked me to rewrite
this in the name of the Holy crab.

### Seriously now.

Come on! I don't want to get into a flame war. You know how it goes, _something
something_ memory _something something_ cargo is nice, don't make me say it,
everybody knows this already.

### I insist.

_Sheesh! Okaaay_. After seeing the huge potential for **llama.cpp**, the first
thing I did was to see how hard would it be to turn it into a library to embed
in my projects. I started digging into the code, and realized the heavy lifting
is done by `ggml` (a C library, easy to bind to Rust) and the whole project was
just around ~2k lines of C++ code (not so easy to bind). After a couple of
(failed) attempts to build an HTTP server into the tool, I realized I'd be much
more productive if I just ported the code to Rust, where I'm more comfortable.

### Is this the real reason?

Haha. Of course _not_. I just like collecting imaginary internet points, in the
form of little stars, that people seem to give to me whenever I embark on
pointless quests for _rewriting X thing, but in Rust_.

### How is this different from `llama.cpp`?

This is a reimplementation of `llama.cpp` that does not share any code with it
outside of `ggml`. This was done for a variety of reasons:

- `llama.cpp` requires a C++ compiler, which can cause problems for
  cross-compilation to more esoteric platforms. An example of such a platform is
  WebAssembly, which can require a non-standard compiler SDK.
- Rust is easier to work with from a development and open-source perspective; it
  offers better tooling for writing "code in the large" with many other authors.
  Additionally, we can benefit from the larger Rust ecosystem with ease.
- We would like to make `ggml` an optional backend (see
  [this issue](https://github.com/rustformers/llm/issues/31)).

In general, we hope to build a solution for model inferencing that is as easy to
use and deploy as any other Rust crate.
