# `llm` - Large Language Models for Everyone, in Rust

`llm` is an ecosystem of Rust libraries for working with large language models -
its built on top of the fast, efficient [GGML](./crates/ggml) library for
machine learning.

![A llama riding a crab, AI-generated](./doc/img/llm-crab-llama.png)

> _Image by [@darthdeus](https://github.com/darthdeus/), using Stable Diffusion_

[![Latest version](https://img.shields.io/crates/v/llm.svg)](https://crates.io/crates/llm)
![MIT/Apache2](https://shields.io/badge/license-MIT%2FApache--2.0-blue)
[![Discord](https://img.shields.io/discord/1085885067601137734)](https://discord.gg/YB9WaXYAWU)

The primary entrypoint [for developers](#getting-started) is
[the `llm` crate](./crates/llm), which wraps [`llm-base`](./crates/llm-base) and
the [supported model](./crates/models) crates.

For end-users, there is [a CLI application](#building-llm-cli),
[`llm-cli`](./binaries/llm-cli), which provides a convenient interface for
interacting with supported models. [Text generation](#running) can be done as a
one-off based on a prompt, or interactively, through
[REPL or chat](#does-the-llm-cli-support-chat-mode) modes. The CLI can also be
used to serialize (print) decoded models,
[quantize](./crates/ggml/README.md#quantization) GGML files, or compute the
[perplexity](https://huggingface.co/docs/transformers/perplexity) of a model. It
can be downloaded from
[the latest GitHub release](https://github.com/rustformers/llm/releases) or by
installing it from `crates.io`.

`llm` is powered by the [`ggml`](https://github.com/ggerganov/ggml) tensor
library, and aims to bring the robustness and ease of use of Rust to the world
of large language models. At present, inference is only on the CPU, but we hope
to support GPU inference in the future through alternate backends.

Currently, the following models are supported:

- [BLOOM](https://huggingface.co/docs/transformers/model_doc/bloom)
- [GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [GPT-J](https://huggingface.co/docs/transformers/model_doc/gptj)
- [GPT-NeoX](https://huggingface.co/docs/transformers/model_doc/gpt_neox):
  GPT-NeoX, StableLM, RedPajama, Dolly v2
- [LLaMA](https://huggingface.co/docs/transformers/model_doc/llama): LLaMA,
  Alpaca, Vicuna, Koala, GPT4All v1, GPT4-X, Wizard
- [MPT](https://www.mosaicml.com/blog/mpt-7b)

## Getting Started

This project depends on Rust v1.65.0 or above and a modern C toolchain.

The `llm` crate exports `llm-base` and the model crates (e.g. `bloom`, `gpt2`
`llama`).

To use `llm`, add it to your `Cargo.toml`:

```toml
[dependencies]
llm = "0.2"
```

**NOTE**: To improve debug performance, exclude `llm` from being built in debug
mode:

```toml
[profile.dev.package.llm]
opt-level = 3
```

### Building `llm-cli`

Follow these steps to build the command line application, which is named `llm`:

#### Using `cargo`

To install `llm` to your Cargo `bin` directory, which `rustup` is likely to have
added to your `PATH`, run:

```shell
cargo install llm-cli
```

The CLI application can then be run through `llm`.

#### From Source

Clone the repository and then build it with

```shell
git clone --recurse-submodules git@github.com:rustformers/llm.git
cargo build --release
```

The resulting binary will be at `target/release/llm[.exe]`.

It can also be run directly through Cargo, with

```shell
cargo run --release -- $ARGS
```

### Getting Models

GGML files are easy to acquire. For a list of models that have been tested, see
the [known-good models](./doc/known-good-models.md).

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

### Running

Once the `llm` executable has been built or is in a `$PATH` directory, try
running it. Here's an example that uses the open-source
[GPT4All](https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin) language
model:

```shell
llm llama infer -m ggml-gpt4all-j-v1.3-groovy.bin -p "Rust is a cool programming language because"
```

For more information about the `llm` CLI, use the `--help` parameter.

There is also a [simple inference example](./crates/llm/examples/inference.rs)
that is helpful for [debugging](./.vscode/launch.json):

```shell
cargo run --release --example inference llama ggml-gpt4all-j-v1.3-groovy.bin $OPTIONAL_PROMPT
```

## Working with Raw Models

Python v3.9 or v3.10 is needed to convert a raw model to a GGML-compatible
format (note that Python v3.11 is not supported):

```shell
python3 util/convert-pth-to-ggml.py $MODEL_HOME/$MODEL/7B/ 1
```

The output of the above command can be used by `llm` to create a
[quantized](./crates/ggml/README.md#quantization) model:

```shell
cargo run --release llama quantize $MODEL_HOME/$MODEL/7B/ggml-model-f16.bin $MODEL_HOME/$MODEL/7B/ggml-model-q4_0.bin q4_0
```

In future, we hope to provide
[a more streamlined way of converting models](https://github.com/rustformers/llm/issues/21).

> **Note**
>
> The [llama.cpp repository](https://github.com/ggerganov/llama.cpp) has
> additional information on how to obtain and run specific models.

## Q&A

### Does the `llm` CLI support chat mode?

Yes, but certain fine-tuned models (e.g.
[Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html),
[Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/),
[Pygmalion](https://docs.alpindale.dev/)) are more more suited to chat use-cases
than so-called "base models". Here's an example of using the `llm` CLI in REPL
(Read-Evaluate-Print Loop) mode with an Alpaca model - note that the
[provided prompt format](./examples/alpaca_prompt.txt) is tailored to the model
that is being used:

```shell
llm llama repl -m ggml-alpaca-7b-q4.bin -f examples/alpaca_prompt.txt
```

There is also a [Vicuna chat example](./crates/llm/examples/vicuna-chat.rs) that
demonstrates how to create a custom chatbot:

```shell
cargo run --release --example vicuna-chat llama ggml-vicuna-7b-q4.bin
```

### Can `llm` sessions be persisted for later use?

Sessions can be loaded (`--load-session`) or saved (`--save-session`) to file.
To automatically load and save the same session, use `--persist-session`. This
can be used to cache prompts to reduce load time, too.

### Do you provide support for Docker and NixOS?

The `llm` [Dockerfile](./util/Dockerfile) is in the `util` directory, as is a
[Flake](./util/flake) manifest and lockfile.

### Do you accept contributions?

Absolutely! Please see the [contributing guide](./doc/CONTRIBUTING.md).

### What applications and libraries use `llm`?

#### Applications

- [llmcord](https://github.com/rustformers/llmcord): Discord bot for generating
  messages using `llm`.
- [local.ai](https://github.com/louisgv/local.ai): Desktop app for hosting an
  inference API on your local machine using `llm`.

#### Libraries

- [llm-chain](https://github.com/sobelio/llm-chain): Build chains in large
  language models for text summarization and completion of more complex tasks
