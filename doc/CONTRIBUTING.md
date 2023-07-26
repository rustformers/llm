# Contributors Guide

The purpose of this document is to make it easy for open-source community
members to contribute to this project. We'd love to discuss your contributions
with you via a GitHub [Issue](https://github.com/rustformers/llm/issues/new) or
[Discussion](https://github.com/rustformers/llm/discussions/new?category=ideas),
or on [Discord](https://discord.gg/YB9WaXYAWU)!

## Checking Changes

This project uses a [GitHub workflow](../.github/workflows/rust.yml) to enforce
code standards.

The `rusty-hook` project is used to run a similar set of checks automatically before committing.
If you would like to run these checks locally, use `cargo run -p precommit-check`.

## Regenerating GGML Bindings

Follow these steps to update the GGML submodule and regenerate the Rust bindings
(this is only necessary if your changes depend on new GGML features):

```shell
git submodule update --remote
cargo run --release --package generate-ggml-bindings
```

## Debugging

This repository includes a [`launch.json` file](../.vscode/launch.json) that can
be used for
[debugging with Visual Studio Code](https://code.visualstudio.com/docs/editor/debugging) -
this file will need to be updated to reflect where models are stored on your
system. Debugging with Visual Studio Code requires a
[language extension](https://code.visualstudio.com/docs/languages/rust#_install-debugging-support)
that depends on your operating system. Keep in mind that debugging text
generation is extremely slow, but debugging model loading is not.

## LLM References

Here are some tried-and-true references for learning more about large language
models:

- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) - an
  excellent technical description of how this seminal language model generates
  text
- [Andrej Karpathy's "Neural Networks: Zero to Hero"](https://karpathy.ai/zero-to-hero.html) -
  a series of in-depth YouTube videos that guide the viewer through creating a
  neural network, a large language model, and a fully functioning chatbot, from
  scratch (in Python)
- [rustygrad](https://github.com/Mathemmagician/rustygrad) - a native Rust
  implementation of Andrej Karpathy's micrograd
- [Understanding Deep Learning](https://udlbook.github.io/udlbook/) (Chapter 12
  specifically)
