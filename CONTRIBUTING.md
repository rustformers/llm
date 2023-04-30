# Contributors guide

This document contains a few things that contributors should know about how the
project is managed. It will be expanded over time with more information.

## Regenerating the GGML Bindings

When new GGML versions are pushed to llama.cpp (or one of the other repos
hosting a copy of it) and we want to update our copy, the process should be as
follows:

- Update the submodule to the latest version of GGML:
  ```shell
  $ git submodule update --remote
  ```
- Run the bindgen script:
  ```shell
  $ cargo run --bin generate-ggml-bindings ggml-sys
  ```
- Fix any compiler errors that pop up due to the new version of the bindings and
  test the changes.
