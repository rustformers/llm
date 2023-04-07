# Contributors guide

This document contains a few things that contributors should know about how the
project is managed. It will be expanded over time with more information.

## Regenerating the GGML Bindings

When new GGML versions are pushed to llama.cpp (or one of the other repos
hosting a copy of it) and we want to update our copy, the process should be as
follows:

- Update the `ggml.c` and `ggml.h` inside `ggml-sys/ggml`.
- In that same folder, update `CREDITS.txt` to indicate the llama.cpp version 
  these files were taken from 
- Run the bindgen script:
    ```shell
    $ cargo run --bin generate-ggml-bindings ggml-sys
    ```
- Fix any compiler errors that pop up due to the new version of the bindings and
  test the changes.
