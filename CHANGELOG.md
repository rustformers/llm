# Changelog

`llm` is actively being iterated upon, so there will be breaking changes to the interface and to compatibility. Where possible, we will try to find ways to mitigate the breaking changes, but we do not expect to have a stable interface for some time.

# 0.2.0-dev (unreleased)

- `llm` now uses the latest GGML version. This limits use to older unquantized models or to models quantized with the latest version (quantization version 2, file format GGJTv3). We are investigating ways to [mitigate this breakage in the future](https://github.com/rustformers/llm/discussions/261).
- `llm::InferenceRequest` no longer implements `Default::default`.
- The `infer` callback now provides an `InferenceResponse` instead of a string to disambiguate the source of the token. Additionally, it now returns an `InferenceFeedback` to control whether or not the generation should continue.
- Several fields have been renamed:
  - `n_context_tokens` -> `context_size`

# 0.1.1 (2023-05-08)

- Fix an issue with the binary build of `llm-cli`.

# 0.1.0 (2023-05-08)

Initial release.
