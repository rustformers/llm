# Known-good models

The following models have been tested and are known to work with `llm`.

We are collecting models in the [rustformers](https://huggingface.co/rustformers) organization,
but this work is ongoing.

The LLaMA architecture is the most well-supported.

## LLaMA

We have chosen not to include any models based on the original LLaMA model due to licensing concerns.
However, the OpenLLaMA models are available under the Apache 2.0 license and are compatible with `llm`.

- <https://huggingface.co/rustformers/open-llama-ggml>
- <https://huggingface.co/TheBloke/open-llama-13b-open-instruct-GGML>
- <https://huggingface.co/TheBloke/Flan-OpenLlama-7B-GGML>

Models based on the original LLaMA model are also compatible, but you will need to find them yourselves.

## GPT-2

- <https://huggingface.co/lxe/Cerebras-GPT-2.7B-Alpaca-SP-ggml>: note that this is `f16`-only and
  we recommend you quantize it using `llm` for best performance.
- <https://huggingface.co/TheBloke/WizardCoder-15B-1.0-GGML>

## GPT-J

- <https://huggingface.co/rustformers/gpt4all-j-ggml>
- <https://huggingface.co/rustformers/gpt-j-ggml>

## MPT

- <https://huggingface.co/rustformers/mpt-7b-ggml>

## GPT-NeoX/RedPajama

- <https://huggingface.co/rustformers/redpajama-ggml>
- <https://huggingface.co/rustformers/pythia-ggml>
- <https://huggingface.co/rustformers/stablelm-ggml>
- <https://huggingface.co/rustformers/dolly-v2-ggml>

## BLOOM

- <https://huggingface.co/rustformers/bloomz-ggml>
- <https://huggingface.co/rustformers/bloom-ggml>
