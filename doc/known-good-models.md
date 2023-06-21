# Known-good models

The following models have been tested and are known to work with `llm`.

We are collecting models in the [rustformers](https://huggingface.co/rustformers) organization,
but this work is ongoing.

## GPT-2

- <https://huggingface.co/lxe/Cerebras-GPT-2.7B-Alpaca-SP-ggml>: note that this is `f16`-only and
  we recommend you quantize it using `llm` for best performance.
- <https://huggingface.co/TheBloke/WizardCoder-15B-1.0-GGML>

## GPT-J

At the time of writing, there are no publicly-released models with the GPT-J architecture and
the currently-supported GGML quantization version.

You will need to obtain a F16 model and quantize it using `llm`.

## LLaMA

We have chosen not to include any models here until we have a better understanding of the licensing situation.

## MPT

- <https://huggingface.co/rustformers/mpt-7b-ggml>

## GPT-NeoX/RedPajama

- <https://huggingface.co/rustformers/redpajama-ggml>
- <https://huggingface.co/rustformers/pythia-ggml>

## BLOOM

- <https://huggingface.co/rustformers/bloom-ggml>
