# GGML - Large Language Models for Everyone

[GGML](https://github.com/ggerganov/ggml) is a format for distributing large
language models (LLMs); the name is a portmanteau of the initials of its
originator ([Georgi Gerganov](https://ggerganov.com/)) and the acronym ML, which
stands for machine learning. This crate provides Rust [bindings](sys) into the
reference implementation of GGML (written in C), as well as a collection of
[native](src) Rust helpers to provide safe, idiomatic access to those bindings.
GGML makes use of a technique called
"[quantization](<https://en.wikipedia.org/wiki/Quantization_(signal_processing)>)"
that allows for large language models to run on consumer hardware. This
documents describes the basics of the GGML format, including how
[quantization](#quantization) is used to democratize access to LLMs.

## Format

GGML files consists of binary-encoded data that is laid out according to a
specified format. The format specifies what kind of data is present in the file,
how it is represented, and the order in which it appears. The first piece of
information present in a valid GGML file is a GGML version number, followed by
three components that define a large language model: the model's
hyperparameters, its vocabulary, and its weights. Continue reading to learn more
about GGML versions and the components of a GGML model.

### GGML Versions

GGML is "bleeding-edge" technology and undergoes frequent changes. In an effort
to support rapid development without sacrificing backwards-compatibility, GGML
uses versioning to introduce improvements that may change the format of the
encoding. For example, newer versions of GGML make use of
[vocabulary](#vocabulary)-scoring, which introduces extra information into the
encoding, as well as [mmap](https://en.wikipedia.org/wiki/Mmap), which enhances
performance through memory-mapping. The first value that is present in a valid
GGML file is a "magic number" that indicates the GGML version that was used to
encode the model.

### Hyperparamaters

The term
"[hyperparameter](<https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)>)"
describes a value that is used to configure the behavior of a large language
model; this is in contrast to the model's **parameters**, which are the
[weights](#weights) that were derived in the training process that was used to
create the model. Each model defines its own hyperparameter structure that
defines the hyperparameter values accepted by that model. Valid GGML files must
list these values in the correct order, and each value must be represented using
the correct data type. Although hyperparameters are different across models,
some attributes appear in the hyperparameters for most models:

- `n_vocab`: the size of the model's [vocabulary](#vocabulary)
- `n_embd`: the size of the model's
  "[embedding](https://en.wikipedia.org/wiki/Word_embedding) layer", which is
  used during prompt ingestion
- `n_layer`: the number of layers in the model; each layer represents a set of
  [weights](#weights).

### Vocabulary

As the name implies, a model's vocabulary comprises components that are used by
the model to generate language (text). However, unlike the vocabulary of a
human, which consists of _words_, the vocabulary of a large language model
consists of "tokens". A token _can_ be an entire word, but oftentimes they are
word _fragments_. Just like humans can compose millions of words from just a
dozen or two letters, large language models use _tokens_ to express a large
number of words from a relatively smaller number of components. Consider a
vocabulary with the following tokens: `whi`, `ch` `le`, `who`, and `a`; this
vocabulary can be used to create the English words "which", "while", "who", "a",
and "leach". How would the behavior change if the model contained the following
tokens: `wh`, `ich`, `ile`, `o`, and `leach`? Choices such as these allow
model-creators to tune the behavior and performance of their models.

As described above, the model's [hyperparameters](#hyperparamaters) typically
contains a value that specifies the number of tokens in the vocabulary. The
vocabulary is encoded as a list of tokens, each of which includes an unsigned
32-bit integer that specifies the length of the token. Depending on the GGML
version, the token may also include a 32-bit floating point score.

[comment]: <> (I need help describing token-scoring)

### Weights

The final, and largest, component of a GGML file is the weights of the LLM that
the file represents. Abstractly, a large language model is software that is used
to generate language - just like software that is used to generate _images_ can
be improved by increasing the number of colors with which images can be
rendered, large language models can be improved by increasing the number of
_weights_ in the model. The total number of a weights in a model are referred to
as the "size" of that model. For example, the
[StableLM](https://github.com/Stability-AI/StableLM) implementation of the
[GPT-NeoX](https://github.com/EleutherAI/gpt-neox) language model architecture
is available in a number of sizes, like 3B and 7B, which stands for 3-billion
and 7-billion, respectively. These numbers refer to the total number of weights
in that model. As described in the [hyperparameters](#hyperparamaters) section,
weights are grouped together in sets called "layers", which, like
hyperparameters, have structures that are uniquely defined by the model
architecture; within a layer, weights are grouped together in structures called
"tensors". So, for instance, both StableLM 3B and StableLM 7B use layers that
comprise the same tensors, but StableLM 3B has relatively _fewer_ layers when
compared to StableLM 7B.

In GGML, a tensor consists of a number of components, including: a name, a
4-element list that represents the number of dimensions in the tensor and their
lengths, and a list of the weights in that tensor. For example, consider the
following 2 ⨯ 2 tensor named `tensor_a0`:

<table style="text-align: center">
  <tr>
    <td colspan="2"><code>tensor_a0</code></td>
  </tr>
  <tr>
    <td>1.0</td>
    <td>0.0</td>
  </tr>
  <tr>
    <td>0.1</td>
    <td>1.1</td>
  </tr>
</table>

A simplification of the GGML representation of `tensor_a0` is
`{"tensor_a0", [2, 2, 1, 1], [1.0, 0.0, 0.1, 1.0]}`. Note that the 4-element
list of dimensions uses `1` as a placeholder for unused dimensions - this is
because the product of the dimensions should not equal zero.

The weights in a GGML file are encoded as a list of layers, the length of which
is typically specified in the model's hyperparameters; each layer is encoded as
an ordered set of tensors.

#### Quantization

LLM weights are floating point (decimal) numbers. Just like it requires more
space to represent a large integer (e.g. 1000) compared to a small integer (e.g.
1), it requires more space to represent a high-precision floating point number
(e.g. 0.0001) compared to a low-precision floating number (e.g. 0.1). The
process of "quantizing" a large language model involves reducing the precision
with which weights are represented in order to reduce the resources required to
use the model. GGML supports a number of different quantization strategies (e.g.
4-bit, 5-bit, and 8-bit quantization), each of which offers different trade-offs
between efficiency and performance.
