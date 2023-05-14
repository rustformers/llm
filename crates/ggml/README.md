# GGML - Large Language Models for Everyone

[GGML](https://github.com/ggerganov/ggml) is a format for distributing large
language models (LLMs); the name is a portmanteau of the initials of its
originator ([Georgi Gerganov](https://ggerganov.com/)) and the acronym ML, which
stands for machine learning. This crate provides Rust [bindings](sys) into the
reference implementation of GGML (written in C), as well as a collection of
native [Rust helpers](src) to provide safe, idiomatic access to those bindings.
GGML makes use of a technique called
"[quantization](<https://en.wikipedia.org/wiki/Quantization_(signal_processing)>)"
that allows for large language models to be run on consumer hardware. Continue
reading to learn more about the basics of the GGML format and how quantization
is used to democratize access to LLMs.

## Format

GGML models consists of binary-encoded data that is laid out according to a
specified format. The format specifies what kind of data is present in the file,
how it is represented, and the order in which it appears. The first piece of
information present in a valid GGML file is a GGML version number, followed by
three components that define a large language model: the model's
hyperparameters, its vocabulary, and its weights. Continue reading to learn more
about GGML versions and the components of a GGML model.

### GGML Versions

GGML is "bleeding-edge" technology and undergoes frequent changes. In an effort
to support rapid development without sacrificing backwards-compatibility, GGML
uses versioning that can be used to specify additional details about the
encoding or make use of performance-enhancing improvements. For example, newer
versions of GGML make use of [vocabulary](#vocabulary)-scoring, which introduces
extra information into the encoding, as well as
[mmap](https://en.wikipedia.org/wiki/Mmap), which enhances performance through
memory-mapping.

### Hyperparamaters

The term
"[hyperparameter](<https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)>)"
describes a value that is used to configure the behavior of a large language
model; this is in contrast to the model's **parameters**, which are the
[weights](#weights) that were derived via the training process that was used to
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
