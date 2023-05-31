//! This crate provides a unified interface for loading and using
//! Large Language Models (LLMs).
//!
//! This is the base crate that implementors can use to implement their own
//! LLMs.
//!
//! As a user, you probably want to use the [llm](https://crates.io/crates/llm) crate instead.
#![deny(missing_docs)]

mod inference_session;
mod loader;
mod lora;
mod quantize;
mod vocabulary;

pub mod model;
pub mod samplers;
pub mod util;

pub use ggml;
pub use ggml::Type as ElementType;

pub use inference_session::{
    feed_prompt_callback, InferenceError, InferenceFeedback, InferenceRequest, InferenceResponse,
    InferenceSession, InferenceSessionConfig, InferenceSnapshot, InferenceStats, ModelKVMemoryType,
    SnapshotError,
};
pub use loader::{
    load, load_progress_callback_stdout, ContainerType, FileType, FileTypeFormat, LoadError,
    LoadProgress, Loader, TensorLoader,
};
pub use lora::{LoraAdapter, LoraParameters};
pub use memmap2::Mmap;
pub use model::{Hyperparameters, KnownModel, Model, ModelParameters, OutputRequest};
pub use quantize::{quantize, QuantizeError, QuantizeProgress};
pub use regex::Regex;
pub use samplers::Sampler;
pub use util::TokenUtf8Buffer;
pub(crate) use vocabulary::ModelVocabulary;
pub use vocabulary::{
    InvalidTokenBias, Prompt, TokenBias, TokenId, TokenizationError, Vocabulary,
    VocabularyLoadError, VocabularySource,
};

#[derive(Clone, Debug)]
/// The parameters for text generation.
///
/// This needs to be provided during all inference calls,
/// but can be changed between calls.
pub struct InferenceParameters<'a> {
    /// The number of threads to use. This is dependent on your user's system,
    /// and should be selected accordingly.
    ///
    /// Note that you should aim for a value close to the number of physical cores
    /// on the system, as this will give the best performance. This means that, for
    /// example, on a 16-core system with hyperthreading, you should set this to 16.
    ///
    /// Also note that not all cores on a system are equal, and that you may need to
    /// experiment with this value to find the optimal value for your use case. For example,
    /// Apple Silicon and modern Intel processors have "performance" and "efficiency" cores,
    /// and you may want to only use the performance cores.
    ///
    /// A reasonable default value is 8, as most modern high-performance computers have
    /// 8 physical cores. Adjust to your needs.
    pub n_threads: usize,
    /// Controls batch/chunk size for prompt ingestion in [InferenceSession::feed_prompt].
    ///
    /// This is the number of tokens that will be ingested at once. This is useful for
    /// trying to speed up the ingestion of prompts, as it allows for parallelization.
    /// However, you will be fundamentally limited by your machine's ability to evaluate
    /// the transformer model, so increasing the batch size will not always help.
    ///
    /// A reasonable default value is 8.
    pub n_batch: usize,
    /// The sampler to use for sampling tokens from the model's probabilities.
    ///
    /// Each time the model runs, it generates a distribution of probabilities; each token
    /// has a probability of being the next token. The sampler is responsible for sampling
    /// from this distribution to generate the next token. Using a different sampler may
    /// change the output of the model, or control how deterministic the generated text is.
    ///
    /// A recommended default sampler is [TopPTopK](samplers::TopPTopK), which is a standard
    /// sampler that offers a [Default](samplers::TopPTopK::default) implementation.
    pub sampler: &'a dyn Sampler,
}
