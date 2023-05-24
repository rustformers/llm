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
pub use model::{
    Hyperparameters, KnownModel, Model, ModelDynamicOverrideValue, ModelDynamicOverrides,
    ModelParameters, OutputRequest,
};
pub use quantize::{quantize, QuantizeError, QuantizeProgress};
pub use util::TokenUtf8Buffer;
pub use vocabulary::{
    ExternalVocabulary, InvalidTokenBias, ModelVocabulary, Prompt, TokenBias, TokenId,
    TokenizationError, Vocabulary, VocabularySource,
};

#[derive(Clone, Debug, PartialEq)]
/// The parameters for text generation.
///
/// This needs to be provided during all inference calls,
/// but can be changed between calls.
pub struct InferenceParameters {
    /// The number of threads to use.
    pub n_threads: usize,
    /// Controls batch/chunk size for prompt ingestion in
    /// [InferenceSession::feed_prompt].
    pub n_batch: usize,
    /// The top K words by score are kept during sampling.
    pub top_k: usize,
    /// The cumulative probability after which no more words are kept for sampling.
    pub top_p: f32,
    /// The penalty for repeating tokens. Higher values make the generation less
    /// likely to get into a loop, but may harm results when repetitive outputs
    /// are desired.
    pub repeat_penalty: f32,
    /// Temperature (randomness) used for sampling. A higher number is more random.
    pub temperature: f32,
    /// A list of tokens to bias against in the process of generation.
    pub bias_tokens: TokenBias,
    /// The number of tokens to consider for the repetition penalty.
    pub repetition_penalty_last_n: usize,
}
impl Default for InferenceParameters {
    /// Returns a reasonable default for the parameters.
    ///
    /// Note that these parameters are not necessarily optimal for all models, and that
    /// you may want to tweak them for your use case.
    fn default() -> Self {
        Self {
            n_threads: 8,
            n_batch: 8,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.30,
            temperature: 0.80,
            bias_tokens: TokenBias::empty(),
            repetition_penalty_last_n: 512,
        }
    }
}
