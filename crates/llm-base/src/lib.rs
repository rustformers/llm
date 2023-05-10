//! This crate provides a unified interface for loading and using
//! Large Language Models (LLMs).
//!
//! This is the base crate that implementors can use to implement their own
//! LLMs.
//!
//! As a user, you probably want to use the [llm](https://crates.io/crates/llm) crate instead.
#![deny(missing_docs)]

use thiserror::Error;

mod inference_session;
mod loader;
mod quantize;
mod vocabulary;

pub mod model;
pub mod util;

pub use ggml;
pub use ggml::Type as ElementType;

pub use inference_session::{
    InferenceFeedback, InferenceRequest, InferenceResponse, InferenceSession,
    InferenceSessionConfig, InferenceSnapshot, InferenceStats, ModelKVMemoryType, SnapshotError,
};
pub use loader::{
    load, load_progress_callback_stdout, ContainerType, FileType, LoadError, LoadProgress, Loader,
    TensorLoader,
};
pub use memmap2::Mmap;
pub use model::{Hyperparameters, KnownModel, Model, ModelParameters, OutputRequest};
pub use quantize::{quantize, QuantizeError, QuantizeProgress};
pub use util::TokenUtf8Buffer;
pub use vocabulary::{InvalidTokenBias, TokenBias, TokenId, Vocabulary};

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
    fn default() -> Self {
        Self {
            n_threads: 8,
            n_batch: 8,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.30,
            temperature: 0.80,
            bias_tokens: TokenBias::default(),
            repetition_penalty_last_n: 512,
        }
    }
}

#[derive(Error, Debug)]
/// Errors encountered during the inference process.
pub enum InferenceError {
    #[error("an invalid token was encountered during tokenization")]
    /// During tokenization, one of the produced tokens was invalid / zero.
    TokenizationFailed,
    #[error("the context window is full")]
    /// The context window for the model is full.
    ContextFull,
    #[error("reached end of text")]
    /// The model has produced an end of text token, signalling that it thinks that the text should end here.
    ///
    /// Note that this error *can* be ignored and inference can continue, but the results are not guaranteed to be sensical.
    EndOfText,
    #[error("the user-specified callback returned an error")]
    /// The user-specified callback returned an error.
    UserCallback(Box<dyn std::error::Error>),
}
