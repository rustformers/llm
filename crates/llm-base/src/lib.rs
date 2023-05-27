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

use std::sync::Arc;

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
pub use regex::Regex;
pub use samplers::Sampler;
pub use util::TokenUtf8Buffer;
pub use vocabulary::{InvalidTokenBias, Prompt, TokenBias, TokenId, TokenizationError, Vocabulary};

#[derive(Clone, Debug)]
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
    /// The sampler to use for sampling tokens from the model's probabilities.
    pub sampler: Arc<dyn Sampler>,
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
            sampler: Arc::new(samplers::TopPTopK::default()),
        }
    }
}
