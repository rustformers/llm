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
mod tokenizer;

pub mod model;
pub mod samplers;
pub mod util;

use std::sync::{Arc, Mutex};

pub use ggml;
pub use ggml::Type as ElementType;

pub use inference_session::{
    conversation_inference_callback, feed_prompt_callback, GraphOutputs, InferenceError,
    InferenceFeedback, InferenceRequest, InferenceResponse, InferenceSession,
    InferenceSessionConfig, InferenceSnapshot, InferenceSnapshotRef, InferenceStats,
    ModelKVMemoryType, RewindError, SnapshotError,
};
pub use llm_samplers::prelude::{Sampler, SamplerChain};
pub use loader::{
    load, load_progress_callback_stdout, ContainerType, FileType, FileTypeFormat, FormatMagic,
    LoadError, LoadProgress, Loader, TensorLoader,
};
pub use lora::{LoraAdapter, LoraParameters};
pub use memmap2::Mmap;
pub use model::{Hyperparameters, KnownModel, Model, ModelParameters, OutputRequest};
pub use quantize::{quantize, QuantizeError, QuantizeProgress};
pub use regex::Regex;
pub use tokenizer::{
    InvalidTokenBias, Prompt, TokenBias, TokenId, TokenizationError, Tokenizer, TokenizerLoadError,
    TokenizerSource,
};
pub use util::TokenUtf8Buffer;

#[derive(Clone, Debug)]
/// The parameters for text generation.
///
/// This needs to be provided during all inference calls,
/// but can be changed between calls.
pub struct InferenceParameters {
    /// The sampler to use for sampling tokens from the model's probabilities.
    ///
    /// Each time the model runs, it generates a distribution of probabilities; each token
    /// has a probability of being the next token. The sampler is responsible for sampling
    /// from this distribution to generate the next token. Using a different sampler may
    /// change the output of the model, or control how deterministic the generated text is.
    ///
    /// This can be anything that implements [Sampler]. Refer to
    /// the `llm-samplers` documentation for possible samplers and suggested
    /// combinations: <https://docs.rs/llm-samplers>
    pub sampler: Arc<Mutex<dyn Sampler<TokenId, f32>>>,
}

//Since Sampler implements Send and Sync, InferenceParameters should too.
unsafe impl Send for InferenceParameters {}
unsafe impl Sync for InferenceParameters {}

impl Default for InferenceParameters {
    fn default() -> Self {
        Self {
            sampler: samplers::default_samplers(),
        }
    }
}
