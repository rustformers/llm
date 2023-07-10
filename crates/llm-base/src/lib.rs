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

use std::sync::Arc;

pub use ggml;
pub use ggml::Type as ElementType;

pub use inference_session::{
    feed_prompt_callback, GraphOutputs, InferenceError, InferenceFeedback, InferenceRequest,
    InferenceResponse, InferenceSession, InferenceSessionConfig, InferenceSnapshot,
    InferenceSnapshotRef, InferenceStats, ModelKVMemoryType, RewindError, SnapshotError,
};
pub use loader::{
    load, load_progress_callback_stdout, ContainerType, FileType, FileTypeFormat, FormatMagic,
    LoadError, LoadProgress, Loader, TensorLoader,
};
pub use lora::{LoraAdapter, LoraParameters};
pub use memmap2::Mmap;
pub use model::{Hyperparameters, KnownModel, Model, ModelParameters, OutputRequest};
pub use quantize::{quantize, QuantizeError, QuantizeProgress};
pub use regex::Regex;
pub use samplers::Sampler;
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
    /// The sampler to use for sampling tokens from the model's probabilities.
    ///
    /// Each time the model runs, it generates a distribution of probabilities; each token
    /// has a probability of being the next token. The sampler is responsible for sampling
    /// from this distribution to generate the next token. Using a different sampler may
    /// change the output of the model, or control how deterministic the generated text is.
    ///
    /// A recommended default sampler is [TopPTopK](samplers::TopPTopK), which is a standard
    /// sampler that offers a [Default](samplers::TopPTopK::default) implementation.
    pub sampler: Arc<dyn Sampler>,
}

//Since Sampler implements Send and Sync, InferenceParameters should too.
unsafe impl Send for InferenceParameters {}
unsafe impl Sync for InferenceParameters {}

impl Default for InferenceParameters {
    fn default() -> Self {
        Self {
            n_threads: 8,
            sampler: Arc::new(samplers::TopPTopK::default()),
        }
    }
}
