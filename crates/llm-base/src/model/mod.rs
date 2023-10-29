//! Large language model traits and types

use std::{fmt::Debug, path::PathBuf, sync::Arc};

use ggml::{
    accelerator::Backend,
    format::gguf::{Gguf, MetadataError},
    sys::llama::llama_ftype,
};
use regex::Regex;
use thiserror::Error;

use crate::{
    loader::{ModelTensorLoader, TensorLoadError},
    tokenizer::TokenId,
    InferenceSession, InferenceSessionConfig, Tokenizer,
};

/// Common functions for model evaluation
pub mod common;

/// All of the arguments required to load a model.
pub struct ModelLoadArgs<'a> {
    /// The GGUF metadata for the model.
    pub gguf: &'a Gguf,
    /// Model metadata.
    pub data: ModelData,
    /// The tensor loader to use for the model.
    pub tensor_loader: ModelTensorLoader<'a>,
}

/// Model data that is required for all models.
pub struct ModelData {
    /// Any parameters that control the behaviour of the model.
    pub params: ModelParameters,
    /// The tokenizer to use for the model.
    pub tokenizer: Tokenizer,
}

/// An error encountered while loading a concrete model.
#[derive(Error, Debug)]
pub enum ModelLoadError {
    /// An error occurred while loading the model's tensors.
    #[error("{0}")]
    TensorLoadError(#[from] TensorLoadError),
    /// An error occurred while reading the model's hyperparameters.
    #[error("{0}")]
    HyperparametersReadError(#[from] HyperparametersReadError),
}

/// Interfaces for creating and interacting with a large language model with a known type
/// of [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)).
pub trait Model: Send + Sync {
    /// Creates a new model from the provided [ModelParameters] hyperparameters.
    /// This function is called by the [load](crate::loader::load) function.
    fn new(args: ModelLoadArgs) -> Result<Self, ModelLoadError>
    where
        Self: Sized;

    /// Starts a new `InferenceSession` for this model.
    fn start_session(&self, config: InferenceSessionConfig) -> InferenceSession;

    /// This function is called by the provided [InferenceSession]; it will use this model
    /// to generate output by evaluating the `input_tokens`.
    /// The [OutputRequest] is used to specify additional data to fetch from the
    /// model.
    fn evaluate(
        &self,
        session: &mut InferenceSession,
        input_tokens: &[TokenId],
        output_request: &mut OutputRequest,
    );

    /// Get the data for this model.
    fn data(&self) -> &ModelData;

    /// Get the tokenizer for this model.
    fn tokenizer(&self) -> &Tokenizer {
        &self.data().tokenizer
    }

    /// Get the context size (configured with [ModelParameters::context_size]) used by
    /// this model.
    fn context_size(&self) -> usize {
        self.data().params.context_size
    }

    /// Get the beginning of text/beginning of string token ID, if available. This value is defined by model implementers.
    fn bot_token_id(&self) -> Option<TokenId>;

    /// Get the end of text/end of string token ID. This value is defined by model implementers.
    fn eot_token_id(&self) -> TokenId;

    /// Get the list of regexes to use to determine if a tensor in this model should be quantized.
    fn quantize_tensors(&self) -> Vec<Regex>;

    /// Get the list of regexes to use to determine if a tensor in this model should not be quantized.
    fn skip_quantize_tensors(&self) -> Vec<Regex>;

    /// Returns whether the model supports deleting tokens.
    fn supports_rewind(&self) -> bool {
        // Assume we can't delete unless otherwise specified
        false
    }
}

#[derive(Error, Debug)]
/// Reported from functions that read hyperparameters
pub enum HyperparametersReadError {
    #[error("{0}")]
    /// A metadata error.
    MetadataError(#[from] MetadataError),
    /// The file type within the model was not supported by this version of `llm`.
    #[error("file type {file_type} is not supported")]
    UnsupportedFileType {
        /// The file type (ignoring the quantization version) that was encountered.
        file_type: llama_ftype,
    },
}

/// Parameters for model-wide behaviour.
#[derive(Debug, Clone)]
pub struct ModelParameters {
    /// For [GGML formats](ggml::format::ContainerType) that support it, [mmap](https://en.wikipedia.org/wiki/Mmap)
    /// is the default. Although mmap typically improves performance, setting this value to `false` may
    /// be preferred in resource-constrained environments.
    pub prefer_mmap: bool,
    /// The context size ("memory") the model should use when evaluating a prompt. A larger context
    /// consumes more resources, but produces more consistent and coherent responses.
    pub context_size: usize,
    /// The [LoRA](https://arxiv.org/abs/2106.09685) adapters to use when loading the model. If `None`, no adapters will be used.
    pub lora_adapters: Option<Vec<PathBuf>>,
    /// Whether to use GPU acceleration when available
    pub use_gpu: bool,
    /// If `use_gpu` is active this defines the number of layers to offload to the gpu. If `None`, all layers will be offloaded.
    pub gpu_layers: Option<usize>,
    /// The arguments/overrides to pass to the [custom RoPE](https://arxiv.org/pdf/2306.15595.pdf) function, if it is used by the model.
    pub rope_overrides: Option<ggml::RoPEOverrides>,
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self {
            prefer_mmap: true,
            context_size: 2048,
            lora_adapters: None,
            use_gpu: false,
            gpu_layers: None,
            rope_overrides: None,
        }
    }
}

impl ModelParameters {
    /// Returns true if the model should offload the given layer to the accelerator.
    pub fn should_offload(&self, layer: usize) -> bool {
        if !self.use_gpu {
            return false;
        }

        self.gpu_layers
            .map(|gpu_layers| layer < gpu_layers)
            .unwrap_or(true)
    }

    /// Returns the backend to use for the given layer.
    pub fn backend(&self, layer: usize) -> Backend {
        if self.should_offload(layer) {
            Backend::Gpu
        } else {
            Backend::Cpu
        }
    }
}

/// Used in a call to [Model::evaluate] or [InferenceSession::infer] to request
/// information from the model. If a value is set to `Some`, the `Vec` will be
/// cleared, resized, and filled with the related data.
#[derive(Default, Debug, PartialEq, Clone)]
pub struct OutputRequest {
    /// Returns all the logits for evaluation. A logit represents the likelihood
    /// that a given token will be generated based on the tokens that have been
    /// evaluated or generated so far. Output shape is `n_batch * n_vocab`.
    pub all_logits: Option<Vec<f32>>,
    /// Returns all the embeddings for an evaluation. An embedding is a vector
    /// that measures the relatedness of text strings. Output shape is
    /// `n_batch * n_embd`.
    pub embeddings: Option<Vec<f32>>,
}

/// Contains the GGML context for a [`Model`]. Implements `Send` and `Sync`
/// to allow for the free transfer of models; this is made possible by this
/// context being effectively inert after creation, so that it cannot be
/// modified across threads.
#[derive(Clone)]
#[allow(clippy::arc_with_non_send_sync)]
pub struct ModelContext(pub(crate) Arc<ggml::Context>);
unsafe impl Send for ModelContext {}
unsafe impl Sync for ModelContext {}
