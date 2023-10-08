//! Large language model traits and types

use std::{fmt::Debug, path::PathBuf, sync::Arc};

use ggml::{
    accelerator::Backend,
    format::gguf::{Metadata, MetadataError},
    sys::llama::llama_ftype,
};
use regex::Regex;
use thiserror::Error;

use crate::{
    loader::{ModelTensorLoader, TensorLoadError},
    tokenizer::TokenId,
    FileType, InferenceSession, InferenceSessionConfig, Tokenizer,
};

/// Common functions for model evaluation
pub mod common;

/// Interfaces for creating and interacting with a large language model with a known type
/// of [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)).
pub trait KnownModel: Send + Sync {
    /// Hyperparameters for the model.
    type Hyperparameters: Hyperparameters;

    /// Creates a new model from the provided [ModelParameters] hyperparameters.
    /// This function is called by the [load](crate::loader::load) function.
    fn new(
        hyperparameters: Self::Hyperparameters,
        params: ModelParameters,
        tokenizer: Tokenizer,
        tensor_loader: ModelTensorLoader,
    ) -> Result<Self, TensorLoadError>
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

    /// Get the hyperparameters for this model.
    fn hyperparameters(&self) -> &Self::Hyperparameters;

    /// Get the tokenizer for this model.
    fn tokenizer(&self) -> &Tokenizer;

    /// Get the context size (configured with [ModelParameters::context_size]) used by
    /// this model.
    fn context_size(&self) -> usize;

    /// Get the beginning of text/beginning of string token ID, if available. This value is defined by model implementers.
    fn bot_token_id(&self) -> Option<TokenId>;

    /// Get the end of text/end of string token ID. This value is defined by model implementers.
    fn eot_token_id(&self) -> TokenId;

    /// Get the list of regexes to use to determine if a tensor in this model should be quantized.
    fn quantize_tensors() -> Vec<Regex>;

    /// Get the list of regexes to use to determine if a tensor in this model should not be quantized.
    fn skip_quantize_tensors() -> Vec<Regex>;

    /// Returns whether the model supports deleting tokens.
    fn supports_rewind(&self) -> bool {
        // Assume we can't delete unless otherwise specified
        false
    }
}

/// A type-erased model to allow for interacting with a model without knowing
/// its hyperparameters.
pub trait Model: Send + Sync {
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

    /// Get the tokenizer for this model.
    fn tokenizer(&self) -> &Tokenizer;

    /// Get the context size (configured with [ModelParameters::context_size]) used by
    /// this model.
    fn context_size(&self) -> usize;

    /// Get the beginning of text/beginning of string token ID, if available. This value is defined by model implementers.
    fn bot_token_id(&self) -> Option<TokenId>;

    /// Get the end of text/end of string token ID. This value is defined by model implementers.
    fn eot_token_id(&self) -> TokenId;

    /// Returns whether the model supports deleting tokens.
    fn supports_rewind(&self) -> bool;
}
impl<H: Hyperparameters, M: KnownModel<Hyperparameters = H>> Model for M {
    fn start_session(&self, config: InferenceSessionConfig) -> InferenceSession {
        KnownModel::start_session(self, config)
    }

    fn evaluate(
        &self,
        session: &mut InferenceSession,
        input_tokens: &[TokenId],
        output_request: &mut OutputRequest,
    ) {
        KnownModel::evaluate(self, session, input_tokens, output_request)
    }

    fn tokenizer(&self) -> &Tokenizer {
        KnownModel::tokenizer(self)
    }

    fn context_size(&self) -> usize {
        KnownModel::context_size(self)
    }

    fn bot_token_id(&self) -> Option<TokenId> {
        KnownModel::bot_token_id(self)
    }

    fn eot_token_id(&self) -> TokenId {
        KnownModel::eot_token_id(self)
    }

    fn supports_rewind(&self) -> bool {
        KnownModel::supports_rewind(self)
    }
}

/// Implemented by model hyperparameters for interacting with hyperparameters
/// without knowing what they are, as well as writing/reading them as required.
pub trait Hyperparameters: Sized + Default + Debug + PartialEq + Eq {
    /// Read the parameters from GGUF metadata.
    fn read_gguf(metadata: &Metadata) -> Result<Self, HyperparametersReadError>;

    /// Write the parameters to GGUF metadata.
    fn write_gguf(&self, metadata: &mut Metadata) -> Result<(), HyperparametersWriteError>;

    /// Get the filetype of the model.
    fn file_type(&self) -> Option<FileType>;

    /// Get mutable access to filetype of the model.
    fn file_type_mut(&mut self) -> Option<&mut FileType>;
}
#[derive(Error, Debug)]
/// Reported from functions that write
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
#[derive(Error, Debug)]
/// Reported from functions that write
pub enum HyperparametersWriteError {
    #[error("non-specific I/O error")]
    /// A non-specific IO error.
    Io(#[from] std::io::Error),
    #[error("invalid integer conversion")]
    /// One of the integers encountered could not be converted to a more appropriate type.
    InvalidIntegerConversion(#[from] std::num::TryFromIntError),
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
