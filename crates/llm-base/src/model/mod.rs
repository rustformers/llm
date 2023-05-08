//! Large language model traits and types

use std::{
    error::Error,
    fmt::Debug,
    io::{BufRead, Write},
};

use thiserror::Error;

use crate::{
    loader::TensorLoader, vocabulary::TokenId, InferenceParameters, InferenceSession,
    InferenceSessionConfig, LoadError, Vocabulary,
};

/// Common functions for model evaluation
pub mod common;

/// Interfaces for creating and interacting with a large language model with a known type
/// of [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)).
pub trait KnownModel: Send + Sync {
    /// Hyperparameters for the model
    type Hyperparameters: Hyperparameters;

    /// Creates a new model from the provided [ModelParameters] hyperparameters.
    /// This function is called by the [load](crate::loader::load) function.
    fn new<E: Error>(
        hyperparameters: Self::Hyperparameters,
        params: ModelParameters,
        vocabulary: Vocabulary,
        tensor_loader: impl TensorLoader<E>,
    ) -> Result<Self, E>
    where
        Self: Sized;

    /// Starts a new `InferenceSession` for this model.
    fn start_session(&self, params: InferenceSessionConfig) -> InferenceSession;

    /// This function is called by the provided [InferenceSession]; it will use this model
    /// and the [InferenceParameters] to generate output by evaluating the `input_tokens`.
    /// The [OutputRequest] is used to specify additional data to fetch from the
    /// model.
    fn evaluate(
        &self,
        session: &mut InferenceSession,
        params: &InferenceParameters,
        input_tokens: &[TokenId],
        output_request: &mut OutputRequest,
    );

    /// Get the vocabulary (loaded from the GGML file) for this model.
    fn vocabulary(&self) -> &Vocabulary;

    /// Get the context size (configured with [ModelParameters::n_context_tokens]) used by
    /// this model.
    fn n_context_tokens(&self) -> usize;

    /// Get the beginning of text/beginning of string token ID, if available. This value is defined by model implementers.
    fn bot_token_id(&self) -> Option<TokenId>;

    /// Get the end of text/end of string token ID. This value is defined by model implementers.
    fn eot_token_id(&self) -> TokenId;

    /// Get the default [InferenceParameters] for this model (used by
    /// [InferenceSession::infer]). This value is configured through
    /// [ModelParameters::inference_parameters].
    fn inference_parameters(&self) -> &InferenceParameters;
}

/// A type-erased model to allow for interacting with a model without knowing
/// its hyperparameters.
pub trait Model: Send + Sync {
    /// Starts a new `InferenceSession` for this model.
    fn start_session(&self, params: InferenceSessionConfig) -> InferenceSession;

    /// This function is called by the provided [InferenceSession]; it will use this model
    /// and the [InferenceParameters] to generate output by evaluating the `input_tokens`.
    /// The [OutputRequest] is used to specify additional data to fetch from the
    /// model.
    fn evaluate(
        &self,
        session: &mut InferenceSession,
        params: &InferenceParameters,
        input_tokens: &[TokenId],
        output_request: &mut OutputRequest,
    );

    /// Get the vocabulary (loaded from the GGML file) for this model.
    fn vocabulary(&self) -> &Vocabulary;

    /// Get the context size (configured with [ModelParameters::n_context_tokens]) used by
    /// this model.
    fn n_context_tokens(&self) -> usize;

    /// Get the beginning of text/beginning of string token ID, if available. This value is defined by model implementers.
    fn bot_token_id(&self) -> Option<TokenId>;

    /// Get the end of text/end of string token ID. This value is defined by model implementers.
    fn eot_token_id(&self) -> TokenId;

    /// Get the default [InferenceParameters] for this model (used by
    /// [InferenceSession::infer]). This value is configured through
    /// [ModelParameters::inference_parameters].
    fn inference_parameters(&self) -> &InferenceParameters;
}
impl<H: Hyperparameters, M: KnownModel<Hyperparameters = H>> Model for M {
    fn start_session(&self, params: InferenceSessionConfig) -> InferenceSession {
        KnownModel::start_session(self, params)
    }

    fn evaluate(
        &self,
        session: &mut InferenceSession,
        params: &InferenceParameters,
        input_tokens: &[TokenId],
        output_request: &mut OutputRequest,
    ) {
        KnownModel::evaluate(self, session, params, input_tokens, output_request)
    }

    fn vocabulary(&self) -> &Vocabulary {
        KnownModel::vocabulary(self)
    }

    fn n_context_tokens(&self) -> usize {
        KnownModel::n_context_tokens(self)
    }

    fn bot_token_id(&self) -> Option<TokenId> {
        KnownModel::bot_token_id(self)
    }

    fn eot_token_id(&self) -> TokenId {
        KnownModel::eot_token_id(self)
    }

    fn inference_parameters(&self) -> &InferenceParameters {
        KnownModel::inference_parameters(self)
    }
}

/// Implemented by model hyperparameters for interacting with hyperparameters
/// without knowing what they are, as well as writing/reading them as required.
pub trait Hyperparameters: Sized + Default + Debug {
    /// Read the parameters in GGML format from a reader.
    fn read_ggml(reader: &mut dyn BufRead) -> Result<Self, LoadError>;

    /// Write the parameters in GGML format to a writer.
    fn write_ggml(&self, writer: &mut dyn Write) -> Result<(), HyperparametersWriteError>;

    /// Get the number of tokens in the vocabulary.
    fn n_vocabulary(&self) -> usize;
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

/// Parameters for tuning model instances
pub struct ModelParameters {
    /// For [GGML formats](ggml::ContainerType) that support it, [mmap](https://en.wikipedia.org/wiki/Mmap)
    /// is the default. Although mmap typically improves performance, setting this value to `false` may
    /// be preferred in resource-constrained environments.
    pub prefer_mmap: bool,
    /// The context size ("memory") the model should use when evaluating a prompt. A larger context
    /// consumes more resources, but produces more consistent and coherent responses.
    pub n_context_tokens: usize,
    /// Default InferenceParameters to use when [evaluating](Model::evaluate) a prompt with this model.
    pub inference_parameters: InferenceParameters,
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self {
            prefer_mmap: true,
            n_context_tokens: 2048,
            inference_parameters: Default::default(),
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
