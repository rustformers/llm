//! Large language model traits and types

use std::{
    error::Error,
    fmt::Debug,
    io::{BufRead, Write},
};

use crate::{
    loader::TensorLoader, vocabulary::TokenId, EvaluateOutputRequest, InferenceParameters,
    InferenceSession, InferenceSessionParameters, InferenceWithPromptParameters, LoadError,
    Vocabulary,
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
    fn start_session(&self, params: InferenceSessionParameters) -> InferenceSession;

    /// This function is called by the provided [InferenceSession]; it will use this model
    /// and the [InferenceParameters] to generate output by evaluating the `input_tokens`.
    /// The [EvaluateOutputRequest] is used to specify additional data to fetch from the
    /// model. For more information, refer to [InferenceSession::infer_with_params]
    fn evaluate(
        &self,
        session: &mut InferenceSession,
        params: &InferenceParameters,
        input_tokens: &[TokenId],
        output_request: &mut EvaluateOutputRequest,
    );

    /// Get the vocabulary (loaded from the GGML file) for this model.
    fn vocabulary(&self) -> &Vocabulary;

    /// Get the context size (configured with [ModelParameters::n_context_tokens]) used by
    /// this model.
    fn n_context_tokens(&self) -> usize;

    /// Get the end of text token ID. This value is defined by model implementers.
    fn eot_token_id(&self) -> TokenId;

    /// Get the default [InferenceSessionParameters] for this model (used by
    /// [InferenceSession::infer]). This value is configured through
    /// [ModelParameters::inference_params].
    fn inference_params(&self) -> InferenceParameters;

    /// Get the default [InferenceWithPromptParameters] for this model (used by
    /// [InferenceSession::infer]). This value is configured through
    /// [ModelParameters::inference_prompt_params].
    fn inference_prompt_params(&self) -> InferenceWithPromptParameters;
}

/// A type-erased model to allow for interacting with a model without knowing
/// its hyperparameters.
pub trait Model: Send + Sync {
    /// Starts a new `InferenceSession` for this model.
    fn start_session(&self, params: InferenceSessionParameters) -> InferenceSession;

    /// This function is called by the provided [InferenceSession]; it will use this model
    /// and the [InferenceParameters] to generate output by evaluating the `input_tokens`.
    /// The [EvaluateOutputRequest] is used to specify additional data to fetch from the
    /// model. For more information, refer to [InferenceSession::infer_with_params]
    fn evaluate(
        &self,
        session: &mut InferenceSession,
        params: &InferenceParameters,
        input_tokens: &[TokenId],
        output_request: &mut EvaluateOutputRequest,
    );

    /// Get the vocabulary (loaded from the GGML file) for this model.
    fn vocabulary(&self) -> &Vocabulary;

    /// Get the context size (configured with [ModelParameters::n_context_tokens]) used by
    /// this model.
    fn n_context_tokens(&self) -> usize;

    /// Get the end of text token ID. This value is defined by model implementers.
    fn eot_token_id(&self) -> TokenId;

    /// Get the default [InferenceSessionParameters] for this model (used by
    /// [InferenceSession::infer]). This value is configured through
    /// [ModelParameters::inference_params].
    fn inference_params(&self) -> InferenceParameters;

    /// Get the default [InferenceWithPromptParameters] for this model (used by
    /// [InferenceSession::infer]). This value is configured through
    /// [ModelParameters::inference_prompt_params].
    fn inference_prompt_params(&self) -> InferenceWithPromptParameters;
}
impl<H: Hyperparameters, M: KnownModel<Hyperparameters = H>> Model for M {
    fn start_session(&self, params: InferenceSessionParameters) -> InferenceSession {
        KnownModel::start_session(self, params)
    }

    fn evaluate(
        &self,
        session: &mut InferenceSession,
        params: &InferenceParameters,
        input_tokens: &[TokenId],
        output_request: &mut EvaluateOutputRequest,
    ) {
        KnownModel::evaluate(self, session, params, input_tokens, output_request)
    }

    fn vocabulary(&self) -> &Vocabulary {
        KnownModel::vocabulary(self)
    }

    fn n_context_tokens(&self) -> usize {
        KnownModel::n_context_tokens(self)
    }

    fn eot_token_id(&self) -> TokenId {
        KnownModel::eot_token_id(self)
    }

    fn inference_params(&self) -> InferenceParameters {
        KnownModel::inference_params(self)
    }

    fn inference_prompt_params(&self) -> InferenceWithPromptParameters {
        KnownModel::inference_prompt_params(self)
    }
}

/// Implemented by model hyperparameters for loading and saving to a GGML model read/writer.
pub trait Hyperparameters: Sized + Default + Debug {
    /// The error type returned during a failure of [Self::write].
    type WriteError: Error + Send + Sync + 'static;

    /// Read the parameters from a reader.
    fn read(reader: &mut dyn BufRead) -> Result<Self, LoadError>;

    /// Write the parameters to a writer.
    fn write(&self, writer: &mut dyn Write) -> Result<(), Self::WriteError>;

    /// Get the number of tokens in the vocabulary.
    fn n_vocabulary(&self) -> usize;
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
    pub inference_params: InferenceParameters,
    /// Default InferenceWithPromptParameters to use when [evaluating](Model::evaluate) a prompt with this model.
    pub inference_prompt_params: InferenceWithPromptParameters,
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self {
            prefer_mmap: true,
            n_context_tokens: 2048,
            inference_params: Default::default(),
            inference_prompt_params: Default::default(),
        }
    }
}
