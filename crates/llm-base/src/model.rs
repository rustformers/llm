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

/// A large language model.
pub trait KnownModel: Send + Sync {
    /// Hyperparameters for the model
    type Hyperparameters: Hyperparameters;

    /// Creates a new model from the provided hyperparameters.
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

    /// Evaluates the transformer.
    ///
    /// The provided `output_request` struct lets you specify which additional
    /// data you are interested in fetching from the transformer. Setting a
    /// field to a `Some` value will clear and fill the provided vector with
    /// data. The provided vector will be resized to the exact output size.
    fn evaluate(
        &self,
        session: &mut InferenceSession,
        params: &InferenceParameters,
        input_tokens: &[TokenId],
        output_request: &mut EvaluateOutputRequest,
    );

    /// Get the vocabulary for this model.
    fn vocabulary(&self) -> &Vocabulary;

    /// Get the context size for this model.
    fn n_context_tokens(&self) -> usize;

    /// Get the end of text token ID.
    fn eot_token_id(&self) -> TokenId;

    /// Get the default InferenceSessionParameters to use with this model
    fn inference_params(&self) -> InferenceParameters;

    /// Get the default InferenceWithPromptParameters to use with this model
    fn inference_prompt_params(&self) -> InferenceWithPromptParameters;
}

/// A type-erased model to allow for interacting with a model without knowing
/// its hyperparameters.
pub trait Model: Send + Sync {
    /// Starts a new `InferenceSession` for this model.
    fn start_session(&self, params: InferenceSessionParameters) -> InferenceSession;

    /// Evaluates the transformer.
    ///
    /// The provided `output_request` struct lets you specify which additional
    /// data you are interested in fetching from the transformer. Setting a
    /// field to a `Some` value will clear and fill the provided vector with
    /// data. The provided vector will be resized to the exact output size.
    fn evaluate(
        &self,
        session: &mut InferenceSession,
        params: &InferenceParameters,
        input_tokens: &[TokenId],
        output_request: &mut EvaluateOutputRequest,
    );

    /// Get the vocabulary for this model.
    fn vocabulary(&self) -> &Vocabulary;

    /// Get the context size for this model.
    fn n_context_tokens(&self) -> usize;

    /// Get the end of text token ID.
    fn eot_token_id(&self) -> TokenId;

    /// Get the default InferenceSessionParameters to use with this model
    fn inference_params(&self) -> InferenceParameters;

    /// Get the default InferenceWithPromptParameters to use with this model
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
    /// Model context size
    pub n_context_tokens: usize,
    /// Default InferenceParameters to use with the model
    pub inference_params: InferenceParameters,
    /// Default InferenceWithPromptParameters to use with the model
    pub inference_prompt_params: InferenceWithPromptParameters,
}
