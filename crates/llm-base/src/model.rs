use std::{
    error::Error,
    io::{BufRead, Write},
};

use crate::{
    loader::TensorLoader, vocabulary::TokenId, EvaluateOutputRequest, InferenceParameters,
    InferenceSession, InferenceSessionParameters, LoadError, Vocabulary,
};

/// A large language model.
pub trait KnownModel: Send + Sync {
    /// Hyperparameters for the model
    type Hyperparameters: Hyperparameters;

    /// Creates a new model from the provided hyperparameters.
    fn new<E: Error>(
        hyperparameters: Self::Hyperparameters,
        n_context_tokens: usize,
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
}

/// A type-erased model to allow for interacting with a model without knowing
/// its hyperparameters.
pub trait Model {
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
}

/// Implemented by model hyperparameters for loading and saving to a GGML model read/writer.
pub trait Hyperparameters: Sized + Default {
    /// The error type returned during a failure of [Self::write].
    type WriteError: Error + Send + Sync + 'static;

    /// Read the parameters from a reader.
    fn read(reader: &mut dyn BufRead) -> Result<Self, LoadError>;

    /// Write the parameters to a writer.
    fn write(&self, writer: &mut dyn Write) -> Result<(), Self::WriteError>;

    /// Get the number of tokens in the vocabulary.
    fn n_vocabulary(&self) -> usize;
}
