use crate::{
    vocabulary::TokenId, EvaluateOutputRequest, InferenceParameters, InferenceSession,
    InferenceSessionParameters, Vocabulary,
};

/// A large language model.
pub trait Model {
    /// The model type.
    type Model;
    /// Hyperparameters for the model
    type Hyperparameters;
    /// Layer for the model
    type Layer;

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

    /// Model vocabulary
    fn vocabulary(&self) -> &Vocabulary;

    /// Model context size
    fn n_ctx(&self) -> usize;
}
