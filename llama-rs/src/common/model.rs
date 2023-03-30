


/// Used in a call to `evaluate` to request information from the transformer.
#[derive(Default)]
pub struct EvaluateOutputRequest {
    /// Returns all the logits for the provided batch of tokens.
    /// Output shape is n_batch * n_vocab
    pub all_logits: Option<Vec<f32>>,
    /// Returns the embeddings for the provided batch of tokens
    /// Output shape is n_batch * n_embd
    pub embeddings: Option<Vec<f32>>,
}

pub trait Model {
    type Error;

    pub fn load(
        path: impl AsRef<Path>,
        n_ctx: i32,
        load_progress_callback: impl Fn(LoadProgress),
    ) -> Result<(Model, Vocabulary), LoadError>;


    pub fn start_session(&self, params: InferenceSessionParameters) -> InferenceSession;


    pub fn sample_top_p_top_k(
        &self,
        session: &InferenceSession,
        params: &InferenceParameters,
        rng: &mut impl rand::Rng,
    ) -> TokenId;

    pub fn sample_top_p_top_k(
        &self,
        session: &InferenceSession,
        params: &InferenceParameters,
        rng: &mut impl rand::Rng,
    ) -> TokenId;

    /// Evaluates the transformer.
    ///
    /// The provided `output_request` struct lets you specify which additional
    /// data you are interested in fetching from the transformer. Setting a
    /// field to a `Some` value will clear and fill the provided vector with
    /// data. The provided vector will be resized to the exact output size.
    pub fn evaluate(
        &self,
        session: &mut InferenceSession,
        params: &InferenceParameters,
        input_tokens: &[TokenId],
        output_request: &mut EvaluateOutputRequest,
    );

    pub fn tokenize(
        &self,
        vocab: &Vocabulary,
        text: &str,
        bos: bool,
    ) -> Result<Vec<TokenId>, InferenceError>;

    pub fn session_from_snapshot(
        &self,
        snapshot: InferenceSnapshot,
    ) -> Result<InferenceSession, SnapshotError>;

}
