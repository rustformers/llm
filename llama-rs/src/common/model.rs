use super::inference::{
    InferenceError, InferenceParameters, InferenceSession, InferenceSessionParameters,
    InferenceSnapshot, SnapshotError,
};
use super::load::{LoadError, LoadProgress};
use super::token::TokenId;
use super::vocabulary::Vocabulary;
use partial_sort::PartialSort;
use rand::{distributions::WeightedIndex, prelude::Distribution};
use std::path::Path;

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
    type Weights;
    type HP;

    fn load(
        path: impl AsRef<Path>,
        n_ctx: usize,
        load_progress_callback: impl Fn(LoadProgress<Self::HP>),
    ) -> Result<(Self::Weights, Vocabulary), LoadError>;

    /// Starts a new `InferenceSession` for this model.
    fn start_session(&self, params: InferenceSessionParameters) -> InferenceSession;

    fn sample_top_p_top_k(
        &self,
        session: &InferenceSession,
        params: &InferenceParameters,
        rng: &mut impl rand::Rng,
    ) -> TokenId {
        let logits = &session.last_logits;
        let n_logits = logits.len();
        let mut logits_id = Vec::<(f32, TokenId)>::with_capacity(n_logits);

        {
            let scale = 1.0 / params.temp;
            for (i, &logit) in logits.iter().enumerate() {
                let tid = i as TokenId;

                // repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
                // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main
                let val = if let Some(logit_override) = params.bias_tokens.get(tid) {
                    logit_override
                } else if session
                    .repetition_penalty_tokens()
                    .contains(&(i as TokenId))
                {
                    // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                    if logits[i] < 0.0 {
                        logit * scale * params.repeat_penalty
                    } else {
                        logit * scale / params.repeat_penalty
                    }
                } else {
                    logit * scale
                };
                logits_id.push((val, tid));
            }
        }

        // find the top K tokens
        {
            logits_id.partial_sort(params.top_k, |a, b| {
                // Sort descending
                b.0.total_cmp(&a.0)
            });
            logits_id.truncate(params.top_k);
        }

        let maxl = logits_id
            .iter()
            .map(|x| x.0)
            .max_by(f32::total_cmp)
            .unwrap();

        // compute probs for the top K tokens
        let mut probs: Vec<f32> = logits_id
            .iter()
            .copied()
            .map(|(k, _)| (k - maxl).exp())
            .collect();
        let sum: f32 = probs.iter().copied().sum();

        // Normalize the probs
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // Top p sampling
        if params.top_p < 1.0 {
            let mut cumsum = 0.0;
            for i in 0..probs.len() {
                cumsum += probs[i];
                if cumsum >= params.top_p {
                    probs.truncate(i + 1);
                    logits_id.truncate(i + 1);
                    break;
                }
            }

            cumsum = 1.0 / cumsum;
            for p in probs.iter_mut() {
                *p *= cumsum;
            }
        }

        let dist = WeightedIndex::new(&probs).expect("WeightedIndex error");
        let idx = dist.sample(rng);

        logits_id[idx].1
    }

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

    fn tokenize(
        &self,
        vocab: &Vocabulary,
        text: &str,
        bos: bool,
    ) -> Result<Vec<TokenId>, InferenceError> {
        Ok(vocab
            .tokenize(text, bos)?
            .iter()
            .map(|(_, tid)| *tid)
            .collect::<Vec<TokenId>>())
    }

    /// Hydrates a previously obtained InferenceSnapshot for this model
    fn session_from_snapshot(
        &self,
        snapshot: InferenceSnapshot,
    ) -> Result<InferenceSession, SnapshotError> {
        let mut session = self.start_session(snapshot.session_params);

        if session.memory_k.nbytes() != snapshot.memory_k.len()
            || session.memory_v.nbytes() != snapshot.memory_v.len()
        {
            return Err(SnapshotError::MemorySizeMismatch {
                self_size: session.memory_k.nbytes() + session.memory_v.nbytes(),
                input_size: snapshot.memory_k.len() + snapshot.memory_v.len(),
            });
        }

        // SAFETY: We have exclusive access to Session, which means no one else
        // should be touching the context's memory. We can write to it because
        // we already checked the size.
        unsafe {
            session.memory_k.write_data(&snapshot.memory_k);
            session.memory_v.write_data(&snapshot.memory_v);
        }

        session.n_past = snapshot.npast;
        session.tokens = snapshot.tokens;
        session.last_logits = snapshot.last_logits;

        Ok(session)
    }
}
