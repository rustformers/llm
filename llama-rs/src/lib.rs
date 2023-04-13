#![deny(missing_docs)]
//! LLaMA-rs is a Rust port of the llama.cpp project. This allows running inference for Facebook's LLaMA model on a CPU with good performance using full precision, f16 or 4-bit quantized versions of the model.

use std::{collections::HashMap, fmt::Display, path::Path};

use serde::Deserialize;
use thiserror::Error;

#[cfg(feature = "convert")]
pub mod convert;

mod inference_session;
mod loader;
mod util;
mod vocabulary;

pub use ggml::Type as ElementType;
pub use inference_session::{
    InferenceSession, InferenceSessionParameters, InferenceSnapshot, ModelKVMemoryType,
    SnapshotError,
};
pub use loader::{LoadError, LoadProgress};
pub use util::TokenUtf8Buffer;
pub use vocabulary::{TokenBias, Vocabulary};

use util::mulf;
use vocabulary::TokenId;

/// The end of text token.
pub const EOT_TOKEN_ID: TokenId = 2; // Hardcoded (for now?)

/// The hyperparameters of the model.
#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub struct Hyperparameters {
    n_vocab: usize,
    n_ctx: usize,
    n_embd: usize,
    n_mult: usize,
    n_head: usize,
    n_layer: usize,
    n_rot: usize,
    f16_: u32,
}

struct Layer {
    attention_norm: ggml::Tensor,

    wq: ggml::Tensor,
    wk: ggml::Tensor,
    wv: ggml::Tensor,
    wo: ggml::Tensor,

    // normalization
    ffn_norm: ggml::Tensor,

    // ff
    w1: ggml::Tensor,
    w2: ggml::Tensor,
    w3: ggml::Tensor,
}

/// The weights for the LLaMA model. All the mutable state is split into a
/// separate struct `InferenceSession`.
pub struct Model {
    hparams: Hyperparameters,

    tok_embeddings: ggml::Tensor,

    norm: ggml::Tensor,
    output: ggml::Tensor,

    layers: Vec<Layer>,

    tensors: HashMap<String, ggml::Tensor>,

    // Must be kept alive for the model
    _context: ggml::Context,
}

#[derive(Clone, Debug, PartialEq)]
/// The parameters that drive text generation.
pub struct InferenceParameters {
    /// The number of threads to use.
    pub n_threads: usize,
    /// [InferenceSession::feed_prompt] processes the prompt in batches of tokens.
    /// This controls how large an individual batch is.
    pub n_batch: usize,
    ///  Top-K: The top K words by score are kept during sampling.
    pub top_k: usize,
    /// Top-p: The cumulative probability after which no more words are kept for sampling.
    pub top_p: f32,
    /// The penalty for repeating tokens. Higher values make the generation less
    /// likely to get into a loop, but may harm results when repetitive outputs
    /// are desired.
    pub repeat_penalty: f32,
    /// Temperature used for sampling.
    pub temperature: f32,
    /// A list of tokens to bias against in the process of generation.
    pub bias_tokens: TokenBias,
    /// Whether or not previous tokens should be played back in [InferenceSession::inference_with_prompt].
    pub play_back_previous_tokens: bool,
}

impl Default for InferenceParameters {
    fn default() -> Self {
        Self {
            n_threads: 8,
            n_batch: 8,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.30,
            temperature: 0.80,
            bias_tokens: TokenBias::default(),
            play_back_previous_tokens: false,
        }
    }
}

/// Statistics about the inference process.
#[derive(Debug, Clone, Copy)]
pub struct InferenceStats {
    /// How long it took to feed the prompt.
    pub feed_prompt_duration: std::time::Duration,
    /// How many tokens the prompt was.
    pub prompt_tokens: usize,
    /// How long it took to predict new tokens.
    pub predict_duration: std::time::Duration,
    /// The number of predicted tokens.
    pub predict_tokens: usize,
}

impl Default for InferenceStats {
    fn default() -> Self {
        Self {
            feed_prompt_duration: std::time::Duration::from_secs(0),
            prompt_tokens: 0,
            predict_duration: std::time::Duration::from_secs(0),
            predict_tokens: 0,
        }
    }
}

impl Display for InferenceStats {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "feed_prompt_duration: {}ms\nprompt_tokens: {}\npredict_duration: {}ms\npredict_tokens: {}\nper_token_duration: {:.3}ms",
            self.feed_prompt_duration.as_millis(),
            self.prompt_tokens,
            self.predict_duration.as_millis(),
            self.predict_tokens,
            (self.predict_duration.as_millis() as f64) / (self.predict_tokens as f64),
        )
    }
}

#[derive(Error, Debug)]
/// Errors encountered during the inferencep rocess.
pub enum InferenceError {
    #[error("an invalid token was encountered during tokenization")]
    /// During tokenization, one of the produced tokens was invalid / zero.
    TokenizationFailed,
    #[error("the context window is full")]
    /// The context window for the model is full.
    ContextFull,
    #[error("reached end of text")]
    /// The model has produced an end of text token, signalling that it thinks that the text should end here.
    ///
    /// Note that this error *can* be ignored and inference can continue, but the results are not guaranteed to be sensical.
    EndOfText,
    #[error("the user-specified callback returned an error")]
    /// The user-specified callback returned an error.
    UserCallback(Box<dyn std::error::Error>),
}

/// Used in a call to `evaluate` to request information from the transformer.
#[derive(Default, Debug, Clone)]
pub struct EvaluateOutputRequest {
    /// Returns all the logits for the provided batch of tokens.
    /// Output shape is `n_batch * n_vocab`.
    pub all_logits: Option<Vec<f32>>,
    /// Returns the embeddings for the provided batch of tokens
    /// Output shape is `n_batch * n_embd`.
    pub embeddings: Option<Vec<f32>>,
}

impl Model {
    /// Load the model from `path` with `n_context_tokens` context tokens.
    ///
    /// The status of the loading process will be reported through `load_progress_callback`.
    pub fn load(
        path: impl AsRef<Path>,
        n_context_tokens: usize,
        load_progress_callback: impl FnMut(LoadProgress),
    ) -> Result<(Model, Vocabulary), LoadError> {
        loader::load(path, n_context_tokens, load_progress_callback)
    }

    /// Starts a new `InferenceSession` for this model.
    pub fn start_session(&self, params: InferenceSessionParameters) -> InferenceSession {
        let Hyperparameters {
            n_ctx,
            n_embd,
            n_layer,
            n_vocab,
            ..
        } = self.hparams;

        let ctx_size = {
            let mut ctx_size = 0;
            ctx_size += mulf!(
                n_ctx,
                n_layer,
                n_embd,
                ggml::type_sizef(params.memory_k_type.into())
            ); // memory_k
            ctx_size += mulf!(
                n_ctx,
                n_layer,
                n_embd,
                ggml::type_sizef(params.memory_v_type.into())
            ); // memory_v
            ctx_size += (5 + 10 * n_layer) * 256; // object overhead
            ctx_size
        };

        let session_ctx = ggml::Context::init(ctx_size);

        // Initialize key + value memory tensors
        let n_mem = n_layer * n_ctx;
        let n_elements = n_embd * n_mem;
        let memory_k = session_ctx.new_tensor_1d(params.memory_k_type.into(), n_elements);
        let memory_v = session_ctx.new_tensor_1d(params.memory_v_type.into(), n_elements);

        InferenceSession {
            _session_ctx: session_ctx,
            memory_size: ctx_size,
            params,
            memory_k,
            memory_v,
            n_past: 0,
            mem_per_token: 0,
            tokens: vec![],
            last_logits: vec![0.0; n_vocab],
            scratch: inference_session::scratch_buffers(),
        }
    }

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
    ) {
        let n = input_tokens.len();
        let n_past = session.n_past;
        let n_threads = params.n_threads;

        let memk_elsize = session.memory_k.element_size();
        let memv_elsize = session.memory_v.element_size();

        let Hyperparameters {
            n_vocab,
            n_ctx,
            n_embd,
            n_mult: _,
            n_head,
            n_layer,
            n_rot,
            f16_: _,
        } = self.hparams;

        // For the first run, we need to guess a maximum buffer size so we can measure
        // the actual memory consumption of the temporary ggml context.
        //
        // These numbers are from `llama.cpp`, and could potentially be more efficient.
        let mut buf_size = {
            let buf_size_mb = if n_layer >= 80 {
                1536
            } else if n_layer >= 60 {
                1280
            } else {
                1024
            };
            buf_size_mb * 1024 * 1024
        };
        if session.mem_per_token > 0 && session.mem_per_token * n > buf_size {
            // add 10% to account for ggml object overhead
            buf_size = (1.1f64 * session.mem_per_token as f64 * n as f64) as usize;
        };
        let ctx0 = ggml::Context::init(buf_size);

        let mut gf = ggml::ComputationGraph::new(n_threads);

        let embd = ctx0.new_tensor_1d(ggml::Type::I32, n);
        unsafe { embd.write_data(bytemuck::cast_slice(input_tokens)) };

        let mut input_layer = ctx0.op_get_rows(&self.tok_embeddings, &embd);

        for il in 0..n_layer {
            let input_self_attention = input_layer.share();
            let mut current: ggml::Tensor;

            ctx0.use_scratch(Some(&mut session.scratch[0]));

            // norm
            {
                current = ctx0.op_rms_norm(&input_layer);

                // cur = attention_norm * cur
                current = ctx0.op_mul(
                    &ctx0.op_repeat(&self.layers[il].attention_norm, &current),
                    &current,
                );
            }

            // self-attention
            {
                // compute Q and K and RoPE them
                let q_current = ctx0.op_rope(
                    &ctx0.op_reshape_3d(
                        &ctx0.op_mul_mat(&self.layers[il].wq, &current),
                        n_embd / n_head,
                        n_head,
                        n,
                    ),
                    n_past,
                    n_rot,
                    0,
                );
                let k_current = ctx0.op_rope(
                    &ctx0.op_reshape_3d(
                        &ctx0.op_mul_mat(&self.layers[il].wk, &current),
                        n_embd / n_head,
                        n_head,
                        n,
                    ),
                    n_past,
                    n_rot,
                    0,
                );

                // store key and value to memory
                {
                    // compute the transposed [N, n_embd] V matrix
                    let v_current = ctx0.op_transpose(&ctx0.op_reshape_2d(
                        &ctx0.op_mul_mat(&self.layers[il].wv, &current),
                        n_embd,
                        n,
                    ));

                    let k = ctx0.op_view_1d(
                        &session.memory_k,
                        n * n_embd,
                        (memk_elsize * n_embd) * (il * n_ctx + n_past),
                    );

                    let v = ctx0.op_view_2d(
                        &session.memory_v,
                        n,
                        n_embd,
                        n_ctx * memv_elsize,
                        (il * n_ctx) * memv_elsize * n_embd + n_past * memv_elsize,
                    );

                    // important: storing RoPE-ed version of K in the KV cache!
                    gf.build_forward_expand(&ctx0.op_cpy(&k_current, &k));
                    gf.build_forward_expand(&ctx0.op_cpy(&v_current, &v));
                }

                let q = ctx0.op_permute(&q_current, 0, 2, 1, 3);

                let k = ctx0.op_permute(
                    &ctx0.op_reshape_3d(
                        &ctx0.op_view_1d(
                            &session.memory_k,
                            (n_past + n) * n_embd,
                            il * n_ctx * memk_elsize * n_embd,
                        ),
                        n_embd / n_head,
                        n_head,
                        n_past + n,
                    ),
                    0,
                    2,
                    1,
                    3,
                );

                // K * Q
                let k_q = ctx0.op_mul_mat(&k, &q);

                // KQ_scaled = KQ / sqrt(n_embd/n_head)
                let k_q_scaled = ctx0.op_scale(
                    &k_q,
                    &ctx0.new_f32(1.0 / f32::sqrt(n_embd as f32 / n_head as f32)),
                );

                // KQ_masked = mask_past(KQ_scaled)
                let k_q_masked = ctx0.op_diag_mask_inf(&k_q_scaled, n_past);

                // KQ = soft_max(KQ_masked)
                let k_q_soft_max = ctx0.op_soft_max(&k_q_masked);

                // split cached V into n_head heads
                let v = ctx0.op_view_3d(
                    &session.memory_v,
                    n_past + n,
                    n_embd / n_head,
                    n_head,
                    n_ctx * memv_elsize,
                    n_ctx * memv_elsize * n_embd / n_head,
                    il * n_ctx * memv_elsize * n_embd,
                );

                let k_q_v = ctx0.op_mul_mat(&v, &k_q_soft_max);

                // KQV_merged = KQV.permute(0, 2, 1, 3)
                let k_q_v_merged = ctx0.op_permute(&k_q_v, 0, 2, 1, 3);

                // cur = KQV_merged.contiguous().view(n_embd, N)
                current = ctx0.op_cpy(
                    &k_q_v_merged,
                    &ctx0.new_tensor_2d(ggml::Type::F32, n_embd, n),
                );

                // projection (no bias)
                current = ctx0.op_mul_mat(&self.layers[il].wo, &current);
            }

            ctx0.use_scratch(Some(&mut session.scratch[1]));

            let input_feed_forward = ctx0.op_add(&current, &input_self_attention);

            // feed-forward network
            {
                // norm
                {
                    current = ctx0.op_rms_norm(&input_feed_forward);

                    // cur = ffn_norm*cur
                    current = ctx0.op_mul(
                        &ctx0.op_repeat(&self.layers[il].ffn_norm, &current),
                        &current,
                    );
                }

                let tmp = ctx0.op_mul_mat(&self.layers[il].w3, &current);

                current = ctx0.op_mul_mat(&self.layers[il].w1, &current);

                // SILU activation
                current = ctx0.op_silu(&current);

                current = ctx0.op_mul(&current, &tmp);

                current = ctx0.op_mul_mat(&self.layers[il].w2, &current);
            }

            current = ctx0.op_add(&current, &input_feed_forward);

            // input for next layer
            input_layer = current;
        }

        ctx0.use_scratch(Some(&mut session.scratch[0]));

        // Used at the end to optionally extract the embeddings.
        let embeddings_tensor;

        // norm
        {
            input_layer = ctx0.op_rms_norm(&input_layer);

            // inpL = norm*inpL
            input_layer = ctx0.op_mul(&ctx0.op_repeat(&self.norm, &input_layer), &input_layer);
            embeddings_tensor = input_layer.share();
        }

        // lm_head
        {
            input_layer = ctx0.op_mul_mat(&self.output, &input_layer);
        }

        ctx0.use_scratch(None);

        // logits -> probs
        // inpL = ctx0.op_soft_max(&inpL);

        // run the computation
        gf.build_forward_expand(&input_layer);
        ctx0.graph_compute(&mut gf);

        // return result for just the last token
        // SAFETY: yolo
        assert_eq!(session.last_logits.len(), n_vocab);
        unsafe {
            input_layer.read_data(
                n_vocab * (n - 1) * std::mem::size_of::<f32>(),
                bytemuck::cast_slice_mut(&mut session.last_logits),
            )
        };

        // Extract logits
        if let Some(all_logits) = &mut output_request.all_logits {
            all_logits.resize(n_vocab * n, 0.0);
            // SAFETY: Tensor data can be read (properly aligned, initialized,
            // data will not be mutated or otherwise aliased during the copy),
            // and we're not reading past the end of the tensor data.
            assert_eq!(input_layer.nelements(), n_vocab * n);
            unsafe {
                input_layer.read_data(0, bytemuck::cast_slice_mut(all_logits));
            }
        }

        // Extract embeddings
        if let Some(embeddings) = &mut output_request.embeddings {
            embeddings.resize(n_embd * n, 0.0);
            // SAFETY: Same rationale as for the "Extract logits" section applies.
            assert_eq!(embeddings_tensor.nelements(), n_embd * n);
            unsafe {
                embeddings_tensor.read_data(0, bytemuck::cast_slice_mut(embeddings));
            }
        }

        // Adjust the required memory per token if we didn't know that already
        if session.mem_per_token == 0 {
            session.mem_per_token = ctx0.used_mem() / n;
        }

        // Adjust n_past to new length.
        session.n_past += input_tokens.len();
    }

    /// Hydrates a previously obtained InferenceSnapshot for this model.
    pub fn session_from_snapshot(
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
