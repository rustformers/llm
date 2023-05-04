//! An implementation of LLaMA (Large Language Model Meta AI) for the `llm` ecosystem.
#![deny(missing_docs)]

use std::{error::Error, path::Path};

use llm_base::{
    ggml, util, BasicWriteError, EvaluateOutputRequest, FileType, InferenceParameters,
    InferenceSession, InferenceSessionParameters, InferenceWithPromptParameters, KnownModel,
    LoadError, LoadProgress, Mmap, ModelParameters, TensorLoader, TokenId, Vocabulary,
};

#[cfg(feature = "convert")]
pub mod convert;

mod old_loader;

/// The LLaMA model.
///
/// # Safety
/// This implements [Send] and [Sync] as it is immutable after construction.
pub struct Llama {
    hyperparameters: Hyperparameters,
    n_context_tokens: usize,

    vocabulary: Vocabulary,

    tok_embeddings: ggml::Tensor,

    norm: ggml::Tensor,
    output: ggml::Tensor,

    layers: Vec<Layer>,

    inference_params: InferenceParameters,
    inference_prompt_params: InferenceWithPromptParameters,

    /// Needs to kept alive while the model is alive
    _mmap: Option<Mmap>,

    // Must be kept alive for the model
    _context: ggml::Context,
}
unsafe impl Send for Llama {}
unsafe impl Sync for Llama {}

impl Llama {
    /// Load the model from `path` with `n_context_tokens` context tokens.
    ///
    /// The status of the loading process will be reported through `load_progress_callback`.
    pub fn load(
        path: &Path,
        prefer_mmap: bool,
        params: ModelParameters,
        load_progress_callback: impl FnMut(LoadProgress),
    ) -> Result<Llama, LoadError> {
        llm_base::load(path, prefer_mmap, params, load_progress_callback)
    }
}
impl KnownModel for Llama {
    type Hyperparameters = Hyperparameters;

    fn new<E: Error>(
        hyperparameters: Self::Hyperparameters,
        params: ModelParameters,
        vocabulary: Vocabulary,
        tensor_loader: impl TensorLoader<E>,
    ) -> Result<Self, E> {
        let mut tl = tensor_loader;

        let tok_embeddings = tl.load("tok_embeddings.weight")?;
        let norm = tl.load("norm.weight")?;
        let output = tl.load("output.weight")?;

        let mut layers = Vec::new();
        for i in 0..hyperparameters.n_layer {
            let layer = Layer {
                attention_norm: tl.load(&format!("layers.{i}.attention_norm.weight"))?,
                wq: tl.load(&format!("layers.{i}.attention.wq.weight"))?,
                wk: tl.load(&format!("layers.{i}.attention.wk.weight"))?,
                wv: tl.load(&format!("layers.{i}.attention.wv.weight"))?,
                wo: tl.load(&format!("layers.{i}.attention.wo.weight"))?,
                ffn_norm: tl.load(&format!("layers.{i}.ffn_norm.weight"))?,
                w1: tl.load(&format!("layers.{i}.feed_forward.w1.weight"))?,
                w2: tl.load(&format!("layers.{i}.feed_forward.w2.weight"))?,
                w3: tl.load(&format!("layers.{i}.feed_forward.w3.weight"))?,
            };

            layers.push(layer);
        }

        let (_context, _tensors, _mmap) = tl.finish();

        let ModelParameters {
            n_context_tokens,
            inference_params,
            inference_prompt_params,
        } = params;

        Ok(Self {
            hyperparameters,
            n_context_tokens,
            vocabulary,
            tok_embeddings,
            norm,
            output,
            layers,
            inference_params,
            inference_prompt_params,
            _context,
            _mmap,
        })
    }

    /// Starts a new `InferenceSession` for this model.
    fn start_session(&self, params: InferenceSessionParameters) -> InferenceSession {
        InferenceSession::new(
            params,
            self.n_context_tokens,
            self.hyperparameters.n_layer,
            self.hyperparameters.n_embd,
            self.hyperparameters.n_vocab,
        )
    }

    fn evaluate(
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
            n_embd,
            n_mult: _,
            n_head,
            n_layer,
            n_rot,
            file_type: _,
        } = self.hyperparameters;
        let n_ctx = self.n_context_tokens;

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
        let ctx0 = ggml::Context::init(buf_size, true);

        let mut gf = ggml::ComputationGraph::new(n_threads);

        let mut embd = ctx0.new_tensor_1d(ggml::Type::I32, n);
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
                        (n, n_embd),
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
                    (n_past + n, n_embd / n_head, n_head),
                    (n_ctx * memv_elsize, n_ctx * memv_elsize * n_embd / n_head),
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

    /// Returns the vocabulary used by this model.
    fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    fn n_context_tokens(&self) -> usize {
        self.n_context_tokens
    }

    fn eot_token_id(&self) -> TokenId {
        2
    }

    fn inference_params(&self) -> InferenceParameters {
        self.inference_params.clone()
    }

    fn inference_prompt_params(&self) -> InferenceWithPromptParameters {
        self.inference_prompt_params
    }
}
#[cfg(test)]
impl Llama {
    /// This does *not* construct a valid model. All of the tensors are entirely
    /// empty. However, it can be used to determine if some code will compile.
    fn new_empty() -> Self {
        let context = ggml::Context::init(1024 * 1024, true);
        let tok_embeddings = context.new_f32(0.0);
        let norm = context.new_f32(0.0);
        let output = context.new_f32(0.0);

        Self {
            hyperparameters: Default::default(),
            n_context_tokens: 0,
            vocabulary: Default::default(),
            tok_embeddings,
            norm,
            output,
            layers: Default::default(),
            _mmap: Default::default(),
            _context: context,
            inference_params: Default::default(),
            inference_prompt_params: Default::default(),
        }
    }
}

/// The hyperparameters of the model.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct Hyperparameters {
    /// n_vocab
    pub n_vocab: usize,
    /// n_embd
    pub n_embd: usize,
    /// n_mult
    pub n_mult: usize,
    /// n_head
    pub n_head: usize,
    /// n_layer
    pub n_layer: usize,
    /// n_rot
    pub n_rot: usize,
    /// file_type
    pub file_type: FileType,
}
impl llm_base::Hyperparameters for Hyperparameters {
    type WriteError = BasicWriteError;

    fn read(reader: &mut dyn std::io::BufRead) -> Result<Self, LoadError> {
        Ok(Hyperparameters {
            n_vocab: util::read_i32(reader)?.try_into()?,
            n_embd: util::read_i32(reader)?.try_into()?,
            n_mult: util::read_i32(reader)?.try_into()?,
            n_head: util::read_i32(reader)?.try_into()?,
            n_layer: util::read_i32(reader)?.try_into()?,
            n_rot: util::read_i32(reader)?.try_into()?,
            file_type: {
                let ftype = util::read_i32(reader)?;
                FileType::try_from(ftype).map_err(|_| LoadError::UnsupportedFileType(ftype))?
            },
        })
    }

    fn write(&self, writer: &mut dyn std::io::Write) -> Result<(), Self::WriteError> {
        util::write_i32(writer, self.n_vocab.try_into()?)?;
        util::write_i32(writer, self.n_embd.try_into()?)?;
        util::write_i32(writer, self.n_mult.try_into()?)?;
        util::write_i32(writer, self.n_head.try_into()?)?;
        util::write_i32(writer, self.n_layer.try_into()?)?;
        util::write_i32(writer, self.n_rot.try_into()?)?;
        util::write_i32(writer, self.file_type.into())?;
        Ok(())
    }

    fn n_vocabulary(&self) -> usize {
        self.n_vocab
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn can_share_model_between_threads() {
        let model = Arc::new(Llama::new_empty());

        for _ in 0..4 {
            let model = model.clone();
            std::thread::spawn(move || {
                let _session = model.start_session(Default::default());
            });
        }

        let session = model.start_session(Default::default());
        std::thread::spawn(move || {
            let _session = session;
        });
    }
}
