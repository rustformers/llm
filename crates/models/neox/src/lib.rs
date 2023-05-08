//! An implementation of [GPT-NeoX](https://huggingface.co/docs/transformers/model_doc/gpt_neox) for the `llm` ecosystem.
#![deny(missing_docs)]

use std::{error::Error, path::Path};

use ggml::Tensor;
use llm_base::{
    ggml,
    model::{common, HyperparametersWriteError},
    util, EvaluateOutputRequest, FileType, InferenceParameters, InferenceSession,
    InferenceSessionConfig, InferenceWithPromptParameters, KnownModel, LoadError, LoadProgress,
    Mmap, ModelParameters, TensorLoader, TokenId, Vocabulary,
};

/// The GPT-NeoX model. Ref: [GitHub](https://github.com/EleutherAI/gpt-neox)
///
/// # Safety
/// This implements [Send] and [Sync] as it is immutable after construction.
pub struct NeoX {
    hyperparameters: Hyperparameters,
    n_context_tokens: usize,

    vocabulary: Vocabulary,

    // normalization
    ln_f_g: Tensor,
    ln_f_b: Tensor,

    // position embedding
    wte: Tensor,

    // language model head
    lmh_g: Tensor,

    layers: Vec<Layer>,

    inference_params: InferenceParameters,
    inference_prompt_params: InferenceWithPromptParameters,

    /// Needs to kept alive while the model is alive
    _mmap: Option<Mmap>,

    // Must be kept alive for the model
    _context: ggml::Context,
}

unsafe impl Send for NeoX {}
unsafe impl Sync for NeoX {}

impl NeoX {
    /// Load a GPT-NeoX model from the `path` and configure it per the `params`. The
    /// status of the loading process will be reported through `load_progress_callback`.
    /// This is a helper function on top of [llm_base::load].
    pub fn load(
        path: &Path,
        params: ModelParameters,
        load_progress_callback: impl FnMut(LoadProgress),
    ) -> Result<NeoX, LoadError> {
        llm_base::load(path, params, load_progress_callback)
    }
}

impl KnownModel for NeoX {
    type Hyperparameters = Hyperparameters;

    fn new<E: Error>(
        hyperparameters: Self::Hyperparameters,
        params: ModelParameters,
        vocabulary: Vocabulary,
        tensor_loader: impl TensorLoader<E>,
    ) -> Result<Self, E>
    where
        Self: Sized,
    {
        let mut tl = tensor_loader;

        // prepare memory for weights
        let wte = tl.load("gpt_neox.embed_in.weight")?;
        let ln_f_g = tl.load("gpt_neox.final_layer_norm.weight")?;
        let ln_f_b = tl.load("gpt_neox.final_layer_norm.bias")?;
        let lmh_g = tl.load("embed_out.weight")?;

        let mut layers = Vec::new();
        for i in 0..hyperparameters.n_layer {
            let layer = Layer {
                ln_1_g: tl.load(&format!("gpt_neox.layers.{i}.input_layernorm.weight"))?,
                ln_1_b: tl.load(&format!("gpt_neox.layers.{i}.input_layernorm.bias"))?,

                c_attn_attn_w: tl.load(&format!(
                    "gpt_neox.layers.{i}.attention.query_key_value.weight"
                ))?,
                c_attn_attn_b: tl.load(&format!(
                    "gpt_neox.layers.{i}.attention.query_key_value.bias"
                ))?,

                c_attn_proj_w: tl.load(&format!("gpt_neox.layers.{i}.attention.dense.weight"))?,
                c_attn_proj_b: tl.load(&format!("gpt_neox.layers.{i}.attention.dense.bias"))?,

                ln_2_g: tl.load(&format!(
                    "gpt_neox.layers.{i}.post_attention_layernorm.weight"
                ))?,
                ln_2_b: tl.load(&format!(
                    "gpt_neox.layers.{i}.post_attention_layernorm.bias"
                ))?,

                c_mlp_fc_w: tl.load(&format!("gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight"))?,
                c_mlp_fc_b: tl.load(&format!("gpt_neox.layers.{i}.mlp.dense_h_to_4h.bias"))?,

                c_mlp_proj_w: tl.load(&format!("gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight"))?,
                c_mlp_proj_b: tl.load(&format!("gpt_neox.layers.{i}.mlp.dense_4h_to_h.bias"))?,
            };

            layers.push(layer);
        }

        let (_context, _, _mmap) = tl.finish();

        let ModelParameters {
            n_context_tokens,
            inference_params,
            inference_prompt_params,
            ..
        } = params;

        Ok(NeoX {
            hyperparameters,
            n_context_tokens,
            vocabulary,
            ln_f_g,
            ln_f_b,
            wte,
            lmh_g,
            layers,
            inference_params,
            inference_prompt_params,
            _context,
            _mmap,
        })
    }

    fn start_session(&self, params: InferenceSessionConfig) -> InferenceSession {
        InferenceSession::new(
            params,
            self.hyperparameters.n_ctx,
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
        let n_threads = params.n_threads;

        let Hyperparameters {
            n_embd,
            n_head,
            n_vocab,
            n_layer,
            n_rot,
            ..
        } = self.hyperparameters;
        let n_ctx = self.n_context_tokens;

        let (ctx0, embd) = common::prepare_for_evaluate(n_layer, session, input_tokens);

        let n_past = session.n_past;

        // wte
        let mut input_layer = ctx0.op_get_rows(&self.wte, &embd);

        let memory_k = &session.memory_k;
        let memory_k_size = memory_k.element_size();

        let memory_v = &session.memory_v;
        let memory_v_size = memory_v.element_size();

        let mut gf = ggml::ComputationGraph::new(n_threads);

        for il in 0..n_layer {
            // self-attention
            let mut current = ctx0.op_norm(&input_layer);
            current = ctx0.op_add(
                &ctx0.op_mul(&ctx0.op_repeat(&self.layers[il].ln_1_g, &current), &current),
                &ctx0.op_repeat(&self.layers[il].ln_1_b, &current),
            );

            // self-attention compute QKV
            current = ctx0.op_mul_mat(&self.layers[il].c_attn_attn_w, &current);
            current = ctx0.op_add(
                &ctx0.op_repeat(&self.layers[il].c_attn_attn_b, &current),
                &current,
            );

            let nb = current.get_nb()[1];
            let f32_size = std::mem::size_of::<f32>();
            let mut qcur = ctx0.op_cont(&ctx0.op_view_3d(
                &current,
                (n_embd / n_head, n_head, n),
                (nb / n_head, nb),
                0,
            ));
            let mut kcur = ctx0.op_cont(&ctx0.op_view_3d(
                &current,
                (n_embd / n_head, n_head, n),
                (nb / n_head, nb),
                f32_size * n_embd / n_head,
            ));
            let mut vcur = ctx0.op_cont(&ctx0.op_view_3d(
                &current,
                (n_embd / n_head, n_head, n),
                (nb / n_head, nb),
                2 * f32_size * n_embd / n_head,
            ));

            // self-attention using mode = 2 for GPT-NeoX mode
            qcur = ctx0.op_rope(&qcur, n_past, n_rot, 2);
            kcur = ctx0.op_rope(&kcur, n_past, n_rot, 2);

            // self-attention store key and value to memory
            vcur = ctx0.op_transpose(&ctx0.op_reshape_2d(&vcur, n_embd, n));

            let little_k = ctx0.op_view_1d(
                memory_k,
                n * n_embd,
                (memory_k_size * n_embd) * (il * n_ctx + n_past),
            );
            let little_v = ctx0.op_view_2d(
                memory_v,
                (n, n_embd),
                n_ctx * memory_v_size,
                (il * n_ctx) * memory_v_size * n_embd + n_past * memory_v_size,
            );

            gf.build_forward_expand(&ctx0.op_cpy(&kcur, &little_k));
            gf.build_forward_expand(&ctx0.op_cpy(&vcur, &little_v));

            let q = ctx0.op_permute(&qcur, 0, 2, 1, 3);
            let big_k = ctx0.op_permute(
                &ctx0.op_reshape_3d(
                    &ctx0.op_view_1d(
                        memory_k,
                        (n_past + n) * n_embd,
                        il * n_ctx * memory_k_size * n_embd,
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

            let kq = ctx0.op_mul_mat(&big_k, &q);
            let kq_scaled = ctx0.op_scale(
                &kq,
                &ctx0.new_f32(1f32 / f32::sqrt(n_embd as f32 / n_head as f32)),
            );

            let kq_masked = ctx0.op_diag_mask_inf(&kq_scaled, n_past);
            let kq_softmax = ctx0.op_soft_max(&kq_masked);

            let big_v = ctx0.op_view_3d(
                memory_v,
                (n_past + n, n_embd / n_head, n_head),
                (
                    n_ctx * memory_v_size,
                    n_ctx * memory_v_size * n_embd / n_head,
                ),
                il * n_ctx * memory_v_size * n_embd,
            );

            let kqv = ctx0.op_mul_mat(&big_v, &kq_softmax);
            let kqv_merged = ctx0.op_permute(&kqv, 0, 2, 1, 3);

            current = ctx0.op_cpy(&kqv_merged, &ctx0.new_tensor_2d(ggml::Type::F32, n_embd, n));

            // self-attention projection
            current = ctx0.op_mul_mat(&self.layers[il].c_attn_proj_w, &current);
            current = ctx0.op_add(
                &ctx0.op_repeat(&self.layers[il].c_attn_proj_b, &current),
                &current,
            );

            // feed-forward
            let ff_in = current.share();

            // feed-forward post attention layer norm
            current = ctx0.op_norm(&input_layer);
            current = ctx0.op_add(
                &ctx0.op_mul(&ctx0.op_repeat(&self.layers[il].ln_2_g, &current), &current),
                &ctx0.op_repeat(&self.layers[il].ln_2_b, &current),
            );

            current = ctx0.op_mul_mat(&self.layers[il].c_mlp_fc_w, &current);
            current = ctx0.op_add(
                &ctx0.op_repeat(&self.layers[il].c_mlp_fc_b, &current),
                &current,
            );

            current = ctx0.op_gelu(&current);

            // feed-forward projection
            current = ctx0.op_mul_mat(&self.layers[il].c_mlp_proj_w, &current);
            current = ctx0.op_add(
                &ctx0.op_repeat(&self.layers[il].c_mlp_proj_b, &current),
                &current,
            );

            current = ctx0.op_add(&current, &ff_in);

            // input for next layer
            input_layer = ctx0.op_add(&current, &input_layer);
        }

        input_layer = ctx0.op_norm(&input_layer);
        input_layer = ctx0.op_add(
            &ctx0.op_mul(&ctx0.op_repeat(&self.ln_f_g, &input_layer), &input_layer),
            &ctx0.op_repeat(&self.ln_f_b, &input_layer),
        );

        input_layer = ctx0.op_mul_mat(&self.lmh_g, &input_layer);

        // run the computation
        gf.build_forward_expand(&input_layer);
        ctx0.graph_compute(&mut gf);

        // finish evaluation
        common::read_last_token(session, &input_layer, n_vocab, n);
        common::extract_logits(output_request, &input_layer, n_vocab, n);
        common::extract_embeddings(output_request, &embd, n_embd, n);
        common::update_session(session, &ctx0, input_tokens.len(), n);
    }

    fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    fn n_context_tokens(&self) -> usize {
        self.hyperparameters.n_ctx
    }

    fn bot_token_id(&self) -> Option<TokenId> {
        None
    }

    fn eot_token_id(&self) -> TokenId {
        self.vocabulary
            .token_to_id
            .get("<|endoftext|>".as_bytes())
            .copied()
            .unwrap()
    }

    fn inference_params(&self) -> InferenceParameters {
        self.inference_params.clone()
    }

    fn inference_prompt_params(&self) -> InferenceWithPromptParameters {
        self.inference_prompt_params
    }
}

/// GPT-NeoX [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct Hyperparameters {
    /// Size of the model's vocabulary
    pub n_vocab: usize,
    /// Size of the model's context
    pub n_ctx: usize,
    /// Size of the model's embedding layer
    pub n_embd: usize,
    /// n_head
    pub n_head: usize,
    /// Number of layers in the model
    pub n_layer: usize,
    /// n_rot
    pub n_rot: usize,
    /// file_type
    pub file_type: FileType,
}
impl llm_base::Hyperparameters for Hyperparameters {
    fn read_ggml(reader: &mut dyn std::io::BufRead) -> Result<Self, LoadError> {
        Ok(Hyperparameters {
            n_vocab: util::read_i32(reader)?.try_into()?,
            n_ctx: util::read_i32(reader)?.try_into()?,
            n_embd: util::read_i32(reader)?.try_into()?,
            n_head: util::read_i32(reader)?.try_into()?,
            n_layer: util::read_i32(reader)?.try_into()?,
            n_rot: util::read_i32(reader)?.try_into()?,
            file_type: {
                let ftype = util::read_i32(reader)?;
                FileType::try_from(ftype).map_err(|_| LoadError::UnsupportedFileType(ftype))?
            },
        })
    }

    fn write_ggml(&self, writer: &mut dyn std::io::Write) -> Result<(), HyperparametersWriteError> {
        util::write_i32(writer, self.n_vocab.try_into()?)?;
        util::write_i32(writer, self.n_ctx.try_into()?)?;
        util::write_i32(writer, self.n_embd.try_into()?)?;
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
    // pre-normalization
    ln_1_g: Tensor,
    ln_1_b: Tensor,

    // attention
    c_attn_attn_w: Tensor,
    c_attn_attn_b: Tensor,

    c_attn_proj_w: Tensor,
    c_attn_proj_b: Tensor,

    // post normalization
    ln_2_g: Tensor,
    ln_2_b: Tensor,

    // feed-forward
    c_mlp_fc_w: Tensor,
    c_mlp_fc_b: Tensor,

    c_mlp_proj_w: Tensor,
    c_mlp_proj_b: Tensor,
}

#[cfg(test)]
impl NeoX {
    /// This does *not* construct a valid model. All of the tensors are entirely
    /// empty. However, it can be used to determine if some code will compile.
    fn new_empty() -> Self {
        let context = ggml::Context::init(1024 * 1024, true);

        Self {
            hyperparameters: Default::default(),
            n_context_tokens: 0,
            vocabulary: Default::default(),
            ln_f_g: context.new_f32(0.0),
            ln_f_b: context.new_f32(0.0),
            wte: context.new_f32(0.0),
            lmh_g: context.new_f32(0.0),
            layers: Default::default(),
            inference_params: Default::default(),
            inference_prompt_params: Default::default(),
            _mmap: Default::default(),
            _context: context,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn can_share_model_between_threads() {
        let model = Arc::new(NeoX::new_empty());

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
