// Ref: https://github.com/ggerganov/ggml/blob/5dd92f4/examples/stablelm/main.cpp

use std::{error::Error, path::Path};

use ggml::Tensor;
use llm_base::{
    util, BasicWriteError, EvaluateOutputRequest, FileType, InferenceParameters, InferenceSession,
    InferenceSessionParameters, KnownModel, LoadError, LoadProgress, Mmap, TensorLoader, TokenId,
    Vocabulary,
};

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

    /// Needs to kept alive while the model is alive
    _mmap: Option<Mmap>,

    // Must be kept alive for the model
    _context: ggml::Context,
}

unsafe impl Send for NeoX {}
unsafe impl Sync for NeoX {}

impl NeoX {
    /// Load the model from `path` with `n_context_tokens` context tokens.
    ///
    /// The status of the loading process will be reported through `load_progress_callback`.
    pub fn load(
        path: &Path,
        prefer_mmap: bool,
        n_context_tokens: usize,
        load_progress_callback: impl FnMut(LoadProgress),
    ) -> Result<NeoX, LoadError> {
        llm_base::load(path, prefer_mmap, n_context_tokens, load_progress_callback)
    }
}

impl KnownModel for NeoX {
    type Hyperparameters = Hyperparameters;

    fn new<E: Error>(
        hyperparameters: Self::Hyperparameters,
        n_context_tokens: usize,
        vocabulary: Vocabulary,
        tensor_loader: impl TensorLoader<E>,
    ) -> Result<Self, E>
    where
        Self: Sized,
    {
        let n_embd = hyperparameters.n_embd;
        let n_layer = hyperparameters.n_layer;
        let n_vocab = hyperparameters.n_vocab;

        let mut tl = tensor_loader;

        // prepare memory for weights
        let wte = tl.load("gpt_neox.embed_in.weight", &[n_embd, n_vocab])?;
        let ln_f_g = tl.load("gpt_neox.final_layer_norm.weight", &[n_embd])?;
        let ln_f_b = tl.load("gpt_neox.final_layer_norm.bias", &[n_embd])?;
        let lmh_g = tl.load("embed_out.weight", &[n_embd, n_vocab])?;

        let mut layers = Vec::new();
        for i in 0..n_layer {
            let layer = Layer {
                ln_1_g: tl.load(
                    &format!("gpt_neox.layers.{i}.input_layernorm.weight"),
                    &[n_embd],
                )?,
                ln_1_b: tl.load(
                    &format!("gpt_neox.layers.{i}.input_layernorm.bias"),
                    &[n_embd],
                )?,

                c_attn_attn_w: tl.load(
                    &format!("gpt_neox.layers.{i}.attention.query_key_value.weight"),
                    &[n_embd, n_embd * 3],
                )?,
                c_attn_attn_b: tl.load(
                    &format!("gpt_neox.layers.{i}.attention.query_key_value.bias"),
                    &[n_embd * 3],
                )?,

                c_attn_proj_w: tl.load(
                    &format!("gpt_neox.layers.{i}.attention.dense.weight"),
                    &[n_embd, n_embd],
                )?,
                c_attn_proj_b: tl.load(
                    &format!("gpt_neox.layers.{i}.attention.dense.bias"),
                    &[n_embd],
                )?,

                ln_2_g: tl.load(
                    &format!("gpt_neox.layers.{i}.post_attention_layernorm.weight"),
                    &[n_embd],
                )?,
                ln_2_b: tl.load(
                    &format!("gpt_neox.layers.{i}.post_attention_layernorm.bias"),
                    &[n_embd],
                )?,

                c_mlp_fc_w: tl.load(
                    &format!("gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight"),
                    &[n_embd, n_embd * 4],
                )?,
                c_mlp_fc_b: tl.load(
                    &format!("gpt_neox.layers.{i}.mlp.dense_h_to_4h.bias"),
                    &[n_embd * 4],
                )?,

                c_mlp_proj_w: tl.load(
                    &format!("gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight"),
                    &[n_embd * 4, n_embd],
                )?,
                c_mlp_proj_b: tl.load(
                    &format!("gpt_neox.layers.{i}.mlp.dense_4h_to_h.bias"),
                    &[n_embd],
                )?,
            };

            layers.push(layer);
        }

        let (_context, _, _mmap) = tl.finish();

        Ok(NeoX {
            hyperparameters,
            n_context_tokens,
            vocabulary,
            ln_f_g,
            ln_f_b,
            wte,
            lmh_g,
            layers,
            _context,
            _mmap,
        })
    }

    fn start_session(&self, params: InferenceSessionParameters) -> InferenceSession {
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

        let n_past = session.n_past;

        // wte
        let mut input_layer = ctx0.op_get_rows(&self.wte, &embd);

        let memory_k = &session.memory_k;
        let memory_k_size = memory_k.element_size();

        let memory_v = &session.memory_v;
        let memory_v_size = memory_v.element_size();

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
            assert_eq!(embd.nelements(), n_embd * n);
            unsafe {
                embd.read_data(0, bytemuck::cast_slice_mut(embeddings));
            }
        }

        // Adjust the required memory per token if we didn't know that already
        if session.mem_per_token == 0 {
            session.mem_per_token = ctx0.used_mem() / n;
        }

        // Adjust n_past to new length.
        session.n_past += input_tokens.len();
    }

    fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    fn n_context_tokens(&self) -> usize {
        self.hyperparameters.n_ctx
    }
}

/// The hyperparameters of the model.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct Hyperparameters {
    /// n_vocab
    pub n_vocab: usize,
    /// n_ctx
    pub n_ctx: usize,
    /// n_embd
    pub n_embd: usize,
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

    fn write(&self, writer: &mut dyn std::io::Write) -> Result<(), Self::WriteError> {
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
