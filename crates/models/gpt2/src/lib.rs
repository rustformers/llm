//! An implementation of [GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2) for the `llm` ecosystem.
#![deny(missing_docs)]

use std::sync::Arc;

use ggml::Tensor;
use llm_base::{
    ggml,
    model::{common, HyperparametersWriteError},
    util, FileType, GraphOutputs, InferenceSession, InferenceSessionConfig, KnownModel, LoadError,
    ModelParameters, OutputRequest, Regex, TokenId, Tokenizer,
};

/// The GPT-2 model. Ref: [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
///
/// # Safety
/// This implements [Send] and [Sync] as it is immutable after construction.
pub struct Gpt2 {
    params: ModelParameters,

    hyperparameters: Hyperparameters,
    tokenizer: Tokenizer,

    // model-global weights
    // normalization gain & bias
    ln_f_g: Tensor,
    ln_f_b: Tensor,
    // weighted token embeddings
    wte: Tensor,
    // weighted positional encodings
    wpe: Tensor,
    // language model head
    //
    // Optional: if not present, the `wte` tensor is used instead.
    lm_head: Option<Tensor>,

    // weights for the model
    layers: Vec<Layer>,

    // must be kept alive for the model
    context: Arc<ggml::Context>,
}

unsafe impl Send for Gpt2 {}
unsafe impl Sync for Gpt2 {}

impl KnownModel for Gpt2 {
    type Hyperparameters = Hyperparameters;

    fn new<E: std::error::Error>(
        hyperparameters: Self::Hyperparameters,
        params: ModelParameters,
        tokenizer: Tokenizer,
        tensor_loader: impl llm_base::TensorLoader<E>,
    ) -> Result<Self, E> {
        let mut tl = tensor_loader;

        // model-global weights
        let ln_f_g = tl.load("model/ln_f/g")?;
        let ln_f_b = tl.load("model/ln_f/b")?;
        let wte = tl.load("model/wte")?;
        let wpe = tl.load("model/wpe")?;

        // GPT-2's language model head is optional; if it is not present,
        // the `wte` tensor is used instead.
        let lm_head = tl.load("model/lm_head").ok();

        let mut layers = Vec::new();
        for i in 0..hyperparameters.n_layer {
            let layer = Layer {
                ln_1_g: tl.load(&format!("model/h{i}/ln_1/g"))?,
                ln_1_b: tl.load(&format!("model/h{i}/ln_1/b"))?,
                ln_2_g: tl.load(&format!("model/h{i}/ln_2/g"))?,
                ln_2_b: tl.load(&format!("model/h{i}/ln_2/b"))?,
                c_attn_attn_w: tl.load(&format!("model/h{i}/attn/c_attn/w"))?,
                c_attn_attn_b: tl.load(&format!("model/h{i}/attn/c_attn/b"))?,
                c_attn_proj_w: tl.load(&format!("model/h{i}/attn/c_proj/w"))?,
                c_attn_proj_b: tl.load(&format!("model/h{i}/attn/c_proj/b"))?,
                c_mlp_fc_w: tl.load(&format!("model/h{i}/mlp/c_fc/w"))?,
                c_mlp_fc_b: tl.load(&format!("model/h{i}/mlp/c_fc/b"))?,
                c_mlp_proj_w: tl.load(&format!("model/h{i}/mlp/c_proj/w"))?,
                c_mlp_proj_b: tl.load(&format!("model/h{i}/mlp/c_proj/b"))?,
            };

            layers.push(layer);
        }

        let context = tl.finish();

        Ok(Gpt2 {
            hyperparameters,
            params,
            tokenizer,
            layers,
            ln_f_g,
            ln_f_b,
            wte,
            wpe,
            lm_head,
            context: Arc::new(context),
        })
    }

    fn start_session(&self, config: InferenceSessionConfig) -> InferenceSession {
        InferenceSession::new(
            config,
            &self.params,
            self.hyperparameters.n_layer,
            self.hyperparameters.n_embd,
            self.hyperparameters.n_vocab,
        )
    }

    fn evaluate(
        &self,
        session: &mut InferenceSession,
        input_tokens: &[TokenId],
        output_request: &mut OutputRequest,
    ) {
        let input_len = input_tokens.len();
        let session_len = session.n_past;
        let ctx_size = self.params.context_size;

        let Hyperparameters {
            n_embd,
            n_head,
            n_vocab,
            n_layer,
            ..
        } = self.hyperparameters;

        let outputs = session.compute(self.context.clone(), input_tokens, |builder| {
            let ctx0 = builder.ctx0.borrow();
            let (memory_k_size, memory_v_size) = (
                builder.memory_k.element_size(),
                builder.memory_v.element_size(),
            );
            let embd = &builder.embd;

            let position_buf: Vec<i32> = (0..input_len).map(|i| (session_len + i) as i32).collect();

            let mut position = ctx0.new_tensor_1d(ggml::Type::I32, input_len);
            unsafe { position.write_data(bytemuck::cast_slice(&position_buf)) };

            let mut input_layer = ctx0.op_add(
                &ctx0.op_get_rows(&self.wte, embd),
                &ctx0.op_get_rows(&self.wpe, &position),
            );

            let mut gf = ggml::ComputationGraph::new();
            for il in 0..n_layer {
                ctx0.use_scratch(builder.get_scratch(0));

                // norm
                let mut current = ctx0.op_norm(&input_layer);
                current = ctx0.op_add(
                    &ctx0.op_mul(&ctx0.op_repeat(&self.layers[il].ln_1_g, &current), &current),
                    &ctx0.op_repeat(&self.layers[il].ln_1_b, &current),
                );

                // attn
                current = ctx0.op_mul_mat(&self.layers[il].c_attn_attn_w, &current);
                current = ctx0.op_add(
                    &ctx0.op_repeat(&self.layers[il].c_attn_attn_b, &current),
                    &current,
                );

                // self-attn
                let nb = current.get_nb()[1];
                let f32_size = std::mem::size_of::<f32>();
                let qcur = ctx0.op_view_2d(&current, (n_embd, input_len), nb, 0);
                let kcur = ctx0.op_view_2d(&current, (n_embd, input_len), nb, f32_size * n_embd);
                let vcur =
                    ctx0.op_view_2d(&current, (n_embd, input_len), nb, f32_size * n_embd * 2);

                if input_len >= 1 {
                    let k = ctx0.op_view_1d(
                        builder.memory_k,
                        input_len * n_embd,
                        (memory_k_size * n_embd) * (il * ctx_size + session_len),
                    );
                    let v = ctx0.op_view_1d(
                        builder.memory_v,
                        input_len * n_embd,
                        (memory_v_size * n_embd) * (il * ctx_size + session_len),
                    );

                    gf.build_forward_expand(&ctx0.op_cpy(&kcur, &k));
                    gf.build_forward_expand(&ctx0.op_cpy(&vcur, &v));
                }

                let q = ctx0.op_permute(
                    &ctx0.op_cpy(
                        &qcur,
                        &ctx0.new_tensor_3d(ggml::Type::F32, n_embd / n_head, n_head, input_len),
                    ),
                    (0, 2, 1, 3),
                );

                let k = ctx0.op_permute(
                    &ctx0.op_reshape_3d(
                        &ctx0.op_view_1d(
                            builder.memory_k,
                            (session_len + input_len) * n_embd,
                            il * ctx_size * memory_k_size * n_embd,
                        ),
                        n_embd / n_head,
                        n_head,
                        session_len + input_len,
                    ),
                    (0, 2, 1, 3),
                );

                let kq = ctx0.op_mul_mat(&k, &q);
                let kq_scaled = ctx0.op_scale_inplace(
                    &kq,
                    &ctx0.new_f32(1f32 / f32::sqrt(n_embd as f32 / n_head as f32)),
                );

                let kq_masked = ctx0.op_diag_mask_inf_inplace(&kq_scaled, session_len);
                let kq_softmax = ctx0.op_soft_max_inplace(&kq_masked);

                let v_trans = ctx0.op_cpy(
                    &ctx0.op_permute(
                        &ctx0.op_reshape_3d(
                            &ctx0.op_view_1d(
                                builder.memory_v,
                                (session_len + input_len) * n_embd,
                                il * ctx_size * memory_v_size * n_embd,
                            ),
                            n_embd / n_head,
                            n_head,
                            session_len + input_len,
                        ),
                        (1, 2, 0, 3),
                    ),
                    &ctx0.new_tensor_3d(
                        builder.memory_v.get_type(),
                        session_len + input_len,
                        n_embd / n_head,
                        n_head,
                    ),
                );

                let kqv = ctx0.op_mul_mat(&v_trans, &kq_softmax);
                let kqv_merged = ctx0.op_permute(&kqv, (0, 2, 1, 3));

                current = ctx0.op_cpy(
                    &kqv_merged,
                    &ctx0.new_tensor_2d(ggml::Type::F32, n_embd, input_len),
                );

                // projection
                current = ctx0.op_mul_mat(&self.layers[il].c_attn_proj_w, &current);
                current = ctx0.op_add(
                    &ctx0.op_repeat(&self.layers[il].c_attn_proj_b, &current),
                    &current,
                );

                // add input
                current = ctx0.op_add(&current, &input_layer);

                // feed-forward
                let ff_in = current.share();

                ctx0.use_scratch(builder.get_scratch(1));

                // feed-forward normalization
                current = ctx0.op_norm(&ff_in);
                current = ctx0.op_add(
                    &ctx0.op_mul(&ctx0.op_repeat(&self.layers[il].ln_2_g, &current), &current),
                    &ctx0.op_repeat(&self.layers[il].ln_2_b, &current),
                );

                // feed-forward fully connected
                current = ctx0.op_mul_mat(&self.layers[il].c_mlp_fc_w, &current);
                current = ctx0.op_add(
                    &ctx0.op_repeat(&self.layers[il].c_mlp_fc_b, &current),
                    &current,
                );

                // feed-forward activation
                current = ctx0.op_gelu(&current);

                // feed-forward projection
                current = ctx0.op_mul_mat(&self.layers[il].c_mlp_proj_w, &current);
                current = ctx0.op_add(
                    &ctx0.op_repeat(&self.layers[il].c_mlp_proj_b, &current),
                    &current,
                );

                // input for next layer
                input_layer = ctx0.op_add(&current, &ff_in);
            }

            ctx0.use_scratch(builder.get_scratch(0));

            // normalization
            input_layer = ctx0.op_norm(&input_layer);
            input_layer = ctx0.op_add(
                &ctx0.op_mul(&ctx0.op_repeat(&self.ln_f_g, &input_layer), &input_layer),
                &ctx0.op_repeat(&self.ln_f_b, &input_layer),
            );

            ctx0.use_scratch(None);

            let embeddings_tensor: ggml::Tensor = input_layer.share();

            let head = self.lm_head.as_ref().unwrap_or(&self.wte);
            input_layer = ctx0.op_mul_mat(head, &input_layer);

            (
                gf,
                GraphOutputs {
                    result: input_layer,
                    embedding_result: embeddings_tensor,
                },
            )
        });

        // finish evaluation
        common::read_last_token(session, &outputs.result, n_vocab, input_len);
        common::extract_logits(output_request, &outputs.result, n_vocab, input_len);
        common::extract_embeddings(output_request, &outputs.embedding_result, n_embd, input_len);
    }

    fn hyperparameters(&self) -> &Self::Hyperparameters {
        &self.hyperparameters
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn context_size(&self) -> usize {
        self.params.context_size
    }

    fn bot_token_id(&self) -> Option<TokenId> {
        None
    }

    fn eot_token_id(&self) -> TokenId {
        self.tokenizer.id("<|endoftext|>".as_bytes()).unwrap()
    }

    fn quantize_tensors() -> Vec<Regex> {
        [
            "model/wte",
            "model/lm_head",
            "model/h.*/attn/c_attn/w",
            "model/h.*/attn/c_proj/w",
            "model/h.*/mlp/c_fc/w",
            "model/h.*/mlp/c_proj/w",
        ]
        .into_iter()
        .map(|s| Regex::new(s).unwrap())
        .collect()
    }

    fn skip_quantize_tensors() -> Vec<Regex> {
        vec![]
    }
}

/// GPT-2 [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct Hyperparameters {
    /// Size of the model's vocabulary
    n_vocab: usize,
    /// Size of the model's context
    n_ctx: usize,
    /// Size of the model's embedding layer
    n_embd: usize,
    /// n_head
    n_head: usize,
    /// Number of layers in the model
    n_layer: usize,
    /// file type
    file_type: FileType,
}

impl llm_base::Hyperparameters for Hyperparameters {
    fn read_ggml(reader: &mut dyn std::io::BufRead) -> Result<Self, LoadError> {
        let hyperparameters = Hyperparameters {
            n_vocab: util::read_i32(reader)?.try_into()?,
            n_ctx: util::read_i32(reader)?.try_into()?,
            n_embd: util::read_i32(reader)?.try_into()?,
            n_head: util::read_i32(reader)?.try_into()?,
            n_layer: util::read_i32(reader)?.try_into()?,
            file_type: util::read_filetype(reader)?,
        };

        let n_vocab = util::read_i32(reader)? as usize;
        if hyperparameters.n_vocab != n_vocab {
            return Err(LoadError::InvariantBroken {
                path: None,
                invariant: format!(
                    "GPT2 model expected n_vocab {} found {}",
                    hyperparameters.n_vocab, n_vocab
                ),
            });
        }

        Ok(hyperparameters)
    }

    fn write_ggml(&self, writer: &mut dyn std::io::Write) -> Result<(), HyperparametersWriteError> {
        util::write_i32(writer, self.n_vocab.try_into()?)?;
        util::write_i32(writer, self.n_ctx.try_into()?)?;
        util::write_i32(writer, self.n_embd.try_into()?)?;
        util::write_i32(writer, self.n_head.try_into()?)?;
        util::write_i32(writer, self.n_layer.try_into()?)?;
        util::write_i32(writer, self.file_type.into())?;
        util::write_i32(writer, self.n_vocab.try_into()?)?;

        Ok(())
    }

    fn n_vocabulary(&self) -> usize {
        self.n_vocab
    }

    fn file_type(&self) -> Option<FileType> {
        Some(self.file_type)
    }

    fn file_type_mut(&mut self) -> Option<&mut FileType> {
        Some(&mut self.file_type)
    }
}

struct Layer {
    // normalization
    ln_1_g: Tensor,
    ln_1_b: Tensor,

    ln_2_g: Tensor,
    ln_2_b: Tensor,

    // attention
    c_attn_attn_w: Tensor,
    c_attn_attn_b: Tensor,

    c_attn_proj_w: Tensor,
    c_attn_proj_b: Tensor,

    // mlp
    c_mlp_fc_w: Tensor,
    c_mlp_fc_b: Tensor,

    c_mlp_proj_w: Tensor,
    c_mlp_proj_b: Tensor,
}
