//! An implementation of [LLaMA](https://huggingface.co/docs/transformers/model_doc/llama) for the `llm` ecosystem.
#![deny(missing_docs)]

use std::{error::Error, sync::Arc};

use llm_base::{
    ggml::{self},
    model::{common, HyperparametersWriteError},
    util, FileType, GraphOutputs, InferenceSession, InferenceSessionConfig, KnownModel, LoadError,
    ModelParameters, OutputRequest, Regex, TensorLoader, TokenId, Tokenizer,
};

/// The LLaMA model. Ref: [Introducing LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
///
/// # Safety
/// This implements [Send] and [Sync] as it is immutable after construction.
pub struct Llama {
    params: ModelParameters,
    hyperparameters: Hyperparameters,
    tokenizer: Tokenizer,

    // model-global weights
    // weighted token embeddings
    wte: ggml::Tensor,
    // normalization
    norm: ggml::Tensor,
    // output weight
    output: ggml::Tensor,

    // weights for the model
    layers: Vec<Layer>,

    // must be kept alive for the model
    context: Arc<ggml::Context>,
}

unsafe impl Send for Llama {}
unsafe impl Sync for Llama {}

impl KnownModel for Llama {
    type Hyperparameters = Hyperparameters;

    fn new<E: Error>(
        hyperparameters: Self::Hyperparameters,
        params: ModelParameters,
        tokenizer: Tokenizer,
        tensor_loader: impl TensorLoader<E>,
    ) -> Result<Self, E> {
        let mut tl = tensor_loader;

        // model-global weights
        let wte = tl.load("tok_embeddings.weight")?;

        let backend = params.backend(0);

        let norm = tl.load("norm.weight")?.transfer_to(backend);
        let output = tl.load("output.weight")?.transfer_to(backend);

        let mut layers = Vec::new();

        for i in 0..hyperparameters.n_layer {
            let backend = params.backend(i);

            let layer = Layer {
                attention_norm: tl
                    .load(&format!("layers.{i}.attention_norm.weight"))?
                    .transfer_to(backend),
                wq: tl
                    .load(&format!("layers.{i}.attention.wq.weight"))?
                    .transfer_to(backend),
                wk: tl
                    .load(&format!("layers.{i}.attention.wk.weight"))?
                    .transfer_to(backend),
                wv: tl
                    .load(&format!("layers.{i}.attention.wv.weight"))?
                    .transfer_to(backend),
                wo: tl
                    .load(&format!("layers.{i}.attention.wo.weight"))?
                    .transfer_to(backend),
                ffn_norm: tl
                    .load(&format!("layers.{i}.ffn_norm.weight"))?
                    .transfer_to(backend),
                w1: tl
                    .load(&format!("layers.{i}.feed_forward.w1.weight"))?
                    .transfer_to(backend),
                w2: tl
                    .load(&format!("layers.{i}.feed_forward.w2.weight"))?
                    .transfer_to(backend),
                w3: tl
                    .load(&format!("layers.{i}.feed_forward.w3.weight"))?
                    .transfer_to(backend),
            };
            layers.push(layer);
        }
        let context = tl.finish();

        Ok(Self {
            hyperparameters,
            params,
            tokenizer,
            wte,
            norm,
            output,
            layers,
            context: Arc::new(context),
        })
    }

    /// Starts a new `InferenceSession` for this model.
    fn start_session(&self, config: InferenceSessionConfig) -> InferenceSession {
        InferenceSession::new(
            config,
            &self.params,
            self.hyperparameters.n_layer,
            self.hyperparameters.n_embd,
            self.hyperparameters.n_vocab,
        )
    }

    #[tracing::instrument(level = "trace", skip_all)]
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
            n_vocab,
            n_embd,
            n_mult: _,
            n_head,
            n_layer,
            n_rot,
            file_type: _,
        } = self.hyperparameters;

        let outputs = session.compute(self.context.clone(), input_tokens, |builder| {
            let mut ctx0 = builder.ctx0.borrow_mut();
            let embd = builder.embd;

            let mut input_layer = ctx0.op_get_rows(&self.wte, embd);

            let mut gf = ggml::ComputationGraph::new();

            for il in 0..n_layer {
                ctx0.set_offloading(self.params.should_offload(il));

                let input_self_attention = input_layer.share();
                let mut current: ggml::Tensor;

                ctx0.use_scratch(builder.get_scratch(0));

                // norm
                current = ctx0.op_rms_norm(&input_layer);

                // cur = attention_norm * cur
                current = ctx0.op_mul(&current, &self.layers[il].attention_norm);

                // self-attention
                // compute Q and K and RoPE them
                let overrides = self.params.rope_overrides.as_ref();
                let q_current = ctx0
                    .op_rope_inplace(
                        &ctx0.op_reshape_3d(
                            &ctx0.op_mul_mat(&self.layers[il].wq, &current),
                            n_embd / n_head,
                            n_head,
                            input_len,
                        ),
                        session_len,
                        n_rot,
                        0,
                        overrides,
                    )
                    .set_name("Qcur");
                let k_current = ctx0
                    .op_rope_inplace(
                        &ctx0.op_reshape_3d(
                            &ctx0.op_mul_mat(&self.layers[il].wk, &current),
                            n_embd / n_head,
                            n_head,
                            input_len,
                        ),
                        session_len,
                        n_rot,
                        0,
                        overrides,
                    )
                    .set_name("Kcur");

                // store key and value to memory
                // compute the transposed [N, n_embd] V matrix
                let v_current = ctx0.op_transpose(&ctx0.op_reshape_2d(
                    &ctx0.op_mul_mat(&self.layers[il].wv, &current),
                    n_embd,
                    input_len,
                ));

                let k = ctx0.op_view_1d(
                    builder.memory_k,
                    input_len * n_embd,
                    (builder.memory_k.element_size() * n_embd) * (il * ctx_size + session_len),
                );

                let v = ctx0.op_view_2d(
                    builder.memory_v,
                    (input_len, n_embd),
                    ctx_size * builder.memory_v.element_size(),
                    (il * ctx_size) * builder.memory_v.element_size() * n_embd
                        + session_len * builder.memory_v.element_size(),
                );

                // important: storing RoPE-ed version of K in the KV cache!
                gf.build_forward_expand(&ctx0.op_cpy(&k_current, &k));
                gf.build_forward_expand(&ctx0.op_cpy(&v_current, &v));

                let q = ctx0.op_permute(&q_current, (0, 2, 1, 3)).set_name("Q");

                let k = ctx0
                    .op_permute(
                        &ctx0.op_reshape_3d(
                            &ctx0.op_view_1d(
                                builder.memory_k,
                                (session_len + input_len) * n_embd,
                                il * ctx_size * builder.memory_k.element_size() * n_embd,
                            ),
                            n_embd / n_head,
                            n_head,
                            session_len + input_len,
                        ),
                        (0, 2, 1, 3),
                    )
                    .set_name("K");

                // K * Q
                let k_q = ctx0.op_mul_mat(&k, &q).set_name("KQ");

                // KQ_scaled = KQ / sqrt(n_embd/n_head)
                let kq_scale = ctx0
                    .new_f32(1.0 / ((n_embd as f32 / n_head as f32).sqrt()))
                    .set_name("1/sqrt(n_embd/n_head)");
                let k_q_scaled = ctx0.op_scale_inplace(&k_q, &kq_scale).set_name("KQ_scaled");

                // KQ_masked = mask_past(KQ_scaled)
                let k_q_masked = ctx0
                    .op_diag_mask_inf_inplace(&k_q_scaled, session_len)
                    .set_name("KQ_masked");

                // KQ = soft_max(KQ_masked)
                let k_q_soft_max = ctx0
                    .op_soft_max_inplace(&k_q_masked)
                    .set_name("KQ_soft_max");

                // split cached V into n_head heads
                let v = ctx0
                    .op_view_3d(
                        builder.memory_v,
                        (session_len + input_len, n_embd / n_head, n_head),
                        (
                            ctx_size * builder.memory_v.element_size(),
                            ctx_size * builder.memory_v.element_size() * n_embd / n_head,
                        ),
                        il * ctx_size * builder.memory_v.element_size() * n_embd,
                    )
                    .set_name("V");

                let k_q_v = ctx0.op_mul_mat(&v, &k_q_soft_max).set_name("KQV");

                // KQV_merged = KQV.permute(0, 2, 1, 3)
                let k_q_v_merged = ctx0.op_permute(&k_q_v, (0, 2, 1, 3)).set_name("KQV_merged");

                // cur = KQV_merged.contiguous().view(n_embd, N)
                current = ctx0
                    .op_cpy(
                        &k_q_v_merged,
                        &ctx0.new_tensor_2d(ggml::Type::F32, n_embd, input_len),
                    )
                    .set_name("KQV_merged_contiguous");

                // projection (no bias)
                current = ctx0.op_mul_mat(&self.layers[il].wo, &current);

                ctx0.use_scratch(builder.get_scratch(1));

                let input_feed_forward = ctx0.op_add(&current, &input_self_attention);

                // feed-forward network
                // norm
                current = ctx0.op_rms_norm(&input_feed_forward);

                // cur = cur*ffn_norm(broadcasted)
                current = ctx0.op_mul(&current, &self.layers[il].ffn_norm);

                let tmp = ctx0.op_mul_mat(&self.layers[il].w3, &current);

                current = ctx0.op_mul_mat(&self.layers[il].w1, &current);

                // SILU activation
                current = ctx0.op_silu(&current);

                current = ctx0.op_mul(&current, &tmp);

                current = ctx0.op_mul_mat(&self.layers[il].w2, &current);

                current = ctx0.op_add(&current, &input_feed_forward);

                // input for next layer
                input_layer = current;
            }

            ctx0.use_scratch(builder.get_scratch(0));

            // norm
            input_layer = ctx0.op_rms_norm(&input_layer);

            // inpL = inpL*norm(broadcasted)
            input_layer = ctx0.op_mul(&input_layer, &self.norm);

            let embedding_result: ggml::Tensor = input_layer.share();

            ctx0.set_offloading(false);
            // lm_head
            input_layer = ctx0.op_mul_mat(&self.output, &input_layer);

            ctx0.use_scratch(None);
            (
                gf,
                GraphOutputs {
                    result: input_layer,
                    embedding_result,
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
        self.tokenizer.id("</s>".as_bytes()).unwrap_or(2)
    }

    fn quantize_tensors() -> Vec<Regex> {
        vec![Regex::new(".*weight").unwrap()]
    }

    fn skip_quantize_tensors() -> Vec<Regex> {
        vec![]
    }

    fn supports_rewind(&self) -> bool {
        true
    }
}

/// LLaMA [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct Hyperparameters {
    /// Size of the model's vocabulary
    pub n_vocab: usize,
    /// Size of the model's embedding layer
    pub n_embd: usize,
    /// n_mult
    pub n_mult: usize,
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
            n_embd: util::read_i32(reader)?.try_into()?,
            n_mult: util::read_i32(reader)?.try_into()?,
            n_head: util::read_i32(reader)?.try_into()?,
            n_layer: util::read_i32(reader)?.try_into()?,
            n_rot: util::read_i32(reader)?.try_into()?,
            file_type: util::read_filetype(reader)?,
        })
    }

    fn write_ggml(&self, writer: &mut dyn std::io::Write) -> Result<(), HyperparametersWriteError> {
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

    fn file_type(&self) -> Option<FileType> {
        Some(self.file_type)
    }

    fn file_type_mut(&mut self) -> Option<&mut FileType> {
        Some(&mut self.file_type)
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
