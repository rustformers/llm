//! An implementation of [BLOOM](https://huggingface.co/docs/transformers/model_doc/bloom)
//! for the `llm` ecosystem.
#![deny(missing_docs)]

use std::sync::Arc;

use llm_base::{
    ggml,
    model::{common, HyperparametersWriteError},
    util, FileType, GraphOutputs, InferenceSession, InferenceSessionConfig, KnownModel,
    ModelParameters, OutputRequest, Regex, TokenId, Tokenizer,
};

/// The BLOOM model. Ref: [Introducing BLOOM](https://bigscience.huggingface.co/blog/bloom)
///
/// # Safety
/// This implements [Send] and [Sync] as it is immutable after construction.
pub struct Bloom {
    params: ModelParameters,

    hyperparameters: Hyperparameters,
    tokenizer: Tokenizer,

    // model-global weights
    // weighted token embeddings
    wte: ggml::Tensor,
    // normalization weight & bias
    norm: ggml::Tensor,
    norm_bias: ggml::Tensor,
    // output normalization weight & bias
    output_norm: ggml::Tensor,
    output_norm_bias: ggml::Tensor,
    // output weight
    output: ggml::Tensor,

    // weights for the model
    layers: Vec<Layer>,

    // must be kept alive for the model
    context: Arc<ggml::Context>,
}

unsafe impl Send for Bloom {}
unsafe impl Sync for Bloom {}

impl KnownModel for Bloom {
    type Hyperparameters = Hyperparameters;

    fn new<E: std::error::Error>(
        hyperparameters: Self::Hyperparameters,
        params: ModelParameters,
        tokenizer: Tokenizer,
        tensor_loader: impl llm_base::TensorLoader<E>,
    ) -> Result<Self, E> {
        let mut tl = tensor_loader;

        // model-global weights
        let wte = tl.load("tok_embeddings.weight")?;
        let norm = tl.load("norm.weight")?;
        let norm_bias = tl.load("norm.bias")?;
        let output_norm = tl.load("output_norm.weight")?;
        let output_norm_bias = tl.load("output_norm.bias")?;
        let output = tl.load("output.weight")?;

        let mut layers = Vec::new();
        for i in 0..hyperparameters.n_layer {
            let layer = Layer {
                attention_norm: tl.load(&format!("layers.{i}.attention_norm.weight"))?,
                attention_norm_b: tl.load(&format!("layers.{i}.attention_norm.bias"))?,

                query_key_value: tl
                    .load(&format!("layers.{i}.attention.query_key_value.weight"))?,
                query_key_value_b: tl
                    .load(&format!("layers.{i}.attention.query_key_value.bias"))?,

                wo: tl.load(&format!("layers.{i}.attention.wo.weight"))?,
                wo_b: tl.load(&format!("layers.{i}.attention.wo.bias"))?,

                ffn_norm: tl.load(&format!("layers.{i}.ffn_norm.weight"))?,
                ffn_norm_b: tl.load(&format!("layers.{i}.ffn_norm.bias"))?,

                w1: tl.load(&format!("layers.{i}.feed_forward.w1.weight"))?,
                w1_b: tl.load(&format!("layers.{i}.feed_forward.w1.bias"))?,
                w2: tl.load(&format!("layers.{i}.feed_forward.w2.weight"))?,
                w2_b: tl.load(&format!("layers.{i}.feed_forward.w2.bias"))?,
            };

            layers.push(layer);
        }

        let context = tl.finish();

        Ok(Bloom {
            hyperparameters,
            params,
            tokenizer,
            wte,
            norm,
            norm_bias,
            output_norm,
            output_norm_bias,
            output,
            layers,
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
            n_vocab,
            n_embd,
            n_mult: _,
            n_head,
            n_layer,
            file_type: _,
        } = self.hyperparameters;

        let outputs = session.compute(self.context.clone(), input_tokens, |builder| {
            let ctx0 = builder.ctx0.borrow();
            let (memory_k_size, memory_v_size) = (
                builder.memory_k.element_size(),
                builder.memory_v.element_size(),
            );
            let embd = &builder.embd;
            let mut input_layer = ctx0.op_get_rows(&self.wte, embd);

            // normalize embeddings
            input_layer = ctx0.op_norm(&input_layer);
            input_layer = ctx0.op_mul(&ctx0.op_repeat(&self.norm, &input_layer), &input_layer);
            input_layer = ctx0.op_add(&ctx0.op_repeat(&self.norm_bias, &input_layer), &input_layer);

            let mut gf = ggml::ComputationGraph::new();
            for il in 0..n_layer {
                let input_self_attention = input_layer.share();
                let mut current: ggml::Tensor;

                // norm
                current = ctx0.op_norm(&input_layer);

                // cur = attention_norm * cur
                current = ctx0.op_mul(
                    &ctx0.op_repeat(&self.layers[il].attention_norm, &current),
                    &current,
                );
                current = ctx0.op_add(
                    &ctx0.op_repeat(&self.layers[il].attention_norm_b, &current),
                    &current,
                );

                //attention
                current = ctx0.op_mul_mat(&self.layers[il].query_key_value, &current);
                current = ctx0.op_add(
                    &ctx0.op_repeat(&self.layers[il].query_key_value_b, &current),
                    &current,
                );

                // self-attention
                let nb = current.get_nb()[1];
                let q_current = ctx0.op_view_2d(
                    &current,
                    (n_embd, input_len),
                    nb,
                    //0 * std::mem::size_of::<f32>() * n_embd as usize,
                    0,
                );
                let k_current = ctx0.op_view_2d(
                    &current,
                    (n_embd, input_len),
                    nb,
                    std::mem::size_of::<f32>() * n_embd,
                );
                let v_current = ctx0.op_view_2d(
                    &current,
                    (n_embd, input_len),
                    nb,
                    2 * std::mem::size_of::<f32>() * n_embd,
                );

                // store key and value to memory
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

                    gf.build_forward_expand(&ctx0.op_cpy(&k_current, &k));
                    gf.build_forward_expand(&ctx0.op_cpy(&v_current, &v));
                }

                // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
                let big_q = ctx0.op_permute(
                    &ctx0.op_cpy(
                        &q_current,
                        &ctx0.new_tensor_3d(ggml::Type::F32, n_embd / n_head, n_head, input_len),
                    ),
                    (0, 2, 1, 3),
                );

                // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
                let big_k = ctx0.op_permute(
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

                // K * Q
                let k_q = ctx0.op_mul_mat(&big_k, &big_q);

                // KQ_scaled = KQ / sqrt(n_embd/n_head)
                let k_q_scaled = ctx0.op_scale(
                    &k_q,
                    &ctx0.new_f32(1.0 / f32::sqrt(n_embd as f32 / n_head as f32)),
                );

                //alibi
                // KQ_scaled_alibi = KQ_scaled + alibi_bias
                let k_q_scaled_alibi = ctx0.op_alibi(&k_q_scaled, session_len, n_head, 8f32);

                // KQ_masked = mask_past(KQ_scaled)
                let k_q_masked = ctx0.op_diag_mask_inf(&k_q_scaled_alibi, session_len);

                // KQ = soft_max(KQ_masked)
                let k_q_soft_max = ctx0.op_soft_max(&k_q_masked);

                let memv_elsize = memory_v_size;

                let v_trans = ctx0.op_cpy(
                    &ctx0.op_permute(
                        &ctx0.op_reshape_3d(
                            &ctx0.op_view_1d(
                                builder.memory_v,
                                (session_len + input_len) * n_embd,
                                il * ctx_size * memv_elsize * n_embd,
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

                let k_q_v = ctx0.op_mul_mat(&v_trans, &k_q_soft_max);

                // KQV_merged = KQV.permute(0, 2, 1, 3)
                let k_q_v_merged = ctx0.op_permute(&k_q_v, (0, 2, 1, 3));

                // cur = KQV_merged.contiguous().view(n_embd, N)
                current = ctx0.op_cpy(
                    &k_q_v_merged,
                    &ctx0.new_tensor_2d(ggml::Type::F32, n_embd, input_len),
                );

                // projection
                current = ctx0.op_mul_mat(&self.layers[il].wo, &current);
                current = ctx0.op_add(&ctx0.op_repeat(&self.layers[il].wo_b, &current), &current);

                let input_feed_forward = ctx0.op_add(&current, &input_self_attention);

                // feed-forward network
                // norm
                current = ctx0.op_norm(&input_feed_forward);

                // cur = ffn_norm*cur + ffn_norm_b
                current = ctx0.op_mul(
                    &ctx0.op_repeat(&self.layers[il].ffn_norm, &current),
                    &current,
                );

                current = ctx0.op_add(
                    &ctx0.op_repeat(&self.layers[il].ffn_norm_b, &current),
                    &current,
                );

                current = ctx0.op_mul_mat(&self.layers[il].w1, &current);

                current = ctx0.op_add(&ctx0.op_repeat(&self.layers[il].w1_b, &current), &current);

                // SILU activation

                current = ctx0.op_gelu(&current);

                current = ctx0.op_mul_mat(&self.layers[il].w2, &current);

                current = ctx0.op_add(&ctx0.op_repeat(&self.layers[il].w2_b, &current), &current);

                current = ctx0.op_add(&current, &input_feed_forward);

                // input for next layer
                input_layer = current;
            }

            // norm
            input_layer = ctx0.op_norm(&input_layer);

            // inpL = norm*inpL
            input_layer = ctx0.op_mul(
                &ctx0.op_repeat(&self.output_norm, &input_layer),
                &input_layer,
            );

            input_layer = ctx0.op_add(
                &ctx0.op_repeat(&self.output_norm_bias, &input_layer),
                &input_layer,
            );

            let embeddings_tensor: ggml::Tensor = input_layer.share();

            // lm_head
            input_layer = ctx0.op_mul_mat(&self.output, &input_layer);

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
        self.tokenizer.id("<s>".as_bytes())
    }

    fn eot_token_id(&self) -> TokenId {
        self.tokenizer.id("</s>".as_bytes()).unwrap()
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

/// BLOOM [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))
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
    /// file_type
    pub file_type: FileType,
}

impl llm_base::Hyperparameters for Hyperparameters {
    fn read_ggml(reader: &mut dyn std::io::BufRead) -> Result<Self, llm_base::LoadError> {
        Ok(Hyperparameters {
            n_vocab: util::read_i32(reader)?.try_into()?,
            n_embd: util::read_i32(reader)?.try_into()?,
            n_mult: util::read_i32(reader)?.try_into()?,
            n_head: util::read_i32(reader)?.try_into()?,
            n_layer: util::read_i32(reader)?.try_into()?,
            file_type: util::read_filetype(reader)?,
        })
    }

    fn write_ggml(&self, writer: &mut dyn std::io::Write) -> Result<(), HyperparametersWriteError> {
        util::write_i32(writer, self.n_vocab.try_into()?)?;
        util::write_i32(writer, self.n_embd.try_into()?)?;
        util::write_i32(writer, self.n_mult.try_into()?)?;
        util::write_i32(writer, self.n_head.try_into()?)?;
        util::write_i32(writer, self.n_layer.try_into()?)?;
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
    pub attention_norm: ggml::Tensor,
    pub attention_norm_b: ggml::Tensor,
    pub wo: ggml::Tensor,
    pub wo_b: ggml::Tensor,
    pub query_key_value: ggml::Tensor,
    pub query_key_value_b: ggml::Tensor,
    // normalization
    pub ffn_norm: ggml::Tensor,
    pub ffn_norm_b: ggml::Tensor,
    // ff
    pub w1: ggml::Tensor,
    pub w1_b: ggml::Tensor,
    pub w2: ggml::Tensor,
    pub w2_b: ggml::Tensor,
}
