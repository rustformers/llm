//! An implementation of [LLaMA](https://huggingface.co/docs/transformers/model_doc/llama) for the `llm` ecosystem.
#![deny(missing_docs)]

use llm_base::{
    ggml::{
        self,
        format::gguf::{Metadata, META_TENSOR_DATA_LAYOUT},
    },
    model::{common, HyperparametersReadError, ModelData, ModelLoadArgs, ModelLoadError},
    FileType, GraphOutputs, InferenceSession, InferenceSessionConfig, Model, ModelContext,
    OutputRequest, Regex, TokenId,
};

/// The LLaMA model. Ref: [Introducing LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
///
/// # Safety
/// This implements [Send] and [Sync] as it is immutable after construction.
pub struct Llama {
    data: ModelData,
    hyperparameters: Hyperparameters,
    // model-global weights
    // weighted token embeddings
    wte: ggml::Tensor,
    // normalization
    norm: ggml::Tensor,
    // output weight
    output: ggml::Tensor,

    // weights for the model
    blocks: Vec<Block>,

    // must be kept alive for the model
    context: ModelContext,
}

unsafe impl Send for Llama {}
unsafe impl Sync for Llama {}

impl Model for Llama {
    fn new(args: ModelLoadArgs) -> Result<Self, ModelLoadError> {
        let hyperparameters = Hyperparameters::read(&args.gguf.metadata)?;
        assert_eq!(hyperparameters.tensor_data_layout, META_TENSOR_DATA_LAYOUT);

        let mut tl = args.tensor_loader;

        // model-global weights
        let wte = tl.load("token_embd.weight")?;

        let data = args.data;

        let backend = data.params.backend(0);

        let norm = tl.load("output_norm.weight")?.transfer_to(backend);
        let output = tl.load("output.weight")?.transfer_to(backend);

        let mut blocks = Vec::new();

        for i in 0..hyperparameters.block_count {
            let backend = data.params.backend(i);

            let block = Block {
                attn_n: tl
                    .load(&format!("blk.{i}.attn_norm.weight"))?
                    .transfer_to(backend),
                attn_q: tl
                    .load(&format!("blk.{i}.attn_q.weight"))?
                    .transfer_to(backend),
                attn_k: tl
                    .load(&format!("blk.{i}.attn_k.weight"))?
                    .transfer_to(backend),
                attn_v: tl
                    .load(&format!("blk.{i}.attn_v.weight"))?
                    .transfer_to(backend),
                attn_output: tl
                    .load(&format!("blk.{i}.attn_output.weight"))?
                    .transfer_to(backend),
                ffn_norm: tl
                    .load(&format!("blk.{i}.ffn_norm.weight"))?
                    .transfer_to(backend),
                ffn_gate: tl
                    .load(&format!("blk.{i}.ffn_gate.weight"))?
                    .transfer_to(backend),
                ffn_down: tl
                    .load(&format!("blk.{i}.ffn_down.weight"))?
                    .transfer_to(backend),
                ffn_up: tl
                    .load(&format!("blk.{i}.ffn_up.weight"))?
                    .transfer_to(backend),
            };
            blocks.push(block);
        }
        let context = tl.finish();

        Ok(Self {
            data,
            hyperparameters,
            wte,
            norm,
            output,
            blocks,
            context,
        })
    }

    /// Starts a new `InferenceSession` for this model.
    fn start_session(&self, config: InferenceSessionConfig) -> InferenceSession {
        InferenceSession::new(
            config,
            &self.data.params,
            self.hyperparameters.block_count,
            self.hyperparameters.embedding_length,
            self.tokenizer().len(),
        )
    }

    #[tracing::instrument(level = "trace", skip_all)]
    fn evaluate(
        &self,
        session: &mut InferenceSession,
        input_tokens: &[TokenId],
        output_request: &mut OutputRequest,
    ) {
        let params = &self.data.params;
        let ctx_size = params.context_size;

        let vocabulary_count = self.tokenizer().len();

        let Hyperparameters {
            embedding_length,
            head_count,
            head_count_kv,
            block_count,
            file_type: _,
            tensor_data_layout: _,
        } = self.hyperparameters;

        let embedding_length_gqa =
            embedding_length / self.hyperparameters.grouped_query_attention();

        let outputs = session.compute(self.context.clone(), input_tokens, |builder| {
            let session_len = builder.n_past;
            let input_len = builder.input_length();
            let mut ctx0 = builder.ctx0.borrow_mut();
            let embd = builder.embd;

            let mut input_layer = ctx0.op_get_rows(&self.wte, embd);

            let mut gf = ctx0.create_compute_graph();

            for il in 0..block_count {
                ctx0.set_offloading(params.should_offload(il));

                let input_self_attention = input_layer.share();
                let mut current: ggml::Tensor;

                // norm
                current = ctx0.op_rms_norm(&input_layer);

                // cur = attention_norm * cur
                current = ctx0.op_mul(&current, &self.blocks[il].attn_n);

                // self-attention
                // compute Q and K and RoPE them
                let overrides = params.rope_overrides.as_ref();
                let n_embd_head = embedding_length / head_count;
                let q_current = ctx0
                    .op_rope_inplace(
                        &ctx0.op_reshape_3d(
                            &ctx0.op_mul_mat(&self.blocks[il].attn_q, &current),
                            n_embd_head,
                            head_count,
                            input_len,
                        ),
                        todo!(),
                        session_len,
                        n_embd_head,
                        0,
                        overrides,
                    )
                    .set_name("Qcur");
                let k_current = ctx0
                    .op_rope_inplace(
                        &ctx0.op_reshape_3d(
                            &ctx0.op_mul_mat(&self.blocks[il].attn_k, &current),
                            n_embd_head,
                            head_count_kv,
                            input_len,
                        ),
                        todo!(),
                        session_len,
                        n_embd_head,
                        0,
                        overrides,
                    )
                    .set_name("Kcur");

                // store key and value to memory
                // compute the transposed [N, embedding_length] V matrix
                let v_current = ctx0.op_transpose(&ctx0.op_reshape_2d(
                    &ctx0.op_mul_mat(&self.blocks[il].attn_v, &current),
                    embedding_length_gqa,
                    input_len,
                ));

                let k = ctx0.op_view_1d(
                    builder.memory_k,
                    input_len * embedding_length_gqa,
                    (builder.memory_k.element_size() * embedding_length_gqa)
                        * (il * ctx_size + session_len),
                );

                let v = ctx0.op_view_2d(
                    builder.memory_v,
                    (input_len, embedding_length_gqa),
                    ctx_size * builder.memory_v.element_size(),
                    (il * ctx_size) * builder.memory_v.element_size() * embedding_length_gqa
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
                                (session_len + input_len) * embedding_length_gqa,
                                il * ctx_size
                                    * builder.memory_k.element_size()
                                    * embedding_length_gqa,
                            ),
                            n_embd_head,
                            head_count_kv,
                            session_len + input_len,
                        ),
                        (0, 2, 1, 3),
                    )
                    .set_name("K");

                // K * Q
                let k_q = ctx0.op_mul_mat(&k, &q).set_name("KQ");

                // KQ_scaled = KQ / sqrt(embedding_length/head_count)
                let kq_scale = ctx0
                    .new_f32(1.0 / ((embedding_length as f32 / head_count as f32).sqrt()))
                    .set_name("1/sqrt(embedding_length/head_count)");
                let k_q_scaled = ctx0.op_scale_inplace(&k_q, &kq_scale).set_name("KQ_scaled");

                // KQ_masked = mask_past(KQ_scaled)
                let k_q_masked = ctx0
                    .op_diag_mask_inf_inplace(&k_q_scaled, session_len)
                    .set_name("KQ_masked");

                // KQ = soft_max(KQ_masked)
                let k_q_soft_max = ctx0
                    .op_soft_max_inplace(&k_q_masked)
                    .set_name("KQ_soft_max");

                // split cached V into head_count heads
                let v = ctx0
                    .op_view_3d(
                        builder.memory_v,
                        (
                            session_len + input_len,
                            embedding_length / head_count,
                            head_count_kv,
                        ),
                        (
                            ctx_size * builder.memory_v.element_size(),
                            ctx_size * builder.memory_v.element_size() * embedding_length
                                / head_count,
                        ),
                        il * ctx_size * builder.memory_v.element_size() * embedding_length_gqa,
                    )
                    .set_name("V");

                let k_q_v = ctx0.op_mul_mat(&v, &k_q_soft_max).set_name("KQV");

                // KQV_merged = KQV.permute(0, 2, 1, 3)
                let k_q_v_merged = ctx0.op_permute(&k_q_v, (0, 2, 1, 3)).set_name("KQV_merged");

                // cur = KQV_merged.contiguous().view(embedding_length, N)
                current = ctx0
                    .op_cpy(
                        &k_q_v_merged,
                        &ctx0.new_tensor_2d(ggml::Type::F32, embedding_length, input_len),
                    )
                    .set_name("KQV_merged_contiguous");

                // projection (no bias)
                current = ctx0.op_mul_mat(&self.blocks[il].attn_output, &current);

                let input_feed_forward = ctx0.op_add(&current, &input_self_attention);

                // feed-forward network
                // norm
                current = ctx0.op_rms_norm(&input_feed_forward);

                // cur = cur*ffn_norm(broadcasted)
                current = ctx0.op_mul(&current, &self.blocks[il].ffn_norm);

                let tmp = ctx0.op_mul_mat(&self.blocks[il].ffn_up, &current);

                current = ctx0.op_mul_mat(&self.blocks[il].ffn_gate, &current);

                // SILU activation
                current = ctx0.op_silu(&current);

                current = ctx0.op_mul(&current, &tmp);

                current = ctx0.op_mul_mat(&self.blocks[il].ffn_down, &current);

                current = ctx0.op_add(&current, &input_feed_forward);

                // input for next layer
                input_layer = current;
            }

            // norm
            input_layer = ctx0.op_rms_norm(&input_layer);

            // inpL = inpL*norm(broadcasted)
            input_layer = ctx0.op_mul(&input_layer, &self.norm);

            let embedding_result: ggml::Tensor = input_layer.share();

            ctx0.set_offloading(false);
            // lm_head
            input_layer = ctx0.op_mul_mat(&self.output, &input_layer);

            (
                gf,
                GraphOutputs {
                    result: input_layer,
                    embedding_result,
                    output_length: input_len,
                },
            )
        });

        // finish evaluation
        common::read_last_token(
            session,
            &outputs.result,
            vocabulary_count,
            outputs.output_length,
        );
        common::extract_logits(
            output_request,
            &outputs.result,
            vocabulary_count,
            outputs.output_length,
        );
        common::extract_embeddings(
            output_request,
            &outputs.embedding_result,
            embedding_length,
            outputs.output_length,
        );
    }

    fn data(&self) -> &ModelData {
        &self.data
    }

    fn bot_token_id(&self) -> Option<TokenId> {
        None
    }

    fn eot_token_id(&self) -> TokenId {
        self.tokenizer().id("</s>".as_bytes()).unwrap_or(2)
    }

    fn quantize_tensors(&self) -> Vec<Regex> {
        vec![Regex::new(".*weight").unwrap()]
    }

    fn skip_quantize_tensors(&self) -> Vec<Regex> {
        vec![]
    }

    fn supports_rewind(&self) -> bool {
        true
    }
}

#[derive(Debug, Default, PartialEq, Eq, Clone)]
struct Hyperparameters {
    /// Size of the model's embedding layer
    embedding_length: usize,
    /// The number of attention heads
    head_count: usize,
    /// The number of grouped-query attention heads
    head_count_kv: usize,
    /// Number of blocks in the model
    block_count: usize,
    /// file_type
    file_type: Option<FileType>,
    /// The tensor data layout that this model was encoded with
    tensor_data_layout: String,
}
impl Hyperparameters {
    pub fn read(metadata: &Metadata) -> Result<Self, HyperparametersReadError> {
        Ok(Self {
            embedding_length: metadata.get_countable("llama.embedding_length")?,
            head_count: metadata.get_countable("llama.attention.head_count")?,
            head_count_kv: metadata.get_countable("llama.attention.head_count_kv")?,
            block_count: metadata.get_countable("llama.block_count")?,
            file_type: FileType::read_for_hyperparameters(metadata)?,
            tensor_data_layout: metadata
                .get_str("llama.tensor_data_layout")
                .unwrap_or(META_TENSOR_DATA_LAYOUT)
                .to_string(),
        })
    }

    /// Returns the number of grouped-query attention heads.
    fn grouped_query_attention(&self) -> usize {
        self.head_count / self.head_count_kv
    }
}

struct Block {
    attn_n: ggml::Tensor,

    attn_q: ggml::Tensor,
    attn_k: ggml::Tensor,
    attn_v: ggml::Tensor,
    attn_output: ggml::Tensor,

    // normalization
    ffn_norm: ggml::Tensor,

    // ff
    ffn_gate: ggml::Tensor,
    ffn_down: ggml::Tensor,
    ffn_up: ggml::Tensor,
}
