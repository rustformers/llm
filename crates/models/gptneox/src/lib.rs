//! An implementation of [GPT-NeoX](https://huggingface.co/docs/transformers/model_doc/gpt_neox) for the `llm` ecosystem.
//! This crate also supports the [RedPajama](https://www.together.xyz/blog/redpajama) GPT-NeoX model.
#![deny(missing_docs)]

use ggml::Tensor;
use llm_base::{
    ggml::{
        self,
        format::gguf::{Metadata, MetadataValue, META_TENSOR_DATA_LAYOUT},
    },
    model::{common, HyperparametersReadError, ModelData, ModelLoadArgs, ModelLoadError},
    FileType, GraphOutputs, InferenceSession, InferenceSessionConfig, Model, ModelContext,
    OutputRequest, Regex, TokenId,
};

/// The GPT-NeoX model. Ref: [GitHub](https://github.com/EleutherAI/gpt-neox)
///
/// # Safety
/// This implements [Send] and [Sync] as it is immutable after construction.
pub struct GptNeoX {
    data: ModelData,
    hyperparameters: Hyperparameters,

    // model-global weights
    // normalization gain & bias
    ln_f_g: Tensor,
    ln_f_b: Tensor,
    // weight token embeddings
    wte: Tensor,
    // language model head gain
    lmh_g: Tensor,

    // weights for the model
    layers: Vec<Block>,

    // must be kept alive for the model
    context: ModelContext,
}

unsafe impl Send for GptNeoX {}
unsafe impl Sync for GptNeoX {}

impl Model for GptNeoX {
    fn new(args: ModelLoadArgs) -> Result<Self, ModelLoadError> {
        let hyperparameters = Hyperparameters::read(&args.gguf.metadata)?;

        let mut tl = args.tensor_loader;

        // model-global weights
        let wte = tl.load("token_embd.weight")?;

        let data = args.data;
        let backend = data.params.backend(0);

        let ln_f_g = tl.load("output_norm.weight")?.transfer_to(backend);
        let ln_f_b = tl.load("output_norm.bias")?.transfer_to(backend);
        let lmh_g = tl.load("output.weight")?.transfer_to(backend);

        let mut layers = Vec::new();
        for i in 0..hyperparameters.block_count {
            let backend = data.params.backend(i);
            let block = Block {
                ln_1_g: tl
                    .load(&format!("blk.{i}.attn_norm.weight"))?
                    .transfer_to(backend),
                ln_1_b: tl
                    .load(&format!("blk.{i}.attn_norm.bias"))?
                    .transfer_to(backend),

                c_attn_attn_w: tl
                    .load(&format!("blk.{i}.attn_qkv.weight"))?
                    .transfer_to(backend),
                c_attn_attn_b: tl
                    .load(&format!("blk.{i}.attn_qkv.bias"))?
                    .transfer_to(backend),

                c_attn_proj_w: tl
                    .load(&format!("blk.{i}.attn_output.weight"))?
                    .transfer_to(backend),
                c_attn_proj_b: tl
                    .load(&format!("blk.{i}.attn_output.bias"))?
                    .transfer_to(backend),

                ln_2_g: tl
                    .load(&format!("blk.{i}.ffn_norm.weight"))?
                    .transfer_to(backend),
                ln_2_b: tl
                    .load(&format!("blk.{i}.ffn_norm.bias"))?
                    .transfer_to(backend),

                c_mlp_fc_w: tl
                    .load(&format!("blk.{i}.ffn_up.weight"))?
                    .transfer_to(backend),
                c_mlp_fc_b: tl
                    .load(&format!("blk.{i}.ffn_up.bias"))?
                    .transfer_to(backend),

                c_mlp_proj_w: tl
                    .load(&format!("blk.{i}.ffn_down.weight"))?
                    .transfer_to(backend),
                c_mlp_proj_b: tl
                    .load(&format!("blk.{i}.ffn_down.bias"))?
                    .transfer_to(backend),
            };

            layers.push(block);
        }

        let context = tl.finish();

        Ok(GptNeoX {
            data,
            hyperparameters,
            ln_f_g,
            ln_f_b,
            wte,
            lmh_g,
            layers,
            context,
        })
    }

    fn start_session(&self, config: InferenceSessionConfig) -> InferenceSession {
        InferenceSession::new(
            config,
            &self.data.params,
            self.hyperparameters.block_count,
            self.hyperparameters.embedding_length,
            self.tokenizer().len(),
        )
    }

    // allow snake case here as its a one-to-one mapping of the original names
    #[allow(non_snake_case)]
    fn evaluate(
        &self,
        session: &mut InferenceSession,
        input_tokens: &[TokenId],
        output_request: &mut OutputRequest,
    ) {
        let n = input_tokens.len();
        let n_past = session.n_past;
        let params = &self.data.params;
        let ctx_size = params.context_size;

        let vocabulary_count = self.tokenizer().len();

        let Hyperparameters {
            embedding_length,
            head_count,
            block_count,
            use_parallel_residual,
            ..
        } = self.hyperparameters;

        let outputs = session.compute(self.context.clone(), input_tokens, |builder| {
            let mut ctx0 = builder.ctx0.borrow_mut();
            let embd = builder.embd;
            let mut input_layer = ctx0.op_get_rows(&self.wte, embd);
            let (memory_k_size, memory_v_size) = (
                builder.memory_k.element_size(),
                builder.memory_v.element_size(),
            );

            let mut gf = ctx0.create_compute_graph();

            for il in 0..block_count {
                ctx0.set_offloading(params.should_offload(il));
                // attention uses first scratch buffer
                ctx0.use_scratch(builder.get_scratch(0));

                // self-attention
                let mut current = ctx0.op_norm(&input_layer);
                current = ctx0.op_add(
                    &ctx0.op_mul(&current, &self.layers[il].ln_1_g),
                    &self.layers[il].ln_1_b,
                );

                // self-attention compute QKV
                current = ctx0.op_mul_mat(&self.layers[il].c_attn_attn_w, &current);
                current = ctx0.op_add(&current, &self.layers[il].c_attn_attn_b);

                let nb = current.get_nb()[1];
                let f32_size = std::mem::size_of::<f32>();

                let mut qcur = ctx0.op_cont(&ctx0.op_view_3d(
                    &current,
                    (embedding_length / head_count, head_count, n),
                    (nb / head_count, nb),
                    0,
                ));
                let mut kcur = ctx0.op_cont(&ctx0.op_view_3d(
                    &current,
                    (embedding_length / head_count, head_count, n),
                    (nb / head_count, nb),
                    f32_size * embedding_length / head_count,
                ));
                let mut vcur = ctx0.op_cont(&ctx0.op_view_3d(
                    &current,
                    (embedding_length / head_count, head_count, n),
                    (nb / head_count, nb),
                    2 * f32_size * embedding_length / head_count,
                ));

                // self-attention using mode = 2 for GPT-NeoX mode
                let overrides = params.rope_overrides.as_ref();
                let n_embd_head = embedding_length / head_count;
                qcur = ctx0.op_rope_inplace(&qcur, n_past, n_embd_head, 2, overrides);
                kcur = ctx0.op_rope_inplace(&kcur, n_past, n_embd_head, 2, overrides);

                // store key and value to memory
                vcur = ctx0.op_transpose(&ctx0.op_reshape_2d(&vcur, embedding_length, n));

                let k = ctx0.op_view_1d(
                    builder.memory_k,
                    n * embedding_length,
                    (memory_k_size * embedding_length) * (il * ctx_size + n_past),
                );

                let v = ctx0.op_view_2d(
                    builder.memory_v,
                    (n, embedding_length),
                    ctx_size * memory_v_size,
                    (il * ctx_size) * memory_v_size * embedding_length + n_past * memory_v_size,
                );

                gf.build_forward_expand(&ctx0.op_cpy(&kcur, &k));
                gf.build_forward_expand(&ctx0.op_cpy(&vcur, &v));

                // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
                let Q = ctx0.op_permute(&qcur, (0, 2, 1, 3));
                // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
                let K = ctx0.op_permute(
                    &ctx0.op_reshape_3d(
                        &ctx0.op_view_1d(
                            builder.memory_k,
                            (n_past + n) * embedding_length,
                            il * ctx_size * memory_k_size * embedding_length,
                        ),
                        embedding_length / head_count,
                        head_count,
                        n_past + n,
                    ),
                    (0, 2, 1, 3),
                );

                // K * Q
                let KQ = ctx0.op_mul_mat(&K, &Q);

                // KQ_scaled = KQ / sqrt(n_embd/n_head)
                let KQ_scaled = ctx0.op_scale_inplace(
                    &KQ,
                    &ctx0.new_f32(1f32 / f32::sqrt(embedding_length as f32 / head_count as f32)),
                );

                // KQ_masked = mask_past(KQ_scaled)
                let KQ_masked = ctx0.op_diag_mask_inf_inplace(&KQ_scaled, n_past);

                // KQ = soft_max(KQ_masked)
                let KQ_softmax = ctx0.op_soft_max_inplace(&KQ_masked);

                // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
                let V = ctx0.op_view_3d(
                    builder.memory_v,
                    (n_past + n, embedding_length / head_count, head_count),
                    (
                        ctx_size * memory_v_size,
                        ctx_size * memory_v_size * embedding_length / head_count,
                    ),
                    il * ctx_size * memory_v_size * embedding_length,
                );

                // KQV = transpose(V) * KQ_soft_max
                let KQV = ctx0.op_mul_mat(&V, &KQ_softmax);
                // KQV_merged = KQV.permute(0, 2, 1, 3)
                let KQV_merged = ctx0.op_permute(&KQV, (0, 2, 1, 3));

                // cur = KQV_merged.contiguous().view(n_embd, N)
                current = ctx0.op_cpy(
                    &KQV_merged,
                    &ctx0.new_tensor_2d(ggml::Type::F32, embedding_length, n),
                );

                // self-attention projection
                current = ctx0.op_mul_mat(&self.layers[il].c_attn_proj_w, &current);
                current = ctx0.op_add(&current, &self.layers[il].c_attn_proj_b);

                // use the second scratch for the feed forward
                ctx0.use_scratch(builder.get_scratch(1));

                let feedforward_input: Tensor;
                if !use_parallel_residual {
                    feedforward_input = ctx0.op_add(&current, &input_layer);
                    current = feed_forward_network(&ctx0, &self.layers[il], &feedforward_input);
                    // input for next layer
                    input_layer = ctx0.op_add(&current, &feedforward_input);
                } else {
                    // calculate with parallel residual
                    feedforward_input = current.share();

                    // this is independent of the self-attention result, so it could be done in parallel to the self-attention
                    // note here we pass inpL instead of cur
                    current = feed_forward_network(&ctx0, &self.layers[il], &input_layer);

                    // layer input + FF
                    current = ctx0.op_add(&current, &feedforward_input);

                    // input for next layer
                    input_layer = ctx0.op_add(&current, &input_layer);
                }
            }

            // use the first scratch for the norm
            ctx0.use_scratch(builder.get_scratch(0));

            // normalize the output
            input_layer = ctx0.op_norm(&input_layer);
            // inpL = ln_f_g*inpL + ln_f_b
            input_layer = ctx0.op_add(&ctx0.op_mul(&input_layer, &self.ln_f_g), &self.ln_f_b);

            let embeddings_tensor: ggml::Tensor = input_layer.share();

            // Disable the scratchbuffer
            ctx0.use_scratch(None);
            ctx0.set_offloading(false);
            // apply language model head
            input_layer = ctx0.op_mul_mat(&self.lmh_g, &input_layer);

            (
                gf,
                GraphOutputs {
                    result: input_layer,
                    embedding_result: embeddings_tensor,
                },
            )
        });

        // finish evaluation
        common::read_last_token(session, &outputs.result, vocabulary_count, n);
        common::extract_logits(output_request, &outputs.result, vocabulary_count, n);
        common::extract_embeddings(
            output_request,
            &outputs.embedding_result,
            embedding_length,
            n,
        );
    }

    fn data(&self) -> &ModelData {
        &self.data
    }

    fn bot_token_id(&self) -> Option<TokenId> {
        None
    }

    fn eot_token_id(&self) -> TokenId {
        self.tokenizer().id("<|endoftext|>".as_bytes()).unwrap()
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

/// GPT-NeoX [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Hyperparameters {
    /// Size of the model's embedding layer
    embedding_length: usize,
    /// n_head
    head_count: usize,
    /// Number of blocks in the model
    block_count: usize,
    /// Whether to use a "parallel" formulation in each Transformer layer.
    /// This is on for most models, but is off for some e.g. RedPajama.
    use_parallel_residual: bool,
    /// file_type
    file_type: Option<FileType>,
    /// The tensor data layout that this model was encoded with
    tensor_data_layout: String,
}

impl Hyperparameters {
    fn read(metadata: &Metadata) -> Result<Self, HyperparametersReadError> {
        Ok(Self {
            embedding_length: metadata.get_countable("gptneox.embedding_length")?,
            head_count: metadata.get_countable("gptneox.attention.head_count")?,
            block_count: metadata.get_countable("gptneox.block_count")?,
            use_parallel_residual: metadata
                .get_with_type("gptneox.use_parallel_residual", MetadataValue::as_bool)?,
            file_type: FileType::read_for_hyperparameters(metadata)?,
            tensor_data_layout: metadata
                .get_str("llama.tensor_data_layout")
                .unwrap_or(META_TENSOR_DATA_LAYOUT)
                .to_string(),
        })
    }
}

struct Block {
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

fn feed_forward_network(context: &ggml::Context, layer: &Block, input: &Tensor) -> Tensor {
    let mut current = context.op_norm(input);

    //gain and bias
    current = context.op_add(&context.op_mul(&current, &layer.ln_2_g), &layer.ln_2_b);

    // apply weights
    current = context.op_mul_mat(&layer.c_mlp_fc_w, &current);

    // apply bias
    current = context.op_add(&current, &layer.c_mlp_fc_b);

    // GELU activation
    current = context.op_gelu(&current);

    // projection
    // cur = proj_w*cur + proj_b
    current = context.op_mul_mat(&layer.c_mlp_proj_w, &current);

    current = context.op_add(&current, &layer.c_mlp_proj_b);

    current
}
