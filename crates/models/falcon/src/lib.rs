//! An implementation of the [Falcon](https://falconllm.tii.ae/) model for the `llm` ecosystem.
//!
//! This implementation only works for Falcon 7B, and with 32-bit memory tensors (i.e. your inference session
//! must be configured with a 32-bit [InferenceSessionConfig]).
//!
//! This model will not be generally available in the `llm` ecosystem until Falcon 40B and 16-bit memory is
//! supported. It is currently only available as a preview.
#![deny(missing_docs)]

use std::sync::Arc;

use ggml::Tensor;
use llm_base::{
    ggml,
    model::{common, HyperparametersWriteError},
    util, FileType, GraphOutputs, InferenceSession, InferenceSessionConfig, KnownModel, LoadError,
    ModelParameters, OutputRequest, Regex, TokenId, Tokenizer,
};

/// The Falcon model. Ref: [Technology Innovation Institute](https://huggingface.co/tiiuae)
///
/// # Safety
/// This implements [Send] and [Sync] as it is immutable after construction.
pub struct Falcon {
    params: ModelParameters,

    hyperparameters: Hyperparameters,

    tokenizer: Tokenizer,

    // model-global weights
    // weighted token embeddings
    tok_embeddings: Tensor,
    output_norm: Tensor,
    output_norm_b: Tensor,
    lm_head: Tensor,

    // weights for the model
    layers: Vec<Layer>,

    // must be kept alive for the model
    context: Arc<ggml::Context>,
}

unsafe impl Send for Falcon {}
unsafe impl Sync for Falcon {}

impl KnownModel for Falcon {
    type Hyperparameters = Hyperparameters;

    fn new<E: std::error::Error>(
        hyperparameters: Self::Hyperparameters,
        params: ModelParameters,
        tokenizer: Tokenizer,
        tensor_loader: impl llm_base::TensorLoader<E>,
    ) -> Result<Self, E> {
        let mut tl = tensor_loader;

        // model-gobal weights
        let tok_embeddings = tl.load("transformer.word_embeddings.weight")?;
        let output_norm = tl.load("transformer.ln_f.weight")?;
        let output_norm_b = tl.load("transformer.ln_f.bias")?;
        let lm_head = tl.load("lm_head.weight")?;

        let mut layers = Vec::new();
        // utilizing n_head_kv to determine the model version (parameters)
        let Hyperparameters { n_head_kv, .. } = hyperparameters;
        for i in 0..hyperparameters.n_layer {
            let (input_layernorm_name, attention_norm_name) = if n_head_kv == 1 {
                // falcon 7b
                (format!("transformer.h.{i}.input_layernorm"), None)
            } else {
                // falcon 40b
                (
                    format!("transformer.h.{i}.ln_mlp"),
                    Some(format!("transformer.h.{i}.ln_attn")),
                )
            };
            let layer = Layer {
                input_layernorm: tl.load(&format!("{}.weight", input_layernorm_name))?,
                input_layernorm_b: tl.load(&format!("{}.bias", input_layernorm_name))?,
                attention_norm: attention_norm_name
                    .as_ref()
                    .map(|path| tl.load(&format!("{}.weight", path)))
                    .transpose()?,
                attention_norm_b: attention_norm_name
                    .map(|path| tl.load(&format!("{}.bias", path)))
                    .transpose()?,

                query_key_value: tl.load(&format!(
                    "transformer.h.{i}.self_attention.query_key_value.weight"
                ))?,
                wo: tl.load(&format!("transformer.h.{i}.self_attention.dense.weight"))?,

                ffn_up: tl.load(&format!("transformer.h.{i}.mlp.dense_h_to_4h.weight"))?,
                ffn_down: tl.load(&format!("transformer.h.{i}.mlp.dense_4h_to_h.weight"))?,
            };

            layers.push(layer);
        }

        let context = tl.finish();

        Ok(Falcon {
            hyperparameters,
            params,
            tokenizer,
            tok_embeddings,
            output_norm,
            output_norm_b,
            lm_head,
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
            n_embd,
            n_head,
            n_head_kv,
            n_vocab,
            n_layer,
            ..
        } = self.hyperparameters;

        let head_dim = n_embd / n_head;
        let n = input_len;

        let outputs = session.compute(self.context.clone(), input_tokens, |builder| {
            let ctx0 = builder.ctx0.borrow();
            let embd = builder.embd;
            let mut input_layer = ctx0.op_get_rows(&self.tok_embeddings, embd);

            let f32_size = std::mem::size_of::<f32>();

            let memory_k = builder.memory_k;
            let memory_k_size = memory_k.element_size();

            let memory_v = builder.memory_v;
            let memory_v_size = memory_v.element_size();

            let mut gf = ggml::ComputationGraph::new();

            let mut current: Tensor;
            let mut layernorm_output: Tensor;

            for il in 0..n_layer {
                // attention uses first scratch buffer
                ctx0.use_scratch(builder.get_scratch(0));

                // self-attention
                layernorm_output = ctx0.op_norm(&input_layer);
                layernorm_output = ctx0.op_add(
                    &ctx0.op_mul(
                        &ctx0.op_repeat(&self.layers[il].input_layernorm, &layernorm_output),
                        &layernorm_output,
                    ),
                    &ctx0.op_repeat(&self.layers[il].input_layernorm_b, &layernorm_output),
                );

                if n_head_kv == 1 {
                    // Falcon-7B only
                    current = layernorm_output.share();
                } else {
                    // Falcon-40B only
                    current = ctx0.op_norm(&input_layer);
                    current = ctx0.op_add(
                        &ctx0.op_mul(
                            &ctx0.op_repeat(
                                self.layers[il].attention_norm.as_ref().unwrap(),
                                &current,
                            ),
                            &current,
                        ),
                        &ctx0.op_repeat(
                            self.layers[il].attention_norm_b.as_ref().unwrap(),
                            &current,
                        ),
                    );
                }

                // compute QKV
                current = ctx0.op_mul_mat(&self.layers[il].query_key_value, &current);

                let fused_qkv_row_nb = head_dim * (n_head + 2 * n_head_kv) * f32_size;

                let mut qcur = ctx0.op_view_3d(
                    &current,
                    (head_dim, n_head, n),
                    (head_dim * f32_size, fused_qkv_row_nb),
                    0,
                );

                let mut kcur = ctx0.op_view_3d(
                    &current,
                    (head_dim, n_head_kv, n),
                    (head_dim * f32_size, fused_qkv_row_nb),
                    head_dim * n_head * f32_size,
                );

                let vcur = ctx0.op_view_3d(
                    &current,
                    (head_dim, n_head_kv, n),
                    (head_dim * f32_size, fused_qkv_row_nb),
                    head_dim * (n_head + n_head_kv) * f32_size,
                );

                // using mode = 2 for neox mode
                let overrides = self.params.rope_overrides.as_ref();
                qcur = ctx0.op_rope_inplace(&qcur, session_len, head_dim, 2, overrides);
                kcur = ctx0.op_rope_inplace(&kcur, session_len, head_dim, 2, overrides);

                // store key and value to memory

                let k = ctx0.op_view_1d(
                    memory_k,
                    n * n_head_kv * head_dim,
                    (memory_k_size * n_head_kv * head_dim) * (il * ctx_size + session_len),
                );
                let v = ctx0.op_view_1d(
                    memory_v,
                    n * n_head_kv * head_dim,
                    (memory_v_size * n_head_kv * head_dim) * (il * ctx_size + session_len),
                );

                gf.build_forward_expand(&ctx0.op_cpy(&kcur, &k));
                gf.build_forward_expand(&ctx0.op_cpy(&vcur, &v));

                // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
                let bigq = ctx0.op_permute(&qcur, (0, 2, 1, 3));

                let bigk = ctx0.op_permute(
                    &ctx0.op_reshape_3d(
                        &ctx0.op_view_1d(
                            memory_k,
                            (session_len + n) * n_head_kv * head_dim,
                            il * ctx_size * memory_k_size * n_head_kv * head_dim,
                        ),
                        head_dim,
                        n_head_kv,
                        session_len + n,
                    ),
                    (0, 2, 1, 3),
                );

                // K * Q
                let big_kq = ctx0.op_mul_mat(&bigk, &bigq);

                // KQ_scaled = KQ / sqrt(n_embd/n_head)
                let big_kq_scaled = ctx0.op_scale_inplace(
                    &big_kq,
                    &ctx0.new_f32(1f32 / f32::sqrt(n_embd as f32 / n_head as f32)),
                );

                let big_kq_masked = ctx0.op_diag_mask_inf_inplace(&big_kq_scaled, session_len);

                let big_kq_softmax = ctx0.op_soft_max_inplace(&big_kq_masked);

                let mut bigv = ctx0.op_permute(
                    &ctx0.op_reshape_3d(
                        &ctx0.op_view_1d(
                            memory_v,
                            (session_len + n) * n_head_kv * head_dim,
                            il * ctx_size * memory_v_size * n_head_kv * head_dim,
                        ),
                        head_dim,
                        n_head_kv,
                        session_len + n,
                    ),
                    (0, 2, 1, 3),
                );
                bigv = ctx0.op_cont(&ctx0.op_transpose(&bigv));

                let big_kqv = ctx0.op_mul_mat(&bigv, &big_kq_softmax);
                // KQV_merged = KQV.permute(0, 2, 1, 3)
                let big_kqv_merged = ctx0.op_permute(&big_kqv, (0, 2, 1, 3));

                // cur = KQV_merged.contiguous().view(n_embd, N)
                current = ctx0.op_cpy(
                    &big_kqv_merged,
                    &ctx0.new_tensor_2d(ggml::Type::F32, n_embd, n),
                );

                // projection
                current = ctx0.op_mul_mat(&self.layers[il].wo, &current);

                // feed forward uses second scratch buffer
                ctx0.use_scratch(builder.get_scratch(1));

                let inp_ff = layernorm_output.share();
                let attn_out =
                    ctx0.op_cpy(&current, &ctx0.new_tensor_2d(ggml::Type::F32, n_embd, n));

                current = ctx0.op_mul_mat(&self.layers[il].ffn_up, &inp_ff);
                current = ctx0.op_gelu(&current);
                current = ctx0.op_mul_mat(&self.layers[il].ffn_down, &current);

                current = ctx0.op_add(&current, &attn_out);
                current = ctx0.op_add(&current, &input_layer);

                input_layer = current.share();
            }

            ctx0.use_scratch(builder.get_scratch(0));

            // norm
            input_layer = ctx0.op_norm(&input_layer);

            input_layer = ctx0.op_add(
                &ctx0.op_mul(
                    &ctx0.op_repeat(&self.output_norm, &input_layer),
                    &input_layer,
                ),
                &ctx0.op_repeat(&self.output_norm_b, &input_layer),
            );

            let embeddings_tensor: ggml::Tensor = input_layer.share();

            ctx0.use_scratch(None);

            // lm_head
            input_layer = ctx0.op_mul_mat(&self.lm_head, &input_layer);

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
        vec![Regex::new(".*weight").unwrap()]
    }

    fn skip_quantize_tensors() -> Vec<Regex> {
        vec![]
    }
}

/// Falcon [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))
#[derive(Debug, Default, PartialEq, Clone, Copy, Eq)]
pub struct Hyperparameters {
    /// Size of the model's vocabulary
    n_vocab: usize,
    /// Size of the model's embedding layer
    n_embd: usize,
    /// n_heads
    n_head: usize,
    // Number of heads for key-value pairs
    n_head_kv: usize,
    /// Number of layers in the model
    n_layer: usize,
    /// file_type
    file_type: FileType,
}

impl llm_base::Hyperparameters for Hyperparameters {
    fn read_ggml(reader: &mut dyn std::io::BufRead) -> Result<Self, LoadError> {
        let hyperparameters = Hyperparameters {
            n_vocab: util::read_i32(reader)?.try_into()?,
            n_embd: util::read_i32(reader)?.try_into()?,
            n_head: util::read_i32(reader)?.try_into()?,
            n_head_kv: util::read_i32(reader)?.try_into()?,
            n_layer: util::read_i32(reader)?.try_into()?,
            file_type: util::read_filetype(reader)?,
        };

        Ok(hyperparameters)
    }

    fn write_ggml(&self, writer: &mut dyn std::io::Write) -> Result<(), HyperparametersWriteError> {
        util::write_i32(writer, self.n_vocab.try_into()?)?;
        util::write_i32(writer, self.n_embd.try_into()?)?;
        util::write_i32(writer, self.n_head.try_into()?)?;
        util::write_i32(writer, self.n_head_kv.try_into()?)?;
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
    // normalization
    input_layernorm: Tensor,
    input_layernorm_b: Tensor,

    // Falcon-40B only
    attention_norm: Option<Tensor>,
    attention_norm_b: Option<Tensor>,

    // attention
    query_key_value: Tensor,
    wo: Tensor,

    // ff
    ffn_up: Tensor,
    ffn_down: Tensor,
}
