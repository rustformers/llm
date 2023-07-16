//! An implementation of [MPT](https://huggingface.co/mosaicml) for the `llm` ecosystem.
#![deny(missing_docs)]

use std::sync::Arc;

use ggml::Tensor;
use llm_base::{
    ggml::{self},
    model::{common, HyperparametersWriteError},
    util, FileType, GraphOutputs, InferenceSession, InferenceSessionConfig, KnownModel, LoadError,
    ModelParameters, OutputRequest, Regex, TokenId, Tokenizer,
};

/// The MosaicML Pretrained Transformer (MPT) model. Ref: [Mosaic ML](https://www.mosaicml.com/blog/mpt-7b)
///
/// # Safety
/// This implements [Send] and [Sync] as it is immutable after construction.
pub struct Mpt {
    params: ModelParameters,

    hyperparameters: Hyperparameters,
    tokenizer: Tokenizer,

    // model-global weights
    // weighted token embeddings
    wte: Tensor,
    // normalization
    norm: Tensor,

    // weights for the model
    layers: Vec<Layer>,

    // must be kept alive for the model
    context: Arc<ggml::Context>,
}

unsafe impl Send for Mpt {}
unsafe impl Sync for Mpt {}

impl KnownModel for Mpt {
    type Hyperparameters = Hyperparameters;

    fn new<E: std::error::Error>(
        hyperparameters: Self::Hyperparameters,
        params: ModelParameters,
        tokenizer: Tokenizer,
        tensor_loader: impl llm_base::TensorLoader<E>,
    ) -> Result<Self, E> {
        let mut tl = tensor_loader;

        // model-gobal weights
        let wte = tl.load("transformer.wte.weight")?;
        let norm = tl.load("transformer.norm_f.weight")?;

        let mut layers = Vec::new();
        for i in 0..hyperparameters.n_layer {
            let layer = Layer {
                norm_1_weight: tl.load(&format!("transformer.blocks.{i}.norm_1.weight"))?,
                c_attn_wqkv_weight: tl.load(&format!("transformer.blocks.{i}.attn.Wqkv.weight"))?,

                c_attn_out_proj_weight: tl
                    .load(&format!("transformer.blocks.{i}.attn.out_proj.weight"))?,
                norm_2_weight: tl.load(&format!("transformer.blocks.{i}.norm_2.weight"))?,

                ffn_up_proj: tl.load(&format!("transformer.blocks.{i}.ffn.up_proj.weight"))?,
                ffn_down_proj: tl.load(&format!("transformer.blocks.{i}.ffn.down_proj.weight"))?,
            };

            layers.push(layer);
        }

        let context = tl.finish();

        Ok(Mpt {
            hyperparameters,
            params,
            tokenizer,
            wte,
            norm,
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
        let n = input_tokens.len();
        let session_len = session.n_past;
        let ctx_size = self.params.context_size;

        let Hyperparameters {
            n_embd,
            n_head,
            n_vocab,
            n_layer,
            alibi_bias_max,
            ..
        } = self.hyperparameters;

        let outputs = session.compute(self.context.clone(), input_tokens, |builder| {
            let ctx0 = builder.ctx0.borrow();
            let (memory_k_size, memory_v_size) = (
                builder.memory_k.element_size(),
                builder.memory_v.element_size(),
            );
            let embd = builder.embd;

            let mut input_layer = ctx0.op_get_rows(&self.wte, embd);

            let f32_size = std::mem::size_of::<f32>();

            let mut gf = ggml::ComputationGraph::new();
            for il in 0..n_layer {
                // attention uses first scratch buffer
                ctx0.use_scratch(builder.get_scratch(0));

                let mut current = ctx0.op_norm(&input_layer);
                current = ctx0.op_mul(
                    &ctx0.op_repeat(&self.layers[il].norm_1_weight, &current),
                    &current,
                );

                current = ctx0.op_mul_mat(&self.layers[il].c_attn_wqkv_weight, &current);

                let nb = current.get_nb()[1];
                let qcur = ctx0.op_view_2d(&current, (n_embd, n), nb, 0);
                let kcur = ctx0.op_view_2d(&current, (n_embd, n), nb, f32_size * n_embd);
                let vcur = ctx0.op_view_2d(&current, (n_embd, n), nb, f32_size * n_embd * 2);

                let k = ctx0.op_view_1d(
                    builder.memory_k,
                    n * n_embd,
                    (memory_k_size * n_embd) * (il * ctx_size + session_len),
                );
                let v = ctx0.op_view_1d(
                    builder.memory_v,
                    n * n_embd,
                    (memory_v_size * n_embd) * (il * ctx_size + session_len),
                );

                gf.build_forward_expand(&ctx0.op_cpy(&kcur, &k));
                gf.build_forward_expand(&ctx0.op_cpy(&vcur, &v));

                let q = ctx0.op_permute(
                    &ctx0.op_cpy(
                        &qcur,
                        &ctx0.new_tensor_3d(ggml::Type::F32, n_embd / n_head, n_head, n),
                    ),
                    (0, 2, 1, 3),
                );

                let bigk = ctx0.op_permute(
                    &ctx0.op_reshape_3d(
                        &ctx0.op_view_1d(
                            builder.memory_k,
                            (session_len + n) * n_embd,
                            il * ctx_size * memory_k_size * n_embd,
                        ),
                        n_embd / n_head,
                        n_head,
                        session_len + n,
                    ),
                    (0, 2, 1, 3),
                );

                let kq = ctx0.op_mul_mat(&bigk, &q);
                let kq_scaled = ctx0.op_scale(
                    &kq,
                    &ctx0.new_f32(1f32 / f32::sqrt(n_embd as f32 / n_head as f32)),
                );
                let kq_scaled_alibi =
                    ctx0.op_alibi(&kq_scaled, session_len, n_head, alibi_bias_max);
                let kq_masked = ctx0.op_diag_mask_inf(&kq_scaled_alibi, session_len);
                let kq_softmax = ctx0.op_soft_max(&kq_masked);

                let v_trans = ctx0.op_cpy(
                    &ctx0.op_permute(
                        &ctx0.op_reshape_3d(
                            &ctx0.op_view_1d(
                                builder.memory_v,
                                (session_len + n) * n_embd,
                                il * ctx_size * memory_v_size * n_embd,
                            ),
                            n_embd / n_head,
                            n_head,
                            session_len + n,
                        ),
                        (1, 2, 0, 3),
                    ),
                    &ctx0.new_tensor_3d(
                        builder.memory_v.get_type(),
                        session_len + n,
                        n_embd / n_head,
                        n_head,
                    ),
                );

                let kqv = ctx0.op_mul_mat(&v_trans, &kq_softmax);
                let kqv_merged = ctx0.op_permute(&kqv, (0, 2, 1, 3));

                current = ctx0.op_cpy(&kqv_merged, &ctx0.new_tensor_2d(ggml::Type::F32, n_embd, n));
                // projection
                current = ctx0.op_mul_mat(&self.layers[il].c_attn_out_proj_weight, &current);

                input_layer = ctx0.op_add(&input_layer, &current);

                // feed forward uses second scratch buffer
                ctx0.use_scratch(builder.get_scratch(1));

                current = ctx0.op_norm(&input_layer);
                current = ctx0.op_mul(
                    &ctx0.op_repeat(&self.layers[il].norm_2_weight, &current),
                    &current,
                );

                current = ctx0.op_mul_mat(&self.layers[il].ffn_up_proj, &current);

                current = ctx0.op_gelu(&current);

                // projection
                current = ctx0.op_mul_mat(&self.layers[il].ffn_down_proj, &current);

                input_layer = ctx0.op_add(&input_layer, &current);
            }

            //use scratch buffer 0 for the rest
            ctx0.use_scratch(builder.get_scratch(0));

            // norm
            input_layer = ctx0.op_norm(&input_layer);
            input_layer = ctx0.op_mul(&ctx0.op_repeat(&self.norm, &input_layer), &input_layer);

            let embeddings_tensor: ggml::Tensor = input_layer.share();

            // disable scratch buffer for last layer
            ctx0.use_scratch(None);
            // output embedding weight tied to input embedding
            input_layer = ctx0.op_mul_mat(&self.wte, &input_layer);

            (
                gf,
                GraphOutputs {
                    result: input_layer,
                    embedding_result: embeddings_tensor,
                },
            )
        });

        // finish evaluation
        common::read_last_token(session, &outputs.result, n_vocab, n);
        common::extract_logits(output_request, &outputs.result, n_vocab, n);
        common::extract_embeddings(output_request, &outputs.embedding_result, n_embd, n);
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
        self.tokenizer.id("<|padding|>".as_bytes())
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

    fn supports_rewind(&self) -> bool {
        true
    }
}

/// MPT [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))
#[derive(Debug, Default, PartialEq, Clone, Copy)]
pub struct Hyperparameters {
    /// Size of the model's embedding layer
    n_embd: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// n_heads
    n_head: usize,
    /// Number of layers in the model
    n_layer: usize,
    /// Size of the model's vocabulary
    n_vocab: usize,
    /// Alibi bias max
    alibi_bias_max: f32,
    /// Clip KQV
    clip_kqv: f32,
    /// file_type
    file_type: FileType,
}
impl Eq for Hyperparameters {}

impl llm_base::Hyperparameters for Hyperparameters {
    fn read_ggml(reader: &mut dyn std::io::BufRead) -> Result<Self, LoadError> {
        let hyperparameters = Hyperparameters {
            n_embd: util::read_i32(reader)?.try_into()?,
            max_seq_len: util::read_i32(reader)?.try_into()?,
            n_head: util::read_i32(reader)?.try_into()?,
            n_layer: util::read_i32(reader)?.try_into()?,
            n_vocab: util::read_i32(reader)?.try_into()?,
            alibi_bias_max: util::read_f32(reader)?,
            clip_kqv: util::read_f32(reader)?,
            file_type: util::read_filetype(reader)?,
        };

        Ok(hyperparameters)
    }

    fn write_ggml(&self, writer: &mut dyn std::io::Write) -> Result<(), HyperparametersWriteError> {
        util::write_i32(writer, self.n_embd.try_into()?)?;
        util::write_i32(writer, self.max_seq_len.try_into()?)?;
        util::write_i32(writer, self.n_head.try_into()?)?;
        util::write_i32(writer, self.n_layer.try_into()?)?;
        util::write_i32(writer, self.n_vocab.try_into()?)?;
        util::write_f32(writer, self.alibi_bias_max)?;
        util::write_f32(writer, self.clip_kqv)?;
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
    // pre normalization
    norm_1_weight: Tensor,

    // attention
    c_attn_wqkv_weight: Tensor,
    c_attn_out_proj_weight: Tensor,

    // post normalization
    norm_2_weight: Tensor,

    // ff
    ffn_up_proj: Tensor,
    ffn_down_proj: Tensor,
}
