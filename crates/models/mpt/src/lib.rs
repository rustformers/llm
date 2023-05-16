//! An implementation of [MPT](https://huggingface.co/mosaicml) for the `llm` ecosystem.
#![deny(missing_docs)]

use ggml::Tensor;
use llm_base::{
    ggml,
    model::{common, HyperparametersWriteError},
    util, FileType, InferenceParameters, InferenceSession, InferenceSessionConfig, KnownModel,
    LoadError, ModelParameters, OutputRequest, TokenId, Vocabulary,
};

/// The MosaicML Pretrained Transformer (MPT) model. Ref: [Mosaic ML](https://www.mosaicml.com/blog/mpt-7b)
///
/// # Safety
/// This implements [Send] and [Sync] as it is immutable after construction.
pub struct Mpt {
    hyperparameters: Hyperparameters,
    n_context_tokens: usize,

    vocabulary: Vocabulary,

    // position embedding
    wte_weight: Tensor,

    // language model head
    norm_f_weight: Tensor,

    layers: Vec<Layer>,

    inference_parameters: InferenceParameters,

    _context: ggml::Context,

    _mmap: Option<llm_base::Mmap>,
}

unsafe impl Send for Mpt {}
unsafe impl Sync for Mpt {}

impl KnownModel for Mpt {
    type Hyperparameters = Hyperparameters;
    type Overrides = ();

    fn new<E: std::error::Error>(
        hyperparameters: Self::Hyperparameters,
        params: ModelParameters,
        _overrides: Option<Self::Overrides>,
        vocabulary: Vocabulary,
        tensor_loader: impl llm_base::TensorLoader<E>,
    ) -> Result<Self, E> {
        let mut tl = tensor_loader;

        // prepare memory for weights
        let wte_weight = tl.load("transformer.wte.weight")?;
        let norm_f_weight = tl.load("transformer.norm_f.weight")?;

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

        let (_context, _, _mmap) = tl.finish();

        let ModelParameters {
            n_context_tokens,
            inference_parameters,
            ..
        } = params;

        Ok(Mpt {
            hyperparameters,
            n_context_tokens,
            vocabulary,
            wte_weight,
            norm_f_weight,
            layers,
            inference_parameters,
            _context,
            _mmap,
        })
    }

    fn start_session(&self, config: InferenceSessionConfig) -> InferenceSession {
        InferenceSession::new(
            config,
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
        output_request: &mut OutputRequest,
    ) {
        let n = input_tokens.len();
        let n_threads = params.n_threads;

        let Hyperparameters {
            n_embd,
            n_head,
            n_vocab,
            n_layer,
            ..
        } = self.hyperparameters;
        let n_ctx = self.n_context_tokens;

        let (ctx0, embd) = common::prepare_for_evaluate(n_layer, session, input_tokens);

        let n_past = session.n_past;

        let mut input_layer = ctx0.op_get_rows(&self.wte_weight, &embd);

        let f32_size = std::mem::size_of::<f32>();

        let memory_k = &session.memory_k;
        let memory_k_size = memory_k.element_size();

        let memory_v = &session.memory_v;
        let memory_v_size = memory_v.element_size();

        let mut gf = ggml::ComputationGraph::new(n_threads);

        for il in 0..n_layer {
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
                memory_k,
                n * n_embd,
                (memory_k_size * n_embd) * (il * n_ctx + n_past),
            );
            let v = ctx0.op_view_1d(
                memory_v,
                n * n_embd,
                (memory_v_size * n_embd) * (il * n_ctx + n_past),
            );

            gf.build_forward_expand(&ctx0.op_cpy(&kcur, &k));
            gf.build_forward_expand(&ctx0.op_cpy(&vcur, &v));

            let q = ctx0.op_permute(
                &ctx0.op_cpy(
                    &qcur,
                    &ctx0.new_tensor_3d(ggml::Type::F32, n_embd / n_head, n_head, n),
                ),
                0,
                2,
                1,
                3,
            );

            let bigk = ctx0.op_permute(
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

            let kq = ctx0.op_mul_mat(&bigk, &q);
            let kq_scaled = ctx0.op_scale(
                &kq,
                &ctx0.new_f32(1f32 / f32::sqrt(n_embd as f32 / n_head as f32)),
            );
            let kq_scaled_alibi = ctx0.op_alibi(&kq_scaled, n_past, n_head);
            let kq_masked = ctx0.op_diag_mask_inf(&kq_scaled_alibi, n_past);
            let kq_softmax = ctx0.op_soft_max(&kq_masked);

            let v_trans = ctx0.op_cpy(
                &ctx0.op_permute(
                    &ctx0.op_reshape_3d(
                        &ctx0.op_view_1d(
                            &session.memory_v,
                            (n_past + n) * n_embd,
                            il * n_ctx * memory_v_size * n_embd,
                        ),
                        n_embd / n_head,
                        n_head,
                        n_past + n,
                    ),
                    1,
                    2,
                    0,
                    3,
                ),
                &ctx0.new_tensor_3d(
                    session.memory_v.get_type(),
                    n_past + n,
                    n_embd / n_head,
                    n_head,
                ),
            );

            let kqv = ctx0.op_mul_mat(&v_trans, &kq_softmax);
            let kqv_merged = ctx0.op_permute(&kqv, 0, 2, 1, 3);

            current = ctx0.op_cpy(&kqv_merged, &ctx0.new_tensor_2d(ggml::Type::F32, n_embd, n));
            // projection
            current = ctx0.op_mul_mat(&self.layers[il].c_attn_out_proj_weight, &current);

            input_layer = ctx0.op_add(&input_layer, &current);

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

        // norm
        input_layer = ctx0.op_norm(&input_layer);
        input_layer = ctx0.op_mul(
            &ctx0.op_repeat(&self.norm_f_weight, &input_layer),
            &input_layer,
        );

        // output embedding weight tied to input embedding
        input_layer = ctx0.op_mul_mat(&self.wte_weight, &input_layer);

        // run the computation
        gf.build_forward_expand(&input_layer);
        ctx0.graph_compute(&mut gf);

        // finish evaluation
        common::read_last_token(session, &input_layer, n_vocab, n);
        common::extract_logits(output_request, &input_layer, n_vocab, n);
        common::extract_embeddings(output_request, &embd, n_embd, n);
        common::update_session(session, &ctx0, input_tokens.len(), n);
    }

    /// Returns the vocabulary used by this model.
    fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    fn n_context_tokens(&self) -> usize {
        self.n_context_tokens
    }

    fn bot_token_id(&self) -> Option<TokenId> {
        self.vocabulary
            .token_to_id
            .get("<|padding|>".as_bytes())
            .copied()
    }

    fn eot_token_id(&self) -> TokenId {
        self.vocabulary
            .token_to_id
            .get("<|endoftext|>".as_bytes())
            .copied()
            .unwrap()
    }

    fn inference_parameters(&self) -> &InferenceParameters {
        &self.inference_parameters
    }
}

/// MPT [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
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
    /// file_type
    file_type: FileType,
}
impl llm_base::Hyperparameters for Hyperparameters {
    fn read_ggml(reader: &mut dyn std::io::BufRead) -> Result<Self, LoadError> {
        let hyperparameters = Hyperparameters {
            n_embd: util::read_i32(reader)?.try_into()?,
            max_seq_len: util::read_i32(reader)?.try_into()?,
            n_head: util::read_i32(reader)?.try_into()?,
            n_layer: util::read_i32(reader)?.try_into()?,
            n_vocab: util::read_i32(reader)?.try_into()?,
            file_type: {
                let ftype = util::read_i32(reader)?;
                FileType::try_from(ftype).map_err(|_| LoadError::UnsupportedFileType(ftype))?
            },
        };

        Ok(hyperparameters)
    }

    fn write_ggml(&self, writer: &mut dyn std::io::Write) -> Result<(), HyperparametersWriteError> {
        util::write_i32(writer, self.n_embd.try_into()?)?;
        util::write_i32(writer, self.max_seq_len.try_into()?)?;
        util::write_i32(writer, self.n_head.try_into()?)?;
        util::write_i32(writer, self.n_layer.try_into()?)?;
        util::write_i32(writer, self.n_vocab.try_into()?)?;
        util::write_i32(writer, self.file_type.into())?;
        Ok(())
    }

    fn n_vocabulary(&self) -> usize {
        self.n_vocab
    }

    fn file_type(&self) -> Option<FileType> {
        Some(self.file_type)
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
