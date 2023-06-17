//! An implementation of [tiiuae](https://huggingface.co/tiiuae)'s [falcon] model for the `llm` ecosystem.
#![deny(missing_docs)]

use ggml::Tensor;
use llm_base::{
    ggml,
    model::{common, HyperparametersWriteError},
    util, FileType, InferenceParameters, InferenceSession, InferenceSessionConfig, KnownModel,
    LoadError, Mmap, ModelParameters, OutputRequest, Regex, TokenId, Vocabulary,
};

/// The falcon model. Ref: [Technology Innovation Institute](https://huggingface.co/tiiuae/falcon-40b)
///
/// # Safety
/// This implements [Send] and [Sync] as it is immutable after construction.
pub struct Falcon {
    // the context size ("memory") the model should use when evaluating a prompt
    context_size: usize,

    hyperparameters: Hyperparameters,

    vocabulary: Vocabulary,

    // model-global weights
    // weighted token embeddings
    tok_embeddings: Tensor,
    output_norm: Tensor,
    output_norm_b: Tensor,
    lm_head: Tensor,

    // weights for the model
    layers: Vec<Layer>,

    // must be kept alive for the model
    _context: ggml::Context,
    _mmap: Option<Mmap>,
}

unsafe impl Send for Falcon {}
unsafe impl Sync for Falcon {}

impl KnownModel for Falcon {
    type Hyperparameters = Hyperparameters;

    fn new<E: std::error::Error>(
        hyperparameters: Self::Hyperparameters,
        params: ModelParameters,
        vocabulary: Vocabulary,
        tensor_loader: impl llm_base::TensorLoader<E>,
    ) -> Result<Self, E> {
        let mut tl = tensor_loader;

        // model-gobal weights
        let tok_embeddings = tl.load("transformer.word_embeddings.weight")?;
        let output_norm = tl.load("transformer.ln_f.weight")?;
        let output_norm_b = tl.load("transformer.ln_f.bias")?;
        let lm_head = tl.load("lm_head.weight")?;

        let mut layers = Vec::new();
        for i in 0..hyperparameters.n_layer {
            let layer = Layer {
                attention_norm: tl.load(&format!("transformer.h.{i}.input_layernorm.weight"))?,
                attention_norm_b: tl.load(&format!("transformer.h.{i}.input_layernorm.bias"))?,

                query_key_value: tl.load(&format!(
                    "transformer.h.{i}.self_attention.query_key_value.weight"
                ))?,
                wo: tl.load(&format!("transformer.h.{i}.self_attention.dense.weight"))?,

                ffn_up: tl.load(&format!("transformer.h.{i}.mlp.dense_h_to_4h.weight"))?,
                ffn_down: tl.load(&format!("transformer.h.{i}.mlp.dense_4h_to_h.weight"))?,
            };

            layers.push(layer);
        }

        let (_context, _, _mmap) = tl.finish();

        let ModelParameters { context_size, .. } = params;

        Ok(Falcon {
            hyperparameters,
            context_size,
            vocabulary,
            tok_embeddings,
            output_norm,
            output_norm_b,
            lm_head,
            layers,
            _context,
            _mmap,
        })
    }

    fn start_session(&self, config: InferenceSessionConfig) -> InferenceSession {
        InferenceSession::new(
            config,
            self.context_size,
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
        let input_len = input_tokens.len();
        let session_len = session.n_past;
        let num_threads = params.n_threads;
        let ctx_size = self.context_size;

        let Hyperparameters {
            n_embd,
            n_head,
            n_vocab,
            n_layer,
            ..
        } = self.hyperparameters;

        let (ctx0, embd) = common::prepare_for_evaluate(n_layer, session, input_tokens);

        let mut input_layer = ctx0.op_get_rows(&self.tok_embeddings, &embd);

        let f32_size = std::mem::size_of::<f32>();

        let memory_k = &session.memory_k;
        let memory_k_size = memory_k.element_size();

        let memory_v = &session.memory_v;
        let memory_v_size = memory_v.element_size();

        let mut gf = ggml::ComputationGraph::new(num_threads);
        // for il in 0..n_layer {
        //     // attention uses first scratch buffer
        //     ctx0.use_scratch(Some(&mut session.scratch[0]));

        //     let mut current = ctx0.op_norm(&input_layer);
        //     current = ctx0.op_mul(
        //         &ctx0.op_repeat(&self.layers[il].norm_1_weight, &current),
        //         &current,
        //     );

        //     current = ctx0.op_mul_mat(&self.layers[il].c_attn_wqkv_weight, &current);

        //     let nb = current.get_nb()[1];
        //     let qcur = ctx0.op_view_2d(&current, (n_embd, input_len), nb, 0);
        //     let kcur = ctx0.op_view_2d(&current, (n_embd, input_len), nb, f32_size * n_embd);
        //     let vcur = ctx0.op_view_2d(&current, (n_embd, input_len), nb, f32_size * n_embd * 2);

        //     let k = ctx0.op_view_1d(
        //         memory_k,
        //         input_len * n_embd,
        //         (memory_k_size * n_embd) * (il * ctx_size + session_len),
        //     );
        //     let v = ctx0.op_view_1d(
        //         memory_v,
        //         input_len * n_embd,
        //         (memory_v_size * n_embd) * (il * ctx_size + session_len),
        //     );

        //     gf.build_forward_expand(&ctx0.op_cpy(&kcur, &k));
        //     gf.build_forward_expand(&ctx0.op_cpy(&vcur, &v));

        //     let q = ctx0.op_permute(
        //         &ctx0.op_cpy(
        //             &qcur,
        //             &ctx0.new_tensor_3d(ggml::Type::F32, n_embd / n_head, n_head, input_len),
        //         ),
        //         (0, 2, 1, 3),
        //     );

        //     let bigk = ctx0.op_permute(
        //         &ctx0.op_reshape_3d(
        //             &ctx0.op_view_1d(
        //                 memory_k,
        //                 (session_len + input_len) * n_embd,
        //                 il * ctx_size * memory_k_size * n_embd,
        //             ),
        //             n_embd / n_head,
        //             n_head,
        //             session_len + input_len,
        //         ),
        //         (0, 2, 1, 3),
        //     );

        //     let kq = ctx0.op_mul_mat(&bigk, &q);
        //     let kq_scaled = ctx0.op_scale(
        //         &kq,
        //         &ctx0.new_f32(1f32 / f32::sqrt(n_embd as f32 / n_head as f32)),
        //     );
        //     let kq_scaled_alibi = ctx0.op_alibi(&kq_scaled, session_len, n_head, alibi_bias_max);
        //     let kq_masked = ctx0.op_diag_mask_inf(&kq_scaled_alibi, session_len);
        //     let kq_softmax = ctx0.op_soft_max(&kq_masked);

        //     let v_trans = ctx0.op_cpy(
        //         &ctx0.op_permute(
        //             &ctx0.op_reshape_3d(
        //                 &ctx0.op_view_1d(
        //                     &session.memory_v,
        //                     (session_len + input_len) * n_embd,
        //                     il * ctx_size * memory_v_size * n_embd,
        //                 ),
        //                 n_embd / n_head,
        //                 n_head,
        //                 session_len + input_len,
        //             ),
        //             (1, 2, 0, 3),
        //         ),
        //         &ctx0.new_tensor_3d(
        //             session.memory_v.get_type(),
        //             session_len + input_len,
        //             n_embd / n_head,
        //             n_head,
        //         ),
        //     );

        //     let kqv = ctx0.op_mul_mat(&v_trans, &kq_softmax);
        //     let kqv_merged = ctx0.op_permute(&kqv, (0, 2, 1, 3));

        //     current = ctx0.op_cpy(
        //         &kqv_merged,
        //         &ctx0.new_tensor_2d(ggml::Type::F32, n_embd, input_len),
        //     );
        //     // projection
        //     current = ctx0.op_mul_mat(&self.layers[il].c_attn_out_proj_weight, &current);

        //     input_layer = ctx0.op_add(&input_layer, &current);

        //     // feed forward uses second scratch buffer
        //     ctx0.use_scratch(Some(&mut session.scratch[1]));

        //     current = ctx0.op_norm(&input_layer);
        //     current = ctx0.op_mul(
        //         &ctx0.op_repeat(&self.layers[il].norm_2_weight, &current),
        //         &current,
        //     );

        //     current = ctx0.op_mul_mat(&self.layers[il].ffn_up_proj, &current);

        //     current = ctx0.op_gelu(&current);

        //     // projection
        //     current = ctx0.op_mul_mat(&self.layers[il].ffn_down_proj, &current);

        //     input_layer = ctx0.op_add(&input_layer, &current);
        // }

        // //use scratch buffer 0 for the rest
        // ctx0.use_scratch(Some(&mut session.scratch[0]));

        // // norm
        // input_layer = ctx0.op_norm(&input_layer);
        // input_layer = ctx0.op_mul(&ctx0.op_repeat(&self.norm, &input_layer), &input_layer);

        // let embeddings_tensor: ggml::Tensor = input_layer.share();

        // // disable scratch buffer for last layer
        // ctx0.use_scratch(None);
        // // output embedding weight tied to input embedding
        // input_layer = ctx0.op_mul_mat(&self.wte, &input_layer);

        // // run the computation
        // gf.build_forward_expand(&input_layer);
        // ctx0.graph_compute(&mut gf);

        // // finish evaluation
        // common::read_last_token(session, &input_layer, n_vocab, input_len);
        // common::extract_logits(output_request, &input_layer, n_vocab, input_len);
        // common::extract_embeddings(output_request, &embeddings_tensor, n_embd, input_len);
        // common::update_session(session, &ctx0, input_tokens.len(), input_len);
    }

    /// Returns the vocabulary used by this model.
    fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    fn context_size(&self) -> usize {
        self.context_size
    }

    fn bot_token_id(&self) -> Option<TokenId> {
        self.vocabulary.id("<|padding|>".as_bytes())
    }

    fn eot_token_id(&self) -> TokenId {
        self.vocabulary.id("<|endoftext|>".as_bytes()).unwrap()
    }

    fn quantize_tensors() -> Vec<Regex> {
        vec![Regex::new(".*weight").unwrap()]
    }

    fn skip_quantize_tensors() -> Vec<Regex> {
        vec![]
    }
}

/// MPT [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))
#[derive(Debug, Default, PartialEq, Clone, Copy)]
pub struct Hyperparameters {
    /// Size of the model's vocabulary
    n_vocab: usize,
    /// Maximum sequence length
    n_ctx: usize,
    /// Size of the model's embedding layer
    n_embd: usize,
    /// n_heads
    n_head: usize,
    /// Number of layers in the model
    n_layer: usize,
    /// file_type
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

        Ok(hyperparameters)
    }

    fn write_ggml(&self, writer: &mut dyn std::io::Write) -> Result<(), HyperparametersWriteError> {
        util::write_i32(writer, self.n_vocab.try_into()?)?;
        util::write_i32(writer, self.n_embd.try_into()?)?;
        util::write_i32(writer, self.n_embd.try_into()?)?;
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
    // normalization
    attention_norm: Tensor,
    attention_norm_b: Tensor,

    // attention
    query_key_value: Tensor,
    wo: Tensor,

    // ff
    ffn_up: Tensor,
    ffn_down: Tensor,
}
