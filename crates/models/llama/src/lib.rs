//! An implementation of [LLaMA](https://huggingface.co/docs/transformers/model_doc/llama) for the `llm` ecosystem.
#![deny(missing_docs)]

use std::error::Error;

use llm_base::{
    ggml,
    model::{common, HyperparametersWriteError},
    util, FileType, InferenceParameters, InferenceSession, InferenceSessionConfig, KnownModel,
    LoadError, Mmap, ModelParameters, OutputRequest, TensorLoader, TokenId, Vocabulary,
};

/// The LLaMA model. Ref: [Introducing LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
///
/// # Safety
/// This implements [Send] and [Sync] as it is immutable after construction.
pub struct Llama {
    // the context size ("memory") the model should use when evaluating a prompt
    context_size: usize,

    hyperparameters: Hyperparameters,
    vocabulary: Vocabulary,

    // model-global weights
    // weighted token embeddings
    wte: ggml::Tensor,
    // normalization
    norm: ggml::Tensor,
    // output weight
    output: ggml::Tensor,

    // weights for the model
    layers: Vec<Layer>,

    // default parameters used by [InferenceSession::infer]
    inference_parameters: InferenceParameters,

    // must be kept alive for the model
    _context: ggml::Context,
    _mmap: Option<Mmap>,
}

unsafe impl Send for Llama {}
unsafe impl Sync for Llama {}

impl KnownModel for Llama {
    type Hyperparameters = Hyperparameters;
    type Overrides = ();

    fn new<E: Error>(
        hyperparameters: Self::Hyperparameters,
        params: ModelParameters,
        _overrides: Option<Self::Overrides>,
        vocabulary: Vocabulary,
        tensor_loader: impl TensorLoader<E>,
    ) -> Result<Self, E> {
        let mut tl = tensor_loader;

        // model-global weights
        let wte = tl.load("tok_embeddings.weight")?;
        let norm = tl.load("norm.weight")?;
        let output = tl.load("output.weight")?;

        let mut layers = Vec::new();
        for i in 0..hyperparameters.n_layer {
            let layer = Layer {
                attention_norm: tl.load(&format!("layers.{i}.attention_norm.weight"))?,
                wq: tl.load(&format!("layers.{i}.attention.wq.weight"))?,
                wk: tl.load(&format!("layers.{i}.attention.wk.weight"))?,
                wv: tl.load(&format!("layers.{i}.attention.wv.weight"))?,
                wo: tl.load(&format!("layers.{i}.attention.wo.weight"))?,
                ffn_norm: tl.load(&format!("layers.{i}.ffn_norm.weight"))?,
                w1: tl.load(&format!("layers.{i}.feed_forward.w1.weight"))?,
                w2: tl.load(&format!("layers.{i}.feed_forward.w2.weight"))?,
                w3: tl.load(&format!("layers.{i}.feed_forward.w3.weight"))?,
            };

            layers.push(layer);
        }

        let (_context, _tensors, _mmap) = tl.finish();

        let ModelParameters {
            context_size,
            inference_parameters,
            ..
        } = params;

        Ok(Self {
            hyperparameters,
            context_size,
            vocabulary,
            wte,
            norm,
            output,
            layers,
            inference_parameters,
            _context,
            _mmap,
        })
    }

    /// Starts a new `InferenceSession` for this model.
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
            n_vocab,
            n_embd,
            n_mult: _,
            n_head,
            n_layer,
            n_rot,
            file_type: _,
        } = self.hyperparameters;

        let (ctx0, embd) = common::prepare_for_evaluate(n_layer, session, input_tokens);

        let mut input_layer = ctx0.op_get_rows(&self.wte, &embd);

        let memory_k_size = session.memory_k.element_size();
        let memory_v_size = session.memory_v.element_size();

        let mut gf = ggml::ComputationGraph::new(num_threads);
        for il in 0..n_layer {
            let input_self_attention = input_layer.share();
            let mut current: ggml::Tensor;

            ctx0.use_scratch(Some(&mut session.scratch[0]));

            // norm
            current = ctx0.op_rms_norm(&input_layer);

            // cur = attention_norm * cur
            current = ctx0.op_mul(
                &ctx0.op_repeat(&self.layers[il].attention_norm, &current),
                &current,
            );

            // self-attention
            // compute Q and K and RoPE them
            let q_current = ctx0.op_rope(
                &ctx0.op_reshape_3d(
                    &ctx0.op_mul_mat(&self.layers[il].wq, &current),
                    n_embd / n_head,
                    n_head,
                    input_len,
                ),
                session_len,
                n_rot,
                0,
            );
            let k_current = ctx0.op_rope(
                &ctx0.op_reshape_3d(
                    &ctx0.op_mul_mat(&self.layers[il].wk, &current),
                    n_embd / n_head,
                    n_head,
                    input_len,
                ),
                session_len,
                n_rot,
                0,
            );

            // store key and value to memory
            // compute the transposed [N, n_embd] V matrix
            let v_current = ctx0.op_transpose(&ctx0.op_reshape_2d(
                &ctx0.op_mul_mat(&self.layers[il].wv, &current),
                n_embd,
                input_len,
            ));

            let k = ctx0.op_view_1d(
                &session.memory_k,
                input_len * n_embd,
                (memory_k_size * n_embd) * (il * ctx_size + session_len),
            );

            let v = ctx0.op_view_2d(
                &session.memory_v,
                (input_len, n_embd),
                ctx_size * memory_v_size,
                (il * ctx_size) * memory_v_size * n_embd + session_len * memory_v_size,
            );

            // important: storing RoPE-ed version of K in the KV cache!
            gf.build_forward_expand(&ctx0.op_cpy(&k_current, &k));
            gf.build_forward_expand(&ctx0.op_cpy(&v_current, &v));

            let q = ctx0.op_permute(&q_current, 0, 2, 1, 3);

            let k = ctx0.op_permute(
                &ctx0.op_reshape_3d(
                    &ctx0.op_view_1d(
                        &session.memory_k,
                        (session_len + input_len) * n_embd,
                        il * ctx_size * memory_k_size * n_embd,
                    ),
                    n_embd / n_head,
                    n_head,
                    session_len + input_len,
                ),
                0,
                2,
                1,
                3,
            );

            // K * Q
            let k_q = ctx0.op_mul_mat(&k, &q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            let k_q_scaled = ctx0.op_scale(
                &k_q,
                &ctx0.new_f32(1.0 / f32::sqrt(n_embd as f32 / n_head as f32)),
            );

            // KQ_masked = mask_past(KQ_scaled)
            let k_q_masked = ctx0.op_diag_mask_inf(&k_q_scaled, session_len);

            // KQ = soft_max(KQ_masked)
            let k_q_soft_max = ctx0.op_soft_max(&k_q_masked);

            // split cached V into n_head heads
            let v = ctx0.op_view_3d(
                &session.memory_v,
                (session_len + input_len, n_embd / n_head, n_head),
                (
                    ctx_size * memory_v_size,
                    ctx_size * memory_v_size * n_embd / n_head,
                ),
                il * ctx_size * memory_v_size * n_embd,
            );

            let k_q_v = ctx0.op_mul_mat(&v, &k_q_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            let k_q_v_merged = ctx0.op_permute(&k_q_v, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            current = ctx0.op_cpy(
                &k_q_v_merged,
                &ctx0.new_tensor_2d(ggml::Type::F32, n_embd, input_len),
            );

            // projection (no bias)
            current = ctx0.op_mul_mat(&self.layers[il].wo, &current);

            ctx0.use_scratch(Some(&mut session.scratch[1]));

            let input_feed_forward = ctx0.op_add(&current, &input_self_attention);

            // feed-forward network
            // norm
            current = ctx0.op_rms_norm(&input_feed_forward);

            // cur = ffn_norm*cur
            current = ctx0.op_mul(
                &ctx0.op_repeat(&self.layers[il].ffn_norm, &current),
                &current,
            );

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

        ctx0.use_scratch(Some(&mut session.scratch[0]));

        // Used at the end to optionally extract the embeddings.
        let embeddings_tensor: ggml::Tensor;

        // norm
        input_layer = ctx0.op_rms_norm(&input_layer);

        // inpL = norm*inpL
        input_layer = ctx0.op_mul(&ctx0.op_repeat(&self.norm, &input_layer), &input_layer);

        embeddings_tensor = input_layer.share();

        // lm_head
        input_layer = ctx0.op_mul_mat(&self.output, &input_layer);

        ctx0.use_scratch(None);

        // run the computation
        gf.build_forward_expand(&input_layer);
        ctx0.graph_compute(&mut gf);

        // finish evaluation
        common::read_last_token(session, &input_layer, n_vocab, input_len);
        common::extract_logits(output_request, &input_layer, n_vocab, input_len);
        common::extract_embeddings(output_request, &embeddings_tensor, n_embd, input_len);
        common::update_session(session, &ctx0, input_tokens.len(), input_len);
    }

    /// Returns the vocabulary used by this model.
    fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    fn context_size(&self) -> usize {
        self.context_size
    }

    fn bot_token_id(&self) -> Option<TokenId> {
        None
    }

    fn eot_token_id(&self) -> TokenId {
        2
    }

    fn inference_parameters(&self) -> &InferenceParameters {
        &self.inference_parameters
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
