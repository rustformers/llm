use std::collections::HashMap;

use ggml::loader::{LoadError, LoadProgress};
use llm_base::{
    EvaluateOutputRequest, InferenceParameters, InferenceSession, InferenceSessionParameters,
    Model, TokenId, Vocabulary,
};

mod ggml_loader;

pub use ggml_loader::load;

/// The weights for the BLOOM model. All the mutable state is split into a
/// separate struct `InferenceSession`.
pub struct Bloom {
    hparams: Hyperparameters,
    vocabulary: Vocabulary,
    tok_embeddings: ggml::Tensor,
    norm: ggml::Tensor,
    norm_b: ggml::Tensor,
    output_norm: ggml::Tensor,
    output_norm_b: ggml::Tensor,
    output: ggml::Tensor,
    layers: Vec<Layer>,
    tensors: HashMap<String, ggml::Tensor>,
    // Must be kept alive for the model
    _context: ggml::Context,
}

impl Model for Bloom {
    type Model = Bloom;
    type Hyperparameters = Hyperparameters;
    type Layer = Layer;

    fn start_session(&self, params: InferenceSessionParameters) -> InferenceSession {
        InferenceSession::new(
            params,
            self.hparams.n_ctx,
            self.hparams.n_layer,
            self.hparams.n_embd,
            self.hparams.n_vocab,
        )
    }

    fn evaluate(
        &self,
        session: &mut InferenceSession,
        params: &InferenceParameters,
        input_tokens: &[TokenId],
        output_request: &mut EvaluateOutputRequest,
    ) {
        let n = input_tokens.len();
        let n_past = session.n_past;
        let n_threads = params.n_threads;

        let Hyperparameters {
            n_vocab,
            n_ctx,
            n_embd,
            n_mult: _,
            n_head,
            n_layer,
            f16_: _,
        } = self.hparams;

        // For the first run, we need to guess a maximum buffer size so we can measure
        // the actual memory consumption of the temporary ggml context.
        let mut buf_size = 1024 * 1024 * 1024;
        if session.mem_per_token > 0 && session.mem_per_token * n > buf_size {
            // add 10% to account for ggml object overhead
            buf_size = (1.1f64 * session.mem_per_token as f64 * n as f64) as usize;
        };
        let ctx0 = ggml::Context::init(buf_size);

        // TODO: REMAKE THIS AFTER CHECKING GGML GRAPH
        let mut gf = ggml::ComputationGraph::new(n_threads);

        let embd = ctx0.new_tensor_1d(ggml::Type::I32, n);
        unsafe { embd.write_data(bytemuck::cast_slice(input_tokens)) };

        let mut input_layer = ctx0.op_get_rows(&self.tok_embeddings, &embd);

        //TODO: word embeddings norm,
        {
            input_layer = ctx0.op_norm(&input_layer);
            input_layer = ctx0.op_mul(&ctx0.op_repeat(&self.norm, &input_layer), &input_layer);
            input_layer = ctx0.op_add(&ctx0.op_repeat(&self.norm_b, &input_layer), &input_layer);
        }

        for il in 0..n_layer {
            let input_self_attention = input_layer.share();
            let mut current: ggml::Tensor;

            // norm
            {
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
            }

            //attention
            {
                current = ctx0.op_mul_mat(&self.layers[il].query_key_value, &current);
                current = ctx0.op_add(
                    &ctx0.op_repeat(&self.layers[il].query_key_value_b, &current),
                    &current,
                );
            }

            // self-attention
            {
                let nb = current.get_nb()[1];
                let q_current = ctx0.op_view_2d(
                    &current, n_embd, n, nb,
                    //0 * std::mem::size_of::<f32>() * n_embd as usize,
                    0,
                );
                let k_current =
                    ctx0.op_view_2d(&current, n_embd, n, nb, std::mem::size_of::<f32>() * n_embd);
                let v_current = ctx0.op_view_2d(
                    &current,
                    n_embd,
                    n,
                    nb,
                    2 * std::mem::size_of::<f32>() * n_embd,
                );

                // store key and value to memory
                if n >= 1 {
                    let k = ctx0.op_view_1d(
                        &session.memory_k,
                        n * n_embd,
                        (session.memory_k.element_size() * n_embd)
                            * (il * n_ctx + n_past),
                    );

                    let v = ctx0.op_view_1d(
                        &session.memory_v,
                        n * n_embd,
                        (session.memory_v.element_size() * n_embd)
                            * (il * n_ctx + n_past),
                    );

                    gf.build_forward_expand(&ctx0.op_cpy(&k_current, &k));
                    gf.build_forward_expand(&ctx0.op_cpy(&v_current, &v));
                }

                // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
                let q = ctx0.op_permute(
                    &ctx0.op_cpy(
                        &q_current,
                        &ctx0.new_tensor_3d(ggml::Type::F32, n_embd / n_head, n_head, n),
                    ),
                    0,
                    2,
                    1,
                    3,
                );

                // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
                let k = ctx0.op_permute(
                    &ctx0.op_reshape_3d(
                        &ctx0.op_view_1d(
                            &session.memory_k,
                            (n_past + n) * n_embd,
                            il * n_ctx * session.memory_k.element_size() * n_embd,
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

                // K * Q
                let k_q = ctx0.op_mul_mat(&k, &q);

                // KQ_scaled = KQ / sqrt(n_embd/n_head)
                let k_q_scaled = ctx0.op_scale(
                    &k_q,
                    &ctx0.new_f32(1.0 / f32::sqrt(n_embd as f32 / n_head as f32)),
                );

                //alibi
                // KQ_scaled_alibi = KQ_scaled + alibi_bias
                // TODO: op_alibi function
                let k_q_scaled_alibi = ctx0.op_alibi(&k_q_scaled, n_past, n_head);

                // KQ_masked = mask_past(KQ_scaled)
                let k_q_masked = ctx0.op_diag_mask_inf(&k_q_scaled_alibi, n_past);

                // KQ = soft_max(KQ_masked)
                let k_q_soft_max = ctx0.op_soft_max(&k_q_masked);

                let memv_elsize = session.memory_v.element_size();

                // split cached V into n_head heads
                let v = ctx0.op_view_3d(
                    &session.memory_v,
                    n_past + n,
                    n_embd / n_head,
                    n_head,
                    n_ctx * memv_elsize,
                    n_ctx * memv_elsize * n_embd / n_head,
                    il * n_ctx * memv_elsize * n_embd,
                );

                // KQV = transpose(V) * KQ_soft_max
                let k_q_v = ctx0.op_mul_mat(&v, &k_q_soft_max);

                // KQV_merged = KQV.permute(0, 2, 1, 3)
                let k_q_v_merged = ctx0.op_permute(&k_q_v, 0, 2, 1, 3);

                // cur = KQV_merged.contiguous().view(n_embd, N)
                current = ctx0.op_cpy(
                    &k_q_v_merged,
                    &ctx0.new_tensor_2d(ggml::Type::F32, n_embd, n),
                );

                // projection
                current = ctx0.op_mul_mat(&self.layers[il].wo, &current);
                current = ctx0.op_add(&ctx0.op_repeat(&self.layers[il].wo_b, &current), &current);
            }

            let input_feed_forward = ctx0.op_add(&current, &input_self_attention);

            // feed-forward network
            {
                // norm
                {
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
                }

                current = ctx0.op_mul_mat(&self.layers[il].w1, &current);

                current = ctx0.op_add(&ctx0.op_repeat(&self.layers[il].w1_b, &current), &current);

                // SILU activation

                current = ctx0.op_gelu(&current);

                current = ctx0.op_mul_mat(&self.layers[il].w2, &current);

                current = ctx0.op_add(&ctx0.op_repeat(&self.layers[il].w2_b, &current), &current);
            }

            current = ctx0.op_add(&current, &input_feed_forward);

            // input for next layer
            input_layer = current;
        }

        // Used at the end to optionally extract the embeddings.
        let embeddings_tensor;

        // norm
        {
            input_layer = ctx0.op_norm(&input_layer);

            // inpL = norm*inpL
            input_layer = ctx0.op_mul(
                &ctx0.op_repeat(&self.output_norm, &input_layer),
                &input_layer,
            );

            input_layer = ctx0.op_add(
                &ctx0.op_repeat(&self.output_norm_b, &input_layer),
                &input_layer,
            );

            embeddings_tensor = input_layer.share(); //TODO: CHECK if this is still necessary, (not in BLOOM C implementation)
        }

        // lm_head
        {
            input_layer = ctx0.op_mul_mat(&self.output, &input_layer);
        }

        // logits -> probs
        // inpL = ctx0.op_soft_max(&inpL);

        // run the computation
        gf.build_forward_expand(&input_layer);
        ctx0.graph_compute(&mut gf);

        // return result for just the last token
        // SAFETY: yolo
        assert_eq!(session.last_logits.len(), { n_vocab });
        unsafe {
            input_layer.read_data(
                n_vocab * (n - 1) * std::mem::size_of::<f32>(),
                bytemuck::cast_slice_mut(&mut session.last_logits),
            )
        };

        // Extract logits
        if let Some(all_logits) = &mut output_request.all_logits {
            all_logits.resize(n_vocab * n, 0.0);
            // SAFETY: Tensor data can be read (properly aligned, initialized,
            // data will not be mutated or otherwise aliased during the copy),
            // and we're not reading past the end of the tensor data.
            assert_eq!(input_layer.nelements(), n_vocab * n);
            unsafe {
                input_layer.read_data(0, bytemuck::cast_slice_mut(all_logits));
            }
        }

        // Extract embeddings
        if let Some(embeddings) = &mut output_request.embeddings {
            embeddings.resize(n_embd * n, 0.0);
            // SAFETY: Same rationale as for the "Extract logits" section applies.
            assert_eq!(embeddings_tensor.nelements(), n_embd * n);
            unsafe {
                embeddings_tensor.read_data(0, bytemuck::cast_slice_mut(embeddings));
            }
        }

        // Adjust the required memory per token if we didn't know that already
        if session.mem_per_token == 0 {
            session.mem_per_token = ctx0.used_mem() / n;
        }

        // Adjust n_past to new length.
        session.n_past += input_tokens.len();
    }

    /// Returns the vocabulary used by this model.
    fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    fn n_ctx(&self) -> usize {
        self.hparams.n_ctx
    }
}

impl Bloom {
    /// Load the model from `path` with `n_context_tokens` context tokens.
    ///
    /// The status of the loading process will be reported through `load_progress_callback`.
    pub fn load(
        path: impl AsRef<std::path::Path>,
        n_context_tokens: usize,
        load_progress_callback: impl FnMut(LoadProgress<Hyperparameters>),
    ) -> Result<Self, LoadError> {
        load(path, n_context_tokens, load_progress_callback)
    }

    pub(crate) fn new(
        context: ggml::Context,
        hparams: Hyperparameters,
        vocabulary: Vocabulary,
        n_ff: usize,
        wtype: ggml::Type,
    ) -> Bloom {
        let n_embd = hparams.n_embd;
        let n_layer = hparams.n_layer;
        let n_vocab = hparams.n_vocab;

        let mut tensors = HashMap::new();

        let tok_embeddings = context.new_tensor_2d(wtype, n_embd, n_vocab);

        let norm = context.new_tensor_1d(ggml::Type::F32, n_embd);
        let norm_b = context.new_tensor_1d(ggml::Type::F32, n_embd);

        let output_norm = context.new_tensor_1d(ggml::Type::F32, n_embd);
        let output_norm_b = context.new_tensor_1d(ggml::Type::F32, n_embd);

        let output = context.new_tensor_2d(wtype, n_embd, n_vocab);

        tensors.insert("tok_embeddings.weight".to_owned(), tok_embeddings.share());

        tensors.insert("norm.weight".to_owned(), norm.share());
        tensors.insert("norm.bias".to_owned(), norm_b.share());

        tensors.insert("output_norm.weight".to_owned(), output_norm.share());
        tensors.insert("output_norm.bias".to_owned(), output_norm_b.share());

        tensors.insert("output.weight".to_owned(), output.share());

        let mut layers = Vec::new();
        for i in 0..n_layer {
            let layer = Layer {
                attention_norm: context.new_tensor_1d(ggml::Type::F32, n_embd),
                attention_norm_b: context.new_tensor_1d(ggml::Type::F32, n_embd),

                query_key_value: context.new_tensor_2d(wtype, n_embd, 3 * n_embd),
                query_key_value_b: context.new_tensor_1d(ggml::Type::F32, 3 * n_embd),

                wo: context.new_tensor_2d(wtype, n_embd, n_embd),
                wo_b: context.new_tensor_1d(ggml::Type::F32, n_embd),

                ffn_norm: context.new_tensor_1d(ggml::Type::F32, n_embd),
                ffn_norm_b: context.new_tensor_1d(ggml::Type::F32, n_embd),

                w1: context.new_tensor_2d(wtype, n_embd, n_ff),
                w1_b: context.new_tensor_1d(ggml::Type::F32, n_ff),
                w2: context.new_tensor_2d(wtype, n_ff, n_embd),
                w2_b: context.new_tensor_1d(ggml::Type::F32, n_embd),
            };

            tensors.insert(
                format!("layers.{i}.attention_norm.weight"),
                layer.attention_norm.share(),
            );

            tensors.insert(
                format!("layers.{i}.attention_norm.bias"),
                layer.attention_norm_b.share(),
            );

            tensors.insert(
                format!("layers.{i}.attention.query_key_value.weight"),
                layer.query_key_value.share(),
            );
            tensors.insert(
                format!("layers.{i}.attention.query_key_value.bias"),
                layer.query_key_value_b.share(),
            );

            tensors.insert(format!("layers.{i}.attention.wo.weight"), layer.wo.share());
            tensors.insert(format!("layers.{i}.attention.wo.bias"), layer.wo_b.share());

            tensors.insert(
                format!("layers.{i}.ffn_norm.weight"),
                layer.ffn_norm.share(),
            );
            tensors.insert(
                format!("layers.{i}.ffn_norm.bias"),
                layer.ffn_norm_b.share(),
            );

            tensors.insert(
                format!("layers.{i}.feed_forward.w1.weight"),
                layer.w1.share(),
            );
            tensors.insert(
                format!("layers.{i}.feed_forward.w1.bias"),
                layer.w1_b.share(),
            );
            tensors.insert(
                format!("layers.{i}.feed_forward.w2.weight"),
                layer.w2.share(),
            );
            tensors.insert(
                format!("layers.{i}.feed_forward.w2.bias"),
                layer.w2_b.share(),
            );

            layers.push(layer);
        }

        Bloom {
            hparams,
            vocabulary,
            tok_embeddings,
            norm,
            norm_b,
            output_norm,
            output_norm_b,
            output,
            layers,
            tensors,
            _context: context,
        }
    }
}

// NOTE: Field order matters! Data is laid out in the file exactly
// in this order.
#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct Hyperparameters {
    pub n_vocab: usize,
    pub n_ctx: usize,
    pub n_embd: usize,
    pub n_mult: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub f16_: u32,
}

pub struct Layer {
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
