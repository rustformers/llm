use std::{collections::HashMap, error::Error, path::Path};

use crate::{
    loader, loader2, loader_common::FileType, vocabulary::TokenId, EvaluateOutputRequest,
    InferenceParameters, InferenceSession, InferenceSessionParameters, LoadError, LoadProgress,
    Vocabulary,
};
use memmap2::Mmap;

/// The weights for the LLaMA model. All the mutable state is split into a
/// separate struct `InferenceSession`.
pub struct Model {
    hyperparameters: Hyperparameters,

    vocabulary: Vocabulary,

    tok_embeddings: ggml::Tensor,

    norm: ggml::Tensor,
    output: ggml::Tensor,

    layers: Vec<Layer>,

    tensors: HashMap<String, ggml::Tensor>,

    /// Needs to kept alive while the model is alive
    _mmap: Option<Mmap>,

    // Must be kept alive for the model
    _context: ggml::Context,
}
impl Model {
    pub(crate) fn new_loader1(
        context: ggml::Context,
        hparams: Hyperparameters,
        vocabulary: Vocabulary,
        n_ff: usize,
        wtype: ggml::Type,
        mmap: Option<Mmap>,
    ) -> Model {
        let n_embd = hparams.n_embd;
        let n_layer = hparams.n_layer;
        let n_vocab = hparams.n_vocab;

        let mut tensors = HashMap::new();

        let tok_embeddings = context.new_tensor_2d(wtype, n_embd, n_vocab);
        let norm = context.new_tensor_1d(ggml::Type::F32, n_embd);
        let output = context.new_tensor_2d(wtype, n_embd, n_vocab);

        tensors.insert("tok_embeddings.weight".to_owned(), tok_embeddings.share());
        tensors.insert("norm.weight".to_owned(), norm.share());
        tensors.insert("output.weight".to_owned(), output.share());

        let mut layers = Vec::new();
        for i in 0..n_layer {
            let layer = Layer {
                attention_norm: context.new_tensor_1d(ggml::Type::F32, n_embd),
                wq: context.new_tensor_2d(wtype, n_embd, n_embd),
                wk: context.new_tensor_2d(wtype, n_embd, n_embd),
                wv: context.new_tensor_2d(wtype, n_embd, n_embd),
                wo: context.new_tensor_2d(wtype, n_embd, n_embd),
                ffn_norm: context.new_tensor_1d(ggml::Type::F32, n_embd),
                w1: context.new_tensor_2d(wtype, n_embd, n_ff),
                w2: context.new_tensor_2d(wtype, n_ff, n_embd),
                w3: context.new_tensor_2d(wtype, n_embd, n_ff),
            };

            tensors.insert(
                format!("layers.{i}.attention_norm.weight"),
                layer.attention_norm.share(),
            );

            tensors.insert(format!("layers.{i}.attention.wq.weight"), layer.wq.share());
            tensors.insert(format!("layers.{i}.attention.wk.weight"), layer.wk.share());
            tensors.insert(format!("layers.{i}.attention.wv.weight"), layer.wv.share());
            tensors.insert(format!("layers.{i}.attention.wo.weight"), layer.wo.share());

            tensors.insert(
                format!("layers.{i}.ffn_norm.weight"),
                layer.ffn_norm.share(),
            );

            tensors.insert(
                format!("layers.{i}.feed_forward.w1.weight"),
                layer.w1.share(),
            );
            tensors.insert(
                format!("layers.{i}.feed_forward.w2.weight"),
                layer.w2.share(),
            );
            tensors.insert(
                format!("layers.{i}.feed_forward.w3.weight"),
                layer.w3.share(),
            );

            layers.push(layer);
        }

        Model {
            hyperparameters: hparams,
            vocabulary,
            tok_embeddings,
            norm,
            output,
            layers,
            tensors,
            _context: context,
            _mmap: mmap,
        }
    }

    pub(crate) fn new_loader2<E: Error>(
        hyperparameters: Hyperparameters,
        vocabulary: Vocabulary,
        n_ff: usize,
        tensor_loader: impl TensorLoader<E>,
    ) -> Result<Model, E> {
        let n_embd = hyperparameters.n_embd;
        let n_layer = hyperparameters.n_layer;
        let n_vocab = hyperparameters.n_vocab;

        let mut tl = tensor_loader;

        let tok_embeddings = tl.load("tok_embeddings.weight", &[n_embd, n_vocab])?;
        let norm = tl.load("norm.weight", &[n_embd])?;
        let output = tl.load("output.weight", &[n_embd, n_vocab])?;

        let mut layers = Vec::new();
        for i in 0..n_layer {
            let layer = Layer {
                attention_norm: tl.load(&format!("layers.{i}.attention_norm.weight"), &[n_embd])?,
                wq: tl.load(
                    &format!("layers.{i}.attention.wq.weight"),
                    &[n_embd, n_embd],
                )?,
                wk: tl.load(
                    &format!("layers.{i}.attention.wk.weight"),
                    &[n_embd, n_embd],
                )?,
                wv: tl.load(
                    &format!("layers.{i}.attention.wv.weight"),
                    &[n_embd, n_embd],
                )?,
                wo: tl.load(
                    &format!("layers.{i}.attention.wo.weight"),
                    &[n_embd, n_embd],
                )?,
                ffn_norm: tl.load(&format!("layers.{i}.ffn_norm.weight"), &[n_embd])?,
                w1: tl.load(
                    &format!("layers.{i}.feed_forward.w1.weight"),
                    &[n_embd, n_ff],
                )?,
                w2: tl.load(
                    &format!("layers.{i}.feed_forward.w2.weight"),
                    &[n_ff, n_embd],
                )?,
                w3: tl.load(
                    &format!("layers.{i}.feed_forward.w3.weight"),
                    &[n_embd, n_ff],
                )?,
            };

            layers.push(layer);
        }

        let (_context, tensors, _mmap) = tl.finish();

        Ok(Model {
            hyperparameters,
            vocabulary,
            tok_embeddings,
            norm,
            output,
            layers,
            tensors,
            _context,
            _mmap,
        })
    }

    /// Load the model from `path` with `n_context_tokens` context tokens.
    ///
    /// The status of the loading process will be reported through `load_progress_callback`.
    pub fn load(
        path: impl AsRef<Path>,
        prefer_mmap: bool,
        n_context_tokens: usize,
        load_progress_callback: impl FnMut(LoadProgress),
    ) -> Result<Model, LoadError> {
        // Loader2 is the default. It can support GGML, GGMF and GGJT, but does not support multipart models.
        //
        // Loader1 is the old loader. It can support multipart models, but will be deprecated.
        let use_loader_2: bool = match std::env::var("GGML_LOADER").as_deref() {
            Ok("2") => true,
            Ok("1") => false,
            Ok(_) => panic!("Please use GGML_LOADER=1 or GGML_LOADER=2"),
            Err(_) => true,
        };

        if use_loader_2 {
            loader2::load(path, prefer_mmap, n_context_tokens, load_progress_callback)
        } else {
            loader::load(path, prefer_mmap, n_context_tokens, load_progress_callback)
        }
    }

    /// Starts a new `InferenceSession` for this model.
    pub fn start_session(&self, params: InferenceSessionParameters) -> InferenceSession {
        InferenceSession::new(
            params,
            self.hyperparameters.n_ctx,
            self.hyperparameters.n_layer,
            self.hyperparameters.n_embd,
            self.hyperparameters.n_vocab,
        )
    }

    /// Evaluates the transformer.
    ///
    /// The provided `output_request` struct lets you specify which additional
    /// data you are interested in fetching from the transformer. Setting a
    /// field to a `Some` value will clear and fill the provided vector with
    /// data. The provided vector will be resized to the exact output size.
    pub fn evaluate(
        &self,
        session: &mut InferenceSession,
        params: &InferenceParameters,
        input_tokens: &[TokenId],
        output_request: &mut EvaluateOutputRequest,
    ) {
        let n = input_tokens.len();
        let n_past = session.n_past;
        let n_threads = params.n_threads;

        let memk_elsize = session.memory_k.element_size();
        let memv_elsize = session.memory_v.element_size();

        let Hyperparameters {
            n_vocab,
            n_ctx,
            n_embd,
            n_mult: _,
            n_head,
            n_layer,
            n_rot,
            file_type: _,
        } = self.hyperparameters;

        // For the first run, we need to guess a maximum buffer size so we can measure
        // the actual memory consumption of the temporary ggml context.
        //
        // These numbers are from `llama.cpp`, and could potentially be more efficient.
        let mut buf_size = {
            let buf_size_mb = if n_layer >= 80 {
                1536
            } else if n_layer >= 60 {
                1280
            } else {
                1024
            };
            buf_size_mb * 1024 * 1024
        };
        if session.mem_per_token > 0 && session.mem_per_token * n > buf_size {
            // add 10% to account for ggml object overhead
            buf_size = (1.1f64 * session.mem_per_token as f64 * n as f64) as usize;
        };
        let ctx0 = ggml::Context::init(buf_size, true);

        let mut gf = ggml::ComputationGraph::new(n_threads);

        let mut embd = ctx0.new_tensor_1d(ggml::Type::I32, n);
        unsafe { embd.write_data(bytemuck::cast_slice(input_tokens)) };

        let mut input_layer = ctx0.op_get_rows(&self.tok_embeddings, &embd);

        for il in 0..n_layer {
            let input_self_attention = input_layer.share();
            let mut current: ggml::Tensor;

            ctx0.use_scratch(Some(&mut session.scratch[0]));

            // norm
            {
                current = ctx0.op_rms_norm(&input_layer);

                // cur = attention_norm * cur
                current = ctx0.op_mul(
                    &ctx0.op_repeat(&self.layers[il].attention_norm, &current),
                    &current,
                );
            }

            // self-attention
            {
                // compute Q and K and RoPE them
                let q_current = ctx0.op_rope(
                    &ctx0.op_reshape_3d(
                        &ctx0.op_mul_mat(&self.layers[il].wq, &current),
                        n_embd / n_head,
                        n_head,
                        n,
                    ),
                    n_past,
                    n_rot,
                    0,
                );
                let k_current = ctx0.op_rope(
                    &ctx0.op_reshape_3d(
                        &ctx0.op_mul_mat(&self.layers[il].wk, &current),
                        n_embd / n_head,
                        n_head,
                        n,
                    ),
                    n_past,
                    n_rot,
                    0,
                );

                // store key and value to memory
                {
                    // compute the transposed [N, n_embd] V matrix
                    let v_current = ctx0.op_transpose(&ctx0.op_reshape_2d(
                        &ctx0.op_mul_mat(&self.layers[il].wv, &current),
                        n_embd,
                        n,
                    ));

                    let k = ctx0.op_view_1d(
                        &session.memory_k,
                        n * n_embd,
                        (memk_elsize * n_embd) * (il * n_ctx + n_past),
                    );

                    let v = ctx0.op_view_2d(
                        &session.memory_v,
                        n,
                        n_embd,
                        n_ctx * memv_elsize,
                        (il * n_ctx) * memv_elsize * n_embd + n_past * memv_elsize,
                    );

                    // important: storing RoPE-ed version of K in the KV cache!
                    gf.build_forward_expand(&ctx0.op_cpy(&k_current, &k));
                    gf.build_forward_expand(&ctx0.op_cpy(&v_current, &v));
                }

                let q = ctx0.op_permute(&q_current, 0, 2, 1, 3);

                let k = ctx0.op_permute(
                    &ctx0.op_reshape_3d(
                        &ctx0.op_view_1d(
                            &session.memory_k,
                            (n_past + n) * n_embd,
                            il * n_ctx * memk_elsize * n_embd,
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

                // KQ_masked = mask_past(KQ_scaled)
                let k_q_masked = ctx0.op_diag_mask_inf(&k_q_scaled, n_past);

                // KQ = soft_max(KQ_masked)
                let k_q_soft_max = ctx0.op_soft_max(&k_q_masked);

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

                let k_q_v = ctx0.op_mul_mat(&v, &k_q_soft_max);

                // KQV_merged = KQV.permute(0, 2, 1, 3)
                let k_q_v_merged = ctx0.op_permute(&k_q_v, 0, 2, 1, 3);

                // cur = KQV_merged.contiguous().view(n_embd, N)
                current = ctx0.op_cpy(
                    &k_q_v_merged,
                    &ctx0.new_tensor_2d(ggml::Type::F32, n_embd, n),
                );

                // projection (no bias)
                current = ctx0.op_mul_mat(&self.layers[il].wo, &current);
            }

            ctx0.use_scratch(Some(&mut session.scratch[1]));

            let input_feed_forward = ctx0.op_add(&current, &input_self_attention);

            // feed-forward network
            {
                // norm
                {
                    current = ctx0.op_rms_norm(&input_feed_forward);

                    // cur = ffn_norm*cur
                    current = ctx0.op_mul(
                        &ctx0.op_repeat(&self.layers[il].ffn_norm, &current),
                        &current,
                    );
                }

                let tmp = ctx0.op_mul_mat(&self.layers[il].w3, &current);

                current = ctx0.op_mul_mat(&self.layers[il].w1, &current);

                // SILU activation
                current = ctx0.op_silu(&current);

                current = ctx0.op_mul(&current, &tmp);

                current = ctx0.op_mul_mat(&self.layers[il].w2, &current);
            }

            current = ctx0.op_add(&current, &input_feed_forward);

            // input for next layer
            input_layer = current;
        }

        ctx0.use_scratch(Some(&mut session.scratch[0]));

        // Used at the end to optionally extract the embeddings.
        let embeddings_tensor;

        // norm
        {
            input_layer = ctx0.op_rms_norm(&input_layer);

            // inpL = norm*inpL
            input_layer = ctx0.op_mul(&ctx0.op_repeat(&self.norm, &input_layer), &input_layer);
            embeddings_tensor = input_layer.share();
        }

        // lm_head
        {
            input_layer = ctx0.op_mul_mat(&self.output, &input_layer);
        }

        ctx0.use_scratch(None);

        // logits -> probs
        // inpL = ctx0.op_soft_max(&inpL);

        // run the computation
        gf.build_forward_expand(&input_layer);
        ctx0.graph_compute(&mut gf);

        // return result for just the last token
        // SAFETY: yolo
        assert_eq!(session.last_logits.len(), n_vocab);
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
    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    pub(crate) fn tensors_mut(&mut self) -> &mut HashMap<String, ggml::Tensor> {
        &mut self.tensors
    }

    pub(crate) fn n_ctx(&self) -> usize {
        self.hyperparameters.n_ctx
    }
}

/// The hyperparameters of the model.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct Hyperparameters {
    /// n_vocab
    pub n_vocab: usize,
    /// n_ctx
    pub n_ctx: usize,
    /// n_embd
    pub n_embd: usize,
    /// n_mult
    pub n_mult: usize,
    /// n_head
    pub n_head: usize,
    /// n_layer
    pub n_layer: usize,
    /// n_rot
    pub n_rot: usize,
    /// file_type
    pub file_type: FileType,
}

pub(crate) trait TensorLoader<E: Error> {
    fn load(&mut self, name: &str, ne: &[usize]) -> Result<ggml::Tensor, E>;
    fn finish(self) -> (ggml::Context, HashMap<String, ggml::Tensor>, Option<Mmap>);
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
