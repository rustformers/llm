use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use crate::common::{helpers::*, inference::*, load::*, model::*, token::*, vocabulary::*};
use crate::mulf;

// NOTE: Field order matters! Data is laid out in the file exactly
// in this order.
#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct Hyperparameters {
    n_vocab: usize,
    n_ctx: usize,
    n_embd: usize,
    n_mult: usize,
    n_head: usize,
    n_layer: usize,
    n_rot: usize,
    f16_: u32,
}

// default
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

/// The weights for the LLaMA model. All the mutable state is split into a
/// separate struct `InferenceSession`.
pub struct Llama {
    hparams: Hyperparameters,

    tok_embeddings: ggml::Tensor,

    norm: ggml::Tensor,
    output: ggml::Tensor,

    layers: Vec<Layer>,

    tensors: HashMap<String, ggml::Tensor>,

    // Must be kept alive for the model
    _context: ggml::Context,
}

impl Model for Llama {
    type Weights = Llama;
    type HP = Hyperparameters;

    fn load(
        path: impl AsRef<Path>,
        n_ctx: usize,
        load_progress_callback: impl Fn(LoadProgress<Self::HP>),
    ) -> Result<(Self::Weights, Vocabulary), LoadError> {
        let main_path = path.as_ref();

        let mut reader =
            BufReader::new(
                File::open(main_path).map_err(|e| LoadError::OpenFileFailed {
                    source: e,
                    path: main_path.to_owned(),
                })?,
            );

        // Verify magic
        let is_legacy_model: bool = match read_u32(&mut reader)? {
            ggml::FILE_MAGIC => false,
            ggml::FILE_MAGIC_UNVERSIONED => true,
            _ => {
                return Err(LoadError::InvalidMagic {
                    path: main_path.to_owned(),
                })
            }
        };

        // Load format version
        if !is_legacy_model {
            #[allow(unused_variables)]
            let version: u32 = match read_u32(&mut reader)? {
                ggml::FORMAT_VERSION => ggml::FORMAT_VERSION,
                version => return Err(LoadError::InvalidFormatVersion { value: version }),
            };
        }
        // =================
        // Load hyper params
        // =================

        // NOTE: Field order matters! Data is laid out in the file exactly
        // in this order.
        let hparams = Hyperparameters {
            n_vocab: read_i32(&mut reader)?.try_into()?,
            n_ctx,
            n_embd: read_i32(&mut reader)?.try_into()?,
            n_mult: read_i32(&mut reader)?.try_into()?,
            n_head: read_i32(&mut reader)?.try_into()?,
            n_layer: read_i32(&mut reader)?.try_into()?,
            n_rot: read_i32(&mut reader)?.try_into()?,
            f16_: read_i32(&mut reader)?.try_into()?,
        };

        let n_ff =
            ((2 * (4 * hparams.n_embd) / 3 + hparams.n_mult - 1) / hparams.n_mult) * hparams.n_mult;

        load_progress_callback(LoadProgress::HyperparametersLoaded(hparams));

        // ===============
        // Load vocabulary
        // ===============
        let vocab = {
            let mut id_to_token = vec![];
            let mut id_to_token_score = vec![];
            let mut token_to_id = HashMap::new();
            let mut max_token_length = 0;

            for i in 0..hparams.n_vocab {
                let len = read_i32(&mut reader)?;
                if let Ok(word) = read_string(&mut reader, len as usize) {
                    max_token_length = max_token_length.max(word.len());
                    id_to_token.push(word.clone());
                    token_to_id.insert(word, TokenId::try_from(i)?);
                } else {
                    load_progress_callback(LoadProgress::BadToken { index: i });
                    id_to_token.push("�".to_string());
                }

                // Token score, currently unused
                if !is_legacy_model {
                    if let Ok(score) = read_f32(&mut reader) {
                        id_to_token_score.push(score);
                    }
                } else {
                    // Legacy model, set empty score
                    id_to_token_score.push(0.);
                }
            }

            Vocabulary {
                id_to_token,
                id_to_token_score,
                token_to_id,
                max_token_length,
            }
        };

        // for the big tensors, we have the option to store the data in 16-bit
        // floats or quantized in order to save memory and also to speed up the
        // computation
        let wtype = match hparams.f16_ {
            0 => ggml::TYPE_F32,
            1 => ggml::TYPE_F16,
            2 => ggml::TYPE_Q4_0,
            3 => ggml::TYPE_Q4_1,
            invalid => return Err(LoadError::HyperparametersF16Invalid { value: invalid }),
        };

        let n_embd = hparams.n_embd;
        let n_layer = hparams.n_layer;
        let n_vocab = hparams.n_vocab;

        let ctx_size = {
            // Use 64-bit math to prevent overflow.
            let n_embd = n_embd as u64;
            let n_layer = n_layer as u64;
            let n_vocab = n_vocab as u64;
            let n_ff = n_ff as u64;

            let mut ctx_size: u64 = 0;

            ctx_size += mulf!(n_embd, n_vocab, ggml::type_sizef(wtype)); // tok_embeddings

            ctx_size += mulf!(n_embd, ggml::type_sizef(ggml::TYPE_F32)); // norm

            ctx_size += mulf!(n_embd, n_vocab, ggml::type_sizef(wtype)); // output

            ctx_size += mulf!(n_layer, n_embd, ggml::type_sizef(ggml::TYPE_F32)); // attention_norm

            ctx_size += mulf!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wq
            ctx_size += mulf!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wk
            ctx_size += mulf!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wv
            ctx_size += mulf!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wo

            ctx_size += mulf!(n_layer, n_embd, ggml::type_sizef(ggml::TYPE_F32)); // ffn_norm

            ctx_size += mulf!(n_layer, n_ff, n_embd, ggml::type_sizef(wtype)); // w1
            ctx_size += mulf!(n_layer, n_ff, n_embd, ggml::type_sizef(wtype)); // w2
            ctx_size += mulf!(n_layer, n_ff, n_embd, ggml::type_sizef(wtype)); // w3

            ctx_size += (5 + 10 * n_layer) * 256; // object overhead

            load_progress_callback(LoadProgress::ContextSize {
                bytes: ctx_size.try_into()?,
            });

            ctx_size
        };

        // Initialize the context
        let context = ggml::Context::init(ctx_size as usize);

        let model = {
            let mut tensors = HashMap::new();

            let tok_embeddings = context.new_tensor_2d(wtype, n_embd, n_vocab);
            let norm = context.new_tensor_1d(ggml::TYPE_F32, n_embd);
            let output = context.new_tensor_2d(wtype, n_embd, n_vocab);

            tensors.insert("tok_embeddings.weight".to_owned(), tok_embeddings.share());
            tensors.insert("norm.weight".to_owned(), norm.share());
            tensors.insert("output.weight".to_owned(), output.share());

            let mut layers = Vec::new();
            for i in 0..n_layer {
                let layer = Layer {
                    attention_norm: context.new_tensor_1d(ggml::TYPE_F32, n_embd),
                    wq: context.new_tensor_2d(wtype, n_embd, n_embd),
                    wk: context.new_tensor_2d(wtype, n_embd, n_embd),
                    wv: context.new_tensor_2d(wtype, n_embd, n_embd),
                    wo: context.new_tensor_2d(wtype, n_embd, n_embd),
                    ffn_norm: context.new_tensor_1d(ggml::TYPE_F32, n_embd),
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

            Llama {
                hparams,
                tok_embeddings,
                norm,
                output,
                layers,
                tensors,
                _context: context,
            }
        };

        // Close the file, but keep its offset. That way we know how to skip the
        // metadata when loading the parts.
        let file_offset = reader.stream_position()?;
        drop(reader);

        let paths = {
            let main_filename = main_path.file_name().and_then(|p| p.to_str());

            let mut paths: Vec<PathBuf> =
                std::fs::read_dir(main_path.parent().ok_or_else(|| LoadError::NoParentPath {
                    path: main_path.to_owned(),
                })?)?
                .filter_map(Result::ok)
                .map(|de| de.path())
                .filter(|p| {
                    p.file_name()
                        .and_then(|p| p.to_str())
                        .zip(main_filename)
                        .map(|(part_filename, main_filename)| {
                            part_filename.starts_with(main_filename)
                        })
                        .unwrap_or(false)
                })
                .collect();
            paths.sort();
            paths
        };

        let n_parts = paths.len();

        for (i, part_path) in paths.into_iter().enumerate() {
            let part_id = i;

            load_progress_callback(LoadProgress::PartLoading {
                file: part_path.clone().into_boxed_path(),
                current_part: i + 1,
                total_parts: n_parts,
            });

            let mut part_reader = BufReader::new(File::open(&part_path)?);

            // Skip metadata
            part_reader.seek(SeekFrom::Start(file_offset))?;

            let mut total_size = 0;
            let mut n_tensors = 0;

            // Load weights
            loop {
                // NOTE: Implementation from #![feature(buf_read_has_data_left)]
                let is_eof = part_reader.fill_buf().map(|b| b.is_empty())?;

                if is_eof {
                    break;
                }

                let n_dims = usize::try_from(read_i32(&mut part_reader)?)?;
                let length = read_i32(&mut part_reader)?;
                let ftype = read_u32(&mut part_reader)?;

                let mut nelements: usize = 1;
                let mut ne = [1i32, 1i32];
                for i in 0..n_dims {
                    ne[i] = read_i32(&mut part_reader)?;
                    nelements *= usize::try_from(ne[i])?;
                }

                let tensor_name = read_string(&mut part_reader, length as usize)?;

                let Some(tensor) = model.tensors.get(&tensor_name)
                    else {
                        return Err(LoadError::UnknownTensor { tensor_name, path: part_path });
                    };

                // split_type = 0: split by columns
                // split_type = 1: split by rows
                //
                // split_type = 0:
                // regex:
                //   - tok_embeddings.*
                //   - layers.*.attention.wo.weight
                //   - layers.*.feed_forward.w2.weight

                // split_type = 1:
                // regex:
                //   - output.*
                //   - layers.*.attention.wq.weight
                //   - layers.*.attention.wk.weight
                //   - layers.*.attention.wv.weight
                //   - layers.*.feed_forward.w1.weight
                //   - layers.*.feed_forward.w3.weight
                #[allow(clippy::if_same_then_else)]
                let split_type = if tensor_name.contains("tok_embeddings") {
                    0
                } else if tensor_name.contains("layers") {
                    if tensor_name.contains("attention.wo.weight") {
                        0
                    } else if tensor_name.contains("feed_forward.w2.weight") {
                        0
                    } else {
                        1
                    }
                } else if tensor_name.contains("output") {
                    1
                } else {
                    0
                };

                if n_dims == 1 {
                    if tensor.nelements() != nelements {
                        return Err(LoadError::TensorWrongSize {
                            tensor_name,
                            path: part_path,
                        });
                    }
                } else if tensor.nelements() / n_parts != nelements {
                    return Err(LoadError::TensorWrongSize {
                        tensor_name,
                        path: part_path,
                    });
                }

                if n_dims == 1 {
                    if tensor.get_ne()[0] != ne[0] || tensor.get_ne()[1] != ne[1] {
                        return Err(LoadError::TensorWrongSize {
                            tensor_name,
                            path: part_path,
                        });
                    }
                } else if split_type == 0 {
                    if tensor.get_ne()[0] / i32::try_from(n_parts)? != ne[0]
                        || tensor.get_ne()[1] != ne[1]
                    {
                        return Err(LoadError::TensorWrongSize {
                            tensor_name,
                            path: part_path,
                        });
                    }
                } else if tensor.get_ne()[0] != ne[0]
                    || tensor.get_ne()[1] / i32::try_from(n_parts)? != ne[1]
                {
                    return Err(LoadError::TensorWrongSize {
                        tensor_name,
                        path: part_path,
                    });
                }

                let bpe = match ftype {
                    0 => ggml::type_size(ggml::TYPE_F32),
                    1 => ggml::type_size(ggml::TYPE_F16),
                    2 => {
                        assert_eq!(ne[0] % 64, 0);
                        ggml::type_size(ggml::TYPE_Q4_0)
                    }
                    3 => {
                        assert_eq!(ne[0] % 64, 0);
                        ggml::type_size(ggml::TYPE_Q4_1)
                    }
                    _ => {
                        return Err(LoadError::InvalidFtype {
                            ftype,
                            path: part_path,
                        })
                    }
                };

                if n_dims == 1 || n_parts == 1 {
                    if (nelements as usize * bpe) / ggml::blck_size(tensor.get_type()) as usize
                        != tensor.nbytes()
                    {
                        return Err(LoadError::TensorWrongSize {
                            tensor_name,
                            path: part_path,
                        });
                    }

                    if part_id == 0 {
                        // SAFETY: yolo, same as original code
                        let slice = unsafe {
                            let data = tensor.data();
                            std::slice::from_raw_parts_mut(data as *mut u8, tensor.nbytes())
                        };
                        part_reader.read_exact(slice)?;
                    } else {
                        part_reader.seek(SeekFrom::Current(tensor.nbytes() as i64))?;
                    }

                    total_size += tensor.nbytes();
                } else {
                    if (nelements as usize * bpe) / ggml::blck_size(tensor.get_type()) as usize
                        != tensor.nbytes() / n_parts
                    {
                        return Err(LoadError::TensorWrongSize {
                            tensor_name,
                            path: part_path,
                        });
                    }

                    if split_type == 0 {
                        let np0 = ne[0];
                        let row_size = (usize::try_from(tensor.get_ne()[0])?
                            / ggml::blck_size(tensor.get_type()))
                            * ggml::type_size(tensor.get_type());

                        assert_eq!(row_size, tensor.get_nb()[1]);

                        for i1 in 0..ne[1] {
                            let offset_row = i1 as usize * row_size;
                            let offset = offset_row
                                + ((part_id * np0 as usize)
                                    / ggml::blck_size(tensor.get_type()) as usize)
                                    * ggml::type_size(tensor.get_type());
                            // SAFETY: yolo, same as original code
                            unsafe {
                                let ptr = tensor.data().add(offset);
                                let slice = std::slice::from_raw_parts_mut(
                                    ptr as *mut u8,
                                    row_size / n_parts,
                                );
                                part_reader.read_exact(slice)?;
                            }
                        }
                    } else {
                        let np1 = ne[1];
                        let row_size = (usize::try_from(tensor.get_ne()[0])?
                            / ggml::blck_size(tensor.get_type()))
                            * ggml::type_size(tensor.get_type());

                        for i1 in 0..ne[1] {
                            let offset_row = (i1 as usize + part_id * np1 as usize) * row_size;
                            // SAFETY: yolo, same as original code
                            unsafe {
                                let ptr = tensor.data().add(offset_row);
                                let slice =
                                    std::slice::from_raw_parts_mut(ptr as *mut u8, row_size);
                                part_reader.read_exact(slice)?;
                            }
                        }
                    }

                    total_size += tensor.nbytes() / n_parts;
                }

                n_tensors += 1;
                load_progress_callback(LoadProgress::PartTensorLoaded {
                    file: part_path.clone().into_boxed_path(),
                    current_tensor: n_tensors.try_into()?,
                    tensor_count: model.tensors.len(),
                });
            }

            load_progress_callback(LoadProgress::PartLoaded {
                file: part_path.into_boxed_path(),
                byte_size: total_size,
                tensor_count: n_tensors.try_into()?,
            });
        }

        Ok((model, vocab))
    }

    fn start_session(&self, params: InferenceSessionParameters) -> InferenceSession {
        let Hyperparameters {
            n_ctx,
            n_embd,
            n_layer,
            n_vocab,
            ..
        } = self.hparams;

        let ctx_size = {
            let mut ctx_size = 0;
            ctx_size += mulf!(
                n_ctx,
                n_layer,
                n_embd,
                ggml::type_sizef(params.memory_k_type.into())
            ); // memory_k
            ctx_size += mulf!(
                n_ctx,
                n_layer,
                n_embd,
                ggml::type_sizef(params.memory_v_type.into())
            ); // memory_v
            ctx_size += (5 + 10 * n_layer as u64) * 256; // object overhead
            ctx_size
        };

        let session_ctx = ggml::Context::init(ctx_size as usize);

        // Initialize key + value memory tensors
        let n_mem = n_layer * n_ctx;
        let n_elements = n_embd * n_mem;
        let memory_k = session_ctx.new_tensor_1d(params.memory_k_type.into(), n_elements);
        let memory_v = session_ctx.new_tensor_1d(params.memory_v_type.into(), n_elements);

        InferenceSession {
            _session_ctx: session_ctx,
            params,
            memory_k,
            memory_v,
            n_past: 0,
            mem_per_token: 0,
            tokens: vec![],
            last_logits: vec![0.0; n_vocab as usize],
        }
    }

    /// Evaluates the transformer.
    ///
    /// The provided `output_request` struct lets you specify which additional
    /// data you are interested in fetching from the transformer. Setting a
    /// field to a `Some` value will clear and fill the provided vector with
    /// data. The provided vector will be resized to the exact output size.
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
        let increased_determinism = params.increased_determinism;

        let Hyperparameters {
            n_vocab,
            n_ctx,
            n_embd,
            n_mult: _,
            n_head,
            n_layer,
            n_rot,
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

        let mut gf = ggml::ComputationGraph::new(n_threads);

        let embd = ctx0.new_tensor_1d(ggml::TYPE_I32, n);
        unsafe { embd.write_data(bytemuck::cast_slice(input_tokens)) };

        let mut input_layer = ctx0.op_get_rows(&self.tok_embeddings, &embd);

        // Defined here to avoid repetition and creating a binding inside nested loops.
        // See the call site below for more context.
        let vtrans_fun = |il: usize| -> ggml::Tensor {
            ctx0.op_permute(
                &ctx0.op_reshape_3d(
                    &ctx0.op_view_1d(
                        &session.memory_v,
                        (n_past + n) * n_embd,
                        il * n_ctx as usize * session.memory_v.element_size() * n_embd as usize,
                    ),
                    n_embd / n_head,
                    n_head,
                    n_past + n,
                ),
                1,
                2,
                0,
                3,
            )
        };

        for il in 0..n_layer as usize {
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
            }

            // self-attention
            {
                let q_current = ctx0.op_mul_mat(&self.layers[il].wq, &current);
                let k_current = ctx0.op_mul_mat(&self.layers[il].wk, &current);
                let v_current = ctx0.op_mul_mat(&self.layers[il].wv, &current);

                // store key and value to memory
                if n >= 1 {
                    let k = ctx0.op_view_1d(
                        &session.memory_k,
                        n * n_embd,
                        (session.memory_k.element_size() * n_embd as usize)
                            * (il * n_ctx as usize + n_past as usize),
                    );

                    let v = ctx0.op_view_1d(
                        &session.memory_v,
                        n * n_embd,
                        (session.memory_v.element_size() * n_embd as usize)
                            * (il * n_ctx as usize + n_past as usize),
                    );

                    gf.build_forward_expand(&ctx0.op_cpy(&k_current, &k));
                    gf.build_forward_expand(&ctx0.op_cpy(&v_current, &v));
                }

                // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
                let q = ctx0.op_permute(
                    &ctx0.op_rope(
                        &ctx0.op_cpy(
                            &q_current,
                            &ctx0.new_tensor_3d(ggml::TYPE_F32, n_embd / n_head, n_head, n),
                        ),
                        n_past,
                        n_rot,
                        0,
                    ),
                    0,
                    2,
                    1,
                    3,
                );

                // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
                let k = ctx0.op_permute(
                    &ctx0.op_rope(
                        &ctx0.op_reshape_3d(
                            &ctx0.op_view_1d(
                                &session.memory_k,
                                (n_past + n) * n_embd,
                                il * n_ctx as usize
                                    * session.memory_k.element_size()
                                    * n_embd as usize,
                            ),
                            n_embd / n_head,
                            n_head,
                            n_past + n,
                        ),
                        n_past,
                        n_rot,
                        1,
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

                // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
                let v_transposed = {
                    if !increased_determinism {
                        vtrans_fun(il)
                    } else {
                        ctx0.op_cpy(
                            &vtrans_fun(il),
                            &ctx0.new_tensor_3d(
                                ggml::TYPE_F32,
                                n_past + n,
                                n_embd / n_head,
                                n_head,
                            ),
                        )
                    }
                };

                // KQV = transpose(V) * KQ_soft_max
                let k_q_v = ctx0.op_mul_mat(&v_transposed, &k_q_soft_max);

                // KQV_merged = KQV.permute(0, 2, 1, 3)
                let k_q_v_merged = ctx0.op_permute(&k_q_v, 0, 2, 1, 3);

                // cur = KQV_merged.contiguous().view(n_embd, N)
                current = ctx0.op_cpy(
                    &k_q_v_merged,
                    &ctx0.new_tensor_2d(ggml::TYPE_F32, n_embd, n),
                );

                // projection (no bias)
                current = ctx0.op_mul_mat(&self.layers[il].wo, &current);
            }

            let input_feed_forward = ctx0.op_add(&current, &input_self_attention);

            // feed-forward network
            {
                // norm
                {
                    current = ctx0.op_norm(&input_feed_forward);

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

        // Used at the end to optionally extract the embeddings.
        let embeddings_tensor;

        // norm
        {
            input_layer = ctx0.op_norm(&input_layer);

            // inpL = norm*inpL
            input_layer = ctx0.op_mul(&ctx0.op_repeat(&self.norm, &input_layer), &input_layer);
            embeddings_tensor = input_layer.share();
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
        assert_eq!(session.last_logits.len(), n_vocab as usize);
        unsafe {
            input_layer.read_data(
                n_vocab as usize * (n - 1) * std::mem::size_of::<f32>(),
                bytemuck::cast_slice_mut(&mut session.last_logits),
            )
        };

        // Extract logits
        if let Some(all_logits) = &mut output_request.all_logits {
            all_logits.resize(n_vocab as usize * n, 0.0);
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
            embeddings.resize(n_embd as usize * n, 0.0);
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
}
