use std::{
    collections::HashMap,
    io::{BufRead, Read, Seek, SeekFrom},
    path::Path,
};

use ggml::loader::{
    find_all_model_files, read_bytes_with_len, read_f32, read_i32, read_string, read_u32,
    LoadError, LoadProgress,
};
use llama_rs::{mulf, TokenId, Vocabulary};

use crate::{Bloom, Hyperparameters};

/// Load a model from disk
pub fn load(
    path: impl AsRef<Path>,
    n_ctx: usize,
    mut load_progress_callback: impl FnMut(LoadProgress<Hyperparameters>),
) -> Result<Bloom, LoadError> {
    use std::fs::File;
    use std::io::BufReader;

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
        f16_: read_i32(&mut reader)?.try_into()?,
    };

    let n_ff = ((4 * hparams.n_embd + hparams.n_mult - 1) / hparams.n_mult) * hparams.n_mult;

    load_progress_callback(LoadProgress::HyperparametersLoaded(&hparams));

    // ===============
    // Load vocabulary
    // ===============
    let vocabulary = {
        let mut id_to_token = vec![];
        let mut id_to_token_score = vec![];
        let mut token_to_id = HashMap::new();
        let mut max_token_length = 0;

        for i in 0..hparams.n_vocab {
            let len = read_i32(&mut reader)?;
            let token = read_bytes_with_len(&mut reader, len as usize)?;
            max_token_length = max_token_length.max(token.len());
            id_to_token.push(token.clone());
            token_to_id.insert(token, TokenId::try_from(i)?);

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
        0 => ggml::Type::F32,
        1 => ggml::Type::F16,
        2 => ggml::Type::Q4_0,
        3 => ggml::Type::Q4_1,
        invalid => return Err(LoadError::HyperparametersF16Invalid { ftype: invalid }),
    };

    let n_embd = hparams.n_embd;
    let n_layer = hparams.n_layer;
    let n_vocab = hparams.n_vocab;

    let ctx_size = {
        let mut ctx_size: usize = 0;

        ctx_size += mulf!(n_embd, n_vocab, ggml::type_sizef(wtype)); // tok_embeddings

        ctx_size += mulf!(n_embd, ggml::type_sizef(ggml::Type::F32)); // norm
        ctx_size += mulf!(n_embd, ggml::type_sizef(ggml::Type::F32)); // norm_b

        ctx_size += mulf!(n_embd, ggml::type_sizef(ggml::Type::F32)); // output_norm
        ctx_size += mulf!(n_embd, ggml::type_sizef(ggml::Type::F32)); // output_norm_b

        ctx_size += mulf!(n_embd, n_vocab, ggml::type_sizef(wtype)); // output

        ctx_size += mulf!(n_layer, n_embd, ggml::type_sizef(ggml::Type::F32)); // attention_norm
        ctx_size += mulf!(n_layer, n_embd, ggml::type_sizef(ggml::Type::F32)); // attention_norm_b

        ctx_size += mulf!(n_layer, 3, n_embd, n_embd, ggml::type_sizef(wtype)); //query_key_value
        ctx_size += mulf!(n_layer, 3, n_embd, ggml::type_sizef(ggml::Type::F32)); //query_key_value_b

        ctx_size += mulf!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wo
        ctx_size += mulf!(n_layer, n_embd, ggml::type_sizef(ggml::Type::F32)); // wo_b

        ctx_size += mulf!(n_layer, n_embd, ggml::type_sizef(ggml::Type::F32)); // ffn_norm
        ctx_size += mulf!(n_layer, n_embd, ggml::type_sizef(ggml::Type::F32)); // ffn_norm_b

        ctx_size += mulf!(n_layer, n_ff, n_embd, ggml::type_sizef(wtype)); // w1
        ctx_size += mulf!(n_layer, n_ff, ggml::type_sizef(ggml::Type::F32)); // w1_b

        ctx_size += mulf!(n_layer, n_ff, n_embd, ggml::type_sizef(wtype)); // w2
        ctx_size += mulf!(n_layer, n_ff, ggml::type_sizef(ggml::Type::F32)); // w2_b

        ctx_size += (5 + 10 * n_layer) * 256; // object overhead

        load_progress_callback(LoadProgress::ContextSize { bytes: ctx_size });

        ctx_size
    };

    // Initialize the context
    let context = ggml::Context::init(ctx_size as usize);

    let model = Bloom::new(context, hparams, vocabulary, n_ff, wtype);

    // Close the file, but keep its offset. That way we know how to skip the
    // metadata when loading the parts.
    let file_offset = reader.stream_position()?;
    drop(reader);

    let paths = find_all_model_files(main_path)?;

    let n_parts = paths.len();

    for (i, part_path) in paths.into_iter().enumerate() {
        let part_id = i;

        load_progress_callback(LoadProgress::PartLoading {
            file: &part_path,
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
            let mut ne = [1i64, 1i64];

            #[allow(clippy::needless_range_loop)]
            for i in 0..n_dims {
                ne[i] = read_i32(&mut part_reader)? as i64;
                nelements *= usize::try_from(ne[i])?;
            }

            let tensor_name = read_string(&mut part_reader, length as usize)?;

            let Some(tensor) = model.tensors.get(&tensor_name)
              else {
                  return Err(LoadError::UnknownTensor { tensor_name, path: part_path.to_path_buf() });
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
                        path: part_path.to_path_buf(),
                    });
                }
            } else if tensor.nelements() / n_parts != nelements {
                return Err(LoadError::TensorWrongSize {
                    tensor_name,
                    path: part_path.to_path_buf(),
                });
            }

            if n_dims == 1 {
                if tensor.get_ne()[0] != ne[0] || tensor.get_ne()[1] != ne[1] {
                    return Err(LoadError::TensorWrongSize {
                        tensor_name,
                        path: part_path.to_path_buf(),
                    });
                }
            } else if split_type == 0 {
                if tensor.get_ne()[0] / i64::try_from(n_parts)? != ne[0]
                    || tensor.get_ne()[1] != ne[1]
                {
                    return Err(LoadError::TensorWrongSize {
                        tensor_name,
                        path: part_path.to_path_buf(),
                    });
                }
            } else if tensor.get_ne()[0] != ne[0]
                || tensor.get_ne()[1] / i64::try_from(n_parts)? != ne[1]
            {
                return Err(LoadError::TensorWrongSize {
                    tensor_name,
                    path: part_path.to_path_buf(),
                });
            }

            let bpe = match ftype {
                0 => ggml::type_size(ggml::Type::F32),
                1 => ggml::type_size(ggml::Type::F16),
                2 => {
                    assert_eq!(ne[0] % 64, 0);
                    ggml::type_size(ggml::Type::Q4_0)
                }
                3 => {
                    assert_eq!(ne[0] % 64, 0);
                    ggml::type_size(ggml::Type::Q4_1)
                }
                _ => {
                    return Err(LoadError::InvalidFtype {
                        ftype,
                        path: part_path.to_path_buf(),
                        tensor_name,
                    })
                }
            };

            if n_dims == 1 || n_parts == 1 {
                if (nelements as usize * bpe) / ggml::blck_size(tensor.get_type()) as usize
                    != tensor.nbytes()
                {
                    return Err(LoadError::TensorWrongSize {
                        tensor_name,
                        path: part_path.to_path_buf(),
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
                        path: part_path.to_path_buf(),
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
                            let slice =
                                std::slice::from_raw_parts_mut(ptr as *mut u8, row_size / n_parts);
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
                            let slice = std::slice::from_raw_parts_mut(ptr as *mut u8, row_size);
                            part_reader.read_exact(slice)?;
                        }
                    }
                }

                total_size += tensor.nbytes() / n_parts;
            }

            n_tensors += 1;
            load_progress_callback(LoadProgress::PartTensorLoaded {
                file: &part_path,
                current_tensor: n_tensors.try_into()?,
                tensor_count: model.tensors.len(),
            });
        }

        load_progress_callback(LoadProgress::PartLoaded {
            file: &part_path,
            byte_size: total_size,
            tensor_count: n_tensors.try_into()?,
        });
    }

    Ok(model)
}
