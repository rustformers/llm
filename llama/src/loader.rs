#![allow(dead_code)]

use std::{
    collections::HashMap,
    io::{BufRead, Read, Seek, SeekFrom},
    path::Path,
};

use crate::{ElementType, Hyperparameters};
use crate::{Llama, LoadError, LoadProgress, TokenId, Vocabulary};
use ggml_format::{
    util::{has_data_left, read_bytes_with_len, read_f32, read_i32, read_u32},
    ContainerType,
};
use llm_base::{mulf, util, FileType};
use memmap2::Mmap;

pub(crate) fn load(
    path: impl AsRef<Path>,
    prefer_mmap: bool,
    n_context_tokens: usize,
    mut load_progress_callback: impl FnMut(LoadProgress<Hyperparameters>),
) -> Result<Llama, LoadError> {
    use std::fs::File;
    use std::io::BufReader;

    let main_path = path.as_ref();

    let file = File::open(main_path).map_err(|e| LoadError::OpenFileFailed {
        source: e,
        path: main_path.to_owned(),
    })?;
    let mut reader = BufReader::new(&file);

    // Verify magic
    let magic = read_u32(&mut reader)?;
    let model_type: ContainerType = match magic {
        ggml::FILE_MAGIC_GGMF => ContainerType::Ggmf,
        ggml::FILE_MAGIC_GGJT => ContainerType::Ggjt,
        ggml::FILE_MAGIC_UNVERSIONED => ContainerType::Ggml,
        _ => {
            return Err(LoadError::InvalidMagic {
                path: main_path.to_owned(),
                magic,
            })
        }
    };

    // Load format version
    match model_type {
        ContainerType::Ggmf | ContainerType::Ggjt => {
            let _version: u32 = match read_u32(&mut reader)? {
                ggml::FORMAT_VERSION => ggml::FORMAT_VERSION,
                version => {
                    return Err(LoadError::InvalidFormatVersion {
                        container_type: model_type,
                        version,
                    })
                }
            };
        }
        ContainerType::Ggml => {}
    }

    // =================
    // Load hyper params
    // =================

    // NOTE: Field order matters! Data is laid out in the file exactly
    // in this order.
    let hparams = Hyperparameters {
        n_vocab: read_i32(&mut reader)?.try_into()?,
        n_ctx: n_context_tokens,
        n_embd: read_i32(&mut reader)?.try_into()?,
        n_mult: read_i32(&mut reader)?.try_into()?,
        n_head: read_i32(&mut reader)?.try_into()?,
        n_layer: read_i32(&mut reader)?.try_into()?,
        n_rot: read_i32(&mut reader)?.try_into()?,
        file_type: {
            let ftype = read_i32(&mut reader)?;
            FileType::try_from(ftype).map_err(|_| LoadError::UnsupportedFileType(ftype))
        }?,
    };

    let n_ff =
        ((2 * (4 * hparams.n_embd) / 3 + hparams.n_mult - 1) / hparams.n_mult) * hparams.n_mult;

    load_progress_callback(LoadProgress::HyperparametersLoaded(&hparams));

    // ===============
    // Load vocabulary
    // ===============
    let vocabulary = {
        let mut vocab = Vocabulary::default();

        for i in 0..hparams.n_vocab {
            let len = read_i32(&mut reader)?;
            let id = i as TokenId;
            let token = read_bytes_with_len(&mut reader, len.try_into()?)?;

            let score = match model_type {
                ContainerType::Ggmf | ContainerType::Ggjt => read_f32(&mut reader)?,
                ContainerType::Ggml => {
                    // Legacy model, set empty score
                    0.
                }
            };

            vocab.push_token(id, token, score);
        }

        vocab
    };

    // for the big tensors, we have the option to store the data in 16-bit
    // floats or quantized in order to save memory and also to speed up the
    // computation
    let wtype = match hparams.file_type {
        FileType::F32 => ggml::Type::F32,
        FileType::MostlyF16 => ggml::Type::F16,
        FileType::MostlyQ4_0 => ggml::Type::Q4_0,
        FileType::MostlyQ4_1 => ggml::Type::Q4_1,
        _ => unimplemented!(),
    };

    let n_embd = hparams.n_embd;
    let n_layer = hparams.n_layer;
    let n_vocab = hparams.n_vocab;

    let alloc = !(prefer_mmap && model_type.support_mmap());

    let ctx_size = {
        // Use 64-bit math to prevent overflow.
        let mut ctx_size: usize = (5 + 10 * n_layer) * 256; // object overhead

        if alloc {
            let mut model_size: usize = 0;

            ctx_size += mulf!(n_embd, n_vocab, ggml::type_sizef(wtype)); // tok_embeddings
            ctx_size += mulf!(n_embd, ggml::type_sizef(ggml::Type::F32)); // norm
            ctx_size += mulf!(n_embd, n_vocab, ggml::type_sizef(wtype)); // output

            model_size += mulf!(n_layer, n_embd, ggml::type_sizef(ggml::Type::F32)); // attention_norm

            model_size += mulf!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wq
            model_size += mulf!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wk
            model_size += mulf!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wv
            model_size += mulf!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wo

            model_size += mulf!(n_layer, n_embd, ggml::type_sizef(ggml::Type::F32)); // ffn_norm

            model_size += mulf!(n_layer, n_ff, n_embd, ggml::type_sizef(wtype)); // w1
            model_size += mulf!(n_layer, n_ff, n_embd, ggml::type_sizef(wtype)); // w2
            model_size += mulf!(n_layer, n_ff, n_embd, ggml::type_sizef(wtype)); // w3

            ctx_size += model_size;
        }

        load_progress_callback(LoadProgress::ContextSize { bytes: ctx_size });

        ctx_size
    };

    // Initialize the context
    let context = ggml::Context::init(ctx_size, alloc);

    let (mmap, mmap_ptr) = if prefer_mmap && model_type.support_mmap() {
        let mmap = unsafe { Mmap::map(&file)? };
        let ptr = mmap.as_ptr();
        (Some(mmap), Some(ptr))
    } else {
        (None, None)
    };

    let mut model = Llama::new_loader1(context, hparams, vocabulary, n_ff, wtype, mmap);
    match model_type {
        ContainerType::Ggmf | ContainerType::Ggml => {
            let file_offset = reader.stream_position()?;
            drop(reader);
            load_weights_ggmf_or_unversioned(
                file_offset,
                main_path,
                load_progress_callback,
                model.tensors_mut(),
            )?
        }
        ContainerType::Ggjt => {
            load_weights_ggjt(
                &mut reader,
                mmap_ptr,
                main_path,
                load_progress_callback,
                model.tensors_mut(),
            )?;
        }
    }

    Ok(model)
}

/// Helper function. Reads a string from the buffer and returns it.
pub(crate) fn read_string(reader: &mut impl BufRead, len: usize) -> Result<String, LoadError> {
    let mut buf = vec![0; len];
    reader
        .read_exact(&mut buf)
        .map_err(|e| LoadError::ReadExactFailed {
            source: e,
            bytes: buf.len(),
        })?;
    let s = String::from_utf8(buf)?;
    Ok(s)
}

fn load_weights_ggmf_or_unversioned(
    file_offset: u64,
    main_path: &Path,
    mut load_progress_callback: impl FnMut(LoadProgress<Hyperparameters>),
    tensors: &mut HashMap<String, ggml::Tensor>,
) -> Result<(), LoadError> {
    use std::{fs::File, io::BufReader};

    let paths = util::find_all_model_files(main_path)?;

    let n_parts = paths.len();
    for (i, part_path) in paths.into_iter().enumerate() {
        let part_id = i;

        load_progress_callback(LoadProgress::PartLoading {
            file: &part_path,
            current_part: i,
            total_parts: n_parts,
        });

        let mut part_reader = BufReader::new(File::open(&part_path)?);

        // Skip metadata
        part_reader.seek(SeekFrom::Start(file_offset))?;

        let mut total_size = 0;
        let mut n_tensors = 0;

        // Load weights
        loop {
            if !has_data_left(&mut part_reader)? {
                break;
            }

            let n_dims = usize::try_from(read_i32(&mut part_reader)?)?;
            let length = read_i32(&mut part_reader)?;
            let ftype = read_i32(&mut part_reader)?;

            let TensorHeaderGgmf {
                nelements,
                ne,
                tensor_name,
                tensor,
                split_type,
                bpe,
            } = load_tensor_header_ggmf(
                n_dims,
                &mut part_reader,
                length,
                tensors,
                &part_path,
                n_parts,
                ftype,
            )?;

            if n_dims == 1 || n_parts == 1 {
                if (nelements * bpe) / ggml::blck_size(tensor.get_type()) != tensor.nbytes() {
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
                if (nelements * bpe) / ggml::blck_size(tensor.get_type())
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
                            + ((part_id * np0 as usize) / ggml::blck_size(tensor.get_type()))
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
                tensor_count: tensors.len(),
            });
        }

        load_progress_callback(LoadProgress::PartLoaded {
            file: &part_path,
            byte_size: total_size,
            tensor_count: n_tensors.try_into()?,
        });
    }
    Ok(())
}

struct TensorHeaderGgmf<'a> {
    nelements: usize,
    ne: [i64; 2],
    tensor_name: String,
    tensor: &'a mut ggml::Tensor,
    split_type: i32,
    bpe: usize,
}
fn load_tensor_header_ggmf<'a>(
    n_dims: usize,
    reader: &mut impl BufRead,
    length: i32,
    tensors: &'a mut HashMap<String, ggml::Tensor>,
    path: &Path,
    n_parts: usize,
    ftype: i32,
) -> Result<TensorHeaderGgmf<'a>, LoadError> {
    let mut nelements = 1;
    let mut ne = [1i64, 1i64];
    assert!(n_dims <= ne.len());
    #[allow(clippy::needless_range_loop)]
    for i in 0..n_dims {
        ne[i] = read_i32(reader)? as i64;
        nelements *= usize::try_from(ne[i])?;
    }
    let tensor_name = read_string(reader, length as usize)?;
    let Some(tensor) = tensors.get_mut(&tensor_name)
        else {
            return Err(LoadError::UnknownTensor { tensor_name, path: path.to_owned() });
        };
    let split_type = if tensor_name.contains("tok_embeddings") {
        0
    } else if tensor_name.contains("layers") {
        if tensor_name.contains("attention.wo.weight")
            || tensor_name.contains("feed_forward.w2.weight")
        {
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
                path: path.to_owned(),
            });
        }
    } else if tensor.nelements() / n_parts != nelements {
        return Err(LoadError::TensorWrongSize {
            tensor_name,
            path: path.to_owned(),
        });
    }
    if n_dims == 1 {
        if tensor.get_ne()[0] != ne[0] || tensor.get_ne()[1] != ne[1] {
            return Err(LoadError::TensorWrongSize {
                tensor_name,
                path: path.to_owned(),
            });
        }
    } else if split_type == 0 {
        if tensor.get_ne()[0] / i64::try_from(n_parts)? != ne[0] || tensor.get_ne()[1] != ne[1] {
            return Err(LoadError::TensorWrongSize {
                tensor_name,
                path: path.to_owned(),
            });
        }
    } else if tensor.get_ne()[0] != ne[0] || tensor.get_ne()[1] / i64::try_from(n_parts)? != ne[1] {
        return Err(LoadError::TensorWrongSize {
            tensor_name,
            path: path.to_owned(),
        });
    }
    let bpe = tensor_type_size(ftype, ne);
    let bpe = match bpe {
        Some(x) => x,
        None => {
            return Err(LoadError::UnsupportedElementType {
                tensor_name,
                ftype,
                path: path.to_owned(),
            });
        }
    };
    Ok(TensorHeaderGgmf {
        nelements,
        ne,
        tensor_name,
        tensor,
        split_type,
        bpe,
    })
}

fn tensor_type_size(ftype: i32, ne: [i64; 2]) -> Option<usize> {
    let ftype = ggml::Type::try_from(ftype).ok()?;
    match ftype {
        ElementType::Q4_0 | ElementType::Q4_1 => {
            assert_eq!(ne[0] % 64, 0);
        }
        _ => {}
    }
    Some(ggml::type_size(ftype))
}

fn load_weights_ggjt(
    reader: &mut (impl BufRead + Seek),
    mmap_base: Option<*const u8>,
    path: &Path,
    mut load_progress_callback: impl FnMut(LoadProgress<Hyperparameters>),
    tensors: &mut HashMap<String, ggml::Tensor>,
) -> Result<(), LoadError>
// where R: std::io::Read
{
    let mut loop_i = 0;
    let mut total_loaded_bytes = 0;
    load_progress_callback(LoadProgress::PartLoading {
        file: path,
        current_part: 0,
        total_parts: 1,
    });

    loop {
        if !has_data_left(reader)? {
            break;
        }

        let n_dims = read_i32(reader)? as usize;
        let length = read_i32(reader)?;
        let ftype = read_i32(reader)?;

        let mut nelements: usize = 1;
        let mut ne = [1i64, 1];
        assert!(n_dims <= ne.len());
        #[allow(clippy::needless_range_loop)]
        for i in 0..n_dims {
            let dim = read_i32(reader)? as usize;
            ne[i] = dim as i64;
            nelements *= dim;
        }
        let tensor_name = read_string(reader, length as usize)?;
        let Some(tensor) = tensors.get_mut(&tensor_name)
        else {
            return Err(LoadError::UnknownTensor { tensor_name, path: path.to_owned() });
        };

        if tensor.nelements() != nelements {
            return Err(LoadError::TensorWrongSize {
                tensor_name,
                path: path.to_owned(),
            });
        }
        let tensor_ne = tensor.get_ne();
        if tensor_ne[0] != ne[0] || tensor_ne[1] != ne[1] {
            return Err(LoadError::TensorWrongSize {
                tensor_name,
                path: path.to_owned(),
            });
        }

        match tensor_type_size(ftype, ne) {
            Some(_) => {}
            None => {
                return Err(LoadError::UnsupportedElementType {
                    tensor_name,
                    ftype,
                    path: path.to_owned(),
                });
            }
        };

        if let Some(mmap_base) = mmap_base {
            load_tensor_ggjt_mmap(reader, mmap_base, tensor)?;
        } else {
            load_tensor_ggjt_copy(reader, tensor)?;
        }

        total_loaded_bytes += tensor.nbytes() as u64;

        load_progress_callback(LoadProgress::PartTensorLoaded {
            file: path,
            current_tensor: loop_i,
            tensor_count: tensors.len(),
        });

        loop_i += 1;
    }

    load_progress_callback(LoadProgress::PartLoaded {
        file: path,
        byte_size: total_loaded_bytes as usize,
        tensor_count: loop_i,
    });

    Ok(())
}

fn load_tensor_ggjt_mmap(
    reader: &mut (impl BufRead + Seek),
    mmap_base: *const u8,
    tensor: &mut ggml::Tensor,
) -> Result<(), LoadError> {
    let offset_curr = reader.stream_position()?;
    let offset_aligned: u64 = (offset_curr + 31) & !31;
    unsafe {
        let ptr = mmap_base.offset(offset_aligned as isize);
        tensor.set_data(ptr as *mut std::ffi::c_void);
    }
    reader.seek(SeekFrom::Start(offset_aligned + tensor.nbytes() as u64))?;
    Ok(())
}

fn load_tensor_ggjt_copy<'a>(
    reader: &mut (impl BufRead + Seek),
    tensor: &'a mut ggml::Tensor,
) -> Result<(), LoadError> {
    let offset_curr = reader.stream_position()?;
    let offset_aligned: u64 = (offset_curr + 31) & !31;
    reader.seek(SeekFrom::Start(offset_aligned))?;

    let buf: &'a mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(tensor.data() as *mut u8, tensor.nbytes()) };
    reader.read_exact(buf)?;

    Ok(())
}
