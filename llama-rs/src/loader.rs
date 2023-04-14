use std::{
    collections::HashMap,
    io::{BufRead, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use crate::{
    util::{self, mulf},
    Mmap, Model, TokenId, Vocabulary,
};
use crate::{ElementType, Hyperparameters};
use ggml_loader::util::*;
use ggml_loader::ContainerType;
use thiserror::Error;

/// Each variant represents a step within the process of loading the model.
/// These can be used to report progress to the user.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum LoadProgress<'a> {
    /// The hyperparameters have been loaded from the model.
    HyperparametersLoaded(&'a Hyperparameters),
    /// The context has been created.
    ContextSize {
        /// The size of the context.
        bytes: usize,
    },
    /// A part of the model is being loaded.
    PartLoading {
        /// The path to the model part.
        file: &'a Path,
        /// The current part (0-indexed).
        current_part: usize,
        /// The number of total parts.
        total_parts: usize,
    },
    /// A tensor from the current part has been loaded.
    PartTensorLoaded {
        /// The path to the model part.
        file: &'a Path,
        /// The current tensor (0-indexed).
        current_tensor: usize,
        /// The number of total tensors.
        tensor_count: usize,
    },
    /// A model part has finished fully loading.
    PartLoaded {
        /// The path to the model part.
        file: &'a Path,
        /// The number of bytes in the part.
        byte_size: usize,
        /// The number of tensors in the part.
        tensor_count: usize,
    },
}

#[derive(Error, Debug)]
/// Errors encountered during the loading process.
pub enum LoadError {
    #[error("could not open file {path:?}")]
    /// A file failed to open.
    OpenFileFailed {
        /// The original error.
        source: std::io::Error,
        /// The path that failed.
        path: PathBuf,
    },
    #[error("no parent path for {path:?}")]
    /// There is no parent path for a given path.
    NoParentPath {
        /// The path without a parent.
        path: PathBuf,
    },
    #[error("unable to read exactly {bytes} bytes")]
    /// Reading exactly `bytes` from a file failed.
    ReadExactFailed {
        /// The original error.
        source: std::io::Error,
        /// The number of bytes that were attempted to be read.
        bytes: usize,
    },
    #[error("non-specific I/O error")]
    /// A non-specific IO error.
    IO(#[from] std::io::Error),
    #[error("could not convert bytes to a UTF-8 string")]
    /// One of the strings encountered was not valid UTF-8.
    InvalidUtf8(#[from] std::string::FromUtf8Error),
    #[error("invalid integer conversion")]
    /// One of the integers encountered could not be converted to a more appropriate type.
    InvalidIntegerConversion(#[from] std::num::TryFromIntError),
    #[error("unsupported f16_: {0}")]
    /// One of the integers encountered could not be converted to a more appropriate type.
    UnsupportedElementType(i32),
    #[error("invalid magic number for {path:?}")]
    /// An invalid magic number was encountered during the loading process.
    InvalidMagic {
        /// The path that failed.
        path: PathBuf,
    },
    #[error("invalid file format version {value}")]
    /// The version of the format is not supported by this version of `llama-rs`.
    InvalidFormatVersion {
        /// The version that was encountered.
        value: u32,
    },
    #[error("invalid value {ftype} for `f16` in hyperparameters")]
    /// The `f16` hyperparameter had an invalid value.
    HyperparametersF16Invalid {
        /// The format type that was encountered.
        ftype: u32,
    },
    #[error("unknown tensor `{tensor_name}` in {path:?}")]
    /// The tensor `tensor_name` was encountered during the loading of `path`, but was not seen during
    /// the model prelude.
    UnknownTensor {
        /// The name of the tensor.
        tensor_name: String,
        /// The path that failed.
        path: PathBuf,
    },
    #[error("the tensor `{tensor_name}` has the wrong size in {path:?}")]
    /// The tensor `tensor_name` did not match its expected size.
    TensorWrongSize {
        /// The name of the tensor.
        tensor_name: String,
        /// The path that failed.
        path: PathBuf,
    },
    /// The tensor `tensor_name` did not have the expected format type.
    #[error("invalid ftype {ftype} for tensor `{tensor_name}` in {path:?}")]
    InvalidFtype {
        /// The name of the tensor.
        tensor_name: String,
        /// The format type that was encountered.
        ftype: i32,
        /// The path that failed.
        path: PathBuf,
    },
}

pub(crate) fn load(
    path: impl AsRef<Path>,
    n_context_tokens: usize,
    mut load_progress_callback: impl FnMut(LoadProgress),
) -> Result<Model, LoadError> {
    use std::fs::File;
    use std::io::BufReader;

    let main_path = path.as_ref();

    let file = File::open(main_path).map_err(|e| LoadError::OpenFileFailed {
        source: e,
        path: main_path.to_owned(),
    })?;
    let mut reader = BufReader::new(&file);

    // Verify magic
    let model_type: ContainerType = match read_u32(&mut reader)? {
        ggml::FILE_MAGIC_GGMF => ContainerType::GGMF,
        ggml::FILE_MAGIC_GGJT => ContainerType::GGJT,
        ggml::FILE_MAGIC_UNVERSIONED => ContainerType::GGML,
        _ => {
            return Err(LoadError::InvalidMagic {
                path: main_path.to_owned(),
            })
        }
    };

    // Load format version
    match model_type {
        ContainerType::GGMF | ContainerType::GGJT => {
            let _version: u32 = match read_u32(&mut reader)? {
                ggml::FORMAT_VERSION => ggml::FORMAT_VERSION,
                version => return Err(LoadError::InvalidFormatVersion { value: version }),
            };
        }
        ContainerType::GGML => {}
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
        element_type: {
            let ftype = read_i32(&mut reader)?;
            decode_element_type(ftype).ok_or_else(|| LoadError::UnsupportedElementType(ftype))
        }?,
    };

    let n_ff =
        ((2 * (4 * hparams.n_embd) / 3 + hparams.n_mult - 1) / hparams.n_mult) * hparams.n_mult;

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
            let token = read_bytes_with_len(&mut reader, len.try_into()?)?;
            max_token_length = max_token_length.max(token.len());
            id_to_token.push(token.clone());
            token_to_id.insert(token, TokenId::try_from(i)?);

            // Token score, currently unused
            match model_type {
                ContainerType::GGMF | ContainerType::GGJT => {
                    let score = read_f32(&mut reader)?;
                    id_to_token_score.push(score);
                }
                ContainerType::GGML => {
                    // Legacy model, set empty score
                    id_to_token_score.push(0.);
                }
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
    let wtype = hparams.element_type;

    let n_embd = hparams.n_embd;
    let n_layer = hparams.n_layer;
    let n_vocab = hparams.n_vocab;

    let ctx_size = {
        // Use 64-bit math to prevent overflow.
        let mut ctx_size: usize = 0;

        ctx_size += mulf!(n_embd, n_vocab, ggml::type_sizef(wtype)); // tok_embeddings

        ctx_size += mulf!(n_embd, ggml::type_sizef(ggml::Type::F32)); // norm

        ctx_size += mulf!(n_embd, n_vocab, ggml::type_sizef(wtype)); // output

        ctx_size += mulf!(n_layer, n_embd, ggml::type_sizef(ggml::Type::F32)); // attention_norm

        ctx_size += mulf!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wq
        ctx_size += mulf!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wk
        ctx_size += mulf!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wv
        ctx_size += mulf!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wo

        ctx_size += mulf!(n_layer, n_embd, ggml::type_sizef(ggml::Type::F32)); // ffn_norm

        ctx_size += mulf!(n_layer, n_ff, n_embd, ggml::type_sizef(wtype)); // w1
        ctx_size += mulf!(n_layer, n_ff, n_embd, ggml::type_sizef(wtype)); // w2
        ctx_size += mulf!(n_layer, n_ff, n_embd, ggml::type_sizef(wtype)); // w3

        ctx_size += (5 + 10 * n_layer) * 256; // object overhead

        load_progress_callback(LoadProgress::ContextSize { bytes: ctx_size });

        ctx_size
    };

    // Initialize the context
    let context = ggml::Context::init(ctx_size);

    let mut model = Model::new(context, hparams, vocabulary, n_ff, wtype, model_type);
    match model_type {
        ContainerType::GGMF | ContainerType::GGML => {
            let file_offset = reader.stream_position()?;
            drop(reader);
            load_weights_ggmf_or_unversioned(
                file_offset,
                main_path,
                load_progress_callback,
                model.tensors_mut(),
            )?
        }
        ContainerType::GGJT => {
            let mmap = unsafe { Mmap::map(&file)? };
            let ptr = mmap.as_ptr();
            model.mmap = Some(mmap);
            load_weights_ggjt(
                &mut reader,
                ptr,
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
    mut load_progress_callback: impl FnMut(LoadProgress),
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

            let (nelements, ne, tensor_name, tensor, split_type, bpe) = load_tensor_header_ggmf(
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

#[allow(clippy::type_complexity)]
fn load_tensor_header_ggmf<'a>(
    n_dims: usize,
    reader: &mut impl BufRead,
    length: i32,
    tensors: &'a mut HashMap<String, ggml::Tensor>,
    path: &Path,
    n_parts: usize,
    ftype: i32,
) -> Result<(usize, [i64; 2], String, &'a mut ggml::Tensor, i32, usize), LoadError> {
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
            return Err(LoadError::InvalidFtype {
                tensor_name,
                ftype,
                path: path.to_owned(),
            });
        }
    };
    Ok((nelements, ne, tensor_name, tensor, split_type, bpe))
}

fn tensor_type_size(ftype: i32, ne: [i64; 2]) -> Option<usize> {
    let ftype = decode_element_type(ftype)?;
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
    mmap_base: *const u8,
    path: &Path,
    mut load_progress_callback: impl FnMut(LoadProgress),
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
                return Err(LoadError::InvalidFtype {
                    tensor_name,
                    ftype,
                    path: path.to_owned(),
                });
            }
        };

        load_tensor_ggjt(reader, mmap_base, tensor)?;

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

#[cfg(feature = "mmap")]
fn load_tensor_ggjt(
    reader: &mut (impl BufRead + Seek),
    mmap_base: *const u8,
    tensor: &ggml::Tensor,
) -> Result<(), LoadError> {
    let offset_curr = reader.stream_position()?;
    let offset_aligned: u64 = (offset_curr + 31) & !31;
    unsafe {
        let ptr = mmap_base.offset(offset_aligned as isize);
        tensor.set_data(ptr as *mut std::ffi::c_void);
    }
    reader.seek(SeekFrom::Start(offset_aligned + tensor.nbytes() as u8))?;
    Ok(())
}

#[cfg(not(feature = "mmap"))]
fn load_tensor_ggjt<'a>(
    reader: &mut (impl BufRead + Seek),
    mmap_base: *const u8,
    tensor: &'a mut ggml::Tensor,
) -> Result<(), LoadError> {
    _ = mmap_base;
    let offset_curr = reader.stream_position()?;
    let offset_aligned: u64 = (offset_curr + 31) & !31;
    reader.seek(SeekFrom::Start(offset_aligned))?;

    let buf: &'a mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(tensor.data() as *mut u8, tensor.nbytes()) };
    reader.read_exact(buf)?;

    Ok(())
}
