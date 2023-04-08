use std::{
    io::{BufRead, Read, Seek, SeekFrom},
    path::Path,
};

use crate::ElementType;
use crate::{util, LoadError, LoadProgress, Model};
use llama_loader::decode_element_type;
use llama_loader::util::*;

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

pub(crate) fn load_weights_ggmf_or_unversioned(
    file_offset: u64,
    main_path: &Path,
    load_progress_callback: impl Fn(LoadProgress),
    model: &Model,
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
                model,
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
                tensor_count: model.tensors.len(),
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
    model: &'a Model,
    path: &Path,
    n_parts: usize,
    ftype: i32,
) -> Result<(usize, [i64; 2], String, &'a ggml::Tensor, i32, usize), LoadError> {
    let mut nelements = 1;
    let mut ne = [1i64, 1i64];
    assert!(n_dims <= ne.len());
    #[allow(clippy::needless_range_loop)]
    for i in 0..n_dims {
        ne[i] = read_i32(reader)? as i64;
        nelements *= usize::try_from(ne[i])?;
    }
    let tensor_name = read_string(reader, length as usize)?;
    let Some(tensor) = model.tensors.get(&tensor_name)
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

pub(crate) fn load_weights_ggjt(
    reader: &mut (impl BufRead + Seek),
    mmap_base: *const u8,
    path: &Path,
    load_progress_callback: impl Fn(LoadProgress),
    model: &Model,
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
        let Some(tensor) = model.tensors.get(&tensor_name)
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
            tensor_count: model.tensors.len(),
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
    tensor: &'a ggml::Tensor,
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
