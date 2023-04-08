#![allow(missing_docs)]

//! standalone model loader 

use std::{
    io::{BufRead, Seek, SeekFrom},
    ops::ControlFlow,
};

use crate::{loader::has_data_left, ElementType, ModelContainerType};

pub(crate) fn decode_element_type(ftype: i32) -> Option<ElementType> {
    match ftype {
        0 => Some(ggml::Type::F32),
        1 => Some(ggml::Type::F16),
        2 => Some(ggml::Type::Q4_0),
        3 => Some(ggml::Type::Q4_1),
        _ => None,
    }
}

pub(crate) fn encode_element_type(element_type: ElementType) -> Option<i32> {
    match element_type {
        ggml::Type::F32 => Some(0),
        ggml::Type::F16 => Some(1),
        ggml::Type::Q4_0 => Some(2),
        ggml::Type::Q4_1 => Some(3),
        _ => None,
    }
}

pub(crate) fn read_bytes<const N: usize>(
    reader: &mut impl BufRead,
) -> Result<[u8; N], std::io::Error> {
    let mut bytes = [0u8; N];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

pub(crate) fn read_i32(reader: &mut impl BufRead) -> Result<i32, std::io::Error> {
    Ok(i32::from_le_bytes(read_bytes::<4>(reader)?))
}

pub(crate) fn read_u32(reader: &mut impl BufRead) -> Result<u32, std::io::Error> {
    Ok(u32::from_le_bytes(read_bytes::<4>(reader)?))
}

pub(crate) fn read_f32(reader: &mut impl BufRead) -> Result<f32, std::io::Error> {
    Ok(f32::from_le_bytes(read_bytes::<4>(reader)?))
}

pub(crate) fn read_bytes_with_len(
    reader: &mut impl BufRead,
    len: usize,
) -> Result<Vec<u8>, std::io::Error> {
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

/// The hyperparameters of the model.
#[derive(Debug, Clone)]
pub struct FixedHyperparameters {
    pub n_vocab: usize,
    pub n_embd: usize,
    pub n_mult: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub n_rot: usize,
    pub tensor_element_type: ElementType,
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError<T> {
    #[error("invalid file magic number: {0}")]
    InvalidMagic(u32),

    #[error("invalid ggml format: version={0}")]
    InvalidFormatVersion(u32),

    #[error("{0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    FailedCast(#[from] std::num::TryFromIntError),

    /// return `ControlFlow::Break` from any of the `cb_*` function to trigger this error
    #[error("user requested interrupt: {0}")]
    UserInterrupted(T),

    #[error("unsupported tensor dtype/f16_: {0}")]
    UnsupportedElementtype(i32),
    
    /// sanity check failed
    #[error("invariant broken: {0}")]
    InvariantBroken(String),
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: Vec<u8>,
    pub n_dims: usize,
    pub dims: [usize; 2],
    pub n_elements: usize,
    pub ftype: ElementType,
}

#[allow(unused_variables)]
pub trait LoadHandler<T> {
    fn cb_container_type(&mut self, model_type: ModelContainerType) -> ControlFlow<T> {
        ControlFlow::Continue(())
    }

    fn cb_hyper_parameters(&mut self, hparams: FixedHyperparameters) -> ControlFlow<T> {
        ControlFlow::Continue(())
    }

    fn cb_vocab_token(&mut self, i: usize, token: Vec<u8>, score: f32) -> ControlFlow<T> {
        ControlFlow::Continue(())
    }

    fn tensor_buffer(&mut self, info: TensorInfo) -> ControlFlow<T, &mut [u8]>;
}

fn retchk<A, B>(model_type: ControlFlow<A, B>) -> Result<B, LoadError<A>> {
    match model_type {
        ControlFlow::Continue(x) => Ok(x),
        ControlFlow::Break(y) => Err(LoadError::UserInterrupted(y)),
    }
}

pub fn load_model_from_reader<T>(
    mut reader: impl BufRead + Seek,
    handler: &mut impl LoadHandler<T>,
) -> Result<(), LoadError<T>> {
    // Verify magic
    let container_type: ModelContainerType = match read_u32(&mut reader)? {
        ggml::FILE_MAGIC_GGMF => ModelContainerType::GGMF,
        ggml::FILE_MAGIC_GGJT => ModelContainerType::GGJT,
        ggml::FILE_MAGIC_UNVERSIONED => ModelContainerType::Unversioned,
        magic => return Err(LoadError::InvalidMagic(magic)),
    };
    retchk(handler.cb_container_type(container_type))?;

    // Load format version
    match container_type {
        ModelContainerType::GGMF | ModelContainerType::GGJT => {
            let _version: u32 = match read_u32(&mut reader)? {
                ggml::FORMAT_VERSION => ggml::FORMAT_VERSION,
                version => return Err(LoadError::InvalidFormatVersion(version)),
            };
        }
        ModelContainerType::Unversioned => {}
    }

    // Load hyper params
    //
    // NOTE: Field order matters! Data is laid out in the file exactly
    // in this order.
    let hparams = FixedHyperparameters {
        n_vocab: read_i32(&mut reader)?.try_into()?,
        n_embd: read_i32(&mut reader)?.try_into()?,
        n_mult: read_i32(&mut reader)?.try_into()?,
        n_head: read_i32(&mut reader)?.try_into()?,
        n_layer: read_i32(&mut reader)?.try_into()?,
        n_rot: read_i32(&mut reader)?.try_into()?,
        tensor_element_type: decode_element_type_res(read_i32(&mut reader)?)?,
    };
    let n_vocab = hparams.n_vocab;
    retchk(handler.cb_hyper_parameters(hparams))?;

    // Load vocabulary
    for i in 0..n_vocab {
        let len = read_u32(&mut reader)?.try_into()?;
        let token = read_bytes_with_len(&mut reader, len)?;
        let token_score = match container_type {
            ModelContainerType::GGMF | ModelContainerType::GGJT => read_f32(&mut reader)?,
            ModelContainerType::Unversioned => {
                // Legacy model, set empty score
                0.
            }
        };
        retchk(handler.cb_vocab_token(i, token, token_score))?;
    }

    // Load tensor data
    match container_type {
        ModelContainerType::GGMF | ModelContainerType::Unversioned => {
            let _file_offset = reader.stream_position()?;
            drop(reader);
            todo!()
        }
        ModelContainerType::GGJT => load_weights_ggjt(&mut reader, handler),
    }
}

fn decode_element_type_res<T>(ftype: i32) -> Result<ElementType, LoadError<T>> {
    match decode_element_type(ftype) {
        Some(x) => Ok(x),
        None => Err(LoadError::UnsupportedElementtype(ftype)),
    }
}

fn load_weights_ggjt<T>(
    reader: &mut (impl BufRead + Seek),
    handler: &mut impl LoadHandler<T>,
) -> Result<(), LoadError<T>> {
    while has_data_left(reader)? {
        // load tensor header
        let n_dims: usize = read_i32(reader)?.try_into()?;
        let name_len = read_i32(reader)?;
        let ftype = decode_element_type_res(read_i32(reader)?)?;

        let mut n_elements: usize = 1;
        let mut dims = [1usize, 1];
        let ne_len = dims.len();
        if !(n_dims <= ne_len) {
            return Err(LoadError::InvariantBroken(format!("{n_dims} <= {ne_len}")));
        }
        #[allow(clippy::needless_range_loop)]
        for i in 0..n_dims {
            let dim: usize = read_i32(reader)?.try_into()?;
            dims[i] = dim;
            n_elements *= dim;
        }

        // load tensor name
        let name = read_bytes_with_len(reader, name_len.try_into()?)?;

        // sanity check
        match ftype {
            ElementType::Q4_0 | ElementType::Q4_1 => {
                if !(dims[0] % 64 == 0) {
                    return Err(LoadError::InvariantBroken(format!("{dims:?}[0] % 64 == 0")));
                }
            }
            _ => {}
        }

        let tensor_info = TensorInfo {
            name, dims, n_dims, n_elements, ftype,
        };

        // load tensor weights
        let offset_curr = reader.stream_position()?;
        let offset_aligned: u64 = (offset_curr + 31) & !31;
        reader.seek(SeekFrom::Start(offset_aligned))?;
    
        let type_size = ggml::type_size(ftype);
        let buf = retchk(handler.tensor_buffer(tensor_info))?;
        let buf_len = buf.len();
        if !(buf_len == type_size * n_elements) {
            return Err(LoadError::InvariantBroken(format!("{buf_len} == {type_size} * {n_elements}")));
        }
        reader.read_exact(buf)?;
    }

    Ok(())
}

