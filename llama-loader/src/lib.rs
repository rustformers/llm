//! standalone model loader
//!
//! Only the hyperparameter is llama-specific. Everything else can be reused for other LLM.
#![allow(clippy::nonminimal_bool)]

pub mod util;

use std::{
    io::{BufRead, Seek, SeekFrom},
    ops::ControlFlow,
};
use util::*;

pub type ElementType = ggml::Type;

/// file type containing the model
#[derive(Debug, PartialEq, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub enum ContainerType {
    /// legacy format, oldest ggml tensor file format
    GGML,
    /// also legacy format, newer than GGML, older than GGJT
    GGMF,
    /// mmap-able format
    GGJT,
}

pub fn decode_element_type(ftype: i32) -> Option<ElementType> {
    match ftype {
        0 => Some(ggml::Type::F32),
        1 => Some(ggml::Type::F16),
        2 => Some(ggml::Type::Q4_0),
        3 => Some(ggml::Type::Q4_1),
        _ => None,
    }
}

pub fn encode_element_type(element_type: ElementType) -> Option<i32> {
    match element_type {
        ggml::Type::F32 => Some(0),
        ggml::Type::F16 => Some(1),
        ggml::Type::Q4_0 => Some(2),
        ggml::Type::Q4_1 => Some(3),
        _ => None,
    }
}

/// The hyperparameters of the model.
#[derive(Debug, Clone)]
pub struct LlamaHyperparameters {
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
    /// start of tensor - start of file
    pub start_offset: u64,
}

#[allow(unused_variables)]
pub trait LoadHandler<T> {
    fn got_container_type(&mut self, model_type: ContainerType) -> ControlFlow<T> {
        ControlFlow::Continue(())
    }

    fn got_hyper_parameters(&mut self, hparams: LlamaHyperparameters) -> ControlFlow<T> {
        ControlFlow::Continue(())
    }

    fn got_vocab_token(&mut self, i: usize, token: Vec<u8>, score: f32) -> ControlFlow<T> {
        ControlFlow::Continue(())
    }

    /// # Returns
    /// 
    /// `None` to skip copying
    /// `Some(buf)` to provide a buffer for copying weights into
    fn get_tensor_buffer(&mut self, info: TensorInfo) -> ControlFlow<T, Option<&mut [u8]>> {
        ControlFlow::Continue(None)
    }
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
    let container_type: ContainerType = match read_u32(&mut reader)? {
        ggml::FILE_MAGIC_GGMF => ContainerType::GGMF,
        ggml::FILE_MAGIC_GGJT => ContainerType::GGJT,
        ggml::FILE_MAGIC_UNVERSIONED => ContainerType::GGML,
        magic => return Err(LoadError::InvalidMagic(magic)),
    };
    retchk(handler.got_container_type(container_type))?;

    // Load format version
    match container_type {
        ContainerType::GGMF | ContainerType::GGJT => {
            let _version: u32 = match read_u32(&mut reader)? {
                ggml::FORMAT_VERSION => ggml::FORMAT_VERSION,
                version => return Err(LoadError::InvalidFormatVersion(version)),
            };
        }
        ContainerType::GGML => {}
    }

    // Load hyper params
    //
    // NOTE: Field order matters! Data is laid out in the file exactly
    // in this order.
    let hparams = LlamaHyperparameters {
        n_vocab: read_i32(&mut reader)?.try_into()?,
        n_embd: read_i32(&mut reader)?.try_into()?,
        n_mult: read_i32(&mut reader)?.try_into()?,
        n_head: read_i32(&mut reader)?.try_into()?,
        n_layer: read_i32(&mut reader)?.try_into()?,
        n_rot: read_i32(&mut reader)?.try_into()?,
        tensor_element_type: decode_element_type_res(read_i32(&mut reader)?)?,
    };
    let n_vocab = hparams.n_vocab;
    retchk(handler.got_hyper_parameters(hparams))?;

    // Load vocabulary
    for i in 0..n_vocab {
        let len = read_u32(&mut reader)?.try_into()?;
        let token = read_bytes_with_len(&mut reader, len)?;
        let token_score = match container_type {
            ContainerType::GGMF | ContainerType::GGJT => read_f32(&mut reader)?,
            ContainerType::GGML => {
                // Legacy model, set empty score
                0.
            }
        };
        retchk(handler.got_vocab_token(i, token, token_score))?;
    }

    // Load tensor data
    match container_type {
        ContainerType::GGMF | ContainerType::GGML => {
            let _file_offset = reader.stream_position()?;
            drop(reader);
            todo!()
        }
        ContainerType::GGJT => load_weights_ggjt(&mut reader, handler),
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

        // load tensor weights
        let offset_curr = reader.stream_position()?;
        let offset_aligned: u64 = (offset_curr + 31) & !31;

        let tensor_info = TensorInfo {
            name,
            dims,
            n_dims,
            n_elements,
            ftype,
            start_offset: offset_aligned
        };

        
        let type_size = ggml::type_size(ftype);
        if let Some(buf) = retchk(handler.get_tensor_buffer(tensor_info))? {
            reader.seek(SeekFrom::Start(offset_aligned))?;
            let buf_len = buf.len();
            if !(buf_len == type_size * n_elements) {
                return Err(LoadError::InvariantBroken(format!(
                    "{buf_len} == {type_size} * {n_elements}"
                )));
            }
            reader.read_exact(buf)?;
        } else {
            // skip if no buffer is given
            reader.seek(SeekFrom::Start(offset_aligned + (type_size * n_elements) as u64))?;
        }
    }

    Ok(())
}
