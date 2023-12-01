//! The loader module contains the code for loading a model from disk.
//!
//! To handle a specific model, implement [LoadHandler] for your model
//! and call [load] with an instance of your handler. It is up to you
//! to process the data from the handler and construct your model.

use std::{
    error::Error,
    io::{BufRead, Seek, SeekFrom},
};

use crate::{
    format::{data_size, header_size, ContainerType, ContainerTypeReadError},
    util::{has_data_left, read_bytes_with_len, read_f32, read_i32, read_u32},
    ElementType,
};

use super::LoadError;

#[derive(Debug, Clone)]
/// Information about a [tensor](https://en.wikipedia.org/wiki/Tensor_(machine_learning)) that is being read.
pub struct TensorLoadInfo {
    /// The name of the tensor.
    pub name: String,
    /// The number of dimensions in the tensor.
    pub n_dims: usize,
    /// The dimensions of the tensor.
    pub dims: [usize; 2],
    /// The number of elements in the tensor.
    pub n_elements: usize,
    /// The type of the elements in the tensor.
    pub element_type: ElementType,
    /// start of tensor - start of file
    pub start_offset: u64,
}
impl TensorLoadInfo {
    /// Get the dimensions of the tensor.
    pub fn dims(&self) -> &[usize] {
        &self.dims[0..self.n_dims]
    }

    /// Calculate the size of the tensor's values in bytes.
    pub fn calc_size(&self) -> usize {
        data_size(self.element_type, self.dims().iter().product())
    }

    /// Calculates the absolute size in bytes of the tensor's data, given the mmap flag.
    pub fn calc_absolute_size(&self, mmap: bool) -> usize {
        if mmap {
            header_size()
        } else {
            header_size() + self.calc_size()
        }
    }

    /// Reads the tensor's data from the given reader in an owned fashion.
    ///
    /// The behaviour is undefined if the reader does not correspond to this info.
    ///
    /// Do not use this if loading with `mmap`.
    pub fn read_data<R: BufRead + Seek>(&self, reader: &mut R) -> std::io::Result<Vec<u8>> {
        let n_bytes = self.n_elements * crate::type_size(self.element_type);
        let mut data = vec![0; n_bytes];
        reader.seek(SeekFrom::Start(self.start_offset))?;
        reader.read_exact(&mut data)?;
        Ok(data)
    }
}

#[derive(Debug, Clone)]
/// Information present within GGML [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))
/// that is required to continue loading the model.
pub struct PartialHyperparameters {
    /// The number of tokens in the model's embedded vocabulary.
    pub n_vocab: usize,
}

/// A handler for loading a GGML model.
pub trait LoadHandler<E: Error> {
    /// Called when the [ContainerType] is read.
    fn container_type(&mut self, container_type: ContainerType) -> Result<(), E>;
    /// Called when a token is read so it can be added to the model's embedded vocabulary.
    fn vocabulary_token(&mut self, i: usize, token: Vec<u8>, score: f32) -> Result<(), E>;
    /// Called when the model's hyperparameters need to be read.
    fn read_hyperparameters(
        &mut self,
        reader: &mut dyn BufRead,
    ) -> Result<PartialHyperparameters, E>;
    /// Called when a new [crate::Tensor] is read for the model.
    fn tensor_buffer(&mut self, info: TensorLoadInfo) -> Result<(), E>;
}

/// Load a GGML model from a `reader` with the [LoadHandler], which will be called when certain events occur.
pub fn load<E: Error, R: BufRead + Seek>(
    reader: &mut R,
    handler: &mut impl LoadHandler<E>,
) -> Result<(), LoadError<E>> {
    // Verify magic
    let container_type = ContainerType::read(reader).map_err(|e| match e {
        ContainerTypeReadError::InvalidMagic(magic) => LoadError::InvalidMagic(magic),
        ContainerTypeReadError::Io(io) => LoadError::Io(io),
    })?;

    match container_type {
        ContainerType::Ggml
        | ContainerType::Ggmf(1)
        | ContainerType::Ggjt(1..=3)
        | ContainerType::Ggla(1) => {}
        _ => return Err(LoadError::InvalidFormatVersion(container_type)),
    }

    handler
        .container_type(container_type)
        .map_err(LoadError::ImplementationError)?;

    // Load hyper params
    let hparams = handler
        .read_hyperparameters(reader)
        .map_err(LoadError::ImplementationError)?;
    let n_vocab = hparams.n_vocab;

    // Load vocabulary
    for i in 0..n_vocab {
        let len = read_u32(reader)?.try_into()?;
        let token = read_bytes_with_len(reader, len)?;
        let token_score = match container_type {
            ContainerType::Ggmf(_version) | ContainerType::Ggjt(_version) => read_f32(reader)?,
            ContainerType::Ggml | ContainerType::Ggla(_) => {
                // Legacy model, set empty score
                0.
            }
            ContainerType::Gguf(_) => {
                unreachable!("This loader should not be used with GGUF")
            }
        };
        handler
            .vocabulary_token(i, token, token_score)
            .map_err(LoadError::ImplementationError)?;
    }

    // Load tensor data
    match container_type {
        ContainerType::Ggmf(_) | ContainerType::Ggml => load_weights(reader, handler, false),
        ContainerType::Ggjt(_version) | ContainerType::Ggla(_version) => {
            load_weights(reader, handler, true)
        }
        ContainerType::Gguf(_) => {
            unreachable!("This loader should not be used with GGUF")
        }
    }
}

/// # Params
///
/// `align`
/// align to 4 bytes before reading tensor weights
fn load_weights<E: Error, R: BufRead + Seek>(
    reader: &mut R,
    handler: &mut impl LoadHandler<E>,
    align: bool,
) -> Result<(), LoadError<E>> {
    while has_data_left(reader)? {
        // load tensor header
        let n_dims: usize = read_i32(reader)?.try_into()?;
        let name_len = read_i32(reader)?;
        let ftype = read_u32(reader)?;

        let mut n_elements: usize = 1;
        let mut dims = [1usize, 1];
        let ne_len = dims.len();
        if n_dims > ne_len {
            return Err(LoadError::InvariantBroken(format!("{n_dims} <= {ne_len}")));
        }

        #[allow(clippy::needless_range_loop)]
        for i in 0..n_dims {
            let dim: usize = read_i32(reader)?.try_into()?;
            dims[i] = dim;
            n_elements *= dim;
        }

        // load tensor name
        let name = String::from_utf8(read_bytes_with_len(reader, name_len.try_into()?)?)?;
        let ftype =
            crate::Type::try_from(ftype).map_err(|_| LoadError::UnsupportedElementType {
                tensor_name: name.clone(),
                ftype,
            })?;

        // sanity check
        match ftype {
            ElementType::Q4_0 | ElementType::Q4_1 => {
                if dims[0] % 64 != 0 {
                    return Err(LoadError::InvariantBroken(format!("{dims:?}[0] % 64 == 0")));
                }
            }
            _ => {}
        }

        // load tensor weights
        let offset_curr = reader.stream_position()?;
        let offset_aligned: u64 = if align {
            (offset_curr + 31) & !31
        } else {
            offset_curr
        };

        let tensor_info = TensorLoadInfo {
            name,
            dims,
            n_dims,
            n_elements,
            element_type: ftype,
            start_offset: offset_aligned,
        };
        let n_bytes = tensor_info.calc_size();
        handler
            .tensor_buffer(tensor_info)
            .map_err(LoadError::ImplementationError)?;
        reader.seek(SeekFrom::Start(offset_aligned + n_bytes as u64))?;
    }

    Ok(())
}
