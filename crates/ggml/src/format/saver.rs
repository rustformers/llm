//! The saver module implements a way to save a model to disk in the GGJT format.
//!
//! To implement a saver for your model, implement [SaveHandler] for your model
//! and provide data as appropriate, then call [save] with an instance of
//! your handler.

use std::{
    error::Error,
    io::{Seek, Write},
};

use crate::{util, ContainerType, ElementType};

#[derive(Debug, thiserror::Error)]
/// Errors that can occur while writing a model.
pub enum SaveError<E: Error> {
    #[error("non-specific I/O error")]
    /// A non-specific IO error.
    Io(#[from] std::io::Error),
    #[error("invalid integer conversion")]
    /// One of the integers encountered could not be converted to a more appropriate type.
    InvalidIntegerConversion(#[from] std::num::TryFromIntError),
    #[error("implementation error")]
    /// An error `E` was returned by the implementation of the loader.
    ImplementationError(#[source] E),
    #[error("invariant broken: {0}")]
    /// An invariant was broken.
    InvariantBroken(String),
    /// An attempt was made to save a model with a container type that does not
    /// support vocabulary scoring, despite the model having a scored vocabulary.
    #[error("container type does not support vocabulary scoring")]
    VocabularyScoringNotSupported,
}

/// A handler for saving a GGML model.
pub trait SaveHandler<E: Error> {
    /// Called when the hyperparameters must be written.
    fn write_hyperparameters(&mut self, writer: &mut dyn Write) -> Result<(), E>;

    /// Called when information for a tensor is to be written.
    fn tensor_data(&mut self, tensor_name: &str) -> Result<TensorSaveInfo, E>;
}

/// Information about a [tensor](https://en.wikipedia.org/wiki/Tensor_(machine_learning)) that is to be saved.
#[derive(Clone, PartialEq, Debug)]
pub struct TensorSaveInfo {
    /// The number of dimensions in the tensor.
    pub n_dims: usize,
    /// The dimensions of the tensor.
    pub dims: [usize; 2],
    /// The type of the elements in the tensor.
    pub element_type: ElementType,
    /// The data to save to disk.
    // TODO: This can be done more efficiently by borrowing the data, but
    // I wanted to avoid the lifetime parameter for now, especially as
    // the naive solution would borrow `TensorData` for the lifetime of the
    // handler, which is obviously not ideal if you're trying to transcode
    // an existing file tensor-by-tensor.
    pub data: Vec<u8>,
}

/// The container of the model to save.
///
/// This is separate from [ContainerType] to ensure that the user
/// does not accidentally use an unsupported container type.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SaveContainerType {
    /// The GGML container.
    Ggml,
    /// The GGJT container.
    GgjtV3,
}
impl From<SaveContainerType> for ContainerType {
    fn from(value: SaveContainerType) -> Self {
        match value {
            SaveContainerType::Ggml => ContainerType::Ggml,
            SaveContainerType::GgjtV3 => ContainerType::Ggjt(3),
        }
    }
}

/// Saves a model to the given writer.
///
/// Only GGML and GGJT version 2 are supported. If using GGML,
/// the vocabulary *must* have scores of 0.0.
pub fn save<E: Error, W: Write + Seek>(
    writer: &mut W,
    handler: &mut dyn SaveHandler<E>,
    container_type: SaveContainerType,
    vocabulary: &[(Vec<u8>, f32)],
    tensor_names: &[String],
) -> Result<(), SaveError<E>> {
    // Write header and hyperparameters
    ContainerType::from(container_type).write(writer)?;

    if container_type == SaveContainerType::Ggml
        && vocabulary.iter().any(|(_, score)| *score != 0.0)
    {
        return Err(SaveError::VocabularyScoringNotSupported);
    }

    handler
        .write_hyperparameters(writer)
        .map_err(SaveError::ImplementationError)?;

    // Write vocabulary
    for (token, score) in vocabulary {
        util::write_u32(writer, token.len().try_into()?)?;
        writer.write_all(token)?;

        if container_type != SaveContainerType::Ggml {
            util::write_f32(writer, *score)?;
        }
    }

    // Write tensors
    for name in tensor_names {
        let TensorSaveInfo {
            n_dims,
            dims,
            element_type,
            data,
        } = handler
            .tensor_data(name)
            .map_err(SaveError::ImplementationError)?;

        match element_type {
            ElementType::Q4_0 | ElementType::Q4_1 => {
                if dims[0] % 64 != 0 {
                    return Err(SaveError::InvariantBroken(format!("{dims:?}[0] % 64 == 0")));
                }
            }
            _ => {}
        }

        // Write tensor header
        util::write_i32(writer, n_dims.try_into()?)?;
        util::write_i32(writer, name.len().try_into()?)?;
        util::write_u32(writer, element_type.into())?;
        for &dim in &dims[0..n_dims] {
            util::write_i32(writer, dim.try_into()?)?;
        }

        // Write tensor name
        writer.write_all(name.as_bytes())?;

        // Align to nearest 32 bytes
        if container_type != SaveContainerType::Ggml {
            let offset_curr = writer.stream_position()?;
            let offset_aligned = (offset_curr + 31) & !31;
            let padding = usize::try_from(offset_aligned - offset_curr)?;
            writer.write_all(&vec![0; padding])?;
        }

        // Write tensor data
        writer.write_all(&data)?;
    }

    Ok(())
}
