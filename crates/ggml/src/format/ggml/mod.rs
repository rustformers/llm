//! Loading and saving of [GGML](https://github.com/ggerganov/ggml) files.

mod loader;
mod saver;

use std::error::Error;

use super::ContainerType;
use crate::util;

pub use loader::*;
pub use saver::*;

#[cfg(test)]
mod tests;

#[derive(Debug, thiserror::Error)]
/// Errors that can occur while loading a model.
pub enum LoadError<E: Error> {
    #[error("invalid file magic value: {0}")]
    /// The file's magic value is invalid.
    InvalidMagic(util::FileMagic),
    #[error("invalid ggml format: format={0:?}")]
    /// An unsupported format version was found.
    InvalidFormatVersion(ContainerType),
    #[error("non-specific I/O error")]
    /// A non-specific IO error.
    Io(#[from] std::io::Error),
    #[error("could not convert bytes to a UTF-8 string")]
    /// One of the strings encountered was not valid UTF-8.
    InvalidUtf8(#[from] std::string::FromUtf8Error),
    #[error("invalid integer conversion")]
    /// One of the integers encountered could not be converted to a more appropriate type.
    InvalidIntegerConversion(#[from] std::num::TryFromIntError),
    #[error("implementation error")]
    /// An error `E` was returned by the implementation of the loader.
    ImplementationError(#[source] E),
    #[error("unsupported tensor type {ftype} for tensor {tensor_name}")]
    /// One of the tensors encountered had an unsupported data type.
    UnsupportedElementType {
        /// The name of the tensor.
        tensor_name: String,
        /// The format type that was encountered.
        ftype: u32,
    },
    #[error("invariant broken: {0}")]
    /// An invariant was broken.
    InvariantBroken(String),
}
