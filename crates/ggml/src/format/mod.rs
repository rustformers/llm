//! Loading and saving of GGML-related files.

use std::error::Error;

use crate::{util::FormatMagic, ElementType};

pub mod ggml;
pub mod gguf;

#[derive(Debug, thiserror::Error)]
/// Errors that can occur while loading a model.
pub enum LoadError<E: Error> {
    #[error("invalid file magic number: {0}")]
    /// The file magic number is invalid.
    InvalidMagic(FormatMagic),
    #[error("invalid ggml format: format={0:?}")]
    /// An unsupported format version was found.
    InvalidFormatVersion(ggml::ContainerType),
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

/// Returns the size occupied by a tensor's data in bytes given the element type and number of elements.
pub(crate) fn data_size(element_type: ElementType, n_elements: usize) -> usize {
    (crate::type_size(element_type) * n_elements) / crate::blck_size(element_type)
}

/// Returns the size of the ggml tensor header in bytes.
pub(crate) fn header_size() -> usize {
    crate::Tensor::C_TYPE_SIZE + crate::OBJECT_SIZE
}

/// Returns the size of a tensor in bytes given the element type and number of elements. This includes the tensor's header.
pub fn tensor_size(element_type: ElementType, n_elements: usize) -> usize {
    header_size() + data_size(element_type, n_elements)
}
