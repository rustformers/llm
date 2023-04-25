#![deny(missing_docs)]
//! A reader and writer for the `ggml` model format.
//!
//! The reader supports the GGML, GGMF and GGJT container formats, but
//! only single-part models.
//!
//! The writer isn't implemented yet. It will support the GGJT container
//! format only.

/// Utilities for reading and writing.
pub mod util;

mod loader;
mod saver;
#[cfg(test)]
mod tests;

pub use loader::{
    data_size, load_model, LoadError, LoadHandler, PartialHyperparameters, TensorInfo,
};
pub use saver::{save_model, SaveError, SaveHandler, TensorData};

/// The type of a tensor element.
pub type ElementType = ggml::Type;

#[derive(Debug, PartialEq, Clone, Copy)]
/// The format of the file containing the model.
pub enum ContainerType {
    /// `GGML`: legacy format, oldest ggml tensor file format
    Ggml,
    /// `GGMF`: also legacy format. Introduces versioning. Newer than GGML, older than GGJT.
    Ggmf,
    /// `GGJT`: mmap-able format.
    Ggjt,
}
impl ContainerType {
    /// Does this container type support mmap?
    pub fn support_mmap(&self) -> bool {
        match self {
            ContainerType::Ggml => false,
            ContainerType::Ggmf => false,
            ContainerType::Ggjt => true,
        }
    }
}
