//! Loading and saving of GGML-related files.

use thiserror::Error;

use crate::{util, ElementType};

#[cfg(feature = "pre-gguf-formats")]
pub mod ggml;
pub mod gguf;

/// Magic constant for `ggml` files (unversioned).
pub const FILE_MAGIC_GGML: [u8; 4] = *b"lmgg";
/// Magic constant for `ggml` files (versioned, ggmf).
pub const FILE_MAGIC_GGMF: [u8; 4] = *b"fmgg";
/// Magic constant for `ggml` files (versioned, ggjt).
pub const FILE_MAGIC_GGJT: [u8; 4] = *b"tjgg";
/// Magic constant for `ggla` files (LoRA adapter).
pub const FILE_MAGIC_GGLA: [u8; 4] = *b"algg";
/// Magic constant for `gguf` files.
pub const FILE_MAGIC_GGUF: [u8; 4] = *b"GGUF";

/// Errors that can occur while reading the container type.
#[derive(Debug, Error)]
pub enum ContainerTypeReadError {
    /// The magic value was invalid.
    #[error("invalid magic value: {0}")]
    InvalidMagic(util::FileMagic),
    /// An I/O error occurred.
    #[error("I/O error")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, PartialEq, Clone, Copy)]
/// The format of the file containing the model.
pub enum ContainerType {
    /// Legacy format, oldest ggml tensor file format
    Ggml,
    /// Legacy format. Introduces versioning. Newer than GGML, older than GGJT.
    Ggmf(u32),
    /// [mmap](https://en.wikipedia.org/wiki/Mmap)-able format.
    Ggjt(u32),
    /// LoRA adapter format.
    Ggla(u32),
    /// GGUF format. Current version of the format.
    Gguf(u32),
}
impl ContainerType {
    /// Does this container type support mmap?
    pub fn support_mmap(&self) -> bool {
        match self {
            ContainerType::Ggml => false,
            ContainerType::Ggmf(_) => false,
            ContainerType::Ggla(_) => false,
            ContainerType::Ggjt(_) => true,
            ContainerType::Gguf(_) => true,
        }
    }

    /// Read the container type from a reader.
    pub fn read(reader: &mut dyn std::io::BufRead) -> Result<Self, ContainerTypeReadError> {
        // Verify magic
        let magic = util::read_bytes::<4>(reader)?;
        let container_type: ContainerType = match magic {
            FILE_MAGIC_GGML => ContainerType::Ggml,
            FILE_MAGIC_GGMF => {
                let version = util::read_u32(reader)?;
                ContainerType::Ggmf(version)
            }
            FILE_MAGIC_GGJT => {
                let version = util::read_u32(reader)?;
                ContainerType::Ggjt(version)
            }
            FILE_MAGIC_GGLA => {
                let version = util::read_u32(reader)?;
                ContainerType::Ggla(version)
            }
            FILE_MAGIC_GGUF => {
                let version = util::read_u32(reader)?;
                ContainerType::Gguf(version)
            }
            magic => return Err(ContainerTypeReadError::InvalidMagic(util::FileMagic(magic))),
        };

        Ok(container_type)
    }

    /// Write the container type to a writer.
    pub fn write(&self, writer: &mut dyn std::io::Write) -> std::io::Result<()> {
        match self {
            ContainerType::Ggml => {
                writer.write_all(&FILE_MAGIC_GGML)?;
            }
            ContainerType::Ggmf(version) => {
                writer.write_all(&FILE_MAGIC_GGMF)?;
                util::write_u32(writer, *version)?;
            }
            ContainerType::Ggjt(version) => {
                writer.write_all(&FILE_MAGIC_GGJT)?;
                util::write_u32(writer, *version)?;
            }
            ContainerType::Ggla(version) => {
                writer.write_all(&FILE_MAGIC_GGLA)?;
                util::write_u32(writer, *version)?;
            }
            ContainerType::Gguf(version) => {
                writer.write_all(&FILE_MAGIC_GGUF)?;
                util::write_u32(writer, *version)?;
            }
        }
        Ok(())
    }
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
