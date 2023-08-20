//! Loading and saving of [GGML](https://github.com/ggerganov/ggml) files.

mod loader;
mod saver;

pub use loader::*;
pub use saver::*;

#[cfg(test)]
mod tests;

use crate::{format::LoadError, util};

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
    pub fn read<E: std::error::Error>(
        reader: &mut dyn std::io::BufRead,
    ) -> Result<Self, LoadError<E>> {
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
            magic => return Err(LoadError::InvalidMagic(util::FormatMagic(magic))),
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
