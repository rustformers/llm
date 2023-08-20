//! Loading and saving of [GGML](https://github.com/ggerganov/ggml) files.

mod loader;
mod saver;

pub use loader::*;
pub use saver::*;

#[cfg(test)]
mod tests;

use crate::{format::LoadError, util};

/// Magic constant for `ggml` files (unversioned).
pub const FILE_MAGIC_GGML: u32 = 0x67676d6c;
/// Magic constant for `ggml` files (versioned, ggmf).
pub const FILE_MAGIC_GGMF: u32 = 0x67676d66;
/// Magic constant for `ggml` files (versioned, ggjt).
pub const FILE_MAGIC_GGJT: u32 = 0x67676a74;
/// Magic constant for `ggla` files (LoRA adapter).
pub const FILE_MAGIC_GGLA: u32 = 0x67676C61;

#[derive(Debug, PartialEq, Clone, Copy)]
/// The format of the file containing the model.
pub enum ContainerType {
    /// Legacy format, oldest ggml tensor file format
    Ggml,
    /// Legacy format. Introduces versioning. Newer than GGML, older than GGJT.
    Ggmf(u32),
    /// [mmap](https://en.wikipedia.org/wiki/Mmap)-able format. Current version of the format.
    Ggjt(u32),
    /// LoRA adapter format.
    Ggla(u32),
}
impl ContainerType {
    /// Does this container type support mmap?
    pub fn support_mmap(&self) -> bool {
        match self {
            ContainerType::Ggml => false,
            ContainerType::Ggmf(_) => false,
            ContainerType::Ggla(_) => false,
            ContainerType::Ggjt(_) => true,
        }
    }

    /// Read the container type from a reader.
    pub fn read<E: std::error::Error>(
        reader: &mut dyn std::io::BufRead,
    ) -> Result<Self, LoadError<E>> {
        // Verify magic
        let magic = util::read_u32(reader)?;
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
            magic => return Err(LoadError::InvalidMagic(util::FormatMagic(magic))),
        };

        Ok(container_type)
    }

    /// Write the container type to a writer.
    pub fn write(&self, writer: &mut dyn std::io::Write) -> std::io::Result<()> {
        match self {
            ContainerType::Ggml => {
                util::write_u32(writer, FILE_MAGIC_GGML)?;
            }
            ContainerType::Ggmf(version) => {
                util::write_u32(writer, FILE_MAGIC_GGMF)?;
                util::write_u32(writer, *version)?;
            }
            ContainerType::Ggjt(version) => {
                util::write_u32(writer, FILE_MAGIC_GGJT)?;
                util::write_u32(writer, *version)?;
            }
            ContainerType::Ggla(version) => {
                util::write_u32(writer, FILE_MAGIC_GGLA)?;
                util::write_u32(writer, *version)?;
            }
        }
        Ok(())
    }
}
