#![allow(missing_docs)]

use std::io::{BufRead, BufWriter, Seek, Write};

use super::{data_size, header_size, ContainerType, ContainerTypeReadError};
use crate::{util, ElementType};

use ggml_sys::ggml_type;
use indexmap::IndexMap;
use thiserror::Error;

mod metadata;
pub use metadata::*;

pub const DEFAULT_ALIGNMENT: u32 = 32;
pub const META_TENSOR_DATA_LAYOUT: &str = "Meta AI original pth";

#[derive(Debug, Error)]
/// Errors that can occur while loading a model.
pub enum GgufLoadError {
    #[error("invalid GGUF file magic value: {0}")]
    /// The file magic number is invalid.
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
    #[error("unsupported tensor type {ftype} for tensor {tensor_name}")]
    /// One of the tensors encountered had an unsupported data type.
    UnsupportedElementType {
        /// The name of the tensor.
        tensor_name: String,
        /// The format type that was encountered.
        ftype: u32,
    },
}

#[derive(Debug, Error)]
/// Errors that can occur while saving a model.
pub enum GgufSaveError {
    // TODO!
}

pub type TensorInfos = IndexMap<String, TensorInfo>;

#[derive(Debug, Clone, PartialEq)]
pub struct Gguf {
    pub metadata: Metadata,
    pub tensor_infos: TensorInfos,
    pub tensor_data_position: u64,
}
impl Gguf {
    pub fn load<R: BufRead + Seek>(reader: &mut R) -> Result<Self, GgufLoadError> {
        let container = ContainerType::read(reader).map_err(|e| match e {
            ContainerTypeReadError::InvalidMagic(magic) => GgufLoadError::InvalidMagic(magic),
            ContainerTypeReadError::Io(io) => GgufLoadError::Io(io),
        })?;
        if ![
            ContainerType::Gguf(1),
            ContainerType::Gguf(2),
            ContainerType::Gguf(3),
        ]
        .contains(&container)
        {
            return Err(GgufLoadError::InvalidFormatVersion(container));
        }

        let ctx = GgufContext {
            use_64_bit_length: container == ContainerType::Gguf(2)
                || container == ContainerType::Gguf(3),
        };

        let tensor_count = util::read_length(reader, ctx.use_64_bit_length)?;
        let metadata_kv_count = util::read_length(reader, ctx.use_64_bit_length)?;

        let mut metadata = IndexMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let (key, value) = MetadataValue::read_key_value(&ctx, reader)?;
            metadata.insert(key, value);
        }
        let metadata = Metadata(metadata);

        let alignment = metadata
            .get_optional("general.alignment")
            .and_then(|v| v.as_uint32())
            .unwrap_or(DEFAULT_ALIGNMENT) as u64;

        let mut tensor_infos = IndexMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let (key, value) = TensorInfo::read_name_value(&ctx, reader)?;
            tensor_infos.insert(key, value);
        }

        let tensor_data_position = align_offset(reader.stream_position()?, alignment);

        Ok(Gguf {
            metadata,
            tensor_infos,
            tensor_data_position,
        })
    }

    /// Saves the GGUF file to the given writer.
    ///
    /// `get_tensor_size` is a function that returns the size of a tensor's data in bytes.
    /// `write_tensor_data` is a function that writes the tensor's data to the writer; the data
    /// must be the same length as the value returned by `get_tensor_size`.
    ///
    /// The `offset` in `TensorInfo` will be ignored and the correct offset will be calculated
    /// automatically.
    pub fn save<W: Write + Seek>(
        &self,
        writer: &mut BufWriter<W>,
        mut write_tensor_data: impl FnMut(&mut BufWriter<W>, &str, &TensorInfo) -> std::io::Result<()>,
    ) -> std::io::Result<()> {
        // Write header
        let container = ContainerType::Gguf(2);
        container.write(writer)?;

        let ctx = GgufContext {
            use_64_bit_length: true,
        };

        util::write_length(writer, ctx.use_64_bit_length, self.tensor_infos.len())?;
        util::write_length(writer, ctx.use_64_bit_length, self.metadata.0.len())?;

        // Write metadata
        for (key, value) in &self.metadata.0 {
            value.write_key_value(&ctx, writer, key)?;
        }

        // Write tensor infos
        let alignment = self
            .metadata
            .get_optional("general.alignment")
            .and_then(|v| v.as_uint32())
            .unwrap_or(DEFAULT_ALIGNMENT) as u64;

        // Pre-plan the write before writing the tensor data.
        #[derive(Debug)]
        struct TensorWrite {
            name: String,
            info: TensorInfo,
            size: usize,
        }
        let mut tensors = vec![];
        let mut next_offset = 0;
        for (name, tensor_info) in &self.tensor_infos {
            let size = tensor_info.calc_size();
            tensors.push(TensorWrite {
                name: name.clone(),
                info: TensorInfo {
                    offset: next_offset,
                    ..tensor_info.clone()
                },
                size,
            });

            next_offset = align_offset(next_offset + size as u64, alignment);
        }

        for write in &tensors {
            write.info.write_name_value(&ctx, writer, &write.name)?;
        }

        // Write tensors
        let stream_position = writer.stream_position()?;
        let tensor_data_position = align_offset(stream_position, alignment);
        assert!(tensor_data_position > stream_position);
        util::write_zero_bytes(writer, (tensor_data_position - stream_position) as usize)?;

        for write in &tensors {
            write_tensor_data(writer, &write.name, &write.info)?;

            let stream_position = writer.stream_position()?;
            assert!(
                stream_position == tensor_data_position + write.info.offset + write.size as u64
            );
            let next_position = align_offset(stream_position, alignment);
            util::write_zero_bytes(writer, (next_position - stream_position) as usize)?;
        }

        Ok(())
    }
}

fn align_offset(offset: u64, alignment: u64) -> u64 {
    offset + (alignment - (offset % alignment)) % alignment
}

struct GgufContext {
    use_64_bit_length: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub dimensions: Vec<usize>,
    pub element_type: ElementType,
    /// This offset is relative to `tensor_data`, not to the start
    /// of the file, to make it easier for writers to write the file.
    pub offset: u64,
}
impl TensorInfo {
    fn read_name_value(
        ctx: &GgufContext,
        reader: &mut dyn BufRead,
    ) -> Result<(String, Self), GgufLoadError> {
        let name = util::read_string(reader, ctx.use_64_bit_length)?;

        let dimension_count = util::read_u32(reader)? as usize;
        let dimensions = (0..dimension_count)
            .map(|_| util::read_length(reader, ctx.use_64_bit_length))
            .collect::<Result<Vec<_>, _>>()?;

        let element_type = util::read_u32(reader)?;
        let element_type = ElementType::try_from(element_type).map_err(|_| {
            GgufLoadError::UnsupportedElementType {
                tensor_name: name.clone(),
                ftype: element_type,
            }
        })?;

        let offset = util::read_u64(reader)?;

        Ok((
            name,
            Self {
                dimensions,
                element_type,
                offset,
            },
        ))
    }

    fn write_name_value(
        &self,
        ctx: &GgufContext,
        writer: &mut dyn Write,
        name: &str,
    ) -> std::io::Result<()> {
        util::write_string(writer, ctx.use_64_bit_length, name)?;

        util::write_u32(writer, self.dimensions.len().try_into().unwrap())?;
        for dimension in &self.dimensions {
            util::write_length(writer, ctx.use_64_bit_length, *dimension)?;
        }

        util::write_u32(writer, ggml_type::from(self.element_type))?;
        util::write_u64(writer, self.offset)?;

        Ok(())
    }

    /// Calculate the size of the tensor's values in bytes.
    pub fn calc_size(&self) -> usize {
        data_size(self.element_type, self.dimensions.iter().product())
    }

    /// Calculates the absolute size in bytes of the tensor's data, given the mmap flag.
    pub fn calc_absolute_size(&self, mmap: bool) -> usize {
        if mmap {
            header_size()
        } else {
            header_size() + self.calc_size()
        }
    }
}
