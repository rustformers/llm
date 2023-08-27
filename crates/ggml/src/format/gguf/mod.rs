#![allow(missing_docs)]

use std::{
    collections::HashMap,
    convert::Infallible,
    io::{BufRead, Seek},
};

use crate::{util, ElementType};

use super::{ggml::ContainerType, LoadError};

pub const DEFAULT_ALIGNMENT: u32 = 32;

#[derive(Debug, Clone, PartialEq)]
pub struct Gguf {
    pub metadata: HashMap<String, MetadataValue>,
    pub tensor_infos: HashMap<String, TensorInfo>,
    pub tensor_data_position: u64,
}
impl Gguf {
    pub fn load<R: BufRead + Seek>(reader: &mut R) -> Result<Self, LoadError<Infallible>> {
        let container = ContainerType::read(reader)?;
        if ![ContainerType::Gguf(1), ContainerType::Gguf(2)].contains(&container) {
            return Err(LoadError::InvalidFormatVersion(container));
        }

        let ctx = GgufContext {
            use_64_bit_length: container == ContainerType::Gguf(2),
        };

        let tensor_count = util::read_length(reader, ctx.use_64_bit_length)?;
        let metadata_kv_count = util::read_length(reader, ctx.use_64_bit_length)?;

        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let (key, value) = MetadataValue::read_key_value(&ctx, reader)?;
            metadata.insert(key, value);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_uint32())
            .unwrap_or(DEFAULT_ALIGNMENT) as u64;

        let mut tensor_infos = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let (key, value) = TensorInfo::read_name_value(&ctx, reader)?;
            tensor_infos.insert(key, value);
        }

        let tensor_data_position = {
            let stream_position = reader.stream_position()?;
            stream_position + (alignment - (stream_position % alignment)) % alignment
        };

        Ok(Gguf {
            metadata,
            tensor_infos,
            tensor_data_position,
        })
    }
}

struct GgufContext {
    use_64_bit_length: bool,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MetadataValueType {
    /// The value is a 8-bit unsigned integer.
    UInt8 = 0,
    /// The value is a 8-bit signed integer.
    Int8 = 1,
    /// The value is a 16-bit unsigned little-endian integer.
    UInt16 = 2,
    /// The value is a 16-bit signed little-endian integer.
    Int16 = 3,
    /// The value is a 32-bit unsigned little-endian integer.
    UInt32 = 4,
    /// The value is a 32-bit signed little-endian integer.
    Int32 = 5,
    /// The value is a 32-bit IEEE754 floating point number.
    Float32 = 6,
    /// The value is a boolean.
    /// 1-byte value where 0 is false and 1 is true.
    /// Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    Bool = 7,
    /// The value is a UTF-8 non-null-terminated string, with length prepended.
    String = 8,
    /// The value is an array of other values, with the length and type prepended.
    ///
    /// Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    Array = 9,
    /// The value is a 64-bit unsigned little-endian integer.
    /// Implemented in GGUFv2.
    UInt64 = 10,
    /// The value is a 64-bit signed little-endian integer.
    /// Implemented in GGUFv2.
    Int64 = 11,
    /// The value is a 64-bit IEEE754 floating point number.
    /// Implemented in GGUFv2.
    Float64 = 12,
}
impl TryFrom<u32> for MetadataValueType {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        // TODO: consider a macro solution to this?
        for test_value in [
            MetadataValueType::UInt8,
            MetadataValueType::Int8,
            MetadataValueType::UInt16,
            MetadataValueType::Int16,
            MetadataValueType::UInt32,
            MetadataValueType::Int32,
            MetadataValueType::Float32,
            MetadataValueType::Bool,
            MetadataValueType::String,
            MetadataValueType::Array,
            MetadataValueType::UInt64,
            MetadataValueType::Int64,
            MetadataValueType::Float64,
        ] {
            if value == test_value as u32 {
                return Ok(test_value);
            }
        }
        Err(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(MetadataArrayValue),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}
impl MetadataValue {
    fn read_key_value(
        ctx: &GgufContext,
        reader: &mut dyn BufRead,
    ) -> Result<(String, Self), LoadError<Infallible>> {
        let key = util::read_string(reader, ctx.use_64_bit_length)?;
        let value_type = MetadataValueType::try_from(util::read_u32(reader)?)
            .expect("TODO: handle invalid value types");
        let value = Self::read_value(ctx, reader, value_type)?;

        Ok((key, value))
    }

    fn read_value(
        ctx: &GgufContext,
        reader: &mut dyn BufRead,
        value_type: MetadataValueType,
    ) -> Result<MetadataValue, LoadError<Infallible>> {
        match value_type {
            MetadataValueType::UInt8 => Self::read_u8(ctx, reader).map(MetadataValue::UInt8),
            MetadataValueType::Int8 => Self::read_i8(ctx, reader).map(MetadataValue::Int8),
            MetadataValueType::UInt16 => Self::read_u16(ctx, reader).map(MetadataValue::UInt16),
            MetadataValueType::Int16 => Self::read_i16(ctx, reader).map(MetadataValue::Int16),
            MetadataValueType::UInt32 => Self::read_u32(ctx, reader).map(MetadataValue::UInt32),
            MetadataValueType::Int32 => Self::read_i32(ctx, reader).map(MetadataValue::Int32),
            MetadataValueType::Float32 => Self::read_f32(ctx, reader).map(MetadataValue::Float32),
            MetadataValueType::Bool => Self::read_bool(ctx, reader).map(MetadataValue::Bool),
            MetadataValueType::String => Self::read_string(ctx, reader).map(MetadataValue::String),
            MetadataValueType::Array => Self::read_array(ctx, reader).map(MetadataValue::Array),
            MetadataValueType::UInt64 => Self::read_u64(ctx, reader).map(MetadataValue::UInt64),
            MetadataValueType::Int64 => Self::read_i64(ctx, reader).map(MetadataValue::Int64),
            MetadataValueType::Float64 => Self::read_f64(ctx, reader).map(MetadataValue::Float64),
        }
    }

    fn read_u8(_ctx: &GgufContext, reader: &mut dyn BufRead) -> Result<u8, LoadError<Infallible>> {
        Ok(util::read_bytes::<1>(reader)?[0])
    }

    fn read_i8(_ctx: &GgufContext, reader: &mut dyn BufRead) -> Result<i8, LoadError<Infallible>> {
        Ok(util::read_bytes::<1>(reader)?[0] as i8)
    }

    fn read_u16(
        _ctx: &GgufContext,
        reader: &mut dyn BufRead,
    ) -> Result<u16, LoadError<Infallible>> {
        Ok(u16::from_le_bytes(util::read_bytes::<2>(reader)?))
    }

    fn read_i16(
        _ctx: &GgufContext,
        reader: &mut dyn BufRead,
    ) -> Result<i16, LoadError<Infallible>> {
        Ok(i16::from_le_bytes(util::read_bytes::<2>(reader)?))
    }

    fn read_u32(
        _ctx: &GgufContext,
        reader: &mut dyn BufRead,
    ) -> Result<u32, LoadError<Infallible>> {
        Ok(util::read_u32(reader)?)
    }

    fn read_i32(
        _ctx: &GgufContext,
        reader: &mut dyn BufRead,
    ) -> Result<i32, LoadError<Infallible>> {
        Ok(util::read_i32(reader)?)
    }

    fn read_f32(
        _ctx: &GgufContext,
        reader: &mut dyn BufRead,
    ) -> Result<f32, LoadError<Infallible>> {
        Ok(util::read_f32(reader)?)
    }

    fn read_bool(
        _ctx: &GgufContext,
        reader: &mut dyn BufRead,
    ) -> Result<bool, LoadError<Infallible>> {
        Ok(util::read_bool(reader)?)
    }

    fn read_string(
        ctx: &GgufContext,
        reader: &mut dyn BufRead,
    ) -> Result<String, LoadError<Infallible>> {
        Ok(util::read_string(reader, ctx.use_64_bit_length)?)
    }

    fn read_array(
        ctx: &GgufContext,
        reader: &mut dyn BufRead,
    ) -> Result<MetadataArrayValue, LoadError<Infallible>> {
        MetadataArrayValue::read_value(ctx, reader)
    }

    fn read_u64(
        _ctx: &GgufContext,
        reader: &mut dyn BufRead,
    ) -> Result<u64, LoadError<Infallible>> {
        Ok(util::read_u64(reader)?)
    }

    fn read_i64(
        _ctx: &GgufContext,
        reader: &mut dyn BufRead,
    ) -> Result<i64, LoadError<Infallible>> {
        Ok(util::read_i64(reader)?)
    }

    fn read_f64(
        _ctx: &GgufContext,
        reader: &mut dyn BufRead,
    ) -> Result<f64, LoadError<Infallible>> {
        Ok(util::read_f64(reader)?)
    }

    pub fn as_uint32(&self) -> Option<u32> {
        match self {
            Self::UInt32(v) => Some(*v),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetadataArrayValue {
    UInt8(Vec<u8>),
    Int8(Vec<i8>),
    UInt16(Vec<u16>),
    Int16(Vec<i16>),
    UInt32(Vec<u32>),
    Int32(Vec<i32>),
    Float32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    Array(Vec<MetadataArrayValue>),
    UInt64(Vec<u64>),
    Int64(Vec<i64>),
    Float64(Vec<f64>),
}
impl MetadataArrayValue {
    fn read_value(
        ctx: &GgufContext,
        reader: &mut dyn BufRead,
    ) -> Result<Self, LoadError<Infallible>> {
        let value_type = MetadataValueType::try_from(util::read_u32(reader)?)
            .expect("TODO: handle invalid value types");
        let length = util::read_length(reader, ctx.use_64_bit_length)?;

        struct ArrayReader<'a> {
            ctx: &'a GgufContext,
            reader: &'a mut dyn BufRead,
            length: usize,
        }
        impl ArrayReader<'_> {
            fn read<T>(
                &mut self,
                value_reader: impl Fn(
                    &GgufContext,
                    &mut dyn BufRead,
                ) -> Result<T, LoadError<Infallible>>,
                value_constructor: impl Fn(Vec<T>) -> MetadataArrayValue,
            ) -> Result<MetadataArrayValue, LoadError<Infallible>> {
                (0..self.length)
                    .map(|_| value_reader(self.ctx, self.reader))
                    .collect::<Result<Vec<T>, _>>()
                    .map(value_constructor)
            }
        }

        let mut reader = ArrayReader {
            ctx,
            reader,
            length,
        };
        use MetadataValue as MV;
        use MetadataValueType as MVT;
        Ok(match value_type {
            MVT::UInt8 => reader.read(MV::read_u8, Self::UInt8),
            MVT::Int8 => reader.read(MV::read_i8, Self::Int8),
            MVT::UInt16 => reader.read(MV::read_u16, Self::UInt16),
            MVT::Int16 => reader.read(MV::read_i16, Self::Int16),
            MVT::UInt32 => reader.read(MV::read_u32, Self::UInt32),
            MVT::Int32 => reader.read(MV::read_i32, Self::Int32),
            MVT::Float32 => reader.read(MV::read_f32, Self::Float32),
            MVT::Bool => reader.read(MV::read_bool, Self::Bool),
            MVT::String => reader.read(MV::read_string, Self::String),
            MVT::Array => reader.read(MV::read_array, Self::Array),
            MVT::UInt64 => reader.read(MV::read_u64, Self::UInt64),
            MVT::Int64 => reader.read(MV::read_i64, Self::Int64),
            MVT::Float64 => reader.read(MV::read_f64, Self::Float64),
        }?)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub dimensions: Vec<usize>,
    pub element_type: ElementType,
    pub offset: u64,
}
impl TensorInfo {
    fn read_name_value(
        ctx: &GgufContext,
        reader: &mut dyn BufRead,
    ) -> Result<(String, Self), LoadError<Infallible>> {
        let name = util::read_string(reader, ctx.use_64_bit_length)?;

        let dimension_count = util::read_u32(reader)? as usize;
        let dimensions = (0..dimension_count)
            .map(|_| util::read_length(reader, ctx.use_64_bit_length))
            .collect::<Result<Vec<_>, _>>()?;

        let element_type = util::read_u32(reader)?;
        let element_type =
            ElementType::try_from(element_type).map_err(|_| LoadError::UnsupportedElementType {
                tensor_name: name.clone(),
                ftype: element_type,
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
}
