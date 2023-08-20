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
        if container != ContainerType::Gguf(1) {
            return Err(LoadError::InvalidFormatVersion(container));
        }

        let tensor_count = util::read_u32(reader)? as usize;
        let metadata_kv_count = util::read_u32(reader)? as usize;

        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let (key, value) = MetadataValue::read_key_value(reader)?;
            metadata.insert(key, value);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_uint32())
            .unwrap_or(DEFAULT_ALIGNMENT) as u64;

        let mut tensor_infos = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let (key, value) = TensorInfo::read_name_value(reader)?;
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

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MetadataValueType {
    // The value is a 8-bit unsigned integer.
    UInt8 = 0,
    // The value is a 8-bit signed integer.
    Int8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    UInt16 = 2,
    // The value is a 16-bit signed little-endian integer.
    Int16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    UInt32 = 4,
    // The value is a 32-bit signed little-endian integer.
    Int32 = 5,
    // The value is a 32-bit IEEE754 floating point number.
    Float32 = 6,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    Bool = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    String = 8,
    // The value is an array of other values, with the length and type prepended.
    ///
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    Array = 9,
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
}
impl MetadataValue {
    fn read_key_value(reader: &mut dyn BufRead) -> Result<(String, Self), LoadError<Infallible>> {
        let key = util::read_string(reader)?;
        let value_type = MetadataValueType::try_from(util::read_u32(reader)?)
            .expect("TODO: handle invalid value types");

        let value = Self::read_value(reader, value_type)?;

        Ok((key, value))
    }

    fn read_value(
        reader: &mut dyn BufRead,
        value_type: MetadataValueType,
    ) -> Result<MetadataValue, LoadError<Infallible>> {
        match value_type {
            MetadataValueType::UInt8 => Self::read_u8(reader).map(MetadataValue::UInt8),
            MetadataValueType::Int8 => Self::read_i8(reader).map(MetadataValue::Int8),
            MetadataValueType::UInt16 => Self::read_u16(reader).map(MetadataValue::UInt16),
            MetadataValueType::Int16 => Self::read_i16(reader).map(MetadataValue::Int16),
            MetadataValueType::UInt32 => Self::read_u32(reader).map(MetadataValue::UInt32),
            MetadataValueType::Int32 => Self::read_i32(reader).map(MetadataValue::Int32),
            MetadataValueType::Float32 => Self::read_f32(reader).map(MetadataValue::Float32),
            MetadataValueType::Bool => Self::read_bool(reader).map(MetadataValue::Bool),
            MetadataValueType::String => Self::read_string(reader).map(MetadataValue::String),
            MetadataValueType::Array => Self::read_array(reader).map(MetadataValue::Array),
        }
    }

    fn read_u8(reader: &mut dyn BufRead) -> Result<u8, LoadError<Infallible>> {
        Ok(util::read_bytes::<1>(reader)?[0])
    }

    fn read_i8(reader: &mut dyn BufRead) -> Result<i8, LoadError<Infallible>> {
        Ok(util::read_bytes::<1>(reader)?[0] as i8)
    }

    fn read_u16(reader: &mut dyn BufRead) -> Result<u16, LoadError<Infallible>> {
        Ok(u16::from_le_bytes(util::read_bytes::<2>(reader)?))
    }

    fn read_i16(reader: &mut dyn BufRead) -> Result<i16, LoadError<Infallible>> {
        Ok(i16::from_le_bytes(util::read_bytes::<2>(reader)?))
    }

    fn read_u32(reader: &mut dyn BufRead) -> Result<u32, LoadError<Infallible>> {
        Ok(util::read_u32(reader)?)
    }

    fn read_i32(reader: &mut dyn BufRead) -> Result<i32, LoadError<Infallible>> {
        Ok(util::read_i32(reader)?)
    }

    fn read_f32(reader: &mut dyn BufRead) -> Result<f32, LoadError<Infallible>> {
        Ok(util::read_f32(reader)?)
    }

    fn read_bool(reader: &mut dyn BufRead) -> Result<bool, LoadError<Infallible>> {
        let v = Self::read_u8(reader)?;
        if v == 0 {
            Ok(false)
        } else if v == 1 {
            Ok(true)
        } else {
            panic!("TODO: error for invalid bools")
        }
    }

    fn read_string(reader: &mut dyn BufRead) -> Result<String, LoadError<Infallible>> {
        Ok(util::read_string(reader)?)
    }

    fn read_array(reader: &mut dyn BufRead) -> Result<MetadataArrayValue, LoadError<Infallible>> {
        MetadataArrayValue::read_value(reader)
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
}
impl MetadataArrayValue {
    fn read_value(reader: &mut dyn BufRead) -> Result<Self, LoadError<Infallible>> {
        let value_type = MetadataValueType::try_from(util::read_u32(reader)?)
            .expect("TODO: handle invalid value types");
        let length = util::read_u32(reader)? as usize;

        fn read_array<T>(
            reader: &mut dyn BufRead,
            length: usize,
            value_reader: impl Fn(&mut dyn BufRead) -> Result<T, LoadError<Infallible>>,
        ) -> Result<Vec<T>, LoadError<Infallible>> {
            (0..length).map(|_| value_reader(reader)).collect()
        }

        use MetadataValue as MV;
        use MetadataValueType as MVT;
        Ok(match value_type {
            MVT::UInt8 => read_array(reader, length, MV::read_u8).map(Self::UInt8),
            MVT::Int8 => read_array(reader, length, MV::read_i8).map(Self::Int8),
            MVT::UInt16 => read_array(reader, length, MV::read_u16).map(Self::UInt16),
            MVT::Int16 => read_array(reader, length, MV::read_i16).map(Self::Int16),
            MVT::UInt32 => read_array(reader, length, MV::read_u32).map(Self::UInt32),
            MVT::Int32 => read_array(reader, length, MV::read_i32).map(Self::Int32),
            MVT::Float32 => read_array(reader, length, MV::read_f32).map(Self::Float32),
            MVT::Bool => read_array(reader, length, MV::read_bool).map(Self::Bool),
            MVT::String => read_array(reader, length, MV::read_string).map(Self::String),
            MVT::Array => read_array(reader, length, MV::read_array).map(Self::Array),
        }?)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub dimensions: Vec<u32>,
    pub element_type: ElementType,
    pub offset: u64,
}
impl TensorInfo {
    fn read_name_value(reader: &mut dyn BufRead) -> Result<(String, Self), LoadError<Infallible>> {
        let name = util::read_string(reader)?;

        let dimension_count = util::read_u32(reader)? as usize;
        let dimensions = (0..dimension_count)
            .map(|_| util::read_u32(reader))
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
