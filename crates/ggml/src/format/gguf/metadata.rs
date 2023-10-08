use std::{collections::HashMap, io::BufRead};

use thiserror::Error;

use crate::util;

use super::{GgufContext, GgufLoadError};

// TODO: make this a newtype instead
pub type Metadata = HashMap<String, MetadataValue>;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetadataValueType {
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
pub trait MetadataValueTypeFromRustType {
    fn value_type() -> MetadataValueType;
}
macro_rules! impl_value_boilerplate {
    ($($value_type:ident($rust_type:ty)),*) => {
        $(
            impl MetadataValueTypeFromRustType for $rust_type {
                fn value_type() -> MetadataValueType {
                    MetadataValueType::$value_type
                }
            }
        )*


        impl TryFrom<u32> for MetadataValueType {
            type Error = ();

            fn try_from(value: u32) -> Result<Self, Self::Error> {
                for test_value in [
                    $(MetadataValueType::$value_type),*
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
            $(
                $value_type($rust_type),
            )*
        }

        // Public
        impl MetadataValue {
            pub fn value_type(&self) -> MetadataValueType {
                match self {
                    $(MetadataValue::$value_type(_) => MetadataValueType::$value_type),*
                }
            }
        }
    };
}
impl_value_boilerplate! {
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
    Float64(f64)
}

// Public
impl MetadataValue {
    pub fn as_uint8(&self) -> Option<u8> {
        match self {
            Self::UInt8(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_int8(&self) -> Option<i8> {
        match self {
            Self::Int8(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_uint16(&self) -> Option<u16> {
        match self {
            Self::UInt16(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_int16(&self) -> Option<i16> {
        match self {
            Self::Int16(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_uint32(&self) -> Option<u32> {
        match self {
            Self::UInt32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_int32(&self) -> Option<i32> {
        match self {
            Self::Int32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_float32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&MetadataArrayValue> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_uint64(&self) -> Option<u64> {
        match self {
            Self::UInt64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_int64(&self) -> Option<i64> {
        match self {
            Self::Int64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_float64(&self) -> Option<f64> {
        match self {
            Self::Float64(v) => Some(*v),
            _ => None,
        }
    }
}
// Private
impl MetadataValue {
    pub(super) fn read_key_value(
        ctx: &GgufContext,
        reader: &mut dyn BufRead,
    ) -> Result<(String, Self), GgufLoadError> {
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
    ) -> Result<MetadataValue, GgufLoadError> {
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

    fn read_u8(_ctx: &GgufContext, reader: &mut dyn BufRead) -> Result<u8, GgufLoadError> {
        Ok(util::read_bytes::<1>(reader)?[0])
    }

    fn read_i8(_ctx: &GgufContext, reader: &mut dyn BufRead) -> Result<i8, GgufLoadError> {
        Ok(util::read_bytes::<1>(reader)?[0] as i8)
    }

    fn read_u16(_ctx: &GgufContext, reader: &mut dyn BufRead) -> Result<u16, GgufLoadError> {
        Ok(u16::from_le_bytes(util::read_bytes::<2>(reader)?))
    }

    fn read_i16(_ctx: &GgufContext, reader: &mut dyn BufRead) -> Result<i16, GgufLoadError> {
        Ok(i16::from_le_bytes(util::read_bytes::<2>(reader)?))
    }

    fn read_u32(_ctx: &GgufContext, reader: &mut dyn BufRead) -> Result<u32, GgufLoadError> {
        Ok(util::read_u32(reader)?)
    }

    fn read_i32(_ctx: &GgufContext, reader: &mut dyn BufRead) -> Result<i32, GgufLoadError> {
        Ok(util::read_i32(reader)?)
    }

    fn read_f32(_ctx: &GgufContext, reader: &mut dyn BufRead) -> Result<f32, GgufLoadError> {
        Ok(util::read_f32(reader)?)
    }

    fn read_bool(_ctx: &GgufContext, reader: &mut dyn BufRead) -> Result<bool, GgufLoadError> {
        Ok(util::read_bool(reader)?)
    }

    fn read_string(ctx: &GgufContext, reader: &mut dyn BufRead) -> Result<String, GgufLoadError> {
        Ok(util::read_string(reader, ctx.use_64_bit_length)?)
    }

    fn read_array(
        ctx: &GgufContext,
        reader: &mut dyn BufRead,
    ) -> Result<MetadataArrayValue, GgufLoadError> {
        MetadataArrayValue::read_value(ctx, reader)
    }

    fn read_u64(_ctx: &GgufContext, reader: &mut dyn BufRead) -> Result<u64, GgufLoadError> {
        Ok(util::read_u64(reader)?)
    }

    fn read_i64(_ctx: &GgufContext, reader: &mut dyn BufRead) -> Result<i64, GgufLoadError> {
        Ok(util::read_i64(reader)?)
    }

    fn read_f64(_ctx: &GgufContext, reader: &mut dyn BufRead) -> Result<f64, GgufLoadError> {
        Ok(util::read_f64(reader)?)
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
// Public
impl MetadataArrayValue {
    pub fn as_uint8_array(&self) -> Option<&[u8]> {
        match self {
            Self::UInt8(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_int8_array(&self) -> Option<&[i8]> {
        match self {
            Self::Int8(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_uint16_array(&self) -> Option<&[u16]> {
        match self {
            Self::UInt16(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_int16_array(&self) -> Option<&[i16]> {
        match self {
            Self::Int16(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_uint32_array(&self) -> Option<&[u32]> {
        match self {
            Self::UInt32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_int32_array(&self) -> Option<&[i32]> {
        match self {
            Self::Int32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_float32_array(&self) -> Option<&[f32]> {
        match self {
            Self::Float32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_bool_array(&self) -> Option<&[bool]> {
        match self {
            Self::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_string_array(&self) -> Option<&[String]> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_array_array(&self) -> Option<&[MetadataArrayValue]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_uint64_array(&self) -> Option<&[u64]> {
        match self {
            Self::UInt64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_int64_array(&self) -> Option<&[i64]> {
        match self {
            Self::Int64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_float64_array(&self) -> Option<&[f64]> {
        match self {
            Self::Float64(v) => Some(v),
            _ => None,
        }
    }
}
impl MetadataArrayValue {
    fn read_value(ctx: &GgufContext, reader: &mut dyn BufRead) -> Result<Self, GgufLoadError> {
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
                value_reader: impl Fn(&GgufContext, &mut dyn BufRead) -> Result<T, GgufLoadError>,
                value_constructor: impl Fn(Vec<T>) -> MetadataArrayValue,
            ) -> Result<MetadataArrayValue, GgufLoadError> {
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
        match value_type {
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
        }
    }

    /// Returns the length of the array.
    pub fn len(&self) -> usize {
        match self {
            Self::UInt8(v) => v.len(),
            Self::Int8(v) => v.len(),
            Self::UInt16(v) => v.len(),
            Self::Int16(v) => v.len(),
            Self::UInt32(v) => v.len(),
            Self::Int32(v) => v.len(),
            Self::Float32(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::String(v) => v.len(),
            Self::Array(v) => v.len(),
            Self::UInt64(v) => v.len(),
            Self::Int64(v) => v.len(),
            Self::Float64(v) => v.len(),
        }
    }
}

#[doc(hidden)]
pub trait MetadataExt {
    fn fallible_get(&self, key: &str) -> Result<&MetadataValue, MetadataError>;
    fn fallible_typed_get<'a, T: MetadataValueTypeFromRustType>(
        &'a self,
        key: &'a str,
        getter: impl Fn(&MetadataValue) -> Option<&T>,
    ) -> Result<&'a T, MetadataError>;
    fn fallible_get_string(&self, key: &str) -> Result<String, MetadataError>;
    fn fallible_get_countable(&self, key: &str) -> Result<usize, MetadataError>;
}
impl MetadataExt for Metadata {
    fn fallible_get(&self, key: &str) -> Result<&MetadataValue, MetadataError> {
        self.get(key).ok_or_else(|| MetadataError::MissingKey {
            key: key.to_owned(),
        })
    }

    fn fallible_typed_get<'a, T: MetadataValueTypeFromRustType>(
        &'a self,
        key: &'a str,
        getter: impl Fn(&MetadataValue) -> Option<&T>,
    ) -> Result<&'a T, MetadataError> {
        let metadata_value = self.fallible_get(key)?;
        getter(metadata_value).ok_or_else(|| MetadataError::InvalidType {
            key: key.to_string(),
            expected_type: T::value_type(),
            actual_type: metadata_value.value_type(),
        })
    }

    // TODO: see if we can generalize this with `ToOwned` or something?
    fn fallible_get_string(&self, key: &str) -> Result<String, MetadataError> {
        let metadata_value = self.fallible_get(key)?;
        Ok(metadata_value
            .as_string()
            .ok_or_else(|| MetadataError::InvalidType {
                key: key.to_string(),
                expected_type: MetadataValueType::String,
                actual_type: metadata_value.value_type(),
            })?
            .to_string())
    }

    fn fallible_get_countable(&self, key: &str) -> Result<usize, MetadataError> {
        let metadata_value = self.fallible_get(key)?;
        match metadata_value {
            MetadataValue::UInt32(v) => Ok(usize::try_from(*v)?),
            MetadataValue::UInt64(v) => Ok(usize::try_from(*v)?),
            _ => Err(MetadataError::InvalidType {
                key: key.to_string(),
                expected_type: MetadataValueType::UInt64,
                actual_type: metadata_value.value_type(),
            }),
        }
    }
}

#[derive(Error, Debug)]
/// Errors encountered during the loading process.
pub enum MetadataError {
    /// The model expected a metadata key-value pair, but the key was missing.
    #[error("missing metadata key {key:?}")]
    MissingKey {
        /// The key that was missing.
        key: String,
    },
    /// The metadata key-value pair was not of the expected type.
    #[error("metadata key {key:?} was not of the expected type")]
    InvalidType {
        /// The key with the invalid type.
        key: String,
        /// The expected type.
        expected_type: MetadataValueType,
        /// The actual type.
        actual_type: MetadataValueType,
    },
    #[error("invalid integer conversion")]
    /// One of the integers encountered could not be converted to a more appropriate type.
    InvalidIntegerConversion(#[from] std::num::TryFromIntError),
}
