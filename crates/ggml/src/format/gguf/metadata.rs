use std::io::{self, BufRead, Write};

use indexmap::IndexMap;
use thiserror::Error;

use crate::util;

use super::{GgufContext, GgufLoadError};

#[derive(Debug, Clone, PartialEq)]
pub struct Metadata(pub IndexMap<String, MetadataValue>);
impl Metadata {
    pub fn iter(&self) -> impl Iterator<Item = (&String, &MetadataValue)> {
        self.0.iter()
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.0.keys()
    }

    pub fn values(&self) -> impl Iterator<Item = &MetadataValue> {
        self.0.values()
    }

    pub fn get_optional(&self, key: &str) -> Option<&MetadataValue> {
        self.0.get(key)
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }

    pub fn get(&self, key: &str) -> Result<&MetadataValue, MetadataError> {
        self.get_optional(key)
            .ok_or_else(|| MetadataError::MissingKey {
                key: key.to_owned(),
            })
    }

    pub fn get_with_type<'a, T: ToMetadataValue>(
        &'a self,
        key: &'a str,
        getter: impl Fn(&MetadataValue) -> Option<T>,
    ) -> Result<T, MetadataError> {
        let metadata_value = self.get(key)?;
        getter(metadata_value).ok_or_else(|| MetadataError::InvalidType {
            key: key.to_string(),
            expected_type: T::value_type(),
            actual_type: metadata_value.value_type(),
        })
    }

    pub fn get_with_ref_type<'a, T: ToMetadataValue>(
        &'a self,
        key: &'a str,
        getter: impl Fn(&MetadataValue) -> Option<&T>,
    ) -> Result<&'a T, MetadataError> {
        let metadata_value = self.get(key)?;
        getter(metadata_value).ok_or_else(|| MetadataError::InvalidType {
            key: key.to_string(),
            expected_type: T::value_type(),
            actual_type: metadata_value.value_type(),
        })
    }

    pub fn get_array_with_type<'a, T: ToMetadataValue>(
        &'a self,
        key: &'a str,
        getter: impl Fn(&MetadataValue) -> Option<&[T]>,
    ) -> Result<&'a [T], MetadataError> {
        let metadata_value = self.get(key)?;
        getter(metadata_value).ok_or_else(|| MetadataError::InvalidType {
            key: key.to_string(),
            expected_type: T::value_type(),
            actual_type: metadata_value.value_type(),
        })
    }

    // TODO: consider
    pub fn get_str(&self, key: &str) -> Result<&str, MetadataError> {
        let metadata_value = self.get(key)?;
        Ok(metadata_value
            .as_string()
            .ok_or_else(|| MetadataError::InvalidType {
                key: key.to_string(),
                expected_type: MetadataValueType::String,
                actual_type: metadata_value.value_type(),
            })?)
    }

    pub fn get_countable(&self, key: &str) -> Result<usize, MetadataError> {
        let metadata_value = self.get(key)?;
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
pub trait ToMetadataValue {
    fn value_type() -> MetadataValueType;
    fn to_value(self) -> MetadataValue;
}
pub trait ToMetadataArrayValue {
    fn to_array_value(self) -> MetadataArrayValue;
}
macro_rules! impl_value_boilerplate {
    ($($value_type:ident($rust_type:ty)),*) => {
        $(
            impl ToMetadataValue for $rust_type {
                fn value_type() -> MetadataValueType {
                    MetadataValueType::$value_type
                }

                fn to_value(self) -> MetadataValue {
                    MetadataValue::$value_type(self)
                }
            }

            impl ToMetadataArrayValue for Vec<$rust_type> {
                fn to_array_value(self) -> MetadataArrayValue {
                    MetadataArrayValue::$value_type(self)
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
        impl MetadataValueType {
            fn read_value(
                self,
                ctx: &GgufContext,
                reader: &mut dyn BufRead,
            ) -> Result<MetadataValue, GgufLoadError> {
                use MetadataValueType as MVT;

                Ok(match self {
                    $(MVT::$value_type => <$rust_type>::read(ctx, reader)?.to_value(),)*
                })
            }
        }

        #[derive(Debug, Clone, PartialEq)]
        pub enum MetadataValue {
            $(
                $value_type($rust_type),
            )*
        }
        impl MetadataValue {
            pub fn value_type(&self) -> MetadataValueType {
                match self {
                    $(MetadataValue::$value_type(_) => MetadataValueType::$value_type),*
                }
            }

            fn write(&self, ctx: &GgufContext, writer: &mut dyn Write) -> io::Result<()> {
                match self {
                    $(MetadataValue::$value_type(v) => v.write(ctx, writer)),*
                }
            }
        }

        #[derive(Debug, Clone, PartialEq)]
        pub enum MetadataArrayValue {
            $($value_type(Vec<$rust_type>),)*
        }
        impl MetadataArrayValue {
            /// Returns the length of the array.
            pub fn len(&self) -> usize {
                match self {
                    $(Self::$value_type(v) => v.len(),)*
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
impl MetadataValue {
    pub(super) fn read_key_value(
        ctx: &GgufContext,
        reader: &mut dyn BufRead,
    ) -> Result<(String, Self), GgufLoadError> {
        let key = util::read_string(reader, ctx.use_64_bit_length)?;
        let value_type = MetadataValueType::try_from(util::read_u32(reader)?)
            .expect("TODO: handle invalid value types");
        let value = value_type.read_value(ctx, reader)?;

        Ok((key, value))
    }

    pub(super) fn write_key_value(
        &self,
        ctx: &GgufContext,
        writer: &mut dyn Write,
        key: &str,
    ) -> io::Result<()> {
        util::write_string(writer, ctx.use_64_bit_length, key)?;
        util::write_u32(writer, self.value_type() as u32)?;
        self.write(ctx, writer)?;

        Ok(())
    }
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

// Shared
trait ValueIO {
    fn read(ctx: &GgufContext, reader: &mut dyn BufRead) -> Result<Self, GgufLoadError>
    where
        Self: Sized;
    fn write(&self, ctx: &GgufContext, writer: &mut dyn Write) -> io::Result<()>;
}
macro_rules! impl_value_io_boilerplate {
    ($($value_type:ident($rust_type:ty, $read_method:ident, $write_method:ident)),*) => {
        $(
            impl ValueIO for $rust_type {
                fn read(_ctx: &GgufContext, reader: &mut dyn BufRead) -> Result<Self, GgufLoadError>
                where
                    Self: Sized,
                {
                    Ok(util::$read_method(reader)?)
                }

                fn write(&self, _ctx: &GgufContext, writer: &mut dyn Write) -> io::Result<()> {
                    util::$write_method(writer, *self)
                }
            }
        )*
    };
}
impl_value_io_boilerplate! {
    UInt8(u8, read_u8, write_u8),
    Int8(i8, read_i8, write_i8),
    UInt16(u16, read_u16, write_u16),
    Int16(i16, read_i16, write_i16),
    UInt32(u32, read_u32, write_u32),
    Int32(i32, read_i32, write_i32),
    Float32(f32, read_f32, write_f32),
    Bool(bool, read_bool, write_bool),
    UInt64(u64, read_u64, write_u64),
    Int64(i64, read_i64, write_i64),
    Float64(f64, read_f64, write_f64)
}
impl ValueIO for String {
    fn read(ctx: &GgufContext, reader: &mut dyn BufRead) -> Result<Self, GgufLoadError>
    where
        Self: Sized,
    {
        Ok(util::read_string(reader, ctx.use_64_bit_length)?)
    }

    fn write(&self, ctx: &GgufContext, writer: &mut dyn Write) -> io::Result<()> {
        util::write_string(writer, ctx.use_64_bit_length, self)
    }
}
impl ValueIO for MetadataArrayValue {
    fn read(ctx: &GgufContext, reader: &mut dyn BufRead) -> Result<Self, GgufLoadError>
    where
        Self: Sized,
    {
        let value_type = MetadataValueType::try_from(util::read_u32(reader)?)
            .expect("TODO: handle invalid value types");
        let length = util::read_length(reader, ctx.use_64_bit_length)?;

        use MetadataValueType as MVT;
        return match value_type {
            MVT::UInt8 => read_array::<u8>(ctx, reader, length),
            MVT::Int8 => read_array::<i8>(ctx, reader, length),
            MVT::UInt16 => read_array::<u16>(ctx, reader, length),
            MVT::Int16 => read_array::<i16>(ctx, reader, length),
            MVT::UInt32 => read_array::<u32>(ctx, reader, length),
            MVT::Int32 => read_array::<i32>(ctx, reader, length),
            MVT::Float32 => read_array::<f32>(ctx, reader, length),
            MVT::Bool => read_array::<bool>(ctx, reader, length),
            MVT::String => read_array::<String>(ctx, reader, length),
            MVT::Array => read_array::<MetadataArrayValue>(ctx, reader, length),
            MVT::UInt64 => read_array::<u64>(ctx, reader, length),
            MVT::Int64 => read_array::<i64>(ctx, reader, length),
            MVT::Float64 => read_array::<f64>(ctx, reader, length),
        };

        fn read_array<T: ValueIO>(
            ctx: &GgufContext,
            reader: &mut dyn BufRead,
            length: usize,
        ) -> Result<MetadataArrayValue, GgufLoadError>
        where
            Vec<T>: ToMetadataArrayValue,
        {
            (0..length)
                .map(|_| T::read(ctx, reader))
                .collect::<Result<Vec<T>, _>>()
                .map(|v| v.to_array_value())
        }
    }

    fn write(&self, ctx: &GgufContext, writer: &mut dyn Write) -> io::Result<()> {
        return match self {
            MetadataArrayValue::UInt8(v) => write_array(ctx, writer, v),
            MetadataArrayValue::Int8(v) => write_array(ctx, writer, v),
            MetadataArrayValue::UInt16(v) => write_array(ctx, writer, v),
            MetadataArrayValue::Int16(v) => write_array(ctx, writer, v),
            MetadataArrayValue::UInt32(v) => write_array(ctx, writer, v),
            MetadataArrayValue::Int32(v) => write_array(ctx, writer, v),
            MetadataArrayValue::Float32(v) => write_array(ctx, writer, v),
            MetadataArrayValue::Bool(v) => write_array(ctx, writer, v),
            MetadataArrayValue::String(v) => write_array(ctx, writer, v),
            MetadataArrayValue::Array(v) => write_array(ctx, writer, v),
            MetadataArrayValue::UInt64(v) => write_array(ctx, writer, v),
            MetadataArrayValue::Int64(v) => write_array(ctx, writer, v),
            MetadataArrayValue::Float64(v) => write_array(ctx, writer, v),
        };

        fn write_array<T: ToMetadataValue + ValueIO>(
            ctx: &GgufContext,
            writer: &mut dyn Write,
            array: &[T],
        ) -> io::Result<()> {
            util::write_u32(writer, T::value_type() as u32)?;
            util::write_length(writer, ctx.use_64_bit_length, array.len())?;
            for value in array {
                value.write(ctx, writer)?;
            }
            Ok(())
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
