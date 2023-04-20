pub use std::io::{BufRead, Seek, SeekFrom};
use std::ops::ControlFlow;

use crate::{ElementType, LoadError};

pub fn read_bytes<const N: usize>(reader: &mut impl BufRead) -> Result<[u8; N], std::io::Error> {
    let mut bytes = [0u8; N];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

pub fn read_i32(reader: &mut impl BufRead) -> Result<i32, std::io::Error> {
    Ok(i32::from_le_bytes(read_bytes::<4>(reader)?))
}

pub fn read_u32(reader: &mut impl BufRead) -> Result<u32, std::io::Error> {
    Ok(u32::from_le_bytes(read_bytes::<4>(reader)?))
}

pub fn read_f32(reader: &mut impl BufRead) -> Result<f32, std::io::Error> {
    Ok(f32::from_le_bytes(read_bytes::<4>(reader)?))
}

pub fn read_bytes_with_len(
    reader: &mut impl BufRead,
    len: usize,
) -> Result<Vec<u8>, std::io::Error> {
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

// NOTE: Implementation from #![feature(buf_read_has_data_left)]
pub fn has_data_left(reader: &mut impl BufRead) -> Result<bool, std::io::Error> {
    reader.fill_buf().map(|b| !b.is_empty())
}

pub fn decode_element_type(ftype: i32) -> Option<ElementType> {
    match ftype {
        0 => Some(ggml::Type::F32),
        1 => Some(ggml::Type::F16),
        2 => Some(ggml::Type::Q4_0),
        3 => Some(ggml::Type::Q4_1),
        _ => None,
    }
}

pub fn encode_element_type(element_type: ElementType) -> Option<i32> {
    match element_type {
        ggml::Type::F32 => Some(0),
        ggml::Type::F16 => Some(1),
        ggml::Type::Q4_0 => Some(2),
        ggml::Type::Q4_1 => Some(3),
        _ => None,
    }
}

pub fn decode_element_type_res<T>(ftype: i32) -> Result<ElementType, LoadError<T>> {
    match decode_element_type(ftype) {
        Some(x) => Ok(x),
        None => Err(LoadError::UnsupportedElementType(ftype)),
    }
}

pub fn retchk<A, B>(x: ControlFlow<A, B>) -> Result<B, LoadError<A>> {
    match x {
        ControlFlow::Continue(x) => Ok(x),
        ControlFlow::Break(y) => Err(LoadError::UserInterrupted(y)),
    }
}

pub fn brkchk<A, B, C: Into<A>>(x: Result<B, C>) -> ControlFlow<A, B> {
    match x {
        Ok(x) => ControlFlow::Continue(x),
        Err(y) => ControlFlow::Break(y.into()),
    }
}
