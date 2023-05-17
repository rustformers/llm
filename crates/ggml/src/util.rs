//! Utilities for reading and writing.

use std::io::{BufRead, Write};

/// Read a fixed-size array of bytes from a reader.
pub fn read_bytes<const N: usize>(reader: &mut dyn BufRead) -> Result<[u8; N], std::io::Error> {
    let mut bytes = [0u8; N];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

/// Read a `i32` from a reader.
pub fn read_i32(reader: &mut dyn BufRead) -> Result<i32, std::io::Error> {
    Ok(i32::from_le_bytes(read_bytes::<4>(reader)?))
}

/// Read a `u32` from a reader.
pub fn read_u32(reader: &mut dyn BufRead) -> Result<u32, std::io::Error> {
    Ok(u32::from_le_bytes(read_bytes::<4>(reader)?))
}

/// Read a `f32` from a reader.
pub fn read_f32(reader: &mut dyn BufRead) -> Result<f32, std::io::Error> {
    Ok(f32::from_le_bytes(read_bytes::<4>(reader)?))
}

/// Read a `bool` represented as an `i32` from a reader.
pub fn read_bool(reader: &mut dyn BufRead) -> Result<bool, std::io::Error> {
    let val = i32::from_le_bytes(read_bytes::<4>(reader)?);
    match val {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Invalid i32 value for bool: '{}'", val),
        )),
    }
}

/// Read a variable-length array of bytes from a reader.
pub fn read_bytes_with_len(
    reader: &mut dyn BufRead,
    len: usize,
) -> Result<Vec<u8>, std::io::Error> {
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

/// Write a `i32` from a writer.
pub fn write_i32(writer: &mut dyn Write, value: i32) -> Result<(), std::io::Error> {
    writer.write_all(&value.to_le_bytes())
}

/// Write a `u32` from a writer.
pub fn write_u32(writer: &mut dyn Write, value: u32) -> Result<(), std::io::Error> {
    writer.write_all(&value.to_le_bytes())
}

/// Write a `f32` from a writer.
pub fn write_f32(writer: &mut dyn Write, value: f32) -> Result<(), std::io::Error> {
    writer.write_all(&value.to_le_bytes())
}

/// Write a `bool` represented as an `i32` to a writer.
pub fn write_bool(writer: &mut dyn Write, value: bool) -> Result<(), std::io::Error> {
    let int_value: i32 = if value { 1 } else { 0 };
    writer.write_all(&int_value.to_le_bytes())
}

// NOTE: Implementation from #![feature(buf_read_has_data_left)]
/// Check if there is any data left in the reader.
pub fn has_data_left(reader: &mut impl BufRead) -> Result<bool, std::io::Error> {
    reader.fill_buf().map(|b| !b.is_empty())
}
