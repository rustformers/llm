//! Utilities for reading and writing.

use std::{
    fmt,
    io::{self, BufRead, Write},
};

/// Helper struct that wraps the magic number of a file format,
/// so that it can be printed in a human-readable format.
pub struct FileMagic(pub [u8; 4]);
impl fmt::Display for FileMagic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:x?} ({})", self.0, String::from_utf8_lossy(&self.0))
    }
}
impl fmt::Debug for FileMagic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

///
/// READERS
///

/// Read a fixed-size array of bytes from a reader.
pub fn read_bytes<const N: usize>(reader: &mut dyn BufRead) -> io::Result<[u8; N]> {
    let mut bytes = [0u8; N];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

/// Read a `i8` from a reader.
pub fn read_i8(reader: &mut dyn BufRead) -> io::Result<i8> {
    Ok(i8::from_le_bytes(read_bytes::<1>(reader)?))
}

/// Read a `u8` from a reader.
pub fn read_u8(reader: &mut dyn BufRead) -> io::Result<u8> {
    Ok(u8::from_le_bytes(read_bytes::<1>(reader)?))
}

/// Read a `i16` from a reader.
pub fn read_i16(reader: &mut dyn BufRead) -> io::Result<i16> {
    Ok(i16::from_le_bytes(read_bytes::<2>(reader)?))
}

/// Read a `u16` from a reader.
pub fn read_u16(reader: &mut dyn BufRead) -> io::Result<u16> {
    Ok(u16::from_le_bytes(read_bytes::<2>(reader)?))
}

/// Read a `i32` from a reader.
pub fn read_i32(reader: &mut dyn BufRead) -> io::Result<i32> {
    Ok(i32::from_le_bytes(read_bytes::<4>(reader)?))
}

/// Read a `u32` from a reader.
pub fn read_u32(reader: &mut dyn BufRead) -> io::Result<u32> {
    Ok(u32::from_le_bytes(read_bytes::<4>(reader)?))
}

/// Read a `i64` from a reader.
pub fn read_i64(reader: &mut dyn BufRead) -> io::Result<i64> {
    Ok(i64::from_le_bytes(read_bytes::<8>(reader)?))
}

/// Read a `u64` from a reader.
pub fn read_u64(reader: &mut dyn BufRead) -> io::Result<u64> {
    Ok(u64::from_le_bytes(read_bytes::<8>(reader)?))
}

/// Read a `f32` from a reader.
pub fn read_f32(reader: &mut dyn BufRead) -> io::Result<f32> {
    Ok(f32::from_le_bytes(read_bytes::<4>(reader)?))
}

/// Read a `f64` from a reader.
pub fn read_f64(reader: &mut dyn BufRead) -> io::Result<f64> {
    Ok(f64::from_le_bytes(read_bytes::<8>(reader)?))
}

/// Read an integer (32-bit or 64-bit) from a reader, and convert it to a usize.
pub fn read_length(reader: &mut dyn BufRead, use_64_bit_length: bool) -> io::Result<usize> {
    let len: usize = if use_64_bit_length {
        read_u64(reader)?.try_into()
    } else {
        read_u32(reader)?.try_into()
    }
    .expect("TODO: invalid usize conversion");
    Ok(len)
}

/// Read a `bool` represented as an `i32` from a reader.
pub fn read_bool(reader: &mut dyn BufRead) -> io::Result<bool> {
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
pub fn read_bytes_with_len(reader: &mut dyn BufRead, len: usize) -> io::Result<Vec<u8>> {
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

/// Read a string from a reader.
pub fn read_string(reader: &mut dyn BufRead, use_64_bit_length: bool) -> io::Result<String> {
    let len = read_length(reader, use_64_bit_length)?;
    let mut bytes = read_bytes_with_len(reader, len)?;
    // The GGUF C writer prior to `llama.cpp@103cfafc774f6feb3172b5d4d39681c965b17eba`
    // wrote a null terminator at the end of strings. As a work-around, we remove
    // them here.
    if bytes.last() == Some(&0) {
        // Remove the null terminator.
        bytes.pop();
    }
    Ok(String::from_utf8(bytes)
        .expect("string was not valid utf-8 (TODO: make this a library error)"))
}

///
/// WRITERS
///

/// Write a `i8` from a writer.
pub fn write_i8(writer: &mut dyn Write, value: i8) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

/// Write a `u8` from a writer.
pub fn write_u8(writer: &mut dyn Write, value: u8) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

/// Write a `i16` from a writer.
pub fn write_i16(writer: &mut dyn Write, value: i16) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

/// Write a `u16` from a writer.
pub fn write_u16(writer: &mut dyn Write, value: u16) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

/// Write a `i32` from a writer.
pub fn write_i32(writer: &mut dyn Write, value: i32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

/// Write a `u32` from a writer.
pub fn write_u32(writer: &mut dyn Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

/// Write a `i64` from a writer.
pub fn write_i64(writer: &mut dyn Write, value: i64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

/// Write a `u64` from a writer.
pub fn write_u64(writer: &mut dyn Write, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

/// Write a `f32` from a writer.
pub fn write_f32(writer: &mut dyn Write, value: f32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

/// Write a `f64` from a writer.
pub fn write_f64(writer: &mut dyn Write, value: f64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

/// Write a `bool` represented as an `i32` to a writer.
pub fn write_bool(writer: &mut dyn Write, value: bool) -> io::Result<()> {
    let int_value: i32 = if value { 1 } else { 0 };
    writer.write_all(&int_value.to_le_bytes())
}

/// Write an integer (32-bit or 64-bit) to a writer, and convert it from a usize.
pub fn write_length(writer: &mut dyn Write, use_64_bit_length: bool, len: usize) -> io::Result<()> {
    if use_64_bit_length {
        write_u64(writer, len as u64)
    } else {
        write_u32(writer, len as u32)
    }
}

/// Read a string from a reader.
pub fn write_string(
    writer: &mut dyn Write,
    use_64_bit_length: bool,
    value: &str,
) -> io::Result<()> {
    write_length(writer, use_64_bit_length, value.len())?;
    writer.write_all(value.as_bytes())
}

/// Write N zero bytes to a writer.
// TODO: is there a more efficient way to do this?
pub fn write_zero_bytes(writer: &mut dyn Write, n: usize) -> io::Result<()> {
    for _ in 0..n {
        writer.write_all(&[0u8])?;
    }
    Ok(())
}

// NOTE: Implementation from #![feature(buf_read_has_data_left)]
/// Check if there is any data left in the reader.
pub fn has_data_left(reader: &mut impl BufRead) -> io::Result<bool> {
    reader.fill_buf().map(|b| !b.is_empty())
}
