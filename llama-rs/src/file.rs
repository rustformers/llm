use crate::LoadError;
pub use std::fs::File;
pub use std::io::{BufRead, BufReader, BufWriter, Read, Write};

pub fn read_bytes<const N: usize>(reader: &mut impl BufRead) -> Result<[u8; N], LoadError> {
    let mut bytes = [0u8; N];
    read(reader, &mut bytes)?;
    Ok(bytes)
}

pub fn read_bytes_with_len(reader: &mut impl BufRead, len: usize) -> Result<Vec<u8>, LoadError> {
    let mut bytes = vec![0u8; len];
    read(reader, &mut bytes)?;
    Ok(bytes)
}

pub fn read_i32(reader: &mut impl BufRead) -> Result<i32, LoadError> {
    Ok(i32::from_le_bytes(read_bytes::<4>(reader)?))
}

pub fn rw_i32(reader: &mut impl BufRead, writer: &mut impl Write) -> Result<i32, LoadError> {
    Ok(i32::from_le_bytes(rw::<4>(reader, writer)?))
}

pub fn read_u32(reader: &mut impl BufRead) -> Result<u32, LoadError> {
    Ok(u32::from_le_bytes(read_bytes::<4>(reader)?))
}

pub fn rw_u32(reader: &mut impl BufRead, writer: &mut impl Write) -> Result<u32, LoadError> {
    Ok(u32::from_le_bytes(rw::<4>(reader, writer)?))
}

pub fn read_f32(reader: &mut impl BufRead) -> Result<f32, LoadError> {
    Ok(f32::from_le_bytes(read_bytes::<4>(reader)?))
}

pub fn rw_f32(reader: &mut impl BufRead, writer: &mut impl Write) -> Result<f32, LoadError> {
    Ok(f32::from_le_bytes(rw::<4>(reader, writer)?))
}

/// Helper function. Reads a string from the buffer and returns it.
pub fn read_string(reader: &mut BufReader<File>, len: usize) -> Result<String, LoadError> {
    Ok(String::from_utf8(read_bytes_with_len(reader, len)?)?)
}

fn read(reader: &mut impl BufRead, bytes: &mut [u8]) -> Result<(), LoadError> {
    reader
        .read_exact(bytes)
        .map_err(|e| LoadError::ReadExactFailed {
            source: e,
            bytes: bytes.len(),
        })
}

pub fn rw_bytes_with_len(
    reader: &mut impl BufRead,
    writer: &mut impl Write,
    len: usize,
) -> Result<Vec<u8>, LoadError> {
    let mut buf = vec![0; len];
    reader
        .read_exact(&mut buf)
        .map_err(|e| LoadError::ReadExactFailed {
            source: e,
            bytes: buf.len(),
        })?;
    writer.write_all(&buf)?;
    Ok(buf)
}

fn rw<const N: usize>(
    reader: &mut impl BufRead,
    writer: &mut impl Write,
) -> Result<[u8; N], LoadError> {
    let mut bytes = [0u8; N];
    read(reader, &mut bytes)?;
    writer.write_all(&bytes)?;
    Ok(bytes)
}
