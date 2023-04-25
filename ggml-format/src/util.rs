pub use std::fs::File;
pub use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::ops::ControlFlow;

use crate::LoadError;

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

pub fn rw_i32(reader: &mut impl BufRead, writer: &mut impl Write) -> Result<i32, std::io::Error> {
    Ok(i32::from_le_bytes(rw::<4>(reader, writer)?))
}

pub fn rw_u32(reader: &mut impl BufRead, writer: &mut impl Write) -> Result<u32, std::io::Error> {
    Ok(u32::from_le_bytes(rw::<4>(reader, writer)?))
}

pub fn rw_f32(reader: &mut impl BufRead, writer: &mut impl Write) -> Result<f32, std::io::Error> {
    Ok(f32::from_le_bytes(rw::<4>(reader, writer)?))
}

pub fn rw_bytes_with_len(
    reader: &mut impl BufRead,
    writer: &mut impl Write,
    len: usize,
) -> Result<Vec<u8>, std::io::Error> {
    let mut buf = vec![0; len];
    reader.read_exact(&mut buf)?;
    writer.write_all(&buf)?;
    Ok(buf)
}

fn rw<const N: usize>(
    reader: &mut impl BufRead,
    writer: &mut impl Write,
) -> Result<[u8; N], std::io::Error> {
    let bytes: [u8; N] = read_bytes(reader)?;
    writer.write_all(&bytes)?;
    Ok(bytes)
}

// NOTE: Implementation from #![feature(buf_read_has_data_left)]
pub fn has_data_left(reader: &mut impl BufRead) -> Result<bool, std::io::Error> {
    reader.fill_buf().map(|b| !b.is_empty())
}

pub(crate) fn controlflow_to_result<A, B>(x: ControlFlow<A, B>) -> Result<B, LoadError<A>> {
    match x {
        ControlFlow::Continue(x) => Ok(x),
        ControlFlow::Break(y) => Err(LoadError::UserInterrupted(y)),
    }
}
