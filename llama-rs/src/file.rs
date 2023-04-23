use crate::LoadError;
pub use std::fs::File;
pub use std::io::{BufRead, BufReader, BufWriter, Read, Write};

pub fn rw_i32(reader: &mut impl BufRead, writer: &mut impl Write) -> Result<i32, LoadError> {
    Ok(i32::from_le_bytes(rw::<4>(reader, writer)?))
}

pub fn rw_u32(reader: &mut impl BufRead, writer: &mut impl Write) -> Result<u32, LoadError> {
    Ok(u32::from_le_bytes(rw::<4>(reader, writer)?))
}

pub fn rw_f32(reader: &mut impl BufRead, writer: &mut impl Write) -> Result<f32, LoadError> {
    Ok(f32::from_le_bytes(rw::<4>(reader, writer)?))
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
    let bytes: [u8; N] = ggml_loader::util::read_bytes(reader)?;
    writer.write_all(&bytes)?;
    Ok(bytes)
}
