//! Utilities for interacting with LLMs and loading them.
pub use ggml::util::*;

use std::{
    io::BufRead,
    path::{Path, PathBuf},
};

/// NOTE: The original code relies in promotion rules and automatic cast between
/// int to float. What we do instead is use this macro to convert every term of
/// the multiplication to f64, which should have enough precision bits to hold
/// the final value, then cast to usize. I have observed a discrepancy between
/// the ctx_size found using this code, and the one in llama.cpp. The number for
/// rust ends up being slightly lower, but no "out of memory" errors are
/// reported by ggml.
#[macro_export]
#[doc(hidden)]
macro_rules! mulf {
    ($term:expr, $($terms:expr),*) => {
        usize::try_from((($term as f64) $(* ($terms as f64))*) as u64).unwrap()
    };
}

use memmap2::{Mmap, MmapAsRawDesc, MmapOptions};
use thiserror::Error;

use crate::{FileType, LoadError};

/// Read the filetype from a reader.
pub fn read_filetype(reader: &mut dyn BufRead) -> Result<FileType, LoadError> {
    let ftype = read_i32(reader)?;
    FileType::try_from(ftype).map_err(|_| LoadError::UnsupportedFileType(ftype))
}

/// Used to buffer incoming tokens until they produce a valid string of UTF-8 text.
///
/// Tokens are *not* valid UTF-8 by themselves. However, the LLM will produce valid UTF-8
/// from multiple tokens. This helps alleviate that issue.
#[derive(Clone, PartialEq, Eq, Default)]
pub struct TokenUtf8Buffer(Vec<u8>);
impl TokenUtf8Buffer {
    /// Create a new buffer.
    pub const fn new() -> Self {
        Self(vec![])
    }

    /// Add a token to the buffer. If the buffer contains a valid string of UTF-8 text,
    /// it is returned and the buffer is cleared for next use.
    pub fn push(&mut self, token: &[u8]) -> Option<String> {
        self.0.extend_from_slice(token);
        match std::str::from_utf8(&self.0) {
            Ok(s) => {
                let out = s.to_owned();
                self.0 = vec![];
                Some(out)
            }
            Err(..) => {
                for i in 1..self.0.len() {
                    let slice = &self.0[i..];
                    if slice.is_empty() {
                        break;
                    }

                    if let Ok(s) = std::str::from_utf8(slice) {
                        let out = s.to_owned();
                        self.0 = vec![];
                        return Some(out);
                    }
                }
                None
            }
        }
    }
}

#[derive(Error, Debug)]
/// Errors encountered during the loading process.
pub enum FindAllModelFilesError {
    #[error("no parent path for {path:?}")]
    /// There is no parent path for a given path.
    NoParentPath {
        /// The path without a parent.
        path: PathBuf,
    },
    #[error("non-specific I/O error")]
    /// A non-specific IO error.
    IO(#[from] std::io::Error),
}

/// Find all the files related to a model.
pub fn find_all_model_files(main_path: &Path) -> Result<Vec<PathBuf>, FindAllModelFilesError> {
    let mut main_path_parent =
        main_path
            .parent()
            .ok_or_else(|| FindAllModelFilesError::NoParentPath {
                path: main_path.to_owned(),
            })?;
    if main_path_parent.to_str() == Some("") {
        main_path_parent = Path::new(".");
    }
    Ok(collect_related_paths(
        main_path,
        std::fs::read_dir(main_path_parent)?
            .filter_map(Result::ok)
            .map(|de| de.path()),
    ))
}

fn collect_related_paths(
    main_path: &Path,
    directory_paths: impl Iterator<Item = PathBuf>,
) -> Vec<PathBuf> {
    let main_filename = main_path.file_name().and_then(|p| p.to_str());

    let mut paths: Vec<PathBuf> = directory_paths
        .filter(|p| {
            p.file_name()
                .and_then(|p| p.to_str())
                .zip(main_filename)
                .map(|(part_filename, main_filename)| {
                    match part_filename.strip_prefix(main_filename) {
                        Some(suffix) => {
                            suffix.is_empty()
                                || (suffix
                                    .strip_prefix('.')
                                    .map(|s| s.parse::<usize>().is_ok())
                                    .unwrap_or(false))
                        }
                        None => false,
                    }
                })
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    paths
}

/// mmap with MAP_POPULATE
pub fn mmap_populate<T: MmapAsRawDesc>(file: T) -> Result<Mmap, std::io::Error> {
    unsafe { MmapOptions::new().populate().map(file) }
}

/// Calculate softmax for a slice
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let mut probs = logits.to_vec();
    let max_logit = probs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f32 = logits.iter().map(|v| (v - max_logit).exp()).sum();
    for v in probs.iter_mut() {
        *v = (*v - max_logit).exp() / sum;
    }
    probs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_related_paths() {
        let main_path = PathBuf::from("/models/llama.bin");
        let directory_paths = [
            "/models/llama.bin",
            "/models/llama.bin.1",
            "/models/llama.bin.2",
            "/models/llama.bin.tmp",
        ]
        .map(PathBuf::from);
        let expected_paths = [
            "/models/llama.bin",
            "/models/llama.bin.1",
            "/models/llama.bin.2",
        ]
        .map(PathBuf::from);

        let output_paths = collect_related_paths(&main_path, directory_paths.into_iter());
        assert_eq!(expected_paths.as_slice(), output_paths);
    }

    #[test]
    fn test_valid_utf8() {
        let mut buffer = TokenUtf8Buffer::new();
        assert_eq!(buffer.push(b"hello").as_deref(), Some("hello"));
        assert_eq!(buffer.push(&[0xE2, 0x82, 0xAC]).as_deref(), Some("€"));
    }

    #[test]
    fn test_partial_utf8() {
        let mut buffer = TokenUtf8Buffer::new();
        assert_eq!(buffer.push(&[0xE2, 0x82]).as_deref(), None);
        assert_eq!(buffer.push(&[0xAC]).as_deref(), Some("€"));
    }

    #[test]
    fn test_invalid_prelude_for_valid_utf8() {
        let mut buffer = TokenUtf8Buffer::new();
        assert_eq!(buffer.push(&[0xD8]).as_deref(), None);
        assert_eq!(buffer.push(&[0xE2, 0x82]).as_deref(), None);
        assert_eq!(buffer.push(&[0xAC]).as_deref(), Some("€"));
    }
}
