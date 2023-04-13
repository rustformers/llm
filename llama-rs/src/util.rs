use std::path::{Path, PathBuf};

use crate::LoadError;

/// Used to buffer incoming tokens until they produce a valid string of UTF-8 text.
///
/// Tokens are *not* valid UTF-8 by themselves. However, the LLM will produce valid UTF-8
/// from multiple tokens. This helps alleviate that issue.
#[derive(Clone, PartialEq, Default)]
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

    /// Adapt a `&str` callback so that it can be used in a `&[u8]` context.
    pub fn adapt_callback<'a, E: std::error::Error + 'static>(
        mut callback: impl FnMut(&str) -> Result<(), E> + 'a,
    ) -> impl FnMut(&[u8]) -> Result<(), E> + 'a {
        let mut buffer = Self::new();
        move |token| match buffer.push(token) {
            Some(tokens) => callback(&tokens),
            None => Ok(()),
        }
    }
}

pub(crate) fn find_all_model_files(main_path: &Path) -> Result<Vec<PathBuf>, LoadError> {
    Ok(collect_related_paths(
        main_path,
        std::fs::read_dir(main_path.parent().ok_or_else(|| LoadError::NoParentPath {
            path: main_path.to_owned(),
        })?)?
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
