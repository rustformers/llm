use std::{
    fmt::Debug,
    io::BufRead,
    path::{Path, PathBuf},
};

use thiserror::Error;

/// Each variant represents a step within the process of loading the model.
/// These can be used to report progress to the user.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum LoadProgress<'a, H> {
    /// The hyperparameters have been loaded from the model.
    HyperparametersLoaded(&'a H),
    /// The context has been created.
    ContextSize {
        /// The size of the context.
        bytes: usize,
    },
    /// A part of the model is being loaded.
    PartLoading {
        /// The path to the model part.
        file: &'a Path,
        /// The current part (0-indexed).
        current_part: usize,
        /// The number of total parts.
        total_parts: usize,
    },
    /// A tensor from the current part has been loaded.
    PartTensorLoaded {
        /// The path to the model part.
        file: &'a Path,
        /// The current tensor (0-indexed).
        current_tensor: usize,
        /// The number of total tensors.
        tensor_count: usize,
    },
    /// A model part has finished fully loading.
    PartLoaded {
        /// The path to the model part.
        file: &'a Path,
        /// The number of bytes in the part.
        byte_size: usize,
        /// The number of tensors in the part.
        tensor_count: usize,
    },
}

#[derive(Error, Debug)]
/// Errors encountered during the loading process.
pub enum LoadError {
    #[error("could not open file {path:?}")]
    /// A file failed to open.
    OpenFileFailed {
        /// The original error.
        source: std::io::Error,
        /// The path that failed.
        path: PathBuf,
    },
    #[error("no parent path for {path:?}")]
    /// There is no parent path for a given path.
    NoParentPath {
        /// The path without a parent.
        path: PathBuf,
    },
    #[error("unable to read exactly {bytes} bytes")]
    /// Reading exactly `bytes` from a file failed.
    ReadExactFailed {
        /// The original error.
        source: std::io::Error,
        /// The number of bytes that were attempted to be read.
        bytes: usize,
    },
    #[error("non-specific I/O error")]
    /// A non-specific IO error.
    IO(#[from] std::io::Error),
    #[error("could not convert bytes to a UTF-8 string")]
    /// One of the strings encountered was not valid UTF-8.
    InvalidUtf8(#[from] std::string::FromUtf8Error),
    #[error("invalid integer conversion")]
    /// One of the integers encountered could not be converted to a more appropriate type.
    InvalidIntegerConversion(#[from] std::num::TryFromIntError),
    #[error("invalid magic number for {path:?}")]
    /// An invalid magic number was encountered during the loading process.
    InvalidMagic {
        /// The path that failed.
        path: PathBuf,
    },
    #[error("invalid file format version {value}")]
    /// The version of the format is not supported by this version of `llama-rs`.
    InvalidFormatVersion {
        /// The version that was encountered.
        value: u32,
    },
    #[error("invalid value {ftype} for `f16` in hyperparameters")]
    /// The `f16` hyperparameter had an invalid value.
    HyperparametersF16Invalid {
        /// The format type that was encountered.
        ftype: u32,
    },
    #[error("unknown tensor `{tensor_name}` in {path:?}")]
    /// The tensor `tensor_name` was encountered during the loading of `path`, but was not seen during
    /// the model prelude.
    UnknownTensor {
        /// The name of the tensor.
        tensor_name: String,
        /// The path that failed.
        path: PathBuf,
    },
    #[error("the tensor `{tensor_name}` has the wrong size in {path:?}")]
    /// The tensor `tensor_name` did not match its expected size.
    TensorWrongSize {
        /// The name of the tensor.
        tensor_name: String,
        /// The path that failed.
        path: PathBuf,
    },
    /// The tensor `tensor_name` did not have the expected format type.
    #[error("invalid ftype {ftype} for tensor `{tensor_name}` in {path:?}")]
    InvalidFtype {
        /// The name of the tensor.
        tensor_name: String,
        /// The format type that was encountered.
        ftype: u32,
        /// The path that failed.
        path: PathBuf,
    },
}

/// Default load progress callback function
pub fn load_progress<H: Debug>(progress: LoadProgress<H>) {
    match progress {
        LoadProgress::HyperparametersLoaded(hparams) => {
            log::debug!("Loaded hyperparameters {hparams:#?}")
        }
        LoadProgress::ContextSize { bytes } => log::info!(
            "ggml ctx size = {:.2} MB\n",
            bytes as f64 / (1024.0 * 1024.0)
        ),
        LoadProgress::PartLoading {
            file,
            current_part,
            total_parts,
        } => {
            let current_part = current_part + 1;
            log::info!(
                "Loading model part {}/{} from '{}'\n",
                current_part,
                total_parts,
                file.to_string_lossy(),
            )
        }
        LoadProgress::PartTensorLoaded {
            current_tensor,
            tensor_count,
            ..
        } => {
            let current_tensor = current_tensor + 1;
            if current_tensor % 8 == 0 {
                log::info!("Loaded tensor {current_tensor}/{tensor_count}");
            }
        }
        LoadProgress::PartLoaded {
            file,
            byte_size,
            tensor_count,
        } => {
            log::info!("Loading of '{}' complete", file.to_string_lossy());
            log::info!(
                "Model size = {:.2} MB / num tensors = {}",
                byte_size as f64 / 1024.0 / 1024.0,
                tensor_count
            );
        }
    }
}

/// Read bytes
pub fn read_bytes<const N: usize>(reader: &mut impl BufRead) -> Result<[u8; N], LoadError> {
    let mut bytes = [0u8; N];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| LoadError::ReadExactFailed {
            source: e,
            bytes: N,
        })?;
    Ok(bytes)
}

/// Ready bytes with length
pub fn read_bytes_with_len(reader: &mut impl BufRead, len: usize) -> Result<Vec<u8>, LoadError> {
    let mut bytes = vec![0u8; len];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| LoadError::ReadExactFailed {
            source: e,
            bytes: len,
        })?;
    Ok(bytes)
}

/// Read an i32
pub fn read_i32(reader: &mut impl BufRead) -> Result<i32, LoadError> {
    Ok(i32::from_le_bytes(read_bytes::<4>(reader)?))
}

/// Read a u32
pub fn read_u32(reader: &mut impl BufRead) -> Result<u32, LoadError> {
    Ok(u32::from_le_bytes(read_bytes::<4>(reader)?))
}

/// Read an f32
pub fn read_f32(reader: &mut impl BufRead) -> Result<f32, LoadError> {
    Ok(f32::from_le_bytes(read_bytes::<4>(reader)?))
}

/// Helper function. Reads a string from the buffer and returns it.
pub fn read_string(reader: &mut impl BufRead, len: usize) -> Result<String, LoadError> {
    Ok(String::from_utf8(read_bytes_with_len(reader, len)?)?)
}

/// Find all model files
pub fn find_all_model_files(main_path: &Path) -> Result<Vec<PathBuf>, LoadError> {
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
    use llm_base::TokenUtf8Buffer;

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
