use std::path::{Path, PathBuf};
use thiserror::Error;

/// Each variant represents a step within the process of loading the model.
/// These can be used to report progress to the user.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum LoadProgress<'a, T> {
    HyperparametersLoaded(&'a T),
    BadToken {
        index: usize,
    },
    ContextSize {
        bytes: usize,
    },
    MemorySize {
        bytes: usize,
        n_mem: usize,
    },
    PartLoading {
        file: &'a Path,
        current_part: usize,
        total_parts: usize,
    },
    PartTensorLoaded {
        file: &'a Path,
        current_tensor: usize,
        tensor_count: usize,
    },
    PartLoaded {
        file: &'a Path,
        byte_size: usize,
        tensor_count: usize,
    },
}

#[derive(Error, Debug)]
pub enum LoadError {
    #[error("could not open file {path:?}")]
    OpenFileFailed {
        source: std::io::Error,
        path: PathBuf,
    },
    #[error("no parent path for {path:?}")]
    NoParentPath { path: PathBuf },
    #[error("unable to read exactly {bytes} bytes")]
    ReadExactFailed {
        source: std::io::Error,
        bytes: usize,
    },
    #[error("non-specific I/O error")]
    IO(#[from] std::io::Error),

    #[error("could not convert bytes to a UTF-8 string")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),
    #[error("invalid integer conversion")]
    InvalidIntegerConversion(#[from] std::num::TryFromIntError),

    #[error("unversioned magic number, regenerate your ggml models")]
    UnversionedMagic,
    #[error("invalid magic number for {path:?}")]
    InvalidMagic { path: PathBuf },
    #[error("invalid file format version {value}")]
    InvalidFormatVersion { value: u32 },
    #[error("invalid value {value} for `f16` in hyperparameters")]
    HyperparametersF16Invalid { value: i32 },
    #[error("unknown tensor `{tensor_name}` in {path:?}")]
    UnknownTensor { tensor_name: String, path: PathBuf },
    #[error("the tensor `{tensor_name}` has the wrong size in {path:?}")]
    TensorWrongSize { tensor_name: String, path: PathBuf },
    #[error("invalid ftype {ftype} in {path:?}")]
    InvalidFtype { ftype: i32, path: PathBuf },
}
