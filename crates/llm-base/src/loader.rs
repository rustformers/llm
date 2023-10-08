//! Functionality for loading models. Very barebones; designed to be driven by `llm`.

use std::{
    fmt::{Display, Formatter},
    io::{BufRead, Seek, SeekFrom},
    path::Path,
    sync::Arc,
};

use crate::{
    model::{Hyperparameters, HyperparametersReadError},
    KnownModel, LoraAdapter, ModelContext, ModelParameters, Tokenizer,
};
use ggml::{
    format::gguf::{Gguf, Metadata, TensorInfo},
    sys::llama::llama_ftype,
    Context, MAX_NAME_LENGTH,
};
pub use ggml::{
    format::gguf::{MetadataError, MetadataExt},
    format::ContainerType,
    util::FileMagic,
};
use thiserror::Error;

#[derive(Debug, PartialEq, Clone, Copy, Eq, Default)]
/// Information about the file.
pub struct FileType {
    /// The format of the tensors.
    pub format: FileTypeFormat,
    /// The quantization version.
    pub quantization_version: u32,
}
impl From<FileType> for i32 {
    fn from(value: FileType) -> Self {
        (value.quantization_version * ggml::QNT_VERSION_FACTOR) as i32
            + llama_ftype::from(value.format)
    }
}
impl TryFrom<i32> for FileType {
    type Error = llama_ftype;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        let format =
            FileTypeFormat::try_from(((value as u32) % ggml::QNT_VERSION_FACTOR) as llama_ftype)?;

        Ok(Self {
            format,
            quantization_version: (value as u32) / ggml::QNT_VERSION_FACTOR,
        })
    }
}
impl Display for FileType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}_qnt{}", self.format, self.quantization_version)
    }
}
impl FileType {
    /// Helper function that reads the file type from the metadata and converts
    /// it to the enum, or fails with a `HyperparametersReadError`.
    pub fn read_for_hyperparameters(
        metadata: &Metadata,
    ) -> Result<Option<Self>, HyperparametersReadError> {
        metadata
            .get("general.file_type")
            .and_then(|v| v.as_uint32())
            .map(|v| {
                FileType::try_from(v as i32).map_err(|ftype| {
                    HyperparametersReadError::UnsupportedFileType { file_type: ftype }
                })
            })
            .transpose()
    }
}

/// How the tensors are stored in GGML LLM models.
#[derive(Debug, PartialEq, Clone, Copy, Eq, Default)]
#[allow(non_camel_case_types)]
pub enum FileTypeFormat {
    /// All tensors are stored as f32.
    F32,
    #[default]
    /// All tensors are mostly stored as `f16`, except for the 1D tensors (32-bit).
    MostlyF16,
    /// All tensors are mostly stored as `Q4_0`, except for the 1D tensors (32-bit).
    MostlyQ4_0,
    /// All tensors are mostly stored as `Q4_1`, except for the 1D tensors (32-bit)
    MostlyQ4_1,
    /// All tensors are mostly stored as `Q4_1`, except for the 1D tensors (32-bit)
    /// and the `tok_embeddings.weight` (f16) and `output.weight` tensors (f16).
    MostlyQ4_1SomeF16,
    /// All tensors are mostly stored as `Q8_0`, except for the 1D tensors (32-bit).
    MostlyQ8_0,
    /// All tensors are mostly stored as `Q5_0`, except for the 1D tensors (32-bit).
    MostlyQ5_0,
    /// All tensors are mostly stored as `Q5_1`, except for the 1D tensors (32-bit).
    MostlyQ5_1,
    /// The tensors are stored using the `Q2_K` quantization scheme.
    MostlyQ2_K,
    /// The tensors are stored using the `Q3_K_S` quantization scheme.
    MostlyQ3_K_S,
    /// The tensors are stored using the `Q3_K_M` quantization scheme.
    MostlyQ3_K_M,
    /// The tensors are stored using the `Q3_K_L` quantization scheme.
    MostlyQ3_K_L,
    /// The tensors are stored using the `Q4_K_S` quantization scheme.
    MostlyQ4_K_S,
    /// The tensors are stored using the `Q4_K_M` quantization scheme.
    MostlyQ4_K_M,
    /// The tensors are stored using the `Q5_K_S` quantization scheme.
    MostlyQ5_K_S,
    /// The tensors are stored using the `Q5_K_M` quantization scheme.
    MostlyQ5_K_M,
    /// The tensors are stored using the `Q6_K` quantization scheme.
    MostlyQ6_K,
}
impl TryFrom<llama_ftype> for FileTypeFormat {
    type Error = llama_ftype;

    fn try_from(value: llama_ftype) -> Result<Self, Self::Error> {
        use ggml::sys::llama::*;
        match value {
            LLAMA_FTYPE_ALL_F32 => Ok(FileTypeFormat::F32),
            LLAMA_FTYPE_MOSTLY_F16 => Ok(FileTypeFormat::MostlyF16),
            LLAMA_FTYPE_MOSTLY_Q4_0 => Ok(FileTypeFormat::MostlyQ4_0),
            LLAMA_FTYPE_MOSTLY_Q4_1 => Ok(FileTypeFormat::MostlyQ4_1),
            LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 => Ok(FileTypeFormat::MostlyQ4_1SomeF16),
            LLAMA_FTYPE_MOSTLY_Q8_0 => Ok(FileTypeFormat::MostlyQ8_0),
            LLAMA_FTYPE_MOSTLY_Q5_0 => Ok(FileTypeFormat::MostlyQ5_0),
            LLAMA_FTYPE_MOSTLY_Q5_1 => Ok(FileTypeFormat::MostlyQ5_1),
            LLAMA_FTYPE_MOSTLY_Q2_K => Ok(FileTypeFormat::MostlyQ2_K),
            LLAMA_FTYPE_MOSTLY_Q3_K_S => Ok(FileTypeFormat::MostlyQ3_K_S),
            LLAMA_FTYPE_MOSTLY_Q3_K_M => Ok(FileTypeFormat::MostlyQ3_K_M),
            LLAMA_FTYPE_MOSTLY_Q3_K_L => Ok(FileTypeFormat::MostlyQ3_K_L),
            LLAMA_FTYPE_MOSTLY_Q4_K_S => Ok(FileTypeFormat::MostlyQ4_K_S),
            LLAMA_FTYPE_MOSTLY_Q4_K_M => Ok(FileTypeFormat::MostlyQ4_K_M),
            LLAMA_FTYPE_MOSTLY_Q5_K_S => Ok(FileTypeFormat::MostlyQ5_K_S),
            LLAMA_FTYPE_MOSTLY_Q5_K_M => Ok(FileTypeFormat::MostlyQ5_K_M),
            LLAMA_FTYPE_MOSTLY_Q6_K => Ok(FileTypeFormat::MostlyQ6_K),
            #[allow(clippy::unnecessary_cast)]
            _ => Err(value),
        }
    }
}
impl From<FileTypeFormat> for llama_ftype {
    fn from(value: FileTypeFormat) -> Self {
        use ggml::sys::llama::*;
        match value {
            FileTypeFormat::F32 => LLAMA_FTYPE_ALL_F32,
            FileTypeFormat::MostlyF16 => LLAMA_FTYPE_MOSTLY_F16,
            FileTypeFormat::MostlyQ4_0 => LLAMA_FTYPE_MOSTLY_Q4_0,
            FileTypeFormat::MostlyQ4_1 => LLAMA_FTYPE_MOSTLY_Q4_1,
            FileTypeFormat::MostlyQ4_1SomeF16 => LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16,
            FileTypeFormat::MostlyQ8_0 => LLAMA_FTYPE_MOSTLY_Q8_0,
            FileTypeFormat::MostlyQ5_0 => LLAMA_FTYPE_MOSTLY_Q5_0,
            FileTypeFormat::MostlyQ5_1 => LLAMA_FTYPE_MOSTLY_Q5_1,
            FileTypeFormat::MostlyQ2_K => LLAMA_FTYPE_MOSTLY_Q2_K,
            FileTypeFormat::MostlyQ3_K_S => LLAMA_FTYPE_MOSTLY_Q3_K_S,
            FileTypeFormat::MostlyQ3_K_M => LLAMA_FTYPE_MOSTLY_Q3_K_M,
            FileTypeFormat::MostlyQ3_K_L => LLAMA_FTYPE_MOSTLY_Q3_K_L,
            FileTypeFormat::MostlyQ4_K_S => LLAMA_FTYPE_MOSTLY_Q4_K_S,
            FileTypeFormat::MostlyQ4_K_M => LLAMA_FTYPE_MOSTLY_Q4_K_M,
            FileTypeFormat::MostlyQ5_K_S => LLAMA_FTYPE_MOSTLY_Q5_K_S,
            FileTypeFormat::MostlyQ5_K_M => LLAMA_FTYPE_MOSTLY_Q5_K_M,
            FileTypeFormat::MostlyQ6_K => LLAMA_FTYPE_MOSTLY_Q6_K,
        }
    }
}
impl Display for FileTypeFormat {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                FileTypeFormat::F32 => "f32",
                FileTypeFormat::MostlyF16 => "f16",
                FileTypeFormat::MostlyQ4_0 => "q4_0",
                FileTypeFormat::MostlyQ4_1 => "q4_1",
                FileTypeFormat::MostlyQ4_1SomeF16 => "q4_1_with_f16",
                FileTypeFormat::MostlyQ8_0 => "q8_0",
                FileTypeFormat::MostlyQ5_0 => "q5_0",
                FileTypeFormat::MostlyQ5_1 => "q5_1",
                FileTypeFormat::MostlyQ2_K => "q2_k",
                FileTypeFormat::MostlyQ3_K_S => "q3_K_S",
                FileTypeFormat::MostlyQ3_K_M => "q3_K_M",
                FileTypeFormat::MostlyQ3_K_L => "q3_K_L",
                FileTypeFormat::MostlyQ4_K_S => "q4_K_S",
                FileTypeFormat::MostlyQ4_K_M => "q4_K_M",
                FileTypeFormat::MostlyQ5_K_S => "q5_K_S",
                FileTypeFormat::MostlyQ5_K_M => "q5_K_M",
                FileTypeFormat::MostlyQ6_K => "q6_k",
            }
        )
    }
}

/// Helper trait that implements traits required for reading.
pub trait Source: BufRead + Seek {}
impl<S: BufRead + Seek> Source for S {}

/// Errors that can occur when loading a known model.
#[derive(Error, Debug)]
pub enum LoadKnownError {
    /// Failed to read the hyperparameters
    #[error("{0}")]
    HyperparametersReadError(#[from] HyperparametersReadError),
    /// Failed to load the tensors
    #[error("{0}")]
    TensorLoadError(#[from] TensorLoadError),
}

/// Each variant represents a step within loading a known model.
#[derive(Debug, Copy, Clone)]
#[doc(hidden)]
pub enum LoadKnownProgress<'a> {
    /// A LoRA has been applied.
    LoraApplied { name: &'a str, source: &'a Path },
    /// A tensor has been loaded.
    TensorLoaded { current_tensor: usize },
}

/// Internal function that takes all of the state that can be derived without
/// knowing a concrete type and loads a concrete model. A *lot* of precondition
/// logic is done in `llm`.
// TODO: think about this design. Do we want to let people to be able to load
// known models directly?
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn load_known_internal<M: KnownModel>(
    source: &mut dyn Source,
    gguf: &Gguf,
    tokenizer: Tokenizer,
    context: Context,
    lora_adapters: Option<Vec<LoraAdapter>>,
    progress_callback: &mut dyn FnMut(LoadKnownProgress),
    params: ModelParameters,
) -> Result<M, LoadKnownError> {
    let hyperparameters = <M::Hyperparameters>::read_gguf(&gguf.metadata)?;
    let tl = ModelTensorLoader {
        tensor_loader: TensorLoader {
            source,
            gguf: &gguf,
            context,
        },
        lora_adapters,
        progress_callback,
        loaded_tensor_count: 0,
    };

    Ok(KnownModel::new(hyperparameters, params, tokenizer, tl)?)
}

/// A helper struct for loading tensors from a model.
pub struct ModelTensorLoader<'a> {
    pub(crate) tensor_loader: TensorLoader<'a>,
    pub(crate) lora_adapters: Option<Vec<LoraAdapter>>,
    pub(crate) progress_callback: &'a mut dyn FnMut(LoadKnownProgress),
    pub(crate) loaded_tensor_count: usize,
}
impl ModelTensorLoader<'_> {
    /// Load a tensor from the model.
    pub fn load(&mut self, name: &str) -> Result<ggml::Tensor, TensorLoadError> {
        let (mut tensor, info) = self.tensor_loader.load(name)?;

        if let Some(lora_adapters) = &mut self.lora_adapters {
            for lora_adapter in lora_adapters {
                lora_adapter.patch(name, info, &mut tensor)?;
                (self.progress_callback)(LoadKnownProgress::LoraApplied {
                    name,
                    source: &lora_adapter.path,
                });
            }
        }

        self.loaded_tensor_count += 1;
        (self.progress_callback)(LoadKnownProgress::TensorLoaded {
            current_tensor: self.loaded_tensor_count,
        });

        Ok(tensor)
    }

    /// Finish loading tensors from the model, and get the model context.
    pub fn finish(self) -> ModelContext {
        // We can ignore this warning as it's OK to share this particular
        // context around, being that it is immutable.
        #[allow(clippy::arc_with_non_send_sync)]
        ModelContext(Arc::new(self.tensor_loader.finish()))
    }
}

pub(crate) struct TensorLoader<'a> {
    pub source: &'a mut dyn Source,
    pub gguf: &'a Gguf,
    pub context: Context,
}
impl TensorLoader<'_> {
    pub fn load(&mut self, name: &str) -> Result<(ggml::Tensor, &TensorInfo), TensorLoadError> {
        let info = self
            .gguf
            .tensor_infos
            .get(name)
            .ok_or(TensorLoadError::UnknownTensor {
                tensor_name: String::from(name),
            })?;

        let ty = info.element_type;
        let dims = &info.dimensions;

        let mut tensor = match dims.len() {
            1 => self.context.new_tensor_1d(ty, dims[0]),
            2 => self.context.new_tensor_2d(ty, dims[0], dims[1]),
            3 => self.context.new_tensor_3d(ty, dims[0], dims[1], dims[2]),
            other => {
                return Err(TensorLoadError::UnsupportedTensorDimensionCount {
                    tensor_name: name.to_string(),
                    dimensions: other,
                });
            }
        };

        let offset = self.gguf.tensor_data_position + info.offset;
        match self.context.storage().as_mmap() {
            Some(mmap) => unsafe {
                let ptr = mmap.as_ptr().offset(offset as isize);
                tensor.set_data(ptr as *mut std::ffi::c_void);
            },
            None => {
                let buf: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(tensor.data() as *mut u8, tensor.nbytes())
                };
                self.source.seek(SeekFrom::Start(offset))?;
                self.source.read_exact(buf)?;
            }
        }

        // The tensor name is truncated to its maximum length.
        let tensor_name = if name.len() >= MAX_NAME_LENGTH {
            &name[name.len() - MAX_NAME_LENGTH..]
        } else {
            name
        };

        Ok((tensor.set_name(tensor_name), info))
    }

    pub fn finish(self) -> Context {
        self.context
    }
}

#[derive(Error, Debug)]
/// Errors encountered during loaing of tensors.
pub enum TensorLoadError {
    #[error("unknown tensor `{tensor_name}`")]
    /// The tensor `tensor_name` is required for this model architecture,
    /// but was not found in the model.
    UnknownTensor {
        /// The name of the tensor.
        tensor_name: String,
    },
    /// A tensor with an unsupported number of dimensions was encountered.
    #[error(
        "tensor {tensor_name} has {dimensions} dimensions, but only 1-3 dimensions are supported"
    )]
    UnsupportedTensorDimensionCount {
        /// The name of the tensor.
        tensor_name: String,
        /// The number of dimensions that were encountered.
        dimensions: usize,
    },
    #[error("non-specific I/O error")]
    /// A non-specific IO error.
    Io(#[from] std::io::Error),
}
