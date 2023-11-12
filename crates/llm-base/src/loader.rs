//! Functionality for loading models. Very barebones; designed to be driven by `llm`.

use std::{
    fmt::{Display, Formatter},
    fs::File,
    io::{BufRead, BufReader, Seek, SeekFrom},
    path::Path,
    sync::Arc,
};

use crate::{
    model::{HyperparametersReadError, ModelData, ModelLoadArgs, ModelLoadError},
    LoraAdapter, Model, ModelContext, ModelParameters, TokenizerLoadError, TokenizerSource,
};
pub use ggml::{format::gguf::MetadataError, format::ContainerType, util::FileMagic};
use ggml::{
    format::gguf::{Gguf, GgufLoadError, Metadata, MetadataValue, MetadataValueType, TensorInfo},
    sys::llama::llama_ftype,
    Context, MAX_NAME_LENGTH,
};
use memmap2::Mmap;
use thiserror::Error;

#[derive(Debug, PartialEq, Clone, Copy, Eq, Default)]
/// Information about the file.
pub struct FileType {
    /// The format of the tensors.
    pub format: FileTypeFormat,
    /// The quantization version.
    pub quantization_version: u32,
}
impl From<FileType> for llama_ftype {
    fn from(value: FileType) -> Self {
        (value.quantization_version * ggml::QNT_VERSION_FACTOR) as llama_ftype
            + llama_ftype::from(value.format)
    }
}
impl TryFrom<llama_ftype> for FileType {
    type Error = llama_ftype;

    fn try_from(value: llama_ftype) -> Result<Self, Self::Error> {
        let format =
            FileTypeFormat::try_from(((value as u32) % ggml::QNT_VERSION_FACTOR) as llama_ftype)?;

        Ok(Self {
            format,
            quantization_version: value / ggml::QNT_VERSION_FACTOR,
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
            .get_optional("general.file_type")
            .and_then(|v| v.as_uint32())
            .map(|v| {
                FileType::try_from(v as llama_ftype).map_err(|ftype| {
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

/// Each variant represents a step within the process of loading the model.
/// These can be used to report progress to the user.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum LoadProgress<'a> {
    /// The hyperparameters have been loaded from the model.
    HyperparametersLoaded,
    /// The context has been created.
    ContextSize {
        /// The size of the context.
        bytes: usize,
    },
    /// A tensor was patched with a LoRA.
    LoraApplied {
        /// The name of the patched tensor.
        name: &'a str,
        /// LoRA file the patch was applied from.
        source: &'a Path,
    },
    /// A tensor from the current part has been loaded.
    TensorLoaded {
        /// The current tensor (0-indexed).
        current_tensor: usize,
        /// The number of total tensors.
        tensor_count: usize,
    },
    /// A model part has finished fully loading.
    Loaded {
        /// The number of bytes in the part.
        file_size: u64,
        /// The number of tensors in the part.
        tensor_count: usize,
    },
}

#[derive(Error, Debug)]
/// Errors encountered during the loading process.
pub enum LoadError {
    #[error("the file does not exist")]
    /// The file does not exist.
    FileDoesNotExist,
    #[error("could not open file")]
    /// A file failed to open.
    OpenFileFailed {
        /// The original error.
        source: std::io::Error,
    },
    #[error("non-specific I/O error")]
    /// A non-specific IO error.
    Io(#[from] std::io::Error),
    #[error("could not convert bytes to a UTF-8 string")]
    /// One of the strings encountered was not valid UTF-8.
    InvalidUtf8(#[from] std::string::FromUtf8Error),
    #[error("invalid integer conversion")]
    /// One of the integers encountered could not be converted to a more appropriate type.
    InvalidIntegerConversion(#[from] std::num::TryFromIntError),
    #[error("invalid magic value {magic}")]
    /// An invalid magic value was encountered during the loading process.
    InvalidMagic {
        /// The magic value that was encountered.
        magic: FileMagic,
    },
    #[error("invalid file format {container_type:?}")]
    /// The version of the format is not supported by this version of `llm`.
    InvalidFormatVersion {
        /// The format that was encountered.
        container_type: ContainerType,
    },
    /// The tensor `tensor_name` had an unsupported element type.
    #[error("invalid element type {element_type} for tensor `{tensor_name}`")]
    UnsupportedElementType {
        /// The name of the tensor.
        tensor_name: String,
        /// The element type that was encountered.
        element_type: u32,
    },
    /// The tokenizer could not be loaded.
    #[error("could not load tokenizer: {0}")]
    TokenizerLoadFail(#[from] TokenizerLoadError),
    /// The quantization version was missing, despite this model containing quantized tensors.
    #[error("quantization version was missing, despite model containing quantized tensors")]
    MissingQuantizationVersion,
    /// The quantization version is not supported by this version of `llm`.
    #[error("quantization version {quantization_version:?} is not supported")]
    UnsupportedQuantizationVersion {
        /// The quantization version that was encountered.
        quantization_version: MetadataValue,
    },
    /// The model expected a metadata key-value pair, but the key was missing.
    #[error("missing metadata key {key:?}")]
    MissingMetadataKey {
        /// The key that was missing.
        key: String,
    },
    /// The metadata key-value pair was not of the expected type.
    #[error("metadata key {key:?} was not of the expected type")]
    InvalidMetadataType {
        /// The key with the invalid type.
        key: String,
        /// The expected type.
        expected_type: MetadataValueType,
        /// The actual type.
        actual_type: MetadataValueType,
    },
    /// The file type within the model was not supported by this version of `llm`.
    #[error("file type {file_type} is not supported")]
    UnsupportedFileType {
        /// The file type (ignoring the quantization version) that was encountered.
        file_type: llama_ftype,
    },
    /// The architecture in the file is not known to the loader.
    #[error("unknown architecture {architecture}")]
    UnknownArchitecture {
        /// The architecture that was encountered.
        architecture: String,
    },
    /// An error occurred while reading the hyperparameters.
    #[error("{0}")]
    HyperparametersReadError(#[from] HyperparametersReadError),
    /// An error occurred while loading the concrete model.
    #[error("{0}")]
    ModelLoadError(#[from] ModelLoadError),
}
impl From<GgufLoadError> for LoadError {
    fn from(value: GgufLoadError) -> Self {
        match value {
            GgufLoadError::InvalidMagic(magic) => LoadError::InvalidMagic { magic },
            GgufLoadError::InvalidFormatVersion(container_type) => {
                LoadError::InvalidFormatVersion { container_type }
            }
            GgufLoadError::Io(err) => LoadError::Io(err),
            GgufLoadError::InvalidUtf8(err) => LoadError::InvalidUtf8(err),
            GgufLoadError::InvalidIntegerConversion(err) => {
                LoadError::InvalidIntegerConversion(err)
            }
            GgufLoadError::UnsupportedElementType { tensor_name, ftype } => {
                LoadError::UnsupportedElementType {
                    tensor_name,
                    element_type: ftype,
                }
            }
        }
    }
}
impl From<MetadataError> for LoadError {
    fn from(value: MetadataError) -> Self {
        Self::HyperparametersReadError(HyperparametersReadError::MetadataError(value))
    }
}

/// When given args, attempt to instantiate a model.
pub type ModelLoadCallback = fn(ModelLoadArgs) -> Result<Box<dyn Model>, ModelLoadError>;

/// A factory that can retrieve the constructor for a given model architecture.
pub trait ModelFactory {
    /// For a given architecture name, return a function that will load the model,
    /// or `None` if the architecture is not supported.
    fn load(&self, architecture: &str) -> Option<ModelLoadCallback>;
}

/// Loads the specified GGUF model from disk, determining its architecture from the metadata.
///
/// This method returns a [`Box`], which means that the model will have single ownership.
/// If you'd like to share ownership (i.e. to use the model in multiple threads), we
/// suggest using [`Arc::from(Box<T>)`](https://doc.rust-lang.org/std/sync/struct.Arc.html#impl-From%3CBox%3CT,+Global%3E%3E-for-Arc%3CT%3E)
/// to convert the [`Box`] into an [`Arc`](std::sync::Arc) after loading.
pub fn load(
    path: &Path,
    tokenizer_source: TokenizerSource,
    params: ModelParameters,
    model_factory: impl ModelFactory,
    mut load_progress_callback: impl FnMut(LoadProgress),
) -> Result<Box<dyn Model>, LoadError> {
    if !path.exists() {
        return Err(LoadError::FileDoesNotExist);
    }

    let file = File::open(path).map_err(|e| LoadError::OpenFileFailed { source: e })?;
    let mut reader = BufReader::new(&file);
    tracing::trace!("Read model file from {:?}", path);

    let gguf = Gguf::load(&mut reader)?;
    tracing::trace!("Loaded GGML model from reader");

    let architecture = gguf.metadata.get_str("general.architecture")?;
    let tokenizer = tokenizer_source.retrieve(&gguf)?;

    let quantization_version = gguf.metadata.get_optional("general.quantization_version");
    tracing::trace!(
        "Determined quantization version of model as {:?}",
        quantization_version
    );

    // TODO: this is temporary while we figure out how to handle this
    let any_quantized = gguf
        .tensor_infos
        .values()
        .any(|t| t.element_type.is_quantized());
    if any_quantized {
        match quantization_version {
            Some(MetadataValue::UInt32(2)) => {
                // Currently supported version
            }
            Some(quantization_version) => {
                return Err(LoadError::UnsupportedQuantizationVersion {
                    quantization_version: quantization_version.clone(),
                })
            }
            None => return Err(LoadError::MissingQuantizationVersion),
        }
    }

    let use_mmap = params.prefer_mmap && params.lora_adapters.is_none();

    let ctx_size = gguf
        .tensor_infos
        .values()
        .map(|ti| ti.calc_absolute_size(use_mmap))
        .sum::<usize>();
    tracing::trace!("Context size: {:?}", ctx_size);

    let mut lora_adapters: Option<Vec<LoraAdapter>> = None;
    if let Some(lora_paths) = &params.lora_adapters {
        let adapters: Result<Vec<_>, _> = lora_paths
        .iter()
        .map(|lora_path| {
            // Read the LoRA file
            let lora_file = File::open(lora_path).map_err(|e| LoadError::OpenFileFailed {
                source: e,
            })?;
            let mut lora_reader = BufReader::new(&lora_file);
            let gguf = Gguf::load(&mut lora_reader)?;

            // Collect the names of the tensors that should be patched
            let tensors_to_patch = gguf
                .tensor_infos
                .keys()
                .filter_map(|k| Some(k.rsplit_once('.')?.0.to_owned()))
                .collect();

            tracing::trace!("Loaded LoRA weights");
            // Return the LoRA patches
            #[allow(unreachable_code)]
            Ok::<_, LoadError>(LoraAdapter {
                tensors: gguf.tensor_infos.clone(),
                tensors_to_patch,
                source: Box::new(lora_reader),
                path: lora_path.to_owned(),
                gguf,
                scaling: todo!("Calculate scaling from LoRA file metadata (GGUF does not have standardised metadata yet)"),
            })
        })
        .collect();
        lora_adapters = Some(adapters?);
    }

    (load_progress_callback)(LoadProgress::ContextSize { bytes: ctx_size });
    let (context, file_size) = if use_mmap {
        unsafe {
            let mmap = Mmap::map(&file)?;
            let file_size = mmap.len() as u64;
            (Context::new_with_mmap(mmap), file_size)
        }
    } else {
        (Context::new_with_allocate(ctx_size), file.metadata()?.len())
    };

    let model_constructor =
        model_factory
            .load(architecture)
            .ok_or_else(|| LoadError::UnknownArchitecture {
                architecture: architecture.to_string(),
            })?;
    let model = (model_constructor)(ModelLoadArgs {
        gguf: &gguf,
        data: ModelData { params, tokenizer },
        tensor_loader: ModelTensorLoader {
            tensor_loader: TensorLoader {
                source: &mut reader,
                gguf: &gguf,
                context,
            },
            lora_adapters,
            progress_callback: &mut load_progress_callback,
            loaded_tensor_count: 0,
        },
    })?;

    (load_progress_callback)(LoadProgress::Loaded {
        file_size,
        tensor_count: gguf.tensor_infos.len(),
    });

    tracing::trace!("Loaded model");

    Ok(model)
}

/// A implementation for `load_progress_callback` that outputs to `stdout`.
pub fn load_progress_callback_stdout(progress: LoadProgress) {
    match progress {
        LoadProgress::HyperparametersLoaded => println!("Loaded hyperparameters"),
        LoadProgress::ContextSize { bytes } => println!(
            "ggml ctx size = {:.2} MB\n",
            bytes as f64 / (1024.0 * 1024.0)
        ),
        LoadProgress::TensorLoaded {
            current_tensor,
            tensor_count,
            ..
        } => {
            let current_tensor = current_tensor + 1;
            if current_tensor % 8 == 0 {
                println!("Loaded tensor {current_tensor}/{tensor_count}");
            }
        }
        LoadProgress::Loaded {
            file_size: byte_size,
            tensor_count,
        } => {
            println!("Loading of model complete");
            println!(
                "Model size = {:.2} MB / num tensors = {}",
                byte_size as f64 / 1024.0 / 1024.0,
                tensor_count
            );
        }
        LoadProgress::LoraApplied { name, source } => {
            println!(
                "Patched tensor {} via LoRA from '{}'",
                name,
                source.file_name().unwrap().to_str().unwrap()
            );
        }
    };
}

/// A helper struct for loading tensors from a model.
pub struct ModelTensorLoader<'a> {
    pub(crate) tensor_loader: TensorLoader<'a>,
    pub(crate) lora_adapters: Option<Vec<LoraAdapter>>,
    pub(crate) progress_callback: &'a mut dyn FnMut(LoadProgress),
    pub(crate) loaded_tensor_count: usize,
}
impl ModelTensorLoader<'_> {
    /// Load a tensor from the model.
    pub fn load(&mut self, name: &str) -> Result<ggml::Tensor, TensorLoadError> {
        let (mut tensor, info) = self.tensor_loader.load(name)?;

        if let Some(lora_adapters) = &mut self.lora_adapters {
            for lora_adapter in lora_adapters {
                lora_adapter.patch(name, info, &mut tensor)?;
                (self.progress_callback)(LoadProgress::LoraApplied {
                    name,
                    source: &lora_adapter.path,
                });
            }
        }

        self.loaded_tensor_count += 1;
        (self.progress_callback)(LoadProgress::TensorLoaded {
            current_tensor: self.loaded_tensor_count,
            tensor_count: self.tensor_loader.gguf.tensor_infos.len(),
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
