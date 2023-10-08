use std::{fs::File, io::BufReader, path::Path};

use llm_base::{
    ggml::{
        format::gguf::{Gguf, GgufLoadError, MetadataValue, MetadataValueType},
        sys::llama::llama_ftype,
        Context,
    },
    loader::{LoadKnownProgress, Source},
    loader::{MetadataError, TensorLoadError},
    model::HyperparametersReadError,
    ContainerType, FileMagic, KnownModel, LoadKnownError, LoraAdapter, Mmap, Model,
    ModelParameters, Tokenizer, TokenizerLoadError, TokenizerSource,
};
use thiserror::Error;

use tracing::log;

use crate::{ModelArchitecture, ModelArchitectureVisitor, UnsupportedModelArchitecture};

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
    /// The architecture specified in this model is not supported by `llm`.
    #[error("architecture is not supported: {0}")]
    UnsupportedArchitecture(#[from] UnsupportedModelArchitecture),
    /// An error occurred while reading the hyperparameters.
    #[error("{0}")]
    HyperparametersReadError(HyperparametersReadError),
    /// An error occurred while reading the tensors.
    #[error("{0}")]
    TensorLoadError(TensorLoadError),
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
impl From<LoadKnownError> for LoadError {
    fn from(value: LoadKnownError) -> Self {
        match value {
            LoadKnownError::HyperparametersReadError(e) => Self::HyperparametersReadError(e),
            LoadKnownError::TensorLoadError(e) => Self::TensorLoadError(e),
        }
    }
}
impl From<MetadataError> for LoadError {
    fn from(value: MetadataError) -> Self {
        Self::HyperparametersReadError(HyperparametersReadError::MetadataError(value))
    }
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
    mut load_progress_callback: impl FnMut(LoadProgress),
) -> Result<Box<dyn Model>, LoadError> {
    if !path.exists() {
        return Err(LoadError::FileDoesNotExist);
    }

    let file = File::open(path).map_err(|e| LoadError::OpenFileFailed { source: e })?;
    let mut reader = BufReader::new(&file);
    log::trace!("Read model file from {:?}", path);

    let gguf = Gguf::load(&mut reader)?;
    log::trace!("Loaded GGML model from reader");

    let architecture = gguf
        .metadata
        .get_string("general.architecture")?
        .parse::<ModelArchitecture>()?;

    let tokenizer = tokenizer_source.retrieve(&gguf)?;

    let quantization_version = gguf.metadata.get_optional("general.quantization_version");
    log::trace!(
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
    log::trace!("Context size: {:?}", ctx_size);

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

            log::trace!("Loaded LoRA weights");
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

    let model = architecture.visit(LoadVisitor {
        source: &mut reader,
        gguf: &gguf,
        tokenizer,
        context,
        lora_adapters,
        load_progress_callback: &mut load_progress_callback,
        params,
    })?;

    (load_progress_callback)(LoadProgress::Loaded {
        file_size,
        tensor_count: gguf.tensor_infos.len(),
    });

    log::trace!("Loaded model");

    Ok(model)
}

struct LoadVisitor<'a, F: FnMut(LoadProgress)> {
    source: &'a mut dyn Source,
    gguf: &'a Gguf,
    tokenizer: Tokenizer,
    context: Context,
    lora_adapters: Option<Vec<LoraAdapter>>,
    load_progress_callback: F,
    params: ModelParameters,
}
impl<'a, F: FnMut(LoadProgress)> ModelArchitectureVisitor<Result<Box<dyn Model>, LoadError>>
    for LoadVisitor<'a, F>
{
    fn visit<M: KnownModel + 'static>(mut self) -> Result<Box<dyn Model>, LoadError> {
        let model = Box::new(llm_base::load_known_internal::<M>(
            self.source,
            self.gguf,
            self.tokenizer,
            self.context,
            self.lora_adapters,
            &mut |step| {
                (self.load_progress_callback)(match step {
                    LoadKnownProgress::LoraApplied { name, source } => {
                        LoadProgress::LoraApplied { name, source }
                    }
                    LoadKnownProgress::TensorLoaded { current_tensor } => {
                        LoadProgress::TensorLoaded {
                            current_tensor,
                            tensor_count: self.gguf.tensor_infos.len(),
                        }
                    }
                })
            },
            self.params,
        )?);

        Ok(model)
    }
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
