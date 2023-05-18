//! Implements quantization of weights.

use crate::{
    model::HyperparametersWriteError, Hyperparameters, KnownModel, LoadError, LoadProgress, Loader,
};
use ggml::format::{SaveError, SaveHandler, TensorLoadInfo, TensorSaveInfo};
use half::f16;
use std::{
    collections::HashMap,
    io::{BufRead, Seek, Write},
    path::PathBuf,
    sync::Arc,
};
use thiserror::Error;

#[derive(Clone, Debug)]

/// Progress of quantization.
pub enum QuantizeProgress<'a> {
    /// Hyperparameters have been loaded.
    HyperparametersLoaded,
    /// A tensor is being loaded.
    TensorLoading {
        /// Name of the tensor.
        name: &'a str,
        /// Size of the tensor.
        dims: [usize; 2],
        /// Type of the tensor.
        element_type: ggml::Type,
        /// Number of elements in the tensor.
        n_elements: usize,
    },
    /// A tensor is being quantized.
    TensorQuantizing {
        /// Name of the tensor.
        name: &'a str,
    },
    /// A tensor has been quantized.
    TensorQuantized {
        /// Name of the tensor.
        name: &'a str,
        /// The original size of the tensor.
        original_size: usize,
        /// The reduced size of the tensor.
        reduced_size: usize,
        /// The history of the quantization.
        history: Vec<f32>,
    },
    /// A tensor has been skipped.
    TensorSkipped {
        /// Name of the tensor.
        name: &'a str,
        /// The original size (in bytes) of the tensor data.
        size: usize,
    },
    /// A model has been quantized.
    Finished {
        /// The original size (in bytes) of the model.
        original_size: usize,
        /// The reduced size (in bytes) of the model.
        reduced_size: usize,
        /// The history of the quantization.
        history: Vec<f32>,
    },
}

#[derive(Error, Debug)]
/// Errors encountered during the quantization process.
pub enum QuantizeError {
    #[error("could not load model")]
    /// There was an error while attempting to load the model.
    Load(#[from] LoadError),
    #[error("non-specific I/O error")]
    /// A non-specific IO error.
    Io(#[from] std::io::Error),
    #[error("could not convert bytes to a UTF-8 string")]
    /// One of the strings encountered was not valid UTF-8.
    InvalidUtf8(#[from] std::string::FromUtf8Error),
    #[error("invalid integer conversion")]
    /// One of the integers encountered could not be converted to a more appropriate type.
    InvalidIntegerConversion(#[from] std::num::TryFromIntError),
    #[error("could not create file {path:?}")]
    /// A file failed to create.
    CreateFileFailed {
        /// The original error.
        source: std::io::Error,
        /// The path that failed.
        path: PathBuf,
    },
    /// An invariant was broken.
    ///
    /// This error is not relevant unless `loader2` is being used.
    #[error("invariant broken: {invariant} in {path:?}")]
    InvariantBroken {
        /// The path that failed.
        path: PathBuf,
        /// The invariant that was broken.
        invariant: String,
    },
    /// Attempted to quantize to an invalid target.
    #[error("invalid quantization target {element_type:?}")]
    InvalidQuantizationTarget {
        /// The quantization target.
        element_type: ggml::Type,
    },
    /// The quantization process encountered an unsupported element type.
    #[error("unsupported element type {element_type:?}")]
    UnsupportedElementType {
        /// The element type.
        element_type: ggml::Type,
    },
    /// An error was encountered while writing the hyperparameters.
    #[error("an error was encountered while writing the hyperparameters")]
    HyperparametersWriteError(#[source] HyperparametersWriteError),
    /// An attempt was made to save a model with a container type that does not
    /// support vocabulary scoring, despite the model having a scored vocabulary.
    #[error("container type does not support vocabulary scoring")]
    VocabularyScoringNotSupported,
}
impl QuantizeError {
    pub(crate) fn from_format_error(value: SaveError<QuantizeError>, path: PathBuf) -> Self {
        match value {
            SaveError::Io(io) => QuantizeError::Io(io),
            SaveError::InvalidIntegerConversion(e) => QuantizeError::InvalidIntegerConversion(e),
            SaveError::ImplementationError(e) => e,
            SaveError::InvariantBroken(invariant) => {
                QuantizeError::InvariantBroken { path, invariant }
            }
            SaveError::VocabularyScoringNotSupported => {
                QuantizeError::VocabularyScoringNotSupported
            }
        }
    }
}

/// Quantizes a model.
pub fn quantize<M: KnownModel, R: BufRead + Seek, W: Write + Seek>(
    reader: &mut R,
    writer: &mut W,
    save_container_type: ggml::format::SaveContainerType,
    desired_type: ggml::Type,
    progress_callback: impl Fn(QuantizeProgress),
) -> Result<(), QuantizeError> {
    // Sanity check
    if !matches!(desired_type, ggml::Type::Q4_0 | ggml::Type::Q4_1) {
        return Err(QuantizeError::InvalidQuantizationTarget {
            element_type: desired_type,
        });
    }

    // Load the model
    let progress_callback = Arc::new(progress_callback);

    let mut loader = Loader::<M::Hyperparameters, _>::new({
        let progress_callback = progress_callback.clone();
        move |p| {
            if let LoadProgress::HyperparametersLoaded = p {
                progress_callback(QuantizeProgress::HyperparametersLoaded)
            }
        }
    });
    ggml::format::load(reader, &mut loader)
        .map_err(|err| LoadError::from_format_error(err, PathBuf::default()))?;

    // Save the quantized model, quantizing as we go
    let Loader {
        mut hyperparameters,
        vocabulary,
        tensors,
        ..
    } = loader;

    if let Some(ft) = hyperparameters.file_type_mut() {
        ft.quantization_version = ggml::QNT_VERSION;
    }

    let vocabulary = vocabulary
        .id_to_token
        .iter()
        .cloned()
        .zip(vocabulary.id_to_token_score)
        .collect::<Vec<_>>();

    let mut saver = QuantizeSaver::new(desired_type, &hyperparameters, &tensors, reader, |p| {
        progress_callback(p)
    });
    ggml::format::save(
        writer,
        &mut saver,
        save_container_type,
        &vocabulary,
        &tensors.keys().cloned().collect::<Vec<_>>(),
    )
    .map_err(|err| QuantizeError::from_format_error(err, PathBuf::default()))?;

    // Final report
    let sum_all: i64 = saver.history_all.iter().sum();
    progress_callback(QuantizeProgress::Finished {
        original_size: saver.total_size_original,
        reduced_size: saver.total_size_new,
        history: saver
            .history_all
            .iter()
            .map(|hist| *hist as f32 / sum_all as f32)
            .collect(),
    });

    Ok(())
}

struct QuantizeSaver<'a, F: Fn(QuantizeProgress), H: Hyperparameters, R: BufRead + Seek> {
    // Input
    quantization_type: ggml::Type,
    hyperparameters: &'a H,
    tensors: &'a HashMap<String, TensorLoadInfo>,
    source_reader: &'a mut R,
    progress_callback: F,

    // Output
    total_size_original: usize,
    total_size_new: usize,
    history_all: Vec<i64>,
}
impl<'a, F: Fn(QuantizeProgress), H: Hyperparameters, R: BufRead + Seek>
    QuantizeSaver<'a, F, H, R>
{
    fn new(
        quantization_type: ggml::Type,
        hyperparameters: &'a H,
        tensors: &'a HashMap<String, TensorLoadInfo>,
        source_reader: &'a mut R,
        progress_callback: F,
    ) -> Self {
        Self {
            quantization_type,
            hyperparameters,
            tensors,
            source_reader,
            progress_callback,

            total_size_original: 0,
            total_size_new: 0,
            history_all: vec![0; 16],
        }
    }
}
impl<F: Fn(QuantizeProgress), H: Hyperparameters, R: BufRead + Seek> SaveHandler<QuantizeError>
    for QuantizeSaver<'_, F, H, R>
{
    fn write_hyperparameters(&mut self, writer: &mut dyn Write) -> Result<(), QuantizeError> {
        self.hyperparameters
            .write_ggml(writer)
            .map_err(QuantizeError::HyperparametersWriteError)?;
        Ok(())
    }

    fn tensor_data(&mut self, tensor_name: &str) -> Result<TensorSaveInfo, QuantizeError> {
        let tensor = self.tensors.get(tensor_name).expect(
            "tensor not found; should be impossible due to handler being populated from loader",
        );

        (self.progress_callback)(QuantizeProgress::TensorLoading {
            name: tensor_name,
            dims: tensor.dims,
            n_elements: tensor.n_elements,
            element_type: tensor.element_type,
        });

        // Quantize only 2D tensors
        let quantize = tensor_name.contains("weight") && tensor.n_dims == 2;
        let raw_data = tensor.read_data(self.source_reader)?;

        if quantize && !matches!(tensor.element_type, ggml::Type::F32 | ggml::Type::F16) {
            return Err(QuantizeError::UnsupportedElementType {
                element_type: tensor.element_type,
            });
        }

        self.total_size_original += raw_data.len();

        let (element_type, data) = if quantize {
            (self.progress_callback)(QuantizeProgress::TensorQuantizing { name: tensor_name });

            let data_f32: Vec<f32> = match tensor.element_type {
                ggml::Type::F32 => raw_data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect(),
                ggml::Type::F16 => raw_data
                    .chunks_exact(2)
                    .map(|chunk| {
                        f16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32()
                    })
                    .collect(),
                _ => unreachable!(),
            };

            let result = match self.quantization_type {
                ggml::Type::Q4_0 => {
                    ggml::quantize_q4_0(&data_f32, tensor.n_elements, tensor.dims[0])
                }
                ggml::Type::Q4_1 => {
                    ggml::quantize_q4_1(&data_f32, tensor.n_elements, tensor.dims[0])
                }
                _ => unreachable!(),
            };
            let new_data = result.output;

            let mut history_new = vec![];
            for (i, val) in result.history.iter().enumerate() {
                self.history_all[i] += val;
                history_new.push(*val as f32 / tensor.n_elements as f32);
            }

            (self.progress_callback)(QuantizeProgress::TensorQuantized {
                name: tensor_name,
                original_size: raw_data.len(),
                reduced_size: new_data.len(),
                history: history_new,
            });

            self.total_size_new += new_data.len();

            (self.quantization_type, new_data)
        } else {
            (self.progress_callback)(QuantizeProgress::TensorSkipped {
                name: tensor_name,
                size: raw_data.len(),
            });
            self.total_size_new += raw_data.len();
            (tensor.element_type, raw_data)
        };

        Ok(TensorSaveInfo {
            n_dims: tensor.n_dims,
            dims: tensor.dims,
            element_type,
            data,
        })
    }
}
