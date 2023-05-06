//! Implements quantization of weights.

use crate::{Hyperparameters, KnownModel, LoadError, LoadProgress, Loader};
use ggml::format::{SaveError, SaveHandler, TensorLoadInfo, TensorSaveInfo};
use half::f16;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
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
        /// The original size of the model.
        original_size: f32,
        /// The reduced size of the model.
        reduced_size: f32,
        /// The history of the quantization.
        history: Vec<f32>,
    },
}

#[derive(Error, Debug)]
/// Errors encountered during the quantization process.
pub enum QuantizeError<E: std::error::Error + Send + Sync> {
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
    /// An error was encountered while writing model-specific data.
    #[error("an error was encountered while writing model-specific data")]
    WriteError(#[source] E),
}
impl<E: std::error::Error + Send + Sync + 'static> QuantizeError<E> {
    pub(crate) fn from_format_error(value: SaveError<QuantizeError<E>>, path: PathBuf) -> Self {
        match value {
            SaveError::Io(io) => QuantizeError::Io(io),
            SaveError::InvalidIntegerConversion(e) => QuantizeError::InvalidIntegerConversion(e),
            SaveError::ImplementationError(e) => e,
            SaveError::InvariantBroken(invariant) => {
                QuantizeError::InvariantBroken { path, invariant }
            }
        }
    }
}

/// Quantizes a model.
pub fn quantize<M: KnownModel>(
    path_in: &Path,
    path_out: &Path,
    desired_type: ggml::Type,
    progress_callback: impl Fn(QuantizeProgress),
) -> Result<(), QuantizeError<<M::Hyperparameters as Hyperparameters>::WriteError>> {
    // Sanity check
    if !matches!(desired_type, ggml::Type::Q4_0 | ggml::Type::Q4_1) {
        return Err(QuantizeError::InvalidQuantizationTarget {
            element_type: desired_type,
        });
    }

    // Load the model
    let progress_callback = Arc::new(progress_callback);

    let mut file_in = File::open(path_in).map_err(|e| LoadError::OpenFileFailed {
        source: e,
        path: path_in.to_owned(),
    })?;
    let mut reader = BufReader::new(&file_in);
    let mut loader = Loader::<M::Hyperparameters, _>::new({
        let progress_callback = progress_callback.clone();
        move |p| {
            if let LoadProgress::HyperparametersLoaded = p {
                progress_callback(QuantizeProgress::HyperparametersLoaded)
            }
        }
    });
    ggml::format::load(&mut reader, &mut loader)
        .map_err(|err| LoadError::from_format_error(err, path_in.to_owned()))?;

    // Save the quantized model, quantizing as we go
    let Loader {
        hyperparameters,
        vocabulary,
        tensors,
        ..
    } = loader;

    let vocabulary = vocabulary
        .id_to_token
        .iter()
        .cloned()
        .zip(vocabulary.id_to_token_score)
        .collect::<Vec<_>>();

    let mut writer = BufWriter::new(File::create(path_out)?);
    let mut saver = QuantizeSaver::new(
        desired_type,
        &hyperparameters,
        &tensors,
        &mut file_in,
        |p| progress_callback(p),
    );
    ggml::format::save(
        &mut writer,
        &mut saver,
        &vocabulary,
        &tensors.keys().cloned().collect::<Vec<_>>(),
    )
    .map_err(|err| QuantizeError::from_format_error(err, path_out.to_owned()))?;

    // Final report
    let sum_all: i64 = saver.history_all.iter().sum();
    progress_callback(QuantizeProgress::Finished {
        original_size: saver.total_size_original as f32 / 1024.0 / 1024.0,
        reduced_size: saver.total_size_new as f32 / 1024.0 / 1024.0,
        history: saver
            .history_all
            .iter()
            .map(|hist| *hist as f32 / sum_all as f32)
            .collect(),
    });

    Ok(())
}

struct QuantizeSaver<'a, F: Fn(QuantizeProgress), H: Hyperparameters> {
    // Input
    quantization_type: ggml::Type,
    hyperparameters: &'a H,
    tensors: &'a HashMap<String, TensorLoadInfo>,
    source_file: &'a mut File,
    progress_callback: F,

    // Output
    total_size_original: usize,
    total_size_new: usize,
    history_all: Vec<i64>,
}
impl<'a, F: Fn(QuantizeProgress), H: Hyperparameters> QuantizeSaver<'a, F, H> {
    fn new(
        quantization_type: ggml::Type,
        hyperparameters: &'a H,
        tensors: &'a HashMap<String, TensorLoadInfo>,
        source_file: &'a mut File,
        progress_callback: F,
    ) -> Self {
        Self {
            quantization_type,
            hyperparameters,
            tensors,
            source_file,
            progress_callback,

            total_size_original: 0,
            total_size_new: 0,
            history_all: vec![0; 16],
        }
    }
}
impl<F: Fn(QuantizeProgress), H: Hyperparameters> SaveHandler<QuantizeError<H::WriteError>>
    for QuantizeSaver<'_, F, H>
{
    fn write_hyperparameters(
        &mut self,
        writer: &mut dyn Write,
    ) -> Result<(), QuantizeError<H::WriteError>> {
        self.hyperparameters
            .write(writer)
            .map_err(QuantizeError::WriteError)?;
        Ok(())
    }

    fn tensor_data(
        &mut self,
        tensor_name: &str,
    ) -> Result<TensorSaveInfo, QuantizeError<H::WriteError>> {
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
        let raw_data = tensor.read_data(&mut BufReader::new(&mut self.source_file))?;

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
