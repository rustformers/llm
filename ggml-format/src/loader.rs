use std::{
    error::Error,
    io::{BufRead, Seek, SeekFrom},
};

use crate::{
    util::{has_data_left, read_bytes_with_len, read_f32, read_i32, read_u32},
    ContainerType, ElementType,
};

#[derive(Debug, thiserror::Error)]
/// Errors that can occur while loading a model.
pub enum LoadError<E: Error> {
    #[error("invalid file magic number: {0}")]
    /// The file magic number is invalid.
    InvalidMagic(u32),
    #[error("invalid ggml format: format={0:?} version={1}")]
    /// An unsupported format version was found.
    InvalidFormatVersion(ContainerType, u32),
    #[error("non-specific I/O error")]
    /// A non-specific IO error.
    Io(#[from] std::io::Error),
    #[error("could not convert bytes to a UTF-8 string")]
    /// One of the strings encountered was not valid UTF-8.
    InvalidUtf8(#[from] std::string::FromUtf8Error),
    #[error("invalid integer conversion")]
    /// One of the integers encountered could not be converted to a more appropriate type.
    InvalidIntegerConversion(#[from] std::num::TryFromIntError),
    #[error("implementation error")]
    /// An error `E` was returned by the implementation of the loader.
    ImplementationError(#[source] E),
    #[error("unsupported tensor type {ftype} for tensor {tensor_name}")]
    /// One of the tensors encountered had an unsupported data type.
    UnsupportedElementType {
        /// The name of the tensor.
        tensor_name: String,
        /// The format type that was encountered.
        ftype: u32,
    },
    #[error("invariant broken: {0}")]
    /// An invariant was broken.
    InvariantBroken(String),
}

#[derive(Debug, Clone)]
/// Information about a tensor that is read.
pub struct TensorInfo {
    /// The name of the tensor.
    pub name: String,
    /// The number of dimensions in the tensor.
    pub n_dims: usize,
    /// The dimensions of the tensor.
    pub dims: [usize; 2],
    /// The number of elements in the tensor.
    pub n_elements: usize,
    /// The type of the elements in the tensor.
    pub element_type: ElementType,
    /// start of tensor - start of file
    pub start_offset: u64,
}
impl TensorInfo {
    /// Get the dimensions of the tensor.
    pub fn dims(&self) -> &[usize] {
        &self.dims[0..self.n_dims]
    }

    /// Calculate the size of the tensor's values in bytes.
    pub fn calc_size(&self) -> usize {
        data_size(self.element_type, self.dims().iter().product())
    }

    /// Reads the tensor's data from the given reader in an owned fashion.
    ///
    /// The behaviour is undefined if the reader does not correspond to this info.
    ///
    /// Do not use this if loading with `mmap`.
    pub fn read_data<R: BufRead + Seek>(&self, reader: &mut R) -> std::io::Result<Vec<u8>> {
        let n_bytes = self.n_elements * ggml::type_size(self.element_type);
        let mut data = vec![0; n_bytes];
        reader.seek(SeekFrom::Start(self.start_offset))?;
        reader.read_exact(&mut data)?;
        Ok(data)
    }
}

/// Returns the size occupied by a tensor's data in bytes given the element type and number of elements.
pub fn data_size(element_type: ElementType, n_elements: usize) -> usize {
    (ggml::type_size(element_type) * n_elements) / ggml::blck_size(element_type)
}

#[derive(Debug, Clone)]
/// Information present within the hyperparameters that is required to continue loading the model.
pub struct PartialHyperparameters {
    /// The number of vocabulary tokens.
    pub n_vocab: usize,
}

/// A handler for loading a model.
pub trait LoadHandler<E: Error> {
    /// Called when the container type is read.
    fn container_type(&mut self, container_type: ContainerType) -> Result<(), E>;
    /// Called when a vocabulary token is read.
    fn vocabulary_token(&mut self, i: usize, token: Vec<u8>, score: f32) -> Result<(), E>;
    /// Called when the hyperparameters need to be read.
    /// You must read the hyperparameters for your model here.
    fn read_hyperparameters(
        &mut self,
        reader: &mut dyn BufRead,
    ) -> Result<PartialHyperparameters, E>;
    /// Called when a new tensor is found.
    fn tensor_buffer(&mut self, info: TensorInfo) -> Result<(), E>;
}

/// Load a model from a `reader` with the `handler`, which will be called when certain events occur.
pub fn load_model<E: Error, R: BufRead + Seek>(
    reader: &mut R,
    handler: &mut impl LoadHandler<E>,
) -> Result<(), LoadError<E>> {
    // Verify magic
    let container_type: ContainerType = match read_u32(reader)? {
        ggml::FILE_MAGIC_GGMF => ContainerType::Ggmf,
        ggml::FILE_MAGIC_GGJT => ContainerType::Ggjt,
        ggml::FILE_MAGIC_UNVERSIONED => ContainerType::Ggml,
        magic => return Err(LoadError::InvalidMagic(magic)),
    };
    handler
        .container_type(container_type)
        .map_err(LoadError::ImplementationError)?;

    // Load format version
    match container_type {
        ContainerType::Ggmf | ContainerType::Ggjt => {
            let _version: u32 = match read_u32(reader)? {
                ggml::FORMAT_VERSION => ggml::FORMAT_VERSION,
                version => return Err(LoadError::InvalidFormatVersion(container_type, version)),
            };
        }
        ContainerType::Ggml => {}
    }

    // Load hyper params
    let hparams = handler
        .read_hyperparameters(reader)
        .map_err(LoadError::ImplementationError)?;
    let n_vocab = hparams.n_vocab;

    // Load vocabulary
    for i in 0..n_vocab {
        let len = read_u32(reader)?.try_into()?;
        let token = read_bytes_with_len(reader, len)?;
        let token_score = match container_type {
            ContainerType::Ggmf | ContainerType::Ggjt => read_f32(reader)?,
            ContainerType::Ggml => {
                // Legacy model, set empty score
                0.
            }
        };
        handler
            .vocabulary_token(i, token, token_score)
            .map_err(LoadError::ImplementationError)?;
    }

    // Load tensor data
    match container_type {
        ContainerType::Ggmf | ContainerType::Ggml => load_weights(reader, handler, false),
        ContainerType::Ggjt => load_weights(reader, handler, true),
    }
}

/// # Params
///
/// `align`
/// align to 4 bytes before reading tensor weights
fn load_weights<E: Error, R: BufRead + Seek>(
    reader: &mut R,
    handler: &mut impl LoadHandler<E>,
    align: bool,
) -> Result<(), LoadError<E>> {
    while has_data_left(reader)? {
        // load tensor header
        let n_dims: usize = read_i32(reader)?.try_into()?;
        let name_len = read_i32(reader)?;
        let ftype = read_u32(reader)?;

        let mut n_elements: usize = 1;
        let mut dims = [1usize, 1];
        let ne_len = dims.len();
        if n_dims > ne_len {
            return Err(LoadError::InvariantBroken(format!("{n_dims} <= {ne_len}")));
        }

        #[allow(clippy::needless_range_loop)]
        for i in 0..n_dims {
            let dim: usize = read_i32(reader)?.try_into()?;
            dims[i] = dim;
            n_elements *= dim;
        }

        // load tensor name
        let name = String::from_utf8(read_bytes_with_len(reader, name_len.try_into()?)?)?;
        let ftype = ggml::Type::try_from(ftype).map_err(|_| LoadError::UnsupportedElementType {
            tensor_name: name.clone(),
            ftype,
        })?;

        // sanity check
        match ftype {
            ElementType::Q4_0 | ElementType::Q4_1 => {
                if dims[0] % 64 != 0 {
                    return Err(LoadError::InvariantBroken(format!("{dims:?}[0] % 64 == 0")));
                }
            }
            _ => {}
        }

        // load tensor weights
        let offset_curr = reader.stream_position()?;
        let offset_aligned: u64 = if align {
            (offset_curr + 31) & !31
        } else {
            offset_curr
        };

        let tensor_info = TensorInfo {
            name,
            dims,
            n_dims,
            n_elements,
            element_type: ftype,
            start_offset: offset_aligned,
        };
        let n_bytes = tensor_info.calc_size();
        handler
            .tensor_buffer(tensor_info)
            .map_err(LoadError::ImplementationError)?;
        reader.seek(SeekFrom::Start(offset_aligned + n_bytes as u64))?;
    }

    Ok(())
}
