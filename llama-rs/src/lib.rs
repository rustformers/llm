#![deny(missing_docs)]
//! LLaMA-rs is a Rust port of the llama.cpp project. This allows running inference for Facebook's LLaMA model on a CPU with good performance using full precision, f16 or 4-bit quantized versions of the model.

use std::{
    collections::HashMap,
    fmt::Display,
    io::{BufRead, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use serde::Deserialize;
use thiserror::Error;

#[cfg(feature = "convert")]
pub mod convert;

mod inference_session;
mod util;
mod vocabulary;

pub use ggml::Type as ElementType;
pub use inference_session::{
    InferenceSession, InferenceSessionParameters, InferenceSnapshot, ModelKVMemoryType,
};
pub use util::TokenUtf8Buffer;
pub use vocabulary::{TokenBias, Vocabulary};

use vocabulary::TokenId;

/// The end of text token.
pub const EOT_TOKEN_ID: TokenId = 2; // Hardcoded (for now?)

/// The hyperparameters of the model.
#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub struct Hyperparameters {
    n_vocab: usize,
    n_ctx: usize,
    n_embd: usize,
    n_mult: usize,
    n_head: usize,
    n_layer: usize,
    n_rot: usize,
    f16_: u32,
}

struct Layer {
    attention_norm: ggml::Tensor,

    wq: ggml::Tensor,
    wk: ggml::Tensor,
    wv: ggml::Tensor,
    wo: ggml::Tensor,

    // normalization
    ffn_norm: ggml::Tensor,

    // ff
    w1: ggml::Tensor,
    w2: ggml::Tensor,
    w3: ggml::Tensor,
}

/// The weights for the LLaMA model. All the mutable state is split into a
/// separate struct `InferenceSession`.
pub struct Model {
    hparams: Hyperparameters,

    tok_embeddings: ggml::Tensor,

    norm: ggml::Tensor,
    output: ggml::Tensor,

    layers: Vec<Layer>,

    tensors: HashMap<String, ggml::Tensor>,

    // Must be kept alive for the model
    _context: ggml::Context,
}

#[derive(Clone, Debug, PartialEq)]
/// The parameters that drive text generation.
pub struct InferenceParameters {
    /// The number of threads to use.
    pub n_threads: usize,
    /// [InferenceSession::feed_prompt] processes the prompt in batches of tokens.
    /// This controls how large an individual batch is.
    pub n_batch: usize,
    ///  Top-K: The top K words by score are kept during sampling.
    pub top_k: usize,
    /// Top-p: The cumulative probability after which no more words are kept for sampling.
    pub top_p: f32,
    /// The penalty for repeating tokens. Higher values make the generation less
    /// likely to get into a loop, but may harm results when repetitive outputs
    /// are desired.
    pub repeat_penalty: f32,
    /// Temperature used for sampling.
    pub temperature: f32,
    /// A list of tokens to bias against in the process of generation.
    pub bias_tokens: TokenBias,
    /// Whether or not previous tokens should be played back in [InferenceSession::inference_with_prompt].
    pub play_back_previous_tokens: bool,
}

impl Default for InferenceParameters {
    fn default() -> Self {
        Self {
            n_threads: 8,
            n_batch: 8,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.30,
            temperature: 0.80,
            bias_tokens: TokenBias::default(),
            play_back_previous_tokens: false,
        }
    }
}

/// Statistics about the inference process.
#[derive(Debug, Clone, Copy)]
pub struct InferenceStats {
    /// How long it took to feed the prompt.
    pub feed_prompt_duration: std::time::Duration,
    /// How many tokens the prompt was.
    pub prompt_tokens: usize,
    /// How long it took to predict new tokens.
    pub predict_duration: std::time::Duration,
    /// The number of predicted tokens.
    pub predict_tokens: usize,
}

impl Default for InferenceStats {
    fn default() -> Self {
        Self {
            feed_prompt_duration: std::time::Duration::from_secs(0),
            prompt_tokens: 0,
            predict_duration: std::time::Duration::from_secs(0),
            predict_tokens: 0,
        }
    }
}

impl Display for InferenceStats {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "feed_prompt_duration: {}ms\nprompt_tokens: {}\npredict_duration: {}ms\npredict_tokens: {}\nper_token_duration: {:.3}ms",
            self.feed_prompt_duration.as_millis(),
            self.prompt_tokens,
            self.predict_duration.as_millis(),
            self.predict_tokens,
            (self.predict_duration.as_millis() as f64) / (self.predict_tokens as f64),
        )
    }
}

/// Each variant represents a step within the process of loading the model.
/// These can be used to report progress to the user.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum LoadProgress<'a> {
    /// The hyperparameters have been loaded from the model.
    HyperparametersLoaded(&'a Hyperparameters),
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

#[derive(Error, Debug)]
/// Errors encountered during the snapshot process.
pub enum SnapshotError {
    /// Arbitrary I/O error.
    #[error("I/O error while reading or writing snapshot")]
    IO(#[from] std::io::Error),
    /// Mismatch between the snapshotted memory and the in-memory memory.
    #[error("could not read snapshot due to size mismatch (self={self_size}, input={input_size})")]
    MemorySizeMismatch {
        /// The size of the session memory in memory.
        self_size: usize,
        /// The size of the session memory in snapshot.
        input_size: usize,
    },
}

#[derive(Error, Debug)]
/// Errors encountered during the inferencep rocess.
pub enum InferenceError {
    #[error("an invalid token was encountered during tokenization")]
    /// During tokenization, one of the produced tokens was invalid / zero.
    TokenizationFailed,
    #[error("the context window is full")]
    /// The context window for the model is full.
    ContextFull,
    #[error("reached end of text")]
    /// The model has produced an end of text token, signalling that it thinks that the text should end here.
    ///
    /// Note that this error *can* be ignored and inference can continue, but the results are not guaranteed to be sensical.
    EndOfText,
    #[error("the user-specified callback returned an error")]
    /// The user-specified callback returned an error.
    UserCallback(Box<dyn std::error::Error>),
}

/// Used in a call to `evaluate` to request information from the transformer.
#[derive(Default, Debug, Clone)]
pub struct EvaluateOutputRequest {
    /// Returns all the logits for the provided batch of tokens.
    /// Output shape is `n_batch * n_vocab`.
    pub all_logits: Option<Vec<f32>>,
    /// Returns the embeddings for the provided batch of tokens
    /// Output shape is `n_batch * n_embd`.
    pub embeddings: Option<Vec<f32>>,
}

/// NOTE: The original code relies in promotion rules and automatic cast between
/// int to float. What we do instead is use this macro to convert every term of
/// the multiplication to f64, which should have enough precision bits to hold
/// the final value, then cast to usize. I have observed a discrepancy between
/// the ctx_size found using this code, and the one in llama.cpp. The number for
/// rust ends up being slightly lower, but no "out of memory" errors are
/// reported by ggml.
macro_rules! mulf {
    ($term:expr, $($terms:expr),*) => {
        usize::try_from((($term as f64) $(* ($terms as f64))*) as u64).unwrap()
    };
}

impl Model {
    /// Load the model from `path` with `n_context_tokens` context tokens.
    ///
    /// The status of the loading process will be reported through `load_progress_callback`.
    pub fn load(
        path: impl AsRef<Path>,
        n_context_tokens: usize,
        mut load_progress_callback: impl FnMut(LoadProgress),
    ) -> Result<(Model, Vocabulary), LoadError> {
        use std::fs::File;
        use std::io::BufReader;

        let main_path = path.as_ref();

        let mut reader =
            BufReader::new(
                File::open(main_path).map_err(|e| LoadError::OpenFileFailed {
                    source: e,
                    path: main_path.to_owned(),
                })?,
            );

        fn read_bytes<const N: usize>(reader: &mut impl BufRead) -> Result<[u8; N], LoadError> {
            let mut bytes = [0u8; N];
            reader
                .read_exact(&mut bytes)
                .map_err(|e| LoadError::ReadExactFailed {
                    source: e,
                    bytes: N,
                })?;
            Ok(bytes)
        }

        fn read_bytes_with_len(
            reader: &mut impl BufRead,
            len: usize,
        ) -> Result<Vec<u8>, LoadError> {
            let mut bytes = vec![0u8; len];
            reader
                .read_exact(&mut bytes)
                .map_err(|e| LoadError::ReadExactFailed {
                    source: e,
                    bytes: len,
                })?;
            Ok(bytes)
        }

        fn read_i32(reader: &mut impl BufRead) -> Result<i32, LoadError> {
            Ok(i32::from_le_bytes(read_bytes::<4>(reader)?))
        }

        fn read_u32(reader: &mut impl BufRead) -> Result<u32, LoadError> {
            Ok(u32::from_le_bytes(read_bytes::<4>(reader)?))
        }

        fn read_f32(reader: &mut impl BufRead) -> Result<f32, LoadError> {
            Ok(f32::from_le_bytes(read_bytes::<4>(reader)?))
        }

        /// Helper function. Reads a string from the buffer and returns it.
        fn read_string(reader: &mut BufReader<File>, len: usize) -> Result<String, LoadError> {
            Ok(String::from_utf8(read_bytes_with_len(reader, len)?)?)
        }

        // Verify magic
        let is_legacy_model: bool = match read_u32(&mut reader)? {
            ggml::FILE_MAGIC => false,
            ggml::FILE_MAGIC_UNVERSIONED => true,
            _ => {
                return Err(LoadError::InvalidMagic {
                    path: main_path.to_owned(),
                })
            }
        };

        // Load format version
        if !is_legacy_model {
            #[allow(unused_variables)]
            let version: u32 = match read_u32(&mut reader)? {
                ggml::FORMAT_VERSION => ggml::FORMAT_VERSION,
                version => return Err(LoadError::InvalidFormatVersion { value: version }),
            };
        }

        // =================
        // Load hyper params
        // =================

        // NOTE: Field order matters! Data is laid out in the file exactly
        // in this order.
        let hparams = Hyperparameters {
            n_vocab: read_i32(&mut reader)?.try_into()?,
            n_ctx: n_context_tokens,
            n_embd: read_i32(&mut reader)?.try_into()?,
            n_mult: read_i32(&mut reader)?.try_into()?,
            n_head: read_i32(&mut reader)?.try_into()?,
            n_layer: read_i32(&mut reader)?.try_into()?,
            n_rot: read_i32(&mut reader)?.try_into()?,
            f16_: read_i32(&mut reader)?.try_into()?,
        };

        let n_ff =
            ((2 * (4 * hparams.n_embd) / 3 + hparams.n_mult - 1) / hparams.n_mult) * hparams.n_mult;

        load_progress_callback(LoadProgress::HyperparametersLoaded(&hparams));

        // ===============
        // Load vocabulary
        // ===============
        let vocab = {
            let mut id_to_token = vec![];
            let mut id_to_token_score = vec![];
            let mut token_to_id = HashMap::new();
            let mut max_token_length = 0;

            for i in 0..hparams.n_vocab {
                let len = read_i32(&mut reader)?;
                let token = read_bytes_with_len(&mut reader, len as usize)?;
                max_token_length = max_token_length.max(token.len());
                id_to_token.push(token.clone());
                token_to_id.insert(token, TokenId::try_from(i)?);

                // Token score, currently unused
                if !is_legacy_model {
                    if let Ok(score) = read_f32(&mut reader) {
                        id_to_token_score.push(score);
                    }
                } else {
                    // Legacy model, set empty score
                    id_to_token_score.push(0.);
                }
            }

            Vocabulary {
                id_to_token,
                id_to_token_score,
                token_to_id,
                max_token_length,
            }
        };

        // for the big tensors, we have the option to store the data in 16-bit
        // floats or quantized in order to save memory and also to speed up the
        // computation
        let wtype = match hparams.f16_ {
            0 => ggml::Type::F32,
            1 => ggml::Type::F16,
            2 => ggml::Type::Q4_0,
            3 => ggml::Type::Q4_1,
            invalid => return Err(LoadError::HyperparametersF16Invalid { ftype: invalid }),
        };

        let n_embd = hparams.n_embd;
        let n_layer = hparams.n_layer;
        let n_vocab = hparams.n_vocab;

        let ctx_size = {
            // Use 64-bit math to prevent overflow.
            let mut ctx_size: usize = 0;

            ctx_size += mulf!(n_embd, n_vocab, ggml::type_sizef(wtype)); // tok_embeddings

            ctx_size += mulf!(n_embd, ggml::type_sizef(ggml::Type::F32)); // norm

            ctx_size += mulf!(n_embd, n_vocab, ggml::type_sizef(wtype)); // output

            ctx_size += mulf!(n_layer, n_embd, ggml::type_sizef(ggml::Type::F32)); // attention_norm

            ctx_size += mulf!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wq
            ctx_size += mulf!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wk
            ctx_size += mulf!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wv
            ctx_size += mulf!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wo

            ctx_size += mulf!(n_layer, n_embd, ggml::type_sizef(ggml::Type::F32)); // ffn_norm

            ctx_size += mulf!(n_layer, n_ff, n_embd, ggml::type_sizef(wtype)); // w1
            ctx_size += mulf!(n_layer, n_ff, n_embd, ggml::type_sizef(wtype)); // w2
            ctx_size += mulf!(n_layer, n_ff, n_embd, ggml::type_sizef(wtype)); // w3

            ctx_size += (5 + 10 * n_layer) * 256; // object overhead

            load_progress_callback(LoadProgress::ContextSize { bytes: ctx_size });

            ctx_size
        };

        // Initialize the context
        let context = ggml::Context::init(ctx_size);

        let model = {
            let mut tensors = HashMap::new();

            let tok_embeddings = context.new_tensor_2d(wtype, n_embd, n_vocab);
            let norm = context.new_tensor_1d(ggml::Type::F32, n_embd);
            let output = context.new_tensor_2d(wtype, n_embd, n_vocab);

            tensors.insert("tok_embeddings.weight".to_owned(), tok_embeddings.share());
            tensors.insert("norm.weight".to_owned(), norm.share());
            tensors.insert("output.weight".to_owned(), output.share());

            let mut layers = Vec::new();
            for i in 0..n_layer {
                let layer = Layer {
                    attention_norm: context.new_tensor_1d(ggml::Type::F32, n_embd),
                    wq: context.new_tensor_2d(wtype, n_embd, n_embd),
                    wk: context.new_tensor_2d(wtype, n_embd, n_embd),
                    wv: context.new_tensor_2d(wtype, n_embd, n_embd),
                    wo: context.new_tensor_2d(wtype, n_embd, n_embd),
                    ffn_norm: context.new_tensor_1d(ggml::Type::F32, n_embd),
                    w1: context.new_tensor_2d(wtype, n_embd, n_ff),
                    w2: context.new_tensor_2d(wtype, n_ff, n_embd),
                    w3: context.new_tensor_2d(wtype, n_embd, n_ff),
                };

                tensors.insert(
                    format!("layers.{i}.attention_norm.weight"),
                    layer.attention_norm.share(),
                );

                tensors.insert(format!("layers.{i}.attention.wq.weight"), layer.wq.share());
                tensors.insert(format!("layers.{i}.attention.wk.weight"), layer.wk.share());
                tensors.insert(format!("layers.{i}.attention.wv.weight"), layer.wv.share());
                tensors.insert(format!("layers.{i}.attention.wo.weight"), layer.wo.share());

                tensors.insert(
                    format!("layers.{i}.ffn_norm.weight"),
                    layer.ffn_norm.share(),
                );

                tensors.insert(
                    format!("layers.{i}.feed_forward.w1.weight"),
                    layer.w1.share(),
                );
                tensors.insert(
                    format!("layers.{i}.feed_forward.w2.weight"),
                    layer.w2.share(),
                );
                tensors.insert(
                    format!("layers.{i}.feed_forward.w3.weight"),
                    layer.w3.share(),
                );

                layers.push(layer);
            }

            Model {
                hparams,
                tok_embeddings,
                norm,
                output,
                layers,
                tensors,
                _context: context,
            }
        };

        // Close the file, but keep its offset. That way we know how to skip the
        // metadata when loading the parts.
        let file_offset = reader.stream_position()?;
        drop(reader);

        let paths = util::find_all_model_files(main_path)?;
        let n_parts = paths.len();

        for (i, part_path) in paths.into_iter().enumerate() {
            let part_id = i;

            load_progress_callback(LoadProgress::PartLoading {
                file: &part_path,
                current_part: i,
                total_parts: n_parts,
            });

            let mut part_reader = BufReader::new(File::open(&part_path)?);

            // Skip metadata
            part_reader.seek(SeekFrom::Start(file_offset))?;

            let mut total_size = 0;
            let mut n_tensors = 0;

            // Load weights
            loop {
                // NOTE: Implementation from #![feature(buf_read_has_data_left)]
                let is_eof = part_reader.fill_buf().map(|b| b.is_empty())?;

                if is_eof {
                    break;
                }

                let n_dims = usize::try_from(read_i32(&mut part_reader)?)?;
                let length = read_i32(&mut part_reader)?;
                let ftype = read_u32(&mut part_reader)?;

                let mut nelements = 1;
                let mut ne = [1i64, 1i64];

                #[allow(clippy::needless_range_loop)]
                for i in 0..n_dims {
                    ne[i] = read_i32(&mut part_reader)? as i64;
                    nelements *= usize::try_from(ne[i])?;
                }

                let tensor_name = read_string(&mut part_reader, length as usize)?;

                let Some(tensor) = model.tensors.get(&tensor_name)
                    else {
                        return Err(LoadError::UnknownTensor { tensor_name, path: part_path });
                    };

                // split_type = 0: split by columns
                // split_type = 1: split by rows
                //
                // split_type = 0:
                // regex:
                //   - tok_embeddings.*
                //   - layers.*.attention.wo.weight
                //   - layers.*.feed_forward.w2.weight

                // split_type = 1:
                // regex:
                //   - output.*
                //   - layers.*.attention.wq.weight
                //   - layers.*.attention.wk.weight
                //   - layers.*.attention.wv.weight
                //   - layers.*.feed_forward.w1.weight
                //   - layers.*.feed_forward.w3.weight
                #[allow(clippy::if_same_then_else)]
                let split_type = if tensor_name.contains("tok_embeddings") {
                    0
                } else if tensor_name.contains("layers") {
                    if tensor_name.contains("attention.wo.weight") {
                        0
                    } else if tensor_name.contains("feed_forward.w2.weight") {
                        0
                    } else {
                        1
                    }
                } else if tensor_name.contains("output") {
                    1
                } else {
                    0
                };

                if n_dims == 1 {
                    if tensor.nelements() != nelements {
                        return Err(LoadError::TensorWrongSize {
                            tensor_name,
                            path: part_path,
                        });
                    }
                } else if tensor.nelements() / n_parts != nelements {
                    return Err(LoadError::TensorWrongSize {
                        tensor_name,
                        path: part_path,
                    });
                }

                if n_dims == 1 {
                    if tensor.get_ne()[0] != ne[0] || tensor.get_ne()[1] != ne[1] {
                        return Err(LoadError::TensorWrongSize {
                            tensor_name,
                            path: part_path,
                        });
                    }
                } else if split_type == 0 {
                    if tensor.get_ne()[0] / i64::try_from(n_parts)? != ne[0]
                        || tensor.get_ne()[1] != ne[1]
                    {
                        return Err(LoadError::TensorWrongSize {
                            tensor_name,
                            path: part_path,
                        });
                    }
                } else if tensor.get_ne()[0] != ne[0]
                    || tensor.get_ne()[1] / i64::try_from(n_parts)? != ne[1]
                {
                    return Err(LoadError::TensorWrongSize {
                        tensor_name,
                        path: part_path,
                    });
                }

                let bpe = match ftype {
                    0 => ggml::type_size(ggml::Type::F32),
                    1 => ggml::type_size(ggml::Type::F16),
                    2 => {
                        assert_eq!(ne[0] % 64, 0);
                        ggml::type_size(ggml::Type::Q4_0)
                    }
                    3 => {
                        assert_eq!(ne[0] % 64, 0);
                        ggml::type_size(ggml::Type::Q4_1)
                    }
                    _ => {
                        return Err(LoadError::InvalidFtype {
                            tensor_name,
                            ftype,
                            path: part_path,
                        })
                    }
                };

                if n_dims == 1 || n_parts == 1 {
                    if (nelements * bpe) / ggml::blck_size(tensor.get_type()) != tensor.nbytes() {
                        return Err(LoadError::TensorWrongSize {
                            tensor_name,
                            path: part_path,
                        });
                    }

                    if part_id == 0 {
                        // SAFETY: yolo, same as original code
                        let slice = unsafe {
                            let data = tensor.data();
                            std::slice::from_raw_parts_mut(data as *mut u8, tensor.nbytes())
                        };
                        part_reader.read_exact(slice)?;
                    } else {
                        part_reader.seek(SeekFrom::Current(tensor.nbytes() as i64))?;
                    }

                    total_size += tensor.nbytes();
                } else {
                    if (nelements * bpe) / ggml::blck_size(tensor.get_type())
                        != tensor.nbytes() / n_parts
                    {
                        return Err(LoadError::TensorWrongSize {
                            tensor_name,
                            path: part_path,
                        });
                    }

                    if split_type == 0 {
                        let np0 = ne[0];
                        let row_size = (usize::try_from(tensor.get_ne()[0])?
                            / ggml::blck_size(tensor.get_type()))
                            * ggml::type_size(tensor.get_type());

                        assert_eq!(row_size, tensor.get_nb()[1]);

                        for i1 in 0..ne[1] {
                            let offset_row = i1 as usize * row_size;
                            let offset = offset_row
                                + ((part_id * np0 as usize) / ggml::blck_size(tensor.get_type()))
                                    * ggml::type_size(tensor.get_type());
                            // SAFETY: yolo, same as original code
                            unsafe {
                                let ptr = tensor.data().add(offset);
                                let slice = std::slice::from_raw_parts_mut(
                                    ptr as *mut u8,
                                    row_size / n_parts,
                                );
                                part_reader.read_exact(slice)?;
                            }
                        }
                    } else {
                        let np1 = ne[1];
                        let row_size = (usize::try_from(tensor.get_ne()[0])?
                            / ggml::blck_size(tensor.get_type()))
                            * ggml::type_size(tensor.get_type());

                        for i1 in 0..ne[1] {
                            let offset_row = (i1 as usize + part_id * np1 as usize) * row_size;
                            // SAFETY: yolo, same as original code
                            unsafe {
                                let ptr = tensor.data().add(offset_row);
                                let slice =
                                    std::slice::from_raw_parts_mut(ptr as *mut u8, row_size);
                                part_reader.read_exact(slice)?;
                            }
                        }
                    }

                    total_size += tensor.nbytes() / n_parts;
                }

                n_tensors += 1;
                load_progress_callback(LoadProgress::PartTensorLoaded {
                    file: &part_path,
                    current_tensor: n_tensors.try_into()?,
                    tensor_count: model.tensors.len(),
                });
            }

            load_progress_callback(LoadProgress::PartLoaded {
                file: &part_path,
                byte_size: total_size,
                tensor_count: n_tensors.try_into()?,
            });
        }

        Ok((model, vocab))
    }

    /// Starts a new `InferenceSession` for this model.
    pub fn start_session(&self, params: InferenceSessionParameters) -> InferenceSession {
        let Hyperparameters {
            n_ctx,
            n_embd,
            n_layer,
            n_vocab,
            ..
        } = self.hparams;

        let ctx_size = {
            let mut ctx_size = 0;
            ctx_size += mulf!(
                n_ctx,
                n_layer,
                n_embd,
                ggml::type_sizef(params.memory_k_type.into())
            ); // memory_k
            ctx_size += mulf!(
                n_ctx,
                n_layer,
                n_embd,
                ggml::type_sizef(params.memory_v_type.into())
            ); // memory_v
            ctx_size += (5 + 10 * n_layer) * 256; // object overhead
            ctx_size
        };

        let session_ctx = ggml::Context::init(ctx_size);

        // Initialize key + value memory tensors
        let n_mem = n_layer * n_ctx;
        let n_elements = n_embd * n_mem;
        let memory_k = session_ctx.new_tensor_1d(params.memory_k_type.into(), n_elements);
        let memory_v = session_ctx.new_tensor_1d(params.memory_v_type.into(), n_elements);

        InferenceSession {
            _session_ctx: session_ctx,
            memory_size: ctx_size,
            params,
            memory_k,
            memory_v,
            n_past: 0,
            mem_per_token: 0,
            tokens: vec![],
            last_logits: vec![0.0; n_vocab],
            scratch: inference_session::scratch_buffers(),
        }
    }

    /// Evaluates the transformer.
    ///
    /// The provided `output_request` struct lets you specify which additional
    /// data you are interested in fetching from the transformer. Setting a
    /// field to a `Some` value will clear and fill the provided vector with
    /// data. The provided vector will be resized to the exact output size.
    pub fn evaluate(
        &self,
        session: &mut InferenceSession,
        params: &InferenceParameters,
        input_tokens: &[TokenId],
        output_request: &mut EvaluateOutputRequest,
    ) {
        let n = input_tokens.len();
        let n_past = session.n_past;
        let n_threads = params.n_threads;

        let memk_elsize = session.memory_k.element_size();
        let memv_elsize = session.memory_v.element_size();

        let Hyperparameters {
            n_vocab,
            n_ctx,
            n_embd,
            n_mult: _,
            n_head,
            n_layer,
            n_rot,
            f16_: _,
        } = self.hparams;

        // For the first run, we need to guess a maximum buffer size so we can measure
        // the actual memory consumption of the temporary ggml context.
        //
        // These numbers are from `llama.cpp`, and could potentially be more efficient.
        let mut buf_size = {
            let buf_size_mb = if n_layer >= 80 {
                1536
            } else if n_layer >= 60 {
                1280
            } else {
                1024
            };
            buf_size_mb * 1024 * 1024
        };
        if session.mem_per_token > 0 && session.mem_per_token * n > buf_size {
            // add 10% to account for ggml object overhead
            buf_size = (1.1f64 * session.mem_per_token as f64 * n as f64) as usize;
        };
        let ctx0 = ggml::Context::init(buf_size);

        let mut gf = ggml::ComputationGraph::new(n_threads);

        let embd = ctx0.new_tensor_1d(ggml::Type::I32, n);
        unsafe { embd.write_data(bytemuck::cast_slice(input_tokens)) };

        let mut input_layer = ctx0.op_get_rows(&self.tok_embeddings, &embd);

        for il in 0..n_layer {
            let input_self_attention = input_layer.share();
            let mut current: ggml::Tensor;

            ctx0.use_scratch(Some(&mut session.scratch[0]));

            // norm
            {
                current = ctx0.op_rms_norm(&input_layer);

                // cur = attention_norm * cur
                current = ctx0.op_mul(
                    &ctx0.op_repeat(&self.layers[il].attention_norm, &current),
                    &current,
                );
            }

            // self-attention
            {
                // compute Q and K and RoPE them
                let q_current = ctx0.op_rope(
                    &ctx0.op_reshape_3d(
                        &ctx0.op_mul_mat(&self.layers[il].wq, &current),
                        n_embd / n_head,
                        n_head,
                        n,
                    ),
                    n_past,
                    n_rot,
                    0,
                );
                let k_current = ctx0.op_rope(
                    &ctx0.op_reshape_3d(
                        &ctx0.op_mul_mat(&self.layers[il].wk, &current),
                        n_embd / n_head,
                        n_head,
                        n,
                    ),
                    n_past,
                    n_rot,
                    0,
                );

                // store key and value to memory
                {
                    // compute the transposed [N, n_embd] V matrix
                    let v_current = ctx0.op_transpose(&ctx0.op_reshape_2d(
                        &ctx0.op_mul_mat(&self.layers[il].wv, &current),
                        n_embd,
                        n,
                    ));

                    let k = ctx0.op_view_1d(
                        &session.memory_k,
                        n * n_embd,
                        (memk_elsize * n_embd) * (il * n_ctx + n_past),
                    );

                    let v = ctx0.op_view_2d(
                        &session.memory_v,
                        n,
                        n_embd,
                        n_ctx * memv_elsize,
                        (il * n_ctx) * memv_elsize * n_embd + n_past * memv_elsize,
                    );

                    // important: storing RoPE-ed version of K in the KV cache!
                    gf.build_forward_expand(&ctx0.op_cpy(&k_current, &k));
                    gf.build_forward_expand(&ctx0.op_cpy(&v_current, &v));
                }

                let q = ctx0.op_permute(&q_current, 0, 2, 1, 3);

                let k = ctx0.op_permute(
                    &ctx0.op_reshape_3d(
                        &ctx0.op_view_1d(
                            &session.memory_k,
                            (n_past + n) * n_embd,
                            il * n_ctx * memk_elsize * n_embd,
                        ),
                        n_embd / n_head,
                        n_head,
                        n_past + n,
                    ),
                    0,
                    2,
                    1,
                    3,
                );

                // K * Q
                let k_q = ctx0.op_mul_mat(&k, &q);

                // KQ_scaled = KQ / sqrt(n_embd/n_head)
                let k_q_scaled = ctx0.op_scale(
                    &k_q,
                    &ctx0.new_f32(1.0 / f32::sqrt(n_embd as f32 / n_head as f32)),
                );

                // KQ_masked = mask_past(KQ_scaled)
                let k_q_masked = ctx0.op_diag_mask_inf(&k_q_scaled, n_past);

                // KQ = soft_max(KQ_masked)
                let k_q_soft_max = ctx0.op_soft_max(&k_q_masked);

                // split cached V into n_head heads
                let v = ctx0.op_view_3d(
                    &session.memory_v,
                    n_past + n,
                    n_embd / n_head,
                    n_head,
                    n_ctx * memv_elsize,
                    n_ctx * memv_elsize * n_embd / n_head,
                    il * n_ctx * memv_elsize * n_embd,
                );

                let k_q_v = ctx0.op_mul_mat(&v, &k_q_soft_max);

                // KQV_merged = KQV.permute(0, 2, 1, 3)
                let k_q_v_merged = ctx0.op_permute(&k_q_v, 0, 2, 1, 3);

                // cur = KQV_merged.contiguous().view(n_embd, N)
                current = ctx0.op_cpy(
                    &k_q_v_merged,
                    &ctx0.new_tensor_2d(ggml::Type::F32, n_embd, n),
                );

                // projection (no bias)
                current = ctx0.op_mul_mat(&self.layers[il].wo, &current);
            }

            ctx0.use_scratch(Some(&mut session.scratch[1]));

            let input_feed_forward = ctx0.op_add(&current, &input_self_attention);

            // feed-forward network
            {
                // norm
                {
                    current = ctx0.op_rms_norm(&input_feed_forward);

                    // cur = ffn_norm*cur
                    current = ctx0.op_mul(
                        &ctx0.op_repeat(&self.layers[il].ffn_norm, &current),
                        &current,
                    );
                }

                let tmp = ctx0.op_mul_mat(&self.layers[il].w3, &current);

                current = ctx0.op_mul_mat(&self.layers[il].w1, &current);

                // SILU activation
                current = ctx0.op_silu(&current);

                current = ctx0.op_mul(&current, &tmp);

                current = ctx0.op_mul_mat(&self.layers[il].w2, &current);
            }

            current = ctx0.op_add(&current, &input_feed_forward);

            // input for next layer
            input_layer = current;
        }

        ctx0.use_scratch(Some(&mut session.scratch[0]));

        // Used at the end to optionally extract the embeddings.
        let embeddings_tensor;

        // norm
        {
            input_layer = ctx0.op_rms_norm(&input_layer);

            // inpL = norm*inpL
            input_layer = ctx0.op_mul(&ctx0.op_repeat(&self.norm, &input_layer), &input_layer);
            embeddings_tensor = input_layer.share();
        }

        // lm_head
        {
            input_layer = ctx0.op_mul_mat(&self.output, &input_layer);
        }

        ctx0.use_scratch(None);

        // logits -> probs
        // inpL = ctx0.op_soft_max(&inpL);

        // run the computation
        gf.build_forward_expand(&input_layer);
        ctx0.graph_compute(&mut gf);

        // return result for just the last token
        // SAFETY: yolo
        assert_eq!(session.last_logits.len(), n_vocab);
        unsafe {
            input_layer.read_data(
                n_vocab * (n - 1) * std::mem::size_of::<f32>(),
                bytemuck::cast_slice_mut(&mut session.last_logits),
            )
        };

        // Extract logits
        if let Some(all_logits) = &mut output_request.all_logits {
            all_logits.resize(n_vocab * n, 0.0);
            // SAFETY: Tensor data can be read (properly aligned, initialized,
            // data will not be mutated or otherwise aliased during the copy),
            // and we're not reading past the end of the tensor data.
            assert_eq!(input_layer.nelements(), n_vocab * n);
            unsafe {
                input_layer.read_data(0, bytemuck::cast_slice_mut(all_logits));
            }
        }

        // Extract embeddings
        if let Some(embeddings) = &mut output_request.embeddings {
            embeddings.resize(n_embd * n, 0.0);
            // SAFETY: Same rationale as for the "Extract logits" section applies.
            assert_eq!(embeddings_tensor.nelements(), n_embd * n);
            unsafe {
                embeddings_tensor.read_data(0, bytemuck::cast_slice_mut(embeddings));
            }
        }

        // Adjust the required memory per token if we didn't know that already
        if session.mem_per_token == 0 {
            session.mem_per_token = ctx0.used_mem() / n;
        }

        // Adjust n_past to new length.
        session.n_past += input_tokens.len();
    }

    /// Hydrates a previously obtained InferenceSnapshot for this model.
    pub fn session_from_snapshot(
        &self,
        snapshot: InferenceSnapshot,
    ) -> Result<InferenceSession, SnapshotError> {
        let mut session = self.start_session(snapshot.session_params);

        if session.memory_k.nbytes() != snapshot.memory_k.len()
            || session.memory_v.nbytes() != snapshot.memory_v.len()
        {
            return Err(SnapshotError::MemorySizeMismatch {
                self_size: session.memory_k.nbytes() + session.memory_v.nbytes(),
                input_size: snapshot.memory_k.len() + snapshot.memory_v.len(),
            });
        }

        // SAFETY: We have exclusive access to Session, which means no one else
        // should be touching the context's memory. We can write to it because
        // we already checked the size.
        unsafe {
            session.memory_k.write_data(&snapshot.memory_k);
            session.memory_v.write_data(&snapshot.memory_v);
        }

        session.n_past = snapshot.npast;
        session.tokens = snapshot.tokens;
        session.last_logits = snapshot.last_logits;

        Ok(session)
    }
}
