mod ggml;

use std::{
    collections::HashMap,
    fmt::Display,
    io::{BufRead, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use thiserror::Error;

use partial_sort::PartialSort;
use rand::{distributions::WeightedIndex, prelude::Distribution};

#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Hyperparameters {
    n_vocab: i32,
    n_ctx: i32,
    n_embd: i32,
    n_mult: i32,
    n_head: i32,
    n_layer: i32,
    n_rot: i32,
    f16_: i32,
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

pub struct Model {
    hparams: Hyperparameters,

    tok_embeddings: ggml::Tensor,

    norm: ggml::Tensor,
    output: ggml::Tensor,

    layers: Vec<Layer>,

    memory_k: ggml::Tensor,
    memory_v: ggml::Tensor,

    tensors: HashMap<String, ggml::Tensor>,

    // Must be kept alive for the model
    _context: ggml::Context,
}

pub struct InferenceParameters {
    pub n_threads: i32,
    pub n_predict: usize,
    pub n_batch: usize,
    pub repeat_last_n: usize,
    pub top_k: i32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub temp: f32,
}

impl Default for InferenceParameters {
    fn default() -> Self {
        Self {
            n_threads: 8,
            n_predict: 128,
            n_batch: 8,
            repeat_last_n: 64,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.30,
            temp: 0.80,
        }
    }
}

type TokenId = i32;
type Token = String;

#[derive(Default)]
pub struct Vocabulary {
    /// Maps every integer (index) token id to its corresponding string
    mapping: Vec<Token>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OutputToken<'a> {
    Token(&'a str),
    EndOfText,
}
impl Display for OutputToken<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                OutputToken::Token(t) => *t,
                OutputToken::EndOfText => "[end of text]",
            }
        )
    }
}

fn llama_n_parts(size: i32) -> i32 {
    match size {
        4096 => 1,
        5120 => 2,
        6656 => 4,
        8192 => 8,
        _ => unreachable!("Invalid size for N_PARTS"),
    }
}

/// Each variant represents a step within the process of loading the model.
/// These can be used to report progress to the user.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum LoadProgress<'a> {
    HyperParamsLoaded(&'a Hyperparameters),
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

    #[error("invalid magic number for {path:?}")]
    InvalidMagic { path: PathBuf },
    #[error("invalid value {value} for `f16` in hyperparameters")]
    HyperparametersF16Invalid { value: i32 },
    #[error("unknown tensor `{tensor_name}` in {path:?}")]
    UnknownTensor { tensor_name: String, path: PathBuf },
    #[error("the tensor `{tensor_name}` has the wrong size in {path:?}")]
    TensorWrongSize { tensor_name: String, path: PathBuf },
    #[error("invalid ftype {ftype} in {path:?}")]
    InvalidFtype { ftype: i32, path: PathBuf },
}

impl Model {
    pub fn load(
        path: impl AsRef<Path>,
        n_ctx: i32,
        load_progress_callback: impl Fn(LoadProgress),
    ) -> Result<(Model, Vocabulary), LoadError> {
        use std::fs::File;
        use std::io::BufReader;

        let path = path.as_ref();

        let mut reader =
            BufReader::new(File::open(path).map_err(|e| LoadError::OpenFileFailed {
                source: e,
                path: path.to_owned(),
            })?);

        /// Helper function. Reads an int from the buffer and returns it.
        fn read_i32(reader: &mut impl BufRead) -> Result<i32, LoadError> {
            let mut bytes = [0u8; 4];
            reader
                .read_exact(&mut bytes)
                .map_err(|e| LoadError::ReadExactFailed {
                    source: e,
                    bytes: bytes.len(),
                })?;
            Ok(i32::from_le_bytes(bytes))
        }

        /// Helper function. Reads a string from the buffer and returns it.
        fn read_string(reader: &mut BufReader<File>, len: usize) -> Result<String, LoadError> {
            let mut buf = vec![0; len];
            reader
                .read_exact(&mut buf)
                .map_err(|e| LoadError::ReadExactFailed {
                    source: e,
                    bytes: buf.len(),
                })?;
            let s = String::from_utf8(buf)?;
            Ok(s)
        }

        // Verify magic
        {
            let magic = read_i32(&mut reader)?;
            if magic != 0x67676d6c {
                return Err(LoadError::InvalidMagic {
                    path: path.to_owned(),
                });
            }
        }

        // =================
        // Load hyper params
        // =================

        // NOTE: Field order matters! Data is laid out in the file exactly
        // in this order.
        let hparams = Hyperparameters {
            n_vocab: read_i32(&mut reader)?,
            n_ctx,
            n_embd: read_i32(&mut reader)?,
            n_mult: read_i32(&mut reader)?,
            n_head: read_i32(&mut reader)?,
            n_layer: read_i32(&mut reader)?,
            n_rot: read_i32(&mut reader)?,
            f16_: read_i32(&mut reader)?,
        };

        let n_ff =
            ((2 * (4 * hparams.n_embd) / 3 + hparams.n_mult - 1) / hparams.n_mult) * hparams.n_mult;
        let n_parts = llama_n_parts(hparams.n_embd);

        load_progress_callback(LoadProgress::HyperParamsLoaded(&hparams));

        // ===============
        // Load vocabulary
        // ===============
        let mut vocab = Vocabulary::default();
        for i in 0..hparams.n_vocab {
            let len = read_i32(&mut reader)?;
            if let Ok(word) = read_string(&mut reader, len as usize) {
                vocab.mapping.push(word);
            } else {
                load_progress_callback(LoadProgress::BadToken {
                    index: i.try_into()?,
                });
                vocab.mapping.push("�".to_string());
            }
        }

        // for the big tensors, we have the option to store the data in 16-bit
        // floats or quantized in order to save memory and also to speed up the
        // computation
        let wtype = match hparams.f16_ {
            0 => ggml::TYPE_F32,
            1 => ggml::TYPE_F16,
            2 => ggml::TYPE_Q4_0,
            3 => ggml::TYPE_Q4_1,
            invalid => return Err(LoadError::HyperparametersF16Invalid { value: invalid }),
        };

        let n_embd = hparams.n_embd;
        let n_layer = hparams.n_layer;
        let n_ctx = hparams.n_ctx;
        let n_vocab = hparams.n_vocab;

        let ctx_size = {
            // Use 64-bit math to prevent overflow.
            let n_embd = n_embd as u64;
            let n_layer = n_layer as u64;
            let n_ctx = n_ctx as u64;
            let n_vocab = n_vocab as u64;
            let n_ff = n_ff as u64;

            /// NOTE: The original code relies in promotion rules and automatic
            /// cast between int to float. What we do instead is use this macro
            /// to convert every term of the multiplication to f64, which should
            /// have enough precision bits to hold the final value, then cast to
            /// usize. I have observed a discrepancy between the ctx_size found
            /// using this code, and the one in llama.cpp. The number for rust
            /// ends up being slightly lower, but no "out of memory" errors are
            /// reported by ggml.
            macro_rules! mul {
                ($term:expr, $($terms:expr),*) => {
                    (($term as f64) $(* ($terms as f64))*) as u64
                };
            }

            let mut ctx_size: u64 = 0;

            ctx_size += mul!(n_embd, n_vocab, ggml::type_sizef(wtype)); // tok_embeddings

            ctx_size += mul!(n_embd, ggml::type_sizef(ggml::TYPE_F32)); // norm

            ctx_size += mul!(n_embd, n_vocab, ggml::type_sizef(wtype)); // output

            ctx_size += mul!(n_layer, n_embd, ggml::type_sizef(ggml::TYPE_F32)); // attention_norm

            ctx_size += mul!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wq
            ctx_size += mul!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wk
            ctx_size += mul!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wv
            ctx_size += mul!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wo

            ctx_size += mul!(n_layer, n_embd, ggml::type_sizef(ggml::TYPE_F32)); // ffn_norm

            ctx_size += mul!(n_layer, n_ff, n_embd, ggml::type_sizef(wtype)); // w1
            ctx_size += mul!(n_layer, n_ff, n_embd, ggml::type_sizef(wtype)); // w2
            ctx_size += mul!(n_layer, n_ff, n_embd, ggml::type_sizef(wtype)); // w3

            ctx_size += mul!(n_ctx, n_layer, n_embd, ggml::type_sizef(ggml::TYPE_F32)); // memory_k
            ctx_size += mul!(n_ctx, n_layer, n_embd, ggml::type_sizef(ggml::TYPE_F32)); // memory_v

            ctx_size += (5 + 10 * n_layer) * 256; // object overhead

            load_progress_callback(LoadProgress::ContextSize {
                bytes: ctx_size.try_into()?,
            });

            ctx_size
        };

        // Initialize the context
        let context = ggml::Context::init(ctx_size as usize);

        let model = {
            let mut tensors = HashMap::new();

            let tok_embeddings = context.new_tensor_2d(wtype, n_embd, n_vocab);
            let norm = context.new_tensor_1d(ggml::TYPE_F32, n_embd);
            let output = context.new_tensor_2d(wtype, n_embd, n_vocab);

            tensors.insert("tok_embeddings.weight".to_owned(), tok_embeddings.share());
            tensors.insert("norm.weight".to_owned(), norm.share());
            tensors.insert("output.weight".to_owned(), output.share());

            let mut layers = Vec::new();
            for i in 0..n_layer {
                let layer = Layer {
                    attention_norm: context.new_tensor_1d(ggml::TYPE_F32, n_embd),
                    wq: context.new_tensor_2d(wtype, n_embd, n_embd),
                    wk: context.new_tensor_2d(wtype, n_embd, n_embd),
                    wv: context.new_tensor_2d(wtype, n_embd, n_embd),
                    wo: context.new_tensor_2d(wtype, n_embd, n_embd),
                    ffn_norm: context.new_tensor_1d(ggml::TYPE_F32, n_embd),
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

            // key + value memory
            let n_mem = n_layer * n_ctx;
            let n_elements = n_embd * n_mem;

            let memory_k = context.new_tensor_1d(ggml::TYPE_F32, n_elements);
            let memory_v = context.new_tensor_1d(ggml::TYPE_F32, n_elements);

            let memory_size = memory_k.nbytes() + memory_v.nbytes();

            load_progress_callback(LoadProgress::MemorySize {
                bytes: memory_size,
                n_mem: n_mem.try_into()?,
            });

            Model {
                hparams,
                tok_embeddings,
                norm,
                output,
                layers,
                memory_k,
                memory_v,
                tensors,
                _context: context,
            }
        };

        // Close the file, but keep its offset. That way we know how to skip the
        // metadata when loading the parts.
        let file_offset = reader.stream_position()?;
        drop(reader);

        for i in 0..n_parts {
            let part_id = i;

            let part_path = if i > 0 {
                let mut path = path.to_owned();
                let mut filename = path.components().last().unwrap().as_os_str().to_owned();
                filename.push(&format!(".{i}"));
                path.pop();
                path.join(filename)
            } else {
                path.to_path_buf()
            };

            load_progress_callback(LoadProgress::PartLoading {
                file: &part_path,
                current_part: (i + 1).try_into()?,
                total_parts: n_parts.try_into()?,
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

                let n_dims = read_i32(&mut part_reader)?;
                let length = read_i32(&mut part_reader)?;
                let ftype = read_i32(&mut part_reader)?;

                let mut nelements = 1;
                let mut ne = [1i32, 1i32];
                for i in 0..n_dims {
                    ne[i as usize] = read_i32(&mut part_reader)?;
                    nelements *= ne[i as usize];
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
                    if tensor.get_ne()[0] / n_parts != ne[0] || tensor.get_ne()[1] != ne[1] {
                        return Err(LoadError::TensorWrongSize {
                            tensor_name,
                            path: part_path,
                        });
                    }
                } else if tensor.get_ne()[0] != ne[0] || tensor.get_ne()[1] / n_parts != ne[1] {
                    return Err(LoadError::TensorWrongSize {
                        tensor_name,
                        path: part_path,
                    });
                }

                let bpe = match ftype {
                    0 => ggml::type_size(ggml::TYPE_F32),
                    1 => ggml::type_size(ggml::TYPE_F16),
                    2 => {
                        assert_eq!(ne[0] % 64, 0);
                        ggml::type_size(ggml::TYPE_Q4_0)
                    }
                    3 => {
                        assert_eq!(ne[0] % 64, 0);
                        ggml::type_size(ggml::TYPE_Q4_1)
                    }
                    _ => {
                        return Err(LoadError::InvalidFtype {
                            ftype,
                            path: part_path,
                        })
                    }
                };

                if n_dims == 1 || n_parts == 1 {
                    if (nelements as usize * bpe) / ggml::blck_size(tensor.get_type()) as usize
                        != tensor.nbytes()
                    {
                        return Err(LoadError::TensorWrongSize {
                            tensor_name,
                            path: part_path,
                        });
                    }

                    let data = tensor.data();

                    if part_id == 0 {
                        // SAFETY: yolo, same as original code
                        let slice = unsafe {
                            std::slice::from_raw_parts_mut(data as *mut u8, tensor.nbytes())
                        };
                        part_reader.read_exact(slice)?;
                    } else {
                        part_reader.seek(SeekFrom::Current(tensor.nbytes() as i64))?;
                    }

                    total_size += tensor.nbytes();
                } else {
                    if (nelements as usize * bpe) / ggml::blck_size(tensor.get_type()) as usize
                        != tensor.nbytes() / n_parts as usize
                    {
                        return Err(LoadError::TensorWrongSize {
                            tensor_name,
                            path: part_path,
                        });
                    }

                    if split_type == 0 {
                        let np0 = ne[0];
                        let row_size = (tensor.get_ne()[0] / ggml::blck_size(tensor.get_type()))
                            as usize
                            * ggml::type_size(tensor.get_type());

                        assert_eq!(row_size, tensor.get_nb()[1]);

                        for i1 in 0..ne[1] {
                            let offset_row = i1 as usize * row_size;
                            let offset = offset_row
                                + ((part_id * np0) as usize
                                    / ggml::blck_size(tensor.get_type()) as usize)
                                    * ggml::type_size(tensor.get_type());
                            // SAFETY: yolo, same as original code
                            unsafe {
                                let ptr = tensor.data().add(offset);
                                let slice = std::slice::from_raw_parts_mut(
                                    ptr as *mut u8,
                                    row_size / n_parts as usize,
                                );
                                part_reader.read_exact(slice)?;
                            }
                        }
                    } else {
                        let np1 = ne[1];
                        let row_size = (tensor.get_ne()[0] / ggml::blck_size(tensor.get_type()))
                            as usize
                            * ggml::type_size(tensor.get_type());

                        for i1 in 0..ne[1] {
                            let offset_row = (i1 + part_id * np1) as usize * row_size;
                            // SAFETY: yolo, same as original code
                            unsafe {
                                let ptr = tensor.data().add(offset_row);
                                let slice =
                                    std::slice::from_raw_parts_mut(ptr as *mut u8, row_size);
                                part_reader.read_exact(slice)?;
                            }
                        }
                    }

                    total_size += tensor.nbytes() / n_parts as usize
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

    pub fn inference_with_prompt(
        &self,
        vocab: &Vocabulary,
        params: &InferenceParameters,
        prompt: &str,
        rng: &mut impl rand::Rng,
        callback: impl Fn(OutputToken),
    ) {
        let embd_inp = self.tokenize(vocab, prompt, true);
        let mut logits = Vec::new();

        // determine the required inference memory per token:
        let mut mem_per_token = 0;
        self.evaluate(
            params.n_threads,
            0,
            &[0, 1, 2, 3],
            &mut logits,
            &mut mem_per_token,
        );

        let last_n_size = params.repeat_last_n;
        let mut last_n_tokens = vec![0 as TokenId; last_n_size];

        let mut remaining_tokens = usize::min(
            params.n_predict,
            self.hparams.n_ctx as usize - embd_inp.len(),
        );
        let mut input_consumed = 0;

        let mut n_past = 0;
        let mut embd = Vec::new();
        while remaining_tokens > 0 {
            // predict
            if !embd.is_empty() {
                self.evaluate(
                    params.n_threads,
                    n_past,
                    &embd,
                    &mut logits,
                    &mut mem_per_token,
                );
            }

            n_past += embd.len() as i32;
            embd.clear();

            if embd_inp.len() <= input_consumed {
                // out of input, sample next token
                let InferenceParameters {
                    top_k,
                    top_p,
                    repeat_penalty,
                    temp,
                    ..
                } = params;

                let n_vocab = self.hparams.n_vocab;

                let id = self.sample_top_p_top_k(
                    vocab,
                    &logits[logits.len() - n_vocab as usize..],
                    &last_n_tokens,
                    *repeat_penalty as f64,
                    *top_k,
                    *top_p as f64,
                    *temp as f64,
                    rng,
                );

                last_n_tokens.remove(0);
                last_n_tokens.push(id);

                // add it to the context
                embd.push(id);

                // decrement remaining sampling budget
                remaining_tokens -= 1;
            } else {
                // if here, it means we are still processing the input prompt
                while embd_inp.len() > input_consumed {
                    embd.push(embd_inp[input_consumed]);
                    last_n_tokens.remove(0);
                    last_n_tokens.push(embd_inp[input_consumed]);
                    input_consumed += 1;
                    if embd.len() > params.n_batch {
                        break;
                    }
                }
            }

            // display text
            let mut eot = false;
            for &id in &embd {
                let output_token = if id == 2 {
                    eot = true;
                    OutputToken::EndOfText
                } else {
                    OutputToken::Token(&vocab.mapping[id as usize])
                };
                callback(output_token);
            }

            if eot {
                break;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_top_p_top_k(
        &self,
        vocab: &Vocabulary,
        logits: &[f32],
        last_n_tokens: &[TokenId],
        repeat_penalty: f64,
        top_k: i32,
        top_p: f64,
        temp: f64,
        rng: &mut impl rand::Rng,
    ) -> TokenId {
        let n_logits = vocab.mapping.len();
        let mut logits_id = Vec::<(f64, TokenId)>::with_capacity(n_logits);

        {
            let scale = 1.0 / temp;
            for (i, &logit) in logits.iter().enumerate() {
                // repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
                // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main
                if last_n_tokens.contains(&(i as TokenId)) {
                    // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                    if logits[i] < 0.0 {
                        logits_id.push((logit as f64 * scale * repeat_penalty, i as TokenId));
                    } else {
                        logits_id.push((logit as f64 * scale / repeat_penalty, i as TokenId));
                    }
                } else {
                    logits_id.push((logit as f64 * scale, i as TokenId));
                }
            }
        }

        // find the top K tokens
        {
            logits_id.partial_sort(top_k as usize, |a, b| {
                // Sort descending
                b.0.total_cmp(&a.0)
            });
            logits_id.truncate(top_k as usize);
        }

        let maxl = logits_id
            .iter()
            .map(|x| x.0)
            .max_by(f64::total_cmp)
            .unwrap();

        // compute probs for the top K tokens
        let mut probs: Vec<f64> = logits_id
            .iter()
            .copied()
            .map(|(k, _)| (k - maxl).exp())
            .collect();
        let sum: f64 = probs.iter().copied().sum();

        // Normalize the probs
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // Top p sampling
        if top_p < 1.0 {
            let mut cumsum = 0.0;
            for i in 0..probs.len() {
                cumsum += probs[i];
                if cumsum >= top_p {
                    probs.truncate(i + 1);
                    logits_id.truncate(i + 1);
                    break;
                }
            }

            cumsum = 1.0 / cumsum;
            for p in probs.iter_mut() {
                *p *= cumsum;
            }
        }

        let dist = WeightedIndex::new(&probs).expect("WeightedIndex error");
        let idx = dist.sample(rng);

        logits_id[idx].1
    }

    pub fn evaluate(
        &self,
        n_threads: i32,
        n_past: i32,
        embd_inp: &[TokenId],
        embd_w: &mut Vec<f32>,
        mem_per_token: &mut usize,
    ) {
        let n = embd_inp.len();

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

        let mut buf_size = 512 * 1024 * 1024;
        if *mem_per_token > 0 && *mem_per_token * n > buf_size {
            // add 10% to account for ggml object overhead
            buf_size = (1.1f64 * *mem_per_token as f64 * n as f64) as usize;
        };
        let ctx0 = ggml::Context::init(buf_size);

        let mut gf = ggml::ComputationGraph::new(n_threads);

        let embd = ctx0.new_tensor_1d(ggml::TYPE_I32, n as i32);
        unsafe { embd.write_data(bytemuck::cast_slice(embd_inp)) };

        let mut input_layer = ctx0.op_get_rows(&self.tok_embeddings, &embd);

        for il in 0..n_layer as usize {
            let input_self_attention = input_layer.share();
            let mut current: ggml::Tensor;

            // norm
            {
                current = ctx0.op_norm(&input_layer);

                // cur = attention_norm * cur
                current = ctx0.op_mul(
                    &ctx0.op_repeat(&self.layers[il].attention_norm, &current),
                    &current,
                );
            }

            // self-attention
            {
                let q_current = ctx0.op_mul_mat(&self.layers[il].wq, &current);
                let k_current = ctx0.op_mul_mat(&self.layers[il].wk, &current);
                let v_current = ctx0.op_mul_mat(&self.layers[il].wv, &current);

                // store key and value to memory
                if n >= 1 {
                    let k = ctx0.op_view_1d(
                        &self.memory_k,
                        n as i32 * n_embd,
                        (self.memory_k.element_size() * n_embd as usize)
                            * (il * n_ctx as usize + n_past as usize),
                    );

                    let v = ctx0.op_view_1d(
                        &self.memory_v,
                        n as i32 * n_embd,
                        (self.memory_v.element_size() * n_embd as usize)
                            * (il * n_ctx as usize + n_past as usize),
                    );

                    gf.build_forward_expand(&ctx0.op_cpy(&k_current, &k));
                    gf.build_forward_expand(&ctx0.op_cpy(&v_current, &v));
                }

                // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
                let q = ctx0.op_permute(
                    &ctx0.op_rope(
                        &ctx0.op_cpy(
                            &q_current,
                            &ctx0.new_tensor_3d(ggml::TYPE_F32, n_embd / n_head, n_head, n as i32),
                        ),
                        n_past,
                        n_rot,
                        0,
                    ),
                    0,
                    2,
                    1,
                    3,
                );

                // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
                let k = ctx0.op_permute(
                    &ctx0.op_rope(
                        &ctx0.op_reshape_3d(
                            &ctx0.op_view_1d(
                                &self.memory_k,
                                (n_past + n as i32) * n_embd,
                                il * n_ctx as usize
                                    * self.memory_k.element_size()
                                    * n_embd as usize,
                            ),
                            n_embd / n_head,
                            n_head,
                            n_past + n as i32,
                        ),
                        n_past,
                        n_rot,
                        1,
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

                // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
                let v_transposed = ctx0.op_permute(
                    &ctx0.op_reshape_3d(
                        &ctx0.op_view_1d(
                            &self.memory_v,
                            (n_past + n as i32) * n_embd,
                            il * n_ctx as usize * self.memory_v.element_size() * n_embd as usize,
                        ),
                        n_embd / n_head,
                        n_head,
                        n_past + n as i32,
                    ),
                    1,
                    2,
                    0,
                    3,
                );

                // KQV = transpose(V) * KQ_soft_max
                let k_q_v = ctx0.op_mul_mat(&v_transposed, &k_q_soft_max);

                // KQV_merged = KQV.permute(0, 2, 1, 3)
                let k_q_v_merged = ctx0.op_permute(&k_q_v, 0, 2, 1, 3);

                // cur = KQV_merged.contiguous().view(n_embd, N)
                current = ctx0.op_cpy(
                    &k_q_v_merged,
                    &ctx0.new_tensor_2d(ggml::TYPE_F32, n_embd, n as i32),
                );

                // projection (no bias)
                current = ctx0.op_mul_mat(&self.layers[il].wo, &current);
            }

            let input_feed_forward = ctx0.op_add(&current, &input_self_attention);

            // feed-forward network
            {
                // norm
                {
                    current = ctx0.op_norm(&input_feed_forward);

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

        // norm
        {
            input_layer = ctx0.op_norm(&input_layer);

            // inpL = norm*inpL
            input_layer = ctx0.op_mul(&ctx0.op_repeat(&self.norm, &input_layer), &input_layer);
        }

        // lm_head
        {
            input_layer = ctx0.op_mul_mat(&self.output, &input_layer);
        }

        // logits -> probs
        // inpL = ctx0.op_soft_max(&inpL);

        // run the computation
        gf.build_forward_expand(&input_layer);
        ctx0.graph_compute(&mut gf);

        // return result for just the last token
        embd_w.resize(n_vocab as usize, 0.0);
        // SAFETY: yolo
        unsafe {
            input_layer.read_data(
                n_vocab as usize * (n - 1) * std::mem::size_of::<f32>(),
                bytemuck::cast_slice_mut(embd_w),
            )
        };

        if *mem_per_token == 0 {
            *mem_per_token = ctx0.used_mem() / n;
        }
    }

    pub fn tokenize(&self, vocab: &Vocabulary, text: &str, bos: bool) -> Vec<TokenId> {
        let mut res = Vec::new();
        if bos {
            res.push(1 as TokenId); // TODO: replace with vocab.bos
        }

        // Find the longest token that matches the text
        let mut pos = 0;
        loop {
            let mut l = 0;
            let mut t = 0;

            for (tk_id, tk) in vocab.mapping.iter().enumerate() {
                if tk.len() < l {
                    continue;
                }
                if tk.len() > text.len() - pos {
                    continue;
                }
                if text[pos..].starts_with(tk) {
                    l = tk.len();
                    t = tk_id;
                }
            }

            if l == 0 {
                break;
            }

            res.push(t as TokenId);
            pos += l;
        }

        res
    }
}
