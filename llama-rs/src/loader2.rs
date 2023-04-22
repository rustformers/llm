use ggml_loader::util::*;
use ggml_loader::*;
use memmap2::Mmap;

use std::{
    fs::File,
    io::{BufRead, BufReader, Seek},
    ops::ControlFlow,
    path::{Path, PathBuf},
};

use crate::{
    util::{self, mulf},
    Hyperparameters, LoadError, LoadProgress, Model, TokenId, Vocabulary,
};

impl LoadError {
    fn from_ggml_loader_error(value: ggml_loader::LoadError<LoadError>, path: PathBuf) -> Self {
        match value {
            ggml_loader::LoadError::InvalidMagic(_magic) => LoadError::InvalidMagic { path },
            ggml_loader::LoadError::InvalidFormatVersion(version) => {
                LoadError::InvalidFormatVersion { version }
            }
            ggml_loader::LoadError::Io(err) => LoadError::Io(err),
            ggml_loader::LoadError::FailedCast(err) => LoadError::InvalidIntegerConversion(err),
            ggml_loader::LoadError::UserInterrupted(err) => err,
            ggml_loader::LoadError::UnsupportedElementType(ty) => {
                LoadError::HyperparametersF16Invalid { ftype: ty }
            }
            ggml_loader::LoadError::InvariantBroken(invariant) => {
                LoadError::InvariantBroken { path, invariant }
            }
        }
    }
}

pub(crate) fn load(
    path: impl AsRef<Path>,
    prefer_mmap: bool,
    n_context_tokens: usize,
    load_progress_callback: impl FnMut(LoadProgress),
) -> Result<Model, LoadError> {
    let main_path = path.as_ref();

    let paths = util::find_all_model_files(main_path)?;
    if paths.len() != 1 {
        return Err(LoadError::MultipartNotSupported { paths });
    }

    let file = File::open(main_path).map_err(|e| LoadError::OpenFileFailed {
        source: e,
        path: main_path.to_owned(),
    })?;
    let mut reader = BufReader::new(&file);

    let path = path.as_ref().to_owned();
    let mut loader = Loader {
        path: path.clone(),
        vocab: Default::default(),
        model: None,
        n_ctx: n_context_tokens,
        load_progress_callback,
        prefer_mmap,

        tensor_accumulator: 0,
        hyperparameters: Hyperparameters::default(),
        container_type: ContainerType::GGJT,
    };

    ggml_loader::load_model_from_reader(&mut reader, &mut loader)
        .map_err(|err| LoadError::from_ggml_loader_error(err, path.clone()))?;

    loader.model.ok_or(LoadError::ModelNotCreated { path })
}

struct Loader<F: FnMut(LoadProgress)> {
    // input data and options
    path: PathBuf,
    n_ctx: usize,
    prefer_mmap: bool,

    // Internal state
    tensor_accumulator: usize,
    container_type: ContainerType,
    hyperparameters: Hyperparameters,
    model: Option<Model>,
    vocab: Vocabulary,
    load_progress_callback: F,
}

impl<F: FnMut(LoadProgress)> ggml_loader::LoadHandler<LoadError, BufReader<&File>> for Loader<F> {
    fn load_hyper_parameters(
        &mut self,
        reader: &mut BufReader<&File>,
    ) -> ControlFlow<LoadError, PartialHyperparameters> {
        let (hyperparameters, partial) = match load_hyperparameters(reader, self.n_ctx) {
            Ok(t) => t,
            Err(err) => {
                return ControlFlow::Break(LoadError::from_ggml_loader_error(
                    err,
                    self.path.clone(),
                ))
            }
        };
        self.hyperparameters = hyperparameters;
        (self.load_progress_callback)(LoadProgress::HyperparametersLoaded(&self.hyperparameters));

        ControlFlow::Continue(partial)
    }

    fn got_container_type(&mut self, t: ContainerType) -> ControlFlow<LoadError> {
        self.container_type = t;
        ControlFlow::Continue(())
    }

    fn got_vocab_token(&mut self, i: usize, token: Vec<u8>, score: f32) -> ControlFlow<LoadError> {
        let id = match TokenId::try_from(i) {
            Ok(id) => id,
            Err(err) => return ControlFlow::Break(LoadError::InvalidIntegerConversion(err)),
        };
        self.vocab.push_token(id, token, score);

        ControlFlow::Continue(())
    }

    fn tensor_buffer(&mut self, info: TensorInfo) -> ControlFlow<LoadError, TensorDataTreatment> {
        let model = match &mut self.model {
            Some(model) => model,
            None => {
                let model = result_to_controlflow(self.create_model(self.vocab.clone()))?;
                self.model.insert(model)
            }
        };

        let tensor_name = match String::from_utf8(info.name) {
            Ok(n) => n,
            Err(err) => return ControlFlow::Break(LoadError::InvalidUtf8(err)),
        };

        let tensor_count = model.tensors_mut().len();

        // to satisfy borrow checker
        macro_rules! get_tensor {
            () => {
                match model.tensors_mut().get_mut(&tensor_name) {
                    Some(tensor) => tensor,
                    None => {
                        return ControlFlow::Break(LoadError::UnknownTensor {
                            path: self.path.clone(),
                            tensor_name,
                        })
                    }
                }
            };
        }

        let ret = match &model.mmap {
            Some(map) => unsafe {
                let ptr = map.as_ptr().offset(info.start_offset as isize);
                let tensor = get_tensor!();
                tensor.set_data(ptr as *mut std::ffi::c_void);
                TensorDataTreatment::SeekPast {
                    n_bytes: tensor.nbytes(),
                }
            },
            None => {
                let tensor = get_tensor!();
                let buf: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(tensor.data() as *mut u8, tensor.nbytes())
                };
                TensorDataTreatment::CopyInto(buf)
            }
        };
        (self.load_progress_callback)(LoadProgress::PartTensorLoaded {
            file: &self.path,
            current_tensor: self.tensor_accumulator,
            tensor_count,
        });
        self.tensor_accumulator += 1;

        ControlFlow::Continue(ret)
    }
}

impl<F: FnMut(LoadProgress)> Loader<F> {
    fn create_model(&mut self, vocabulary: Vocabulary) -> Result<Model, LoadError> {
        (self.load_progress_callback)(LoadProgress::PartLoading {
            file: &self.path,
            current_part: 0,
            total_parts: 1,
        });
        let alloc = !(self.use_mmap());
        let Hyperparameters {
            n_vocab,
            n_embd,
            n_mult,
            n_layer,
            element_type,
            ..
        } = self.hyperparameters;
        let n_ff = ((2 * (4 * n_embd) / 3 + n_mult - 1) / n_mult) * n_mult;
        let wtype = element_type;
        let ctx_size = {
            // Use 64-bit math to prevent overflow.
            let mut ctx_size: usize = (5 + 10 * n_layer) * 256; // object overhead

            if alloc {
                let mut model_size: usize = 0;

                ctx_size += mulf!(n_embd, n_vocab, ggml::type_sizef(wtype)); // tok_embeddings
                ctx_size += mulf!(n_embd, ggml::type_sizef(ggml::Type::F32)); // norm
                ctx_size += mulf!(n_embd, n_vocab, ggml::type_sizef(wtype)); // output

                model_size += mulf!(n_layer, n_embd, ggml::type_sizef(ggml::Type::F32)); // attention_norm

                model_size += mulf!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wq
                model_size += mulf!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wk
                model_size += mulf!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wv
                model_size += mulf!(n_layer, n_embd, n_embd, ggml::type_sizef(wtype)); // wo

                model_size += mulf!(n_layer, n_embd, ggml::type_sizef(ggml::Type::F32)); // ffn_norm

                model_size += mulf!(n_layer, n_ff, n_embd, ggml::type_sizef(wtype)); // w1
                model_size += mulf!(n_layer, n_ff, n_embd, ggml::type_sizef(wtype)); // w2
                model_size += mulf!(n_layer, n_ff, n_embd, ggml::type_sizef(wtype)); // w3

                ctx_size += model_size;
            }

            (self.load_progress_callback)(LoadProgress::ContextSize { bytes: ctx_size });

            ctx_size
        };
        // Initialize the context
        let context = ggml::Context::init(ctx_size, alloc);

        let mmap = if self.use_mmap() {
            let file = File::open(&self.path)?;
            Some(unsafe { Mmap::map(&file)? })
        } else {
            None
        };

        Ok(Model::new(
            context,
            self.hyperparameters,
            vocabulary,
            n_ff,
            wtype,
            self.container_type,
            mmap,
        ))
    }

    fn use_mmap(&mut self) -> bool {
        self.prefer_mmap && self.container_type.support_mmap()
    }
}

/// use this to load params for llama model inside [`LoadHandler::load_hyper_parameters`]
fn load_hyperparameters<T, R: BufRead + Seek>(
    reader: &mut R,
    n_ctx: usize,
) -> Result<(Hyperparameters, PartialHyperparameters), ggml_loader::LoadError<T>> {
    // NOTE: Field order matters! Data is laid out in the file exactly in this order.
    let hparams = Hyperparameters {
        n_vocab: read_i32(reader)?.try_into()?,
        n_embd: read_i32(reader)?.try_into()?,
        n_mult: read_i32(reader)?.try_into()?,
        n_head: read_i32(reader)?.try_into()?,
        n_layer: read_i32(reader)?.try_into()?,
        n_rot: read_i32(reader)?.try_into()?,
        element_type: decode_element_type_res(read_i32(reader)?)?,
        n_ctx,
    };
    let partial = PartialHyperparameters {
        n_vocab: hparams.n_vocab,
    };
    Ok((hparams, partial))
}
