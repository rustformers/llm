//! This is an experimental, *incomplete* implementation of a loader based on `ggml_loader`.
//!
//! At the time of writing, it does not successfully load any models.
//!
//! GGML/GGMF fails with an invariant broken error, and GGJT fails with an unexpected state error.
//!
//! It also does not support mmap, but it shouldn't be too hard to add: mmap as is done in `loader`, then populate
//! the tensor from the [TensorInfo].

use ggml_loader::util::*;
use ggml_loader::*;

use std::{
    fs::File,
    io::{BufRead, BufReader, Seek},
    ops::ControlFlow,
    path::{Path, PathBuf},
};

use crate::{
    util::mulf, Hyperparameters, LoadError, LoadProgress, Model, TokenId, UnexpectedState,
    Vocabulary,
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
                LoadError::HyperparametersF16Invalid {
                    ftype: ty.try_into().unwrap(),
                }
            }
            ggml_loader::LoadError::InvariantBroken(invariant) => {
                LoadError::InvariantBroken { path, invariant }
            }
        }
    }
}

pub(crate) fn load(
    path: impl AsRef<Path>,
    n_context_tokens: usize,
    load_progress_callback: impl FnMut(LoadProgress),
) -> Result<Model, LoadError> {
    let main_path = path.as_ref();

    let file = File::open(main_path).map_err(|e| LoadError::OpenFileFailed {
        source: e,
        path: main_path.to_owned(),
    })?;
    let mut reader = BufReader::new(&file);

    let path = path.as_ref().to_owned();
    let mut loader = Loader {
        path: path.clone(),
        state: LoadState::Vocabulary(Vocabulary::default()),
        hyperparameters: Hyperparameters::default(),
        container_type: ContainerType::GGJT,
        load_progress_callback,
        n_ctx: n_context_tokens,
    };

    ggml_loader::load_model_from_reader(&mut reader, &mut loader)
        .map_err(|err| LoadError::from_ggml_loader_error(err, path.clone()))?;

    match loader.state {
        LoadState::Vocabulary(_) => Err(LoadError::UnexpectedState {
            path,
            state: UnexpectedState::Vocabulary,
            context: "Encountered vocabulary state while finalizing model".to_string(),
        }),
        LoadState::Model(model) => Ok(model),
    }
}

enum LoadState {
    Vocabulary(Vocabulary),
    Model(Model),
}
struct Loader<F: FnMut(LoadProgress)> {
    // Context
    path: PathBuf,
    n_ctx: usize,
    load_progress_callback: F,

    // Internal state
    hyperparameters: Hyperparameters,
    container_type: ContainerType,

    state: LoadState,
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

    fn got_container_type(&mut self, model_type: ContainerType) -> ControlFlow<LoadError> {
        self.container_type = model_type;
        ControlFlow::Continue(())
    }

    fn got_vocab_token(&mut self, i: usize, token: Vec<u8>, score: f32) -> ControlFlow<LoadError> {
        let vocab = match &mut self.state {
            LoadState::Vocabulary(v) => v,
            LoadState::Model(_) => {
                return ControlFlow::Break(LoadError::UnexpectedState {
                    path: self.path.clone(),
                    state: UnexpectedState::Model,
                    context: "Encountered model state while loading vocabulary".to_string(),
                })
            }
        };
        vocab.max_token_length = vocab.max_token_length.max(token.len());
        vocab.id_to_token.push(token.clone());
        let id = match TokenId::try_from(i) {
            Ok(id) => id,
            Err(err) => return ControlFlow::Break(LoadError::InvalidIntegerConversion(err)),
        };
        vocab.token_to_id.insert(token, id);
        vocab.id_to_token_score.push(score);

        ControlFlow::Continue(())
    }

    fn load_multipart(&mut self, _reader: &mut BufReader<&File>) -> ControlFlow<LoadError> {
        // TODO: implement multipart loading

        (self.load_progress_callback)(LoadProgress::PartLoading {
            file: &self.path,
            current_part: 0,
            total_parts: 1,
        });

        let vocabulary = match &self.state {
            LoadState::Vocabulary(v) => v.clone(),
            LoadState::Model(_) => {
                return ControlFlow::Break(LoadError::UnexpectedState {
                    path: self.path.clone(),
                    state: UnexpectedState::Model,
                    context: "Encountered model state while transitioning into model state"
                        .to_string(),
                })
            }
        };
        let alloc = !(cfg!(feature = "mmap") && self.container_type == ContainerType::GGJT);

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

        self.state = LoadState::Model(Model::new(
            context,
            self.hyperparameters,
            vocabulary,
            n_ff,
            wtype,
            self.container_type,
        ));
        ControlFlow::Continue(())
    }

    fn tensor_buffer(&mut self, info: TensorInfo) -> ControlFlow<LoadError, Option<&mut [u8]>> {
        let model = match &mut self.state {
            LoadState::Model(m) => m,
            LoadState::Vocabulary(_) => {
                return ControlFlow::Break(LoadError::UnexpectedState {
                    path: self.path.clone(),
                    state: UnexpectedState::Vocabulary,
                    context: "Encountered vocabulary state while populating tensors".to_string(),
                })
            }
        };

        let tensor_name = match String::from_utf8(info.name) {
            Ok(n) => n,
            Err(err) => return ControlFlow::Break(LoadError::InvalidUtf8(err)),
        };

        let tensor = match model.tensors_mut().get_mut(&tensor_name) {
            Some(tensor) => tensor,
            None => {
                return ControlFlow::Break(LoadError::UnknownTensor {
                    path: self.path.clone(),
                    tensor_name,
                })
            }
        };

        let buf: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(tensor.data() as *mut u8, tensor.nbytes()) };

        (self.load_progress_callback)(LoadProgress::PartTensorLoaded {
            file: &self.path,
            // TODO: keep track of tensors loaded
            current_tensor: 0,
            tensor_count: model.tensors_mut().len(),
        });

        ControlFlow::Continue(Some(buf))
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
