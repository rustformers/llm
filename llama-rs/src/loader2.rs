use ggml_loader::util::*;
use ggml_loader::*;
use memmap2::Mmap;

use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Seek},
    ops::ControlFlow,
    path::{Path, PathBuf},
};

use crate::{
    loader_common::FileType, util, Hyperparameters, LoadError, LoadProgress, Model, TokenId,
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
    mut load_progress_callback: impl FnMut(LoadProgress),
) -> Result<Model, LoadError> {
    let main_path = path.as_ref();

    let paths = util::find_all_model_files(main_path)?;
    if paths.len() != 1 {
        return Err(LoadError::MultipartNotSupported { paths });
    }

    let mut file = File::open(main_path).map_err(|e| LoadError::OpenFileFailed {
        source: e,
        path: main_path.to_owned(),
    })?;
    let mut reader = BufReader::new(&file);

    let path = path.as_ref().to_owned();

    (load_progress_callback)(LoadProgress::PartLoading {
        file: &path,
        current_part: 0,
        total_parts: 1,
    });

    let mut loader = Loader::new(
        path.clone(),
        n_context_tokens,
        prefer_mmap,
        load_progress_callback,
    );
    let use_mmap = loader.mmap_active();

    ggml_loader::load_model_from_reader(&mut reader, &mut loader)
        .map_err(|err| LoadError::from_ggml_loader_error(err, path.clone()))?;

    let Loader {
        hyperparameters,
        vocabulary,
        tensors,
        mut load_progress_callback,
        ..
    } = loader;

    let Hyperparameters { n_embd, n_mult, .. } = hyperparameters;
    let n_ff = ((2 * (4 * n_embd) / 3 + n_mult - 1) / n_mult) * n_mult;

    let ctx_size = tensors
        .values()
        .map(|ti| {
            ggml::Tensor::C_TYPE_SIZE
                + ggml::OBJECT_SIZE
                + if use_mmap { 0 } else { ti.calc_size() }
        })
        .sum::<usize>();
    (load_progress_callback)(LoadProgress::ContextSize { bytes: ctx_size });
    let context = ggml::Context::init(ctx_size, !use_mmap);

    let mmap = if use_mmap {
        let file = File::open(&path)?;
        Some(unsafe { Mmap::map(&file)? })
    } else {
        None
    };

    let model = Model::new_loader2(
        context,
        hyperparameters,
        vocabulary,
        n_ff,
        path.clone(),
        &mut file,
        &tensors,
        mmap,
        |tensor_index| {
            (load_progress_callback)(LoadProgress::PartTensorLoaded {
                file: &path,
                current_tensor: tensor_index,
                tensor_count: tensors.len(),
            });
        },
    )?;

    (load_progress_callback)(LoadProgress::PartLoaded {
        file: &path,
        byte_size: 0,
        tensor_count: tensors.len(),
    });

    Ok(model)
}

struct Loader<F: FnMut(LoadProgress)> {
    // Input
    path: PathBuf,
    n_ctx: usize,
    prefer_mmap: bool,
    load_progress_callback: F,

    // Output
    container_type: ContainerType,
    hyperparameters: Hyperparameters,
    vocabulary: Vocabulary,
    tensors: HashMap<String, TensorInfo>,
}
impl<F: FnMut(LoadProgress)> Loader<F> {
    fn new(path: PathBuf, n_ctx: usize, prefer_mmap: bool, load_progress_callback: F) -> Self {
        Self {
            path,
            n_ctx,
            prefer_mmap,
            load_progress_callback,

            container_type: ContainerType::GGJT,
            hyperparameters: Hyperparameters::default(),
            vocabulary: Vocabulary::default(),
            tensors: HashMap::default(),
        }
    }
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
        self.vocabulary.push_token(id, token, score);

        ControlFlow::Continue(())
    }

    fn tensor_buffer(&mut self, info: TensorInfo) -> ControlFlow<LoadError, TensorDataTreatment> {
        let tensor_name = match String::from_utf8(info.name.clone()) {
            Ok(n) => n,
            Err(err) => return ControlFlow::Break(LoadError::InvalidUtf8(err)),
        };

        self.tensors.insert(tensor_name, info);
        ControlFlow::Continue(TensorDataTreatment::Skip)
    }
}

impl<F: FnMut(LoadProgress)> Loader<F> {
    fn mmap_active(&mut self) -> bool {
        self.prefer_mmap && self.container_type.support_mmap()
    }
}

/// use this to load params for llama model inside [`LoadHandler::load_hyper_parameters`]
fn load_hyperparameters<R: BufRead + Seek>(
    reader: &mut R,
    n_ctx: usize,
) -> Result<(Hyperparameters, PartialHyperparameters), ggml_loader::LoadError<LoadError>> {
    // NOTE: Field order matters! Data is laid out in the file exactly in this order.
    let hparams = Hyperparameters {
        n_vocab: read_i32(reader)?.try_into()?,
        n_embd: read_i32(reader)?.try_into()?,
        n_mult: read_i32(reader)?.try_into()?,
        n_head: read_i32(reader)?.try_into()?,
        n_layer: read_i32(reader)?.try_into()?,
        n_rot: read_i32(reader)?.try_into()?,
        file_type: {
            let ftype = read_i32(reader)?;
            FileType::try_from(ftype).map_err(|_| {
                ggml_loader::LoadError::UserInterrupted(LoadError::UnsupportedFileType(ftype))
            })?
        },
        n_ctx,
    };
    let partial = PartialHyperparameters {
        n_vocab: hparams.n_vocab,
    };
    Ok((hparams, partial))
}
