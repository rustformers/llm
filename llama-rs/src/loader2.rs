use ggml_format::{
    util::read_i32, ContainerType, PartialHyperparameters, TensorDataTreatment, TensorInfo,
};
use memmap2::Mmap;

use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Read, Seek, SeekFrom},
    ops::ControlFlow,
    path::{Path, PathBuf},
};

use crate::{
    loader_common::FileType, model::TensorLoader, util, Hyperparameters, LoadError, LoadProgress,
    Model, TokenId, Vocabulary,
};

impl LoadError {
    pub(crate) fn from_format_error(
        value: ggml_format::LoadError<LoadError>,
        path: PathBuf,
    ) -> Self {
        match value {
            ggml_format::LoadError::InvalidMagic(_magic) => LoadError::InvalidMagic { path },
            ggml_format::LoadError::InvalidFormatVersion(container_type, version) => {
                LoadError::InvalidFormatVersion {
                    container_type,
                    version,
                }
            }
            ggml_format::LoadError::Io(err) => LoadError::Io(err),
            ggml_format::LoadError::FailedCast(err) => LoadError::InvalidIntegerConversion(err),
            ggml_format::LoadError::UserInterrupted(err) => err,
            ggml_format::LoadError::UnsupportedElementType(ty) => {
                LoadError::HyperparametersF16Invalid { ftype: ty }
            }
            ggml_format::LoadError::InvariantBroken(invariant) => {
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

    let file = File::open(main_path).map_err(|e| LoadError::OpenFileFailed {
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

    ggml_format::load_model_from_reader(&mut reader, &mut loader)
        .map_err(|err| LoadError::from_format_error(err, path.clone()))?;

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

    struct TensorLoader2<'a> {
        path: PathBuf,
        file: File,
        tensors: HashMap<String, TensorInfo>,
        context: ggml::Context,
        mmap: Option<Mmap>,
        load_progress_callback: &'a mut dyn FnMut(LoadProgress),
        loaded_tensors: HashMap<String, ggml::Tensor>,
    }
    impl TensorLoader<LoadError> for TensorLoader2<'_> {
        fn load(&mut self, name: &str, ne: &[usize]) -> Result<ggml::Tensor, LoadError> {
            let info = self
                .tensors
                .get(name)
                .ok_or_else(|| LoadError::UnknownTensor {
                    path: self.path.clone(),
                    tensor_name: name.to_owned(),
                })?;

            let ctx = &self.context;
            let mut tensor = match ne.len() {
                1 => ctx.new_tensor_1d(info.element_type, ne[0]),
                2 => ctx.new_tensor_2d(info.element_type, ne[0], ne[1]),
                3 => ctx.new_tensor_3d(info.element_type, ne[0], ne[1], ne[2]),
                _ => {
                    return Err(LoadError::InvariantBroken {
                        path: self.path.clone(),
                        invariant: format!(
                            "the tensor {name} had an unsupported dimension count: {ne:?}"
                        ),
                    })
                }
            };

            match self.mmap.as_ref() {
                Some(mmap) => unsafe {
                    let ptr = mmap.as_ptr().offset(info.start_offset as isize);
                    tensor.set_data(ptr as *mut std::ffi::c_void);
                },
                None => {
                    let buf: &mut [u8] = unsafe {
                        std::slice::from_raw_parts_mut(tensor.data() as *mut u8, tensor.nbytes())
                    };
                    self.file.seek(SeekFrom::Start(info.start_offset))?;
                    self.file.read_exact(buf)?;
                }
            }

            self.loaded_tensors.insert(name.to_owned(), tensor.share());
            (self.load_progress_callback)(LoadProgress::PartTensorLoaded {
                file: &self.path,
                current_tensor: self.loaded_tensors.len(),
                tensor_count: self.tensors.len(),
            });

            Ok(tensor)
        }

        fn finish(self) -> (ggml::Context, HashMap<String, ggml::Tensor>, Option<Mmap>) {
            (self.context, self.loaded_tensors, self.mmap)
        }
    }

    let tensors_len = tensors.len();
    let tl = TensorLoader2 {
        path: path.clone(),
        file,
        tensors,
        context,
        mmap,
        load_progress_callback: &mut load_progress_callback,
        loaded_tensors: Default::default(),
    };

    let model = Model::new_loader2(hyperparameters, vocabulary, n_ff, tl)?;

    (load_progress_callback)(LoadProgress::PartLoaded {
        file: &path,
        byte_size: 0,
        tensor_count: tensors_len,
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

impl<F: FnMut(LoadProgress)> ggml_format::LoadHandler<LoadError, BufReader<&File>> for Loader<F> {
    fn load_hyper_parameters(
        &mut self,
        reader: &mut BufReader<&File>,
    ) -> ControlFlow<LoadError, PartialHyperparameters> {
        let (hyperparameters, partial) = match load_hyperparameters(reader, self.n_ctx) {
            Ok(t) => t,
            Err(err) => {
                return ControlFlow::Break(LoadError::from_format_error(err, self.path.clone()))
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
) -> Result<(Hyperparameters, PartialHyperparameters), ggml_format::LoadError<LoadError>> {
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
                ggml_format::LoadError::UserInterrupted(LoadError::UnsupportedFileType(ftype))
            })?
        },
        n_ctx,
    };
    let partial = PartialHyperparameters {
        n_vocab: hparams.n_vocab,
    };
    Ok((hparams, partial))
}
