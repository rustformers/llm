use ggml_format::{
    util::read_i32, ContainerType, LoadError as FormatLoadError, PartialHyperparameters, TensorInfo,
};
use memmap2::Mmap;

use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use crate::{
    loader_common::FileType, model::TensorLoader, util, Hyperparameters, LoadError, LoadProgress,
    Model, TokenId, Vocabulary,
};

impl LoadError {
    pub(crate) fn from_format_error(value: FormatLoadError<LoadError>, path: PathBuf) -> Self {
        match value {
            FormatLoadError::InvalidMagic(magic) => LoadError::InvalidMagic { path, magic },
            FormatLoadError::InvalidFormatVersion(container_type, version) => {
                LoadError::InvalidFormatVersion {
                    container_type,
                    version,
                }
            }
            FormatLoadError::Io(err) => LoadError::Io(err),
            FormatLoadError::InvalidUtf8(err) => LoadError::InvalidUtf8(err),
            FormatLoadError::InvalidIntegerConversion(err) => {
                LoadError::InvalidIntegerConversion(err)
            }
            FormatLoadError::ImplementationError(err) => err,
            FormatLoadError::UnsupportedElementType { tensor_name, ftype } => {
                LoadError::UnsupportedElementType {
                    path,
                    tensor_name,
                    ftype,
                }
            }
            FormatLoadError::InvariantBroken(invariant) => {
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

    let mut loader = Loader::new(n_context_tokens, load_progress_callback);

    ggml_format::load_model(&mut reader, &mut loader)
        .map_err(|err| LoadError::from_format_error(err, path.clone()))?;

    let Loader {
        hyperparameters,
        vocabulary,
        tensors,
        mut load_progress_callback,
        container_type,
        ..
    } = loader;

    let Hyperparameters { n_embd, n_mult, .. } = hyperparameters;
    let n_ff = ((2 * (4 * n_embd) / 3 + n_mult - 1) / n_mult) * n_mult;

    let use_mmap = prefer_mmap && container_type.support_mmap();

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

pub(crate) struct Loader<F: FnMut(LoadProgress)> {
    // Input
    n_ctx: usize,
    load_progress_callback: F,

    // Output
    pub(crate) container_type: ContainerType,
    pub(crate) hyperparameters: Hyperparameters,
    pub(crate) vocabulary: Vocabulary,
    pub(crate) tensors: HashMap<String, TensorInfo>,
}
impl<F: FnMut(LoadProgress)> Loader<F> {
    pub(crate) fn new(n_ctx: usize, load_progress_callback: F) -> Self {
        Self {
            n_ctx,
            load_progress_callback,

            container_type: ContainerType::Ggjt,
            hyperparameters: Hyperparameters::default(),
            vocabulary: Vocabulary::default(),
            tensors: HashMap::default(),
        }
    }
}
impl<F: FnMut(LoadProgress)> ggml_format::LoadHandler<LoadError> for Loader<F> {
    fn container_type(&mut self, container_type: ContainerType) -> Result<(), LoadError> {
        self.container_type = container_type;
        Ok(())
    }

    fn vocabulary_token(&mut self, i: usize, token: Vec<u8>, score: f32) -> Result<(), LoadError> {
        let id = match TokenId::try_from(i) {
            Ok(id) => id,
            Err(err) => return Err(LoadError::InvalidIntegerConversion(err)),
        };
        self.vocabulary.push_token(id, token, score);

        Ok(())
    }

    fn read_hyperparameters(
        &mut self,
        reader: &mut dyn BufRead,
    ) -> Result<PartialHyperparameters, LoadError> {
        // NOTE: Field order matters! Data is laid out in the file exactly in this order.
        let hyperparameters = Hyperparameters {
            n_vocab: read_i32(reader)?.try_into()?,
            n_embd: read_i32(reader)?.try_into()?,
            n_mult: read_i32(reader)?.try_into()?,
            n_head: read_i32(reader)?.try_into()?,
            n_layer: read_i32(reader)?.try_into()?,
            n_rot: read_i32(reader)?.try_into()?,
            file_type: {
                let ftype = read_i32(reader)?;
                FileType::try_from(ftype).map_err(|_| LoadError::UnsupportedFileType(ftype))?
            },
            n_ctx: self.n_ctx,
        };
        let partial = PartialHyperparameters {
            n_vocab: hyperparameters.n_vocab,
        };
        self.hyperparameters = hyperparameters;
        (self.load_progress_callback)(LoadProgress::HyperparametersLoaded(&self.hyperparameters));

        Ok(partial)
    }

    fn tensor_buffer(&mut self, info: TensorInfo) -> Result<(), LoadError> {
        self.tensors.insert(info.name.clone(), info);
        Ok(())
    }
}
