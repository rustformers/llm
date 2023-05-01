//! An implementation of BLOOM (BigScience Large Open-science Open-access Multilingual Language Model)
//! for the `llm` ecosystem.
//!
//! This implementation of BLOOM may not be fully correct. More work may be required.
#![deny(missing_docs)]

use std::path::Path;

use llm_base::{
    util, EvaluateOutputRequest, FileType, InferenceParameters, InferenceSession,
    InferenceSessionParameters, KnownModel, LoadError, LoadProgress, Mmap, TokenId, Vocabulary,
};


// layer for the model
struct Layer {
    // normalization
    ln_1_g: Tensor,
    ln_1_b: Tensor,

    // attention
    c_attn_q_proj_w: Tensor,
    c_attn_k_proj_w: Tensor,
    c_attn_v_proj_w: Tensor,

    c_attn_proj_w: Tensor,
    c_attn_proj_b: Tensor,

    // ff
    c_mlp_fc_w: Tensor,
    c_mlp_fc_b: Tensor,

    c_mlp_proj_w: Tensor,
    c_mlp_proj_b: Tensor,
}

/// The hyperparameters of the model.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct Hyperparameters {
    /// n_vocab
    n_vocab: usize,
    /// n_ctx
    n_ctx: usize,
    /// n_embd
    n_embd: usize,
    /// n_head
    n_head: usize,
    /// n_layer
    n_layer: usize,
    /// n_rot
    n_rot: usize,
    /// f_16
    f_16: usize,
    /// file type
    file_type: FileType,
}

/// The Salesforce Codegen model.
///
/// # Safety
/// This implements [Send] and [Sync] as it is immutable after construction.
pub struct CodeGen {
    hyperparameters: Hyperparameters,
    n_context_tokens: usize,

    vocabulary: Vocabulary,
    tok_embeddings: ggml::Tensor,

    norm: ggml::Tensor,
    norm_b: ggml::Tensor,

    output_norm: ggml::Tensor,
    output_norm_b: ggml::Tensor,

    output: ggml::Tensor,
    layers: Vec<Layer>,

    // Must be kept alive for the model
    _context: ggml::Context,
    _mmap: Option<Mmap>,
}


unsafe impl Send for CodeGen {}
unsafe impl Sync for CodeGen {}

impl KnownModel for CodeGen {
    type Hyperparameters = Hyperparameters;


    fn new<E: std::error::Error>(
        hyperparameters: Self::Hyperparameters,
        n_context_tokens: usize,
        vocabulary: Vocabulary,
        tensor_loader: impl llm_base::TensorLoader<E>,
    ) -> Result<Self, E> {
        todo!()
    }


    fn start_session(&self, params: InferenceSessionParameters) -> InferenceSession {
        InferenceSession::new(
            params,
            self.hyperparameters.n_ctx,
            self.hyperparameters.n_layer,
            self.hyperparameters.n_embd,
            self.hyperparameters.n_vocab,
        )
    }

    fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    fn n_context_tokens(&self) -> usize {
        self.hyperparameters.n_ctx
    }

}

impl CodeGen {
    /// Load the model from `path` with `n_context_tokens` context tokens.
    ///
    /// The status of the loading process will be reported through `load_progress_callback`.
    pub fn load(
        path: &Path,
        prefer_mmap: bool,
        n_context_tokens: usize,
        load_progress_callback: impl FnMut(LoadProgress),
    ) -> Result<CodeGen, LoadError> {
        llm_base::load(path, prefer_mmap, n_context_tokens, load_progress_callback)
    }
}

impl llm_base::Hyperparameters for Hyperparameters {
    type WriteError = llm_base::BasicWriteError;

    fn read(reader: &mut dyn std::io::BufRead) -> Result<Self, LoadError> {
        let hyperparameters = Hyperparameters {
            n_vocab: util::read_i32(reader)?.try_into()?,
            n_ctx: util::read_i32(reader)?.try_into()?,
            n_embd: util::read_i32(reader)?.try_into()?,
            n_head: util::read_i32(reader)?.try_into()?,
            n_layer: util::read_i32(reader)?.try_into()?,
            file_type: {
                let ftype = util::read_i32(reader)?;
                FileType::try_from(ftype).map_err(|_| LoadError::UnsupportedFileType(ftype))?
            },
        };

        let n_vocab = util::read_i32(reader)? as usize;
        if hyperparameters.n_vocab != n_vocab {
            return Err(LoadError::InvariantBroken {
                path: None,
                invariant: format!(
                    "GPT2 model expected n_vocab {} found {}",
                    hyperparameters.n_vocab, n_vocab
                ),
            });
        }

        Ok(hyperparameters)
    }

    fn write(&self, writer: &mut dyn std::io::Write) -> Result<(), Self::WriteError> {
        util::write_i32(writer, self.n_vocab.try_into()?)?;
        util::write_i32(writer, self.n_ctx.try_into()?)?;
        util::write_i32(writer, self.n_embd.try_into()?)?;
        util::write_i32(writer, self.n_head.try_into()?)?;
        util::write_i32(writer, self.n_layer.try_into()?)?;
        util::write_i32(writer, self.file_type.into())?;
        util::write_i32(writer, self.n_vocab.try_into()?)?;

        Ok(())
    }

    fn n_vocabulary(&self) -> usize {
        self.n_vocab
    }
}
