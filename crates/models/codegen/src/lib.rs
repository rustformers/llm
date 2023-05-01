//! An implementation of BLOOM (BigScience Large Open-science Open-access Multilingual Language Model)
//! for the `llm` ecosystem.
//!
//! This implementation of BLOOM may not be fully correct. More work may be required.
#![deny(missing_docs)]

use std::{path::Path, fmt::format};

use ggml::{Tensor, Context};
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

    ln_f_g: Tensor,
    ln_f_b: Tensor,

    wte: Tensor, // position embedding

    lmh_g: Tensor, // language model head
    lmh_b: Tensor, // language model bias

    layers: Vec<Layer>,

    context: Context,
    tensors: HashMap<String, Tensor>
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
        let n_embd = hyperparameters.n_embd;
        let n_layer = hyperparameters.n_layer;
        let n_vocab = hyperparameters.n_vocab;
        let n_ctx = hyperparameters.n_ctx;

        let mut tl = tensor_loader;

        let ln_f_g = tl.load("transformer.ln_f.weight", &[n_embd])?;
        let ln_f_b = tl.load("transformer.lm_f.bias", &[n_embd])?;

        let wte = tl.load("transformer.wte.weight", &[n_embd, n_vocab])?;


        let lmh_g = tl.load("lm_head.weight", &[n_embd. n_vocab])?;
        let lmh_b = tl.load("lm_head.bias", &[n_embd])?;


        let mut layers = Vec::new();
        for i in 0..n_layer {
            let layer = Layer {
                ln_1_g: tl.load(&format!("transformer.h.{i}.ln_1.weight"), &[n_embd])?,
                ln_1_b: tl.load(&format!("transformer.h.{i}.ln_1.bias"), &[n_embd])?,

                c_attn_q_proj_w: tl
                    .load(&format!("transformer.h.{i}.attn.q_proj.weight"), &[n_embd, n_embd * 3])?,

                c_attn_k_proj_w: tl
                    .load(&format!("transformer.h.{i}.attn.k_proj.weight"), &[n_embd, n_embd * 3])?,

                c_attn_v_proj_w: tl
                    .load(&format!("transformer.h.{i}.attn.v_proj.weight"), &[n_embd, n_embd * 3])?,

                c_attn_proj_w: tl.load(&format!("transformer.h.{i}.attn.out_proj.weight"), &[n_embd, n_embd])?,
                c_attn_proj_b: tl.load(&format!("transformer.h.{i}.attn.out_proj.bias"), &[n_embd])?,

                c_mlp_fc_w: tl.load(&format!("transformer.h.{i}.mlp.fc_in.weight"), &[n_embd, n_embd * 4])?,
                c_mlp_fc_b: tl.load(&format!("transformer.h.{i}.mlp.fc_in.bias"), &[n_embd * 4])?,

                c_mlp_proj_w: tl
                    .load(&format!("transformer.h.{i}.mlp.fc_out.weight"), &[n_embd * 4, n_embd])?,
                c_mlp_proj_b: tl.load(&format!("transformer.h.{i}.mlp.fc_out.bias"), &[n_embd])?,
            };

            layers.push(layer);
        }

        let (context, tensors, _) = tl.finish();

        Ok(CodeGen {
            hyperparameters,
            ln_f_g,
            ln_f_b,
            wte,
            lmh_g,
            lmh_b,
            layers,
            context,
            tensors,
        })

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


    fn evaluate(
        &self,
        session: &mut InferenceSession,
        params: &InferenceParameters,
        input_tokens: &[TokenId],
        output_request: &mut EvaluateOutputRequest,
    ) {
        todo!()
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
