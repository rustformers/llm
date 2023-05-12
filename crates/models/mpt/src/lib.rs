//! An implementation of [GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2) for the `llm` ecosystem.
#![deny(missing_docs)]

use ggml::Tensor;
use llm_base::{
    ggml,
    model::{common, HyperparametersWriteError},
    util, FileType, InferenceParameters, InferenceSession, InferenceSessionConfig, KnownModel,
    LoadError, ModelParameters, OutputRequest, TokenId, Vocabulary,
};

/// The  MosaicPretrainedTransformer (MPT) model. Ref: [Mosaic ML](https://huggingface.co/mosaicml)
///
/// # Safety
/// This implements [Send] and [Sync] as it is immutable after construction.
pub struct Mpt {
    hyperparameters: Hyperparameters,
    n_context_tokens: usize,
    vocabulary: Vocabulary,
    ln_f_g: Tensor,
    ln_f_b: Tensor,
    wte: Tensor,
    wpe: Tensor,
    lm_head: Tensor,
    layers: Vec<Layer>,
    inference_params: InferenceParameters,
    _context: ggml::Context,
    _mmap: Option<llm_base::Mmap>,
}
unsafe impl Send for Mpt {}
unsafe impl Sync for Mpt {}

impl KnownModel for Mpt {
    type Hyperparameters = Hyperparameters;

    fn new<E: std::error::Error>(
        hyperparameters: Self::Hyperparameters,
        params: ModelParameters,
        vocabulary: Vocabulary,
        tensor_loader: impl llm_base::TensorLoader<E>,
    ) -> Result<Self, E> {
       todo!("implement this")
    }

    fn start_session(&self, config: InferenceSessionConfig) -> InferenceSession {
        todo!("implement this")
    }

    fn evaluate(
        &self,
        session: &mut InferenceSession,
        params: &InferenceParameters,
        input_tokens: &[TokenId],
        output_request: &mut OutputRequest,
    ) {
        todo!("implement this")
    }

    /// Returns the vocabulary used by this model.
    fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    fn n_context_tokens(&self) -> usize {
        self.n_context_tokens
    }

    fn bot_token_id(&self) -> Option<TokenId> {
        todo!("implement this")
    }

    fn eot_token_id(&self) -> TokenId {
        todo!("implement this")
    }

    fn inference_parameters(&self) -> &InferenceParameters {
        &self.inference_params
    }

}

/// MPT [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct Hyperparameters {
    //for the love of god please rename these to something more descriptive and standardise the names accross the different models 
    /// TODO: document
    d_model: usize,
    /// TODO: document
    max_seq_len: usize,
    /// TODO: document
    n_heads: usize,
    /// TODO: document
    n_layers: usize,
    /// TODO: document
    vocab_size: usize,
    /// TODO: document
    file_type: FileType,
}
impl llm_base::Hyperparameters for Hyperparameters {
    fn read_ggml(reader: &mut dyn std::io::BufRead) -> Result<Self, LoadError> {
        let hyperparameters = Hyperparameters {
            d_model: util::read_i32(reader)?.try_into()?,
            max_seq_len: util::read_i32(reader)?.try_into()?,
            n_heads: util::read_i32(reader)?.try_into()?,
            n_layers: util::read_i32(reader)?.try_into()?,
            vocab_size: util::read_i32(reader)?.try_into()?,
            file_type: {
                let ftype = util::read_i32(reader)?;
                FileType::try_from(ftype).map_err(|_| LoadError::UnsupportedFileType(ftype))?
            },
        };

        Ok(hyperparameters)
    }

    fn write_ggml(&self, writer: &mut dyn std::io::Write) -> Result<(), HyperparametersWriteError> {
        util::write_i32(writer, self.d_model.try_into()?)?;
        util::write_i32(writer, self.max_seq_len.try_into()?)?;
        util::write_i32(writer, self.n_heads.try_into()?)?;
        util::write_i32(writer, self.n_layers.try_into()?)?;
        util::write_i32(writer, self.vocab_size.try_into()?)?;
        util::write_i32(writer, self.file_type.into())?;
        Ok(())
    }

    fn n_vocabulary(&self) -> usize {
        self.vocab_size
    }
}

struct Layer {
    // TODO
}
