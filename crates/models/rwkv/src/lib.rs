// Ref: https://github.com/saharNooby/rwkv.cpp/blob/5eb8f09/rwkv.cpp

use std::path::Path;

use ggml::Tensor;
use llm_base::{
    util, BasicWriteError, EvaluateOutputRequest, FileType, InferenceParameters, InferenceSession,
    InferenceSessionParameters, KnownModel, LoadError, LoadProgress, Mmap, TokenId, Vocabulary,
};

pub struct Rwkv {
    hyperparameters: Hyperparameters,
    n_context_tokens: usize,

    vocabulary: Vocabulary,

    emb: Tensor,

    ln0_weight: Tensor,
    ln0_bias: Tensor,

    ln_out_weight: Tensor,
    ln_out_bias: Tensor,

    head: Tensor,

    layers: Vec<Layer>,

    /// Needs to kept alive while the model is alive
    _mmap: Option<Mmap>,

    // Must be kept alive for the model
    _context: ggml::Context,
}

unsafe impl Send for Rwkv {}
unsafe impl Sync for Rwkv {}

impl Rwkv {
    /// Load the model from `path` with `n_context_tokens` context tokens.
    ///
    /// The status of the loading process will be reported through `load_progress_callback`.
    pub fn load(
        path: &Path,
        prefer_mmap: bool,
        n_context_tokens: usize,
        load_progress_callback: impl FnMut(LoadProgress),
    ) -> Result<Rwkv, LoadError> {
        llm_base::load(path, prefer_mmap, n_context_tokens, load_progress_callback)
    }
}

impl KnownModel for Rwkv {
    type Hyperparameters = Hyperparameters;

    fn new<E: std::error::Error>(
        hyperparameters: Self::Hyperparameters,
        n_context_tokens: usize,
        vocabulary: Vocabulary,
        tensor_loader: impl llm_base::TensorLoader<E>,
    ) -> Result<Self, E>
    where
        Self: Sized,
    {
        let n_embd = hyperparameters.n_embd;
        let n_layer = hyperparameters.n_layer;
        let n_vocab = hyperparameters.n_vocab;

        let mut tl = tensor_loader;

        // prepare memory for weights
        let emb = tl.load_from_info("emb.weight")?;
        let ln0_weight = tl.load_from_info("blocks.0.ln0.weight")?;
        let ln0_bias = tl.load_from_info("blocks.0.ln0.bias")?;

        let mut layers = Vec::new();
        for i in 0..n_layer {
            let layer = Layer {
                ln1_weight: tl.load_from_info(&format!("blocks.{i}.ln1.weight"))?,
                ln1_bias: tl.load_from_info(&format!("blocks.{i}.ln1.bias"))?,
                att_time_mix_k: tl.load_from_info(&format!("blocks.{i}.att.time_mix_k"))?,
                att_time_mix_v: tl.load_from_info(&format!("blocks.{i}.att.time_mix_v"))?,
                att_time_mix_r: tl.load_from_info(&format!("blocks.{i}.att.time_mix_r"))?,
                att_time_first: tl.load_from_info(&format!("blocks.{i}.att.time_first"))?,
                att_time_decay: tl.load_from_info(&format!("blocks.{i}.att.time_decay"))?,
                att_key: tl.load_from_info(&format!("blocks.{i}.att.key.weight"))?,
                att_value: tl.load_from_info(&format!("blocks.{i}.att.value.weight"))?,
                att_receptance: tl.load_from_info(&format!("blocks.{i}.att.receptance.weight"))?,
                att_output: tl.load_from_info(&format!("blocks.{i}.att.output.weight"))?,
                ln2_weight: tl.load_from_info(&format!("blocks.{i}.ln2.weight"))?,
                ln2_bias: tl.load_from_info(&format!("blocks.{i}.ln2.bias"))?,
                ffn_time_mix_k: tl.load_from_info(&format!("blocks.{i}.ffn.time_mix_k"))?,
                ffn_time_mix_r: tl.load_from_info(&format!("blocks.{i}.ffn.time_mix_r"))?,
                ffn_key: tl.load_from_info(&format!("blocks.{i}.ffn.key.weight"))?,
                ffn_value: tl.load_from_info(&format!("blocks.{i}.ffn.value.weight"))?,
                ffn_receptance: tl.load_from_info(&format!("blocks.{i}.ffn.receptance.weight"))?,
            };

            layers.push(layer);
        }

        let ln_out_weight = tl.load_from_info("ln_out.weight")?;
        let ln_out_bias = tl.load_from_info("ln_out.bias")?;
        let head = tl.load_from_info("head.weight")?;

        let (_context, _, _mmap) = tl.finish();

        Ok(Rwkv {
            hyperparameters,
            n_context_tokens,
            vocabulary,
            emb,
            ln0_weight,
            ln0_bias,
            ln_out_weight,
            ln_out_bias,
            head,
            layers,
            _mmap,
            _context,
        })
    }

    fn start_session(&self, params: InferenceSessionParameters) -> InferenceSession {
        InferenceSession::new(
            params,
            self.n_context_tokens,
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
        self.n_context_tokens
    }

    fn eot_token_id(&self) -> llm_base::TokenId {
        todo!()
    }
}

/// The hyperparameters of the model.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct Hyperparameters {
    /// n_vocab
    pub n_vocab: usize,
    /// n_embd
    pub n_embd: usize,
    /// n_layer
    pub n_layer: usize,
    /// file_type
    pub file_type: FileType,
}

impl llm_base::Hyperparameters for Hyperparameters {
    type WriteError = BasicWriteError;

    fn read(reader: &mut dyn std::io::BufRead) -> Result<Self, LoadError> {
        Ok(Hyperparameters {
            n_vocab: util::read_i32(reader)?.try_into()?,
            n_embd: util::read_i32(reader)?.try_into()?,
            n_layer: util::read_i32(reader)?.try_into()?,
            file_type: {
                let ftype = util::read_i32(reader)?;
                FileType::try_from(ftype).map_err(|_| LoadError::UnsupportedFileType(ftype))?
            },
        })
    }

    fn write(&self, writer: &mut dyn std::io::Write) -> Result<(), Self::WriteError> {
        util::write_i32(writer, self.n_vocab.try_into()?)?;
        util::write_i32(writer, self.n_embd.try_into()?)?;
        util::write_i32(writer, self.n_layer.try_into()?)?;
        util::write_i32(writer, self.file_type.into())?;
        Ok(())
    }

    fn n_vocabulary(&self) -> usize {
        self.n_vocab
    }
}

struct Layer {
    ln1_weight: Tensor,
    ln1_bias: Tensor,

    // RWKV, also called "attention" by the author.
    att_time_mix_k: Tensor,
    att_time_mix_v: Tensor,
    att_time_mix_r: Tensor,
    att_time_first: Tensor,
    att_time_decay: Tensor,
    att_key: Tensor,
    att_value: Tensor,
    att_receptance: Tensor,
    att_output: Tensor,

    ln2_weight: Tensor,
    ln2_bias: Tensor,

    // FFN.
    ffn_time_mix_k: Tensor,
    ffn_time_mix_r: Tensor,
    ffn_key: Tensor,
    ffn_value: Tensor,
    ffn_receptance: Tensor,
}
