//! ggml-loader aux

use ggml_loader::util::*;
use ggml_loader::*;

/// The hyperparameters of the model.
#[derive(Debug, Clone)]
pub struct LlamaHyperparameters {
    pub n_vocab: usize,
    pub n_embd: usize,
    pub n_mult: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub n_rot: usize,
    pub tensor_element_type: ElementType,
}

/// use this to load params for llama model inside [`LoadHandler::load_hyper_parameters`]
#[allow(dead_code)]
pub fn load_llama_hparams<T, R: BufRead + Seek>(
    reader: &mut R,
) -> Result<(LlamaHyperparameters, PartialHyperparameters), LoadError<T>> {
    // NOTE: Field order matters! Data is laid out in the file exactly in this order.
    let hparams = LlamaHyperparameters {
        n_vocab: read_i32(reader)?.try_into()?,
        n_embd: read_i32(reader)?.try_into()?,
        n_mult: read_i32(reader)?.try_into()?,
        n_head: read_i32(reader)?.try_into()?,
        n_layer: read_i32(reader)?.try_into()?,
        n_rot: read_i32(reader)?.try_into()?,
        tensor_element_type: decode_element_type_res(read_i32(reader)?)?,
    };
    let partial = PartialHyperparameters {
        n_vocab: hparams.n_vocab,
    };
    Ok((hparams, partial))
}
