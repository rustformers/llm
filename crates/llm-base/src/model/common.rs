use ggml::Tensor;

use crate::{InferenceSession, OutputRequest};

/// Return result for just the last token
pub fn read_last_token(
    session: &mut InferenceSession,
    input_layer: &Tensor,
    n_vocab: usize,
    n: usize,
) {
    assert_eq!(session.last_logits.len(), n_vocab);
    unsafe {
        input_layer.read_data(
            n_vocab * (n - 1) * std::mem::size_of::<f32>(),
            bytemuck::cast_slice_mut(&mut session.last_logits),
        )
    };
}

/// Extract logits from [OutputRequest] evaluation
pub fn extract_logits(
    output_request: &mut OutputRequest,
    input_layer: &Tensor,
    n_vocab: usize,
    n: usize,
) {
    if let Some(all_logits) = &mut output_request.all_logits {
        all_logits.resize(n_vocab * n, 0.0);
        // SAFETY: Tensor data can be read (properly aligned, initialized,
        // data will not be mutated or otherwise aliased during the copy),
        // and we're not reading past the end of the tensor data.
        assert_eq!(input_layer.nelements(), n_vocab * n);
        unsafe {
            input_layer.read_data(0, bytemuck::cast_slice_mut(all_logits));
        }
    }
}

/// Extract embeddings from [OutputRequest] evaluation
pub fn extract_embeddings(
    output_request: &mut OutputRequest,
    embeddings_tensor: &Tensor,
    n_embd: usize,
    n: usize,
) {
    // Extract embeddings
    if let Some(embeddings) = &mut output_request.embeddings {
        embeddings.resize(n_embd, 0.0);
        // Create a new vector to hold all embeddings
        let mut all_embeddings = vec![0.0; n_embd * n];
        // SAFETY: Same rationale as for the "Extract logits" section applies.
        assert_eq!(embeddings_tensor.nelements(), n_embd * n);
        unsafe {
            embeddings_tensor.read_data(0, bytemuck::cast_slice_mut(&mut all_embeddings));
        }
        embeddings.copy_from_slice(&all_embeddings[n_embd * (n - 1)..]);
    }
}
