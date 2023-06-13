use ggml::{Context, Tensor};

use crate::{InferenceSession, OutputRequest, TokenId};

/// Common code to prepare a model to evaluate input
pub fn prepare_for_evaluate(
    n_layer: usize,
    session: &InferenceSession,
    input_tokens: &[TokenId],
) -> (Context, Tensor) {
    // For the first run, we need to guess a maximum buffer size so we can measure
    // the actual memory consumption of the temporary ggml context.
    //
    // These numbers are from `llama.cpp`, and could potentially be more efficient.
    let mut buf_size = {
        let buf_size_mb = if n_layer >= 80 {
            1536
        } else if n_layer >= 60 {
            1280
        } else {
            1024
        };
        buf_size_mb * 1024 * 1024
    };

    let n = input_tokens.len();
    if session.mem_per_token > 0 && session.mem_per_token * n > buf_size {
        // add 10% to account for ggml object overhead
        buf_size = (1.1f64 * session.mem_per_token as f64 * n as f64) as usize;
    };
    let ctx0 = ggml::Context::init(buf_size, true, session.config.use_gpu);

    let mut embd = ctx0.new_tensor_1d(ggml::Type::I32, n);
    unsafe { embd.write_data(bytemuck::cast_slice(input_tokens)) };
    ggml::set_name(&embd, "embd");

    (ctx0, embd)
}

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

/// Update an [InferenceSession] after evaluation
pub fn update_session(session: &mut InferenceSession, ctx0: &Context, n_input: usize, n: usize) {
    // Adjust the required memory per token if we didn't know that already
    if session.mem_per_token == 0 {
        session.mem_per_token = ctx0.used_mem() / n;
    }

    // Adjust n_past to new length.
    session.n_past += n_input;
}
