//! The `llm` crate provides a unified interface for loading and using
//! Large Language Models (LLMs) such as LLaMA.
//!
//! At present, the only supported backend is GGML, but this is expected to
//! change in the future.
#![deny(missing_docs)]

// Try not to expose too many GGML details here.
// This is the "user-facing" API, and GGML may not always be our backend.
pub use llm_base::{
    ggml::format as ggml_format, load, load_progress_callback_stdout, quantize, ElementType,
    FileType, InferenceError, InferenceParameters, InferenceSession, InferenceSessionParameters,
    InferenceSnapshot, InferenceWithPromptParameters, KnownModel, LoadError, LoadProgress, Loader,
    Model, ModelKVMemoryType, SnapshotError, TokenBias, TokenId, TokenUtf8Buffer, Vocabulary,
};

/// All available models.
pub mod models {
    #[cfg(feature = "bloom")]
    pub use llm_bloom::{self as bloom, Bloom};
    #[cfg(feature = "gpt2")]
    pub use llm_gpt2::{self as gpt2, Gpt2};
    #[cfg(feature = "gptj")]
    pub use llm_gptj::{self as gptj, GptJ};
    #[cfg(feature = "llama")]
    pub use llm_llama::{self as llama, Llama};
    #[cfg(feature = "neox")]
    pub use llm_neox::{self as neox, NeoX};
    #[cfg(feature = "rwkv")]
    pub use llm_rwkv::{self as rwkv, Rwkv};
}
