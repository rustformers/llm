//! The `llm` crate provides a unified interface for loading and using
//! Large Language Models (LLMs) such as LLaMA.
//!
//! At present, the only supported backend is GGML, but this is expected to
//! change in the future.
#![deny(missing_docs)]

pub use llm_base::{
    load, quantize, ElementType, FileType, InferenceError, InferenceParameters, InferenceSession,
    InferenceSessionParameters, InferenceSnapshot, InferenceWithPromptParameters, KnownModel,
    LoadError, LoadProgress, Model, ModelKVMemoryType, SnapshotError, TokenBias, TokenId,
    TokenUtf8Buffer, Vocabulary, EOT_TOKEN_ID,
};

/// All available models.
pub mod models {
    #[cfg(feature = "bloom")]
    pub use llm_bloom::{self as bloom, Bloom};
    #[cfg(feature = "gpt2")]
    pub use llm_gpt2::{self as gpt2, Gpt2};
    #[cfg(feature = "llama")]
    pub use llm_llama::{self as llama, Llama};
    #[cfg(feature = "neox")]
    pub use llm_neox::{self as neox, NeoX};
}
