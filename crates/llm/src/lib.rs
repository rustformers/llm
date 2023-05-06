//! This crate provides a unified interface for loading and using
//! Large Language Models (LLMs). The following models are supported:
//!
//! - [BLOOM](llm_bloom)
//! - [GPT-2](llm_gpt2)
//! - [GPT-J](llm_gptj)
//! - [LLaMA](llm_llama)
//! - [GPT-NeoX](llm_neox)
//!
//! At present, the only supported backend is [GGML](https://github.com/ggerganov/ggml), but this is expected to
//! change in the future.
//!
//! # Example
//!
//! ```no_run
//! use std::io::Write;
//! use llm::Model;
//!
//! // load a GGML model from disk
//! let model_load = llm::load::<llm::models::Llama>(
//!     // path to GGML file
//!     std::path::Path::new("/path/to/model"),
//!     // llm::ModelParameters
//!     Default::default(),
//!     // load progress callback
//!     llm::load_progress_callback_stdout
//! );
//!    
//! let llama = match model_load {
//!     Ok(model) => model,
//!     Err(e) => panic!("Failed to load model: {e}"),
//! };
//!   
//! // use the model to generate text from a prompt
//! let mut session = llama.start_session(Default::default());
//! let res = session.infer::<std::convert::Infallible>(
//!     // model to use for text generation
//!     &llama,
//!     // text generation prompt
//!     "Rust is a cool programming language because",
//!     // llm::EvaluateOutputRequest
//!     &mut Default::default(),
//!     // randomness provider
//!     &mut rand::thread_rng(),
//!     // output callback
//!     |t| {
//!         print!("{t}");
//!         std::io::stdout().flush().unwrap();
//!   
//!         Ok(())
//!     }
//! );
//!   
//! match res {
//!     Ok(result) => println!("\n\nInference stats:\n{result}"),
//!     Err(err) => println!("\n{err}"),
//! }
//! ```
#![deny(missing_docs)]

// Try not to expose too many GGML details here.
// This is the "user-facing" API, and GGML may not always be our backend.
pub use llm_base::{
    ggml::format as ggml_format, load, load_progress_callback_stdout, quantize, ElementType,
    FileType, InferenceError, InferenceParameters, InferenceSession, InferenceSessionParameters,
    InferenceSnapshot, InferenceWithPromptParameters, KnownModel, LoadError, LoadProgress, Loader,
    Model, ModelKVMemoryType, ModelParameters, SnapshotError, TokenBias, TokenId, TokenUtf8Buffer,
    Vocabulary,
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
}
