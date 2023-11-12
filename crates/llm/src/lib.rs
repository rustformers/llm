//! This crate provides a unified interface for loading and using
//! Large Language Models (LLMs). The following models are supported:
//!
//! - [BLOOM](llm_bloom)
//! - [GPT-2](llm_gpt2)
//! - [GPT-J](llm_gptj)
//! - [GPT-NeoX](llm_gptneox)
//! - [LLaMA](llm_llama)
//! - [MPT](llm_mpt)
//! - Falcon (currently disabled due to incompleteness)
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
//! let llama = llm::load::<llm::models::Llama>(
//!     // path to GGML file
//!     std::path::Path::new("/path/to/model"),
//!     // llm::TokenizerSource
//!     llm::TokenizerSource::Embedded,
//!     // llm::ModelParameters
//!     Default::default(),
//!     // load progress callback
//!     llm::load_progress_callback_stdout
//! )
//! .unwrap_or_else(|err| panic!("Failed to load model: {err}"));
//!
//! // use the model to generate text from a prompt
//! let mut session = llama.start_session(Default::default());
//! let res = session.infer::<std::convert::Infallible>(
//!     // model to use for text generation
//!     &llama,
//!     // randomness provider
//!     &mut rand::thread_rng(),
//!     // the prompt to use for text generation, as well as other
//!     // inference parameters
//!     &llm::InferenceRequest {
//!         prompt: "Rust is a cool programming language because".into(),
//!         parameters: &llm::InferenceParameters::default(),
//!         play_back_previous_tokens: false,
//!         maximum_token_count: None,
//!     },
//!     // llm::OutputRequest
//!     &mut Default::default(),
//!     // output callback
//!     |r| match r {
//!         llm::InferenceResponse::PromptToken(t) | llm::InferenceResponse::InferredToken(t) => {
//!             print!("{t}");
//!             std::io::stdout().flush().unwrap();
//!
//!             Ok(llm::InferenceFeedback::Continue)
//!         }
//!         _ => Ok(llm::InferenceFeedback::Continue),
//!     }
//! );
//!
//! match res {
//!     Ok(result) => println!("\n\nInference stats:\n{result}"),
//!     Err(err) => println!("\n{err}"),
//! }
//! ```
#![deny(missing_docs)]

use std::{
    error::Error,
    fmt::{Debug, Display},
    path::Path,
    str::FromStr,
};

// Try not to expose too many GGML details here.
// This is the "user-facing" API, and GGML may not always be our backend.
pub use llm_base::{
    conversation_inference_callback, feed_prompt_callback,
    ggml::accelerator::get_accelerator as ggml_get_accelerator,
    ggml::accelerator::Accelerator as GgmlAccelerator, ggml::format as ggml_format,
    ggml::RoPEOverrides, load, load_progress_callback_stdout, quantize, samplers, ElementType,
    FileType, FileTypeFormat, FormatMagic, Hyperparameters, InferenceError, InferenceFeedback,
    InferenceParameters, InferenceRequest, InferenceResponse, InferenceSession,
    InferenceSessionConfig, InferenceSnapshot, InferenceSnapshotRef, InferenceStats,
    InvalidTokenBias, KnownModel, LoadError, LoadProgress, Loader, Model, ModelKVMemoryType,
    ModelParameters, OutputRequest, Prompt, QuantizeError, QuantizeProgress, RewindError,
    SnapshotError, TokenBias, TokenId, TokenUtf8Buffer, TokenizationError, Tokenizer,
    TokenizerSource,
};

use serde::Serialize;

macro_rules! define_models {
    ($(($model_lowercase:ident, $model_lowercase_str:literal, $model_pascalcase:ident, $krate_ident:ident, $display_name:literal)),*) => {
        /// All available models.
        pub mod models {
            $(
                #[cfg(feature = $model_lowercase_str)]
                pub use $krate_ident::{self as $model_lowercase, $model_pascalcase};
            )*
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
        /// All available model architectures.
        pub enum ModelArchitecture {
            $(
                #[cfg(feature = $model_lowercase_str)]
                #[doc = concat!("[", $display_name, "](", stringify!($krate_ident), ")")]
                $model_pascalcase,
            )*
        }

        impl ModelArchitecture {
            /// All available model architectures
            pub const ALL: &[Self] = &[
                $(
                    #[cfg(feature = $model_lowercase_str)]
                    Self::$model_pascalcase,
                )*
            ];
        }

        impl ModelArchitecture {
            /// Use a visitor to dispatch some code based on the model architecture.
            pub fn visit<R>(&self, visitor: &mut impl ModelArchitectureVisitor<R>) -> R {
                match self {
                    $(
                        #[cfg(feature = $model_lowercase_str)]
                        Self::$model_pascalcase => visitor.visit::<models::$model_pascalcase>(),
                    )*
                }
            }
        }

        impl FromStr for ModelArchitecture {
            type Err = UnsupportedModelArchitecture;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                use ModelArchitecture::*;
                match s
                    .to_lowercase()
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .as_str()
                {
                    $(
                        #[cfg(feature = $model_lowercase_str)]
                        $model_lowercase_str => Ok($model_pascalcase),
                    )*

                    _ => Err(UnsupportedModelArchitecture(format!(
                        "{s} is not one of supported model architectures: {:?}", ModelArchitecture::ALL
                    ))),
                }
            }
        }

        impl Display for ModelArchitecture {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $(
                        #[cfg(feature = $model_lowercase_str)]
                        Self::$model_pascalcase => write!(f, $display_name),
                    )*
                }
            }
        }
    };
}

define_models!(
    (bert, "bert", Bert, llm_bert, "Bert"),
    (bloom, "bloom", Bloom, llm_bloom, "BLOOM"),
    (gpt2, "gpt2", Gpt2, llm_gpt2, "GPT-2"),
    (gptj, "gptj", GptJ, llm_gptj, "GPT-J"),
    (gptneox, "gptneox", GptNeoX, llm_gptneox, "GPT-NeoX"),
    (llama, "llama", Llama, llm_llama, "LLaMA"),
    (mpt, "mpt", Mpt, llm_mpt, "MPT"),
    (falcon, "falcon", Falcon, llm_falcon, "Falcon")
);

/// Used to dispatch some code based on the model architecture.
pub trait ModelArchitectureVisitor<R> {
    /// Visit a model architecture.
    fn visit<M: KnownModel + 'static>(&mut self) -> R;
}

/// An unsupported model architecture was specified.
pub struct UnsupportedModelArchitecture(String);
impl Display for UnsupportedModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl Error for UnsupportedModelArchitecture {}
impl Debug for UnsupportedModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("UnsupportedModelArchitecture")
            .field(&self.0)
            .finish()
    }
}

/// A helper function that loads the specified model from disk using an architecture
/// specified at runtime. If no architecture is specified, it will try to infer it
/// from the model's metadata.
///
/// This method returns a [`Box`], which means that the model will have single ownership.
/// If you'd like to share ownership (i.e. to use the model in multiple threads), we
/// suggest using [`Arc::from(Box<T>)`](https://doc.rust-lang.org/std/sync/struct.Arc.html#impl-From%3CBox%3CT,+Global%3E%3E-for-Arc%3CT%3E)
/// to convert the [`Box`] into an [`Arc`](std::sync::Arc) after loading.
pub fn load_dynamic(
    architecture: Option<ModelArchitecture>,
    path: &Path,
    tokenizer_source: TokenizerSource,
    params: ModelParameters,
    load_progress_callback: impl FnMut(LoadProgress),
) -> Result<Box<dyn Model>, LoadError> {
    fn load_model<M: KnownModel + 'static>(
        path: &Path,
        tokenizer_source: TokenizerSource,
        params: ModelParameters,
        load_progress_callback: impl FnMut(LoadProgress),
    ) -> Result<Box<dyn Model>, LoadError> {
        Ok(Box::new(load::<M>(
            path,
            tokenizer_source,
            params,
            load_progress_callback,
        )?))
    }

    let architecture = architecture.ok_or_else(|| LoadError::MissingModelArchitecture {
        path: path.to_owned(),
    })?;

    struct LoadVisitor<'a, F: FnMut(LoadProgress)> {
        path: &'a Path,
        tokenizer_source: TokenizerSource,
        params: ModelParameters,
        load_progress_callback: F,
    }
    impl<'a, F: FnMut(LoadProgress)> ModelArchitectureVisitor<Result<Box<dyn Model>, LoadError>>
        for LoadVisitor<'a, F>
    {
        fn visit<M: KnownModel + 'static>(&mut self) -> Result<Box<dyn Model>, LoadError> {
            load_model::<M>(
                self.path,
                self.tokenizer_source.clone(),
                self.params.clone(),
                &mut self.load_progress_callback,
            )
        }
    }

    architecture.visit(&mut LoadVisitor {
        path,
        tokenizer_source,
        params,
        load_progress_callback,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_architecture_from_str() {
        for arch in ModelArchitecture::ALL {
            assert_eq!(
                arch,
                &arch.to_string().parse::<ModelArchitecture>().unwrap()
            );
        }
    }
}
