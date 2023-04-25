//! standalone model loader
//!
//! Only the hyperparameter is llama-specific. Everything else can be reused for other LLM.
#![allow(clippy::nonminimal_bool)]

pub mod util;

mod loader;

pub use loader::{
    load_model_from_reader, LoadError, LoadHandler, PartialHyperparameters, TensorDataTreatment,
    TensorInfo,
};

pub type ElementType = ggml::Type;

/// the format of the file containing the model
#[derive(Debug, PartialEq, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub enum ContainerType {
    /// legacy format, oldest ggml tensor file format
    GGML,
    /// also legacy format, newer than GGML, older than GGJT
    GGMF,
    /// mmap-able format
    GGJT,
}
impl ContainerType {
    pub fn support_mmap(&self) -> bool {
        match self {
            ContainerType::GGML => false,
            ContainerType::GGMF => false,
            ContainerType::GGJT => true,
        }
    }
}
