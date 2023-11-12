use std::path::Path;

pub use llm_base::loader::{load_progress_callback_stdout, LoadError, LoadProgress};
use llm_base::{
    loader::{ModelFactory, ModelLoadCallback},
    model::{ModelLoadArgs, ModelLoadError},
    Model, ModelParameters, TokenizerSource,
};

use crate::{ModelArchitecture, ModelArchitectureVisitor};

/// Loads the specified GGUF model from disk, determining its architecture from the metadata,
/// and loading it with one of the supported modules. If you want to load a custom model,
/// consider using [llm_base::loader::load] directly.
///
/// This method returns a [`Box`], which means that the model will have single ownership.
/// If you'd like to share ownership (i.e. to use the model in multiple threads), we
/// suggest using [`Arc::from(Box<T>)`](https://doc.rust-lang.org/std/sync/struct.Arc.html#impl-From%3CBox%3CT,+Global%3E%3E-for-Arc%3CT%3E)
/// to convert the [`Box`] into an [`Arc`](std::sync::Arc) after loading.
pub fn load(
    path: &Path,
    tokenizer_source: TokenizerSource,
    params: ModelParameters,
    load_progress_callback: impl FnMut(LoadProgress),
) -> Result<Box<dyn Model>, LoadError> {
    llm_base::loader::load(
        path,
        tokenizer_source,
        params,
        VisitorModelFactory,
        load_progress_callback,
    )
}

struct VisitorModelFactory;
impl ModelFactory for VisitorModelFactory {
    fn load(&self, architecture: &str) -> Option<ModelLoadCallback> {
        let architecture = architecture.parse::<ModelArchitecture>().ok()?;
        Some(architecture.visit(VisitorModelFactoryVisitor))
    }
}

struct VisitorModelFactoryVisitor;
impl ModelArchitectureVisitor<ModelLoadCallback> for VisitorModelFactoryVisitor {
    fn visit<M: Model + 'static>(self) -> ModelLoadCallback {
        Self::new_for_model::<M>
    }
}
impl VisitorModelFactoryVisitor {
    fn new_for_model<M: Model + 'static>(
        args: ModelLoadArgs,
    ) -> Result<Box<dyn Model>, ModelLoadError> {
        Ok(M::new(args).map(Box::new)?)
    }
}
