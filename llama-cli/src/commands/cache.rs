use clap::Args;
use llama_rs::{InferenceError, InferenceParameters, InferenceSnapshot, InferenceSession, Model, Vocabulary};
use std::{convert::Infallible, io::Write};

#[derive(Debug, Args)]
pub struct Cache {
    /// Stores a cached prompt at the given path. The same prompt can then be
    /// loaded from disk using --restore-prompt
    #[arg(long, default_value = None)]
    cache_prompt: Option<String>,

}

impl Cache {

    fn cache_current_prompt(&self, session: &InferenceSession, model: &Model, vocab: &Vocabulary, prompt: &str, inference_params: &InferenceParameters) {
        if self.cache_prompt.is_some() {
            let res = session.feed_prompt::<Infallible>(
                &model,
                &vocab,
                &inference_params,
                &prompt,
                |t| {
                    print!("{t}");
                    std::io::stdout().flush().unwrap();

                    Ok(())
                },
            );

            println!();

            match res {
                Ok(_) => (),
                Err(InferenceError::ContextFull) => {
                    log::warn!(
                        "Context is not large enough to fit the prompt. Saving intermediate state."
                    );
                }
                Err(InferenceError::UserCallback(_)) => unreachable!("cannot fail"),
            }

            // Write the memory to the cache file
            // SAFETY: no other model functions used inside the block
            unsafe {
                let memory = session.get_snapshot();
                match memory.write_to_disk(self.cache_prompt) {
                    Ok(_) => {
                        log::info!("Successfully written prompt cache to {0}", self.cache_prompt);
                    }
                    Err(err) => {
                        eprintln!("Could not restore prompt. Error: {err}");
                        std::process::exit(1);
                    }
                }
            }
        }
    }
}
