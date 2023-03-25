use clap::Parser;
use llama_rs::{
    InferenceError, InferenceParameters, InferenceSession, InferenceSnapshot, Model, Vocabulary,
};
use std::{convert::Infallible, io::Write};

#[derive(Debug, Parser)]
pub enum Prompts {
    Prompt {
        /// The prompt to feed the generator
        #[arg(long, short = 'p', default_value = None)]
        prompt: Option<String>,
    },
    PromptFile {
        /// A file to read the prompt from. Takes precedence over `prompt` if set.
        #[arg(long, short = 'f', default_value = None)]
        prompt_file: Option<String>,
    },

    RestorePrompt {
        /// Restores a cached prompt at the given path, previously using
        /// --cache-prompt
        #[arg(long, default_value = None)]
        restore_prompt: Option<String>,
    },

    CachePrompt {
        /// Stores a cached prompt at the given path. The same prompt can then be
        /// loaded from disk using --restore-prompt
        #[arg(long, default_value = None)]
        cache_prompt: Option<String>,
    },
}

impl Prompts {
    fn cache_current_prompt(
        &self,
        session: &InferenceSession,
        model: &Model,
        vocab: &Vocabulary,
        inference_params: &InferenceParameters,
    ) {
        // TODO: refactor this to decouple model generation and prompt creation
        // TODO: check run model then store prompt if successful
        let res = session.feed_prompt::<Infallible>(
            &model,
            &vocab,
            &inference_params,
            self.cache_prompt,
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
                    log::info!(
                        "Successfully written prompt cache to {0}",
                        self.cache_prompt
                    );
                }
                Err(err) => {
                    eprintln!("Could not restore prompt. Error: {err}");
                    std::process::exit(1);
                }
            }
        }
    }

    fn read_prompt_from_file(&self) -> Result<String, String> {
        match std::fs::read_to_string(self.prompt_file) {
            Ok(prompt) => Ok(prompt),
            Err(err) => {
                log::error!(
                    "Could not read prompt file at {}. Error: {}",
                    self.prompt_file,
                    err
                );
                return Err(format!(
                    "Could not read prompt file at {}. Error: {}",
                    self.prompt_file, err
                ));
            }
        }
    }

    fn create_prompt(&self) -> Result<String, String> {
        match self.prompt {
            Some(prompt) => Ok(prompt),
            None => {}
        }
    }

    fn restore_previous_prompt(&self, model: &Model) -> Result<String, String> {
        if self.restore_prompt.is_some() {
            let snapshot = InferenceSnapshot::load_from_disk(&self.restore_prompt);
            match snapshot.and_then(|snapshot| model.session_from_snapshot(snapshot)) {
                Ok(session) => {
                    log::info!("Restored cached memory from {0}", self.restore_prompt);
                    session
                }
                Err(err) => {
                    log::error!("{err}");
                    std::process::exit(1);
                }
            }
        }
    }

    fn run(&self) -> Result<String, String> {
        match self {
            Self::Prompt { prompt } => self.create_prompt(),
            Self::PromptFile { prompt_file } => self.create_prompt(),
            Self::RestorePrompt { restore_prompt } => self.restore_previous_prompt(),
            Self::CachePrompt { cache_prompt } => self.cache_current_prompt(),
        }
    }
}
