use clap::{Parser, Args};
use llama_rs::{InferenceError, InferenceParameters, InferenceSnapshot, InferenceSession, Model, Vocabulary};
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
}

impl Prompts {
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
        // interactive, repl, cache_prompt
        // if just plain prompt file or prompt fun this
        Ok(self.prompt);
    }

    fn create_session(&self, session: &InferenceSession) {
        let res = session.inference_with_prompt::<Infallible>(
            &model,
            &vocab,
            &inference_params,
            &prompt,
            args.num_predict,
            &mut rng,
            |t| {
                print!("{t}");
                std::io::stdout().flush().unwrap();

                Ok(())
            },
        );

        println!();

        match res {
            Ok(stats) => {
                println!("{}", stats);
            }
            Err(llama_rs::InferenceError::ContextFull) => {
                log::warn!("Context window full, stopping inference.")
            }
            Err(InferenceError::UserCallback(_)) => unreachable!("cannot fail"),
        }
        Ok(())
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
        }
    }
}
