use clap::{Args, Parser};
use env_logger::Builder;
use llama_rs::{InferenceError, InferenceParameters, InferenceSnapshot, Model, Vocabulary};
use once_cell::sync::Lazy;
use std::{convert::Infallible, io::Write};

mod cache;
mod generate;
mod mode;
mod prompt;

use cache::Cache;
use generate::Generate;
use mode::Mode;
use prompt::Prompts;

#[derive(Debug, Args)]
pub struct LlamaCmd {
    #[command(flatten)]
    pub cache: Cache,

    #[command(flatten)]
    pub generate: Generate,

    #[command(flatten)]
    pub mode: Mode,

    #[command(flatten)]
    pub prompts: Prompts,
}

impl LlamaCmd {
    pub fn run(&self) -> Result<(), String> {
        //  create and run the actual session here
        //  use match and if statements to build up the
        Builder::new()
            .filter_level(log::LevelFilter::Info)
            .parse_default_env()
            .init();

        let prompt = self.prompts.run();
        let generate = self.generate.inference_parameters();
        let mode = self.mode.run();
        let (mut model, vocab) = select_model_and_vocab(args).unwrap();
    }
}
