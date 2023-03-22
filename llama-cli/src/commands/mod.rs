use clap::{Parser, Subcommand};
use once_cell::sync::Lazy;

mod cache;
mod mode;
mod model;
mod prompt;
mod generate;


#[derive(Debug, Subcommand)]
pub enum LlamaCmd {

    Cache(cache::Cmd),

    Generate(generate::Cmd),

    Mode(mode::Cmd),

    Model(model::Cmd),

    Prompt(prompt::Cmd)

}

impl LlamaCmd {
    fn run(&self) {
        match self {

        }

    }
}
