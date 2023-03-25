use clap::Args;
use env_logger::Builder;

mod generate;
mod mode;
mod prompt;

use generate::Generate;
use mode::Mode;
use prompt::Prompts;

#[derive(Debug, Args)]
pub struct LlamaCmd {
    #[command(flatten)]
    pub generate: Generate,

    #[command(flatten)]
    pub mode: Mode,

    #[command(flatten)]
    pub prompts: Prompts,
}

impl LlamaCmd {
    pub fn run(&self) -> Result<(), String> {
        Builder::new()
            .filter_level(log::LevelFilter::Info)
            .parse_default_env()
            .init();

        let prompt = self.prompts.run();
        let generate = self.generate.run(&prompt);
        self.mode.run(&generate);
    }
}
