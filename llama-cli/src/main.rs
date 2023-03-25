use clap::Parser;

use Commands::LlamaCmd;
mod Commands;

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[command(flatten)]
    pub cmds: LlamaCmd,
}

impl Args {
    fn run(self) -> Result<(), String> {
        self.cmds.run()
    }
}

fn main() -> Result<(), String> {
    Args::parse().run()
}
