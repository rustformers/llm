use std::path::PathBuf;

use clap::Parser;
use llama_rs::TokenBias;
use once_cell::sync::Lazy;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub enum Args {
    #[command(name = "generate")]
    Generate(Box<Generate>),

    #[command(name = "convert")]
    Convert(Box<Convert>),
}

#[derive(Parser, Debug)]
pub struct Convert {
    /// Path to model directory
    #[arg(long, short = 'd')]
    pub dir: String,

    /// File type to convert to
    #[arg(long, default_value_t = false)]
    pub f32: bool,
}

#[derive(Parser, Debug)]
pub struct Generate {
    /// Where to load the model path from
    #[arg(long, short = 'm')]
    pub model_path: String,

    /// The prompt to feed the generator
    #[arg(long, short = 'p', default_value = None)]
    pub prompt: Option<String>,

    /// A file to read the prompt from. Takes precedence over `prompt` if set.
    #[arg(long, short = 'f', default_value = None)]
    pub prompt_file: Option<String>,

    /// Run in REPL mode.
    #[arg(long, short = 'R', default_value_t = false)]
    pub repl: bool,

    /// Sets the number of threads to use
    #[arg(long, short = 't', default_value_t = num_cpus::get_physical())]
    pub num_threads: usize,

    /// Sets how many tokens to predict
    #[arg(long, short = 'n')]
    pub num_predict: Option<usize>,

    /// Sets the size of the context (in tokens). Allows feeding longer prompts.
    /// Note that this affects memory. TODO: Unsure how large the limit is.
    #[arg(long, default_value_t = 512)]
    pub num_ctx_tokens: usize,

    /// How many tokens from the prompt at a time to feed the network. Does not
    /// affect generation.
    #[arg(long, default_value_t = 8)]
    pub batch_size: usize,

    /// Size of the 'last N' buffer that is used for the `repeat_penalty`
    /// option. In tokens.
    #[arg(long, default_value_t = 64)]
    pub repeat_last_n: usize,

    /// The penalty for repeating tokens. Higher values make the generation less
    /// likely to get into a loop, but may harm results when repetitive outputs
    /// are desired.
    #[arg(long, default_value_t = 1.30)]
    pub repeat_penalty: f32,

    /// Temperature
    #[arg(long, default_value_t = 0.80)]
    pub temp: f32,

    /// Top-K: The top K words by score are kept during sampling.
    #[arg(long, default_value_t = 40)]
    pub top_k: usize,

    /// Top-p: The cumulative probability after which no more words are kept
    /// for sampling.
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,

    /// Saves an inference session at the given path. The same session can then be
    /// loaded from disk using `--load-session`.
    ///
    /// Use this with `-n 0` to save just the prompt
    #[arg(long, default_value = None)]
    pub save_session: Option<PathBuf>,

    /// Loads a saved inference session from the given path, previously saved using
    /// `--save-session`
    #[arg(long, default_value = None)]
    pub load_session: Option<PathBuf>,

    /// Loads an inference session from the given path if present, and then saves
    /// the result to the same path after inference is completed.
    ///
    /// Equivalent to `--load-session` and `--save-session` with the same path,
    /// but will not error if the path does not exist
    #[arg(long, default_value = None)]
    pub persist_session: Option<PathBuf>,

    /// Specifies the seed to use during sampling. Note that, depending on
    /// hardware, the same seed may lead to different results on two separate
    /// machines.
    #[arg(long, default_value = None)]
    pub seed: Option<u64>,

    /// Use 16-bit floats for model memory key and value. Ignored when restoring
    /// from the cache.
    #[arg(long, default_value_t = false)]
    pub float16: bool,

    /// A comma separated list of token biases. The list should be in the format
    /// "TID=BIAS,TID=BIAS" where TID is an integer token ID and BIAS is a
    /// floating point number.
    /// For example, "1=-1.0,2=-1.0" sets the bias for token IDs 1
    /// (start of document) and 2 (end of document) to -1.0 which effectively
    /// disables the model from generating responses containing those token IDs.
    #[arg(long, default_value = None, value_parser = parse_bias)]
    pub token_bias: Option<TokenBias>,

    /// Prevent the end of stream (EOS/EOD) token from being generated. This will allow the
    /// model to generate text until it runs out of context space. Note: The --token-bias
    /// option will override this if specified.
    #[arg(long, default_value_t = false)]
    pub ignore_eos: bool,

    /// Dumps the prompt to console and exits, first as a comma seperated list of token IDs
    /// and then as a list of comma seperated string keys and token ID values.
    #[arg(long, default_value_t = false)]
    pub dump_prompt_tokens: bool,
}

fn parse_bias(s: &str) -> Result<TokenBias, String> {
    s.parse()
}

/// CLI args are stored in a lazy static variable so they're accessible from
/// everywhere. Arguments are parsed on first access.
pub static CLI_ARGS: Lazy<Args> = Lazy::new(Args::parse);
