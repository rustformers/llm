use clap::Parser;
use once_cell::sync::Lazy;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
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

    /// Top-p: The cummulative probability after which no more words are kept
    /// for sampling.
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,

    /// Stores a cached prompt at the given path. The same prompt can then be
    /// loaded from disk using --restore-prompt
    #[arg(long, default_value = None)]
    pub cache_prompt: Option<String>,

    /// Restores a cached prompt at the given path, previously using
    /// --cache-prompt
    #[arg(long, default_value = None)]
    pub restore_prompt: Option<String>,
}

/// CLI args are stored in a lazy static variable so they're accessible from
/// everywhere. Arguments are parsed on first access.
pub static CLI_ARGS: Lazy<Args> = Lazy::new(Args::parse);
