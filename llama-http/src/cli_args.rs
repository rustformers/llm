use clap::Parser;
use once_cell::sync::Lazy;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// The port to listen on
    #[arg(long, short = 'P', default_value_t = 8080)]
    pub port: u16,

    /// Where to load the model path from
    #[arg(long, short = 'm')]
    pub model_path: String,

    /// Use 16-bit floats for model memory key and value. Ignored when restoring
    /// from the cache.
    #[arg(long, default_value_t = false)]
    pub float16: bool,

    /// Sets the number of threads to use
    #[arg(long, short = 't', default_value_t = num_cpus::get_physical())]
    pub num_threads: usize,

    /// Sets the size of the context (in tokens). Allows feeding longer prompts.
    /// Note that this affects memory. TODO: Unsure how large the limit is.
    #[arg(long, default_value_t = 512)]
    pub num_ctx_tokens: usize,

    /// How many tokens from the prompt at a time to feed the network. Does not
    /// affect generation.
    /// This is the default value unless overridden by the request.
    #[arg(long, default_value_t = 8)]
    pub batch_size: usize,

    /// Size of the 'last N' buffer that is used for the `repeat_penalty`
    /// option. In tokens.
    #[arg(long, default_value_t = 64)]
    pub repeat_last_n: usize,

    /// The penalty for repeating tokens. Higher values make the generation less
    /// likely to get into a loop, but may harm results when repetitive outputs
    /// are desired.
    /// This is the default value unless overridden by the request.
    #[arg(long, default_value_t = 1.30)]
    pub repeat_penalty: f32,

    /// Temperature
    /// This is the default value unless overridden by the request.
    #[arg(long, default_value_t = 0.80)]
    pub temp: f32,

    /// Top-K: The top K words by score are kept during sampling.
    /// This is the default value unless overridden by the request.
    #[arg(long, default_value_t = 40)]
    pub top_k: usize,

    /// Top-p: The cummulative probability after which no more words are kept
    /// for sampling.
    /// This is the default value unless overridden by the request.
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,

    /// Restores a cached prompt at the given path, previously using
    /// --cache-prompt
    #[arg(long, default_value = None)]
    pub restore_prompt: Option<String>,
}

/// CLI args are stored in a lazy static variable so they're accessible from
/// everywhere. Arguments are parsed on first access.
pub static CLI_ARGS: Lazy<Args> = Lazy::new(Args::parse);
