use std::path::PathBuf;

use clap::Parser;

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
}
