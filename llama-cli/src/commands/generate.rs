
    pub struct Cmd {
        /// Sets the number of threads to use
        #[arg(long, short = 't', default_value_t = num_cpus::get_physical())]
        num_threads: usize,

        /// Sets how many tokens to predict
        #[arg(long, short = 'n')]
        num_predict: Option<usize>,

        /// Sets the size of the context (in tokens). Allows feeding longer prompts.
        /// Note that this affects memory. TODO: Unsure how large the limit is.
        #[arg(long, default_value_t = 512)]
        num_ctx_tokens: usize,

        /// How many tokens from the prompt at a time to feed the network. Does not
        /// affect generation.
        #[arg(long, default_value_t = 8)]
        batch_size: usize,

        /// Size of the 'last N' buffer that is used for the `repeat_penalty`
        /// option. In tokens.
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// The penalty for repeating tokens. Higher values make the generation less
        /// likely to get into a loop, but may harm results when repetitive outputs
        /// are desired.
        #[arg(long, default_value_t = 1.30)]
        repeat_penalty: f32,

        /// Temperature
        #[arg(long, default_value_t = 0.80)]
        temp: f32,

        /// Top-K: The top K words by score are kept during sampling.
        #[arg(long, default_value_t = 40)]
        top_k: usize,

        /// Top-p: The cummulative probability after which no more words are kept
        /// for sampling.
        #[arg(long, default_value_t = 0.95)]
        top_p: f32,

        /// Specifies the seed to use during sampling. Note that, depending on
        /// hardware, the same seed may lead to different results on two separate
        /// machines.
        #[arg(long, default_value = None)]
        seed: Option<u64>,
    },
