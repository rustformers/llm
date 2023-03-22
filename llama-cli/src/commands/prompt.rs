    pub struct Cmd {
        /// The prompt to feed the generator
        #[arg(long, short = 'p', default_value = None)]
        prompt: Option<String>,

        /// A file to read the prompt from. Takes precedence over `prompt` if set.
        #[arg(long, short = 'f', default_value = None)]
        prompt_file: Option<String>,
    }
