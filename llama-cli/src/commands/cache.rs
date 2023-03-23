    pub struct Cmd {
        /// Stores a cached prompt at the given path. The same prompt can then be
        /// loaded from disk using --restore-prompt
        #[arg(long, default_value = None)]
        cache_prompt: Option<String>,

        /// Restores a cached prompt at the given path, previously using
        /// --cache-prompt
        #[arg(long, default_value = None)]
        restore_prompt: Option<String>,
    }
