use clap::Args;
use llama_rs::{InferenceError, InferenceParameters, InferenceSnapshot, Model, Vocabulary};
use rand::SeedableRng;

#[derive(Debug, Args)]
pub struct Generate {
    /// Where to load the model path from
    #[arg(long, short = 'm')]
    model_path: String,

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
}

impl Generate {
    fn create_seed(&self) -> rand::rngs::StdRng {
        if self.seed.is_some() {
            rand::rngs::StdRng::seed_from_u64(self.seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        }
    }

    fn select_model_and_vocab(&self) -> Result<(Model, Vocabulary), String> {
        let (model, vocab) =
            llama_rs::Model::load(&self.model_path, self.num_ctx_tokens as i32, |progress| {
                use llama_rs::LoadProgress;

                match progress {
                    LoadProgress::HyperparametersLoaded(hparams) => {
                        log::debug!("Loaded HyperParams {hparams:#?}")
                    }
                    LoadProgress::BadToken { index } => {
                        log::info!("Warning: Bad token in vocab at index {index}")
                    }
                    LoadProgress::ContextSize { bytes } => log::info!(
                        "ggml ctx size = {:.2} MB\n",
                        bytes as f64 / (1024.0 * 1024.0)
                    ),
                    LoadProgress::MemorySize { bytes, n_mem } => log::info!(
                        "Memory size: {} MB {}",
                        bytes as f32 / 1024.0 / 1024.0,
                        n_mem
                    ),
                    LoadProgress::PartLoading {
                        file,
                        current_part,
                        total_parts,
                    } => log::info!(
                        "Loading model part {}/{} from '{}'\n",
                        current_part,
                        total_parts,
                        file.to_string_lossy(),
                    ),
                    LoadProgress::PartTensorLoaded {
                        current_tensor,
                        tensor_count,
                        ..
                    } => {
                        if current_tensor % 8 == 0 {
                            log::info!("Loaded tensor {current_tensor}/{tensor_count}");
                        }
                    }
                    LoadProgress::PartLoaded {
                        file,
                        byte_size,
                        tensor_count,
                    } => {
                        log::info!("Loading of '{}' complete", file.to_string_lossy());
                        log::info!(
                            "Model size = {:.2} MB / num tensors = {}",
                            byte_size as f64 / 1024.0 / 1024.0,
                            tensor_count
                        );
                    }
                }
            })
            .expect("Could not load model");

        log::info!("Model fully loaded!");
        Ok((model, vocab))
    }

    fn create_inference_parameters(&self) -> InferenceParameters {
        InferenceParameters {
            n_threads: self.num_threads as i32,
            n_batch: self.batch_size,
            top_k: self.top_k,
            top_p: self.top_p,
            repeat_penalty: self.repeat_penalty,
            temp: self.temp,
        };
    }
}
