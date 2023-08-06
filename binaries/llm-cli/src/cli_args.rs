use std::{
    fmt,
    ops::Deref,
    path::{Path, PathBuf},
};

use clap::{Parser, ValueEnum};
use color_eyre::eyre::{self, WrapErr};
use llm::{
    ggml_format, samplers::build_sampler, ElementType, InferenceParameters, InferenceSessionConfig,
    InvalidTokenBias, LoadProgress, Model, ModelKVMemoryType, ModelParameters, RoPEOverrides,
    TokenBias, TokenId, TokenizerSource,
};
use rand::SeedableRng;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub enum Args {
    #[command()]
    /// Use a model to infer the next tokens in a sequence, and exit.
    Infer(Box<Infer>),

    #[command()]
    /// Measure a model's perplexity for a given prompt.
    Perplexity(Box<Perplexity>),

    #[command()]
    /// Get information about a GGML model.
    Info(Box<Info>),

    #[command()]
    /// Dumps the prompt to console and exits, first as a comma-separated list of token IDs
    /// and then as a list of comma-separated string keys and token ID values.
    PromptTokens(Box<PromptTokens>),

    #[command()]
    /// Use a model to interactively prompt it multiple times, while
    /// resetting the context between invocations.
    Repl(Box<Repl>),

    #[command()]
    /// Use a model to interactively generate tokens, and chat with it.
    ///
    /// Note that most, if not all, existing models are not trained for this
    /// and do not support a long enough context window to be able to
    /// have an extended conversation.
    Chat(Box<Chat>),

    /// Quantize a GGML model to 4-bit.
    Quantize(Box<Quantize>),
}

#[derive(Parser, Debug)]
pub struct Infer {
    #[command(flatten)]
    pub model_load: ModelLoad,

    #[command(flatten)]
    pub prompt_file: PromptFile,

    #[command(flatten)]
    pub generate: Generate,

    #[command(flatten)]
    pub prompt: Prompt,

    /// Hide the prompt in the generation.
    ///
    /// By default, the prompt tokens will be shown as they are fed to the model.
    /// This option will only show the inferred tokens.
    #[arg(long, default_value_t = false)]
    pub hide_prompt: bool,

    /// Loads a saved inference session from the given path, previously saved using
    /// `--save-session`
    #[arg(long, default_value = None)]
    pub load_session: Option<PathBuf>,

    /// Saves an inference session at the given path. The same session can then be
    /// loaded from disk using `--load-session`.
    ///
    /// Use this with `-n 0` to save just the prompt
    #[arg(long, default_value = None)]
    pub save_session: Option<PathBuf>,

    /// Loads an inference session from the given path if present, and then saves
    /// the result to the same path after inference is completed.
    ///
    /// Equivalent to `--load-session` and `--save-session` with the same path,
    /// but will not error if the path does not exist
    #[arg(long, default_value = None)]
    pub persist_session: Option<PathBuf>,

    /// Output statistics about the time taken to perform inference, among other
    /// things.
    #[arg(long, default_value_t = false)]
    pub stats: bool,
}

#[derive(Parser, Debug)]
pub struct Perplexity {
    #[command(flatten)]
    pub model_load: ModelLoad,

    #[command(flatten)]
    pub prompt_file: PromptFile,

    #[command(flatten)]
    pub generate: Generate,

    #[command(flatten)]
    pub prompt: Prompt,
}

#[derive(Parser, Debug)]
pub struct Info {
    #[command(flatten)]
    pub model_and_tokenizer: ModelAndTokenizer,

    /// Show all of the tensors in the model, including their names, formats and shapes.
    #[arg(long, short = 't')]
    pub tensors: bool,

    /// Show all of the tokens in the tokenizer.
    #[arg(long, short = 'k')]
    pub tokenizer: bool,
}

#[derive(Parser, Debug)]
pub struct PromptTokens {
    #[command(flatten)]
    pub model_load: ModelLoad,

    #[command(flatten)]
    pub prompt_file: PromptFile,

    #[command(flatten)]
    pub prompt: Prompt,
}

#[derive(Parser, Debug)]
pub struct Prompt {
    /// The prompt to feed the generator.
    ///
    /// If used with `--prompt-file`/`-f`, the prompt from the file will be used
    /// and `{{PROMPT}}` will be replaced with the value of `--prompt`/`-p`.
    #[arg(long, short = 'p', default_value = None)]
    prompt: Option<String>,
}
impl Deref for Prompt {
    type Target = Option<String>;

    fn deref(&self) -> &Self::Target {
        &self.prompt
    }
}

#[derive(Parser, Debug)]
pub struct Repl {
    #[command(flatten)]
    pub model_load: ModelLoad,

    #[command(flatten)]
    pub prompt_file: PromptFile,

    #[command(flatten)]
    pub generate: Generate,
}

#[derive(Parser, Debug)]
pub struct Chat {
    #[command(flatten)]
    pub model_load: ModelLoad,

    /// The file to read the initial prompt/prelude from.
    ///
    /// Must contain a `{{PROMPT}}` placeholder, which will be replaced with the
    /// first user prompt.
    #[arg(long, short = 'f')]
    pub prelude_prompt_file: PathBuf,

    /// The per-message prefix to be prepended to the user's message.
    ///
    /// The `{{PROMPT}}` will automatically be appended to this prefix.
    #[arg(long, short = 'p')]
    pub message_prompt_prefix: Option<String>,

    /// The file containing the per-message prefix to be prepended to the user's message.
    ///
    /// The `{{PROMPT}}` will automatically be appended to this prefix.
    #[arg(long, short = 'q')]
    pub message_prompt_prefix_file: Option<PathBuf>,

    #[command(flatten)]
    pub generate: Generate,
}
impl Chat {
    pub fn message_prompt_prefix(&self) -> eyre::Result<String> {
        const MESSAGE_PROMPT_PREFIX_ERROR: &str = concat!(
            "Message prompt prefix must not contain a `{{PROMPT}}` placeholder. ",
            "The prompt will be automatically appended to the prefix."
        );

        match (
            &self.message_prompt_prefix,
            &self.message_prompt_prefix_file,
        ) {
            (None, None) => eyre::bail!(
                "Must specify either --message-prompt-prefix or --message-prompt-prefix-file"
            ),
            (Some(_), Some(_)) => eyre::bail!(
                "Cannot specify both --message-prompt-prefix and --message-prompt-prefix-file"
            ),
            (Some(message_prompt_prefix), None) => {
                if message_prompt_prefix.contains("{{PROMPT}}") {
                    eyre::bail!("{MESSAGE_PROMPT_PREFIX_ERROR}");
                }
                Ok(message_prompt_prefix.clone())
            }
            (None, Some(message_prompt_prefix_file)) => {
                let prompt = read_prompt_file(message_prompt_prefix_file)?;
                if prompt.contains("{{PROMPT}}") {
                    eyre::bail!("{MESSAGE_PROMPT_PREFIX_ERROR}");
                }
                Ok(prompt)
            }
        }
    }
}

#[derive(Parser, Debug)]
pub struct Generate {
    /// Sets the number of threads to use
    #[arg(long, short = 't')]
    pub num_threads: Option<usize>,

    /// Sets how many tokens to predict
    #[arg(long, short = 'n')]
    pub num_predict: Option<usize>,

    /// How many tokens from the prompt at a time to feed the network. Does not
    /// affect generation.
    #[arg(long, default_value_t = 8)]
    pub batch_size: usize,

    /// Configure sampler settings using a string in the format: sampler_name:key1=value1:key2=value2
    /// To configure multiple samplers at once, separate the sampler configuration strings with space or '/' (forward slash).
    /// NOTE: Mirostat samplers are incompatible with top-p, top-k, locally typical and tail free samplers.
    /// TIPS:
    ///   1. Sampler options aren't required. For example "mirostat1" will enable Mirostat 1 with its default options.
    ///   2. It's possible to specify partial option names, as long as they are unambiguous.
    ///   3. Underscore and dash are ignored in sampler names, so "top-p" is the same as "topp" or "top_p".
    ///
    /// Configurable samplers (defaults shown in parenthesis):
    ///
    /// freq_presence (default: disabled) - Allows penalizing tokens for presence and frequency. May be specified more than once.
    ///   frequency_penalty(0.0): Penalty to apply to tokens based on frequency. For example, if a token has appeared 3 times within the last_n range then it will have its probability decreased by 3 * frequency_penalty.
    ///   presence_penalty(0.0): Penalty to apply to tokens that are already present within the last_n tokens.
    ///   last_n(64): Number of previous tokens to consider.
    ///
    /// locally_typical (default: disabled) - An approach to sampling that attempts to maximize natural and human-like output. See: <https://arxiv.org/abs/2202.00666>
    ///   p(1.0): Referred to as τ in the paper. It suggests using 0.2 as a value for story generation and `0.95` for "abstractive summarization" (presumably this means more factual output). 1.0 appears to be the same as disabled which is similar to top-p sampling.
    ///   min_keep(1): Minimum tokens to keep. Setting this to 0 is not recommended.
    ///
    /// mirostat1 (default: disabled) - See: <https://arxiv.org/abs/2007.14966>
    ///   eta(0.1): Learning rate
    ///   tau(5.0): Target entropy
    ///   mu(tau * 2): Initial learning state value. Setting this is generally not recommended.
    ///
    /// mirostat2 (default: disabled) - See: <https://arxiv.org/abs/2007.14966>
    ///   eta(0.1): Learning rate
    ///   tau(5.0): Target entropy
    ///   mu(tau * 2): Initial learning state value. Setting this is generally not recommended.
    ///
    /// repetition - Allows setting a repetition penalty. May be specified more than once.
    ///   penalty(1.30): The penalty for repeating tokens. Higher values make the generation less likely to get into a loop, but may harm results when repetitive outputs are desired.
    ///   last_n(64): Number of previous tokens to consider.
    ///
    /// tail_free (default: disabled) - An approach to sampling that attempts to outperform existing nucleus (top-p and top-k) methods. See: <https://trentbrick.github.io/Tail-Free-Sampling/>
    ///   z(1.0): It is not entirely clear what a reasonable value here is but 1.0 appears to be the same as disabled which is similar to top-p sampling.
    ///   min_keep(1): Minimum tokens to keep. Setting this to 0 is not recommended.
    ///
    /// temperature - Temperature used for sampling.
    ///   temperature(0.8): Temperature (randomness) used for sampling. A higher number is more random.
    ///
    /// top_k - The top k (or min_keep if it is greater) tokens by score are kept during sampling.
    ///   k(40): Number of tokens to keep.
    ///   min_keep(1): Minimum tokens to keep. Setting this to 0 is not recommended.
    ///
    /// top_p - The probability for the top tokens are added until the result is greater or equal to P and at least min_keep tokens have been seen.
    ///   p(0.95): The cumulative probability after which no more tokens are kept for sampling.
    ///   min_keep(1): Minimum tokens to keep. Setting this to 0 is not recommended.
    #[arg(long = "sampler", short = 's', verbatim_doc_comment)]
    pub sampler_options: Vec<String>,

    /// Specifies the seed to use during sampling. Note that, depending on
    /// hardware, the same seed may lead to different results on two separate
    /// machines.
    #[arg(long, default_value = None)]
    pub seed: Option<u64>,

    /// Use 16-bit floats for model memory key and value. Ignored but allowed for
    /// backwards compatibility: this is now the default
    #[arg(long = "float16", hide = true)]
    pub _float16: bool,

    /// Use 32-bit floats for model memory key and value.
    /// Not recommended: doubles size without a measurable quality increase.
    /// Ignored when restoring from the cache
    #[arg(long = "no-float16", default_value_t = false)]
    pub no_float16: bool,

    /// A comma separated list of token biases. The list should be in the format
    /// "TID=BIAS,TID=BIAS" where TID is an integer token ID and BIAS is a
    /// floating point number.
    /// For example, "1=-1.0,2=-1.0" sets the bias for token IDs 1
    /// (start of document) and 2 (end of document) to -1.0 which effectively
    /// disables the model from generating responses containing those token IDs.
    #[arg(long, default_value = None, value_parser = parse_bias)]
    pub token_bias: Option<TokenBias>,

    /// Prevent the end of stream (EOS/EOD) token from being generated. This will allow the
    /// model to generate text until it runs out of context space.
    #[arg(long, default_value_t = false)]
    pub ignore_eos: bool,

    /// Whether to use GPU acceleration when available
    #[arg(long, default_value_t = false)]
    pub use_gpu: bool,
}
impl Generate {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn autodetect_num_threads(&self) -> usize {
        std::process::Command::new("sysctl")
            .arg("-n")
            .arg("hw.perflevel0.physicalcpu")
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok()?.trim().parse().ok())
            .unwrap_or(num_cpus::get_physical())
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    pub fn autodetect_num_threads(&self) -> usize {
        num_cpus::get_physical()
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
            .unwrap_or_else(|| self.autodetect_num_threads())
    }

    pub fn inference_session_config(&self) -> InferenceSessionConfig {
        let mem_typ = if self.no_float16 {
            ModelKVMemoryType::Float32
        } else {
            ModelKVMemoryType::Float16
        };
        InferenceSessionConfig {
            memory_k_type: mem_typ,
            memory_v_type: mem_typ,
            n_batch: self.batch_size,
            n_threads: self.num_threads(),
        }
    }

    pub fn rng(&self) -> rand::rngs::StdRng {
        if let Some(seed) = self.seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        }
    }

    pub fn inference_parameters(
        &self,
        eot: TokenId,
        n_vocab: usize,
    ) -> eyre::Result<InferenceParameters> {
        let mut bias: Vec<(TokenId, f32)> = self.token_bias.clone().unwrap_or_default().into();
        if self.ignore_eos {
            bias.push((eot, f32::NEG_INFINITY));
        }
        Ok(InferenceParameters {
            sampler: build_sampler(n_vocab, &bias, &self.sampler_options)
                .map_err(|e| eyre::eyre!("Invalid sampler configuration: {e}"))?,
        })
    }
}

fn parse_bias(s: &str) -> Result<TokenBias, InvalidTokenBias> {
    s.parse()
}

#[derive(Parser, Debug)]
pub struct ModelTokenizer {
    /// Local path to Hugging Face tokenizer file
    #[arg(long, short = 'v')]
    pub tokenizer_path: Option<PathBuf>,

    /// Remote Hugging Face repository containing a tokenizer
    #[cfg(feature = "tokenizers-remote")]
    #[arg(long, short = 'r')]
    pub tokenizer_repository: Option<String>,
}
impl ModelTokenizer {
    pub fn to_source(&self) -> eyre::Result<TokenizerSource> {
        let tokenizer_path = self.tokenizer_path.as_deref();
        #[cfg(feature = "tokenizers-remote")]
        let tokenizer_repository = self.tokenizer_repository.as_deref();

        #[cfg(feature = "tokenizers-remote")]
        if tokenizer_path.is_some() && tokenizer_repository.is_some() {
            eyre::bail!("Cannot specify both --tokenizer-path and --tokenizer-repository");
        }

        if let Some(path) = tokenizer_path {
            return Ok(TokenizerSource::HuggingFaceTokenizerFile(path.to_owned()));
        }

        #[cfg(feature = "tokenizers-remote")]
        if let Some(repository) = tokenizer_repository {
            return Ok(TokenizerSource::HuggingFaceRemote(repository.to_owned()));
        }

        Ok(TokenizerSource::Embedded)
    }
}

#[derive(Parser, Debug)]
pub struct ModelArchitecture {
    /// The model architecture to use. Will attempt to guess if not specified.
    #[arg(long, short = 'a')]
    pub model_architecture: Option<llm::ModelArchitecture>,
}

#[derive(Parser, Debug)]
pub struct ModelAndTokenizer {
    /// Where to load the model from
    #[arg(long, short = 'm')]
    pub model_path: PathBuf,

    #[command(flatten)]
    pub architecture: ModelArchitecture,

    #[command(flatten)]
    pub tokenizer: ModelTokenizer,
}
impl ModelAndTokenizer {
    pub fn to_source(&self) -> eyre::Result<TokenizerSource> {
        self.tokenizer.to_source()
    }
}

#[derive(Parser, Debug)]
pub struct RoPEScaling {
    #[arg(long)]
    pub rope_freq_base: Option<usize>,

    #[arg(long)]
    pub rope_freq_scale: Option<f32>,
}

impl RoPEScaling {
    pub fn to_rope_arguments(&self) -> Option<RoPEOverrides> {
        if self.rope_freq_base.is_none() && self.rope_freq_scale.is_none() {
            return None;
        }

        let default = RoPEOverrides::default();
        Some(RoPEOverrides {
            frequency_base: self.rope_freq_base.unwrap_or(default.frequency_base),
            frequency_scale: self.rope_freq_scale.unwrap_or(default.frequency_scale),
        })
    }
}

#[derive(Parser, Debug)]
pub struct ModelLoad {
    #[command(flatten)]
    pub model_and_tokenizer: ModelAndTokenizer,

    /// Sets the size of the context (in tokens). Allows feeding longer prompts.
    /// Note that this affects memory.
    ///
    /// LLaMA models are trained with a context size of 2048 tokens. If you
    /// want to use a larger context size, you will need to retrain the model,
    /// or use a model that was trained with a larger context size.
    ///
    /// Alternate methods to extend the context, including
    /// [context clearing](https://github.com/rustformers/llm/issues/77) are
    /// being investigated, but are not yet implemented. Additionally, these
    /// will likely not perform as well as a model with a larger context size.
    #[arg(long, default_value_t = 2048)]
    pub num_ctx_tokens: usize,

    /// Don't use mmap to load the model.
    #[arg(long)]
    pub no_mmap: bool,

    /// LoRA adapter to use for the model
    #[arg(long, num_args(0..))]
    pub lora_paths: Option<Vec<PathBuf>>,

    /// Number of layers to run on the GPU. If not specified, all layers will be run on the GPU.
    #[arg(long)]
    pub gpu_layers: Option<usize>,

    #[command(flatten)]
    pub rope_scaling: RoPEScaling,
}

impl ModelLoad {
    pub fn load(&self, use_gpu: bool) -> eyre::Result<Box<dyn Model>> {
        let params = ModelParameters {
            prefer_mmap: !self.no_mmap,
            context_size: self.num_ctx_tokens,
            lora_adapters: self.lora_paths.clone(),
            use_gpu,
            gpu_layers: self.gpu_layers,
            rope_overrides: self.rope_scaling.to_rope_arguments(),
        };

        let mut sp = Some(spinoff::Spinner::new(
            spinoff::spinners::Dots2,
            "Loading model...",
            None,
        ));
        let now = std::time::Instant::now();
        let mut prev_load_time = now;

        let tokenizer_source = match self.model_and_tokenizer.to_source() {
            Ok(vs) => vs,
            Err(err) => {
                if let Some(sp) = sp.take() {
                    sp.fail(&format!("Failed to load tokenizer: {}", err));
                }
                return Err(err);
            }
        };

        let model = llm::load_dynamic(
            self.model_and_tokenizer.architecture.model_architecture,
            &self.model_and_tokenizer.model_path,
            tokenizer_source,
            params,
            |progress| match progress {
                LoadProgress::HyperparametersLoaded => {
                    if let Some(sp) = sp.as_mut() {
                        sp.update_text("Loaded hyperparameters")
                    };
                }
                LoadProgress::ContextSize { bytes } => log::debug!(
                    "ggml ctx size = {}",
                    bytesize::to_string(bytes as u64, false)
                ),
                LoadProgress::LoraApplied { name, source } => {
                    if let Some(sp) = sp.as_mut() {
                        sp.update_text(format!(
                            "Patched tensor {} via LoRA from '{}'",
                            name,
                            source.file_name().unwrap().to_str().unwrap()
                        ));
                    }
                }
                LoadProgress::TensorLoaded {
                    current_tensor,
                    tensor_count,
                    ..
                } => {
                    if prev_load_time.elapsed().as_millis() > 500 {
                        // We don't want to re-render this on every message, as that causes the
                        // spinner to constantly reset and not look like it's spinning (and
                        // it's obviously wasteful).
                        if let Some(sp) = sp.as_mut() {
                            sp.update_text(format!(
                                "Loaded tensor {}/{tensor_count}",
                                current_tensor + 1,
                            ));
                        };
                        prev_load_time = std::time::Instant::now();
                    }
                }
                LoadProgress::Loaded {
                    file_size,
                    tensor_count,
                } => {
                    if let Some(sp) = sp.take() {
                        sp.success(&format!(
                            "Loaded {tensor_count} tensors ({}) after {}ms",
                            bytesize::to_string(file_size, false),
                            now.elapsed().as_millis()
                        ));
                    };
                }
            },
        )
        .wrap_err("Could not load model");

        if model.is_err() {
            // If we've failed at loading the model, we probably haven't stopped the spinner yet.
            // Cancel it now if needed.
            if let Some(sp) = sp {
                sp.fail("Failed to load model")
            }
        }

        model
    }
}

#[derive(Parser, Debug)]
pub struct PromptFile {
    /// A file to read the prompt from.
    #[arg(long, short = 'f', default_value = None)]
    pub prompt_file: Option<PathBuf>,
}
impl PromptFile {
    pub fn contents(&self) -> eyre::Result<Option<String>> {
        Ok(match &self.prompt_file {
            Some(path) => Some(read_prompt_file(path)?),
            _ => None,
        })
    }
}

pub fn read_prompt_file(path: &Path) -> eyre::Result<String> {
    std::fs::read_to_string(path)
        .wrap_err_with(|| format!("Could not read prompt file at {path:?}"))
}

#[derive(Parser, Debug)]
pub struct Quantize {
    #[command(flatten)]
    pub architecture: ModelArchitecture,

    /// The path to the model to quantize
    #[arg()]
    pub source: PathBuf,

    /// The path to save the quantized model to
    #[arg()]
    pub destination: PathBuf,

    #[command(flatten)]
    pub tokenizer: ModelTokenizer,

    /// The GGML container type to target.
    ///
    /// Note that using GGML requires the original model to have
    /// an unscored vocabulary, which is not the case for newer models.
    #[arg(short, long, default_value_t = SaveContainerType::GgjtV3)]
    pub container_type: SaveContainerType,

    /// The format to convert to
    pub target: QuantizationTarget,
}

#[derive(Parser, Debug, ValueEnum, Clone, Copy)]
pub enum SaveContainerType {
    /// GGML container.
    Ggml,
    /// GGJT v3 container.
    GgjtV3,
}
impl fmt::Display for SaveContainerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SaveContainerType::Ggml => write!(f, "ggml"),
            SaveContainerType::GgjtV3 => write!(f, "ggjt-v3"),
        }
    }
}
impl From<SaveContainerType> for ggml_format::SaveContainerType {
    fn from(value: SaveContainerType) -> Self {
        match value {
            SaveContainerType::Ggml => ggml_format::SaveContainerType::Ggml,
            SaveContainerType::GgjtV3 => ggml_format::SaveContainerType::GgjtV3,
        }
    }
}

#[derive(Parser, Debug, ValueEnum, Clone, Copy)]
#[clap(rename_all = "snake_case")]
pub enum QuantizationTarget {
    /// Quantized 4-bit (type 0).
    Q4_0,
    /// Quantized 4-bit (type 1).
    Q4_1,
    /// Quantized 5-bit (type 0).
    Q5_0,
    /// Quantized 5-bit (type 1).
    Q5_1,
    /// Quantized 8-bit (type 0).
    Q8_0,
}
impl From<QuantizationTarget> for ElementType {
    fn from(t: QuantizationTarget) -> Self {
        match t {
            QuantizationTarget::Q4_0 => ElementType::Q4_0,
            QuantizationTarget::Q4_1 => ElementType::Q4_1,
            QuantizationTarget::Q5_0 => ElementType::Q5_0,
            QuantizationTarget::Q5_1 => ElementType::Q5_1,
            QuantizationTarget::Q8_0 => ElementType::Q8_0,
        }
    }
}
