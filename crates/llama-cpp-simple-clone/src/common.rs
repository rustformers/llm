use std::path::PathBuf;

use crate::llama::*;
use crate::sampling::*;

/// Structure to hold parameters for the GPT model.
#[derive(Debug)]
pub struct GptParams {
    /// RNG seed
    pub seed: u32,

    /// Number of threads for processing (default is the number of physical cores)
    pub n_threads: u32,
    /// Number of threads for batch processing (-1 = use `n_threads`)
    pub n_threads_batch: Option<u32>,
    /// New tokens to predict
    pub n_predict: Option<u32>,
    /// Context size
    pub n_ctx: u32,
    /// Batch size for prompt processing (must be >=32 to use BLAS)
    pub n_batch: u32,
    /// Number of tokens to keep from initial prompt
    pub n_keep: u32,
    /// Number of tokens to draft during speculative decoding
    pub n_draft: u32,
    /// Max number of chunks to process (-1 = unlimited)
    pub n_chunks: Option<u32>,
    /// Number of parallel sequences to decode
    pub n_parallel: u32,
    /// Number of sequences to decode
    pub n_sequences: u32,
    /// Speculative decoding accept probability
    pub p_accept: f32,
    /// Speculative decoding split probability
    pub p_split: f32,
    /// Number of layers to store in VRAM (-1 - use default)
    pub n_gpu_layers: Option<u32>,
    /// Number of layers to store in VRAM for the draft model (-1 - use default)
    pub n_gpu_layers_draft: Option<u32>,
    /// The GPU that is used for scratch and small tensors
    pub main_gpu: u32,
    /// How split tensors should be distributed across GPUs
    pub tensor_split: [f32; LLAMA_MAX_DEVICES], // Replace LLAMA_MAX_DEVICES with the actual value
    /// If non-zero then use beam search of given width
    pub n_beams: u32,
    /// RoPE base frequency
    pub rope_freq_base: f32,
    /// RoPE frequency scaling factor
    pub rope_freq_scale: f32,
    /// YaRN extrapolation mix factor
    pub yarn_ext_factor: f32,
    /// YaRN magnitude scaling factor
    pub yarn_attn_factor: f32,
    /// YaRN low correction dim
    pub yarn_beta_fast: f32,
    /// YaRN high correction dim
    pub yarn_beta_slow: f32,
    /// YaRN original context length
    pub yarn_orig_ctx: u32,
    /// RoPE scaling type (use int32_t for alignment)
    pub rope_scaling_type: LlamaRopeScaling,

    // Sampling parameters
    pub sparams: LlamaSamplingParams, // Define or import LlamaSamplingParams struct

    // Model and prompt related parameters
    pub model: PathBuf,
    pub model_draft: String,
    pub model_alias: String,
    pub prompt: String,
    pub prompt_file: String,
    pub path_prompt_cache: String,
    pub input_prefix: String,
    pub input_suffix: String,
    pub antiprompt: Vec<String>,
    pub logdir: String,

    // lora adapter
    pub lora_adapter: Vec<(String, f32)>,
    pub lora_base: String,

    // Perplexity calculations
    pub ppl_stride: i32,
    pub ppl_output_type: i32,

    // HellaSwag parameters
    pub hellaswag: bool,
    pub hellaswag_tasks: usize,

    // Additional parameters
    pub mul_mat_q: bool,
    pub memory_f16: bool,
    pub random_prompt: bool,
    pub use_color: bool,
    pub interactive: bool,
    pub prompt_cache_all: bool,
    pub prompt_cache_ro: bool,
    pub embedding: bool,
    pub escape: bool,
    pub interactive_first: bool,
    pub multiline_input: bool,
    pub simple_io: bool,
    pub cont_batching: bool,
    pub input_prefix_bos: bool,
    pub ignore_eos: bool,
    pub instruct: bool,
    pub logits_all: bool,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub numa: bool,
    pub verbose_prompt: bool,
    pub infill: bool,

    // Multimodal models
    pub mmproj: String,
    pub image: String,
}

impl Default for GptParams {
    fn default() -> Self {
        Self {
            seed: u32::MAX,                      // Equivalent to -1 for a 32-bit integer
            n_threads: get_num_physical_cores(), // Implement or import `get_num_physical_cores`
            n_threads_batch: None,
            n_predict: None,
            n_ctx: 512,
            n_batch: 512,
            n_keep: 0,
            n_draft: 16,
            n_chunks: None,
            n_parallel: 1,
            n_sequences: 1,
            p_accept: 0.5,
            p_split: 0.1,
            n_gpu_layers: None,
            n_gpu_layers_draft: None,
            main_gpu: 0,
            tensor_split: [0.0; LLAMA_MAX_DEVICES], // Replace LLAMA_MAX_DEVICES with the actual value
            n_beams: 0,
            rope_freq_base: 0.0,
            rope_freq_scale: 0.0,
            yarn_ext_factor: -1.0,
            yarn_attn_factor: 1.0,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            yarn_orig_ctx: 0,
            rope_scaling_type: LlamaRopeScaling::Unspecified,
            sparams: LlamaSamplingParams::default(), // Implement Default for LlamaSamplingParams
            model: PathBuf::from("models/7B/ggml-model-f16.gguf"),
            model_draft: String::new(),
            model_alias: String::from("unknown"),
            prompt: String::new(),
            prompt_file: String::new(),
            path_prompt_cache: String::new(),
            input_prefix: String::new(),
            input_suffix: String::new(),
            antiprompt: Vec::new(),
            logdir: String::new(),
            lora_adapter: Vec::new(),
            lora_base: String::new(),
            ppl_stride: 0,
            ppl_output_type: 0,
            hellaswag: false,
            hellaswag_tasks: 400,
            mul_mat_q: true,
            memory_f16: true,
            random_prompt: false,
            use_color: false,
            interactive: false,
            prompt_cache_all: false,
            prompt_cache_ro: false,
            embedding: false,
            escape: false,
            interactive_first: false,
            multiline_input: false,
            simple_io: false,
            cont_batching: false,
            input_prefix_bos: false,
            ignore_eos: false,
            instruct: false,
            logits_all: false,
            use_mmap: true,
            use_mlock: false,
            numa: false,
            verbose_prompt: false,
            infill: false,
            mmproj: String::new(),
            image: String::new(),
        }
    }
}

fn get_num_physical_cores() -> u32 {
    // Not relevant to the exercise we're doing here
    4
}
