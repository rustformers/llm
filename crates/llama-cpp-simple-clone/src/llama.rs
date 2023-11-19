use anyhow::Context;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::{
    alloc::{alloc, dealloc, Layout},
    collections::{BTreeMap, HashMap, HashSet},
    ffi::{c_char, c_void},
    path::Path,
    ptr::{null_mut, NonNull},
};

pub type LlamaToken = i32;
pub type LlamaPos = i32;
pub type LlamaSeqId = i32;

#[cfg(feature = "cublas")]
pub const LLAMA_MAX_DEVICES: usize = ggml::accelerator::cublas::MAX_DEVICES as usize;
#[cfg(not(feature = "cublas"))]
pub const LLAMA_MAX_DEVICES: usize = 1;

#[derive(Copy, Clone, Debug)]
pub enum LlamaRopeScaling {
    Unspecified,
    None,
    Linear,
    Yarn,
}

#[derive(Debug, Clone, Copy)]
pub enum LlamaTokenType {
    Undefined,
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

/// Parameters for a llama model.
pub struct LlamaModelParams {
    pub n_gpu_layers: i32,
    pub main_gpu: i32,
    pub tensor_split: Option<NonNull<f32>>,
    pub progress_callback: Option<LlamaProgressCallback>,
    pub progress_callback_user_data: Option<NonNull<c_void>>,
    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_mlock: bool,
}

impl Default for LlamaModelParams {
    fn default() -> Self {
        Self {
            n_gpu_layers: 0,
            main_gpu: 0,
            tensor_split: None,
            progress_callback: None,
            progress_callback_user_data: None,
            vocab_only: false,
            use_mmap: true,
            use_mlock: false,
        }
    }
}

#[derive(Clone)]
/// Batch structure for llama.
pub struct LlamaBatch {
    pub tokens: Vec<LlamaToken>,
    pub embd: Vec<f32>,
    pub pos: Vec<LlamaPos>,
    pub seq_id: Vec<LlamaSeqId>,
    pub logits: Vec<bool>,
    pub all_pos_0: LlamaPos,
    pub all_pos_1: LlamaPos,
    pub all_seq_id: LlamaSeqId,
}
impl LlamaBatch {
    pub fn new(n_tokens: i32, embd: i32, n_seq_max: i32) -> Self {
        todo!()
    }

    pub fn add(
        &mut self,
        id: LlamaToken,         // Assuming LlamaToken is the type for token ID
        pos: LlamaPos,          // Assuming LlamaPos is the type for position
        seq_ids: &[LlamaSeqId], // Using a slice of LlamaSeqId
        logits: bool,
    ) {
        todo!()
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
    }
}
impl Drop for LlamaBatch {
    fn drop(&mut self) {
        todo!("batch_free")
    }
}

/// Token data structure for llama.
pub struct LlamaTokenData {
    pub token_id: LlamaToken,
    pub logit: f32,
    pub probability: f32,
}

pub struct LlamaTokenDataArray {
    pub data: Vec<LlamaTokenData>, // Assuming LlamaTokenData is another struct you have
    pub sorted: bool,
}

pub type LlamaProgressCallback = fn(progress: f32, ctx: &mut c_void);

pub enum EModel {
    ModelUnknown,
    Model1B,
    Model3B,
    Model7B,
    Model8B,
    Model13B,
    Model15B,
    Model30B,
    Model34B,
    Model40B,
    Model65B,
    Model70B,
}

pub enum LlmArch {
    LLAMA,
    FALCON,
    BAICHUAN,
    GPT2,
    GPTJ,
    GPTNEOX,
    MPT,
    STARCODER,
    PERSIMMON,
    REFACT,
    BLOOM,
    STABLELM,
    UNKNOWN,
}

/// model file types
#[allow(non_camel_case_types)]
pub enum LlamaFtype {
    AllF32 = 0,
    /// except 1d tensors
    MostlyF16 = 1,
    /// except 1d tensors
    MostlyQ4_0 = 2,
    /// except 1d tensors
    MostlyQ4_1 = 3,
    /// tok_embeddings.weight and output.weight are F16
    MostlyQ4_1SomeF16 = 4,
    /// support has been removed
    // MostlyQ4_2       = 5,
    /// support has been removed
    // MostlyQ4_3       = 6,
    /// except 1d tensors
    MostlyQ8_0 = 7,
    /// except 1d tensors
    MostlyQ5_0 = 8,
    /// except 1d tensors
    MostlyQ5_1 = 9,
    /// except 1d tensors
    MostlyQ2_K = 10,
    /// except 1d tensors
    MostlyQ3_K_S = 11,
    /// except 1d tensors
    MostlyQ3_K_M = 12,
    /// except 1d tensors
    MostlyQ3_K_L = 13,
    /// except 1d tensors
    MostlyQ4_K_S = 14,
    /// except 1d tensors
    MostlyQ4_K_M = 15,
    /// except 1d tensors
    MostlyQ5_K_S = 16,
    /// except 1d tensors
    MostlyQ5_K_M = 17,
    /// except 1d tensors
    MostlyQ6_K = 18,

    Guessed = 1024, // not specified in the model file
}

#[derive(Debug, Clone, Copy)]
pub struct LlamaHparams {
    pub vocab_only: bool,
    pub n_vocab: u32,
    pub n_ctx_train: u32,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub n_layer: u32,
    pub n_rot: u32,
    pub n_ff: u32,
    pub f_norm_eps: f32,
    pub f_norm_rms_eps: f32,
    pub rope_freq_base_train: f32,
    pub rope_freq_scale_train: f32,
    pub n_yarn_orig_ctx: u32,
    pub rope_scaling_type_train: i8, // 3-bit field in C++, but Rust does not support bitfields
    pub rope_finetuned: bool,
    pub f_clamp_kqv: f32,
    pub f_max_alibi_bias: f32,
}

impl PartialEq for LlamaHparams {
    fn eq(&self, other: &Self) -> bool {
        const EPSILON: f32 = 1e-9;

        self.vocab_only == other.vocab_only
            && self.n_vocab == other.n_vocab
            && self.n_ctx_train == other.n_ctx_train
            && self.n_embd == other.n_embd
            && self.n_head == other.n_head
            && self.n_head_kv == other.n_head_kv
            && self.n_layer == other.n_layer
            && self.n_rot == other.n_rot
            && self.n_ff == other.n_ff
            && self.rope_finetuned == other.rope_finetuned
            && self.n_yarn_orig_ctx == other.n_yarn_orig_ctx
            && is_float_close(self.f_norm_eps, other.f_norm_eps, EPSILON)
            && is_float_close(self.f_norm_rms_eps, other.f_norm_rms_eps, EPSILON)
            && is_float_close(
                self.rope_freq_base_train,
                other.rope_freq_base_train,
                EPSILON,
            )
            && is_float_close(
                self.rope_freq_scale_train,
                other.rope_freq_scale_train,
                EPSILON,
            )
    }
}

impl Eq for LlamaHparams {}

impl LlamaHparams {
    pub fn n_gqa(&self) -> u32 {
        self.n_head / self.n_head_kv
    }

    pub fn n_embd_head(&self) -> u32 {
        self.n_embd / self.n_head
    }

    pub fn n_embd_gqa(&self) -> u32 {
        self.n_embd / self.n_gqa()
    }
}

fn is_float_close(a: f32, b: f32, abs_tol: f32) -> bool {
    // Check for non-negative tolerance
    if abs_tol < 0.0 {
        panic!("Tolerance must be non-negative");
    }

    // Exact equality check
    if a == b {
        return true;
    }

    // Check for infinities
    if a.is_infinite() || b.is_infinite() {
        return false;
    }

    // Regular comparison using the provided absolute tolerance
    (b - a).abs() <= abs_tol
}

#[derive(Debug, Clone)]
pub struct LlamaVocab {
    pub token_to_id: HashMap<String, i32>,
    pub id_to_token: Vec<TokenData>,
    pub special_tokens_cache: HashMap<String, i32>,
    pub bpe_ranks: BTreeMap<(String, String), i32>,
    pub special_bos_id: i32,
    pub special_eos_id: i32,
    pub special_unk_id: i32,
    pub special_sep_id: i32,
    pub special_pad_id: i32,
    pub special_add_bos: i32,
    pub special_add_eos: i32,
    pub linefeed_id: i32,
    pub special_prefix_id: i32,
    pub special_middle_id: i32,
    pub special_suffix_id: i32,
    pub special_eot_id: i32,
    pub type_: LlamaVocabType,
}

#[derive(Debug, Clone)]
pub struct TokenData {
    pub text: String,
    pub score: f32,
    pub type_: LlamaTokenType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaVocabType {
    LlamaVocabTypeSpm,
    // Other variants as needed
}

impl LlamaVocab {
    pub fn find_bpe_rank(&self, token_left: &str, token_right: &str) -> i32 {
        assert!(!token_left.contains(' '));
        assert!(!token_left.contains('\n'));
        assert!(!token_right.contains(' '));
        assert!(!token_right.contains('\n'));

        *self
            .bpe_ranks
            .get(&(token_left.to_string(), token_right.to_string()))
            .unwrap_or(&-1)
    }
}

pub struct LlamaBuffer {
    data: *mut u8,
    size: usize,
    fallback: bool,
}

impl LlamaBuffer {
    pub fn new() -> Self {
        Self {
            data: null_mut(),
            size: 0,
            fallback: false,
        }
    }

    pub fn resize(&mut self, n: usize) {
        unsafe {
            self.free_data();

            let layout = Layout::array::<u8>(n).unwrap();
            self.data = alloc(layout) as *mut u8;

            if self.data.is_null() {
                self.fallback = true;
                self.data = malloc(n) as *mut u8;
            } else {
                self.fallback = false;
            }

            assert!(!self.data.is_null());
            self.size = n;
        }
    }

    unsafe fn free_data(&mut self) {
        if !self.data.is_null() {
            let layout = Layout::array::<u8>(self.size).unwrap();

            if self.fallback {
                free(self.data as *mut c_void);
            } else {
                dealloc(self.data, layout);
            }

            self.data = null_mut();
        }
    }
}

fn malloc(n: usize) -> *mut c_void {
    todo!()
}

fn free(ptr: *mut c_void) {
    todo!()
}

impl Drop for LlamaBuffer {
    fn drop(&mut self) {
        unsafe {
            self.free_data();
        }
    }
}

/// Represents a llama model.
pub struct LlamaModel {
    pub type_: EModel,
    pub arch: LlmArch,
    pub ftype: LlamaFtype,
    pub name: String,
    pub hparams: LlamaHparams,
    pub vocab: LlamaVocab,
    pub tok_embd: ggml::Tensor,
    pub pos_embd: ggml::Tensor,
    pub tok_norm: ggml::Tensor,
    pub tok_norm_b: ggml::Tensor,
    pub output_norm: ggml::Tensor,
    pub output_norm_b: ggml::Tensor,
    pub output: ggml::Tensor,
    pub layers: Vec<LlamaLayer>,
    pub n_gpu_layers: i32,
    pub gguf_kv: HashMap<String, String>,
    pub ctx: ggml::Context,
    pub buf: LlamaBuffer,
    pub mapping: Option<memmap2::Mmap>,
    // pub mlock_buf: LlamaMlock,
    // pub mlock_mmap: LlamaMlock,
    pub tensors_by_name: Vec<(String, ggml::Tensor)>,
    pub t_load_us: i64,
    pub t_start_us: i64,
}

pub struct LlamaLayer {
    // normalization
    attn_norm: ggml::Tensor,
    attn_norm_b: ggml::Tensor,
    attn_norm_2: ggml::Tensor,
    attn_norm_2_b: ggml::Tensor,
    attn_q_norm: ggml::Tensor,
    attn_q_norm_b: ggml::Tensor,
    attn_k_norm: ggml::Tensor,
    attn_k_norm_b: ggml::Tensor,

    // attention
    wq: ggml::Tensor,
    wk: ggml::Tensor,
    wv: ggml::Tensor,
    wo: ggml::Tensor,
    wqkv: ggml::Tensor,

    // attention bias
    bo: ggml::Tensor,
    bqkv: ggml::Tensor,

    // normalization
    ffn_norm: ggml::Tensor,
    ffn_norm_b: ggml::Tensor,

    // ff
    ffn_gate: ggml::Tensor, // w1
    ffn_down: ggml::Tensor, // w2
    ffn_up: ggml::Tensor,   // w3

    // ff bias
    ffn_down_b: ggml::Tensor, // b2
    ffn_up_b: ggml::Tensor,   // b3
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        #[cfg(feature = "cublas")]
        if ggml_cublas_loaded() {
            for tensor in &self.tensors_by_name {
                unsafe { ggml_cuda_free_data(tensor.1.as_ptr()) };
            }
            ggml_cuda_free_scratch();
        }

        #[cfg(feature = "clblast")]
        for tensor in &self.tensors_by_name {
            unsafe { ggml_cl_free_data(tensor.1.as_ptr()) };
        }
    }
}

#[derive(Default)]
pub struct LlamaCparams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: u32,
    pub n_threads_batch: u32,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub n_yarn_orig_ctx: u32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub mul_mat_q: bool,
}

#[derive(Default, Clone)]
pub struct LlamaKvCell {
    pos: i32, // assuming llama_pos is an i32
    delta: i32,
    seq_id: HashSet<i32>, // assuming llama_seq_id is an i32
}

impl LlamaKvCell {
    pub fn has_seq_id(&self, id: i32) -> bool {
        self.seq_id.contains(&id)
    }
}

pub struct LlamaKvCache {
    has_shift: bool,
    head: u32,
    size: u32,
    n: u32,
    cells: Vec<LlamaKvCell>,
    k: ggml::Tensor,
    v: ggml::Tensor,
    ctx: ggml::Context,
}
impl LlamaKvCache {
    pub fn new(
        hparams: &LlamaHparams,
        wtype: ggml::Type,
        n_ctx: u32,
        n_gpu_layers: i32,
    ) -> anyhow::Result<Self> {
        let n_embd = hparams.n_embd_gqa();
        let n_layer = hparams.n_layer;

        let n_mem = (n_layer as i64) * (n_ctx as i64);
        let n_elements = ((n_embd as i64) * n_mem) as usize;

        let ggml_type_size = ggml::type_size(wtype) as usize;
        let ggml_tensor_overhead = ggml::tensor_overhead() as usize;

        let buffer_size = (2 * n_elements * ggml_type_size + 2 * ggml_tensor_overhead) as usize;

        let ctx = ggml::Context::new(ggml::ContextStorage::Buffer {
            buffer: ggml::Buffer::new(buffer_size),
            allocate: false,
        });

        let k = ctx.new_tensor_1d(wtype, n_elements).set_name("cache_k");
        let v = ctx.new_tensor_1d(wtype, n_elements).set_name("cache_v");

        let cache = LlamaKvCache {
            has_shift: false,
            head: 0,
            size: n_ctx,
            n: 0,
            cells: vec![Default::default(); n_ctx as usize],
            k,
            v,
            ctx,
        };

        // Handle GPU layers
        #[cfg(feature = "cublas")]
        if ggml_cublas_loaded() && n_gpu_layers > n_layer as i32 {
            // GPU-related logic
            // ...
        }

        Ok(cache)
    }
}

impl Drop for LlamaKvCache {
    fn drop(&mut self) {
        #[cfg(feature = "cublas")]
        if ggml_cublas_loaded() {
            if let Some(k) = self.k {
                unsafe { ggml_cuda_free_data(k.as_ptr()) };
            }
            if let Some(v) = self.v {
                unsafe { ggml_cuda_free_data(v.as_ptr()) };
            }
        }
    }
}

pub struct LlamaContext<'a> {
    cparams: LlamaCparams,
    pub model: &'a LlamaModel,
    kv_self: LlamaKvCache,
    rng: StdRng,
    has_evaluated_once: bool,
    t_start_us: i64,
    t_load_us: i64,
    t_sample_us: i64,
    t_p_eval_us: i64,
    t_eval_us: i64,
    n_sample: i32,
    n_p_eval: i32,
    n_eval: i32,
    logits: Vec<f32>,
    logits_all: bool,
    embedding: Vec<f32>,
    work_buffer: Vec<u8>,
    buf_compute: LlamaBuffer,
    buf_alloc: LlamaBuffer,
    alloc: Option<NonNull<ggml::sys::ggml_allocr>>,
    #[cfg(feature = "metal")]
    ctx_metal: Option<GgmlMetalContext>,
    #[cfg(feature = "mpi")]
    ctx_mpi: Option<GgmlMpiContext>,
}
pub struct LlamaContextParams {
    pub seed: u32, // Note: Rust doesn't have unsigned -1, so you might need to use Option<u32> or a special value to indicate 'random'
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: u32,
    pub n_threads_batch: u32,
    pub rope_scaling_type: LlamaRopeScaling,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub mul_mat_q: bool,
    pub f16_kv: bool,
    pub logits_all: bool,
    pub embedding: bool,
}

impl Default for LlamaContextParams {
    fn default() -> Self {
        Self {
            seed: 42,
            n_ctx: 512,
            n_batch: 512,
            n_threads: 4,
            n_threads_batch: 4,
            rope_scaling_type: LlamaRopeScaling::Unspecified,
            rope_freq_base: 0.0,
            rope_freq_scale: 0.0,
            yarn_ext_factor: -1.0,
            yarn_attn_factor: 1.0,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            yarn_orig_ctx: 0,
            mul_mat_q: true,
            f16_kv: true,
            logits_all: false,
            embedding: false,
        }
    }
}

impl<'a> LlamaContext<'a> {
    pub fn new(model: &'a LlamaModel, params: LlamaContextParams) -> anyhow::Result<Self> {
        let hparams = &model.hparams;

        let cparams = LlamaCparams {
            n_ctx: if params.n_ctx == 0 {
                hparams.n_ctx_train
            } else {
                params.n_ctx
            },
            n_batch: params.n_batch,
            n_threads: params.n_threads,
            n_threads_batch: params.n_threads_batch,
            rope_freq_base: if params.rope_freq_base == 0.0 {
                hparams.rope_freq_base_train
            } else {
                params.rope_freq_base
            },
            rope_freq_scale: if params.rope_freq_scale == 0.0 {
                hparams.rope_freq_scale_train
            } else {
                params.rope_freq_scale
            },
            n_yarn_orig_ctx: if params.yarn_orig_ctx == 0 {
                if hparams.n_yarn_orig_ctx == 0 {
                    hparams.n_ctx_train
                } else {
                    hparams.n_yarn_orig_ctx
                }
            } else {
                params.yarn_orig_ctx
            },
            yarn_ext_factor: params.yarn_ext_factor,
            yarn_attn_factor: params.yarn_attn_factor,
            yarn_beta_fast: params.yarn_beta_fast,
            yarn_beta_slow: params.yarn_beta_slow,
            mul_mat_q: params.mul_mat_q,
        };

        let memory_type = if params.f16_kv {
            ggml::Type::F16
        } else {
            ggml::Type::F32
        };

        let n_ctx = cparams.n_ctx;

        Ok(Self {
            cparams,
            model,
            kv_self: LlamaKvCache::new(&model.hparams, memory_type, n_ctx, model.n_gpu_layers)
                .context("failed to create kv cache for self-attention")?,
            rng: StdRng::from_entropy(),
            has_evaluated_once: false,
            t_start_us: model.t_start_us,
            t_load_us: model.t_load_us,
            t_sample_us: 0,
            t_p_eval_us: 0,
            t_eval_us: 0,
            n_sample: 0,
            n_p_eval: 0,
            n_eval: 0,
            logits: Vec::new(),
            logits_all: false,
            embedding: Vec::new(),
            work_buffer: Vec::new(),
            buf_compute: LlamaBuffer::new(), // Assuming LlamaBuffer has a constructor
            buf_alloc: LlamaBuffer::new(),
            alloc: None,
            #[cfg(feature = "metal")]
            ctx_metal: None,
            #[cfg(feature = "mpi")]
            ctx_mpi: None,
        })
    }
}

impl Drop for LlamaContext<'_> {
    fn drop(&mut self) {
        #[cfg(feature = "metal")]
        if let Some(ctx_metal) = self.ctx_metal {
            ggml_metal_free(ctx_metal);
        }

        if let Some(alloc) = self.alloc {
            unsafe {
                ggml::sys::ggml_allocr_free(alloc.as_ptr());
            }
        }

        // The other fields are automatically dropped by Rust
    }
}

/// Initialize the llama + ggml backend.
pub fn backend_init(numa: bool) {
    todo!()
}

/// Load a model from file.
pub fn load_model_from_file(
    path_model: &Path,
    params: LlamaModelParams,
) -> anyhow::Result<Box<LlamaModel>> {
    todo!()
}

/// Create a new context with a model.
pub fn new_context_with_model<'a>(
    model: &'a LlamaModel,
    params: LlamaContextParams,
) -> anyhow::Result<LlamaContext<'a>> {
    LlamaContext::new(model, params)
}

pub fn tokenize(
    ctx: &LlamaContext,
    text: &str,
    add_bos: bool,
    special: Option<bool>,
) -> Vec<LlamaToken> {
    tokenize_with_model(ctx.model, text, add_bos, special.unwrap_or(false))
}

fn tokenize_with_model(
    model: &LlamaModel,
    text: &str,
    add_bos: bool,
    special: bool,
) -> Vec<LlamaToken> {
    todo!()
}

fn tokenize_with_all(
    model: &LlamaModel,
    text: &str,
    tokens: &mut [LlamaToken], // Assuming LlamaToken is a type you have defined
    add_bos: bool,
    special: bool,
) -> usize {
    todo!()
}

/// Get the context size of a llama context.
pub fn n_ctx(ctx: &LlamaContext) -> usize {
    todo!()
}

/// Decode using a llama context and batch.
pub fn decode(ctx: &mut LlamaContext, batch: LlamaBatch) -> anyhow::Result<()> {
    todo!()
}

/// Get the vocabulary size of a model.
pub fn n_vocab(model: &LlamaModel) -> usize {
    todo!()
}

/// Get logits for the ith token.
pub fn get_logits_ith<'a, 'b>(ctx: &'a mut LlamaContext<'b>, i: usize) -> &'a mut [f32] {
    todo!()
}

/// Select the token with the highest probability.
pub fn sample_token_greedy<'a, 'b, 'c>(
    ctx: &'a mut LlamaContext<'b>,
    candidates: &'c mut LlamaTokenDataArray,
) -> LlamaToken {
    todo!()
}

/// Get end-of-sentence token.
pub fn token_eos(model: &LlamaModel) -> LlamaToken {
    todo!()
}

// tokenizes a token into a piece
// should work similar to Python's `tokenizer.id_to_piece`
pub fn token_to_piece(ctx: &LlamaContext, token: LlamaToken) -> String {
    todo!();
}

/// Token Id to Piece conversion.
pub fn token_to_piece_with_buffer(
    model: &LlamaModel,
    token: LlamaToken,
    buf: &mut c_char,
    length: i32,
) -> i32 {
    todo!()
}

/// Print timings for a llama context.
pub fn print_timings(ctx: &mut LlamaContext) {
    todo!()
}

/// Free the backend resources.
pub fn backend_free() {
    todo!()
}
