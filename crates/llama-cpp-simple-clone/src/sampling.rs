use std::collections::HashMap;

use crate::llama::LlamaToken;

/// Sampling parameters for LLaMa.
#[derive(Debug)]
pub struct LlamaSamplingParams {
    /// Number of previous tokens to remember.
    pub n_prev: i32,
    /// If greater than 0, output the probabilities of top `n_probs` tokens.
    pub n_probs: i32,
    /// Less than or equal to 0 to use vocab size.
    pub top_k: i32,
    /// 1.0 = disabled.
    pub top_p: f32,
    /// 0.0 = disabled.
    pub min_p: f32,
    /// 1.0 = disabled.
    pub tfs_z: f32,
    /// 1.0 = disabled.
    pub typical_p: f32,
    /// 1.0 = disabled.
    pub temp: f32,
    /// Last n tokens to penalize (0 = disable penalty, -1 = context size).
    pub penalty_last_n: i32,
    /// 1.0 = disabled.
    pub penalty_repeat: f32,
    /// 0.0 = disabled.
    pub penalty_freq: f32,
    /// 0.0 = disabled.
    pub penalty_present: f32,
    /// 0 = disabled, 1 = mirostat, 2 = mirostat 2.0.
    pub mirostat: i32,
    /// Target entropy.
    pub mirostat_tau: f32,
    /// Learning rate.
    pub mirostat_eta: f32,
    /// Consider newlines as a repeatable token.
    pub penalize_nl: bool,
    /// Optional BNF-like grammar to constrain sampling.
    pub grammar: String,
    /// String to help guidance.
    pub cfg_negative_prompt: String,
    /// How strong is guidance.
    pub cfg_scale: f32,
    /// Logit bias for specific tokens.
    pub logit_bias: HashMap<LlamaToken, f32>,
}

impl Default for LlamaSamplingParams {
    fn default() -> Self {
        Self {
            n_prev: 64,
            n_probs: 0,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            tfs_z: 1.0,
            typical_p: 1.0,
            temp: 0.8,
            penalty_last_n: 64,
            penalty_repeat: 1.1,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            mirostat: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            penalize_nl: true,
            grammar: String::new(),
            cfg_negative_prompt: String::new(),
            cfg_scale: 1.0,
            logit_bias: HashMap::new(),
        }
    }
}
