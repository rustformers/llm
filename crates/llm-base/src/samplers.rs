//! Types and methods used for constructing and running
//! the samplers used for generation.

use std::{
    error::Error,
    fmt,
    str::FromStr,
    sync::{Arc, Mutex},
};

use llm_samplers::prelude::*;

use crate::TokenId;

/// This structure holds specific samplers that have already
/// been configured and provides some convenience methods
/// for constructing samplers with default settings.
#[derive(Debug, Default, Clone)]
pub struct ConfiguredSamplers {
    bias: Option<SampleFlatBias<TokenId, f32>>,
    repetition: Option<SampleRepetition<TokenId, f32>>,
    freq_presence: Option<SampleFreqPresence<TokenId, f32>>,
    top_k: Option<SampleTopK>,
    tail_free: Option<SampleTailFree<f32>>,
    locally_typical: Option<SampleLocallyTypical<f32>>,
    top_p: Option<SampleTopP<f32>>,
    temperature: Option<SampleTemperature<f32>>,
    mirostat1: Option<SampleMirostat1<TokenId, f32>>,
    mirostat2: Option<SampleMirostat2<TokenId, f32>>,
}

impl ConfiguredSamplers {
    /// Sets the token bias list
    pub fn set_token_bias(&mut self, bias: impl IntoIterator<Item = (TokenId, f32)>) {
        self.bias = Some(SampleFlatBias::new(bias))
    }

    /// Creates a temperature new sampler with default options.
    pub fn new_temperature() -> SampleTemperature<f32> {
        SampleTemperature::default().temperature(0.8)
    }

    /// Creates a new repetition sampler with default options.
    pub fn new_repetition() -> SampleRepetition<TokenId, f32> {
        SampleRepetition::default().penalty(1.30).last_n(64)
    }

    /// Creates a new frequency/presence sampler with default options.
    pub fn new_freq_presence() -> SampleFreqPresence<TokenId, f32> {
        SampleFreqPresence::default()
            .frequency(0.0)
            .presence(0.0)
            .last_n(64)
    }

    /// Creates a new top k sampler with default options.
    pub fn new_top_k() -> SampleTopK {
        SampleTopK::default().k(40)
    }

    /// Creates a new top p sampler with default options.
    pub fn new_top_p() -> SampleTopP<f32> {
        SampleTopP::default().p(0.95)
    }

    /// Creates a new tail free sampler with default options.
    pub fn new_tail_free() -> SampleTailFree<f32> {
        SampleTailFree::default().z(1.0)
    }

    /// Creates a new locally typical sampler with default options.
    pub fn new_locally_typical() -> SampleLocallyTypical<f32> {
        SampleLocallyTypical::default().p(1.0)
    }

    /// Creates a new mirostat 1 sampler with default options.
    pub fn new_mirostat1() -> SampleMirostat1<TokenId, f32> {
        SampleMirostat1::default().eta(0.1).tau(5.0)
    }

    /// Creates a new mirostat 2 sampler with default options.
    pub fn new_mirostat2() -> SampleMirostat2<TokenId, f32> {
        SampleMirostat2::default().eta(0.1).tau(5.0)
    }
}

impl From<ConfiguredSamplers> for SamplerChain<TokenId, f32> {
    fn from(val: ConfiguredSamplers) -> Self {
        let mut chain = SamplerChain::new();

        if let Some(sampler) = val.bias {
            chain += sampler;
        }
        if let Some(sampler) = val.repetition {
            chain += sampler;
        }
        if let Some(sampler) = val.freq_presence {
            chain += sampler;
        }

        if let Some(mirosampler) = val.mirostat1 {
            if let Some(sampler) = val.temperature {
                chain += sampler;
            }
            chain += mirosampler;
            return chain;
        } else if let Some(mirosampler) = val.mirostat2 {
            if let Some(sampler) = val.temperature {
                chain += sampler;
            }
            chain += mirosampler;
            return chain;
        }

        if let Some(sampler) = val.top_k {
            chain += sampler;
        }
        if let Some(sampler) = val.tail_free {
            chain += sampler;
        }
        if let Some(sampler) = val.locally_typical {
            chain += sampler;
        }
        if let Some(sampler) = val.top_p {
            chain += sampler;
        }
        if let Some(sampler) = val.temperature {
            chain += sampler;
        }
        chain += SampleRandDistrib::new();
        chain
    }
}

impl ConfiguredSamplers {
    fn from_args(args: Vec<ConfiguredSampler>, n_vocab: usize) -> Self {
        let mut result = Self::default();

        args.into_iter().for_each(|arg| match arg {
            ConfiguredSampler::Repetition(sampler) => result.repetition = Some(sampler),
            ConfiguredSampler::FreqPresence(sampler) => result.freq_presence = Some(sampler),
            ConfiguredSampler::TopK(sampler) => result.top_k = Some(sampler),
            ConfiguredSampler::TailFree(sampler) => result.tail_free = Some(sampler),
            ConfiguredSampler::LocallyTypical(sampler) => result.locally_typical = Some(sampler),
            ConfiguredSampler::TopP(sampler) => result.top_p = Some(sampler),
            ConfiguredSampler::Temperature(sampler) => result.temperature = Some(sampler),
            ConfiguredSampler::Mirostat1(sampler) => {
                result.mirostat1 = Some(sampler.n_vocab(n_vocab))
            }
            ConfiguredSampler::Mirostat2(sampler) => result.mirostat2 = Some(sampler),
        });

        if result.temperature.is_none() {
            result.temperature = Some(ConfiguredSamplers::new_temperature())
        }
        if result.repetition.is_none() {
            result.repetition = Some(ConfiguredSamplers::new_repetition())
        }
        if result.mirostat1.is_some() || result.mirostat2.is_some() {
            return result;
        }

        if result.top_k.is_none() {
            result.top_k = Some(ConfiguredSamplers::new_top_k())
        }
        if result.top_p.is_none() {
            result.top_p = Some(ConfiguredSamplers::new_top_p())
        }
        result
    }
}

/// A specific type of sampler that has been configured
#[derive(Clone, Debug)]
pub enum ConfiguredSampler {
    /// Holds the configured sampler
    Repetition(SampleRepetition<TokenId, f32>),
    /// Holds the configured sampler
    FreqPresence(SampleFreqPresence<TokenId, f32>),
    /// Holds the configured sampler
    TopK(SampleTopK),
    /// Holds the configured sampler
    TailFree(SampleTailFree<f32>),
    /// Holds the configured sampler
    LocallyTypical(SampleLocallyTypical<f32>),
    /// Holds the configured sampler
    TopP(SampleTopP<f32>),
    /// Holds the configured sampler
    Temperature(SampleTemperature<f32>),
    /// Holds the configured sampler
    Mirostat1(SampleMirostat1<TokenId, f32>),
    /// Holds the configured sampler
    Mirostat2(SampleMirostat2<TokenId, f32>),
}

impl FromStr for ConfiguredSampler {
    type Err = Box<dyn Error + Send + Sync + 'static>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (name, args) = if let Some(val) = s.split_once(':') {
            val
        } else {
            return Err(Box::from("Bad format for sampler argument"));
        };

        Ok(match name.trim() {
            "repetition" => ConfiguredSamplers::new_repetition()
                .configure(args)
                .map(Self::Repetition)?,
            "frequency" | "presence" | "freqpresence" => ConfiguredSamplers::new_freq_presence()
                .configure(args)
                .map(Self::FreqPresence)?,
            "topk" | "top_k" => {
                ConfigurableSampler::<_, f32>::configure(ConfiguredSamplers::new_top_k(), args)
                    .map(Self::TopK)?
            }
            "topp" | "top_p" => ConfiguredSamplers::new_top_p()
                .configure(args)
                .map(Self::TopP)?,
            "temperature" | "temp" => ConfigurableSampler::<TokenId, _>::configure(
                ConfiguredSamplers::new_temperature(),
                args,
            )
            .map(Self::Temperature)?,
            "tailfree" | "tail_free" => ConfiguredSamplers::new_tail_free()
                .configure(args)
                .map(Self::TailFree)?,
            "locallytypical" | "locally_typical" => ConfiguredSamplers::new_locally_typical()
                .configure(args)
                .map(Self::LocallyTypical)?,
            "mirostat1" => ConfiguredSamplers::new_mirostat1()
                .configure(args)
                .map(Self::Mirostat1)?,
            "mirostat2" => ConfiguredSamplers::new_mirostat2()
                .configure(args)
                .map(Self::Mirostat2)?,
            unknown => return Err(Box::from(format!("Unknown sampler: {unknown}"))),
        })
    }
}

/// Sample a token. This convenience function handles building
/// the sampler resources and logits objects the sampler needs.
pub fn sample_token(
    mut sampler: impl Sampler<TokenId, f32>,
    rng: &mut impl rand::Rng,
    previous_tokens: &[TokenId],
    last_logits: impl IntoIterator<Item = f32>,
) -> Result<TokenId, Box<dyn Error + Send + Sync>> {
    Logits::try_from_iter(last_logits.into_iter())?
        .sample_token(
            &mut SamplerResources {
                previous_tokens,
                rng,
            },
            &mut sampler,
        )?
        .ok_or_else(|| Box::from("sampler did not return a token"))
}

/// Build a sampler with the supplied options, vocab size and token bias list.
pub fn build_sampler(
    n_vocab: usize,
    bias: &[(TokenId, f32)],
    args: Vec<ConfiguredSampler>,
) -> Arc<Mutex<dyn Sampler<TokenId, f32>>> {
    let mut settings = ConfiguredSamplers::from_args(args, n_vocab);
    if !bias.is_empty() {
        settings.set_token_bias(bias.iter().copied())
    }
    let chain: SamplerChain<TokenId, f32> = settings.into();
    Arc::new(Mutex::new(chain))
}

// Struct used to temporarily hold resources for the `llm_samplers`
// sampler.
struct SamplerResources<'pt, 'r> {
    previous_tokens: &'pt [TokenId],
    rng: &'r mut dyn rand::RngCore,
}

impl<'pt, 'r> fmt::Debug for SamplerResources<'pt, 'r> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SamplerResources")
            .field("previous_tokens", &self.previous_tokens)
            .field("rng", &"<dyn RngCore>")
            .finish()
    }
}

impl<'pt, 'r> HasSamplerResources for SamplerResources<'pt, 'r> {
    type TokenId = TokenId;

    fn with_rng_mut(
        &mut self,
        fun: &mut dyn FnMut(&mut dyn rand::RngCore),
    ) -> Result<(), SamplerError> {
        fun(self.rng);
        Ok(())
    }

    fn with_last_tokens(&self, fun: &mut dyn FnMut(&[Self::TokenId])) -> Result<(), SamplerError> {
        fun(self.previous_tokens);
        Ok(())
    }
}
