//! Types and methods used for constructing and running
//! the samplers used for generation.

use std::{
    error::Error,
    fmt,
    str::FromStr,
    sync::{Arc, Mutex},
};

use llm_samplers::{configure::*, prelude::*};

use crate::TokenId;

#[derive(Debug)]
struct ConfiguredSamplers {
    builder: SamplerChainBuilder,
    mirostat1: bool,
    mirostat2: bool,
    incompat_mirostat: bool,
}

impl Default for ConfiguredSamplers {
    fn default() -> Self {
        Self {
            builder: SamplerChainBuilder::from([
                (
                    "repetition",
                    SamplerSlot::new_chain(
                        || Box::new(SampleRepetition::default().penalty(1.30).last_n(64)),
                        [],
                    ),
                ),
                (
                    "freqpresence",
                    SamplerSlot::new_chain(
                        || Box::new(SampleFreqPresence::default().last_n(64)),
                        [],
                    ),
                ),
                (
                    "seqrepetition",
                    SamplerSlot::new_chain(|| Box::<SampleSeqRepetition>::default(), []),
                ),
                (
                    "topk",
                    SamplerSlot::new_single(
                        || Box::new(SampleTopK::default().k(40)),
                        Option::<SampleTopK>::None,
                    ),
                ),
                (
                    "tailfree",
                    SamplerSlot::new_single(
                        || Box::<SampleTailFree>::default(),
                        Option::<SampleTailFree>::None,
                    ),
                ),
                (
                    "locallytypical",
                    SamplerSlot::new_single(
                        || Box::<SampleLocallyTypical>::default(),
                        Option::<SampleLocallyTypical>::None,
                    ),
                ),
                (
                    "topp",
                    SamplerSlot::new_single(
                        || Box::new(SampleTopP::default().p(0.95)),
                        Option::<SampleTopP>::None,
                    ),
                ),
                (
                    "temperature",
                    SamplerSlot::new_single(
                        || Box::new(SampleTemperature::default().temperature(0.8)),
                        Option::<SampleTemperature>::None,
                    ),
                ),
                (
                    "mirostat1",
                    SamplerSlot::new_single(
                        || Box::<SampleMirostat1>::default(),
                        Option::<SampleMirostat1>::None,
                    ),
                ),
                (
                    "mirostat2",
                    SamplerSlot::new_single(
                        || Box::<SampleMirostat2>::default(),
                        Option::<SampleMirostat2>::None,
                    ),
                ),
            ]),
            mirostat1: false,
            mirostat2: false,
            incompat_mirostat: false,
        }
    }
}

impl ConfiguredSamplers {
    pub fn ensure_default_slots(&mut self) {
        self.builder.iter_mut().for_each(|(name, slot)| {
            let mirostat = self.mirostat1 || self.mirostat2;
            match name as &str {
                "temperature" | "repetition" => slot.ensure_present(),
                "topp" | "topk" if !mirostat => slot.ensure_present(),
                _ => (),
            }
        });

        if !(self.mirostat1 || self.mirostat2) {
            self.builder += (
                "randdistrib".to_string(),
                SamplerSlot::new_static(|| Box::<SampleRandDistrib>::default()),
            )
        }
    }

    pub fn ensure_valid(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        if self.mirostat1 && self.mirostat2 {
            Err(Box::<dyn Error + Send + Sync>::from(
                "Cannot enable both Mirostat 1 and Mirostat 2 samplers",
            ))?
        } else if (self.mirostat1 || self.mirostat2) && self.incompat_mirostat {
            Err(Box::<dyn Error + Send + Sync>::from(
                "Cannot enable top-p, top-k, locally typical or tail free samplers with Mirostat 1 or 2",
            ))?
        }
        Ok(())
    }
}

impl FromStr for ConfiguredSamplers {
    type Err = Box<dyn Error + Send + Sync + 'static>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut result = Self::default();

        let s = s.trim().to_lowercase();
        let opts = s
            .split(|c: char| c == '/' || c.is_whitespace())
            .filter(|s| !s.is_empty())
            .map(|s| {
                if let Some((name, opts)) = s.split_once(':') {
                    (
                        name.trim()
                            .chars()
                            .filter(|c| *c != '_' && *c != '-')
                            .collect(),
                        opts.trim(),
                    )
                } else {
                    (s.trim().to_string(), "")
                }
            })
            .inspect(|(name, _slot)| match name.as_str() {
                "mirostat1" => result.mirostat1 = true,
                "mirostat2" => result.mirostat2 = true,
                "topp" | "topk" | "locallytypical" | "tailfree" => result.incompat_mirostat = true,
                _ => (),
            })
            .collect::<Vec<_>>();

        opts.into_iter()
            .try_for_each(|(name, args)| result.builder.configure(name, args))?;

        result.ensure_default_slots();
        result.ensure_valid()?;

        Ok(result)
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
#[allow(clippy::type_complexity)]
pub fn build_sampler(
    n_vocab: usize,
    bias: &[(TokenId, f32)],
    args: &[impl AsRef<str>],
) -> Result<Arc<Mutex<dyn Sampler<TokenId, f32>>>, Box<dyn std::error::Error + Send + Sync>> {
    let mut samplers = SamplerChain::new();

    if !bias.is_empty() {
        samplers += SampleFlatBias::new(bias.iter().copied());
    }

    let mut sampler_options = args
        .iter()
        .map(|s| s.as_ref().trim())
        .filter(|s| !s.is_empty())
        .map(|s| "/".to_string() + s)
        .collect::<String>();
    if sampler_options.contains("/mirostat1") {
        sampler_options += &format!("/mirostat1:n_vocab={n_vocab}");
    }
    let configured_samplers = ConfiguredSamplers::from_str(&sampler_options)?.builder;
    samplers += configured_samplers.into_chain();
    Ok(Arc::new(Mutex::new(samplers)))
}

/// Get the default sampler chain.
pub fn default_samplers() -> Arc<Mutex<dyn Sampler<TokenId, f32>>> {
    let mut result = ConfiguredSamplers::default();
    result.ensure_default_slots();
    Arc::new(Mutex::new(result.builder.into_chain()))
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
