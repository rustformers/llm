//! Types and methods used for constructing and running
//! the samplers used for generation.
//!
//! The `llm-samplers` crate is also re-exported here for convenient use as `llm_samplers`.

use std::{
    error::Error,
    fmt,
    str::FromStr,
    sync::{Arc, Mutex},
};

use thiserror::Error;

pub use llm_samplers;

use llm_samplers::{configure::*, prelude::*};

use crate::TokenId;

#[derive(Debug, Error)]
/// Errors related to constructing samplers from string definitions.
pub enum SamplerConfigurationError {
    #[error("An incompatible combination of samplers was requested: {0}")]
    /// Not all combinations of samplers are valid. This error will be returned
    /// when an invalid combination is specified.
    SamplerCombinationError(String),

    #[error("Error configuring sampler {name}: {err}")]
    /// The sampler name was unknown or the options to it were invalid.
    BuildSamplerError {
        /// Name of the sampler that failed.
        name: String,
        /// The actual error.
        err: Box<dyn Error + Send + Sync + 'static>,
    },
}

#[derive(Debug, Error)]
/// Errors that occured during sampling.
pub enum SamplingError {
    #[error("Sampling failed to produce a token")]
    /// Sampling didn't produce a token.
    NoToken,

    #[error("An error occured constructing logits for sampling: {0}")]
    /// Constructing logits failed. This can usually only happen if a logit is NaN.
    LogitsError(Box<dyn Error + Send + Sync + 'static>),

    #[error("An internal error occured during sampling: {0}")]
    /// Sampling failed.
    InternalSamplingError(Box<dyn Error + Send + Sync + 'static>),
}

#[derive(Debug)]
/// Used for configuring samplers dynamically from string definitions.
/// For example, commandline arguments. Constructing this structure manually is
/// not recommended. Use the [build_sampler] function or the [FromStr] instance
/// to ensure a valid configuration.
pub struct ConfiguredSamplers {
    /// A builder from the `llm-samplers` crate.
    pub builder: SamplerChainBuilder,
    /// Mirostat 1 is present.
    pub mirostat1: bool,
    /// Mirostat 2 is present.
    pub mirostat2: bool,
    /// Samplers incompatible with Mirostat 1 and 2 are present.
    pub incompat_mirostat: bool,
}

/// Construct a default instance of the structure. The `builder`
/// field contains a list of slots that may be optional.
///
/// We call a configuration of samplers that run in a certain order a "chain".
/// Here is a description of the default chain `llm` uses:
///
/// 1. Repetition (present by default, multiple allowed)
/// 2. Frequency/Presence (optional, multiple allowed)
/// 3. Sequence Repetition (optional, multiple allowed)
/// 4. Top-K (present by default - incompatible with Mirostat)
/// 5. Tail Free (optional - incompatible with Mirostat)
/// 6. Locally Typical (optional - incompatible with Mirostat)
/// 7. Top-P (present by default - incompatible with Mirostat)
/// 8. Temperature (present by default)
/// 9. A Mirostat 1 or 2 sampler if configured, otherwise Random Distribution.
///
/// Samplers listed as "present by default" but incompatible with Mirostat will
/// only be enabled by default if there is no Mirostat sampler enabled.
///
/// It's worth mentioning that "present by default" samplers that allow multiple instances
/// will add at least one entry if the user didn't specify the sampler. If they _did_ specify
/// it then no extra "default" sampler of that type will be added. So, for example,
/// if you wanted both the default Repetition sampler _and_ one with custom options, you'd
/// need to configure the Repetition sampler twice.
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
    /// Ensures the default slots are populated after processing options.
    /// Currently this is: temperature and repetition samplers
    /// Then if neither Mirostat 1 or 2 are enabled: top-p and top-k.
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

    /// Ensure that the configured samplers are compatible with each other.
    /// For example, if Mirostat 1 and Mirostat 2 are enabled, this would
    /// be invalid.
    pub fn ensure_valid(&self) -> Result<(), SamplerConfigurationError> {
        if self.mirostat1 && self.mirostat2 {
            Err(SamplerConfigurationError::SamplerCombinationError(
                "Cannot enable both Mirostat 1 and Mirostat 2 samplers".to_string(),
            ))?
        } else if (self.mirostat1 || self.mirostat2) && self.incompat_mirostat {
            Err(SamplerConfigurationError::SamplerCombinationError(
                "Cannot enable top-p, top-k, locally typical or tail free samplers with Mirostat 1 or 2".to_string(),
            ))?
        }
        Ok(())
    }
}

/// The structure is generally build from a string definition.
/// Configuring as individual sampler takes the form `sampler_name:key1=value1:key2=value2`.
/// Underscore and dash are ignored when comparing sampler names and comparison is
/// case-insensitive. A partial key name may be specified as long as it's not ambiguous.
/// If the sampler only has one option (for example Temperature) the key and equals sign can
/// be left out entirely.
///
/// Separate multiple sampler configuration strings with space or forward slash.
/// Blank entries are allowed.
impl FromStr for ConfiguredSamplers {
    type Err = SamplerConfigurationError;

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

        opts.into_iter().try_for_each(|(name, args)| {
            result.builder.configure(&name, args).map_err(|err| {
                SamplerConfigurationError::BuildSamplerError {
                    name: name.to_string(),
                    err: err.into(),
                }
            })
        })?;

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
) -> Result<TokenId, SamplingError> {
    Logits::try_from_iter(last_logits.into_iter())
        .map_err(|err| SamplingError::LogitsError(err.into()))?
        .sample_token(
            &mut SamplerResources {
                previous_tokens,
                rng,
            },
            &mut sampler,
        )
        .map_err(|err| SamplingError::InternalSamplingError(err.into()))?
        .ok_or_else(|| SamplingError::NoToken)
}

/// Build a sampler object with the supplied options, vocab size and token bias list.
///
/// Note that this is just a convenience function for building a sampler from
/// string definitions such as commandline arguments. The only limit on constructing
/// your own samplers is your sampler or samplers must implement the [Sampler] trait
/// from the `llm-samplers` crate.
pub fn build_sampler(
    n_vocab: usize,
    bias: &[(TokenId, f32)],
    args: &[impl AsRef<str>],
) -> Result<Arc<Mutex<dyn Sampler<TokenId, f32>>>, SamplerConfigurationError> {
    let mut samplers = SamplerChain::new();

    if !bias.is_empty() {
        samplers += SampleFlatBias::new(bias.iter().copied());
    }

    let sampler_options = args
        .iter()
        .map(|s| s.as_ref().trim())
        .filter(|s| !s.is_empty())
        .map(|s| "/".to_string() + s)
        .collect::<String>();

    let mut configured_samplers = ConfiguredSamplers::from_str(&sampler_options)?;
    if configured_samplers.mirostat1 {
        configured_samplers
            .builder
            .configure("mirostat1", format!("n_vocab={n_vocab}"))
            .map_err(|err| SamplerConfigurationError::BuildSamplerError {
                name: "mirostat1".to_string(),
                err: err.into(),
            })?;
    }
    samplers += configured_samplers.builder.into_chain();
    Ok(Arc::new(Mutex::new(samplers)))
}

/// Get the default sampler chain.
pub fn default_samplers() -> Arc<Mutex<dyn Sampler<TokenId, f32>>> {
    let mut result = ConfiguredSamplers::default();
    result.ensure_default_slots();
    Arc::new(Mutex::new(result.builder.into_chain()))
}

// Structure used to temporarily hold resources for the `llm-samplers`
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
