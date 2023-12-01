//! Tokenizer-related functionality.

use std::{error::Error, fmt::Display, path::PathBuf, str::FromStr};

use ggml::format::gguf::{Gguf, MetadataError};
use thiserror::Error;

mod embedded;
pub use embedded::*;
mod huggingface;
pub use huggingface::*;
pub use tokenizers as huggingface_tokenizers;

/// The identifier of a token in a tokenizer.
pub type TokenId = u32;
pub(crate) type Token = Vec<u8>;
pub(crate) type TokenScore = f32;

#[derive(Error, Debug)]
/// Errors related to tokenization.
pub enum TokenizationError {
    #[error("an invalid token was encountered during tokenization: {error}")]
    /// During tokenization, one of the produced tokens was invalid / zero.
    TokenizationFailed {
        #[source]
        /// The error that occurred during tokenization.
        error: Box<dyn Error + Send + Sync>,
    },
    #[error("the token ID {0} was invalid for this model")]
    /// One of the tokens provided by the user was invalid, and did not belong to this model's tokenizer.
    InvalidTokenId(TokenId),
}

#[derive(Error, Debug)]
/// Errors related to loading the tokenizer.
#[error("error loading tokenizer from {path}: {error}")]
pub enum TokenizerLoadError {
    #[error("error loading Hugging Face tokenizer from {tokenizer_source}: {error}")]
    /// An error occurred while loading a Hugging Face tokenizer.
    HuggingFaceTokenizerError {
        /// The source of the tokenizer that failed.
        tokenizer_source: HuggingFaceTokenizerErrorSource,
        /// The error that occurred during loading.
        error: Box<dyn Error + Send + Sync>,
    },
    #[error("no supported tokenizers were found, including in the model file: {unsupported_tokenizers:?}")]
    /// No supported tokenizers were found, including in the model file.
    NoSupportedTokenizersFound {
        /// The list of tokenizers that were found, but not supported.
        unsupported_tokenizers: Vec<String>,
    },
    #[error("{0}")]
    /// An error occured with retrieving data from the metadata.
    MetadataError(#[from] MetadataError),
}

/// Used to identify where the tokenizer that errored came from.
// NOTE: We could potentially reuse `TokenizerSource` for this, but I want to avoid
// cloning and/or displaying the entire `String` case. Revisit in future and see if
// I still feel the same.
#[derive(Debug)]
pub enum HuggingFaceTokenizerErrorSource {
    /// The tokenizer was loaded from this file.
    File(PathBuf),
    /// The tokenizer was loaded from thep rovided string.
    String,
    #[cfg(feature = "tokenizers-remote")]
    /// The tokenizer was loaded from the given HF ID.
    Remote(String),
}
impl Display for HuggingFaceTokenizerErrorSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::File(file) => write!(f, "file {file:?}"),
            Self::String => write!(f, "string"),
            #[cfg(feature = "tokenizers-remote")]
            Self::Remote(remote) => write!(f, "HF ID {remote:?}"),
        }
    }
}

/// At the time of writing, the embedded tokenizer is not enabled as it has
/// some bugs. We're just not enabling the option while it's broken.
const EMBEDDED_TOKENIZER_ENABLED: bool = false;

#[derive(Clone, Debug, PartialEq)]
/// The source of a tokenizer.
pub enum TokenizerSource {
    /// Read the vocabulary from the model if available, and use a simplistic tokenizer.
    ///
    /// This is easy to use, but may not be the best choice for your use case, and is not
    /// guaranteed to be available for all models.
    Embedded,

    /// Read a Hugging Face tokenizer from a local Hugging Face tokenizer file.
    HuggingFaceTokenizerFile(PathBuf),

    /// Read a Hugging Face tokenizer from the provided string.
    HuggingFaceTokenizerString(String),

    /// Fetch a Hugging Face tokenizer from a remote Hugging Face repository.
    /// This will make a blocking HTTP request to Hugging Face to retrieve the tokenizer
    /// and may store files locally, so it is not recommended for production use.
    #[cfg(feature = "tokenizers-remote")]
    HuggingFaceRemote(String),
}
impl TokenizerSource {
    /// Retrieve the tokenizer from the source.
    ///
    /// Note that this may make a blocking HTTP request to Hugging Face to retrieve the tokenizer.
    /// if `self` is `Self::HuggingFaceRemote`.
    pub fn retrieve(self, gguf: &Gguf) -> Result<Tokenizer, TokenizerLoadError> {
        match self {
            #[cfg(feature = "tokenizers-remote")]
            Self::HuggingFaceRemote(identifier) => Ok(HuggingFaceTokenizer::new(
                tokenizers::Tokenizer::from_pretrained(&identifier, None).map_err(|error| {
                    TokenizerLoadError::HuggingFaceTokenizerError {
                        tokenizer_source: HuggingFaceTokenizerErrorSource::Remote(
                            identifier.clone(),
                        ),
                        error,
                    }
                })?,
            )
            .into()),

            Self::HuggingFaceTokenizerFile(path) => Ok(HuggingFaceTokenizer::new(
                tokenizers::Tokenizer::from_file(&path).map_err(|error| {
                    TokenizerLoadError::HuggingFaceTokenizerError {
                        tokenizer_source: HuggingFaceTokenizerErrorSource::File(path.clone()),
                        error,
                    }
                })?,
            )
            .into()),

            Self::HuggingFaceTokenizerString(s) => Ok(Self::load_huggingface_json(&s)?),

            Self::Embedded => {
                if let Ok(hf) = gguf.metadata.get_str("tokenizer.huggingface.json") {
                    Ok(Self::load_huggingface_json(hf)?)
                } else if EmbeddedTokenizer::is_present_in_metadata(&gguf.metadata) {
                    if EMBEDDED_TOKENIZER_ENABLED {
                        Ok(EmbeddedTokenizer::from_metadata(&gguf.metadata)?.into())
                    } else {
                        Err(TokenizerLoadError::NoSupportedTokenizersFound {
                            unsupported_tokenizers: vec!["embedded".to_owned()],
                        })
                    }
                } else {
                    Err(TokenizerLoadError::NoSupportedTokenizersFound {
                        unsupported_tokenizers: vec![],
                    })
                }
            }
        }
    }

    fn load_huggingface_json(tokenizer_json: &str) -> Result<Tokenizer, TokenizerLoadError> {
        Ok(
            HuggingFaceTokenizer::new(tokenizers::Tokenizer::from_str(tokenizer_json).map_err(
                |error| TokenizerLoadError::HuggingFaceTokenizerError {
                    tokenizer_source: HuggingFaceTokenizerErrorSource::String,
                    error,
                },
            )?)
            .into(),
        )
    }
}
/// Encapsulates the tokenizer for a model, and provides methods to tokenize text.
pub enum Tokenizer {
    /// The vocabulary built-in to the model.
    Embedded(EmbeddedTokenizer),

    /// A Hugging Face tokenizer.
    HuggingFace(HuggingFaceTokenizer),
}
impl From<EmbeddedTokenizer> for Tokenizer {
    fn from(v: EmbeddedTokenizer) -> Self {
        Self::Embedded(v)
    }
}
impl From<HuggingFaceTokenizer> for Tokenizer {
    fn from(v: HuggingFaceTokenizer) -> Self {
        Self::HuggingFace(v)
    }
}
impl Tokenizer {
    /// Converts a token to the token ID it represents in this tokenizer.
    pub fn id(&self, token: &[u8]) -> Option<TokenId> {
        match self {
            Tokenizer::Embedded(v) => v.id(token),
            Tokenizer::HuggingFace(v) => v.id(token),
        }
    }

    /// Converts a token index to the token it represents in this tokenizer.
    pub fn token(&self, idx: usize) -> Vec<u8> {
        match self {
            Tokenizer::Embedded(v) => v.token(idx),
            Tokenizer::HuggingFace(v) => v.token(idx),
        }
    }

    /// Returns the number of tokens in the tokenizer.
    pub fn len(&self) -> usize {
        match self {
            Tokenizer::Embedded(v) => v.len(),
            Tokenizer::HuggingFace(v) => v.len(),
        }
    }

    /// Returns whether the tokenizer is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            Tokenizer::Embedded(v) => v.is_empty(),
            Tokenizer::HuggingFace(v) => v.is_empty(),
        }
    }

    /// Tokenize a `text` with this tokenizer.
    ///
    /// `bos` controls whether a beginning-of-string token should be inserted.
    pub fn tokenize(
        &self,
        text: &str,
        bos: bool,
    ) -> Result<Vec<(Vec<u8>, TokenId)>, TokenizationError> {
        match self {
            Tokenizer::Embedded(v) => v.tokenize(text, bos),
            Tokenizer::HuggingFace(v) => v.tokenize(text, bos),
        }
    }

    /// Decode a list `tokens` with this tokenizer.
    pub fn decode(&self, tokens: Vec<TokenId>, bos: bool) -> Vec<u8> {
        match self {
            Tokenizer::Embedded(v) => v.decode(tokens, bos),
            Tokenizer::HuggingFace(v) => v.decode(tokens, bos),
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
/// Represents the prompt, which can be specified as either text or tokens.
///
/// This type implements [From] for the following types:
/// - `&str`
/// - `&String`
/// - `&[TokenId]`
/// - `&Vec<TokenId>`
///
/// This allows you to pass any of these types to where this type is expected.
pub enum Prompt<'a> {
    /// A prompt specified as text.
    Text(&'a str),
    /// A prompt specified as tokens for this model's tokenizer.
    Tokens(&'a [TokenId]),
}
impl Prompt<'_> {
    /// Converts this prompt to a list of tokens for this model's tokenizer.
    ///
    /// Can return an error if [Self::Tokens] is used and includes a token ID that is not
    /// in this model's tokenizer.
    pub fn to_tokens(
        &self,
        vocab: &Tokenizer,
        beginning_of_sentence: bool,
    ) -> Result<Vec<TokenId>, TokenizationError> {
        Ok(match self {
            Self::Text(text) => vocab
                .tokenize(text, beginning_of_sentence)?
                .iter()
                .map(|(_, tok)| *tok)
                .collect(),
            Self::Tokens(tokens) => {
                if let Some(t) = tokens
                    .iter()
                    .copied()
                    .find(|t| vocab.token(*t as usize).is_empty())
                {
                    return Err(TokenizationError::InvalidTokenId(t));
                }
                tokens.to_vec()
            }
        })
    }

    /// Returns whether this prompt is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Text(text) => text.is_empty(),
            Self::Tokens(tokens) => tokens.is_empty(),
        }
    }
}
impl<'a> Default for Prompt<'a> {
    fn default() -> Self {
        Self::Text("")
    }
}
impl<'a> From<&'a str> for Prompt<'a> {
    fn from(v: &'a str) -> Self {
        Self::Text(v)
    }
}
impl<'a> From<&'a String> for Prompt<'a> {
    fn from(v: &'a String) -> Self {
        Self::from(v.as_str())
    }
}
impl<'a> From<&'a [TokenId]> for Prompt<'a> {
    fn from(v: &'a [TokenId]) -> Self {
        Self::Tokens(v)
    }
}
impl<'a> From<&'a Vec<TokenId>> for Prompt<'a> {
    fn from(v: &'a Vec<TokenId>) -> Self {
        Self::from(v.as_slice())
    }
}

#[derive(Default, Clone, Debug, PartialEq)]
/// A list of tokens to bias during the process of inferencing.
///
/// When a biased token is encountered, the bias will be used
/// instead of the inferred logit during the sampling process.
///
/// This can be used to disable the generation of responses
/// with specific tokens by setting their corresponding bias
/// to -1.0.
pub struct TokenBias(Vec<(TokenId, f32)>);

impl TokenBias {
    /// Create an empty [TokenBias].
    pub const fn empty() -> Self {
        Self(Vec::new())
    }

    /// Create a [TokenBias] from an existing `Vec`.
    pub fn new(mut v: Vec<(TokenId, f32)>) -> Self {
        v.sort_by_cached_key(|(tid, _)| *tid);
        v.dedup_by_key(|(tid, _)| *tid);
        Self(v)
    }

    /// Retrieves the bias for a given token, if available.
    pub fn get(&self, tid: TokenId) -> Option<f32> {
        self.0
            .binary_search_by_key(&tid, |(tid, _)| *tid)
            .map(|idx| self.0[idx].1)
            .ok()
    }
}

impl From<TokenBias> for Vec<(TokenId, f32)> {
    fn from(val: TokenBias) -> Self {
        val.0
    }
}

impl FromStr for TokenBias {
    type Err = InvalidTokenBias;

    /// A comma separated list of token biases. The list should be in the format
    /// "TID=BIAS,TID=BIAS" where TID is an integer token ID and BIAS is a
    /// floating point number.
    /// For example, "1=-1.0,2=-1.0" sets the bias for token IDs 1
    /// (start of document) and 2 (end of document) to -1.0 which effectively
    /// disables the model from generating responses containing those token IDs.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let x = s
            .split(',')
            .map(|kv| {
                let (k, v) = kv
                    .trim()
                    .split_once('=')
                    .ok_or_else(|| "Missing '=' in bias item".to_owned())?;
                let tid: TokenId = k
                    .trim()
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?;
                let bias: f32 = v
                    .trim()
                    .parse()
                    .map_err(|e: std::num::ParseFloatError| e.to_string())?;
                Result::<_, String>::Ok((tid, bias))
            })
            .collect::<Result<_, _>>()
            .map_err(InvalidTokenBias)?;
        Ok(TokenBias::new(x))
    }
}

/// An error was encountered when parsing a token bias string, which should be
/// in the format "TID=BIAS,TID=BIAS" where TID is an integer token ID and BIAS
/// is a floating point number.
#[derive(Debug)]
pub struct InvalidTokenBias(String);

impl Display for InvalidTokenBias {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "should be in the format <int>=<float>,<int>=<float>: {:?}",
            self.0
        )
    }
}

impl Error for InvalidTokenBias {}

impl std::fmt::Display for TokenBias {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
