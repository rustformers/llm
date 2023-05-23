use std::{collections::HashMap, error::Error, fmt::Display, str::FromStr};

use thiserror::Error;
use tokenizers::Tokenizer;

/// The identifier of a token in a vocabulary.
pub type TokenId = u32;
pub(crate) type Token = Vec<u8>;
pub(crate) type TokenScore = f32;

#[derive(Error, Debug)]
/// Errors related to tokenization.
pub enum TokenizationError {
    #[error("an invalid token was encountered during tokenization")]
    /// During tokenization, one of the produced tokens was invalid / zero.
    TokenizationFailed,
    #[error("the token ID {0} was invalid for this model")]
    /// One of the tokens provided by the user was invalid, and did not belong to this model's vocabulary.
    InvalidTokenId(TokenId),
}

/// The vocabulary used by a model.
#[derive(Debug, Clone, Default)]
pub struct Vocabulary {
    // TODO: make these private
    /// Maps every integer (index) token ID to its corresponding token.
    pub id_to_token: Vec<Token>,

    /// Maps every integer (index) token ID to corresponding score.
    pub id_to_token_score: Vec<TokenScore>,

    // todo: use a radix tree
    /// Maps a token to a token ID.
    pub token_to_id: HashMap<Token, TokenId>,

    /// The longest token in this vocabulary.
    pub max_token_length: usize,

    /// The tokenizer
    pub tokenizer: Option<Tokenizer>,
}

impl Vocabulary {
    /// Intialize a new vocabulary.
    pub fn new(tokenizer: Option<Tokenizer>) -> Vocabulary {
        let mut vocab = Vocabulary::default();
        vocab.tokenizer = tokenizer;

        vocab
    }
    /// Add a token to the vocabulary.
    ///
    /// The token added must have `id` directly after the last token in the vocabulary.
    ///
    /// # Panics
    /// - This function can panic if `id` does not correspond to the next token in the vocabulary.
    ///   That is, if there are already `n` tokens in the vocabulary, then `id` must be `n`.
    pub fn push_token(&mut self, id: TokenId, content: Token, score: TokenScore) {
        if self.tokenizer.is_some() {
            return;
        }

        // These are loader invariants. If this is broken, then the loader is broken and this is a bug,
        // not an issue with the model itself.
        assert_eq!(self.id_to_token.len(), self.id_to_token_score.len());
        if self.id_to_token.len() != id as usize || self.id_to_token_score.len() != id as usize {
            let expected_id = self.id_to_token.len() as TokenId;
            panic!("the id of token added should be {expected_id}; is {id}");
        }

        self.max_token_length = self.max_token_length.max(content.len());
        self.id_to_token.push(content.clone());
        self.id_to_token_score.push(score);
        self.token_to_id.insert(content, id);
    }

    /// Converts a token index to the token it represents in this vocabulary.
    pub fn token(&self, idx: usize) -> Vec<u8> {
        if let Some(tokenizer) = &self.tokenizer {
            return tokenizer
                .decode(vec![idx as u32], true)
                .unwrap()
                .as_bytes()
                .to_vec();
        }

        (&self.id_to_token[idx]).clone()
    }

    /// Returns the number of tokens in the vocabulary.
    pub fn len(&self) -> usize {
        if let Some(tokenizer) = &self.tokenizer {
            return tokenizer.get_vocab_size(false) as usize;
        }

        self.id_to_token.len()
    }

    /// Returns whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        if let Some(tokenizer) = &self.tokenizer {
            return tokenizer.get_vocab_size(false) == 0;
        }

        self.id_to_token.is_empty()
    }

    // SentencePiece implementation after https://guillaume-be.github.io/2020-05-30/sentence_piece
    /// Tokenize a `text` with this vocabulary.
    ///
    /// `bos` controls whether a beginning-of-string token should be inserted.
    pub fn tokenize<'a>(
        &'a self,
        text: &str,
        bos: bool,
    ) -> Result<Vec<(Vec<u8>, TokenId)>, TokenizationError> {
        if let Some(tokenizer) = &self.tokenizer {
            let res = tokenizer.encode(text, bos);
            if res.is_err() {
                return Err(TokenizationError::TokenizationFailed);
            } else {
                Ok(tokenizer
                    .encode(text, bos)
                    .unwrap()
                    .get_ids()
                    .iter()
                    .map(|id| (self.token(*id as usize), *id))
                    .collect::<Vec<(Vec<u8>, TokenId)>>())
            }
        } else {
            let len = text.len();

            let mut score = vec![0usize; len + 1];
            let mut prev = vec![TokenId::default(); len + 1];

            for i in 0..len {
                let max_len = (len - i).min(self.max_token_length);
                for sub_len in 1..=max_len {
                    let sub = &text.as_bytes()[i..i + sub_len];
                    let token = self.token_to_id.get(sub);

                    if let Some(token) = token {
                        let token_score = sub.len() * sub.len();
                        let local_score = score[i] + token_score;
                        let next = i + sub_len;

                        if score[next] < local_score {
                            score[next] = local_score;
                            prev[next] = *token;
                        }
                    }
                }
            }

            // Backward pass
            let mut res = vec![];
            let mut i = len;
            while i > 0 {
                let token_id = prev[i];
                if token_id == 0 {
                    return Err(TokenizationError::TokenizationFailed);
                }
                let token = self.id_to_token[token_id as usize].as_slice();
                res.push((token.to_vec(), token_id));
                i -= token.len();
            }

            if bos {
                // TODO: replace with vocab.bos
                res.push((vec![], 1));
            }

            // Pieces are in reverse order so correct that
            res.reverse();

            Ok(res)
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
    /// A prompt specified as tokens for this model's vocabulary.
    Tokens(&'a [TokenId]),
}
impl Prompt<'_> {
    /// Converts this prompt to a list of tokens for this model's vocabulary.
    ///
    /// Can return an error if [Self::Tokens] is used and includes a token ID that is not
    /// in this model's vocabulary.
    pub fn to_tokens(
        &self,
        vocab: &Vocabulary,
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
                    .find(|t| vocab.id_to_token.get(*t as usize).is_none())
                {
                    return Err(TokenizationError::InvalidTokenId(t));
                }
                tokens.to_vec()
            }
        })
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
