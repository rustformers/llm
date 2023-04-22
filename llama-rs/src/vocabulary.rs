use std::{collections::HashMap, str::FromStr};

use crate::InferenceError;

/// The identifier of a token in a vocabulary.
pub type TokenId = i32;
pub(crate) type Token = Vec<u8>;
pub(crate) type TokenScore = f32;

/// The vocabulary used by a model.
#[derive(Debug, Clone, Default)]
pub struct Vocabulary {
    /// Maps every integer (index) token id to its corresponding token
    pub(crate) id_to_token: Vec<Token>,

    /// Maps every integer (index) token id to corresponding score
    pub(crate) id_to_token_score: Vec<TokenScore>,

    // todo: use a radix tree
    /// Maps a token to a token id
    pub(crate) token_to_id: HashMap<Token, TokenId>,

    /// The longest token in this vocabulary
    pub(crate) max_token_length: usize,
}

impl Vocabulary {
    /// Add a token to the vocabulary.
    ///
    /// The token added must have `id` directly after the last token in the vocabulary.
    pub fn push_token(&mut self, id: TokenId, content: Token, score: TokenScore) {
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

    pub(crate) fn token(&self, idx: usize) -> &[u8] {
        &self.id_to_token[idx]
    }

    // SentencePiece implementation after https://guillaume-be.github.io/2020-05-30/sentence_piece
    /// Tokenize a `text` with this vocabulary.
    ///
    /// `bos` controls whether a beginning-of-string token should be inserted.
    pub fn tokenize<'a>(
        &'a self,
        text: &str,
        bos: bool,
    ) -> Result<Vec<(&'a [u8], TokenId)>, InferenceError> {
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
                return Err(InferenceError::TokenizationFailed);
            }
            let token = self.id_to_token[token_id as usize].as_slice();
            res.push((token, token_id));
            i -= token.len();
        }

        if bos {
            // TODO: replace with vocab.bos
            res.push((&[], 1));
        }

        // Pieces are in reverse order so correct that
        res.reverse();

        Ok(res)
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
    type Err = String;

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
            .collect::<Result<_, _>>()?;
        Ok(TokenBias::new(x))
    }
}

impl std::fmt::Display for TokenBias {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
