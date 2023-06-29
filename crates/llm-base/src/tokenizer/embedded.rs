use std::collections::HashMap;

use thiserror::Error;

use super::{Token, TokenId, TokenScore, TokenizationError};

#[derive(Debug, Error)]
/// Errors that can occur when using a model tokenizer.
pub enum EmbeddedTokenizerError {
    /// Arbitrary error that occurred during use of the model tokenizer.
    #[error("Arbitrary error: {0:?}")]
    Arbitrary(String),
}

/// The built-in GGML tokenizer.
#[derive(Debug, Clone, Default)]
pub struct EmbeddedTokenizer {
    /// Maps every integer (index) token ID to its corresponding token.
    id_to_token: Vec<Token>,

    /// Maps every integer (index) token ID to corresponding score.
    id_to_token_score: Vec<TokenScore>,

    // todo: use a radix tree
    /// Maps a token to a token ID.
    token_to_id: HashMap<Token, TokenId>,

    /// The longest token in this tokenizer.
    max_token_length: usize,
}

impl EmbeddedTokenizer {
    /// Add a token to the internal vocabulary.
    ///
    /// The token added must have `id` directly after the last token in the vocabulary.
    ///
    /// # Panics
    /// - This function can panic if `id` does not correspond to the next token in the vocabulary.
    ///   That is, if there are already `n` tokens in the vocabulary, then `id` must be `n`.
    pub(crate) fn push_token(&mut self, id: TokenId, content: Token, score: TokenScore) {
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

    pub(crate) fn id(&self, token: &[u8]) -> Option<TokenId> {
        self.token_to_id.get(token).copied()
    }

    /// Converts a token index to the token it represents in this tokenizer.
    pub(crate) fn token(&self, idx: usize) -> Vec<u8> {
        self.id_to_token[idx].clone()
    }

    /// Returns the number of tokens in the tokenizer.
    pub(crate) fn len(&self) -> usize {
        self.id_to_token.len()
    }

    /// Returns whether the tokenizer is empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }

    // SentencePiece implementation after https://guillaume-be.github.io/2020-05-30/sentence_piece
    /// Tokenize a `text` with this tokenizer.
    ///
    /// `bos` controls whether a beginning-of-string token should be inserted.
    pub(crate) fn tokenize(
        &self,
        text: &str,
        bos: bool,
    ) -> Result<Vec<(Vec<u8>, TokenId)>, TokenizationError> {
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
                return Err(TokenizationError::TokenizationFailed {
                    error: Box::new(EmbeddedTokenizerError::Arbitrary(
                        "the backward pass for the tokenizer encountered a non-set token"
                            .to_string(),
                    )),
                });
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

    /// Decode a list `tokens` with this tokenizer.
    pub(crate) fn decode(&self, tokens: Vec<TokenId>, skip_special_tokens: bool) -> Vec<u8> {
        let mut vec = vec![];

        for token in tokens {
            if skip_special_tokens && token == 1 {
                continue;
            }

            vec.append(&mut self.id_to_token[token as usize].to_vec());
        }

        vec
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (Token, f32)> + '_ {
        self.id_to_token
            .iter()
            .zip(self.id_to_token_score.iter())
            .map(|(token, score)| (token.clone(), *score))
    }
}
