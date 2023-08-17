use super::{TokenId, TokenizationError};

/// A Hugging Face tokenizer.
#[derive(Debug, Clone)]
pub struct HuggingFaceTokenizer {
    pub(crate) tokenizer: tokenizers::Tokenizer,
}

impl HuggingFaceTokenizer {
    /// Create a new `HuggingFaceTokenizer`.
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self { tokenizer }
    }
}

impl HuggingFaceTokenizer {
    pub(crate) fn id(&self, token: &[u8]) -> Option<TokenId> {
        self.tokenizer
            .token_to_id(std::str::from_utf8(token).unwrap())
    }

    /// Converts a token index to the token it represents in this tokenizer.
    pub(crate) fn token(&self, idx: usize) -> Vec<u8> {
        self.tokenizer
            .decode(&[idx as u32], true)
            .expect("Cannot decode token from tokenizer tokenizer.")
            .as_bytes()
            .to_vec()
    }

    /// Returns the number of tokens in the tokenizer.
    pub(crate) fn len(&self) -> usize {
        self.tokenizer.get_vocab_size(false)
    }

    /// Returns whether the tokenizer is empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.tokenizer.get_vocab_size(false) == 0
    }

    /// Tokenize a `text` with this tokenizer.
    ///
    /// `bos` controls whether a beginning-of-string token should be inserted.
    pub(crate) fn tokenize(
        &self,
        text: &str,
        bos: bool,
    ) -> Result<Vec<(Vec<u8>, TokenId)>, TokenizationError> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| TokenizationError::TokenizationFailed { error: e })?;

        let encoding = self
            .tokenizer
            .post_process(encoding, None, bos)
            .map_err(|e| TokenizationError::TokenizationFailed { error: e })?;

        Ok(encoding
            .get_tokens()
            .iter()
            .map(|t| t.as_bytes().to_vec())
            .zip(encoding.get_ids().iter().copied())
            .collect())
    }

    /// Decode a list `tokens` with this tokenizer.
    pub(crate) fn decode(&self, tokens: Vec<TokenId>, skip_special_tokens: bool) -> Vec<u8> {
        self.tokenizer
            .decode(&tokens, skip_special_tokens)
            .expect("Cannot decode token from tokenizer.")
            .as_bytes()
            .to_vec()
    }
}
