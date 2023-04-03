use super::inference::InferenceError;
use super::token::{Token, TokenId, TokenScore};
use std::collections::HashMap;

pub struct Vocabulary {
    /// Maps every integer (index) token id to its corresponding token
    pub id_to_token: Vec<Token>,

    /// Maps every integer (index) token id to corresponding score
    #[allow(dead_code)]
    pub id_to_token_score: Vec<TokenScore>,

    /// Maps a token to a token id
    pub token_to_id: HashMap<Token, TokenId>,

    /// The longest token in this vocabulary
    pub max_token_length: usize,
}

impl Vocabulary {
    // SentencePiece implementation after https://guillaume-be.github.io/2020-05-30/sentence_piece
    pub fn tokenize<'a>(
        &'a self,
        text: &str,
        bos: bool,
    ) -> Result<Vec<(&'a str, TokenId)>, InferenceError> {
        let len = text.len();

        let mut score = vec![0usize; len + 1];
        let mut prev = vec![TokenId::default(); len + 1];

        for i in 0..len {
            let max_len = (len - i).min(self.max_token_length);
            for sub_len in 1..=max_len {
                let sub = &text.as_bytes()[i..i + sub_len];
                let Ok(sub) = std::str::from_utf8(sub) else { continue; };
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
            let token = self.id_to_token[token_id as usize].as_str();
            res.push((token, token_id));
            i -= token.len();
        }

        if bos {
            // TODO: replace with vocab.bos
            res.push(("", 1));
        }

        // Pieces are in reverse order so correct that
        res.reverse();

        Ok(res)
    }
    fn token(&self, idx: usize) -> &str {
        &self.id_to_token[idx]
    }
}
