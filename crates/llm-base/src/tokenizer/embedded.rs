use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap},
    str::FromStr,
};

use ggml::format::gguf::{Metadata, MetadataError};
use thiserror::Error;

use crate::TokenizerLoadError;

use super::{Token, TokenId, TokenScore, TokenizationError};

#[derive(Debug, Error)]
/// Errors that can occur when using a model tokenizer.
pub enum EmbeddedTokenizerError {
    /// Arbitrary error that occurred during use of the model tokenizer.
    #[error("Arbitrary error: {0:?}")]
    Arbitrary(String),
}

/// The built-in GGML tokenizer.
#[derive(Debug, Clone)]
pub struct EmbeddedTokenizer {
    /// Maps every integer (index) token ID to its corresponding token.
    id_to_token: Vec<TokenData>,

    // todo: use a radix tree
    /// Maps a token to a token ID.
    token_to_id: HashMap<Token, TokenId>,

    model: GgufEmbeddedTokenizerModel,
    bos_id: TokenId,
    _eos_id: TokenId,
    _unknown_id: TokenId,
    linefeed_id: TokenId,
    _separator_id: Option<TokenId>,
    _padding_id: Option<TokenId>,
}
#[derive(Debug, Clone, Default)]
struct TokenData {
    text: Token,
    score: TokenScore,
    ty: TokenType,
}
impl EmbeddedTokenizer {
    pub(crate) fn is_present_in_metadata(metadata: &Metadata) -> bool {
        metadata.contains_key("tokenizer.ggml.scores")
    }

    pub(crate) fn from_metadata(metadata: &Metadata) -> Result<Self, TokenizerLoadError> {
        let tok = GgufEmbeddedTokenizer::from_metadata(metadata)?;

        let model = if let Some(model) = tok.model {
            model
                .parse::<GgufEmbeddedTokenizerModel>()
                .expect("TODO: handle invalid tokenizer model")
        } else {
            GgufEmbeddedTokenizerModel::Llama
        };

        match model {
            GgufEmbeddedTokenizerModel::Llama => {
                let bos_id = metadata
                    .get_with_type("tokenizer.ggml.bos_token_id", |v| v.as_uint32())
                    .unwrap_or(1);
                let eos_id = metadata
                    .get_with_type("tokenizer.ggml.eos_token_id", |v| v.as_uint32())
                    .unwrap_or(2);
                let unknown_id = metadata
                    .get_with_type("tokenizer.ggml.unknown_token_id", |v| v.as_uint32())
                    .unwrap_or(0);
                let separator_id = metadata
                    .get_with_type("tokenizer.ggml.separator_token_id", |v| v.as_uint32())
                    .ok();
                let padding_id = metadata
                    .get_with_type("tokenizer.ggml.padding_token_id", |v| v.as_uint32())
                    .ok();

                let tokens = metadata.get_array_with_type("tokenizer.ggml.tokens", |v| {
                    v.as_array()?.as_string_array()
                })?;
                let scores = metadata
                    .get_array_with_type("tokenizer.ggml.scores", |v| {
                        v.as_array()?.as_float32_array()
                    })
                    .unwrap_or_default();
                let types = metadata
                    .get_array_with_type("tokenizer.ggml.token_type", |v| {
                        v.as_array()?.as_int32_array()
                    })
                    .unwrap_or_default();

                let mut token_to_id = HashMap::default();
                let mut id_to_token = vec![TokenData::default(); tokens.len()];

                for (i, token) in tokens.iter().enumerate() {
                    let word = token.as_bytes().to_vec();
                    token_to_id.insert(word.clone(), i as TokenId);
                    id_to_token[i] = TokenData {
                        text: word.clone(),
                        score: scores.get(i).copied().unwrap_or(0.0),
                        ty: match types.get(i) {
                            Some(tok) => {
                                TokenType::try_from(*tok).expect("TODO: handle invalid token type")
                            }
                            None => TokenType::Normal,
                        },
                    };
                }

                let mut tokenizer = EmbeddedTokenizer {
                    token_to_id,
                    id_to_token,
                    model: GgufEmbeddedTokenizerModel::Llama,
                    bos_id,
                    _eos_id: eos_id,
                    _unknown_id: unknown_id,
                    linefeed_id: 0,
                    _separator_id: separator_id,
                    _padding_id: padding_id,
                };

                tokenizer.linefeed_id = tokenizer.byte_to_token(b'\n');

                Ok(tokenizer)
            }
            _ => unimplemented!(),
        }
    }

    pub(crate) fn id(&self, token: &[u8]) -> Option<TokenId> {
        self.token_to_id.get(token).copied()
    }

    /// Converts a token index to the token it represents in this tokenizer.
    pub(crate) fn token(&self, idx: usize) -> Token {
        self.id_to_token[idx].text.clone()
    }

    /// Returns the number of tokens in the tokenizer.
    pub(crate) fn len(&self) -> usize {
        self.id_to_token.len()
    }

    /// Returns whether the tokenizer is empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }

    /// Tokenize a `text` with this tokenizer.
    ///
    /// `bos` controls whether a beginning-of-string token should be inserted.
    pub(crate) fn tokenize(
        &self,
        text: &str,
        bos: bool,
    ) -> Result<Vec<(Token, TokenId)>, TokenizationError> {
        let mut output = vec![];

        if bos {
            output.push((
                self.id_to_token[self.bos_id as usize].text.clone(),
                self.bos_id,
            ));
        }

        if text.is_empty() {
            return Ok(output);
        }

        match self.model {
            GgufEmbeddedTokenizerModel::Llama => {
                let text = escape_whitespace(format!(" {text}").as_bytes());

                Ok(TokenizerSpm::new(self)
                    .tokenize(&text)
                    .into_iter()
                    .map(|id| {
                        // TODO: see if this can be made more efficient
                        (self.id_to_token[id as usize].text.clone(), id)
                    })
                    .collect())
            }
            _ => unimplemented!(),
        }
    }

    /// Decode a list `tokens` with this tokenizer.
    pub(crate) fn decode(&self, tokens: Vec<TokenId>, _skip_special_tokens: bool) -> Vec<u8> {
        let mut ret = vec![];

        match self.model {
            GgufEmbeddedTokenizerModel::Llama => {
                for token_id in tokens {
                    let token = &self.id_to_token[token_id as usize];
                    match token.ty {
                        TokenType::Normal => {
                            ret.append(&mut unescape_whitespace(&token.text));
                        }
                        TokenType::Unknown => {
                            assert_eq!(token.text.len(), 3);
                            ret.extend_from_slice(&[0xE2, 0x96, 0x85]);
                        }
                        TokenType::Byte => {
                            ret.push(self.token_to_byte(token_id));
                        }
                        TokenType::Control | TokenType::UserDefined | TokenType::Unused => {}
                    }
                }
            }
            _ => unimplemented!(),
        }

        ret
    }
}
impl EmbeddedTokenizer {
    fn byte_to_token(&self, ch: u8) -> TokenId {
        let token = format!("<0x{ch:02X}>");
        self.token_to_id.get(token.as_bytes()).copied().unwrap()
    }

    fn token_to_byte(&self, token_id: TokenId) -> u8 {
        let data = &self.id_to_token[token_id as usize];
        assert_eq!(data.ty, TokenType::Byte);

        match self.model {
            GgufEmbeddedTokenizerModel::Llama => {
                u8::from_str_radix(std::str::from_utf8(&data.text[3..5]).unwrap(), 16).unwrap()
            }
            _ => unimplemented!(),
        }
    }
}

/// An embedded tokenizer definition in a GGUF.
pub struct GgufEmbeddedTokenizer<'a> {
    /// The model type.
    pub model: Option<&'a str>,
    /// The tokens.
    pub tokens: &'a [String],
    /// The token scores.
    pub scores: &'a [f32],
    /// The token types.
    pub types: Option<&'a [u32]>,
}
impl GgufEmbeddedTokenizer<'_> {
    /// Attempt to retrieve the embedded tokenizer from the metadata.
    pub fn from_metadata(metadata: &Metadata) -> Result<GgufEmbeddedTokenizer, MetadataError> {
        Ok(GgufEmbeddedTokenizer {
            model: metadata
                .get_optional("tokenizer.ggml.model")
                .and_then(|v| v.as_string()),
            tokens: metadata.get_array_with_type("tokenizer.ggml.tokens", |v| {
                v.as_array()?.as_string_array()
            })?,
            scores: metadata.get_array_with_type("tokenizer.ggml.scores", |v| {
                v.as_array()?.as_float32_array()
            })?,
            types: metadata
                .get_array_with_type("tokenizer.ggml.token_type", |v| {
                    v.as_array()?.as_uint32_array()
                })
                .ok(),
        })
    }
}

/// Typesafe tokenizer models.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum GgufEmbeddedTokenizerModel {
    /// Llama style SentencePiece (tokens and scores extracted from HF `tokenizer.model`)
    Llama,
    /// Replit style SentencePiece (tokens and scores extracted from HF `spiece.model`)
    Replit,
    /// GPT-2 / GPT-NeoX style BPE (tokens extracted from HF `tokenizer.json`)
    Gpt2,
    /// RWKV tokenizer
    Rwkv,
}
impl FromStr for GgufEmbeddedTokenizerModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "llama" => Ok(Self::Llama),
            "replit" => Ok(Self::Replit),
            "gpt2" => Ok(Self::Gpt2),
            "rwkv" => Ok(Self::Rwkv),
            other => Err(other.to_string()),
        }
    }
}

/// The type of a token.
#[allow(missing_docs)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Default)]
pub enum TokenType {
    #[default]
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}
impl TryFrom<i32> for TokenType {
    type Error = i32;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Normal),
            2 => Ok(Self::Unknown),
            3 => Ok(Self::Control),
            4 => Ok(Self::UserDefined),
            5 => Ok(Self::Unused),
            6 => Ok(Self::Byte),
            other => Err(other),
        }
    }
}

#[derive(Clone)]
struct Symbol {
    prev: isize,
    next: isize,
    text: Vec<u8>,
    n: usize,
}

struct LlmBigramSpm {
    left: isize,
    right: isize,
    score: f32,
    size: usize,
}
impl PartialOrd for LlmBigramSpm {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for LlmBigramSpm {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialEq for LlmBigramSpm {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.left == other.left
    }
}

impl Eq for LlmBigramSpm {}

struct TokenizerSpm<'a> {
    vocab: &'a EmbeddedTokenizer,
    symbols: Vec<Symbol>,
    work_queue: BinaryHeap<LlmBigramSpm>,
    rev_merge: HashMap<Token, (isize, isize)>,
}

impl<'a> TokenizerSpm<'a> {
    fn new(vocab: &'a EmbeddedTokenizer) -> Self {
        Self {
            vocab,
            symbols: Vec::new(),
            work_queue: BinaryHeap::new(),
            rev_merge: HashMap::new(),
        }
    }

    fn tokenize(&mut self, text: &[u8]) -> Vec<TokenId> {
        let mut output = vec![];
        let mut index = 0;
        let mut offs = 0;
        while offs < text.len() {
            let len = text[offs..].len();
            let sym = Symbol {
                text: text[offs..offs + len].to_vec(),
                n: len.min(text.len() - offs),
                prev: index - 1,
                next: if offs + len == text.len() {
                    -1
                } else {
                    index + 1
                },
            };
            offs += sym.n;
            index += 1;
            self.symbols.push(sym);
        }

        for i in 1..self.symbols.len() {
            self.try_add_bigram((i - 1) as isize, i as isize);
        }

        while let Some(bigram) = self.work_queue.pop() {
            let mut left_sym = self.symbols[bigram.left as usize].clone();
            let mut right_sym = self.symbols[bigram.right as usize].clone();

            if left_sym.n == 0 || right_sym.n == 0 || left_sym.n + right_sym.n != bigram.size {
                continue;
            }

            left_sym.n += right_sym.n;
            right_sym.n = 0;

            left_sym.next = right_sym.next;
            if right_sym.next >= 0 {
                self.symbols[right_sym.next as usize].prev = bigram.left;
            }

            let left_sym_prev = left_sym.prev;
            let left_sym_next = left_sym.next;

            self.symbols[bigram.left as usize] = left_sym;
            self.symbols[bigram.right as usize] = right_sym;

            self.try_add_bigram(left_sym_prev, bigram.left);
            self.try_add_bigram(bigram.left, left_sym_next);
        }

        let mut i = 0;
        while i != -1 {
            let symbol = &self.symbols[i as usize];
            self.resegment(symbol, &mut output);
            i = symbol.next;
        }
        output
    }

    fn resegment(&self, symbol: &Symbol, output: &mut Vec<TokenId>) {
        let text = symbol.text.clone();
        if let Some(&token_id) = self.vocab.token_to_id.get(&text) {
            output.push(token_id);
            return;
        }

        if let Some(&(left, right)) = self.rev_merge.get(&text) {
            self.resegment(&self.symbols[left as usize], output);
            self.resegment(&self.symbols[right as usize], output);
        } else {
            for &ch in &text {
                let token_id = self.vocab.byte_to_token(ch);
                output.push(token_id);
            }
        }
    }

    fn try_add_bigram(&mut self, left: isize, right: isize) {
        if left == -1 || right == -1 {
            return;
        }

        let text = [
            self.symbols[left as usize].text.clone(),
            self.symbols[right as usize].text.clone(),
        ]
        .concat();
        if let Some(&token_id) = self.vocab.token_to_id.get(&text) {
            if (token_id as usize) < self.vocab.id_to_token.len() {
                let tok_data = &self.vocab.id_to_token[token_id as usize];
                let bigram = LlmBigramSpm {
                    left,
                    right,
                    score: tok_data.score,
                    size: text.len(),
                };
                self.work_queue.push(bigram);
                self.rev_merge.insert(text, (left, right));
            }
        }
    }
}

fn escape_whitespace(text: &[u8]) -> Vec<u8> {
    let mut out = vec![];

    for &b in text {
        if b == b' ' {
            out.extend_from_slice(&[0xE2, 0x96, 0x81]);
        } else {
            out.push(b);
        }
    }

    out
}

fn unescape_whitespace(text: &[u8]) -> Vec<u8> {
    let mut out = vec![];
    let mut buffer: Vec<u8> = vec![];

    for &b in text {
        #[allow(clippy::if_same_then_else)]
        if b == 0xE2 {
            // If the current byte is 0xE2, start buffering and check for the sequence.
            buffer.push(b);
        } else if buffer.len() == 1 && b == 0x96 {
            // If the previous byte was 0xE2 and the current byte is 0x96, continue buffering.
            buffer.push(b);
        } else if buffer.len() == 2 && b == 0x81 {
            // If the previous bytes were 0xE2 and 0x96 and the current byte is 0x81, replace with space and reset buffer.
            out.push(b' ');
            buffer.clear();
        } else {
            // If no match, flush the buffer and append the current byte.
            out.append(&mut buffer);
            out.push(b);
        }
    }

    // If there are any remaining bytes in the buffer, append them.
    out.append(&mut buffer);

    out
}
