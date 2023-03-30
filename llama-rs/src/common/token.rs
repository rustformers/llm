pub const EOD_TOKEN_ID: TokenId = 2; // Hardcoded (for now?)

type TokenId = i32;
type Token = String;
type TokenScore = f32;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OutputToken<'a> {
    Token(&'a str),
    EndOfText,
}
impl<'a> OutputToken<'a> {
    fn from_id(vocab: &'a Vocabulary, id: TokenId) -> Self {
        if id == 2 {
            Self::EndOfText
        } else {
            Self::Token(&vocab.id_to_token[id as usize])
        }
    }
}
impl Display for OutputToken<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                OutputToken::Token(t) => *t,
                OutputToken::EndOfText => "[end of text]",
            }
        )
    }
}

#[derive(Default, Clone, Debug, PartialEq)]
pub struct TokenBias(Vec<(TokenId, f32)>);

impl TokenBias {
    pub fn new(mut v: Vec<(TokenId, f32)>) -> Self {
        v.sort_by_cached_key(|(tid, _)| *tid);
        v.dedup_by_key(|(tid, _)| *tid);
        Self(v)
    }

    pub fn get(&self, tid: TokenId) -> Option<f32> {
        self.0
            .binary_search_by_key(&tid, |(tid, _)| *tid)
            .map(|idx| self.0[idx].1)
            .ok()
    }
}

impl FromStr for TokenBias {
    type Err = String;

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
