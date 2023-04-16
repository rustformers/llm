/// NOTE: The original code relies in promotion rules and automatic cast between
/// int to float. What we do instead is use this macro to convert every term of
/// the multiplication to f64, which should have enough precision bits to hold
/// the final value, then cast to usize. I have observed a discrepancy between
/// the ctx_size found using this code, and the one in llama.cpp. The number for
/// rust ends up being slightly lower, but no "out of memory" errors are
/// reported by ggml.
#[macro_export]
macro_rules! mulf {
    ($term:expr, $($terms:expr),*) => {
        usize::try_from((($term as f64) $(* ($terms as f64))*) as u64).unwrap()
    };
}

/// Used to buffer incoming tokens until they produce a valid string of UTF-8 text.
///
/// Tokens are *not* valid UTF-8 by themselves. However, the LLM will produce valid UTF-8
/// from multiple tokens. This helps alleviate that issue.
#[derive(Clone, PartialEq, Default)]
pub struct TokenUtf8Buffer(Vec<u8>);
impl TokenUtf8Buffer {
    /// Create a new buffer.
    pub const fn new() -> Self {
        Self(vec![])
    }

    /// Add a token to the buffer. If the buffer contains a valid string of UTF-8 text,
    /// it is returned and the buffer is cleared for next use.
    pub fn push(&mut self, token: &[u8]) -> Option<String> {
        self.0.extend_from_slice(token);
        match std::str::from_utf8(&self.0) {
            Ok(s) => {
                let out = s.to_owned();
                self.0 = vec![];
                Some(out)
            }
            Err(..) => {
                for i in 1..self.0.len() {
                    let slice = &self.0[i..];
                    if slice.is_empty() {
                        break;
                    }

                    if let Ok(s) = std::str::from_utf8(slice) {
                        let out = s.to_owned();
                        self.0 = vec![];
                        return Some(out);
                    }
                }
                None
            }
        }
    }

    /// Adapt a `&str` callback so that it can be used in a `&[u8]` context.
    pub fn adapt_callback<'a, E: std::error::Error + 'static>(
        mut callback: impl FnMut(&str) -> Result<(), E> + 'a,
    ) -> impl FnMut(&[u8]) -> Result<(), E> + 'a {
        let mut buffer = Self::new();
        move |token| match buffer.push(token) {
            Some(tokens) => callback(&tokens),
            None => Ok(()),
        }
    }
}
