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
        (($term as f64) $(* ($terms as f64))*) as u64
    };
}
