//! Metal support.
use std::{ptr::NonNull, sync::Arc};

use crate::sys::metal;

/// Acts as a RAII-guard over a `sys::metal::ggml_metal_context`, allocating via
/// `ggml_metal_init` and dropping via `ggml_metal_free`.
pub struct MetalContext {
    ptr: Arc<NonNull<metal::ggml_metal_context>>,
}

impl MetalContext {
    /// Creates a new [MetalContext]
    pub fn new() -> Self {
        let raw = unsafe { metal::ggml_metal_init() };

        MetalContext {
            ptr: Arc::new(NonNull::new(raw).expect("Should not be null")),
        }
    }
}

impl Drop for MetalContext {
    fn drop(&mut self) {
        // SAFETY: The only non-weak copy of ptr is no longer accessible after
        // this drop call.
        unsafe { metal::ggml_metal_free(self.ptr.as_ptr()) }
    }
}
