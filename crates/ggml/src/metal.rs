//! Metal support.
use std::{ptr::NonNull, sync::Arc};

use crate::{sys::metal, ComputationGraph, Tensor};

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

    /// Computes the specified graph using Metal.
    pub fn graph_compute(&self, graph: &mut ComputationGraph) {
        unsafe {
            metal::ggml_metal_graph_compute(
                self.ptr.as_ptr(),
                &mut graph.inner as *mut ggml_sys::ggml_cgraph as *mut metal::ggml_cgraph,
            );
        }
    }

    /// Reads a tensor from Metal
    pub fn get_tensor(&self, tensor: &Tensor) {
        unsafe {
            metal::ggml_metal_get_tensor(
                self.ptr.as_ptr(),
                tensor.ptr.as_ptr() as *mut metal::ggml_tensor,
            )
        }
    }
}

impl Drop for MetalContext {
    fn drop(&mut self) {
        panic!();
        // SAFETY: The only non-weak copy of ptr is no longer accessible after
        // this drop call.
        unsafe { metal::ggml_metal_free(self.ptr.as_ptr()) }
    }
}
