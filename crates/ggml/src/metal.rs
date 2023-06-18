//! Metal support.
use crate::{sys::metal, Buffer, ComputationGraph, Context, Tensor};
use std::{ffi::c_void, ptr::NonNull, sync::Arc};

/// Acts as a RAII-guard over a `sys::metal::ggml_metal_context`, allocating via
/// `ggml_metal_init` and dropping via `ggml_metal_free`.
pub struct MetalContext {
    ptr: Arc<NonNull<metal::ggml_metal_context>>,

    /// References to the context that hold buffers that are used in this Metal context. As Metal does not need to copy
    /// buffers to VRAM, we do need to keep the original buffers alive through this reference.
    contexts: Vec<Arc<Context>>,
}

impl MetalContext {
    /// Create a new Metal context
    pub fn new() -> Self {
        let raw = unsafe { metal::ggml_metal_init() };

        MetalContext {
            contexts: vec![],
            ptr: Arc::new(NonNull::new(raw).expect("Should not be null")),
        }
    }

    /// Register a buffer mapping
    pub fn add_scratch_buffer(&mut self, buf: &Buffer) {
        unsafe {
            let raw_metal_context = self.ptr.as_ptr();

            //Last we need to add the scratch buffers to the buffers
            assert!(
                metal::ggml_metal_add_buffer(
                    raw_metal_context,
                    "scratch\0".as_ptr().cast(), // FIXME: allocate string and insert number in name
                    buf.data.as_ptr() as *mut core::ffi::c_void,
                    buf.data.len(),
                    buf.data.len()
                ),
                "{}",
                format!("Could not add scratch buffer to metal context")
            );
        }
    }

    /// Add a context's memory as buffer to this Metal context
    pub fn add_context(&mut self, from_context: Arc<Context>) {
        self.ref_context(from_context.clone());

        unsafe {
            let raw_context = from_context.ptr.as_ptr();

            let (data_ptr, data_size): (*mut c_void, usize) =
                if let Some(ref mmap) = from_context.mmap {
                    // This is a bit naughty...
                    (mmap.as_ptr().cast_mut().cast(), mmap.len())
                } else {
                    (
                        ggml_sys::ggml_get_mem_buffer(raw_context),
                        ggml_sys::ggml_get_mem_size(raw_context),
                    )
                };

            let max_size = ggml_sys::ggml_get_max_tensor_size(raw_context);
            assert!(
                metal::ggml_metal_add_buffer(
                    self.ptr.as_ptr(),
                    "wt\0".as_ptr().cast(), // FIXME provide an actual name
                    data_ptr,
                    data_size,
                    max_size
                ),
                "Could not add weight buffer to metal context"
            );
        }
    }
}

impl Default for MetalContext {
    fn default() -> Self {
        Self::new()
    }
}

impl MetalContext {
    fn ref_context(&mut self, context: Arc<Context>) {
        self.contexts.push(context);
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
        // SAFETY: The only non-weak copy of ptr is no longer accessible after
        // this drop call.
        unsafe { metal::ggml_metal_free(self.ptr.as_ptr()) }
    }
}
