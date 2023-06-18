//! Metal support.
use crate::{sys::metal, Buffer, ComputationGraph, Context, Tensor};
use std::{ptr::NonNull, sync::Arc};

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
}

impl Default for MetalContext {
    fn default() -> Self {
        Self::new()
    }
}

impl MetalContext {
    /// Initializes the buffers needed for a metal forward pass.
    pub fn initialize_eval_buffers(&mut self, eval_context: Arc<Context>) {
        unsafe {
            let raw_metal_context = self.ptr.as_ptr();

            // TODO check if this works with mmap
            // in our implementation this should be the `ctx0` buffer
            // Original code: ggml_metal_add_buffer(ctx->ctx_metal, "eval", ctx->buf_compute.addr, ctx->buf_compute.size)
            let raw_eval_context = eval_context.ptr.as_ptr();
            self.ref_context(eval_context);
            let eval_ptr = ggml_sys::ggml_get_mem_buffer(raw_eval_context);
            let eval_size = ggml_sys::ggml_get_mem_size(raw_eval_context);
            assert!(
                metal::ggml_metal_add_buffer(
                    raw_metal_context,
                    "eval".as_ptr().cast(),
                    eval_ptr,
                    eval_size
                ),
                "Could not add eval buffer to metal context"
            );
        }
    }

    fn ref_context(&mut self, context: Arc<Context>) {
        self.contexts.push(context);
    }

    /// Initializes the buffers needed for a metal forward pass.
    pub fn initialize_buffers(
        &mut self,
        context: Arc<Context>,
        memory_k: &mut Tensor,
        memory_v: &mut Tensor,
        scratch: &mut [Buffer],
    ) {
        unsafe {
            let raw_metal_context = self.ptr.as_ptr();

            //TODO check if this works with mmap
            let raw_context = context.ptr.as_ptr();
            let data_ptr = ggml_sys::ggml_get_mem_buffer(raw_context);
            let data_size = ggml_sys::ggml_get_mem_size(raw_context);
            assert!(
                metal::ggml_metal_add_buffer(
                    raw_metal_context,
                    "data\0".as_ptr().cast(),
                    data_ptr,
                    data_size
                ),
                "Could not add data buffer to metal context"
            );

            //This is the `kv` section from the original code, we dont have a joined kv buffer, so we need to add them seperately
            assert!(
                metal::ggml_metal_add_buffer(
                    raw_metal_context,
                    "k\0".as_ptr().cast(),
                    memory_k.data(),
                    memory_k.element_size()
                ),
                "Could not add k buffer to metal context"
            );

            assert!(
                metal::ggml_metal_add_buffer(
                    raw_metal_context,
                    "v\0".as_ptr().cast(),
                    memory_v.data(),
                    memory_v.element_size()
                ),
                "Could not add v buffer to metal context"
            );

            //Last we need to add the scratch buffers to the buffers
            for (i, buf) in scratch.iter().enumerate() {
                assert!(
                    metal::ggml_metal_add_buffer(
                        raw_metal_context,
                        "scrN\0".as_ptr().cast(), // FIXME: allocate string and insert number in name
                        buf.data.as_ptr() as *mut core::ffi::c_void,
                        buf.data.len()
                    ),
                    "{}",
                    format!("Could not add scratch buffer {} to metal context", i)
                );
            }
        }

        self.ref_context(context);
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
