//! Metal support.
use std::{ptr::NonNull, sync::Arc, ffi::CString, borrow::BorrowMut};
use crate::{sys::metal, ComputationGraph, Tensor, Context, Buffer};

/// Acts as a RAII-guard over a `sys::metal::ggml_metal_context`, allocating via
/// `ggml_metal_init` and dropping via `ggml_metal_free`.
pub struct MetalContext {
    ptr: Arc<NonNull<metal::ggml_metal_context>>,
}

impl Default for MetalContext {
    fn default() -> Self {
        let raw = unsafe { metal::ggml_metal_init() };

        MetalContext {
            ptr: Arc::new(NonNull::new(raw).expect("Should not be null")),
        }
    }


    /// Initializes the buffers needed for a metal forward pass.
    pub fn initialize_buffers(&self, 
        context: &Context,
        eval_context: &Context,
        memory_k:&mut Tensor,
        memory_v:&mut Tensor,
        scratch: &mut [Buffer]){
        unsafe{
            let raw_metal_context = self.ptr.as_ptr();

            //TODO check if this works with mmap
            let raw_context = context.ptr.as_ptr();
            let data_ptr = ggml_sys::ggml_get_mem_buffer(raw_context);
            let data_size = ggml_sys::ggml_get_mem_size(raw_context);
            let data_name = CString::new("data").unwrap();
            assert!(metal::ggml_metal_add_buffer(raw_metal_context, data_name.as_ptr() ,data_ptr, data_size),"Could not add data buffer to metal context");

            // in our implementation this should be the `ctx0` buffer
            // Original code: ggml_metal_add_buffer(ctx->ctx_metal, "eval", ctx->buf_compute.addr, ctx->buf_compute.size)
            let raw_eval_context = eval_context.ptr.as_ptr();
            let eval_ptr = ggml_sys::ggml_get_mem_buffer(raw_eval_context);
            let eval_size = ggml_sys::ggml_get_mem_size(raw_eval_context);
            let eval_name = CString::new("eval").unwrap();
            assert!(metal::ggml_metal_add_buffer(raw_metal_context, eval_name.as_ptr() ,eval_ptr, eval_size),"Could not add eval buffer to metal context");

            //This is the `kv` section from the original code, we dont have a joined kv buffer, so we need to add them seperately
            let k_name = CString::new("k").unwrap();
            assert!(metal::ggml_metal_add_buffer(raw_metal_context, k_name.as_ptr() ,memory_k.data(), memory_k.element_size()), "Could not add k buffer to metal context");

            let v_name = CString::new("v").unwrap();
            assert!(metal::ggml_metal_add_buffer(raw_metal_context, v_name.as_ptr() ,memory_v.data(), memory_v.element_size()), "Could not add v buffer to metal context");

            //Last we need to add the scratch buffers to the buffers
            for (i,buf) in scratch.iter().enumerate(){
                let name = CString::new(format!("scr{}",i)).unwrap();
                assert!(metal::ggml_metal_add_buffer(raw_metal_context, name.as_ptr() ,buf.data.as_ptr() as *mut core::ffi::c_void, buf.data.len()), "{}", format!("Could not add scratch buffer {} to metal context",i));
            }
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
        // SAFETY: The only non-weak copy of ptr is no longer accessible after
        // this drop call.
        unsafe { metal::ggml_metal_free(self.ptr.as_ptr()) }
    }
}
