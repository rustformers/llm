use std::{
    ffi::c_void,
    marker::PhantomData,
    ptr::{addr_of, NonNull},
    sync::{Arc, Weak},
};

pub const GGML_TYPE_Q4_0: ggml_raw::ggml_type = ggml_raw::ggml_type_GGML_TYPE_Q4_0;
pub const GGML_TYPE_Q4_1: ggml_raw::ggml_type = ggml_raw::ggml_type_GGML_TYPE_Q4_1;
pub const GGML_TYPE_I8: ggml_raw::ggml_type = ggml_raw::ggml_type_GGML_TYPE_I8;
pub const GGML_TYPE_I16: ggml_raw::ggml_type = ggml_raw::ggml_type_GGML_TYPE_I16;
pub const GGML_TYPE_I32: ggml_raw::ggml_type = ggml_raw::ggml_type_GGML_TYPE_I32;
pub const GGML_TYPE_F16: ggml_raw::ggml_type = ggml_raw::ggml_type_GGML_TYPE_F16;
pub const GGML_TYPE_F32: ggml_raw::ggml_type = ggml_raw::ggml_type_GGML_TYPE_F32;
pub const GGML_TYPE_COUNT: ggml_raw::ggml_type = ggml_raw::ggml_type_GGML_TYPE_COUNT;

/// Acts as a RAII-guard over a `ggml_raw::ggml_context`, allocating via
/// ggml_init and dropping via ggml_free
pub struct GgmlContext {
    /// An `Arc` is used to model the relation between the context and the
    /// allocated tensors. Tensors are owned by the object, so a [`GgmlTensor`]
    /// contains a `Weak` reference underneath and doesn't let you do anything
    /// with it if the underlying context has been deallocated.
    ptr: Arc<NonNull<ggml_raw::ggml_context>>,
}
impl GgmlContext {
    pub fn init(mem_size: usize) -> Self {
        let raw = unsafe {
            ggml_raw::ggml_init(ggml_raw::ggml_init_params {
                mem_size,
                // Null here means we want ggml to own this memory. We don't
                // support passing an owned buffer from the Rust side.
                mem_buffer: std::ptr::null_mut(),
            })
        };
        Self {
            ptr: Arc::new(NonNull::new(raw).expect("Should not be null")),
        }
    }

    fn new_tensor_raw(&self, raw: *mut ggml_raw::ggml_tensor) -> GgmlTensor {
        GgmlTensor {
            ptr: NonNull::new(raw).expect("Should not be null"),
            ctx: Arc::downgrade(&self.ptr),
        }
    }

    pub fn new_tensor_1d(&self, typ: ggml_raw::ggml_type, ne0: i32) -> GgmlTensor {
        let raw = unsafe { ggml_raw::ggml_new_tensor_1d(self.ptr.as_ptr(), typ, ne0) };
        self.new_tensor_raw(raw)
    }

    pub fn new_tensor_2d(&self, typ: ggml_raw::ggml_type, ne0: i32, ne1: i32) -> GgmlTensor {
        let raw = unsafe { ggml_raw::ggml_new_tensor_2d(self.ptr.as_ptr(), typ, ne0, ne1) };
        self.new_tensor_raw(raw)
    }

    pub fn new_tensor_3d(
        &self,
        typ: ggml_raw::ggml_type,
        ne0: i32,
        ne1: i32,
        ne2: i32,
    ) -> GgmlTensor {
        let raw = unsafe { ggml_raw::ggml_new_tensor_3d(self.ptr.as_ptr(), typ, ne0, ne1, ne2) };
        self.new_tensor_raw(raw)
    }

    pub fn new_f32(&self, x: f32) -> GgmlTensor {
        let raw = unsafe { ggml_raw::ggml_new_f32(self.ptr.as_ptr(), x) };
        self.new_tensor_raw(raw)
    }

    pub fn op_get_rows(&self, a: &GgmlTensor, b: &GgmlTensor) -> GgmlTensor {
        let tensor =
            unsafe { ggml_raw::ggml_get_rows(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    pub fn op_norm(&self, a: &GgmlTensor) -> GgmlTensor {
        let tensor = unsafe { ggml_raw::ggml_norm(self.ptr.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    pub fn op_mul(&self, a: &GgmlTensor, b: &GgmlTensor) -> GgmlTensor {
        let tensor =
            unsafe { ggml_raw::ggml_mul(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    pub fn op_repeat(&self, a: &GgmlTensor, b: &GgmlTensor) -> GgmlTensor {
        let tensor =
            unsafe { ggml_raw::ggml_repeat(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    pub fn op_mul_mat(&self, a: &GgmlTensor, b: &GgmlTensor) -> GgmlTensor {
        let tensor =
            unsafe { ggml_raw::ggml_mul_mat(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    pub fn op_add(&self, a: &GgmlTensor, b: &GgmlTensor) -> GgmlTensor {
        let tensor =
            unsafe { ggml_raw::ggml_add(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    pub fn op_silu(&self, a: &GgmlTensor) -> GgmlTensor {
        let tensor = unsafe { ggml_raw::ggml_silu(self.ptr.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    pub fn op_scale(&self, a: &GgmlTensor, b: &GgmlTensor) -> GgmlTensor {
        let tensor =
            unsafe { ggml_raw::ggml_scale(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    pub fn op_diag_mask_inf(&self, a: &GgmlTensor, n_past: i32) -> GgmlTensor {
        let tensor =
            unsafe { ggml_raw::ggml_diag_mask_inf(self.ptr.as_ptr(), a.ptr.as_ptr(), n_past) };
        self.new_tensor_raw(tensor)
    }

    pub fn op_soft_max(&self, a: &GgmlTensor) -> GgmlTensor {
        let tensor = unsafe { ggml_raw::ggml_soft_max(self.ptr.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    pub fn op_view_1d(&self, a: &GgmlTensor, ne0: i32, offset: usize) -> GgmlTensor {
        let tensor =
            unsafe { ggml_raw::ggml_view_1d(self.ptr.as_ptr(), a.ptr.as_ptr(), ne0, offset) };
        self.new_tensor_raw(tensor)
    }

    pub fn op_cpy(&self, a: &GgmlTensor, b: &GgmlTensor) -> GgmlTensor {
        let tensor =
            unsafe { ggml_raw::ggml_cpy(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    pub fn op_permute(
        &self,
        a: &GgmlTensor,
        axis0: i32,
        axis1: i32,
        axis2: i32,
        axis3: i32,
    ) -> GgmlTensor {
        let tensor = unsafe {
            ggml_raw::ggml_permute(
                self.ptr.as_ptr(),
                a.ptr.as_ptr(),
                axis0,
                axis1,
                axis2,
                axis3,
            )
        };
        self.new_tensor_raw(tensor)
    }
    pub fn op_reshape_3d(&self, a: &GgmlTensor, ne0: i32, ne1: i32, ne2: i32) -> GgmlTensor {
        let tensor =
            unsafe { ggml_raw::ggml_reshape_3d(self.ptr.as_ptr(), a.ptr.as_ptr(), ne0, ne1, ne2) };
        self.new_tensor_raw(tensor)
    }

    pub fn op_rope(&self, a: &GgmlTensor, npast: i32, ndims: i32, mode: i32) -> GgmlTensor {
        let tensor =
            unsafe { ggml_raw::ggml_rope(self.ptr.as_ptr(), a.ptr.as_ptr(), npast, ndims, mode) };
        self.new_tensor_raw(tensor)
    }

    pub fn graph_compute(&self, graph: &mut GgmlCGraph) {
        unsafe {
            ggml_raw::ggml_graph_compute(self.ptr.as_ptr(), &mut graph.inner);
        }
    }

    pub fn used_mem(&self) -> usize {
        unsafe { ggml_raw::ggml_used_mem(self.ptr.as_ptr()) }
    }
}

impl Drop for GgmlContext {
    fn drop(&mut self) {
        // SAFETY: The only non-weak copy of ptr is no longer accessible after
        // this drop call.
        unsafe {
            ggml_raw::ggml_free(self.ptr.as_ptr());
        }
    }
}

/// Tensors are owned by the context. A tensor is alive as long as the
/// underlying context it was created with is alive.
pub struct GgmlTensor {
    ptr: NonNull<ggml_raw::ggml_tensor>,
    ctx: Weak<NonNull<ggml_raw::ggml_context>>,
}

impl GgmlTensor {
    /// Creates a shared copy of this tensor pointer.
    pub fn share(&self) -> Self {
        GgmlTensor {
            ptr: self.ptr,
            ctx: Weak::clone(&self.ctx),
        }
    }

    fn with_alive_ctx<U>(&self, f: impl Fn() -> U) -> U {
        if let Some(_ctx) = self.ctx.upgrade() {
            f()
        } else {
            panic!("Using a tensor after the context was dropped")
        }
    }

    pub fn nbytes(&self) -> usize {
        self.with_alive_ctx(|| {
            // SAFETY: The with_alive_call guarantees the context is alive
            unsafe { ggml_raw::ggml_nbytes(self.ptr.as_ptr()) }
        })
    }

    pub fn data(&self) -> *mut c_void {
        self.with_alive_ctx(|| {
            // SAFETY: The with_alive_call guarantees the context is alive
            unsafe { *self.ptr.as_ptr() }.data
        })
    }

    pub fn nelements(&self) -> i32 {
        self.with_alive_ctx(|| {
            // SAFETY: The with_alive_call guarantees the context is alive
            unsafe { ggml_raw::ggml_nelements(self.ptr.as_ptr()) }
        })
    }

    pub fn get_ne(&self) -> [i32; 4] {
        self.with_alive_ctx(|| unsafe { *self.ptr.as_ptr() }.ne)
    }

    pub fn get_nb(&self) -> [usize; 4] {
        self.with_alive_ctx(|| unsafe { *self.ptr.as_ptr() }.nb)
    }

    pub fn get_type(&self) -> ggml_raw::ggml_type {
        self.with_alive_ctx(|| unsafe { *self.ptr.as_ptr() }.type_)
    }

    pub fn element_size(&self) -> usize {
        self.with_alive_ctx(|| unsafe { ggml_raw::ggml_element_size(self.ptr.as_ptr()) })
    }

    pub unsafe fn write_data(&self, src: &[u8]) {
        std::ptr::copy_nonoverlapping(src.as_ptr(), self.data() as *mut u8, src.len())
    }

    pub unsafe fn read_data(&self, offset: usize, dst: &mut [u8]) {
        let data = unsafe { ggml_raw::ggml_get_data(self.ptr.as_ptr()) };
        std::ptr::copy_nonoverlapping(data, dst as *mut _ as _, dst.len())
    }
}

pub struct GgmlCGraph {
    inner: ggml_raw::ggml_cgraph,
}

impl GgmlCGraph {
    pub fn new(n_threads: i32) -> Self {
        Self {
            inner: ggml_raw::ggml_cgraph {
                n_threads,
                // SAFETY: This should be safe to zero. The original C++ impl
                // just leaves it uninitialized
                ..unsafe { std::mem::zeroed::<ggml_raw::ggml_cgraph>() }
            },
        }
    }

    pub fn build_forward_expand(&mut self, tensor: &GgmlTensor) {
        unsafe { ggml_raw::ggml_build_forward_expand(&mut self.inner, tensor.ptr.as_ptr()) }
    }
}
