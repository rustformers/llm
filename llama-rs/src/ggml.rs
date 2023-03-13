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
    pub fn init(params: ggml_raw::ggml_init_params) -> Self {
        let raw = unsafe { ggml_raw::ggml_init(params) };
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
}
