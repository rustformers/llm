use std::{
    collections::HashMap,
    ffi::c_void,
    os::raw::c_int,
    ptr::NonNull,
    sync::{Arc, Mutex},
};

use memmap2::Mmap;

use crate::{
    accelerator::Backend, sys, usize_to_i32, usize_to_i64, Buffer, ComputationGraph, RoPEOverrides,
    Tensor, Type,
};

/// Acts as a RAII-guard over a `sys::ggml_context`, allocating via
/// `ggml_init` and dropping via `ggml_free`.
#[derive(PartialEq, Eq)]
pub struct Context {
    /// An `Arc` is used to model the relation between the context and the
    /// allocated tensors. Tensors are owned by the object, so a [`Tensor`]
    /// contains a `Weak` reference underneath and doesn't let you do anything
    /// with it if the underlying context has been deallocated.
    inner: Arc<ContextInner>,

    /// The storage for this context. This is stored so that the buffer can be dropped when the context is dropped.
    storage: Option<ContextStorage>,

    /// Whether the context can offload tensors to the GPU
    pub can_offload: bool,
}

/// Contains state shared between a context and its tensors
pub(crate) struct ContextInner {
    pub ptr: NonNull<sys::ggml_context>,

    /// Offloaded tensors. Used to free them when the context is dropped.
    // TODO: revisit this. What it means for a tensor to be "offloaded",
    // "transferred", etc. is not clear. This map is necessary because
    // there is no obvious heuristic for whether a given `Tensor`
    // should have the accelerator free method called on it.
    //
    // This is because tensors can be present on the accelerator without
    // having data (i.e. compute nodes), or they can refer to the scratch buffers.
    // Freeing these offloaded-but-not-allocated tensors will lead to crashes.
    //
    // Hopefully, this is resolved by GGML redesigning both its accelerator
    // interface and its scratch buffer solution.
    pub offloaded_tensors: Mutex<HashMap<String, Tensor>>,
}
impl PartialEq for ContextInner {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}
impl Eq for ContextInner {}
impl ContextInner {
    pub(crate) fn new(ptr: *mut ggml_sys::ggml_context) -> Arc<Self> {
        Arc::new(Self {
            ptr: NonNull::new(ptr).expect("Should not be null"),
            offloaded_tensors: Default::default(),
        })
    }
}

/// Controls how the context uses memory.
pub enum ContextStorage {
    /// Use the provided buffer as memory.
    Buffer(Buffer),
    /// Use the provided memory mapped file as memory.
    Mmap(Mmap),
    /// Allocate `mem_size` bytes of memory.
    Allocate {
        /// The size, in bytes, of the memory in to allocate.
        mem_size: usize,
    },
}
impl ContextStorage {
    /// Returns the `Mmap` if this is a `Mmap` variant.
    pub fn as_mmap(&self) -> Option<&Mmap> {
        match self {
            Self::Mmap(v) => Some(v),
            _ => None,
        }
    }

    /// Returns the `Buffer` if this is a `Buffer` variant.
    pub fn as_buffer(&self) -> Option<&Buffer> {
        match self {
            Self::Buffer(v) => Some(v),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn as_ptr_and_size(&self, ctx: &Context) -> (*mut c_void, usize) {
        match self {
            // This is a bit naughty...
            Self::Mmap(mmap) => (mmap.as_ptr().cast_mut() as *mut c_void, mmap.len()),
            _ => (
                ggml_sys::ggml_get_mem_buffer(ctx.as_ptr()),
                ggml_sys::ggml_get_mem_size(ctx.as_ptr()),
            ),
        }
    }
}
impl PartialEq for ContextStorage {
    fn eq(&self, other: &Self) -> bool {
        use ContextStorage::*;
        match (self, other) {
            (Buffer(l0), Buffer(r0)) => l0 == r0,
            (Mmap(l0), Mmap(r0)) => l0.as_ptr() == r0.as_ptr(),
            (Allocate { mem_size: l }, Allocate { mem_size: r }) => l == r,
            _ => false,
        }
    }
}
impl Eq for ContextStorage {}

impl Context {
    /// Creates a new [Context] with the given storage..
    pub fn new(storage: ContextStorage) -> Self {
        let init_params = match &storage {
            ContextStorage::Buffer(buffer) => sys::ggml_init_params {
                mem_size: buffer.size(),
                mem_buffer: buffer.data,
                no_alloc: false,
            },
            ContextStorage::Mmap(mmap) => sys::ggml_init_params {
                mem_size: mmap.len(),
                mem_buffer: std::ptr::null_mut(),
                // We are mmapping so ggml does not need to allocate any memory for us
                no_alloc: true,
            },
            ContextStorage::Allocate { mem_size } => sys::ggml_init_params {
                mem_size: *mem_size,
                // Null here means we want ggml to own this memory.
                mem_buffer: std::ptr::null_mut(),
                // It doesn't make sense to `no_alloc` when passing in a `mem_size` in this mode.
                no_alloc: false,
            },
        };

        let raw = unsafe { sys::ggml_init(init_params) };
        Self {
            inner: ContextInner::new(raw),
            storage: Some(storage),
            can_offload: false,
        }
    }

    /// Creates a new [Context] with the specified buffer.
    /// The buffer will be used by GGML.
    pub fn new_with_buffer(buffer: Buffer) -> Self {
        Self::new(ContextStorage::Buffer(buffer))
    }

    /// Creates a new [Context] with the specified memory mapped file.
    pub fn new_with_mmap(mmap: Mmap) -> Self {
        Self::new(ContextStorage::Mmap(mmap))
    }

    /// Creates a new [Context] with the specified memory size.
    /// The memory will be allocated by GGML.
    pub fn new_with_allocate(mem_size: usize) -> Self {
        Self::new(ContextStorage::Allocate { mem_size })
    }

    /// Recreates this context using the same storage.
    pub fn recreate(&mut self) {
        // This is the only operation that can consume the `self.storage`, so we can unwrap here.
        *self = Self::new(self.storage.take().unwrap());
    }

    ///Crate a new [ComputationGraph] in this context.
    pub fn create_compute_graph(&self) -> ComputationGraph {
        let context = self.inner.to_owned().ptr.as_ptr();
        unsafe {
            let graph = sys::ggml_new_graph(context);
            ComputationGraph::from_raw(graph)
        }
    }

    /// Prints all ggml objects in this context. Mainly used for debugging.
    pub fn list_ggml_objects(&self) {
        let context = self.inner.to_owned().ptr.as_ptr();
        unsafe { sys::ggml_print_objects(context) }
    }

    /// If offloading is enabled, all tensors created by this context will be offloaded to the GPU
    pub fn set_offloading(&mut self, can_offload: bool) {
        self.can_offload = can_offload;
    }

    /// Retrieves the memory used by this [Context].
    pub fn used_mem(&self) -> usize {
        unsafe { sys::ggml_used_mem(self.as_ptr()) }
    }

    /// Sets the scratch buffer to be used by this [Context].
    ///
    /// If `scratch_buffer` is `None`, the scratch buffer will be disabled.
    pub fn use_scratch<'a>(&'a self, scratch_buffer: Option<&'a Buffer>) {
        let (size, data) = if let Some(buffer) = scratch_buffer {
            (buffer.size(), buffer.data)
        } else {
            (0, std::ptr::null_mut())
        };
        // SAFETY: this just passes (most likely uninitialized) memory buffer to the ggml C API
        unsafe {
            sys::ggml_set_scratch(
                self.as_ptr(),
                sys::ggml_scratch {
                    offs: 0,
                    size,
                    data,
                },
            );
        }
    }

    /// Creates a new 1D tensor.
    pub fn new_tensor_1d(&self, typ: Type, ne0: usize) -> Tensor {
        let raw = unsafe { sys::ggml_new_tensor_1d(self.as_ptr(), typ.into(), usize_to_i64(ne0)) };
        self.new_tensor_raw(raw)
    }

    /// Creates a new 2D tensor.
    pub fn new_tensor_2d(&self, typ: Type, ne0: usize, ne1: usize) -> Tensor {
        let raw = unsafe {
            sys::ggml_new_tensor_2d(
                self.as_ptr(),
                typ.into(),
                usize_to_i64(ne0),
                usize_to_i64(ne1),
            )
        };
        self.new_tensor_raw(raw)
    }

    /// Creates a new 3D tensor.
    pub fn new_tensor_3d(&self, typ: Type, ne0: usize, ne1: usize, ne2: usize) -> Tensor {
        let raw = unsafe {
            sys::ggml_new_tensor_3d(
                self.as_ptr(),
                typ.into(),
                usize_to_i64(ne0),
                usize_to_i64(ne1),
                usize_to_i64(ne2),
            )
        };
        self.new_tensor_raw(raw)
    }

    /// Creates a new 1D tensor with the specified value.
    pub fn new_f32(&self, x: f32) -> Tensor {
        let raw = unsafe { sys::ggml_new_f32(self.as_ptr(), x) };
        self.new_tensor_raw(raw)
    }

    /// Returns a reference to the [ContextStorage] used by this [Context].
    pub fn storage(&self) -> &ContextStorage {
        self.storage.as_ref().unwrap()
    }
}
// Operations
impl Context {
    /// Unknown, aside from the obvious. It's transposing something!
    pub fn op_transpose(&self, a: &Tensor) -> Tensor {
        let tensor = unsafe { sys::ggml_transpose(self.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Unknown.
    pub fn op_get_rows(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor = unsafe { sys::ggml_get_rows(self.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with the values of `a`, but normalized.
    pub fn op_norm(&self, a: &Tensor) -> Tensor {
        let tensor = unsafe { sys::ggml_norm(self.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with the values of `a`, but normalized using RMSNorm.
    pub fn op_rms_norm(&self, a: &Tensor) -> Tensor {
        let tensor =
            unsafe { sys::ggml_rms_norm(self.as_ptr(), a.ptr.as_ptr(), crate::DEFAULT_EPS) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with the multiplication of `a` and `b`. Supports broadcasting if the dimensions are compatible, menaing the first dimensions of `a` must be devisible by the first dimensions of `b`.
    pub fn op_mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor = unsafe { sys::ggml_mul(self.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Repeats the `a` tensor along the first dimension of the `b` tensor.  
    pub fn op_repeat(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor = unsafe { sys::ggml_repeat(self.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with the multiplication of `a` and `b` as if they were matrices.
    ///
    /// `a`: m rows, n columns
    ///
    /// `b`: p rows, n columns (i.e. we transpose it internally)
    ///
    /// Result is m columns, p rows
    pub fn op_mul_mat(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor = unsafe { sys::ggml_mul_mat(self.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with the addition of `a` and `b`. Supports broadcasting if the dimensions are compatible, menaing the first dimensions of `a` must be devisible by the first dimensions of `b`.
    pub fn op_add(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor = unsafe { sys::ggml_add(self.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with the [SiLU](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html) activation function applied to `a`.
    pub fn op_silu(&self, a: &Tensor) -> Tensor {
        let tensor = unsafe { sys::ggml_silu(self.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Scales `a` by the 1D tensor `b`.
    pub fn op_scale(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor = unsafe { sys::ggml_scale(self.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// In-place, scales `a` by the 1D tensor `b`.
    pub fn op_scale_inplace(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor =
            unsafe { sys::ggml_scale_inplace(self.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Sets the elements above the diagonal to -INF.
    pub fn op_diag_mask_inf(&self, a: &Tensor, n_past: usize) -> Tensor {
        let tensor =
            unsafe { sys::ggml_diag_mask_inf(self.as_ptr(), a.ptr.as_ptr(), usize_to_i32(n_past)) };
        self.new_tensor_raw(tensor)
    }

    /// In-place, sets the elements above the diagonal to -INF.
    pub fn op_diag_mask_inf_inplace(&self, a: &Tensor, n_past: usize) -> Tensor {
        let tensor = unsafe {
            sys::ggml_diag_mask_inf_inplace(self.as_ptr(), a.ptr.as_ptr(), usize_to_i32(n_past))
        };
        self.new_tensor_raw(tensor)
    }

    /// Applies the [Softmax function](https://en.wikipedia.org/wiki/Softmax_function) to `a`.
    pub fn op_soft_max(&self, a: &Tensor) -> Tensor {
        let tensor = unsafe { sys::ggml_soft_max(self.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// In-place, applies the [Softmax function](https://en.wikipedia.org/wiki/Softmax_function) to `a`.
    pub fn op_soft_max_inplace(&self, a: &Tensor) -> Tensor {
        let tensor = unsafe { sys::ggml_soft_max_inplace(self.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with result of mapping `fun` with `a`.
    ///
    /// `cnt` is the number of `f32` elements to be mapped.
    /// `src` is source for elements to be mapped.
    /// `dst` is the destination for mapped elements.
    ///
    /// # Safety
    ///
    /// This is marked unsafe since we're passing pointers into C code, and not
    /// only vanilla pointers but a pointer to a function. For obvious reasons, it's
    /// important not to do anything crazy like mutate any of these values concurrently.
    ///
    /// Don't make assumptions about how/when the function will be called. It may be called
    /// on a row, it may be called on a whole tensor. It may be called concurrently or not.
    /// Once you give that function pointer to C land, all bets are off.
    pub unsafe fn op_map_unary(
        &self,
        a: &Tensor,
        fun: unsafe extern "C" fn(cnt: c_int, dst: *mut f32, src: *const f32),
    ) -> Tensor {
        let tensor = unsafe { sys::ggml_map_unary_f32(self.as_ptr(), a.ptr.as_ptr(), Some(fun)) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with result of mapping `fun` with `a` and `b`.
    ///
    /// `cnt` is the number of `f32` elements to be mapped.
    /// `src0`, `src1` are the sources of elements to be mapped.
    /// `dst` is the destination for mapped elements.
    ///
    /// # Safety
    ///
    /// This is marked unsafe since we're passing pointers into C code, and not
    /// only vanilla pointers but a pointer to a function. For obvious reasons, it's
    /// important not to do anything crazy like mutate any of these values concurrently.
    ///
    /// Don't make assumptions about how/when the function will be called. It may be called
    /// on a row, it may be called on a whole tensor. It may be called concurrently or not.
    /// Once you give that function pointer to C land, all bets are off.
    pub unsafe fn op_map_binary(
        &self,
        a: &Tensor,
        b: &Tensor,
        fun: unsafe extern "C" fn(cnt: c_int, dst: *mut f32, src0: *const f32, src1: *const f32),
    ) -> Tensor {
        let tensor = unsafe {
            sys::ggml_map_binary_f32(self.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr(), Some(fun))
        };
        self.new_tensor_raw(tensor)
    }

    /// Creates a 1D view over `a`.
    pub fn op_view_1d(&self, a: &Tensor, ne0: usize, offset: usize) -> Tensor {
        #[cfg(debug_assertions)]
        assert!(
            offset < a.nbytes(),
            "Cannot create tensor view with offset larger than tensor"
        );
        let tensor =
            unsafe { sys::ggml_view_1d(self.as_ptr(), a.ptr.as_ptr(), usize_to_i64(ne0), offset) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a 2D view over `a`.
    pub fn op_view_2d(&self, a: &Tensor, ne: (usize, usize), nb1: usize, offset: usize) -> Tensor {
        let (ne0, ne1) = ne;
        let tensor = unsafe {
            sys::ggml_view_2d(
                self.as_ptr(),
                a.ptr.as_ptr(),
                usize_to_i64(ne0),
                usize_to_i64(ne1),
                nb1,
                offset,
            )
        };
        self.new_tensor_raw(tensor)
    }

    /// Creates a 3d view over `a`.
    pub fn op_view_3d(
        &self,
        a: &Tensor,
        ne: (usize, usize, usize),
        nb: (usize, usize),
        offset: usize,
    ) -> Tensor {
        let (ne0, ne1, ne2) = ne;
        let (nb1, nb2) = nb;
        let tensor = unsafe {
            sys::ggml_view_3d(
                self.as_ptr(),
                a.ptr.as_ptr(),
                usize_to_i64(ne0),
                usize_to_i64(ne1),
                usize_to_i64(ne2),
                nb1,
                nb2,
                offset,
            )
        };
        self.new_tensor_raw(tensor)
    }

    /// Copies `a` to `b` and returns `b`.
    pub fn op_cpy(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor = unsafe { sys::ggml_cpy(self.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with the axes of `a` permuted as described by the parameters.
    pub fn op_permute(&self, a: &Tensor, axes: (usize, usize, usize, usize)) -> Tensor {
        let tensor = unsafe {
            sys::ggml_permute(
                self.as_ptr(),
                a.ptr.as_ptr(),
                usize_to_i32(axes.0),
                usize_to_i32(axes.1),
                usize_to_i32(axes.2),
                usize_to_i32(axes.3),
            )
        };
        self.new_tensor_raw(tensor)
    }

    /// In-place; reshapes `a` in accordance with the dimensions of `b`
    pub fn op_reshape(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor = unsafe { sys::ggml_reshape(self.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// In-place; reshapes `a` in accordance with the specified dimensions.
    pub fn op_reshape_2d(&self, a: &Tensor, ne0: usize, ne1: usize) -> Tensor {
        let tensor = unsafe {
            sys::ggml_reshape_2d(
                self.as_ptr(),
                a.ptr.as_ptr(),
                usize_to_i64(ne0),
                usize_to_i64(ne1),
            )
        };
        self.new_tensor_raw(tensor)
    }

    /// In-place; reshapes `a` in accordance with the specified dimensions.
    pub fn op_reshape_3d(&self, a: &Tensor, ne0: usize, ne1: usize, ne2: usize) -> Tensor {
        let tensor = unsafe {
            sys::ggml_reshape_3d(
                self.as_ptr(),
                a.ptr.as_ptr(),
                usize_to_i64(ne0),
                usize_to_i64(ne1),
                usize_to_i64(ne2),
            )
        };
        self.new_tensor_raw(tensor)
    }

    /// ggml_cont
    pub fn op_cont(&self, a: &Tensor) -> Tensor {
        let tensor = unsafe { sys::ggml_cont(self.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Applies ROtary Positional Encoding.
    pub fn op_rope(&self, a: &Tensor, npast: usize, ndims: usize, mode: i32) -> Tensor {
        let tensor = unsafe {
            sys::ggml_rope(
                self.as_ptr(),
                a.ptr.as_ptr(),
                usize_to_i32(npast),
                usize_to_i32(ndims),
                mode,
                0,
            )
        };
        self.new_tensor_raw(tensor)
    }

    /// In-place; applies ROtary Positional Encoding.
    pub fn op_rope_inplace(
        &self,
        a: &Tensor,
        npast: usize,
        ndims: usize,
        mode: i32,
        overrides: Option<&RoPEOverrides>,
    ) -> Tensor {
        let tensor = unsafe {
            if let Some(custom_args) = overrides {
                sys::ggml_rope_custom_inplace(
                    self.as_ptr(),
                    a.ptr.as_ptr(),
                    usize_to_i32(npast),
                    usize_to_i32(ndims),
                    mode,
                    1,
                    custom_args.frequency_base as f32,
                    custom_args.frequency_scale,
                )
            } else {
                sys::ggml_rope_inplace(
                    self.as_ptr(),
                    a.ptr.as_ptr(),
                    usize_to_i32(npast),
                    usize_to_i32(ndims),
                    mode,
                    0,
                )
            }
        };
        self.new_tensor_raw(tensor)
    }

    /// Attention with LInear BIases (Ref: <https://arxiv.org/pdf/2108.12409.pdf>)
    pub fn op_alibi(&self, a: &Tensor, n_past: usize, n_head: usize, bias_max: f32) -> Tensor {
        let tensor = unsafe {
            sys::ggml_alibi(
                self.as_ptr(),
                a.ptr.as_ptr(),
                usize_to_i32(n_past),
                usize_to_i32(n_head),
                bias_max,
            )
        };

        self.new_tensor_raw(tensor)
    }

    /// Gaussian Error Linear Units
    pub fn op_gelu(&self, a: &Tensor) -> Tensor {
        let tensor = unsafe { sys::ggml_gelu(self.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// flash attention.
    pub fn op_flash_attn(&self, q: &Tensor, k: &Tensor, v: &Tensor, masked: bool) -> Tensor {
        let tensor = unsafe {
            sys::ggml_flash_attn(
                self.as_ptr(),
                q.ptr.as_ptr(),
                k.ptr.as_ptr(),
                v.ptr.as_ptr(),
                masked,
            )
        };
        self.new_tensor_raw(tensor)
    }
}
// Public to this crate methods
impl Context {
    pub(crate) fn as_ptr(&self) -> *mut sys::ggml_context {
        self.inner.ptr.as_ptr()
    }
}
// Private methods
impl Context {
    /// Wraps a raw tensor with a weak pointer to the context.
    fn new_tensor_raw(&self, raw: *mut sys::ggml_tensor) -> Tensor {
        let tensor = Tensor {
            ptr: NonNull::new(raw).expect("Should not be null"),
            inner: Arc::downgrade(&self.inner),
        };

        if self.can_offload {
            tensor.offload();
        }
        tensor
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: The only non-weak copy of ptr is no longer accessible after this drop call.
        unsafe {
            // if we moved tensors to an accelerator we need to free them
            for (_, tensor) in self.inner.offloaded_tensors.lock().unwrap().drain() {
                if tensor.backend() != Backend::Cpu {
                    tensor.free_accelerator();
                }
            }
            sys::ggml_free(self.as_ptr());
        }
    }
}
