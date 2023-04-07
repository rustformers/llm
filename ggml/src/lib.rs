#![deny(missing_docs)]

//! `ggml` is a semi-idiomatic wrapper for the `ggml` C library.
//!
//! It exposes a subset of operations (currently used to implement the [llama-rs](https://crates.io/crates/llama-rs) library).
//! Note that it does not expose a fully-idiomatic safe Rust interface; operations that could be potentially unsafe are marked as such.
//!
//! `ggml` operates on a computational graph; no values will be computed until [Context::graph_compute] is executed.
//! All [Tensor]s are nodes in this computational graph, and values cannot be retrieved until computation is completed.

use std::{
    ffi::c_void,
    ptr::NonNull,
    sync::{Arc, Weak},
};

/// Magic constant for `ggml` files (versioned, ggmf).
pub const FILE_MAGIC_GGMF: u32 = 0x67676d66;
/// Magic constant for `ggml` files (versioned, ggjt).
pub const FILE_MAGIC_GGJT: u32 = 0x67676a74;
/// Magic constant for `ggml` files (unversioned).
pub const FILE_MAGIC_UNVERSIONED: u32 = 0x67676d6c;

/// The currently-supported format version for `ggml` files.
pub const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
/// The type of a value in `ggml`.
pub enum Type {
    /// Quantized 4-bit (type 0).
    Q4_0,
    /// Quantized 4-bit (type 1); used by GPTQ.
    Q4_1,
    /// Integer 32-bit.
    I32,
    /// Float 16-bit.
    F16,
    /// Float 32-bit.
    F32,
}
impl From<Type> for ggml_sys::ggml_type {
    fn from(t: Type) -> Self {
        match t {
            Type::Q4_0 => ggml_sys::ggml_type_GGML_TYPE_Q4_0,
            Type::Q4_1 => ggml_sys::ggml_type_GGML_TYPE_Q4_1,
            Type::I32 => ggml_sys::ggml_type_GGML_TYPE_I32,
            Type::F16 => ggml_sys::ggml_type_GGML_TYPE_F16,
            Type::F32 => ggml_sys::ggml_type_GGML_TYPE_F32,
        }
    }
}
impl TryFrom<ggml_sys::ggml_type> for Type {
    type Error = ();
    fn try_from(t: ggml_sys::ggml_type) -> Result<Self, Self::Error> {
        match t {
            ggml_sys::ggml_type_GGML_TYPE_Q4_0 => Ok(Type::Q4_0),
            ggml_sys::ggml_type_GGML_TYPE_Q4_1 => Ok(Type::Q4_1),
            ggml_sys::ggml_type_GGML_TYPE_I32 => Ok(Type::I32),
            ggml_sys::ggml_type_GGML_TYPE_F16 => Ok(Type::F16),
            ggml_sys::ggml_type_GGML_TYPE_F32 => Ok(Type::F32),
            _ => Err(()),
        }
    }
}
impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Q4_0 => write!(f, "q4_0"),
            Type::Q4_1 => write!(f, "q4_1"),
            Type::I32 => write!(f, "i32"),
            Type::F16 => write!(f, "f16"),
            Type::F32 => write!(f, "f32"),
        }
    }
}

/// Acts as a RAII-guard over a `ggml_sys::ggml_context`, allocating via
/// `ggml_init` and dropping via `ggml_free`.
pub struct Context {
    /// An `Arc` is used to model the relation between the context and the
    /// allocated tensors. Tensors are owned by the object, so a [`Tensor`]
    /// contains a `Weak` reference underneath and doesn't let you do anything
    /// with it if the underlying context has been deallocated.
    ptr: Arc<NonNull<ggml_sys::ggml_context>>,
}
impl Context {
    /// Creates a new [Context] with the specified `mem_size` as a working area.
    pub fn init(mem_size: usize) -> Self {
        let raw = unsafe {
            ggml_sys::ggml_init(ggml_sys::ggml_init_params {
                mem_size,
                // Null here means we want ggml to own this memory. We don't
                // support passing an owned buffer from the Rust side.
                mem_buffer: std::ptr::null_mut(),
                no_alloc: false,
            })
        };
        Self {
            ptr: Arc::new(NonNull::new(raw).expect("Should not be null")),
        }
    }

    /// Wraps a raw tensor with a weak pointer to the context.
    fn new_tensor_raw(&self, raw: *mut ggml_sys::ggml_tensor) -> Tensor {
        Tensor {
            ptr: NonNull::new(raw).expect("Should not be null"),
            ctx: Arc::downgrade(&self.ptr),
        }
    }

    /// Creates a new 1D tensor.
    pub fn new_tensor_1d(&self, typ: Type, ne0: usize) -> Tensor {
        let raw = unsafe {
            ggml_sys::ggml_new_tensor_1d(self.ptr.as_ptr(), typ.into(), usize_to_i64(ne0))
        };
        self.new_tensor_raw(raw)
    }

    /// Creates a new 2D tensor.
    pub fn new_tensor_2d(&self, typ: Type, ne0: usize, ne1: usize) -> Tensor {
        let raw = unsafe {
            ggml_sys::ggml_new_tensor_2d(
                self.ptr.as_ptr(),
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
            ggml_sys::ggml_new_tensor_3d(
                self.ptr.as_ptr(),
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
        let raw = unsafe { ggml_sys::ggml_new_f32(self.ptr.as_ptr(), x) };
        self.new_tensor_raw(raw)
    }

    /// Unknown, aside from the obvious. It's transposing something!
    pub fn op_transpose(&self, a: &Tensor) -> Tensor {
        let tensor = unsafe { ggml_sys::ggml_transpose(self.ptr.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Unknown.
    pub fn op_get_rows(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor =
            unsafe { ggml_sys::ggml_get_rows(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with the values of `a`, but normalized.
    pub fn op_norm(&self, a: &Tensor) -> Tensor {
        let tensor = unsafe { ggml_sys::ggml_norm(self.ptr.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with the values of `a`, but normalized using RMSNorm.
    pub fn op_rms_norm(&self, a: &Tensor) -> Tensor {
        let tensor = unsafe { ggml_sys::ggml_rms_norm(self.ptr.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with the multiplication of `a` and `b`.
    pub fn op_mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor =
            unsafe { ggml_sys::ggml_mul(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Unknown.
    pub fn op_repeat(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor =
            unsafe { ggml_sys::ggml_repeat(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
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
        let tensor =
            unsafe { ggml_sys::ggml_mul_mat(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with the addition of `a` and `b`.
    pub fn op_add(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor =
            unsafe { ggml_sys::ggml_add(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with the [SiLU](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html) activation function applied to `a`.
    pub fn op_silu(&self, a: &Tensor) -> Tensor {
        let tensor = unsafe { ggml_sys::ggml_silu(self.ptr.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// In-place, scales `a` by the 1D tensor `b`.
    pub fn op_scale(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor =
            unsafe { ggml_sys::ggml_scale(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// In-place, sets the elements above the diagonal to -INF.
    pub fn op_diag_mask_inf(&self, a: &Tensor, n_past: usize) -> Tensor {
        let tensor = unsafe {
            ggml_sys::ggml_diag_mask_inf(self.ptr.as_ptr(), a.ptr.as_ptr(), usize_to_i32(n_past))
        };
        self.new_tensor_raw(tensor)
    }

    /// In-place, applies the [Softmax function](https://en.wikipedia.org/wiki/Softmax_function) to `a`.
    pub fn op_soft_max(&self, a: &Tensor) -> Tensor {
        let tensor = unsafe { ggml_sys::ggml_soft_max(self.ptr.as_ptr(), a.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a 1D view over `a`.
    pub fn op_view_1d(&self, a: &Tensor, ne0: usize, offset: usize) -> Tensor {
        let tensor = unsafe {
            ggml_sys::ggml_view_1d(self.ptr.as_ptr(), a.ptr.as_ptr(), usize_to_i64(ne0), offset)
        };
        self.new_tensor_raw(tensor)
    }

    /// Creates a 2D view over `a`.
    pub fn op_view_2d(
        &self,
        a: &Tensor,
        ne0: usize,
        ne1: usize,
        nb1: usize,
        offset: usize,
    ) -> Tensor {
        let tensor = unsafe {
            ggml_sys::ggml_view_2d(
                self.ptr.as_ptr(),
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
    #[allow(clippy::too_many_arguments)]
    pub fn op_view_3d(
        &self,
        a: &Tensor,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        nb1: usize,
        nb2: usize,
        offset: usize,
    ) -> Tensor {
        let tensor = unsafe {
            ggml_sys::ggml_view_3d(
                self.ptr.as_ptr(),
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
        let tensor =
            unsafe { ggml_sys::ggml_cpy(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// Creates a new tensor with the axes of `a` permuted as described by the parameters.
    pub fn op_permute(
        &self,
        a: &Tensor,
        axis0: usize,
        axis1: usize,
        axis2: usize,
        axis3: usize,
    ) -> Tensor {
        let tensor = unsafe {
            ggml_sys::ggml_permute(
                self.ptr.as_ptr(),
                a.ptr.as_ptr(),
                usize_to_i32(axis0),
                usize_to_i32(axis1),
                usize_to_i32(axis2),
                usize_to_i32(axis3),
            )
        };
        self.new_tensor_raw(tensor)
    }

    /// In-place; reshapes `a` in accordance with the dimensions of `b`
    pub fn op_reshape(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let tensor =
            unsafe { ggml_sys::ggml_reshape(self.ptr.as_ptr(), a.ptr.as_ptr(), b.ptr.as_ptr()) };
        self.new_tensor_raw(tensor)
    }

    /// In-place; reshapes `a` in accordance with the specified dimensions.
    pub fn op_reshape_2d(&self, a: &Tensor, ne0: usize, ne1: usize) -> Tensor {
        let tensor = unsafe {
            ggml_sys::ggml_reshape_2d(
                self.ptr.as_ptr(),
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
            ggml_sys::ggml_reshape_3d(
                self.ptr.as_ptr(),
                a.ptr.as_ptr(),
                usize_to_i64(ne0),
                usize_to_i64(ne1),
                usize_to_i64(ne2),
            )
        };
        self.new_tensor_raw(tensor)
    }

    /// In-place; applies ROtary Positional Encoding.
    pub fn op_rope(&self, a: &Tensor, npast: usize, ndims: usize, mode: i32) -> Tensor {
        let tensor = unsafe {
            ggml_sys::ggml_rope(
                self.ptr.as_ptr(),
                a.ptr.as_ptr(),
                usize_to_i32(npast),
                usize_to_i32(ndims),
                mode,
            )
        };
        self.new_tensor_raw(tensor)
    }

    /// Computes the specified graph. Must be run in order to evaluate the graph.
    pub fn graph_compute(&self, graph: &mut ComputationGraph) {
        unsafe {
            ggml_sys::ggml_graph_compute(self.ptr.as_ptr(), &mut graph.inner);
        }
    }

    /// Retrieves the memory used by this [Context].
    pub fn used_mem(&self) -> usize {
        unsafe { ggml_sys::ggml_used_mem(self.ptr.as_ptr()) }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: The only non-weak copy of ptr is no longer accessible after
        // this drop call.
        unsafe {
            ggml_sys::ggml_free(self.ptr.as_ptr());
        }
    }
}

/// Tensors are owned by the context. A tensor is alive as long as the
/// underlying context it was created with is alive.
pub struct Tensor {
    ptr: NonNull<ggml_sys::ggml_tensor>,
    ctx: Weak<NonNull<ggml_sys::ggml_context>>,
}

impl Tensor {
    /// Creates a shared copy of this tensor pointer.
    pub fn share(&self) -> Self {
        Tensor {
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

    /// Number of bytes used by this tensor.
    pub fn nbytes(&self) -> usize {
        self.with_alive_ctx(|| {
            // SAFETY: The with_alive_call guarantees the context is alive
            unsafe { ggml_sys::ggml_nbytes(self.ptr.as_ptr()) }
        })
    }

    /// Provides raw mutable access to the data contained within the tensor.
    ///
    /// # Safety
    ///
    /// Only `std::slice::from_raw_parts_mut(tensor.data(), tensor.nbytes())` is safe to mutate.
    pub unsafe fn data(&self) -> *mut c_void {
        self.with_alive_ctx(|| {
            // SAFETY: The with_alive_call guarantees the context is alive
            unsafe { *self.ptr.as_ptr() }.data
        })
    }

    // /// Set the tensor's data pointer (useful for mmap-ed data)
    // ///
    // /// # Safety
    // /// 
    // /// The memory region from `data_ptr` to `data_ptr.offset(tensor.nbytes())` will be read from.
    // pub unsafe fn set_data(&self, data_ptr: *mut c_void) {
    //     self.with_alive_ctx(|| {
    //         // SAFETY: The with_alive_call guarantees the context is alive
    //         unsafe { *self.ptr.as_ptr() }.data = data_ptr;
    //     })
    // }

    /// Number of elements in this tensor.
    pub fn nelements(&self) -> usize {
        self.with_alive_ctx(|| {
            // SAFETY: The with_alive_call guarantees the context is alive
            i64_to_usize(unsafe { ggml_sys::ggml_nelements(self.ptr.as_ptr()) })
        })
    }

    /// Number of elements in each dimension.
    pub fn get_ne(&self) -> [i64; 4] {
        self.with_alive_ctx(|| unsafe { *self.ptr.as_ptr() }.ne)
    }

    /// Stride of each dimension.
    pub fn get_nb(&self) -> [usize; 4] {
        self.with_alive_ctx(|| unsafe { *self.ptr.as_ptr() }.nb)
    }

    /// The data type.
    pub fn get_type(&self) -> Type {
        self.with_alive_ctx(|| unsafe { *self.ptr.as_ptr() }.type_.try_into().unwrap())
    }

    /// The size of the element type in bytes.
    pub fn element_size(&self) -> usize {
        self.with_alive_ctx(|| unsafe { ggml_sys::ggml_element_size(self.ptr.as_ptr()) })
    }

    /// Writes `src` to this tensor.
    ///
    /// # Safety
    ///
    /// This tensor must not be written to or read by from any other code.
    pub unsafe fn write_data(&self, src: &[u8]) {
        std::ptr::copy_nonoverlapping(src.as_ptr(), self.data() as *mut u8, src.len())
    }

    /// Zeroes out this tensor.
    pub fn zero_data(&self) {
        unsafe { std::ptr::write_bytes(self.data() as *mut u8, 0, self.nbytes()) }
    }

    /// Reads this tensor into `dst`, starting from `offset`.
    ///
    /// # Safety
    ///
    /// This tensor must not be written to or read by from any other code.
    pub unsafe fn read_data(&self, offset: usize, dst: &mut [u8]) {
        let data = unsafe { ggml_sys::ggml_get_data(self.ptr.as_ptr()).add(offset) };
        std::ptr::copy_nonoverlapping(data, dst as *mut _ as _, dst.len())
    }
}

/// A `ggml` computation graph. Keeps track of all state during computation.
pub struct ComputationGraph {
    inner: ggml_sys::ggml_cgraph,
}

impl ComputationGraph {
    /// Create a new [ComputationGraph] with the specified `n_threads`.
    pub fn new(n_threads: usize) -> Self {
        Self {
            inner: ggml_sys::ggml_cgraph {
                n_threads: usize_to_i32(n_threads),
                // SAFETY: This should be safe to zero. The original C++ impl
                // just leaves it uninitialized
                ..unsafe { std::mem::zeroed::<ggml_sys::ggml_cgraph>() }
            },
        }
    }

    /// Build this computational graph in the forward direction in preparation for computation.
    pub fn build_forward_expand(&mut self, tensor: &Tensor) {
        unsafe { ggml_sys::ggml_build_forward_expand(&mut self.inner, tensor.ptr.as_ptr()) }
    }
}

/// The size of `t` as bytes.
pub fn type_size(t: Type) -> usize {
    unsafe { ggml_sys::ggml_type_size(t.into()) }
}

/// [type_size]/[blck_size] as float.
pub fn type_sizef(x: Type) -> f64 {
    (unsafe { ggml_sys::ggml_type_sizef(x.into()) }) as f64
}

/// The size of a block for `t`. Only relevant for quantized types.
pub fn blck_size(t: Type) -> usize {
    i32_to_usize(unsafe { ggml_sys::ggml_blck_size(t.into()) })
}

fn usize_to_i32(val: usize) -> i32 {
    i32::try_from(val).unwrap()
}

fn usize_to_i64(val: usize) -> i64 {
    i64::try_from(val).unwrap()
}

fn i32_to_usize(val: i32) -> usize {
    usize::try_from(val).unwrap()
}

fn i64_to_usize(val: i64) -> usize {
    usize::try_from(val).unwrap()
}
