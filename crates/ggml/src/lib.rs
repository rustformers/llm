//! `ggml` is a semi-idiomatic wrapper for the `ggml` C library.
//!
//! It exposes a subset of operations (currently used to implement the [llm](https://crates.io/crates/llm) library).
//! Note that it does not expose a fully-idiomatic safe Rust interface; operations that could be potentially unsafe are marked as such.
//!
//! `ggml` operates on a computational graph; no values will be computed until the [Context] is executed via an [GraphExecutionPlan].
//! All [Tensor]s are nodes in this computational graph, and values cannot be retrieved until computation is completed.
#![deny(missing_docs)]

use std::{
    alloc::Layout,
    os::raw::{c_int, c_void},
    ptr::NonNull,
    sync::Arc,
};

mod context;
mod tensor;

pub mod format;
pub mod util;

pub mod accelerator;

pub use context::{Context, ContextStorage};

pub use tensor::Tensor;

pub use ggml_sys as sys;

/// The type of a tensor element.
pub type ElementType = Type;

/// The current quantization version.
pub const QNT_VERSION: u32 = sys::GGML_QNT_VERSION;
/// The factor by which to divide `ftype` to determine the current quantization version.
pub const QNT_VERSION_FACTOR: u32 = sys::GGML_QNT_VERSION_FACTOR;

/// The size of a `ggml` object.
pub const OBJECT_SIZE: usize = sys::GGML_OBJECT_SIZE;

/// The maximum length of a `ggml` tensor-name.
pub const MAX_NAME_LENGTH: usize = sys::GGML_MAX_NAME as usize;

/// Default epsilon to use for RMS computation.
pub const DEFAULT_EPS: f32 = 0.000005;

/// Alignment used for the Tensors in a `ggml` graph.
pub const TENSOR_ALIGNMENT: usize = 32;

/// Value overrides to use for RoPE.
///
/// Formula: `theta_i = scale * base^(−2(i−1)/d), for i in [1, 2, ..., d/2]`
#[derive(Debug, Clone)]
pub struct RoPEOverrides {
    /// The original context length.
    pub original_context_length: usize,
    /// The frequency scale to use.
    pub frequency_scale: f32,
    /// The frequency base value to use.
    pub frequency_base: usize,

    /// TODO
    pub ext_factor: f32,
    /// TODO
    pub attn_factor: f32,
    /// TODO
    pub beta_fast: f32,
    /// TODO
    pub beta_slow: f32,
}

impl Default for RoPEOverrides {
    fn default() -> Self {
        Self {
            // Not really sure this should have a default, but if we're the only users for now, it's probably OK?
            original_context_length: 2048,
            frequency_scale: 1.0,
            frequency_base: 10_000,
            ext_factor: -1.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default, PartialOrd, Ord)]
/// The type of a value in `ggml`.
pub enum Type {
    /// Quantized 4-bit (type 0).
    #[default]
    Q4_0,
    /// Quantized 4-bit (type 1).
    Q4_1,
    /// Quantized 5-bit (type 0).
    Q5_0,
    /// Quantized 5-bit (type 1).
    Q5_1,
    /// Quantized 8-bit (type 0).
    Q8_0,
    /// Quantized 8-bit (type 1).
    Q8_1,
    /// K-Quantized 2-bit.
    #[allow(non_camel_case_types)]
    Q2_K,
    /// K-Quantized 3-bit.
    #[allow(non_camel_case_types)]
    Q3_K,
    /// K-Quantized 4-bit.
    #[allow(non_camel_case_types)]
    Q4_K,
    /// K-Quantized 5-bit.
    #[allow(non_camel_case_types)]
    Q5_K,
    /// K-Quantized 6-bit.
    #[allow(non_camel_case_types)]
    Q6_K,
    /// Integer 32-bit.
    I32,
    /// Float 16-bit.
    F16,
    /// Float 32-bit.
    F32,
    /// Integer 8-bit.
    I8,
}
impl From<Type> for sys::ggml_type {
    fn from(t: Type) -> Self {
        match t {
            Type::Q4_0 => sys::ggml_type_GGML_TYPE_Q4_0,
            Type::Q4_1 => sys::ggml_type_GGML_TYPE_Q4_1,
            Type::Q5_0 => sys::ggml_type_GGML_TYPE_Q5_0,
            Type::Q5_1 => sys::ggml_type_GGML_TYPE_Q5_1,
            Type::Q8_0 => sys::ggml_type_GGML_TYPE_Q8_0,
            Type::Q8_1 => sys::ggml_type_GGML_TYPE_Q8_1,
            Type::Q2_K => sys::ggml_type_GGML_TYPE_Q2_K,
            Type::Q3_K => sys::ggml_type_GGML_TYPE_Q3_K,
            Type::Q4_K => sys::ggml_type_GGML_TYPE_Q4_K,
            Type::Q5_K => sys::ggml_type_GGML_TYPE_Q5_K,
            Type::Q6_K => sys::ggml_type_GGML_TYPE_Q6_K,
            Type::I32 => sys::ggml_type_GGML_TYPE_I32,
            Type::F16 => sys::ggml_type_GGML_TYPE_F16,
            Type::F32 => sys::ggml_type_GGML_TYPE_F32,
            Type::I8 => sys::ggml_type_GGML_TYPE_I8,
        }
    }
}
impl TryFrom<sys::ggml_type> for Type {
    type Error = ();
    fn try_from(t: sys::ggml_type) -> Result<Self, Self::Error> {
        match t {
            sys::ggml_type_GGML_TYPE_Q4_0 => Ok(Type::Q4_0),
            sys::ggml_type_GGML_TYPE_Q4_1 => Ok(Type::Q4_1),
            sys::ggml_type_GGML_TYPE_Q5_0 => Ok(Type::Q5_0),
            sys::ggml_type_GGML_TYPE_Q5_1 => Ok(Type::Q5_1),
            sys::ggml_type_GGML_TYPE_Q8_0 => Ok(Type::Q8_0),
            sys::ggml_type_GGML_TYPE_Q8_1 => Ok(Type::Q8_1),
            sys::ggml_type_GGML_TYPE_Q2_K => Ok(Type::Q2_K),
            sys::ggml_type_GGML_TYPE_Q3_K => Ok(Type::Q3_K),
            sys::ggml_type_GGML_TYPE_Q4_K => Ok(Type::Q4_K),
            sys::ggml_type_GGML_TYPE_Q5_K => Ok(Type::Q5_K),
            sys::ggml_type_GGML_TYPE_Q6_K => Ok(Type::Q6_K),
            sys::ggml_type_GGML_TYPE_I32 => Ok(Type::I32),
            sys::ggml_type_GGML_TYPE_F16 => Ok(Type::F16),
            sys::ggml_type_GGML_TYPE_F32 => Ok(Type::F32),
            sys::ggml_type_GGML_TYPE_I8 => Ok(Type::I8),

            _ => Err(()),
        }
    }
}
impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Q4_0 => write!(f, "q4_0"),
            Type::Q4_1 => write!(f, "q4_1"),
            Type::Q5_0 => write!(f, "q5_0"),
            Type::Q5_1 => write!(f, "q5_1"),
            Type::Q8_0 => write!(f, "q8_0"),
            Type::Q8_1 => write!(f, "q8_1"),
            Type::Q2_K => write!(f, "q2_k"),
            Type::Q3_K => write!(f, "q3_k"),
            Type::Q4_K => write!(f, "q4_k"),
            Type::Q5_K => write!(f, "q5_k"),
            Type::Q6_K => write!(f, "q6_k"),
            Type::I32 => write!(f, "i32"),
            Type::F16 => write!(f, "f16"),
            Type::F32 => write!(f, "f32"),
            Type::I8 => write!(f, "i8"),
        }
    }
}
impl Type {
    /// Returns whether this type is quantized.
    pub fn is_quantized(&self) -> bool {
        match self {
            Type::Q4_0 => true,
            Type::Q4_1 => true,
            Type::Q5_0 => true,
            Type::Q5_1 => true,
            Type::Q8_0 => true,
            Type::Q8_1 => true,
            Type::Q2_K => true,
            Type::Q3_K => true,
            Type::Q4_K => true,
            Type::Q5_K => true,
            Type::Q6_K => true,
            Type::I32 => false,
            Type::F16 => false,
            Type::F32 => false,
            Type::I8 => false,
        }
    }
}

/// A buffer of memory that can be used as a buffer for a [Context] or [GraphAllocator].
#[derive(PartialEq, Eq, Debug)]
pub struct Buffer {
    data: *mut c_void,
    layout: Layout,
}

const BUFFER_ALIGN: usize = 16384;

impl Buffer {
    /// Creates a new buffer of the specified size.
    pub fn new(size: usize) -> Self {
        let layout = Layout::from_size_align(size, BUFFER_ALIGN).unwrap();

        unsafe {
            Buffer {
                data: std::alloc::alloc(layout).cast(),
                layout,
            }
        }
    }

    /// Creates a new buffer of the specified size, without aligning it.
    pub fn new_unaligned(size: usize) -> Self {
        let layout = Layout::from_size_align(size, 1).unwrap();

        unsafe {
            Buffer {
                data: std::alloc::alloc(layout).cast(),
                layout,
            }
        }
    }

    /// Returns the size of the buffer in bytes
    pub fn size(&self) -> usize {
        self.layout.size()
    }

    /// Returns a pointer to the data in this buffer.
    pub fn data(&mut self) -> *mut c_void {
        self.data
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(self.data.cast(), self.layout);
        }
    }
}

/// A `ggml` computation graph. Keeps track of all state during computation.
pub struct ComputationGraph {
    inner: *mut sys::ggml_cgraph,
}

impl ComputationGraph {
    /// Create a new [ComputationGraph] from a raw [sys::ggml_cgraph].
    pub fn from_raw(raw_context: *mut sys::ggml_cgraph) -> Self {
        Self { inner: raw_context }
    }

    /// Build this computational graph in the forward direction in preparation for computation.
    pub fn build_forward_expand(&mut self, tensor: &Tensor) {
        unsafe { sys::ggml_build_forward_expand(self.inner, tensor.ptr.as_ptr()) }
    }

    /// Returns the leafs in this graph.
    pub fn leafs(&self, context: &Context) -> Vec<Tensor> {
        let mut wrapped_leafs: Vec<Tensor> = vec![];

        for leaf in self.leafs_slice() {
            if !leaf.is_null() {
                wrapped_leafs.push(Tensor {
                    ptr: NonNull::new(*leaf).expect("Should not be null"),
                    inner: Arc::downgrade(&context.inner),
                })
            }
        }
        wrapped_leafs
    }
    /// Returns the nodes in this graph.
    pub fn nodes(&self, context: &Context) -> Vec<Tensor> {
        let mut wrapped_nodes: Vec<Tensor> = vec![];

        for leaf in self.nodes_slice() {
            if !leaf.is_null() {
                wrapped_nodes.push(Tensor {
                    ptr: NonNull::new(*leaf).expect("Should not be null"),
                    inner: Arc::downgrade(&context.inner),
                })
            }
        }
        wrapped_nodes
    }
}
impl ComputationGraph {
    fn leafs_slice(&self) -> &[*mut sys::ggml_tensor] {
        unsafe {
            std::slice::from_raw_parts(
                self.inner.as_ref().unwrap().leafs,
                self.inner.as_ref().unwrap().n_leafs as usize,
            )
        }
    }

    fn nodes_slice(&self) -> &[*mut sys::ggml_tensor] {
        unsafe {
            std::slice::from_raw_parts(
                self.inner.as_ref().unwrap().nodes,
                self.inner.as_ref().unwrap().n_nodes as usize,
            )
        }
    }
}

/// A `ggml` execution plan. Contains the information needed to execute a computation graph.
pub struct GraphExecutionPlan {
    inner: sys::ggml_cplan,
    inner_graph: *mut sys::ggml_cgraph,
}

impl GraphExecutionPlan {
    /// Create a new [GraphExecutionPlan] from a [ComputationGraph] and the number of threads to use.
    pub fn new(graph: &mut ComputationGraph, n_threads: usize) -> Self {
        Self {
            inner: unsafe { sys::ggml_graph_plan(graph.inner, usize_to_i32(n_threads)) },
            inner_graph: graph.inner,
        }
    }

    /// Execute this [GraphExecutionPlan] in the given [Context].
    pub fn execute(&mut self, buffer: &mut Vec<u8>) {
        if self.inner.work_size > 0 {
            buffer.resize(self.inner.work_size, 0);
            self.inner.work_data = buffer.as_mut_ptr().cast();
        }

        unsafe {
            sys::ggml_graph_compute(self.inner_graph, &mut self.inner);
        }
    }
}

#[derive(PartialEq, Eq, Debug)]
/// Acts as a RAII-guard over a `sys::ggml_allocr`, allocating via
/// `ggml_allocr_new` and dropping via `ggml_allocr_free`.
/// Used to allocate the memory used by a computational graph.
pub struct GraphAllocator {
    /// The underlying `sys::ggml_allocr` pointer.
    pub ptr: *mut sys::ggml_allocr,
    /// The buffer used by this allocator.
    pub buffer: Buffer,
}

impl GraphAllocator {
    /// Create a new allocator with the specified buffer.
    pub fn new(buffer: Buffer, tensor_alignment: usize) -> Self {
        let ptr = unsafe { sys::ggml_allocr_new(buffer.data, buffer.size(), tensor_alignment) };
        Self { ptr, buffer }
    }

    /// Create a new allocator to measure a computational graph.
    pub fn new_measurement(tensor_alignment: usize) -> Self {
        let ptr = unsafe { sys::ggml_allocr_new_measure(tensor_alignment) };
        let buffer = Buffer::new(tensor_alignment);
        Self { ptr, buffer }
    }

    /// Allocates a computational graph in the allocator and returns the size in bytes.
    pub fn allocate_graph(&self, graph: &ComputationGraph) -> usize {
        unsafe { sys::ggml_allocr_alloc_graph(self.ptr, graph.inner) }
    }

    /// Resets the allocator for a new forward pass.
    pub fn reset(&self) {
        unsafe { sys::ggml_allocr_reset(self.ptr) }
    }

    /// Returns true if the allocator is in measuring mode.
    pub fn in_measuring_mode(&self) -> bool {
        unsafe { sys::ggml_allocr_is_measure(self.ptr) }
    }

    /// Allocates memory for a given tensor in the allocator.
    pub fn allocate(&self, tensor: &Tensor) {
        unsafe { sys::ggml_allocr_alloc(self.ptr, tensor.ptr.as_ptr()) }
    }

    /// Switches the buffer used by the allocator.
    pub fn resize_buffer(&mut self, graph_size: usize, tensor_alignment: usize) {
        // Free the old allocator
        unsafe { sys::ggml_allocr_free(self.ptr) }
        //Resize the buffer
        self.buffer = Buffer::new_unaligned(graph_size);
        // Create a new allocator with the new buffer
        self.ptr =
            unsafe { sys::ggml_allocr_new(self.buffer.data, self.buffer.size(), tensor_alignment) };
    }
}

impl Drop for GraphAllocator {
    fn drop(&mut self) {
        unsafe { sys::ggml_allocr_free(self.ptr) }
    }
}

/// The size of `t` as bytes.
pub fn type_size(t: Type) -> usize {
    unsafe { sys::ggml_type_size(t.into()) }
}

/// [type_size]/[blck_size] as float.
pub fn type_sizef(x: Type) -> f64 {
    (unsafe { sys::ggml_type_sizef(x.into()) }) as f64
}

/// The size of a block for `t`. Only relevant for quantized types.
pub fn blck_size(t: Type) -> usize {
    i32_to_usize(unsafe { sys::ggml_blck_size(t.into()) })
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

/// Contains the result of a quantization operation.
pub struct QuantizationResult {
    /// The quantized output.
    pub output: Vec<u8>,
    /// The quantization history.
    pub history: Vec<i64>,
}

/// Quantizes `src` into `dst` using `q4_0` quantization.
///
/// You must ensure that `src.len() == n_elements`, and `n_elements_0`
/// is the first dimension of `src`.
pub fn quantize_q4_0(src: &[f32], n_elements: usize, n_elements_0: usize) -> QuantizationResult {
    quantize_impl(src, n_elements, n_elements_0, sys::ggml_quantize_q4_0)
}

/// Quantizes `src` into `dst` using `q4_1` quantization.
///
/// You must ensure that `src.len() == n_elements`, and `n_elements_0`
/// is the first dimension of `src`.
pub fn quantize_q4_1(src: &[f32], n_elements: usize, n_elements_0: usize) -> QuantizationResult {
    quantize_impl(src, n_elements, n_elements_0, sys::ggml_quantize_q4_1)
}

/// Quantizes `src` into `dst` using `q5_0` quantization.
///
/// You must ensure that `src.len() == n_elements`, and `n_elements_0`
/// is the first dimension of `src`.
pub fn quantize_q5_0(src: &[f32], n_elements: usize, n_elements_0: usize) -> QuantizationResult {
    quantize_impl(src, n_elements, n_elements_0, sys::ggml_quantize_q5_0)
}

/// Quantizes `src` into `dst` using `q5_1` quantization.
///
/// You must ensure that `src.len() == n_elements`, and `n_elements_0`
/// is the first dimension of `src`.
pub fn quantize_q5_1(src: &[f32], n_elements: usize, n_elements_0: usize) -> QuantizationResult {
    quantize_impl(src, n_elements, n_elements_0, sys::ggml_quantize_q5_1)
}

/// Quantizes `src` into `dst` using `q8_0` quantization.
///
/// You must ensure that `src.len() == n_elements`, and `n_elements_0`
/// is the first dimension of `src`.
pub fn quantize_q8_0(src: &[f32], n_elements: usize, n_elements_0: usize) -> QuantizationResult {
    quantize_impl(src, n_elements, n_elements_0, sys::ggml_quantize_q8_0)
}

fn quantize_impl(
    src: &[f32],
    n_elements: usize,
    n_elements_0: usize,
    quantizer: unsafe extern "C" fn(*const f32, *mut c_void, c_int, c_int, *mut i64) -> usize,
) -> QuantizationResult {
    assert_eq!(src.len(), n_elements);
    assert_eq!(n_elements % n_elements_0, 0);

    // A conservative multiplier of 4 is used here.
    let mut output = vec![0u8; n_elements * 4];
    let mut history = vec![0i64; 16];
    let output_size = unsafe {
        quantizer(
            src.as_ptr(),
            output.as_mut_ptr() as *mut c_void,
            n_elements.try_into().unwrap(),
            n_elements_0.try_into().unwrap(),
            history.as_mut_ptr(),
        )
    };

    output.resize(output_size, 0u8);
    QuantizationResult { output, history }
}

/// Returns true if the current system has BLAS support.
pub fn cpu_has_blas() -> bool {
    unsafe { sys::ggml_cpu_has_blas() != 0 }
}

/// Returns true if the current system has GPU BLAS support.
pub fn cpu_has_gpublas() -> bool {
    unsafe { sys::ggml_cpu_has_gpublas() != 0 }
}

/// Returns the graph overhead in bytes.
pub fn graph_overhead() -> usize {
    unsafe { sys::ggml_graph_overhead() }
}

/// Returns the tensor overhead in bytes.
pub fn tensor_overhead() -> usize {
    unsafe { sys::ggml_tensor_overhead() }
}
