#![allow(non_camel_case_types)]

use std::os::raw::{c_int, c_void};

pub type ggml_type = c_int;
pub const GGML_TYPE_Q4_0: ggml_type = 0;
pub const GGML_TYPE_Q4_1: ggml_type = 1;
pub const GGML_TYPE_I8: ggml_type = 2;
pub const GGML_TYPE_I16: ggml_type = 3;
pub const GGML_TYPE_I32: ggml_type = 4;
pub const GGML_TYPE_F16: ggml_type = 5;
pub const GGML_TYPE_F32: ggml_type = 6;
pub const GGML_TYPE_COUNT: ggml_type = 7;

pub type ggml_op = c_int;
pub const GGML_OP_NONE: ggml_op = 0;
pub const GGML_OP_DUP: ggml_op = 1;
pub const GGML_OP_ADD: ggml_op = 2;
pub const GGML_OP_SUB: ggml_op = 3;
pub const GGML_OP_MUL: ggml_op = 4;
pub const GGML_OP_DIV: ggml_op = 5;
pub const GGML_OP_SQR: ggml_op = 6;
pub const GGML_OP_SQRT: ggml_op = 7;
pub const GGML_OP_SUM: ggml_op = 8;
pub const GGML_OP_MEAN: ggml_op = 9;
pub const GGML_OP_REPEAT: ggml_op = 10;
pub const GGML_OP_ABS: ggml_op = 11;
pub const GGML_OP_SGN: ggml_op = 12;
pub const GGML_OP_NEG: ggml_op = 13;
pub const GGML_OP_STEP: ggml_op = 14;
pub const GGML_OP_RELU: ggml_op = 15;
pub const GGML_OP_GELU: ggml_op = 16;
pub const GGML_OP_SILU: ggml_op = 17;
pub const GGML_OP_NORM: ggml_op = 18;
pub const GGML_OP_MUL_MAT: ggml_op = 19;
pub const GGML_OP_SCALE: ggml_op = 20;
pub const GGML_OP_CPY: ggml_op = 21;
pub const GGML_OP_RESHAPE: ggml_op = 22;
pub const GGML_OP_VIEW: ggml_op = 23;
pub const GGML_OP_PERMUTE: ggml_op = 24;
pub const GGML_OP_TRANSPOSE: ggml_op = 25;
pub const GGML_OP_GET_ROWS: ggml_op = 26;
pub const GGML_OP_DIAG_MASK_INF: ggml_op = 27;
pub const GGML_OP_SOFT_MAX: ggml_op = 28;
pub const GGML_OP_ROPE: ggml_op = 29;
pub const GGML_OP_CONV_1D_1S: ggml_op = 30;
pub const GGML_OP_CONV_1D_2S: ggml_op = 31;
pub const GGML_OP_FLASH_ATTN: ggml_op = 32;
pub const GGML_OP_FLASH_FF: ggml_op = 33;
pub const GGML_OP_COUNT: ggml_op = 34;

pub type ggml_context = c_void;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_tensor {
    pub type_: ggml_type,
    pub n_dims: c_int,
    pub ne: [c_int; 4usize],
    pub nb: [usize; 4usize],
    pub op: ggml_op,
    pub is_param: bool,
    pub grad: *mut ggml_tensor,
    pub src0: *mut ggml_tensor,
    pub src1: *mut ggml_tensor,
    pub opt: [*mut ggml_tensor; 4usize],
    pub n_tasks: c_int,
    pub perf_runs: c_int,
    pub perf_cycles: i64,
    pub perf_time_us: i64,
    pub data: *mut c_void,
    pub padding: [::std::os::raw::c_char; 8usize],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_cgraph {
    pub n_nodes: c_int,
    pub n_leafs: c_int,
    pub n_threads: c_int,
    pub work_size: usize,
    pub work: *mut ggml_tensor,
    pub nodes: [*mut ggml_tensor; 4096usize],
    pub grads: [*mut ggml_tensor; 4096usize],
    pub leafs: [*mut ggml_tensor; 4096usize],
    pub perf_runs: c_int,
    pub perf_cycles: i64,
    pub perf_time_us: i64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_init_params {
    pub mem_size: usize,
    pub mem_buffer: *mut c_void,
}

extern "C" {
    pub fn ggml_nelements(tensor: *const ggml_tensor) -> c_int;

    pub fn ggml_nbytes(tensor: *const ggml_tensor) -> usize;

    pub fn ggml_blck_size(type_: ggml_type) -> c_int;

    pub fn ggml_type_size(type_: ggml_type) -> usize;

    pub fn ggml_type_sizef(type_: ggml_type) -> f32;

    pub fn ggml_element_size(tensor: *const ggml_tensor) -> usize;

    pub fn ggml_init(params: ggml_init_params) -> *mut ggml_context;

    pub fn ggml_free(ctx: *mut ggml_context);

    pub fn ggml_used_mem(ctx: *const ggml_context) -> usize;

    pub fn ggml_new_tensor_1d(
        ctx: *mut ggml_context,
        type_: ggml_type,
        ne0: c_int,
    ) -> *mut ggml_tensor;

    pub fn ggml_new_tensor_2d(
        ctx: *mut ggml_context,
        type_: ggml_type,
        ne0: c_int,
        ne1: c_int,
    ) -> *mut ggml_tensor;

    pub fn ggml_new_tensor_3d(
        ctx: *mut ggml_context,
        type_: ggml_type,
        ne0: c_int,
        ne1: c_int,
        ne2: c_int,
    ) -> *mut ggml_tensor;

    pub fn ggml_new_f32(ctx: *mut ggml_context, value: f32) -> *mut ggml_tensor;

    pub fn ggml_get_data(tensor: *const ggml_tensor) -> *mut c_void;

    pub fn ggml_add(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub fn ggml_mul(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub fn ggml_repeat(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub fn ggml_silu(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;

    pub fn ggml_norm(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;

    pub fn ggml_mul_mat(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub fn ggml_scale(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub fn ggml_cpy(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub fn ggml_reshape_3d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: c_int,
        ne1: c_int,
        ne2: c_int,
    ) -> *mut ggml_tensor;

    pub fn ggml_view_1d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: c_int,
        offset: usize,
    ) -> *mut ggml_tensor;

    pub fn ggml_permute(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        axis0: c_int,
        axis1: c_int,
        axis2: c_int,
        axis3: c_int,
    ) -> *mut ggml_tensor;

    pub fn ggml_get_rows(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;

    pub fn ggml_diag_mask_inf(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        n_past: c_int,
    ) -> *mut ggml_tensor;

    pub fn ggml_soft_max(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;

    pub fn ggml_rope(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        n_past: c_int,
        n_dims: c_int,
        mode: c_int,
    ) -> *mut ggml_tensor;

    pub fn ggml_build_forward_expand(cgraph: *mut ggml_cgraph, tensor: *mut ggml_tensor);

    pub fn ggml_graph_compute(ctx: *mut ggml_context, cgraph: *mut ggml_cgraph);

    pub fn ggml_quantize_q4_0(
        src: *mut f32,
        work: *mut c_void,
        n: i32,
        k: i32,
        qk: i32,
        hist: *mut i64,
    ) -> usize;

    pub fn ggml_quantize_q4_1(
        src: *mut f32,
        work: *mut c_void,
        n: i32,
        k: i32,
        qk: i32,
        hist: *mut i64,
    ) -> usize;

    pub fn ggml_fp16_to_fp32(x: u16) -> f32;
}
