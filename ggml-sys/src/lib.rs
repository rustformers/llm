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
    pub fn ggml_rms_norm(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;

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
}
