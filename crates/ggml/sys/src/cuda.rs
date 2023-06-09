/* automatically generated by rust-bindgen 0.65.1 */

use super::ggml_compute_params;
use super::ggml_tensor;

pub const GGML_CUDA_MAX_DEVICES: u32 = 16;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ggml_tensor_extra_gpu {
    pub data_device: [*mut ::std::os::raw::c_void; 16usize],
}
#[test]
fn bindgen_test_layout_ggml_tensor_extra_gpu() {
    const UNINIT: ::std::mem::MaybeUninit<ggml_tensor_extra_gpu> =
        ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<ggml_tensor_extra_gpu>(),
        128usize,
        concat!("Size of: ", stringify!(ggml_tensor_extra_gpu))
    );
    assert_eq!(
        ::std::mem::align_of::<ggml_tensor_extra_gpu>(),
        8usize,
        concat!("Alignment of ", stringify!(ggml_tensor_extra_gpu))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).data_device) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(ggml_tensor_extra_gpu),
            "::",
            stringify!(data_device)
        )
    );
}
extern "C" {
    pub fn ggml_init_cublas();
}
extern "C" {
    pub fn ggml_cuda_set_tensor_split(tensor_split: *const f32);
}
extern "C" {
    pub fn ggml_cuda_mul(src0: *const ggml_tensor, src1: *const ggml_tensor, dst: *mut ggml_tensor);
}
extern "C" {
    pub fn ggml_cuda_can_mul_mat(
        src0: *const ggml_tensor,
        src1: *const ggml_tensor,
        dst: *mut ggml_tensor,
    ) -> bool;
}
extern "C" {
    pub fn ggml_cuda_mul_mat_get_wsize(
        src0: *const ggml_tensor,
        src1: *const ggml_tensor,
        dst: *mut ggml_tensor,
    ) -> usize;
}
extern "C" {
    pub fn ggml_cuda_mul_mat(
        src0: *const ggml_tensor,
        src1: *const ggml_tensor,
        dst: *mut ggml_tensor,
        wdata: *mut ::std::os::raw::c_void,
        wsize: usize,
    );
}
extern "C" {
    pub fn ggml_cuda_host_malloc(size: usize) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    pub fn ggml_cuda_host_free(ptr: *mut ::std::os::raw::c_void);
}
extern "C" {
    pub fn ggml_cuda_transform_tensor(data: *mut ::std::os::raw::c_void, tensor: *mut ggml_tensor);
}
extern "C" {
    pub fn ggml_cuda_free_data(tensor: *mut ggml_tensor);
}
extern "C" {
    pub fn ggml_cuda_assign_buffers(tensor: *mut ggml_tensor);
}
extern "C" {
    pub fn ggml_cuda_assign_buffers_no_scratch(tensor: *mut ggml_tensor);
}
extern "C" {
    pub fn ggml_cuda_set_main_device(main_device: ::std::os::raw::c_int);
}
extern "C" {
    pub fn ggml_cuda_set_scratch_size(scratch_size: usize);
}
extern "C" {
    pub fn ggml_cuda_free_scratch();
}
extern "C" {
    pub fn ggml_cuda_compute_forward(
        params: *mut ggml_compute_params,
        tensor: *mut ggml_tensor,
    ) -> bool;
}
