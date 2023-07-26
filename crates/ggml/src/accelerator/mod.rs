//! Functionality related to hardware acceleration of GGML (GPU, etc.)
use crate::sys;

#[cfg(feature = "metal")]
pub mod metal;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
/// Accelerators supported by `ggml`.
pub enum Accelerator {
    /// CuBLAS accelerated
    CuBLAS,
    /// CLBlast accelerated
    CLBlast,
    /// Metal accelerated
    Metal,
    /// Cpu accelerated
    None,
}

/// Returns the accelerator `ggml` was compiled with.
pub fn get_accelerator() -> Accelerator {
    #[cfg(feature = "clblast")]
    return Accelerator::CLBlast;
    #[cfg(feature = "cublas")]
    return Accelerator::CuBLAS;
    #[cfg(feature = "metal")]
    return Accelerator::Metal;
    #[cfg(not(any(feature = "cublas", feature = "clblast", feature = "metal")))]
    return Accelerator::None;
}

#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
/// Backend to use for a tensor.
pub enum Backend {
    /// CPU backend
    #[default]
    Cpu,
    /// GPU backend
    Gpu,
    /// Multi-GPU backend
    GpuSplit,
}

impl From<Backend> for sys::ggml_backend {
    fn from(b: Backend) -> Self {
        match b {
            Backend::Cpu => sys::ggml_backend_GGML_BACKEND_CPU,
            Backend::Gpu => sys::ggml_backend_GGML_BACKEND_GPU,
            Backend::GpuSplit => sys::ggml_backend_GGML_BACKEND_GPU_SPLIT,
        }
    }
}

impl TryFrom<sys::ggml_backend> for Backend {
    type Error = ();
    fn try_from(b: sys::ggml_backend) -> Result<Self, Self::Error> {
        match b {
            sys::ggml_backend_GGML_BACKEND_CPU => Ok(Backend::Cpu),
            sys::ggml_backend_GGML_BACKEND_GPU => Ok(Backend::Gpu),
            sys::ggml_backend_GGML_BACKEND_GPU_SPLIT => Ok(Backend::GpuSplit),
            _ => Err(()),
        }
    }
}

/// Initialize the accelerator. If ggml-sys is compiled with CUDA or CLBlast support, this function will initialize the accelerator. If not this is a no-op.
#[allow(unused_variables)]
pub fn initialize(device: i32) {
    #[cfg(feature = "cublas")]
    unsafe {
        //TODO: Make this configurable
        sys::cuda::ggml_init_cublas();
        sys::cuda::ggml_cuda_set_main_device(device);
        let split = 1.0f32;
        sys::cuda::ggml_cuda_set_tensor_split(&split as *const f32);
    }
}

///  Sets the scratch size for the GPU. If ggml-sys is compiled with CUDA support, this function will set the scratch size. If not this is a no-op.
#[allow(unused_variables)]
pub fn set_scratch_size(size: usize) {
    #[cfg(feature = "cublas")]
    unsafe {
        sys::cuda::ggml_cuda_set_scratch_size(size);
    }
}

/// Frees the scratch memory. If ggml-sys is compiled with CUDA support, this function will free the scratch memory. If not this is a no-op.
pub fn free_scratch() {
    #[cfg(feature = "cublas")]
    unsafe {
        sys::cuda::ggml_cuda_free_scratch();
    }
}
