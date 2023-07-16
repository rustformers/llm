use std::{os::raw::c_void, ptr::NonNull, sync::Weak};

use crate::{
    accelerator::Backend, context::ContextInner, i64_to_usize, sys, Type, MAX_NAME_LENGTH,
};

/// Tensors are owned by the context. A tensor is alive as long as the
/// underlying context it was created with is alive.
pub struct Tensor {
    pub(crate) ptr: NonNull<sys::ggml_tensor>,
    pub(crate) inner: Weak<ContextInner>,
}

impl Tensor {
    /// Size of the `ggml_tensor` struct in bytes.
    ///
    /// Exposed for purposes of determining context size.
    pub const C_TYPE_SIZE: usize = std::mem::size_of::<sys::ggml_tensor>();

    /// Sets the name of the tensor.
    ///
    /// # Safety
    ///
    /// The name must be a valid UTF-8 string and must not be longer than [`MAX_NAME_LENGTH`] bytes.
    pub fn set_name(mut self, name: &str) -> Tensor {
        assert!(
            name.len() <= MAX_NAME_LENGTH,
            "Tensor name must be less than {} bytes",
            MAX_NAME_LENGTH
        );

        let c_name = std::ffi::CString::new(name).unwrap();
        self.with_alive_ctx_mut(|t| unsafe { sys::ggml_set_name(t.ptr.as_ptr(), c_name.as_ptr()) });
        self
    }

    /// Gets the name of the tensor
    pub fn name(&self) -> String {
        self.with_alive_ctx(|| {
            let name_ptr = unsafe { sys::ggml_get_name(self.ptr.as_ptr()) };
            let name = unsafe { std::ffi::CStr::from_ptr(name_ptr) };
            name.to_string_lossy().into_owned()
        })
    }

    /// Gets the acceleration backend of the tensor
    pub fn backend(&self) -> Backend {
        self.with_alive_ctx(|| unsafe {
            (self.ptr.as_ref().backend as sys::ggml_backend)
                .try_into()
                .unwrap()
        })
    }

    /// Sets the tensor's acceleration backend and moves the tensor's data to the new backend.
    pub fn transfer_to(mut self, backend: Backend) -> Tensor {
        self.with_alive_ctx_mut(|t| {
            let current_backend = t.backend();

            if current_backend != Backend::Cpu && backend == Backend::Cpu {
                unimplemented!("Tensors cannot be moved from an accelerator to the CPU at present");
            }
            if backend == Backend::Cpu {
                return;
            }
            t.set_backend(backend);

            #[cfg(feature = "cublas")]
            unsafe {
                sys::cuda::ggml_cuda_transform_tensor(t.data(), t.ptr.as_ptr());
            }
            #[cfg(feature = "clblast")]
            unsafe {
                sys::opencl::ggml_cl_transform_tensor(t.data(), t.ptr.as_ptr());
            }

            t.mark_as_offloaded();
        });
        self
    }

    /// If ggml-sys is compiled with CUDA support, this function will offload the tensor to the GPU.
    /// If not, this is a no-op.
    ///
    /// It will not transfer the data. Use `transfer_to` for that.
    #[allow(unused_variables)]
    pub fn offload(&self) {
        self.with_alive_ctx(|| {
            #[cfg(feature = "cublas")]
            unsafe {
                sys::cuda::ggml_cuda_assign_buffers(self.ptr.as_ptr());
            }
        })
    }

    /// If ggml-sys is compiled with CUDA support, this function will offload the tensor to the GPU without using the scratch buffer.
    /// If not, this is a no-op.
    ///
    /// It will not transfer the data. Use `transfer_to` for that.
    ///
    /// Unlike `offload`, this function will add the tensor to the offloaded tensors map. This is because the non-use of a scratch buffer
    /// allows us to safely assume that this tensor will actually point to data.
    #[allow(unused_variables)]
    pub fn offload_no_scratch(&self) {
        self.with_alive_ctx(|| {
            #[cfg(feature = "cublas")]
            unsafe {
                sys::cuda::ggml_cuda_assign_buffers_no_scratch(self.ptr.as_ptr());
            }
            self.mark_as_offloaded();
        })
    }

    /// Creates a shared copy of this tensor pointer.
    pub fn share(&self) -> Self {
        Tensor {
            ptr: self.ptr,
            inner: Weak::clone(&self.inner),
        }
    }

    /// Number of bytes used by this tensor.
    pub fn nbytes(&self) -> usize {
        self.with_alive_ctx(|| {
            // SAFETY: The with_alive_call guarantees the context is alive
            unsafe { sys::ggml_nbytes(self.ptr.as_ptr()) }
        })
    }

    /// Provides raw mutable access to the data contained within the tensor.
    ///
    /// # Safety
    ///
    /// Only `std::slice::from_raw_parts_mut(tensor.data(), tensor.nbytes())` is safe to mutate.
    pub unsafe fn data(&mut self) -> *mut c_void {
        self.with_alive_ctx(|| {
            // SAFETY: The with_alive_call guarantees the context is alive
            unsafe { *self.ptr.as_ptr() }.data
        })
    }

    /// Set the tensor's data pointer (useful for mmap-ed data)
    ///
    /// # Safety
    ///
    /// The memory region from `data_ptr` to `data_ptr.offset(tensor.nbytes())` will be read from.
    pub unsafe fn set_data(&mut self, data_ptr: *mut c_void) {
        self.with_alive_ctx_mut(|t| {
            let tensor = t.ptr.as_mut();
            // SAFETY: The with_alive_call guarantees the context is alive
            tensor.data = data_ptr;
        })
    }

    /// Number of elements in this tensor.
    pub fn nelements(&self) -> usize {
        self.with_alive_ctx(|| {
            // SAFETY: The with_alive_call guarantees the context is alive
            i64_to_usize(unsafe { sys::ggml_nelements(self.ptr.as_ptr()) })
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
        self.with_alive_ctx(|| unsafe { sys::ggml_element_size(self.ptr.as_ptr()) })
    }

    /// Writes `src` to this tensor.
    ///
    /// # Safety
    ///
    /// This tensor must not be written to or read by from any other code.
    pub unsafe fn write_data(&mut self, src: &[u8]) {
        std::ptr::copy_nonoverlapping(src.as_ptr(), self.data() as *mut u8, src.len())
    }

    /// Zeroes out this tensor.
    pub fn zero_data(&mut self) {
        unsafe { std::ptr::write_bytes(self.data() as *mut u8, 0, self.nbytes()) }
    }

    /// Reads this tensor into `dst`, starting from `offset`. The size of `dst`
    /// will be used to determine how many bytes to read.
    ///
    /// # Safety
    ///
    /// This tensor must not be written to or read by from any other code.
    pub unsafe fn read_data(&self, offset: usize, dst: &mut [u8]) {
        let data = unsafe { sys::ggml_get_data(self.ptr.as_ptr()).add(offset) };
        std::ptr::copy_nonoverlapping(data, dst as *mut _ as _, dst.len())
    }

    /// Frees the memory of a tensor on an accelerator if ggml-sys is compiled with CUDA or CLBlast support.
    /// If not, this is a no-op.
    ///
    /// This is temporary while GGML improves their context memory management. This should only be called by
    /// `Context` when it is dropped.
    pub(crate) fn free_accelerator(self) {
        #[cfg(feature = "cublas")]
        unsafe {
            sys::cuda::ggml_cuda_free_data(self.ptr.as_ptr());
        }
        #[cfg(feature = "clblast")]
        unsafe {
            sys::opencl::ggml_cl_free_data(self.ptr.as_ptr());
        }
    }
}
impl Tensor {
    fn with_alive_ctx<U>(&self, mut f: impl FnMut() -> U) -> U {
        let _ctx = self
            .inner
            .upgrade()
            .expect("Using a tensor after the context was dropped");
        f()
    }

    fn with_alive_ctx_mut<U>(&mut self, mut f: impl FnMut(&mut Tensor) -> U) -> U {
        let _ctx = self
            .inner
            .upgrade()
            .expect("Using a tensor after the context was dropped");
        f(self)
    }

    /// Sets the acceleration backend of the tensor.
    ///
    /// # Caution
    ///
    /// This will not move the data to the new backend! See [Tensor::transfer_to] if you want to move the data to the new backend.
    fn set_backend(&mut self, backend: Backend) {
        unsafe {
            self.ptr.as_mut().backend = backend.try_into().unwrap();
        }
    }

    /// Adds this tensor to the context's list of offloaded tensors, so that it will be automatically freed.
    fn mark_as_offloaded(&self) {
        self.inner
            .upgrade()
            .expect("Attempted to update a dropped context's offloaded tensors")
            .offloaded_tensors
            .lock()
            .unwrap()
            .insert(self.name(), self.share());
    }
}
