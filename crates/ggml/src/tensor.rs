use std::{os::raw::c_void, ptr::NonNull, sync::Weak};

use crate::{i64_to_usize, sys, Type};

/// Tensors are owned by the context. A tensor is alive as long as the
/// underlying context it was created with is alive.
pub struct Tensor {
    pub(crate) ptr: NonNull<sys::ggml_tensor>,
    pub(crate) ctx: Weak<NonNull<sys::ggml_context>>,
}

impl Tensor {
    /// Size of the `ggml_tensor` struct in bytes.
    ///
    /// Exposed for purposes of determining context size.
    pub const C_TYPE_SIZE: usize = std::mem::size_of::<sys::ggml_tensor>();

    ///Sets the name of the tensor
    pub fn set_name(&mut self, name: &str) -> &Tensor {
        assert!(name.len() <= 48, "Name is too long!");

        let bytes = name.as_bytes();
        let mut array = [0i8; 48];
        array[..bytes.len()].copy_from_slice(&bytes.iter().map(|&x| x as i8).collect::<Vec<_>>());

        unsafe { self.ptr.as_mut().name = array }
        self
    }

    ///Gets the name of the tensor
    pub fn get_name(&self) -> String {
        let name = unsafe { self.ptr.as_ref().name };
        let mut name = name.iter().map(|&x| x as u8).collect::<Vec<_>>();
        name.retain(|&x| x != 0);
        String::from_utf8(name).unwrap()
    }

    ///Sets the acceleration backend of the tensor
    pub fn set_backend(&mut self, backend: crate::Backend) {
        unsafe { crate::set_tensor_backend(self.ptr.as_mut(), backend) }
    }

    ///Gets the acceleration backend of the tensor
    pub fn get_backend(&self) -> crate::Backend {
        unsafe { crate::get_tensor_backend(self.ptr.as_ref()) }
    }

    /// Creates a shared copy of this tensor pointer.
    pub fn share(&self) -> Self {
        Tensor {
            ptr: self.ptr,
            ctx: Weak::clone(&self.ctx),
        }
    }

    fn with_alive_ctx<U>(&self, mut f: impl FnMut() -> U) -> U {
        if let Some(_ctx) = self.ctx.upgrade() {
            f()
        } else {
            panic!("Using a tensor after the context was dropped")
        }
    }

    fn with_alive_ctx_mut<U>(&self, mut f: impl FnMut() -> U) -> U {
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
        let tensor = self.ptr.as_mut();
        self.with_alive_ctx_mut(|| {
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
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if self.get_backend() != crate::Backend::Cpu {
            // if the tensor is not on the cpu, free it from the accelerator
            crate::accelerator_free_tensor(self)
        }
    }
}
