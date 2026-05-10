// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      memory.rs
// PATH:      /crates/swiftllm-cuda/src/memory.rs
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// Safe RAII wrapper for device memory allocations.
//
// `DeviceBuffer<T>` owns a GPU allocation and frees it on drop.
// It exposes:
//   - `as_ptr()` / `as_mut_ptr()` — raw device pointers for kernel calls
//   - `copy_from_host()`           — H2D transfer
//   - `copy_to_host()`             — D2H transfer into a pre-allocated Vec
//   - `zeros()`                    — constructor that memsets to zero
//
// `T` must be `bytemuck::Pod` (plain-old-data, safe to copy byte-for-byte).
// Licensed under the Apache License, Version 2.0
// ==============================================================================

use super::{CudaError, Result};

/// Owned GPU buffer of `len` elements of type `T`.
///
/// Freed automatically when dropped.
pub struct DeviceBuffer<T: bytemuck::Pod> {
    ptr:    *mut u8,
    len:    usize,       // element count
    device: usize,
    _phantom: std::marker::PhantomData<T>,
}

// SAFETY: GPU memory is not aliased; we enforce exclusive mutability via &mut.
unsafe impl<T: bytemuck::Pod> Send for DeviceBuffer<T> {}
unsafe impl<T: bytemuck::Pod> Sync for DeviceBuffer<T> {}

impl<T: bytemuck::Pod> DeviceBuffer<T> {
    /// Allocate `len` elements of `T` on the given CUDA device (uninitialized).
    pub fn new(len: usize, device: usize) -> Result<Self> {
        let _ = device; // cudarc device selection handled globally for now
        let size_bytes = len * std::mem::size_of::<T>();
        let ptr = super::malloc(size_bytes)?;
        Ok(Self {
            ptr,
            len,
            device,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Allocate `len` elements of `T` and zero-initialise them.
    pub fn zeros(len: usize, device: usize) -> Result<Self> {
        let buf = Self::new(len, device)?;
        super::memset_zero(buf.ptr, len * std::mem::size_of::<T>())?;
        Ok(buf)
    }

    /// Create a device buffer and copy `data` into it.
    pub fn from_slice(data: &[T], device: usize) -> Result<Self> {
        let buf = Self::new(data.len(), device)?;
        let bytes = bytemuck::cast_slice(data);
        super::copy_to_device(buf.ptr, bytes)?;
        Ok(buf)
    }

    /// Element count.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if no elements are stored.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw const device pointer (for kernel arguments).
    pub fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    /// Raw mutable device pointer (for kernel output arguments).
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr as *mut T
    }

    /// Copy host slice into this buffer (must be same length).
    pub fn copy_from_host(&mut self, src: &[T]) -> Result<()> {
        if src.len() != self.len {
            return Err(CudaError::InvalidParameter(format!(
                "copy_from_host: length mismatch src={} dst={}", src.len(), self.len
            )));
        }
        let bytes = bytemuck::cast_slice(src);
        super::copy_to_device(self.ptr, bytes)
    }

    /// Copy device buffer into `dst` (must be pre-allocated to `self.len`).
    pub fn copy_to_host(&self, dst: &mut Vec<T>) -> Result<()> {
        dst.resize(self.len, T::zeroed());
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(dst.as_mut_slice());
        super::copy_to_host(bytes, self.ptr as *const u8)
    }

    /// Copy device buffer into a newly allocated `Vec<T>`.
    pub fn to_vec(&self) -> Result<Vec<T>> {
        let mut dst = Vec::with_capacity(self.len);
        self.copy_to_host(&mut dst)?;
        Ok(dst)
    }

    /// Size of the allocation in bytes.
    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// CUDA device index this buffer lives on.
    pub fn device(&self) -> usize {
        self.device
    }
}

impl<T: bytemuck::Pod> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // Ignore errors on drop; log for debugging.
            if let Err(e) = super::free(self.ptr) {
                tracing::warn!("DeviceBuffer<{}> drop: free failed: {}", std::any::type_name::<T>(), e);
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}

impl<T: bytemuck::Pod + std::fmt::Debug> std::fmt::Debug for DeviceBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeviceBuffer<{}>[len={}, device={}]",
            std::any::type_name::<T>(), self.len, self.device)
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: memory.rs
// REPO PATH:   /swiftllm/crates/swiftllm-cuda/src/memory.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
