// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      lib.rs
// PATH:      /crates/swiftllm-cuda/src/lib.rs
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

//! SwiftLLM CUDA Kernels
//!
//! High-performance CUDA kernels for LLM inference:
//! - PagedAttention
//! - Mamba-3 selective SSM scan (prefill + decode)
//! - LatentMoE dispatch (compress → gate → expert FFN → expand)
//! - Dense Verification cross-attention + confidence scoring
//! - RLM depth embedding, confidence MLP, sub-problem gating
//! - F16 linear (GEMM) fallback

#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)] // CUDA kernel signatures require many parameters

pub mod bindings;
pub mod forward;
pub mod memory;

use thiserror::Error;

/// CUDA error type
#[derive(Error, Debug)]
pub enum CudaError {
    /// CUDA initialization failed
    #[error("CUDA initialization failed: {0}")]
    InitError(String),

    /// Kernel launch failed
    #[error("Kernel launch failed: {0}")]
    KernelError(String),

    /// Memory allocation failed
    #[error("Memory allocation failed: {0}")]
    MemoryError(String),

    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Device not found
    #[error("CUDA device not found")]
    DeviceNotFound,

    /// Not supported
    #[error("Operation not supported: {0}")]
    NotSupported(String),
}

pub type Result<T> = std::result::Result<T, CudaError>;

/// CUDA device information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device index
    pub index: usize,

    /// Device name
    pub name: String,

    /// Total memory in bytes
    pub total_memory: usize,

    /// Compute capability (major, minor)
    pub compute_capability: (u32, u32),

    /// Number of multiprocessors
    pub multiprocessor_count: u32,

    /// Max threads per block
    pub max_threads_per_block: u32,

    /// Max shared memory per block
    pub max_shared_memory_per_block: usize,
}

/// Get the number of CUDA devices
pub fn device_count() -> Result<usize> {
    // In a real implementation, this would call cuDeviceGetCount
    #[cfg(has_cuda)]
    {
        // cudarc implementation
        Ok(1) // Placeholder
    }

    #[cfg(not(has_cuda))]
    {
        Err(CudaError::DeviceNotFound)
    }
}

/// Get device information
#[allow(unused_variables)]
pub fn get_device_info(device_id: usize) -> Result<DeviceInfo> {
    // In a real implementation, this would query CUDA device properties
    #[cfg(has_cuda)]
    {
        Ok(DeviceInfo {
            index: device_id,
            name: format!("NVIDIA GPU {}", device_id),
            total_memory: 16 * 1024 * 1024 * 1024, // 16 GB placeholder
            compute_capability: (8, 0),
            multiprocessor_count: 108,
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 163840,
        })
    }

    #[cfg(not(has_cuda))]
    {
        Err(CudaError::DeviceNotFound)
    }
}

/// Set the current CUDA device
#[allow(unused_variables)]
pub fn set_device(device_id: usize) -> Result<()> {
    #[cfg(has_cuda)]
    {
        // cuDevicePrimaryCtxRetain and cuCtxSetCurrent
        Ok(())
    }

    #[cfg(not(has_cuda))]
    {
        Err(CudaError::DeviceNotFound)
    }
}

/// Synchronize the current device
pub fn synchronize() -> Result<()> {
    #[cfg(has_cuda)]
    {
        // cuCtxSynchronize
        Ok(())
    }

    #[cfg(not(has_cuda))]
    {
        Ok(())
    }
}

/// Allocate `size` bytes on the current CUDA device.
///
/// Returns a raw device pointer.  Must be freed with [`free`].
pub fn malloc(size: usize) -> Result<*mut u8> {
    #[cfg(has_cuda)]
    {
        use std::ffi::c_void;
        extern "C" {
            fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
        }
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { cudaMalloc(&mut ptr as *mut *mut c_void, size) };
        if rc != 0 {
            return Err(CudaError::MemoryError(format!(
                "cudaMalloc({} bytes) failed with code {}", size, rc
            )));
        }
        Ok(ptr as *mut u8)
    }

    #[cfg(not(has_cuda))]
    {
        let _ = size;
        Err(CudaError::DeviceNotFound)
    }
}

/// Free device memory previously allocated with [`malloc`].
pub fn free(ptr: *mut u8) -> Result<()> {
    #[cfg(has_cuda)]
    {
        use std::ffi::c_void;
        extern "C" {
            fn cudaFree(devPtr: *mut c_void) -> i32;
        }
        let rc = unsafe { cudaFree(ptr as *mut c_void) };
        if rc != 0 {
            return Err(CudaError::MemoryError(format!("cudaFree failed with code {}", rc)));
        }
        Ok(())
    }

    #[cfg(not(has_cuda))]
    {
        let _ = ptr;
        Err(CudaError::DeviceNotFound)
    }
}

/// Copy `src` (host slice) to `dst` (device pointer).
pub fn copy_to_device(dst: *mut u8, src: &[u8]) -> Result<()> {
    #[cfg(has_cuda)]
    {
        use std::ffi::c_void;
        // cudaMemcpyKind::cudaMemcpyHostToDevice = 1
        extern "C" {
            fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
        }
        let rc = unsafe {
            cudaMemcpy(
                dst as *mut c_void,
                src.as_ptr() as *const c_void,
                src.len(),
                1, // H2D
            )
        };
        if rc != 0 {
            return Err(CudaError::MemoryError(format!("cudaMemcpy H2D failed with code {}", rc)));
        }
        Ok(())
    }

    #[cfg(not(has_cuda))]
    {
        let _ = (dst, src);
        Err(CudaError::DeviceNotFound)
    }
}

/// Copy `src` (device pointer) to `dst` (host slice).
pub fn copy_to_host(dst: &mut [u8], src: *const u8) -> Result<()> {
    #[cfg(has_cuda)]
    {
        use std::ffi::c_void;
        // cudaMemcpyKind::cudaMemcpyDeviceToHost = 2
        extern "C" {
            fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
        }
        let rc = unsafe {
            cudaMemcpy(
                dst.as_mut_ptr() as *mut c_void,
                src as *const c_void,
                dst.len(),
                2, // D2H
            )
        };
        if rc != 0 {
            return Err(CudaError::MemoryError(format!("cudaMemcpy D2H failed with code {}", rc)));
        }
        Ok(())
    }

    #[cfg(not(has_cuda))]
    {
        let _ = (dst, src);
        Err(CudaError::DeviceNotFound)
    }
}

/// Set `count` bytes starting at `ptr` to zero on device.
pub fn memset_zero(ptr: *mut u8, count: usize) -> Result<()> {
    #[cfg(has_cuda)]
    {
        use std::ffi::c_void;
        extern "C" {
            fn cudaMemset(devPtr: *mut c_void, value: i32, count: usize) -> i32;
        }
        let rc = unsafe { cudaMemset(ptr as *mut c_void, 0, count) };
        if rc != 0 {
            return Err(CudaError::MemoryError(format!("cudaMemset failed with code {}", rc)));
        }
        Ok(())
    }

    #[cfg(not(has_cuda))]
    {
        let _ = (ptr, count);
        Err(CudaError::DeviceNotFound)
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: lib.rs
// REPO PATH:   /swiftllm/crates/swiftllm-cuda/src/lib.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
