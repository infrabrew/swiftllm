// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      tensor.rs
// PATH:      /crates/swiftllm-core/src/tensor.rs
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

//! Tensor abstractions for SwiftLLM
//!
//! This module provides a unified tensor interface that works across
//! different backends (CPU, CUDA, etc.)

use crate::config::DataType;
use crate::error::{Error, Result};
use bytemuck::{Pod, Zeroable};
use half::{bf16, f16};
use std::sync::Arc;

/// Tensor shape
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// Create a new shape
    pub fn new(dims: Vec<usize>) -> Self {
        Self(dims)
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Get the dimensions
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Get a specific dimension
    pub fn dim(&self, index: usize) -> usize {
        self.0[index]
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    /// Check if the shape is scalar
    pub fn is_scalar(&self) -> bool {
        self.0.is_empty()
    }

    /// Check if the shape is empty (has zero elements)
    pub fn is_empty(&self) -> bool {
        self.0.iter().any(|&d| d == 0)
    }

    /// Reshape to a new shape (must have same number of elements)
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Shape> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(Error::ShapeMismatch {
                expected: new_shape,
                actual: self.0.clone(),
            });
        }
        Ok(Shape::new(new_shape))
    }

    /// Broadcast with another shape
    pub fn broadcast_with(&self, other: &Shape) -> Result<Shape> {
        let max_ndim = self.ndim().max(other.ndim());
        let mut result = Vec::with_capacity(max_ndim);

        for i in 0..max_ndim {
            let dim_self = if i < max_ndim - self.ndim() {
                1
            } else {
                self.0[i - (max_ndim - self.ndim())]
            };

            let dim_other = if i < max_ndim - other.ndim() {
                1
            } else {
                other.0[i - (max_ndim - other.ndim())]
            };

            if dim_self != dim_other && dim_self != 1 && dim_other != 1 {
                return Err(Error::ShapeMismatch {
                    expected: self.0.clone(),
                    actual: other.0.clone(),
                });
            }

            result.push(dim_self.max(dim_other));
        }

        Ok(Shape::new(result))
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self::new(dims.to_vec())
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "]")
    }
}

/// Device type for tensor storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU memory
    Cpu,
    /// CUDA GPU memory
    Cuda(usize),
}

impl Device {
    /// Check if this is a CPU device
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Check if this is a CUDA device
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }

    /// Get the CUDA device index (panics if CPU)
    ///
    /// DEPRECATED: Use `try_cuda_device_index()` instead to avoid panics.
    pub fn cuda_device_index(&self) -> usize {
        match self {
            Device::Cuda(idx) => *idx,
            Device::Cpu => panic!("Not a CUDA device"),
        }
    }

    /// Get the CUDA device index, returning an error if this is not a CUDA device.
    pub fn try_cuda_device_index(&self) -> crate::error::Result<usize> {
        match self {
            Device::Cuda(idx) => Ok(*idx),
            Device::Cpu => Err(crate::error::Error::Device(
                "Expected CUDA device, got CPU".to_string(),
            )),
        }
    }
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(idx) => write!(f, "cuda:{}", idx),
        }
    }
}

/// Storage backend for tensor data
#[derive(Debug)]
pub enum Storage {
    /// CPU storage (owned data)
    Cpu(Vec<u8>),

    /// CUDA storage (device pointer and size)
    #[cfg(feature = "cuda")]
    Cuda {
        ptr: *mut u8,
        size: usize,
        device: usize,
    },
}

// Safety: Storage is Send/Sync because CUDA operations are synchronized
unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}

impl Storage {
    /// Create a new CPU storage with given size in bytes
    pub fn cpu(size: usize) -> Self {
        Storage::Cpu(vec![0u8; size])
    }

    /// Create a new CPU storage from data
    pub fn cpu_from_data(data: Vec<u8>) -> Self {
        Storage::Cpu(data)
    }

    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            Storage::Cpu(data) => data.len(),
            #[cfg(feature = "cuda")]
            Storage::Cuda { size, .. } => *size,
        }
    }

    /// Get the device
    pub fn device(&self) -> Device {
        match self {
            Storage::Cpu(_) => Device::Cpu,
            #[cfg(feature = "cuda")]
            Storage::Cuda { device, .. } => Device::Cuda(*device),
        }
    }

    /// Get a slice of the data (CPU only)
    pub fn as_slice(&self) -> Option<&[u8]> {
        match self {
            Storage::Cpu(data) => Some(data),
            #[cfg(feature = "cuda")]
            Storage::Cuda { .. } => None,
        }
    }

    /// Get a mutable slice of the data (CPU only)
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        match self {
            Storage::Cpu(data) => Some(data),
            #[cfg(feature = "cuda")]
            Storage::Cuda { .. } => None,
        }
    }
}

/// A multi-dimensional tensor
#[derive(Debug)]
pub struct Tensor {
    /// Shape of the tensor
    shape: Shape,

    /// Data type
    dtype: DataType,

    /// Storage backend
    storage: Arc<Storage>,

    /// Offset into storage (for views)
    offset: usize,

    /// Strides for each dimension
    strides: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor with zeros
    pub fn zeros(shape: impl Into<Shape>, dtype: DataType, device: Device) -> Result<Self> {
        let shape = shape.into();
        let size_bytes = shape.numel() * dtype.size_bytes();

        let storage = match device {
            Device::Cpu => Storage::cpu(size_bytes),
            #[cfg(feature = "cuda")]
            Device::Cuda(dev_idx) => {
                // Allocate zeroed device memory using the swiftllm-cuda layer.
                // The swiftllm-cuda crate is an optional dependency; when
                // available it delegates to cudaMalloc + cudaMemset(0).
                let ptr = swiftllm_cuda::malloc(size_bytes)
                    .map_err(|e| Error::Device(format!("CUDA malloc: {}", e)))?;
                swiftllm_cuda::memset_zero(ptr, size_bytes)
                    .map_err(|e| Error::Device(format!("CUDA memset: {}", e)))?;
                Storage::Cuda { ptr, size: size_bytes, device: dev_idx }
            }
            #[cfg(not(feature = "cuda"))]
            _ => return Err(Error::Device("CUDA not available (build without cuda feature)".to_string())),
        };

        let strides = compute_strides(&shape);

        Ok(Self {
            shape,
            dtype,
            storage: Arc::new(storage),
            offset: 0,
            strides,
        })
    }

    /// Create a new tensor from raw data
    pub fn from_data(
        data: Vec<u8>,
        shape: impl Into<Shape>,
        dtype: DataType,
    ) -> Result<Self> {
        let shape = shape.into();
        let expected_size = shape.numel() * dtype.size_bytes();

        if data.len() != expected_size {
            return Err(Error::Tensor(format!(
                "Data size mismatch: expected {} bytes, got {}",
                expected_size,
                data.len()
            )));
        }

        let strides = compute_strides(&shape);

        Ok(Self {
            shape,
            dtype,
            storage: Arc::new(Storage::cpu_from_data(data)),
            offset: 0,
            strides,
        })
    }

    /// Create a tensor from a slice of f32 values
    pub fn from_f32(data: &[f32], shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        if data.len() != shape.numel() {
            return Err(Error::Tensor(format!(
                "Data length {} doesn't match shape {:?}",
                data.len(),
                shape
            )));
        }

        let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
        Self::from_data(bytes, shape, DataType::Float32)
    }

    /// Create a tensor from a slice of f16 values
    pub fn from_f16(data: &[f16], shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        if data.len() != shape.numel() {
            return Err(Error::Tensor(format!(
                "Data length {} doesn't match shape {:?}",
                data.len(),
                shape
            )));
        }

        let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
        Self::from_data(bytes, shape, DataType::Float16)
    }

    /// Get the shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get the dimensions
    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Get the data type
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    /// Get the device
    pub fn device(&self) -> Device {
        self.storage.device()
    }

    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.size_bytes()
    }

    /// Check if the tensor is contiguous
    pub fn is_contiguous(&self) -> bool {
        let expected_strides = compute_strides(&self.shape);
        self.strides == expected_strides
    }

    /// Get the data as a slice (CPU only)
    pub fn as_bytes(&self) -> Option<&[u8]> {
        let slice = self.storage.as_slice()?;
        Some(&slice[self.offset..self.offset + self.size_bytes()])
    }

    /// Get the data as a typed slice
    pub fn as_slice<T: Pod>(&self) -> Option<&[T]> {
        let bytes = self.as_bytes()?;
        Some(bytemuck::cast_slice(bytes))
    }

    /// Reshape the tensor
    pub fn reshape(&self, new_shape: impl Into<Shape>) -> Result<Self> {
        let new_shape = new_shape.into();
        if new_shape.numel() != self.shape.numel() {
            return Err(Error::ShapeMismatch {
                expected: new_shape.dims().to_vec(),
                actual: self.shape.dims().to_vec(),
            });
        }

        Ok(Self {
            shape: new_shape.clone(),
            dtype: self.dtype,
            storage: Arc::clone(&self.storage),
            offset: self.offset,
            strides: compute_strides(&new_shape),
        })
    }

    /// Transpose the tensor (swap last two dimensions)
    pub fn transpose(&self) -> Result<Self> {
        if self.ndim() < 2 {
            return Err(Error::Tensor(
                "Cannot transpose tensor with less than 2 dimensions".to_string(),
            ));
        }

        let mut new_shape = self.shape.dims().to_vec();
        let n = new_shape.len();
        new_shape.swap(n - 1, n - 2);

        let mut new_strides = self.strides.clone();
        new_strides.swap(n - 1, n - 2);

        Ok(Self {
            shape: Shape::new(new_shape),
            dtype: self.dtype,
            storage: Arc::clone(&self.storage),
            offset: self.offset,
            strides: new_strides,
        })
    }

    /// Create a contiguous copy of the tensor
    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(Self {
                shape: self.shape.clone(),
                dtype: self.dtype,
                storage: Arc::clone(&self.storage),
                offset: self.offset,
                strides: self.strides.clone(),
            });
        }

        // For non-contiguous tensors, we need to copy the data
        // This is a simplified implementation
        Err(Error::not_implemented("Non-contiguous tensor copy"))
    }

    /// Move tensor to a device.
    ///
    /// When CUDA is enabled this performs real H2D or D2H transfers.
    pub fn to(&self, device: Device) -> Result<Self> {
        if self.device() == device {
            return Ok(Self {
                shape: self.shape.clone(),
                dtype: self.dtype,
                storage: Arc::clone(&self.storage),
                offset: self.offset,
                strides: self.strides.clone(),
            });
        }

        match (self.device(), device) {
            (Device::Cpu, Device::Cuda(dev_idx)) => {
                #[cfg(feature = "cuda")]
                {
                    let src_bytes = self.as_bytes()
                        .ok_or_else(|| Error::Tensor("Cannot read CPU bytes for H2D transfer".into()))?;
                    let size = src_bytes.len();
                    let dst_ptr = swiftllm_cuda::malloc(size)
                        .map_err(|e| Error::Device(format!("CUDA malloc (H2D): {}", e)))?;
                    swiftllm_cuda::copy_to_device(dst_ptr, src_bytes)
                        .map_err(|e| Error::Device(format!("cudaMemcpy H2D: {}", e)))?;
                    let storage = Storage::Cuda { ptr: dst_ptr, size, device: dev_idx };
                    let strides = compute_strides(&self.shape);
                    return Ok(Self {
                        shape:   self.shape.clone(),
                        dtype:   self.dtype,
                        storage: Arc::new(storage),
                        offset:  0,
                        strides,
                    });
                }
                #[cfg(not(feature = "cuda"))]
                Err(Error::not_implemented("CPU to CUDA transfer (build with cuda feature)"))
            }
            #[cfg(feature = "cuda")]
            (Device::Cuda(_), Device::Cpu) => {
                if let Storage::Cuda { ptr, size, .. } = self.storage.as_ref() {
                    let mut host_bytes = vec![0u8; *size];
                    swiftllm_cuda::copy_to_host(&mut host_bytes, *ptr as *const u8)
                        .map_err(|e| Error::Device(format!("cudaMemcpy D2H: {}", e)))?;
                    let strides = compute_strides(&self.shape);
                    return Ok(Self {
                        shape:   self.shape.clone(),
                        dtype:   self.dtype,
                        storage: Arc::new(Storage::Cpu(host_bytes)),
                        offset:  0,
                        strides,
                    });
                }
                Err(Error::Tensor("Expected CUDA storage for D2H transfer".into()))
            }
            _ => Err(Error::Device("Unsupported device transfer".to_string())),
        }
    }

    /// Return the raw device pointer for CUDA tensors, or `None` for CPU tensors.
    ///
    /// The returned pointer is valid only as long as this `Tensor` is alive.
    /// Used to pass device pointers directly to kernel launch wrappers.
    pub fn cuda_data_ptr(&self) -> Option<*mut u8> {
        #[cfg(feature = "cuda")]
        {
            if let Storage::Cuda { ptr, .. } = self.storage.as_ref() {
                return Some(*ptr);
            }
        }
        None
    }

    /// Cast tensor to a different dtype
    pub fn to_dtype(&self, dtype: DataType) -> Result<Self> {
        if self.dtype == dtype {
            return Ok(Self {
                shape: self.shape.clone(),
                dtype: self.dtype,
                storage: Arc::clone(&self.storage),
                offset: self.offset,
                strides: self.strides.clone(),
            });
        }

        // Type casting implementation would go here
        Err(Error::not_implemented("Dtype casting"))
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            dtype: self.dtype,
            storage: Arc::clone(&self.storage),
            offset: self.offset,
            strides: self.strides.clone(),
        }
    }
}

/// Compute strides for a contiguous tensor
fn compute_strides(shape: &Shape) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.ndim());
    let mut stride = 1;

    for &dim in shape.dims().iter().rev() {
        strides.push(stride);
        stride *= dim;
    }

    strides.reverse();
    strides
}

/// Tensor operations trait
pub trait TensorOps {
    /// Matrix multiplication
    fn matmul(&self, other: &Tensor) -> Result<Tensor>;

    /// Element-wise addition
    fn add(&self, other: &Tensor) -> Result<Tensor>;

    /// Element-wise multiplication
    fn mul(&self, other: &Tensor) -> Result<Tensor>;

    /// Softmax
    fn softmax(&self, dim: i32) -> Result<Tensor>;

    /// Layer normalization
    fn layer_norm(&self, weight: &Tensor, bias: Option<&Tensor>, eps: f32) -> Result<Tensor>;

    /// RMS normalization
    fn rms_norm(&self, weight: &Tensor, eps: f32) -> Result<Tensor>;

    /// SiLU activation
    fn silu(&self) -> Result<Tensor>;

    /// GELU activation
    fn gelu(&self) -> Result<Tensor>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.numel(), 24);
        assert_eq!(shape.dim(0), 2);
        assert_eq!(shape.dim(1), 3);
        assert_eq!(shape.dim(2), 4);
    }

    #[test]
    fn test_shape_broadcast() {
        let a = Shape::new(vec![3, 1, 5]);
        let b = Shape::new(vec![1, 4, 5]);
        let c = a.broadcast_with(&b).unwrap();
        assert_eq!(c.dims(), &[3, 4, 5]);
    }

    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::zeros(vec![2, 3], DataType::Float32, Device::Cpu).unwrap();
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.dtype(), DataType::Float32);
        assert_eq!(tensor.device(), Device::Cpu);
    }

    #[test]
    fn test_tensor_from_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_f32(&data, vec![2, 3]).unwrap();

        assert_eq!(tensor.dims(), &[2, 3]);
        let slice = tensor.as_slice::<f32>().unwrap();
        assert_eq!(slice, &data);
    }

    #[test]
    fn test_tensor_reshape() {
        let data = vec![1.0f32; 24];
        let tensor = Tensor::from_f32(&data, vec![2, 3, 4]).unwrap();
        let reshaped = tensor.reshape(vec![4, 6]).unwrap();

        assert_eq!(reshaped.dims(), &[4, 6]);
        assert_eq!(reshaped.numel(), 24);
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: tensor.rs
// REPO PATH:   /swiftllm/crates/swiftllm-core/src/tensor.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
