// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      gpu_integration.rs
// PATH:      /crates/swiftllm-cuda/tests/gpu_integration.rs
// AUTHOR:    Peter A. Aldrich Jr.
// DATE:      2026
// ------------------------------------------------------------------------------
// Integration tests for CUDA kernels on real GPU hardware.
//
// These tests require:
//   - A CUDA-capable GPU visible to the process
//   - The `has_cuda` cfg flag (set automatically by build.rs when CUDA is found)
//
// Run with:
//   CUDA_PATH=/usr/local/cuda cargo test -p swiftllm-cuda --test gpu_integration
//
// Licensed under the Apache License, Version 2.0
// ==============================================================================

/// Guard: skip all tests at the module level when compiled without CUDA.
/// Each test additionally checks runtime GPU availability.
#[cfg(not(has_cuda))]
mod tests {
    #[test]
    fn skipped_no_cuda() {
        eprintln!("GPU integration tests skipped: compiled without CUDA support.");
    }
}

#[cfg(has_cuda)]
mod tests {
    use swiftllm_cuda::memory::DeviceBuffer;

    // -----------------------------------------------------------------------
    // Helper: check if we can actually allocate GPU memory (driver present)
    // -----------------------------------------------------------------------
    fn gpu_available() -> bool {
        DeviceBuffer::<f32>::zeros(1, 0).is_ok()
    }

    macro_rules! require_gpu {
        () => {
            if !gpu_available() {
                eprintln!("  [SKIP] No GPU available at runtime");
                return;
            }
        };
    }

    // =======================================================================
    // 1. DeviceBuffer RAII lifecycle
    // =======================================================================

    #[test]
    fn test_device_buffer_alloc_and_drop() {
        require_gpu!();
        let buf = DeviceBuffer::<f32>::new(1024, 0).expect("alloc 1024 f32");
        assert_eq!(buf.len(), 1024);
        assert_eq!(buf.size_bytes(), 1024 * 4);
        assert!(!buf.is_empty());
        drop(buf); // should not panic or leak
    }

    #[test]
    fn test_device_buffer_zeros() {
        require_gpu!();
        let buf = DeviceBuffer::<f32>::zeros(512, 0).expect("zeros 512 f32");
        let host = buf.to_vec().expect("D2H copy");
        assert_eq!(host.len(), 512);
        for &v in &host {
            assert_eq!(v, 0.0f32, "zeros buffer should be all 0.0");
        }
    }

    #[test]
    fn test_device_buffer_roundtrip() {
        require_gpu!();
        let src: Vec<f32> = (0..256).map(|i| i as f32 * 0.5).collect();
        let buf = DeviceBuffer::<f32>::from_slice(&src, 0).expect("H2D");
        let dst = buf.to_vec().expect("D2H");
        assert_eq!(src, dst, "H2D -> D2H roundtrip must preserve data");
    }

    #[test]
    fn test_device_buffer_copy_from_host() {
        require_gpu!();
        let mut buf = DeviceBuffer::<u32>::zeros(64, 0).expect("zeros");
        let data: Vec<u32> = (100..164).collect();
        buf.copy_from_host(&data).expect("copy_from_host");
        let out = buf.to_vec().expect("D2H");
        assert_eq!(data, out);
    }

    #[test]
    fn test_device_buffer_length_mismatch() {
        require_gpu!();
        let mut buf = DeviceBuffer::<f32>::zeros(10, 0).expect("zeros");
        let bad_data = vec![1.0f32; 20];
        let result = buf.copy_from_host(&bad_data);
        assert!(result.is_err(), "copy_from_host with wrong length should fail");
    }

    #[test]
    fn test_device_buffer_empty() {
        require_gpu!();
        let buf = DeviceBuffer::<f32>::new(0, 0).expect("empty alloc");
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    // =======================================================================
    // 2. Raw malloc / free / memset
    // =======================================================================

    #[test]
    fn test_raw_malloc_free() {
        require_gpu!();
        let ptr = swiftllm_cuda::malloc(4096).expect("malloc 4096");
        assert!(!ptr.is_null());
        swiftllm_cuda::free(ptr).expect("free");
    }

    #[test]
    fn test_raw_memset_zero() {
        require_gpu!();
        let ptr = swiftllm_cuda::malloc(256).expect("malloc");
        swiftllm_cuda::memset_zero(ptr, 256).expect("memset");
        let mut host = vec![0xFFu8; 256];
        swiftllm_cuda::copy_to_host(&mut host, ptr).expect("D2H");
        for &b in &host {
            assert_eq!(b, 0u8, "memset_zero should zero all bytes");
        }
        swiftllm_cuda::free(ptr).expect("free");
    }

    // =======================================================================
    // 3. Linear F16 kernel
    // =======================================================================

    #[test]
    fn test_linear_f16_identity() {
        require_gpu!();

        // Test: y = x @ I (identity matrix), no bias
        // x: [2, 4], W: [4, 4] = I, y: [2, 4]
        let m = 2usize;
        let n = 4usize;
        let k = 4usize;

        // Input: [[1, 2, 3, 4], [5, 6, 7, 8]]
        let x_host: Vec<half::f16> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .iter()
            .map(|&v| half::f16::from_f32(v))
            .collect();

        // Weight: 4x4 identity (row-major, W^T applied inside kernel)
        let mut w_host = vec![half::f16::from_f32(0.0); n * k];
        for i in 0..n.min(k) {
            w_host[i * k + i] = half::f16::from_f32(1.0);
        }

        let x_buf = DeviceBuffer::<half::f16>::from_slice(&x_host, 0).expect("x H2D");
        let w_buf = DeviceBuffer::<half::f16>::from_slice(&w_host, 0).expect("w H2D");
        let mut y_buf = DeviceBuffer::<half::f16>::zeros(m * n, 0).expect("y alloc");

        unsafe {
            swiftllm_cuda::bindings::linear_f16(
                x_buf.as_ptr(),
                w_buf.as_ptr(),
                None,
                y_buf.as_mut_ptr(),
                m, n, k,
            )
            .expect("linear_f16 kernel");
        }

        swiftllm_cuda::synchronize().expect("sync");

        let y_host = y_buf.to_vec().expect("y D2H");
        let y_f32: Vec<f32> = y_host.iter().map(|v| v.to_f32()).collect();

        // With identity weight, output should equal input
        let x_f32: Vec<f32> = x_host.iter().map(|v| v.to_f32()).collect();
        for i in 0..y_f32.len() {
            assert!(
                (y_f32[i] - x_f32[i]).abs() < 0.01,
                "y[{}] = {} (expected {})", i, y_f32[i], x_f32[i]
            );
        }
    }

    #[test]
    fn test_linear_f16_scale() {
        require_gpu!();

        // Test: y = x @ W where W = 2*I (scale by 2)
        let m = 1usize;
        let n = 4usize;
        let k = 4usize;

        let x_host: Vec<half::f16> = [1.0, 2.0, 3.0, 4.0]
            .iter()
            .map(|&v| half::f16::from_f32(v))
            .collect();

        let mut w_host = vec![half::f16::from_f32(0.0); n * k];
        for i in 0..n.min(k) {
            w_host[i * k + i] = half::f16::from_f32(2.0);
        }

        let x_buf = DeviceBuffer::<half::f16>::from_slice(&x_host, 0).expect("x H2D");
        let w_buf = DeviceBuffer::<half::f16>::from_slice(&w_host, 0).expect("w H2D");
        let mut y_buf = DeviceBuffer::<half::f16>::zeros(m * n, 0).expect("y alloc");

        unsafe {
            swiftllm_cuda::bindings::linear_f16(
                x_buf.as_ptr(),
                w_buf.as_ptr(),
                None,
                y_buf.as_mut_ptr(),
                m, n, k,
            )
            .expect("linear_f16 kernel");
        }

        swiftllm_cuda::synchronize().expect("sync");

        let y_host = y_buf.to_vec().expect("y D2H");
        let y_f32: Vec<f32> = y_host.iter().map(|v| v.to_f32()).collect();

        let expected = [2.0, 4.0, 6.0, 8.0];
        for i in 0..y_f32.len() {
            assert!(
                (y_f32[i] - expected[i]).abs() < 0.1,
                "y[{}] = {} (expected {})", i, y_f32[i], expected[i]
            );
        }
    }

    // =======================================================================
    // 4. Half-precision DeviceBuffer roundtrip
    // =======================================================================

    #[test]
    fn test_half_precision_roundtrip() {
        require_gpu!();
        let src: Vec<half::f16> = (0..128)
            .map(|i| half::f16::from_f32(i as f32 * 0.125))
            .collect();
        let buf = DeviceBuffer::<half::f16>::from_slice(&src, 0).expect("H2D");
        let dst = buf.to_vec().expect("D2H");
        assert_eq!(src.len(), dst.len());
        for (i, (a, b)) in src.iter().zip(dst.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "mismatch at index {}", i);
        }
    }

    // =======================================================================
    // 5. Large allocation stress test
    // =======================================================================

    #[test]
    fn test_large_allocation() {
        require_gpu!();
        // Allocate 64 MB of f32 (16M elements)
        let n = 16 * 1024 * 1024;
        let buf = DeviceBuffer::<f32>::zeros(n, 0).expect("64MB allocation");
        assert_eq!(buf.len(), n);
        assert_eq!(buf.size_bytes(), n * 4);
        // Just verify it doesn't crash; don't D2H the whole thing
        drop(buf);
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: gpu_integration.rs
// REPO PATH:   /swiftllm/crates/swiftllm-cuda/tests/gpu_integration.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
