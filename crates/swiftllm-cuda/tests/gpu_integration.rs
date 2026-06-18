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

    // =======================================================================
    // 5. End-to-end model forward pass on the GPU
    //    A real Gemma-style gated FFN block — gate/up/down projections run on
    //    the device via linear_f16 — verified against a pure-CPU reference.
    // =======================================================================

    #[test]
    fn test_gemma_ffn_forward_on_gpu() {
        require_gpu!();
        use swiftllm_cuda::forward::{ffn_forward_gpu, ffn_reference, FfnWeights};

        let hidden = 8usize;
        let inter = 16usize;
        let m = 2usize;

        // Deterministic small f16 weights in roughly [-0.25, 0.25].
        let mk = |count: usize, seed: u32| -> Vec<half::f16> {
            (0..count)
                .map(|j| {
                    let h = (j as u32)
                        .wrapping_mul(2_654_435_761)
                        .wrapping_add(seed.wrapping_mul(40_503));
                    let v = ((h % 1000) as f32 / 1000.0 - 0.5) * 0.5;
                    half::f16::from_f32(v)
                })
                .collect()
        };
        let weights = FfnWeights {
            hidden_size: hidden,
            intermediate_size: inter,
            gate_w: mk(inter * hidden, 1),
            up_w: mk(inter * hidden, 2),
            down_w: mk(hidden * inter, 3),
        };

        // Input [m, hidden].
        let x_f16: Vec<half::f16> = (0..m * hidden)
            .map(|j| half::f16::from_f32((j as f32) / 16.0 - 0.25))
            .collect();
        // Reference uses the SAME f16-rounded input so only kernel accumulation differs.
        let x_ref: Vec<f32> = x_f16.iter().map(|v| v.to_f32()).collect();

        let gpu = ffn_forward_gpu(&x_f16, &weights, m).expect("GPU FFN forward");
        let cpu = ffn_reference(&x_ref, &weights, m);

        assert_eq!(gpu.len(), cpu.len());
        assert_eq!(gpu.len(), m * hidden);
        for j in 0..gpu.len() {
            let g = gpu[j].to_f32();
            assert!(
                (g - cpu[j]).abs() < 0.05,
                "FFN output mismatch at {}: gpu={} cpu={}",
                j,
                g,
                cpu[j]
            );
        }
    }

    #[test]
    fn test_rmsnorm_on_gpu() {
        require_gpu!();
        use swiftllm_cuda::forward::{rmsnorm_forward_gpu, rmsnorm_reference};

        let rows = 3usize;
        let dim = 16usize;
        let x_f16: Vec<half::f16> = (0..rows * dim)
            .map(|i| half::f16::from_f32((i as f32 % 7.0) - 3.0))
            .collect();
        let w_f16: Vec<half::f16> = (0..dim)
            .map(|i| half::f16::from_f32(0.5 + (i as f32) / 32.0))
            .collect();
        let x_ref: Vec<f32> = x_f16.iter().map(|v| v.to_f32()).collect();
        let w_ref: Vec<f32> = w_f16.iter().map(|v| v.to_f32()).collect();

        // Gemma unit-offset RMSNorm (offset = 1.0), eps = 1e-6.
        let gpu = rmsnorm_forward_gpu(&x_f16, &w_f16, rows, dim, 1e-6, 1.0).expect("GPU rmsnorm");
        let cpu = rmsnorm_reference(&x_ref, &w_ref, rows, dim, 1e-6, 1.0);

        assert_eq!(gpu.len(), cpu.len());
        for i in 0..gpu.len() {
            let g = gpu[i].to_f32();
            assert!(
                (g - cpu[i]).abs() < 0.02,
                "RMSNorm mismatch at {}: gpu={} cpu={}",
                i,
                g,
                cpu[i]
            );
        }
    }

    // Deterministic small f16 weights in ~[-0.2, 0.2].
    fn mk_f16(n: usize, seed: u32) -> Vec<half::f16> {
        (0..n)
            .map(|i| {
                let hsh = (i as u32)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(seed.wrapping_mul(40_503));
                half::f16::from_f32(((hsh % 1000) as f32 / 1000.0 - 0.5) * 0.4)
            })
            .collect()
    }

    fn make_layer(seed: u32) -> swiftllm_cuda::forward::LayerWeights {
        use swiftllm_cuda::forward::LayerWeights;
        let (h, nh, d, inter) = (8usize, 2usize, 4usize, 16usize); // attn_dim = nh*d = 8 = h
        let zeros = |n: usize| vec![half::f16::from_f32(0.0); n]; // norm weight 0 (+unit offset = identity)
        LayerWeights {
            hidden_size: h,
            num_heads: nh,
            head_dim: d,
            intermediate_size: inter,
            input_norm_w: zeros(h),
            q_w: mk_f16(nh * d * h, seed + 1),
            k_w: mk_f16(nh * d * h, seed + 2),
            v_w: mk_f16(nh * d * h, seed + 3),
            o_w: mk_f16(h * nh * d, seed + 4),
            post_norm_w: zeros(h),
            gate_w: mk_f16(inter * h, seed + 5),
            up_w: mk_f16(inter * h, seed + 6),
            down_w: mk_f16(h * inter, seed + 7),
            rms_eps: 1e-6,
            norm_unit_offset: 1.0,
        }
    }

    #[test]
    fn test_attention_on_gpu() {
        require_gpu!();
        use swiftllm_cuda::forward::{attention_forward_gpu, attention_reference};
        let (s, nh, d) = (4usize, 2usize, 4usize);
        let hd = nh * d;
        let scale = 1.0f32 / (d as f32).sqrt();
        let q = mk_f16(s * hd, 1);
        let k = mk_f16(s * hd, 2);
        let v = mk_f16(s * hd, 3);
        let f = |x: &[half::f16]| -> Vec<f32> { x.iter().map(|v| v.to_f32()).collect() };

        let gpu = attention_forward_gpu(&q, &k, &v, s, nh, d, scale, true).expect("GPU attention");
        let cpu = attention_reference(&f(&q), &f(&k), &f(&v), s, nh, d, scale, true);
        assert_eq!(gpu.len(), cpu.len());
        for i in 0..gpu.len() {
            assert!(
                (gpu[i].to_f32() - cpu[i]).abs() < 0.02,
                "attention mismatch at {}: gpu={} cpu={}",
                i,
                gpu[i].to_f32(),
                cpu[i]
            );
        }
    }

    #[test]
    fn test_transformer_layer_on_gpu() {
        require_gpu!();
        use swiftllm_cuda::forward::{transformer_layer_forward_gpu, transformer_layer_reference};
        let (h, s) = (8usize, 3usize);
        let w = make_layer(10);
        let x_f16: Vec<half::f16> = (0..s * h)
            .map(|i| half::f16::from_f32((i as f32) / 8.0 - 0.5))
            .collect();
        let x_ref: Vec<f32> = x_f16.iter().map(|v| v.to_f32()).collect();

        let gpu = transformer_layer_forward_gpu(&x_f16, &w, s).expect("GPU layer");
        let cpu = transformer_layer_reference(&x_ref, &w, s);
        assert_eq!(gpu.len(), cpu.len());
        let mut max_err = 0.0f32;
        for i in 0..gpu.len() {
            max_err = max_err.max((gpu[i].to_f32() - cpu[i]).abs());
        }
        assert!(max_err < 0.2, "transformer layer max f16 error {} too high", max_err);
    }

    #[test]
    fn test_transformer_stack_on_gpu() {
        require_gpu!();
        use swiftllm_cuda::forward::{
            transformer_layer_reference, transformer_stack_forward_gpu,
        };
        let (h, s) = (8usize, 3usize);
        let layers = vec![make_layer(100), make_layer(200)]; // 2-layer stack
        let x_f16: Vec<half::f16> = (0..s * h)
            .map(|i| half::f16::from_f32((i as f32) / 10.0 - 0.4))
            .collect();
        let x_ref: Vec<f32> = x_f16.iter().map(|v| v.to_f32()).collect();

        let gpu = transformer_stack_forward_gpu(&x_f16, &layers, s).expect("GPU stack");
        // CPU reference: apply each layer in sequence.
        let mut cpu = x_ref;
        for layer in &layers {
            cpu = transformer_layer_reference(&cpu, layer, s);
        }
        assert_eq!(gpu.len(), cpu.len());
        let mut max_err = 0.0f32;
        for i in 0..gpu.len() {
            max_err = max_err.max((gpu[i].to_f32() - cpu[i]).abs());
        }
        assert!(max_err < 0.4, "2-layer stack max f16 error {} too high", max_err);
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: gpu_integration.rs
// REPO PATH:   /swiftllm/crates/swiftllm-cuda/tests/gpu_integration.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
