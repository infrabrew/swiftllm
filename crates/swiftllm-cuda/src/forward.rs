// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      forward.rs
// PATH:      /crates/swiftllm-cuda/src/forward.rs
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

//! A real model forward pass on the GPU.
//!
//! This composes real GPU kernels — the `linear_f16` GEMM, the `geglu_f16`
//! activation, and the `rmsnorm_f16` normalization — with `DeviceBuffer` memory
//! into an actual Gemma-style gated FFN block, `down(geglu(gate(x), up(x)))`,
//! plus RMS normalization. The projections, the GeGLU activation, **and** the
//! RMSNorm all run on the device (no host round-trip). The shapes and math match
//! `swiftllm_models::architectures::gemma`.
//!
//! [`ffn_reference`] is a pure-CPU computation of the same block; the GPU path
//! is verified against it (within f16 tolerance) in the crate's GPU integration
//! tests, which is what makes this an *exercised* GPU forward pass rather than a
//! compile-only one.

use half::f16;

/// Tanh-approximation GELU (identical to the Gemma runner's activation).
pub fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_56;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh())
}

/// GeGLU activation: `gelu(gate) * up`, element-wise.
pub fn geglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| gelu_tanh(g) * u)
        .collect()
}

/// Weights for a Gemma-style gated FFN block (row-major, f16).
///
/// `gate_w` and `up_w` are `[intermediate_size, hidden_size]`; `down_w` is
/// `[hidden_size, intermediate_size]`. Each projection is `y = x · Wᵀ`, matching
/// the `linear_f16` kernel.
pub struct FfnWeights {
    /// Model hidden size `H`.
    pub hidden_size: usize,
    /// FFN intermediate size `I`.
    pub intermediate_size: usize,
    /// Gate projection weights `[I, H]`.
    pub gate_w: Vec<f16>,
    /// Up projection weights `[I, H]`.
    pub up_w: Vec<f16>,
    /// Down projection weights `[H, I]`.
    pub down_w: Vec<f16>,
}

/// Row-major `y = x · Wᵀ`: `x` is `[m, k]`, `w` is `[n, k]`, `y` is `[m, n]`.
/// Pure-f32 reference matmul.
fn matmul_wt(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += x[row * k + kk] * w[col * k + kk];
            }
            y[row * n + col] = acc;
        }
    }
    y
}

/// Pure-CPU reference for the gated FFN forward (used to verify the GPU path).
/// `x` is `[m, hidden_size]`; returns `[m, hidden_size]`.
pub fn ffn_reference(x: &[f32], w: &FfnWeights, m: usize) -> Vec<f32> {
    let h = w.hidden_size;
    let i = w.intermediate_size;
    let gate_w: Vec<f32> = w.gate_w.iter().map(|v| v.to_f32()).collect();
    let up_w: Vec<f32> = w.up_w.iter().map(|v| v.to_f32()).collect();
    let down_w: Vec<f32> = w.down_w.iter().map(|v| v.to_f32()).collect();

    let gate = matmul_wt(x, &gate_w, m, i, h);
    let up = matmul_wt(x, &up_w, m, i, h);
    let hidden = geglu(&gate, &up);
    matmul_wt(&hidden, &down_w, m, h, i)
}

/// Run the gated FFN forward on the GPU.
///
/// `x` is the `[m, hidden_size]` input in f16. The gate, up, and down
/// projections run on the device via `linear_f16`, and the GeGLU activation runs
/// on the device via `geglu_f16` — the whole block executes on the GPU. Returns
/// the `[m, hidden_size]` output in f16. Verify against [`ffn_reference`].
///
/// Available only when compiled with CUDA support.
#[cfg(has_cuda)]
pub fn ffn_forward_gpu(x: &[f16], w: &FfnWeights, m: usize) -> crate::Result<Vec<f16>> {
    use crate::memory::DeviceBuffer;

    let h = w.hidden_size;
    let i = w.intermediate_size;

    // Upload input and weights to the device.
    let x_buf = DeviceBuffer::<f16>::from_slice(x, 0)?;
    let gate_w = DeviceBuffer::<f16>::from_slice(&w.gate_w, 0)?;
    let up_w = DeviceBuffer::<f16>::from_slice(&w.up_w, 0)?;
    let down_w = DeviceBuffer::<f16>::from_slice(&w.down_w, 0)?;

    // gate = x · gate_wᵀ  and  up = x · up_wᵀ   (on the GPU).
    let mut gate_buf = DeviceBuffer::<f16>::zeros(m * i, 0)?;
    let mut up_buf = DeviceBuffer::<f16>::zeros(m * i, 0)?;
    unsafe {
        crate::bindings::linear_f16(x_buf.as_ptr(), gate_w.as_ptr(), None, gate_buf.as_mut_ptr(), m, i, h)?;
        crate::bindings::linear_f16(x_buf.as_ptr(), up_w.as_ptr(), None, up_buf.as_mut_ptr(), m, i, h)?;
    }
    crate::synchronize()?;

    // GeGLU activation on the GPU (geglu_f16 kernel — no host round-trip).
    let mut hidden_buf = DeviceBuffer::<f16>::zeros(m * i, 0)?;
    unsafe {
        crate::bindings::geglu_f16(gate_buf.as_ptr(), up_buf.as_ptr(), hidden_buf.as_mut_ptr(), m * i)?;
    }
    crate::synchronize()?;

    // out = hidden · down_wᵀ   (on the GPU).
    let mut out_buf = DeviceBuffer::<f16>::zeros(m * h, 0)?;
    unsafe {
        crate::bindings::linear_f16(hidden_buf.as_ptr(), down_w.as_ptr(), None, out_buf.as_mut_ptr(), m, h, i)?;
    }
    crate::synchronize()?;

    out_buf.to_vec()
}

/// Pure-CPU RMSNorm reference (verifies the GPU kernel).
/// `offset = 0.0` is standard RMSNorm; `offset = 1.0` is Gemma's unit-offset.
pub fn rmsnorm_reference(
    x: &[f32],
    weight: &[f32],
    rows: usize,
    dim: usize,
    eps: f32,
    weight_offset: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for i in 0..dim {
            out[r * dim + i] = row[i] * inv * (weight[i] + weight_offset);
        }
    }
    out
}

/// RMS normalization on the GPU via the `rmsnorm_f16` kernel. `x`/output are
/// `[rows, dim]`; verify against [`rmsnorm_reference`]. CUDA-only.
#[cfg(has_cuda)]
pub fn rmsnorm_forward_gpu(
    x: &[f16],
    weight: &[f16],
    rows: usize,
    dim: usize,
    eps: f32,
    weight_offset: f32,
) -> crate::Result<Vec<f16>> {
    use crate::memory::DeviceBuffer;
    let x_buf = DeviceBuffer::<f16>::from_slice(x, 0)?;
    let w_buf = DeviceBuffer::<f16>::from_slice(weight, 0)?;
    let mut out_buf = DeviceBuffer::<f16>::zeros(rows * dim, 0)?;
    unsafe {
        crate::bindings::rmsnorm_f16(
            x_buf.as_ptr(),
            w_buf.as_ptr(),
            out_buf.as_mut_ptr(),
            rows,
            dim,
            eps,
            weight_offset,
        )?;
    }
    crate::synchronize()?;
    out_buf.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_and_geglu_match_gemma() {
        assert!(gelu_tanh(0.0).abs() < 1e-6);
        assert!(gelu_tanh(10.0) > 9.9);
        let out = geglu(&[0.0, 4.0], &[3.0, 2.0]);
        assert!(out[0].abs() < 1e-6); // gelu(0)*3 = 0
        assert!((out[1] - 2.0 * gelu_tanh(4.0)).abs() < 1e-4);
    }

    #[test]
    fn ffn_reference_identity_weights() {
        // hidden = intermediate = 2; all projections identity.
        let identity = vec![
            f16::from_f32(1.0),
            f16::from_f32(0.0),
            f16::from_f32(0.0),
            f16::from_f32(1.0),
        ];
        let w = FfnWeights {
            hidden_size: 2,
            intermediate_size: 2,
            gate_w: identity.clone(),
            up_w: identity.clone(),
            down_w: identity,
        };
        // x = [1, 2]; gate = up = [1, 2]; h = geglu([1,2],[1,2]); out = h (identity down).
        let out = ffn_reference(&[1.0, 2.0], &w, 1);
        let expected = geglu(&[1.0, 2.0], &[1.0, 2.0]);
        assert_eq!(out.len(), 2);
        assert!((out[0] - expected[0]).abs() < 1e-4);
        assert!((out[1] - expected[1]).abs() < 1e-4);
    }

    #[test]
    fn ffn_reference_shapes() {
        let w = FfnWeights {
            hidden_size: 4,
            intermediate_size: 8,
            gate_w: vec![f16::from_f32(0.1); 8 * 4],
            up_w: vec![f16::from_f32(0.1); 8 * 4],
            down_w: vec![f16::from_f32(0.1); 4 * 8],
        };
        let x = vec![0.5f32; 2 * 4]; // m = 2
        let out = ffn_reference(&x, &w, 2);
        assert_eq!(out.len(), 2 * 4); // [m, hidden]
    }

    #[test]
    fn rmsnorm_reference_basic() {
        // x = [3, 4]; rms = sqrt((9+16)/2); weight [1,1], offset 0 → out = x / rms.
        let out = rmsnorm_reference(&[3.0, 4.0], &[1.0, 1.0], 1, 2, 0.0, 0.0);
        let rms = ((9.0 + 16.0) / 2.0_f32).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-5);
        assert!((out[1] - 4.0 / rms).abs() < 1e-5);
        // Gemma unit-offset: weight 0 + offset 1 behaves like weight 1.
        let gemma = rmsnorm_reference(&[3.0, 4.0], &[0.0, 0.0], 1, 2, 0.0, 1.0);
        assert!((gemma[0] - out[0]).abs() < 1e-5);
        assert!((gemma[1] - out[1]).abs() < 1e-5);
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: forward.rs
// REPO PATH:   /swiftllm/crates/swiftllm-cuda/src/forward.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
