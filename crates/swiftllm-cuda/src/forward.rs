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

// ----------------------------------------------------------------------------
// Multi-head attention
// ----------------------------------------------------------------------------

/// Pure-CPU multi-head scaled-dot-product attention reference (verifies the GPU
/// kernel). `q`/`k`/`v` are `[seq_len, num_heads * head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn attention_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let hd = num_heads * head_dim;
    let mut out = vec![0.0f32; seq_len * hd];
    for i in 0..seq_len {
        for h in 0..num_heads {
            let qbase = i * hd + h * head_dim;
            let jmax = if causal { i + 1 } else { seq_len };
            let mut scores = vec![0.0f32; jmax];
            let mut maxv = f32::NEG_INFINITY;
            for (j, score) in scores.iter_mut().enumerate() {
                let kbase = j * hd + h * head_dim;
                let mut s = 0.0f32;
                for d in 0..head_dim {
                    s += q[qbase + d] * k[kbase + d];
                }
                s *= scale;
                *score = s;
                if s > maxv {
                    maxv = s;
                }
            }
            let mut denom = 0.0f32;
            for score in scores.iter_mut() {
                *score = (*score - maxv).exp();
                denom += *score;
            }
            let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (j, &sc) in scores.iter().enumerate() {
                    acc += sc * v[j * hd + h * head_dim + d];
                }
                out[i * hd + h * head_dim + d] = acc * inv;
            }
        }
    }
    out
}

/// Multi-head attention on the GPU via the `attention_f16` kernel. CUDA-only.
#[cfg(has_cuda)]
#[allow(clippy::too_many_arguments)]
pub fn attention_forward_gpu(
    q: &[f16],
    k: &[f16],
    v: &[f16],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) -> crate::Result<Vec<f16>> {
    use crate::memory::DeviceBuffer;
    let hd = num_heads * head_dim;
    let q_buf = DeviceBuffer::<f16>::from_slice(q, 0)?;
    let k_buf = DeviceBuffer::<f16>::from_slice(k, 0)?;
    let v_buf = DeviceBuffer::<f16>::from_slice(v, 0)?;
    let mut o_buf = DeviceBuffer::<f16>::zeros(seq_len * hd, 0)?;
    unsafe {
        crate::bindings::attention_f16(
            q_buf.as_ptr(), k_buf.as_ptr(), v_buf.as_ptr(), o_buf.as_mut_ptr(),
            seq_len, num_heads, head_dim, scale, causal,
        )?;
    }
    crate::synchronize()?;
    o_buf.to_vec()
}

// ----------------------------------------------------------------------------
// Full transformer decoder layer
// ----------------------------------------------------------------------------

/// Weights for one Gemma-style transformer decoder layer. Attention dimension is
/// `num_heads * head_dim`; projections are `y = x · Wᵀ`.
pub struct LayerWeights {
    /// Hidden size `H`.
    pub hidden_size: usize,
    /// Attention heads.
    pub num_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// FFN intermediate size.
    pub intermediate_size: usize,
    /// Pre-attention RMSNorm weight `[H]`.
    pub input_norm_w: Vec<f16>,
    /// Query projection `[num_heads*head_dim, H]`.
    pub q_w: Vec<f16>,
    /// Key projection `[num_heads*head_dim, H]`.
    pub k_w: Vec<f16>,
    /// Value projection `[num_heads*head_dim, H]`.
    pub v_w: Vec<f16>,
    /// Output projection `[H, num_heads*head_dim]`.
    pub o_w: Vec<f16>,
    /// Post-attention RMSNorm weight `[H]`.
    pub post_norm_w: Vec<f16>,
    /// FFN gate projection `[intermediate, H]`.
    pub gate_w: Vec<f16>,
    /// FFN up projection `[intermediate, H]`.
    pub up_w: Vec<f16>,
    /// FFN down projection `[H, intermediate]`.
    pub down_w: Vec<f16>,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
    /// RMSNorm weight offset (`1.0` for Gemma unit-offset).
    pub norm_unit_offset: f32,
}

/// Pure-CPU reference for one transformer decoder layer:
/// `h = x + attn(rmsnorm(x))`, `out = h + ffn(rmsnorm(h))`, causal attention.
pub fn transformer_layer_reference(x: &[f32], w: &LayerWeights, seq_len: usize) -> Vec<f32> {
    let h = w.hidden_size;
    let attn_dim = w.num_heads * w.head_dim;
    let inter = w.intermediate_size;
    let s = seq_len;
    let scale = 1.0 / (w.head_dim as f32).sqrt();
    let f = |v: &[f16]| -> Vec<f32> { v.iter().map(|x| x.to_f32()).collect() };

    let normed = rmsnorm_reference(x, &f(&w.input_norm_w), s, h, w.rms_eps, w.norm_unit_offset);
    let q = matmul_wt(&normed, &f(&w.q_w), s, attn_dim, h);
    let k = matmul_wt(&normed, &f(&w.k_w), s, attn_dim, h);
    let v = matmul_wt(&normed, &f(&w.v_w), s, attn_dim, h);
    let attn = attention_reference(&q, &k, &v, s, w.num_heads, w.head_dim, scale, true);
    let attn_out = matmul_wt(&attn, &f(&w.o_w), s, h, attn_dim);
    let hres: Vec<f32> = x.iter().zip(&attn_out).map(|(a, b)| a + b).collect();

    let normed2 = rmsnorm_reference(&hres, &f(&w.post_norm_w), s, h, w.rms_eps, w.norm_unit_offset);
    let gate = matmul_wt(&normed2, &f(&w.gate_w), s, inter, h);
    let up = matmul_wt(&normed2, &f(&w.up_w), s, inter, h);
    let ffn_hidden = geglu(&gate, &up);
    let ffn_out = matmul_wt(&ffn_hidden, &f(&w.down_w), s, h, inter);
    hres.iter().zip(&ffn_out).map(|(a, b)| a + b).collect()
}

/// Run one full transformer decoder layer on the GPU: RMSNorm → QKV → attention
/// → output projection → residual → RMSNorm → GeGLU FFN → residual. Every step
/// runs on the device. Verify against [`transformer_layer_reference`]. CUDA-only.
#[cfg(has_cuda)]
pub fn transformer_layer_forward_gpu(x: &[f16], w: &LayerWeights, seq_len: usize) -> crate::Result<Vec<f16>> {
    use crate::bindings::{add_f16, attention_f16, geglu_f16, linear_f16, rmsnorm_f16};
    use crate::memory::DeviceBuffer;

    let h = w.hidden_size;
    let attn_dim = w.num_heads * w.head_dim;
    let inter = w.intermediate_size;
    let s = seq_len;
    let scale = 1.0f32 / (w.head_dim as f32).sqrt();

    // Upload input + weights.
    let x_buf = DeviceBuffer::<f16>::from_slice(x, 0)?;
    let in_norm = DeviceBuffer::<f16>::from_slice(&w.input_norm_w, 0)?;
    let qw = DeviceBuffer::<f16>::from_slice(&w.q_w, 0)?;
    let kw = DeviceBuffer::<f16>::from_slice(&w.k_w, 0)?;
    let vw = DeviceBuffer::<f16>::from_slice(&w.v_w, 0)?;
    let ow = DeviceBuffer::<f16>::from_slice(&w.o_w, 0)?;
    let post_norm = DeviceBuffer::<f16>::from_slice(&w.post_norm_w, 0)?;
    let gatew = DeviceBuffer::<f16>::from_slice(&w.gate_w, 0)?;
    let upw = DeviceBuffer::<f16>::from_slice(&w.up_w, 0)?;
    let downw = DeviceBuffer::<f16>::from_slice(&w.down_w, 0)?;

    let mut normed = DeviceBuffer::<f16>::zeros(s * h, 0)?;
    let mut q = DeviceBuffer::<f16>::zeros(s * attn_dim, 0)?;
    let mut k = DeviceBuffer::<f16>::zeros(s * attn_dim, 0)?;
    let mut v = DeviceBuffer::<f16>::zeros(s * attn_dim, 0)?;
    let mut attn = DeviceBuffer::<f16>::zeros(s * attn_dim, 0)?;
    let mut attn_out = DeviceBuffer::<f16>::zeros(s * h, 0)?;
    let mut hres = DeviceBuffer::<f16>::zeros(s * h, 0)?;
    let mut normed2 = DeviceBuffer::<f16>::zeros(s * h, 0)?;
    let mut gate = DeviceBuffer::<f16>::zeros(s * inter, 0)?;
    let mut up = DeviceBuffer::<f16>::zeros(s * inter, 0)?;
    let mut ffn_hidden = DeviceBuffer::<f16>::zeros(s * inter, 0)?;
    let mut ffn_out = DeviceBuffer::<f16>::zeros(s * h, 0)?;
    let mut out = DeviceBuffer::<f16>::zeros(s * h, 0)?;

    unsafe {
        // Attention sub-block.
        rmsnorm_f16(x_buf.as_ptr(), in_norm.as_ptr(), normed.as_mut_ptr(), s, h, w.rms_eps, w.norm_unit_offset)?;
        linear_f16(normed.as_ptr(), qw.as_ptr(), None, q.as_mut_ptr(), s, attn_dim, h)?;
        linear_f16(normed.as_ptr(), kw.as_ptr(), None, k.as_mut_ptr(), s, attn_dim, h)?;
        linear_f16(normed.as_ptr(), vw.as_ptr(), None, v.as_mut_ptr(), s, attn_dim, h)?;
        attention_f16(q.as_ptr(), k.as_ptr(), v.as_ptr(), attn.as_mut_ptr(), s, w.num_heads, w.head_dim, scale, true)?;
        linear_f16(attn.as_ptr(), ow.as_ptr(), None, attn_out.as_mut_ptr(), s, h, attn_dim)?;
        add_f16(x_buf.as_ptr(), attn_out.as_ptr(), hres.as_mut_ptr(), s * h)?;

        // FFN sub-block.
        rmsnorm_f16(hres.as_ptr(), post_norm.as_ptr(), normed2.as_mut_ptr(), s, h, w.rms_eps, w.norm_unit_offset)?;
        linear_f16(normed2.as_ptr(), gatew.as_ptr(), None, gate.as_mut_ptr(), s, inter, h)?;
        linear_f16(normed2.as_ptr(), upw.as_ptr(), None, up.as_mut_ptr(), s, inter, h)?;
        geglu_f16(gate.as_ptr(), up.as_ptr(), ffn_hidden.as_mut_ptr(), s * inter)?;
        linear_f16(ffn_hidden.as_ptr(), downw.as_ptr(), None, ffn_out.as_mut_ptr(), s, h, inter)?;
        add_f16(hres.as_ptr(), ffn_out.as_ptr(), out.as_mut_ptr(), s * h)?;
    }
    crate::synchronize()?;
    out.to_vec()
}

/// Run a stack of transformer layers on the GPU (the multi-layer forward).
#[cfg(has_cuda)]
pub fn transformer_stack_forward_gpu(
    x: &[f16],
    layers: &[LayerWeights],
    seq_len: usize,
) -> crate::Result<Vec<f16>> {
    let mut cur = x.to_vec();
    for layer in layers {
        cur = transformer_layer_forward_gpu(&cur, layer, seq_len)?;
    }
    Ok(cur)
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

    #[test]
    fn attention_reference_causal() {
        // S=2, H=1, D=2, causal. Query 0 attends only to key 0 → out[0] = v[0].
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];
        let out = attention_reference(&q, &k, &v, 2, 1, 2, 1.0, true);
        assert!((out[0] - 5.0).abs() < 1e-5);
        assert!((out[1] - 6.0).abs() < 1e-5);
        // Row 1 mixes v[0] and v[1], so each component lies between them.
        assert!(out[2] >= 5.0 && out[2] <= 7.0);
        assert!(out[3] >= 6.0 && out[3] <= 8.0);
    }

    #[test]
    fn transformer_layer_reference_shapes() {
        let (h, nh, d, inter, s) = (4usize, 2usize, 2usize, 8usize, 3usize); // attn_dim = nh*d = 4 = h
        let w = LayerWeights {
            hidden_size: h,
            num_heads: nh,
            head_dim: d,
            intermediate_size: inter,
            input_norm_w: vec![f16::from_f32(0.0); h],
            q_w: vec![f16::from_f32(0.1); nh * d * h],
            k_w: vec![f16::from_f32(0.1); nh * d * h],
            v_w: vec![f16::from_f32(0.1); nh * d * h],
            o_w: vec![f16::from_f32(0.1); h * nh * d],
            post_norm_w: vec![f16::from_f32(0.0); h],
            gate_w: vec![f16::from_f32(0.1); inter * h],
            up_w: vec![f16::from_f32(0.1); inter * h],
            down_w: vec![f16::from_f32(0.1); h * inter],
            rms_eps: 1e-6,
            norm_unit_offset: 1.0,
        };
        let x = vec![0.5f32; s * h];
        let out = transformer_layer_reference(&x, &w, s);
        assert_eq!(out.len(), s * h);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: forward.rs
// REPO PATH:   /swiftllm/crates/swiftllm-cuda/src/forward.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
