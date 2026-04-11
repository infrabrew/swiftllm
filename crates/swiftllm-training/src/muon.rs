// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      muon.rs
// PATH:      /crates/swiftllm-training/src/muon.rs
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

//! Muon optimizer — Momentum Orthogonalization Update
//!
//! Muon applies Nesterov momentum followed by Newton-Schulz orthogonalization
//! to produce update directions that lie on the Stiefel manifold (orthogonal
//! matrices). This yields faster convergence for matrix-shaped parameters
//! (e.g. linear layers) compared to Adam.
//!
//! For 1D parameters (biases, LayerNorm/RMSNorm weights) Muon falls back to
//! AdamW since orthogonalization is undefined for vectors.
//!
//! # Reference
//!
//! Bernstein et al., "Old Optimizer, New Norm: An Anthology" (2024)
//! <https://arxiv.org/abs/2409.20325>

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::optimizer::Optimizer;

/// Configuration for the Muon optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MuonConfig {
    /// Learning rate for matrix-shaped (>=2D) parameters
    pub lr: f64,

    /// Nesterov momentum coefficient (typical: 0.95)
    pub momentum: f64,

    /// Number of Newton-Schulz iterations for orthogonalization (typical: 5-10, max: 20)
    pub ns_steps: usize,

    /// Weight decay (decoupled, applied before the update)
    pub weight_decay: f64,

    /// Learning rate for the AdamW fallback (1D params)
    pub adamw_lr: f64,

    /// Beta1 for AdamW fallback
    pub adamw_beta1: f64,

    /// Beta2 for AdamW fallback
    pub adamw_beta2: f64,

    /// Epsilon for AdamW fallback
    pub adamw_eps: f64,

    /// Weight decay for AdamW fallback
    pub adamw_weight_decay: f64,
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            lr: 0.02,
            momentum: 0.95,
            ns_steps: 5,
            weight_decay: 0.0,
            adamw_lr: 3e-4,
            adamw_beta1: 0.9,
            adamw_beta2: 0.999,
            adamw_eps: 1e-8,
            adamw_weight_decay: 0.01,
        }
    }
}

/// Per-parameter state for Muon (2D+) parameters
struct MuonState {
    /// Momentum buffer (flat, same length as param)
    buf: Vec<f32>,
    /// Rows (product of all dims except last)
    rows: usize,
    /// Columns (last dim)
    cols: usize,
}

/// Per-parameter state for the AdamW fallback (1D params)
struct AdamWState {
    m: Vec<f32>,
    v: Vec<f32>,
    step_count: u64,
}

/// Unified per-parameter state
enum ParamState {
    Muon(MuonState),
    AdamW(AdamWState),
}

/// The Muon optimizer
///
/// Applies Newton-Schulz orthogonalization to the Nesterov momentum buffer
/// for >=2D parameters and falls back to AdamW for 1D parameters.
///
/// # Shape detection
///
/// The optimizer infers parameter dimensionality from the `param_name`:
/// - Names containing `".bias"`, `"bias"`, `"_norm"`, `"ln_"`, `"layernorm"`,
///   or `"rmsnorm"` are treated as 1D (AdamW fallback).
/// - All other parameters are treated as 2D matrices. If the parameter length
///   is not factored by `set_shape`, the optimizer uses a square-ish layout.
///
/// For explicit shape control, call [`Muon::set_shape`] before the first step.
///
/// # Example
///
/// ```
/// use swiftllm_training::muon::{Muon, MuonConfig};
/// use swiftllm_training::optimizer::Optimizer;
///
/// let mut opt = Muon::new(MuonConfig::default());
///
/// // Tell Muon this param is 128×64
/// opt.set_shape("layer0.weight", 128, 64);
///
/// let mut param = vec![0.01f32; 128 * 64];
/// let grad = vec![0.001f32; 128 * 64];
/// opt.step(&mut param, &grad, "layer0.weight");
/// ```
pub struct Muon {
    config: MuonConfig,
    states: HashMap<String, ParamState>,
    shapes: HashMap<String, (usize, usize)>,
}

impl Muon {
    /// Create a new Muon optimizer with the given configuration
    pub fn new(config: MuonConfig) -> Self {
        Self {
            config,
            states: HashMap::new(),
            shapes: HashMap::new(),
        }
    }

    /// Create a Muon optimizer with default settings
    pub fn with_defaults() -> Self {
        Self::new(MuonConfig::default())
    }

    /// Register the (rows, cols) shape for a named parameter.
    /// Must be called before the first `step` for that parameter if you want
    /// explicit shape control. Otherwise the optimizer infers shape heuristically.
    pub fn set_shape(&mut self, param_name: &str, rows: usize, cols: usize) {
        self.shapes.insert(param_name.to_string(), (rows, cols));
    }

    /// Return a reference to the current configuration
    pub fn config(&self) -> &MuonConfig {
        &self.config
    }

    /// Check whether a parameter name looks like a 1D parameter (bias / norm)
    fn is_1d_param(name: &str) -> bool {
        let lower = name.to_lowercase();
        lower.contains("bias")
            || lower.contains("_norm")
            || lower.contains("ln_")
            || lower.contains("layernorm")
            || lower.contains("rmsnorm")
            || lower.contains("embed")
    }

    /// Infer (rows, cols) for a parameter
    fn infer_shape(&self, param_name: &str, n: usize) -> (usize, usize) {
        if let Some(&(r, c)) = self.shapes.get(param_name) {
            return (r, c);
        }
        // Best-effort square-ish factorisation
        let sqrt = (n as f64).sqrt() as usize;
        for c in (1..=sqrt).rev() {
            if n % c == 0 {
                return (n / c, c);
            }
        }
        (n, 1)
    }

    // ── Muon step (>=2D) ───────────────────────────────────────────────

    fn step_muon(&mut self, param: &mut [f32], grad: &[f32], param_name: &str) {
        let n = param.len();
        let (rows, cols) = self.infer_shape(param_name, n);
        let mu = self.config.momentum as f32;
        let lr = self.config.lr as f32;
        let wd = self.config.weight_decay as f32;

        // Lazily initialise state
        let state = self.states
            .entry(param_name.to_string())
            .or_insert_with(|| {
                ParamState::Muon(MuonState {
                    buf: vec![0.0f32; n],
                    rows,
                    cols,
                })
            });

        let ms = match state {
            ParamState::Muon(s) => s,
            _ => return, // shouldn't happen
        };

        // 1. Nesterov momentum: buf = mu * buf + grad
        for i in 0..n {
            ms.buf[i] = mu * ms.buf[i] + grad[i];
        }

        // 2. Newton-Schulz orthogonalization
        //    Treat buf as (rows x cols). If rows < cols, transpose so inner dim is smaller.
        let (work_rows, work_cols, transposed) = if ms.rows >= ms.cols {
            (ms.rows, ms.cols, false)
        } else {
            (ms.cols, ms.rows, true)
        };

        let mut x = vec![0.0f32; work_rows * work_cols];
        if transposed {
            transpose(&ms.buf, ms.rows, ms.cols, &mut x);
        } else {
            x.copy_from_slice(&ms.buf[..n]);
        }

        // Spectral normalisation: scale so Frobenius norm ~ 1
        let frob = frobenius_norm(&x);
        if frob > 0.0 {
            let scale = 1.0 / frob;
            for v in x.iter_mut() {
                *v *= scale;
            }
        }

        // Newton-Schulz iterations: X <- X * (3I - X^T X) / 2
        let ns_steps = self.config.ns_steps.min(20); // Cap to prevent runaway loops
        let mut xtx = vec![0.0f32; work_cols * work_cols];
        let mut temp = vec![0.0f32; work_rows * work_cols];

        for _ in 0..ns_steps {
            // xtx = X^T * X
            matmul_at_b(&x, &x, work_rows, work_cols, work_cols, &mut xtx);

            // xtx = 3I - X^T X
            for r in 0..work_cols {
                for c in 0..work_cols {
                    let idx = r * work_cols + c;
                    let identity = if r == c { 3.0f32 } else { 0.0f32 };
                    xtx[idx] = identity - xtx[idx];
                }
            }

            // temp = X * (3I - X^T X)
            matmul_ab(&x, &xtx, work_rows, work_cols, work_cols, &mut temp);

            // X = temp / 2
            for i in 0..x.len() {
                x[i] = temp[i] * 0.5;
            }
        }

        // Rescale back
        if frob > 0.0 {
            for v in x.iter_mut() {
                *v *= frob;
            }
        }

        // Un-transpose if needed
        if transposed {
            transpose(&x, work_rows, work_cols, &mut ms.buf);
        } else {
            ms.buf[..n].copy_from_slice(&x);
        }

        // 3. Apply decoupled weight decay + update
        for i in 0..n {
            param[i] -= lr * (ms.buf[i] + wd * param[i]);
        }
    }

    // ── AdamW step (1D) ────────────────────────────────────────────────

    fn step_adamw(&mut self, param: &mut [f32], grad: &[f32], param_name: &str) {
        let n = param.len();
        let lr = self.config.adamw_lr as f32;
        let beta1 = self.config.adamw_beta1 as f32;
        let beta2 = self.config.adamw_beta2 as f32;
        let eps = self.config.adamw_eps as f32;
        let wd = self.config.adamw_weight_decay as f32;

        let state = self.states
            .entry(param_name.to_string())
            .or_insert_with(|| {
                ParamState::AdamW(AdamWState {
                    m: vec![0.0f32; n],
                    v: vec![0.0f32; n],
                    step_count: 0,
                })
            });

        let aws = match state {
            ParamState::AdamW(s) => s,
            _ => return,
        };

        aws.step_count += 1;
        let t = aws.step_count as f64;
        let bc1 = (1.0 - (self.config.adamw_beta1).powf(t)).max(f64::EPSILON);
        let bc2 = (1.0 - (self.config.adamw_beta2).powf(t)).max(f64::EPSILON);

        for i in 0..n {
            aws.m[i] = beta1 * aws.m[i] + (1.0 - beta1) * grad[i];
            aws.v[i] = beta2 * aws.v[i] + (1.0 - beta2) * grad[i] * grad[i];

            let m_hat = aws.m[i] / bc1 as f32;
            let v_hat = aws.v[i] / bc2 as f32;

            param[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * param[i]);
        }
    }
}

impl Optimizer for Muon {
    fn name(&self) -> &str {
        "Muon"
    }

    fn learning_rate(&self) -> f64 {
        self.config.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.config.lr = lr;
    }

    fn step(&mut self, param: &mut [f32], grad: &[f32], param_name: &str) {
        assert_eq!(param.len(), grad.len(), "param and grad length mismatch");

        if Self::is_1d_param(param_name) {
            self.step_adamw(param, grad, param_name);
        } else {
            self.step_muon(param, grad, param_name);
        }
    }

    fn reset(&mut self) {
        self.states.clear();
    }
}

// ── Linear-algebra helpers (CPU, f32) ───────────────────────────────────────

/// Frobenius norm of a flat matrix
fn frobenius_norm(a: &[f32]) -> f32 {
    a.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Transpose an (m x n) row-major matrix into (n x m)
fn transpose(src: &[f32], m: usize, n: usize, dst: &mut [f32]) {
    for r in 0..m {
        for c in 0..n {
            dst[c * m + r] = src[r * n + c];
        }
    }
}

/// C = A^T * B  where A is (m x k), B is (m x n), C is (k x n)
fn matmul_at_b(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, c: &mut [f32]) {
    c.iter_mut().for_each(|v| *v = 0.0);
    for i in 0..m {
        for j in 0..k {
            let a_val = a[i * k + j];
            for l in 0..n {
                c[j * n + l] += a_val * b[i * n + l];
            }
        }
    }
}

/// C = A * B  where A is (m x k), B is (k x n), C is (m x n)
fn matmul_ab(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, c: &mut [f32]) {
    c.iter_mut().for_each(|v| *v = 0.0);
    for i in 0..m {
        for j in 0..k {
            let a_val = a[i * k + j];
            for l in 0..n {
                c[i * n + l] += a_val * b[j * n + l];
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = vec![0.0; 6];
        transpose(&src, 2, 3, &mut dst);
        assert_eq!(dst, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_frobenius_norm() {
        let v = vec![3.0, 4.0];
        assert!((frobenius_norm(&v) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_ab() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        matmul_ab(&a, &b, 2, 2, 2, &mut c);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_at_b_identity() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![0.0; 4];
        matmul_at_b(&a, &a, 2, 2, 2, &mut c);
        assert_eq!(c, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_muon_2d_step() {
        let mut opt = Muon::new(MuonConfig {
            lr: 0.01,
            momentum: 0.95,
            ns_steps: 5,
            ..Default::default()
        });
        opt.set_shape("layer.weight", 8, 4);

        let mut param = vec![0.1f32; 32];
        let grad = vec![0.01f32; 32];

        opt.step(&mut param, &grad, "layer.weight");

        let unchanged = param.iter().all(|&v| (v - 0.1).abs() < 1e-12);
        assert!(!unchanged, "Muon step should modify parameters");
    }

    #[test]
    fn test_bias_uses_adamw() {
        let mut opt = Muon::new(MuonConfig::default());

        let mut param = vec![0.5f32; 128];
        let grad = vec![0.01f32; 128];

        opt.step(&mut param, &grad, "layer.bias");

        let unchanged = param.iter().all(|&v| (v - 0.5).abs() < 1e-12);
        assert!(!unchanged, "AdamW fallback should modify bias parameters");

        // Verify the state is AdamW
        match opt.states.get("layer.bias") {
            Some(ParamState::AdamW(_)) => {},
            _ => panic!("bias should use AdamW state"),
        }
    }

    #[test]
    fn test_norm_uses_adamw() {
        let mut opt = Muon::new(MuonConfig::default());

        let mut param = vec![1.0f32; 64];
        let grad = vec![0.01f32; 64];

        opt.step(&mut param, &grad, "model.layernorm.weight");

        match opt.states.get("model.layernorm.weight") {
            Some(ParamState::AdamW(_)) => {},
            _ => panic!("layernorm should use AdamW state"),
        }
    }

    #[test]
    fn test_multiple_steps_converge() {
        let mut opt = Muon::new(MuonConfig {
            lr: 0.01,
            momentum: 0.95,
            ns_steps: 5,
            ..Default::default()
        });
        opt.set_shape("w", 4, 4);

        let mut param = vec![1.0f32; 16];

        for _ in 0..20 {
            let grad: Vec<f32> = param.iter().map(|&p| p).collect();
            opt.step(&mut param, &grad, "w");
        }

        let dist: f32 = param.iter().map(|p| p * p).sum::<f32>().sqrt();
        assert!(
            dist < 1.0,
            "After 20 Muon steps, parameter norm should decrease (got {})",
            dist
        );
    }

    #[test]
    fn test_wide_matrix() {
        let mut opt = Muon::new(MuonConfig {
            lr: 0.01,
            ns_steps: 3,
            ..Default::default()
        });
        opt.set_shape("w", 4, 16);

        let mut param = vec![0.1f32; 64];
        let grad = vec![0.01f32; 64];

        opt.step(&mut param, &grad, "w");

        let unchanged = param.iter().all(|&v| (v - 0.1).abs() < 1e-12);
        assert!(!unchanged, "Wide matrix Muon step should modify parameters");
    }

    #[test]
    fn test_reset_clears_state() {
        let mut opt = Muon::new(MuonConfig::default());
        opt.set_shape("w", 4, 4);

        let mut param = vec![0.1f32; 16];
        let grad = vec![0.01f32; 16];
        opt.step(&mut param, &grad, "w");

        assert!(!opt.states.is_empty());
        opt.reset();
        assert!(opt.states.is_empty());
    }

    #[test]
    fn test_optimizer_name() {
        let opt = Muon::with_defaults();
        assert_eq!(opt.name(), "Muon");
    }

    #[test]
    fn test_learning_rate_get_set() {
        let mut opt = Muon::new(MuonConfig {
            lr: 0.02,
            ..Default::default()
        });

        assert!((opt.learning_rate() - 0.02).abs() < 1e-12);
        opt.set_learning_rate(0.001);
        assert!((opt.learning_rate() - 0.001).abs() < 1e-12);
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: muon.rs
// REPO PATH:   /swiftllm/crates/swiftllm-training/src/muon.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
