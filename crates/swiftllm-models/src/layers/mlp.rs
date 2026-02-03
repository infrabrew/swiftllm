//! MLP (Feed-Forward) Layer Implementation
//!
//! This module implements the MLP/FFN layers used in transformer models,
//! including gated variants (SwiGLU, GeGLU) used in LLaMA and similar models.

use super::Linear;
use swiftllm_core::error::Result;
use swiftllm_core::tensor::Tensor;

/// MLP configuration
#[derive(Debug, Clone)]
pub struct MlpConfig {
    /// Hidden size
    pub hidden_size: usize,

    /// Intermediate size
    pub intermediate_size: usize,

    /// Activation function
    pub activation: Activation,

    /// Use bias
    pub use_bias: bool,
}

impl MlpConfig {
    /// Create new MLP config
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            activation: Activation::SiLU,
            use_bias: false,
        }
    }

    /// Set activation function
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Enable bias
    pub fn with_bias(mut self) -> Self {
        self.use_bias = true;
        self
    }
}

/// Activation functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Rectified Linear Unit
    ReLU,
    /// Gaussian Error Linear Unit
    GELU,
    /// Sigmoid Linear Unit (SiLU / Swish)
    SiLU,
    /// GELU approximation (tanh)
    GELUTanh,
    /// Quick GELU
    QuickGELU,
}

impl Activation {
    /// Apply activation function element-wise
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::GELU => gelu(x),
            Activation::SiLU => silu(x),
            Activation::GELUTanh => gelu_tanh(x),
            Activation::QuickGELU => quick_gelu(x),
        }
    }
}

/// Standard MLP layer
#[derive(Debug)]
pub struct Mlp {
    /// Configuration
    config: MlpConfig,

    /// Up projection
    up_proj: Linear,

    /// Down projection
    down_proj: Linear,
}

impl Mlp {
    /// Create a new MLP layer
    pub fn new(config: MlpConfig, up_proj: Linear, down_proj: Linear) -> Self {
        Self {
            config,
            up_proj,
            down_proj,
        }
    }

    /// Forward pass
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // hidden_states: [batch, seq, hidden_size]
        // output: [batch, seq, hidden_size]

        // 1. Up projection: [batch, seq, intermediate_size]
        // 2. Activation
        // 3. Down projection: [batch, seq, hidden_size]

        // Placeholder
        Tensor::zeros(
            hidden_states.dims().to_vec(),
            hidden_states.dtype(),
            hidden_states.device(),
        )
    }
}

/// Gated MLP layer (used in LLaMA, Mistral, etc.)
///
/// Uses SwiGLU: output = down_proj(silu(gate_proj(x)) * up_proj(x))
#[derive(Debug)]
pub struct GatedMlp {
    /// Configuration
    config: MlpConfig,

    /// Gate projection
    gate_proj: Linear,

    /// Up projection
    up_proj: Linear,

    /// Down projection
    down_proj: Linear,
}

impl GatedMlp {
    /// Create a new gated MLP layer
    pub fn new(
        config: MlpConfig,
        gate_proj: Linear,
        up_proj: Linear,
        down_proj: Linear,
    ) -> Self {
        Self {
            config,
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// Forward pass
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // SwiGLU:
        // gate = gate_proj(x)
        // up = up_proj(x)
        // output = down_proj(silu(gate) * up)

        // Placeholder
        Tensor::zeros(
            hidden_states.dims().to_vec(),
            hidden_states.dtype(),
            hidden_states.device(),
        )
    }
}

/// Fused Gated MLP (gate and up projections merged)
#[derive(Debug)]
pub struct FusedGatedMlp {
    /// Configuration
    config: MlpConfig,

    /// Merged gate+up projection
    gate_up_proj: Linear,

    /// Down projection
    down_proj: Linear,
}

impl FusedGatedMlp {
    /// Create a new fused gated MLP
    pub fn new(config: MlpConfig, gate_up_proj: Linear, down_proj: Linear) -> Self {
        Self {
            config,
            gate_up_proj,
            down_proj,
        }
    }

    /// Forward pass
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Fused version:
        // gate_up = gate_up_proj(x)  # [batch, seq, 2 * intermediate]
        // gate, up = split(gate_up)
        // output = down_proj(silu(gate) * up)

        Tensor::zeros(
            hidden_states.dims().to_vec(),
            hidden_states.dtype(),
            hidden_states.device(),
        )
    }
}

/// Mixture of Experts (MoE) MLP layer
#[derive(Debug)]
pub struct MoeMlp {
    /// Number of experts
    num_experts: usize,

    /// Number of experts to route to
    num_experts_per_tok: usize,

    /// Expert MLPs
    experts: Vec<GatedMlp>,

    /// Router (gate) weights
    gate: Linear,
}

impl MoeMlp {
    /// Create a new MoE layer
    pub fn new(
        num_experts: usize,
        num_experts_per_tok: usize,
        experts: Vec<GatedMlp>,
        gate: Linear,
    ) -> Self {
        Self {
            num_experts,
            num_experts_per_tok,
            experts,
            gate,
        }
    }

    /// Forward pass
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // MoE forward:
        // 1. Compute router logits: [batch * seq, num_experts]
        // 2. Select top-k experts per token
        // 3. Route tokens to experts
        // 4. Combine expert outputs

        Tensor::zeros(
            hidden_states.dims().to_vec(),
            hidden_states.dtype(),
            hidden_states.device(),
        )
    }

    /// Get number of experts
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }
}

// Activation function implementations

/// SiLU (Swish) activation: x * sigmoid(x)
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GELU activation (exact)
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (x / std::f32::consts::SQRT_2).erf())
}

/// GELU approximation using tanh
fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x.powi(3))).tanh())
}

/// Quick GELU approximation
fn quick_gelu(x: f32) -> f32 {
    x / (1.0 + (-1.702 * x).exp())
}

/// erf function approximation (for CPU)
trait Erf {
    fn erf(self) -> Self;
}

impl Erf for f32 {
    fn erf(self) -> f32 {
        // Horner form approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if self < 0.0 { -1.0 } else { 1.0 };
        let x = self.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_silu() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!(silu(1.0) > 0.7);
        assert!(silu(-1.0) < -0.2);
    }

    #[test]
    fn test_activation_gelu() {
        assert!((gelu(0.0) - 0.0).abs() < 1e-6);
        assert!(gelu(1.0) > 0.8);
        // GELU is approximately linear for large positive values
        assert!((gelu(3.0) - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_mlp_config() {
        let config = MlpConfig::new(4096, 11008);

        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.intermediate_size, 11008);
        assert_eq!(config.activation, Activation::SiLU);
        assert!(!config.use_bias);
    }

    #[test]
    fn test_erf() {
        assert!((0.0f32.erf() - 0.0).abs() < 1e-6);
        assert!((1.0f32.erf() - 0.8427).abs() < 0.001);
        assert!((-1.0f32.erf() - (-0.8427)).abs() < 0.001);
    }
}
