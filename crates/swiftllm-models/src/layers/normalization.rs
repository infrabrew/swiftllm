//! Normalization Layers
//!
//! This module implements normalization layers used in transformer models,
//! including LayerNorm and RMSNorm.

use swiftllm_core::error::Result;
use swiftllm_core::tensor::Tensor;

/// RMS Normalization (used in LLaMA, Mistral, etc.)
///
/// Applies: output = weight * (x / sqrt(mean(x^2) + eps))
#[derive(Debug)]
pub struct RMSNorm {
    /// Weight parameter [hidden_size]
    weight: Tensor,

    /// Epsilon for numerical stability
    eps: f32,

    /// Hidden size
    hidden_size: usize,
}

impl RMSNorm {
    /// Create a new RMSNorm layer
    pub fn new(weight: Tensor, eps: f32) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 1 {
            return Err(swiftllm_core::error::Error::Tensor(
                "RMSNorm weight must be 1D".to_string(),
            ));
        }

        Ok(Self {
            hidden_size: dims[0],
            weight,
            eps,
        })
    }

    /// Forward pass
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // 1. Compute variance: var = mean(x^2)
        // 2. Compute scale: rsqrt = 1 / sqrt(var + eps)
        // 3. Apply: output = x * rsqrt * weight

        // For a real implementation on GPU, this would be a fused kernel

        // Placeholder
        Tensor::zeros(
            hidden_states.dims().to_vec(),
            hidden_states.dtype(),
            hidden_states.device(),
        )
    }

    /// Get the weight tensor
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Get epsilon
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Get hidden size
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

/// Layer Normalization
///
/// Applies: output = ((x - mean) / sqrt(var + eps)) * weight + bias
#[derive(Debug)]
pub struct LayerNorm {
    /// Weight parameter [hidden_size]
    weight: Tensor,

    /// Bias parameter [hidden_size]
    bias: Option<Tensor>,

    /// Epsilon for numerical stability
    eps: f32,

    /// Hidden size
    hidden_size: usize,
}

impl LayerNorm {
    /// Create a new LayerNorm layer
    pub fn new(weight: Tensor, bias: Option<Tensor>, eps: f32) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 1 {
            return Err(swiftllm_core::error::Error::Tensor(
                "LayerNorm weight must be 1D".to_string(),
            ));
        }

        if let Some(ref b) = bias {
            if b.dims() != weight.dims() {
                return Err(swiftllm_core::error::Error::Tensor(
                    "LayerNorm bias must have same shape as weight".to_string(),
                ));
            }
        }

        Ok(Self {
            hidden_size: dims[0],
            weight,
            bias,
            eps,
        })
    }

    /// Forward pass
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // 1. Compute mean: mean = mean(x, dim=-1)
        // 2. Compute variance: var = mean((x - mean)^2, dim=-1)
        // 3. Normalize: x_norm = (x - mean) / sqrt(var + eps)
        // 4. Scale and shift: output = x_norm * weight + bias

        // Placeholder
        Tensor::zeros(
            hidden_states.dims().to_vec(),
            hidden_states.dtype(),
            hidden_states.device(),
        )
    }

    /// Get the weight tensor
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Get the bias tensor
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// Get epsilon
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Get hidden size
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

/// Fused RMSNorm with residual add
///
/// Applies: output = residual + rmsnorm(input)
/// This is more efficient than separate operations
#[derive(Debug)]
pub struct FusedAddRMSNorm {
    /// RMSNorm layer
    norm: RMSNorm,
}

impl FusedAddRMSNorm {
    /// Create a new fused layer
    pub fn new(norm: RMSNorm) -> Self {
        Self { norm }
    }

    /// Forward pass with residual
    pub fn forward(&self, input: &Tensor, residual: &Tensor) -> Result<(Tensor, Tensor)> {
        // Returns (normalized output, new residual)
        // new_residual = input + residual
        // output = rmsnorm(new_residual)

        let output = self.norm.forward(input)?;
        // For simplicity, return the same tensor for residual
        // Real implementation would compute residual connection
        Ok((output.clone(), output))
    }
}

// CPU implementations for testing

/// Compute RMS norm for a 1D slice
pub fn rms_norm_1d(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    assert_eq!(n, weight.len());

    // Compute mean of squares
    let mean_sq: f32 = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;

    // Compute scaling factor
    let rsqrt = 1.0 / (mean_sq + eps).sqrt();

    // Apply normalization and scale
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| xi * rsqrt * wi)
        .collect()
}

/// Compute layer norm for a 1D slice
pub fn layer_norm_1d(x: &[f32], weight: &[f32], bias: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let n = x.len();
    assert_eq!(n, weight.len());

    // Compute mean
    let mean: f32 = x.iter().sum::<f32>() / n as f32;

    // Compute variance
    let var: f32 = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n as f32;

    // Compute scaling factor
    let inv_std = 1.0 / (var + eps).sqrt();

    // Apply normalization, scale, and shift
    x.iter()
        .zip(weight.iter())
        .enumerate()
        .map(|(i, (&xi, &wi))| {
            let normalized = (xi - mean) * inv_std;
            let scaled = normalized * wi;
            if let Some(b) = bias {
                scaled + b[i]
            } else {
                scaled
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_1d() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-5;

        let result = rms_norm_1d(&x, &weight, eps);

        // Check output has same length
        assert_eq!(result.len(), x.len());

        // RMS of x is sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.74
        // Each element should be x_i / rms
        let rms = (30.0f32 / 4.0).sqrt();
        for (i, &r) in result.iter().enumerate() {
            let expected = x[i] / rms;
            assert!((r - expected).abs() < 1e-4, "Mismatch at {}: {} vs {}", i, r, expected);
        }
    }

    #[test]
    fn test_layer_norm_1d() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let bias = vec![0.0, 0.0, 0.0, 0.0];
        let eps = 1e-5;

        let result = layer_norm_1d(&x, &weight, Some(&bias), eps);

        // Mean = 2.5, Var = 1.25, std = 1.118
        // Normalized values should sum to 0 and have unit variance
        let sum: f32 = result.iter().sum();
        assert!(sum.abs() < 1e-4, "Sum should be close to 0: {}", sum);
    }

    #[test]
    fn test_layer_norm_no_bias() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![2.0, 2.0, 2.0, 2.0];
        let eps = 1e-5;

        let result = layer_norm_1d(&x, &weight, None, eps);

        // Should be 2x the normalized values
        let result_unit = layer_norm_1d(&x, &[1.0, 1.0, 1.0, 1.0], None, eps);
        for (r, u) in result.iter().zip(result_unit.iter()) {
            assert!((r - 2.0 * u).abs() < 1e-4);
        }
    }
}
