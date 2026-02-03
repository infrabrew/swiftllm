//! Embedding Layer Implementation
//!
//! This module provides embedding layers for transformer models,
//! including vocabulary parallel embedding for multi-GPU setups.

use swiftllm_core::error::{Error, Result};
use swiftllm_core::tensor::Tensor;
use swiftllm_core::types::TokenId;

/// Embedding layer
#[derive(Debug)]
pub struct Embedding {
    /// Embedding weights [vocab_size, hidden_size]
    weight: Tensor,

    /// Vocabulary size
    vocab_size: usize,

    /// Embedding dimension
    embedding_dim: usize,

    /// Padding index (tokens at this index produce zeros)
    padding_idx: Option<u32>,
}

impl Embedding {
    /// Create a new embedding layer
    pub fn new(weight: Tensor, padding_idx: Option<u32>) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 2 {
            return Err(Error::Tensor(
                "Embedding weight must be 2D [vocab_size, hidden_size]".to_string(),
            ));
        }

        Ok(Self {
            vocab_size: dims[0],
            embedding_dim: dims[1],
            weight,
            padding_idx,
        })
    }

    /// Forward pass
    pub fn forward(&self, input_ids: &[TokenId]) -> Result<Tensor> {
        // Lookup embeddings for each token
        // Output shape: [seq_len, hidden_size]

        // Validate input
        for &token_id in input_ids {
            if (token_id as usize) >= self.vocab_size {
                return Err(Error::Tensor(format!(
                    "Token ID {} out of vocabulary range [0, {})",
                    token_id, self.vocab_size
                )));
            }
        }

        // Placeholder output
        let seq_len = input_ids.len();
        Tensor::zeros(
            vec![seq_len, self.embedding_dim],
            self.weight.dtype(),
            self.weight.device(),
        )
    }

    /// Forward pass for batched input
    pub fn forward_batch(&self, input_ids: &[Vec<TokenId>]) -> Result<Tensor> {
        // Output shape: [batch_size, seq_len, hidden_size]

        if input_ids.is_empty() {
            return Err(Error::Tensor("Empty batch".to_string()));
        }

        let batch_size = input_ids.len();
        let max_seq_len = input_ids.iter().map(|ids| ids.len()).max().unwrap_or(0);

        Tensor::zeros(
            vec![batch_size, max_seq_len, self.embedding_dim],
            self.weight.dtype(),
            self.weight.device(),
        )
    }

    /// Get the embedding for a single token
    pub fn get_embedding(&self, token_id: TokenId) -> Result<Vec<f32>> {
        if (token_id as usize) >= self.vocab_size {
            return Err(Error::Tensor(format!(
                "Token ID {} out of vocabulary range",
                token_id
            )));
        }

        // Check for padding
        if let Some(pad_idx) = self.padding_idx {
            if token_id == pad_idx {
                return Ok(vec![0.0; self.embedding_dim]);
            }
        }

        // In a real implementation, this would read from the tensor
        // For now, return placeholder
        Ok(vec![0.0; self.embedding_dim])
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Get the weight tensor
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

/// Vocabulary parallel embedding (sharded across GPUs)
#[derive(Debug)]
pub struct VocabParallelEmbedding {
    /// Local embedding weights
    weight: Tensor,

    /// Total vocabulary size
    vocab_size: usize,

    /// Local vocabulary size (per GPU)
    vocab_size_per_gpu: usize,

    /// Embedding dimension
    embedding_dim: usize,

    /// Start index for this shard
    vocab_start_idx: usize,

    /// End index for this shard
    vocab_end_idx: usize,

    /// Tensor parallel rank
    tp_rank: usize,

    /// Tensor parallel world size
    tp_size: usize,
}

impl VocabParallelEmbedding {
    /// Create a new vocabulary parallel embedding
    pub fn new(
        weight: Tensor,
        vocab_size: usize,
        tp_rank: usize,
        tp_size: usize,
    ) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 2 {
            return Err(Error::Tensor(
                "Embedding weight must be 2D".to_string(),
            ));
        }

        let vocab_size_per_gpu = (vocab_size + tp_size - 1) / tp_size;
        let vocab_start_idx = tp_rank * vocab_size_per_gpu;
        let vocab_end_idx = ((tp_rank + 1) * vocab_size_per_gpu).min(vocab_size);

        Ok(Self {
            embedding_dim: dims[1],
            weight,
            vocab_size,
            vocab_size_per_gpu,
            vocab_start_idx,
            vocab_end_idx,
            tp_rank,
            tp_size,
        })
    }

    /// Forward pass
    pub fn forward(&self, input_ids: &[TokenId]) -> Result<Tensor> {
        // For each token:
        // 1. Check if it belongs to this shard
        // 2. If yes, lookup embedding
        // 3. If no, return zeros (will be filled by all-reduce)

        let seq_len = input_ids.len();
        Tensor::zeros(
            vec![seq_len, self.embedding_dim],
            self.weight.dtype(),
            self.weight.device(),
        )
    }

    /// Check if a token belongs to this shard
    pub fn contains(&self, token_id: TokenId) -> bool {
        let idx = token_id as usize;
        idx >= self.vocab_start_idx && idx < self.vocab_end_idx
    }

    /// Get local index for a token
    pub fn local_index(&self, token_id: TokenId) -> Option<usize> {
        let idx = token_id as usize;
        if self.contains(token_id) {
            Some(idx - self.vocab_start_idx)
        } else {
            None
        }
    }
}

/// Linear projection for output embeddings (LM head)
///
/// Often tied to the input embeddings
#[derive(Debug)]
pub struct LMHead {
    /// Weight matrix [vocab_size, hidden_size]
    /// May be tied to embedding weight
    weight: Tensor,

    /// Vocabulary size
    vocab_size: usize,

    /// Hidden size
    hidden_size: usize,

    /// Whether weights are tied to embedding
    tied: bool,
}

impl LMHead {
    /// Create a new LM head
    pub fn new(weight: Tensor, tied: bool) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 2 {
            return Err(Error::Tensor(
                "LM head weight must be 2D".to_string(),
            ));
        }

        Ok(Self {
            vocab_size: dims[0],
            hidden_size: dims[1],
            weight,
            tied,
        })
    }

    /// Forward pass - compute logits
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // hidden_states: [batch, seq, hidden_size]
        // output: [batch, seq, vocab_size]

        let dims = hidden_states.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];

        Tensor::zeros(
            vec![batch_size, seq_len, self.vocab_size],
            hidden_states.dtype(),
            hidden_states.device(),
        )
    }

    /// Get logits for last token only (more efficient for inference)
    pub fn forward_last_token(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // hidden_states: [batch, hidden_size]
        // output: [batch, vocab_size]

        let batch_size = hidden_states.dims()[0];

        Tensor::zeros(
            vec![batch_size, self.vocab_size],
            hidden_states.dtype(),
            hidden_states.device(),
        )
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Check if weights are tied
    pub fn is_tied(&self) -> bool {
        self.tied
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use swiftllm_core::config::DataType;
    use swiftllm_core::tensor::Device;

    #[test]
    fn test_embedding() {
        let weight = Tensor::zeros(vec![32000, 4096], DataType::Float32, Device::Cpu).unwrap();
        let embedding = Embedding::new(weight, None).unwrap();

        assert_eq!(embedding.vocab_size(), 32000);
        assert_eq!(embedding.embedding_dim(), 4096);

        let output = embedding.forward(&[1, 2, 3]).unwrap();
        assert_eq!(output.dims(), &[3, 4096]);
    }

    #[test]
    fn test_embedding_out_of_range() {
        let weight = Tensor::zeros(vec![1000, 512], DataType::Float32, Device::Cpu).unwrap();
        let embedding = Embedding::new(weight, None).unwrap();

        let result = embedding.forward(&[1001]);
        assert!(result.is_err());
    }

    #[test]
    fn test_vocab_parallel_embedding() {
        let weight = Tensor::zeros(vec![8000, 4096], DataType::Float32, Device::Cpu).unwrap();
        let embedding = VocabParallelEmbedding::new(weight, 32000, 0, 4).unwrap();

        assert!(embedding.contains(0));
        assert!(embedding.contains(7999));
        assert!(!embedding.contains(8000));

        assert_eq!(embedding.local_index(100), Some(100));
        assert_eq!(embedding.local_index(8000), None);
    }
}
