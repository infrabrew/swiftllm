// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      data.rs
// PATH:      /crates/swiftllm-training/src/data.rs
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

//! Data loading and preprocessing for training

use crate::config::{DataConfig, DataFormat};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;
use thiserror::Error;

/// Data loading errors
#[derive(Error, Debug)]
pub enum DataError {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON parsing error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// Invalid data format
    #[error("Invalid data: {0}")]
    InvalidData(String),
    /// Empty dataset
    #[error("Dataset is empty")]
    EmptyDataset,
}

type Result<T> = std::result::Result<T, DataError>;

/// A single training sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    /// Input text (prompt/instruction)
    pub input: String,
    /// Output text (response/completion) — None for unsupervised
    pub output: Option<String>,
    /// Full formatted text for causal language modeling
    pub text: String,
    /// Token IDs (populated after tokenization)
    pub token_ids: Vec<u32>,
    /// Labels for loss computation (-100 = ignore)
    pub labels: Vec<i64>,
}

/// An instruction-format sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstructionSample {
    /// Instruction/prompt
    pub instruction: String,
    /// Optional input context
    #[serde(default)]
    pub input: Option<String>,
    /// Expected output
    pub output: String,
}

/// Trait for datasets
pub trait Dataset: Send + Sync {
    /// Get the number of samples
    fn len(&self) -> usize;
    /// Check if empty
    fn is_empty(&self) -> bool { self.len() == 0 }
    /// Get a sample by index
    fn get(&self, index: usize) -> Option<&TrainingSample>;
    /// Get a batch of samples
    fn get_batch(&self, indices: &[usize]) -> Vec<&TrainingSample> {
        indices.iter().filter_map(|&i| self.get(i)).collect()
    }
}

/// Plain text dataset (one sample per line)
pub struct TextDataset {
    samples: Vec<TrainingSample>,
}

impl TextDataset {
    /// Load from a text file (one sample per line)
    pub fn from_file(path: &Path, max_seq_len: usize) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let samples: Vec<TrainingSample> = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                let text = if line.len() > max_seq_len * 4 {
                    // Rough char-to-token ratio ~4:1
                    // Use char boundary-safe truncation to avoid panics on multi-byte UTF-8
                    let limit = max_seq_len * 4;
                    let end = line.char_indices()
                        .take_while(|&(i, _)| i < limit)
                        .last()
                        .map(|(i, c)| i + c.len_utf8())
                        .unwrap_or(0);
                    line[..end].to_string()
                } else {
                    line.to_string()
                };
                TrainingSample {
                    input: text.clone(),
                    output: None,
                    text,
                    token_ids: Vec::new(),
                    labels: Vec::new(),
                }
            })
            .collect();

        if samples.is_empty() {
            return Err(DataError::EmptyDataset);
        }

        Ok(Self { samples })
    }
}

impl Dataset for TextDataset {
    fn len(&self) -> usize { self.samples.len() }
    fn get(&self, index: usize) -> Option<&TrainingSample> { self.samples.get(index) }
}

/// Instruction dataset (JSONL with instruction/input/output fields)
pub struct InstructionDataset {
    samples: Vec<TrainingSample>,
    template: String,
}

impl InstructionDataset {
    /// Default instruction template (Alpaca-style)
    const DEFAULT_TEMPLATE: &'static str =
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}";

    /// Load from JSONL file
    pub fn from_jsonl(path: &Path, template: Option<&str>, max_seq_len: usize) -> Result<Self> {
        let template = template.unwrap_or(Self::DEFAULT_TEMPLATE).to_string();
        let content = std::fs::read_to_string(path)?;
        let mut samples = Vec::new();

        for (line_num, line) in content.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }

            let sample: InstructionSample = serde_json::from_str(line).map_err(|e| {
                DataError::InvalidData(format!("Line {}: {}", line_num + 1, e))
            })?;

            let text = template
                .replace("{instruction}", &sample.instruction)
                .replace("{input}", sample.input.as_deref().unwrap_or(""))
                .replace("{output}", &sample.output);

            let truncated = if text.len() > max_seq_len * 4 {
                text[..max_seq_len * 4].to_string()
            } else {
                text
            };

            samples.push(TrainingSample {
                input: sample.instruction,
                output: Some(sample.output),
                text: truncated,
                token_ids: Vec::new(),
                labels: Vec::new(),
            });
        }

        if samples.is_empty() {
            return Err(DataError::EmptyDataset);
        }

        Ok(Self { samples, template })
    }

    /// Get the template
    pub fn template(&self) -> &str {
        &self.template
    }
}

impl Dataset for InstructionDataset {
    fn len(&self) -> usize { self.samples.len() }
    fn get(&self, index: usize) -> Option<&TrainingSample> { self.samples.get(index) }
}

/// DataLoader with batching and shuffling
pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    position: usize,
    epoch: usize,
}

impl<D: Dataset> DataLoader<D> {
    /// Create a new DataLoader
    pub fn new(dataset: D, batch_size: usize, shuffle: bool) -> Self {
        let len = dataset.len();
        let indices: Vec<usize> = (0..len).collect();
        Self {
            dataset,
            batch_size,
            shuffle,
            indices,
            position: 0,
            epoch: 0,
        }
    }

    /// Reset for a new epoch (and optionally shuffle)
    pub fn reset(&mut self) {
        self.position = 0;
        self.epoch += 1;
        if self.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }

    /// Get the next batch of indices
    pub fn next_batch_indices(&mut self) -> Option<Vec<usize>> {
        if self.position >= self.indices.len() {
            return None;
        }

        let end = (self.position + self.batch_size).min(self.indices.len());
        let batch_indices: Vec<usize> = self.indices[self.position..end].to_vec();
        self.position = end;

        Some(batch_indices)
    }

    /// Get the next batch of samples
    pub fn next_batch(&mut self) -> Option<Vec<&TrainingSample>> {
        let indices = self.next_batch_indices()?;
        Some(self.dataset.get_batch(&indices))
    }

    /// Number of batches per epoch
    pub fn num_batches(&self) -> usize {
        (self.dataset.len() + self.batch_size - 1) / self.batch_size
    }

    /// Current epoch
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Total number of samples
    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_temp_text_file(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "{}", content).unwrap();
        file
    }

    #[test]
    fn test_text_dataset() {
        let file = create_temp_text_file("Hello world\nThis is a test\nThird line\n");
        let dataset = TextDataset::from_file(file.path(), 2048).unwrap();
        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.get(0).unwrap().text, "Hello world");
    }

    #[test]
    fn test_instruction_dataset() {
        let content = r#"{"instruction":"Say hello","output":"Hello!"}
{"instruction":"Count to 3","input":"starting from 1","output":"1, 2, 3"}"#;
        let file = create_temp_text_file(content);
        let dataset = InstructionDataset::from_jsonl(file.path(), None, 2048).unwrap();
        assert_eq!(dataset.len(), 2);
        assert!(dataset.get(0).unwrap().text.contains("Say hello"));
    }

    #[test]
    fn test_dataloader() {
        let file = create_temp_text_file("a\nb\nc\nd\ne\n");
        let dataset = TextDataset::from_file(file.path(), 2048).unwrap();
        let mut loader = DataLoader::new(dataset, 2, false);

        assert_eq!(loader.num_batches(), 3); // ceil(5/2)

        let batch1 = loader.next_batch().unwrap();
        assert_eq!(batch1.len(), 2);

        let batch2 = loader.next_batch().unwrap();
        assert_eq!(batch2.len(), 2);

        let batch3 = loader.next_batch().unwrap();
        assert_eq!(batch3.len(), 1);

        assert!(loader.next_batch().is_none());
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: data.rs
// REPO PATH:   /swiftllm/crates/swiftllm-training/src/data.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
