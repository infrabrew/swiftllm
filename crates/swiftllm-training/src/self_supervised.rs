// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      self_supervised.rs
// PATH:      /crates/swiftllm-training/src/self_supervised.rs
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

//! Self-supervised learning objectives.
//!
//! Self-supervised learning builds its own supervision from unlabeled text: the
//! target is derived from the input itself, so no human labels are required.
//! This module turns a token sequence into a training example for three classic
//! objectives:
//!
//! * [`SslObjective::CausalLm`] — next-token prediction (decoder-only LMs).
//! * [`SslObjective::MaskedLm`] — BERT-style masked LM (the 80/10/10 corruption).
//! * [`SslObjective::SpanCorruption`] — T5-style span denoising with sentinels.
//!
//! These are pure token-level transforms ([`build_example`]) and are fully
//! deterministic under a seed, so they unit-test on CPU. The forward pass /
//! loss is the model seam — the example produced here is exactly what a trainer
//! feeds the model.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use swiftllm_core::types::TokenId;

/// A self-supervised training example.
///
/// `labels[i] == None` marks a position that contributes no loss (the HF
/// `-100` ignore-index convention). For sequence-to-sequence objectives
/// (span corruption) the supervision lives in `target_ids` and `labels` is all
/// `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct SslExample {
    /// Model input tokens (possibly corrupted).
    pub input_ids: Vec<TokenId>,
    /// Per-position target token, or `None` where no loss is applied.
    pub labels: Vec<Option<TokenId>>,
    /// Decoder targets for seq2seq objectives (span corruption); else `None`.
    pub target_ids: Option<Vec<TokenId>>,
}

impl SslExample {
    /// Number of positions that contribute to the loss.
    pub fn loss_positions(&self) -> usize {
        self.labels.iter().filter(|l| l.is_some()).count()
    }

    /// Render labels with an explicit ignore index (e.g. `-100`), the format
    /// most autograd training loops expect.
    pub fn labels_with_ignore_index(&self, ignore: i64) -> Vec<i64> {
        self.labels
            .iter()
            .map(|l| l.map(|t| t as i64).unwrap_or(ignore))
            .collect()
    }
}

/// Configuration for masked language modeling (BERT-style).
#[derive(Debug, Clone)]
pub struct MlmConfig {
    /// Fraction of tokens selected for prediction (typ. 0.15).
    pub mask_prob: f32,
    /// The `[MASK]` token id.
    pub mask_token_id: TokenId,
    /// Vocabulary size (for random replacement).
    pub vocab_size: u32,
    /// Of selected tokens, fraction replaced with `[MASK]` (typ. 0.8).
    pub replace_with_mask_prob: f32,
    /// Of selected tokens, fraction replaced with a random token (typ. 0.1);
    /// the remainder are kept unchanged.
    pub replace_with_random_prob: f32,
}

impl Default for MlmConfig {
    fn default() -> Self {
        Self {
            mask_prob: 0.15,
            mask_token_id: 103, // BERT [MASK]
            vocab_size: 30522,
            replace_with_mask_prob: 0.8,
            replace_with_random_prob: 0.1,
        }
    }
}

/// Configuration for T5-style span corruption.
#[derive(Debug, Clone)]
pub struct SpanConfig {
    /// Fraction of tokens corrupted (typ. 0.15).
    pub mask_prob: f32,
    /// Mean masked-span length (typ. 3).
    pub mean_span_len: usize,
    /// Sentinel ids count *down* from here (`<extra_id_0> = sentinel_start_id`).
    pub sentinel_start_id: TokenId,
}

impl Default for SpanConfig {
    fn default() -> Self {
        Self {
            mask_prob: 0.15,
            mean_span_len: 3,
            sentinel_start_id: 32099, // T5 <extra_id_0>
        }
    }
}

/// A self-supervised objective.
#[derive(Debug, Clone)]
pub enum SslObjective {
    /// Next-token prediction.
    CausalLm,
    /// Masked language modeling.
    MaskedLm(MlmConfig),
    /// Span corruption / denoising.
    SpanCorruption(SpanConfig),
}

/// Build a self-supervised example from a token sequence under `objective`.
/// `seed` makes corruption deterministic.
pub fn build_example(objective: &SslObjective, tokens: &[TokenId], seed: u64) -> SslExample {
    match objective {
        SslObjective::CausalLm => build_causal_lm(tokens),
        SslObjective::MaskedLm(cfg) => {
            let mut rng = StdRng::seed_from_u64(seed);
            build_masked_lm(tokens, cfg, &mut rng)
        }
        SslObjective::SpanCorruption(cfg) => {
            let mut rng = StdRng::seed_from_u64(seed);
            build_span_corruption(tokens, cfg, &mut rng)
        }
    }
}

/// Causal LM: predict token *t+1* from tokens *0..=t*. Inputs and labels are the
/// sequence shifted by one.
pub fn build_causal_lm(tokens: &[TokenId]) -> SslExample {
    if tokens.len() < 2 {
        return SslExample {
            input_ids: Vec::new(),
            labels: Vec::new(),
            target_ids: None,
        };
    }
    let input_ids = tokens[..tokens.len() - 1].to_vec();
    let labels = tokens[1..].iter().map(|&t| Some(t)).collect();
    SslExample {
        input_ids,
        labels,
        target_ids: None,
    }
}

/// Masked LM with the BERT 80/10/10 corruption.
pub fn build_masked_lm(tokens: &[TokenId], cfg: &MlmConfig, rng: &mut StdRng) -> SslExample {
    let mut input_ids = tokens.to_vec();
    let mut labels = vec![None; tokens.len()];

    for i in 0..tokens.len() {
        if rng.gen::<f32>() >= cfg.mask_prob {
            continue;
        }
        // Selected for prediction: label is the original token.
        labels[i] = Some(tokens[i]);
        let r: f32 = rng.gen();
        if r < cfg.replace_with_mask_prob {
            input_ids[i] = cfg.mask_token_id; // 80%: [MASK]
        } else if r < cfg.replace_with_mask_prob + cfg.replace_with_random_prob {
            input_ids[i] = rng.gen_range(0..cfg.vocab_size.max(1)); // 10%: random
        }
        // else (remaining 10%): keep the original token unchanged.
    }

    SslExample {
        input_ids,
        labels,
        target_ids: None,
    }
}

/// T5-style span corruption: contiguous masked spans are each replaced in the
/// input by a single sentinel; the target is `<sentinel_i> span_i …` followed by
/// a final sentinel.
pub fn build_span_corruption(tokens: &[TokenId], cfg: &SpanConfig, rng: &mut StdRng) -> SslExample {
    let n = tokens.len();
    if n == 0 {
        return SslExample {
            input_ids: Vec::new(),
            labels: Vec::new(),
            target_ids: Some(vec![cfg.sentinel_start_id]),
        };
    }

    // 1. Choose masked positions by placing spans until the budget is reached.
    let budget = ((n as f32) * cfg.mask_prob).round() as usize;
    let span_len = cfg.mean_span_len.max(1);
    let mut masked = vec![false; n];
    let mut masked_count = 0usize;
    let mut attempts = 0usize;
    while masked_count < budget && attempts < n * 4 {
        attempts += 1;
        let start = rng.gen_range(0..n);
        for j in start..(start + span_len).min(n) {
            if !masked[j] {
                masked[j] = true;
                masked_count += 1;
                if masked_count >= budget {
                    break;
                }
            }
        }
    }

    // 2. Walk the sequence, emitting sentinels for masked spans.
    let mut input_ids = Vec::with_capacity(n);
    let mut target_ids = Vec::new();
    let mut sentinel = 0u32;
    let mut i = 0;
    while i < n {
        if masked[i] {
            let sid = cfg.sentinel_start_id - sentinel;
            input_ids.push(sid);
            target_ids.push(sid);
            while i < n && masked[i] {
                target_ids.push(tokens[i]);
                i += 1;
            }
            sentinel += 1;
        } else {
            input_ids.push(tokens[i]);
            i += 1;
        }
    }
    // Final sentinel terminates the target sequence.
    target_ids.push(cfg.sentinel_start_id - sentinel);

    SslExample {
        input_ids,
        labels: Vec::new(), // supervision is in target_ids (seq2seq)
        target_ids: Some(target_ids),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_lm_shifts_by_one() {
        let tokens = vec![10, 20, 30, 40];
        let ex = build_causal_lm(&tokens);
        assert_eq!(ex.input_ids, vec![10, 20, 30]);
        assert_eq!(ex.labels, vec![Some(20), Some(30), Some(40)]);
        assert_eq!(ex.loss_positions(), 3);
        assert!(ex.target_ids.is_none());
    }

    #[test]
    fn causal_lm_handles_short_sequences() {
        assert_eq!(build_causal_lm(&[]).input_ids.len(), 0);
        assert_eq!(build_causal_lm(&[5]).labels.len(), 0);
    }

    #[test]
    fn labels_with_ignore_index() {
        let ex = SslExample {
            input_ids: vec![1, 2, 3],
            labels: vec![Some(7), None, Some(9)],
            target_ids: None,
        };
        assert_eq!(ex.labels_with_ignore_index(-100), vec![7, -100, 9]);
    }

    #[test]
    fn masked_lm_selects_and_labels() {
        let tokens: Vec<TokenId> = (0..1000).collect();
        let cfg = MlmConfig {
            mask_prob: 0.15,
            mask_token_id: 999_999,
            vocab_size: 1000,
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(7);
        let ex = build_masked_lm(&tokens, &cfg, &mut rng);

        // Roughly 15% of positions are selected for prediction.
        let selected = ex.loss_positions();
        assert!(selected > 100 && selected < 200, "selected={}", selected);

        // Every selected position's label is the ORIGINAL token; unselected are None.
        for (i, label) in ex.labels.iter().enumerate() {
            match label {
                Some(t) => assert_eq!(*t, tokens[i]),
                None => assert_eq!(ex.input_ids[i], tokens[i]), // untouched
            }
        }
        // At least some inputs were replaced with [MASK].
        assert!(ex.input_ids.iter().any(|&t| t == 999_999));
    }

    #[test]
    fn masked_lm_is_deterministic_under_seed() {
        let tokens: Vec<TokenId> = (0..200).collect();
        let cfg = MlmConfig::default();
        let a = build_example(&SslObjective::MaskedLm(cfg.clone()), &tokens, 42);
        let b = build_example(&SslObjective::MaskedLm(cfg), &tokens, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn span_corruption_reconstructs_original() {
        let tokens: Vec<TokenId> = (1..=40).collect();
        let cfg = SpanConfig {
            mask_prob: 0.25,
            mean_span_len: 3,
            sentinel_start_id: 100_000,
        };
        let mut rng = StdRng::seed_from_u64(3);
        let ex = build_span_corruption(&tokens, &cfg, &mut rng);
        let target = ex.target_ids.clone().unwrap();

        // Sentinels appear in the input, one per masked span.
        let input_sentinels = ex.input_ids.iter().filter(|&&t| t >= 99_000).count();
        let target_sentinels = target.iter().filter(|&&t| t >= 99_000).count();
        assert!(input_sentinels >= 1);
        // Target has one extra (terminal) sentinel beyond the spans.
        assert_eq!(target_sentinels, input_sentinels + 1);

        // Reconstruct: weave the non-sentinel input tokens with the masked spans
        // recovered from the target back into the original sequence.
        let is_sentinel = |t: TokenId| t > cfg.sentinel_start_id - 50;
        // Map sentinel -> the span tokens following it in the target.
        let mut spans: std::collections::HashMap<TokenId, Vec<TokenId>> = Default::default();
        let mut cur: Option<TokenId> = None;
        for &t in &target {
            if is_sentinel(t) {
                cur = Some(t);
                spans.entry(t).or_default();
            } else if let Some(s) = cur {
                spans.get_mut(&s).unwrap().push(t);
            }
        }
        let mut recon = Vec::new();
        for &t in &ex.input_ids {
            if is_sentinel(t) {
                recon.extend(spans.get(&t).cloned().unwrap_or_default());
            } else {
                recon.push(t);
            }
        }
        assert_eq!(recon, tokens);
    }

    #[test]
    fn span_corruption_handles_empty() {
        let ex = build_span_corruption(&[], &SpanConfig::default(), &mut StdRng::seed_from_u64(1));
        assert!(ex.input_ids.is_empty());
        assert_eq!(ex.target_ids.unwrap().len(), 1); // a lone terminal sentinel
    }
}

// ------------------------------------------------------------------------------
// END OF FILE: self_supervised.rs
// REPO PATH:   /swiftllm/crates/swiftllm-training/src/self_supervised.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
