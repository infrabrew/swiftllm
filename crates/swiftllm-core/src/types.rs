//! Core types for SwiftLLM inference engine

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Unique identifier for a request
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(Uuid);

impl RequestId {
    /// Create a new random request ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create from a UUID string
    pub fn from_str(s: &str) -> Result<Self, uuid::Error> {
        Ok(Self(Uuid::parse_str(s)?))
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a sequence within a request
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SequenceId(u64);

impl SequenceId {
    /// Create a new sequence ID
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Get the raw ID value
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl Default for SequenceId {
    fn default() -> Self {
        Self::new()
    }
}

/// Token ID type
pub type TokenId = u32;

/// A token with optional logprob information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    /// Token ID
    pub id: TokenId,

    /// Decoded text (if available)
    pub text: Option<String>,

    /// Log probability (if available)
    pub logprob: Option<f32>,

    /// Top log probabilities (if requested)
    pub top_logprobs: Option<Vec<(TokenId, f32)>>,
}

impl Token {
    /// Create a new token
    pub fn new(id: TokenId) -> Self {
        Self {
            id,
            text: None,
            logprob: None,
            top_logprobs: None,
        }
    }

    /// Create a token with text
    pub fn with_text(id: TokenId, text: String) -> Self {
        Self {
            id,
            text: Some(text),
            logprob: None,
            top_logprobs: None,
        }
    }
}

/// Request status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RequestStatus {
    /// Request is waiting in queue
    Pending,
    /// Request is being processed
    Running,
    /// Request has been preempted
    Preempted,
    /// Request completed successfully
    Completed,
    /// Request was cancelled
    Cancelled,
    /// Request failed with an error
    Failed,
}

impl RequestStatus {
    /// Check if the request is finished (completed, cancelled, or failed)
    pub fn is_finished(&self) -> bool {
        matches!(
            self,
            RequestStatus::Completed | RequestStatus::Cancelled | RequestStatus::Failed
        )
    }
}

/// Finish reason for a sequence
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Reached stop token
    Stop,
    /// Reached maximum length
    Length,
    /// Matched stop sequence
    StopSequence,
    /// Cancelled by user
    Cancelled,
    /// Error occurred
    Error,
}

/// An inference request
#[derive(Debug, Clone)]
pub struct Request {
    /// Unique request ID
    pub id: RequestId,

    /// Input prompt (text)
    pub prompt: Option<String>,

    /// Input prompt (tokenized)
    pub prompt_token_ids: Vec<TokenId>,

    /// Sampling parameters
    pub sampling_params: crate::config::SamplingConfig,

    /// Request arrival time
    pub arrival_time: Instant,

    /// Request status
    pub status: RequestStatus,

    /// Priority (higher = more urgent)
    pub priority: i32,

    /// Arbitrary metadata
    pub metadata: HashMap<String, String>,

    /// Multi-modal inputs (if any)
    pub multi_modal_data: Option<MultiModalData>,
}

impl Request {
    /// Create a new request from tokenized input
    pub fn new(prompt_token_ids: Vec<TokenId>) -> Self {
        Self {
            id: RequestId::new(),
            prompt: None,
            prompt_token_ids,
            sampling_params: crate::config::SamplingConfig::default(),
            arrival_time: Instant::now(),
            status: RequestStatus::Pending,
            priority: 0,
            metadata: HashMap::new(),
            multi_modal_data: None,
        }
    }

    /// Create a new request from text
    pub fn from_text(prompt: String, prompt_token_ids: Vec<TokenId>) -> Self {
        Self {
            id: RequestId::new(),
            prompt: Some(prompt),
            prompt_token_ids,
            sampling_params: crate::config::SamplingConfig::default(),
            arrival_time: Instant::now(),
            status: RequestStatus::Pending,
            priority: 0,
            metadata: HashMap::new(),
            multi_modal_data: None,
        }
    }

    /// Set sampling parameters
    pub fn with_sampling_params(mut self, params: crate::config::SamplingConfig) -> Self {
        self.sampling_params = params;
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Get the prompt length
    pub fn prompt_len(&self) -> usize {
        self.prompt_token_ids.len()
    }

    /// Get the time elapsed since request arrival
    pub fn elapsed(&self) -> Duration {
        self.arrival_time.elapsed()
    }
}

/// Multi-modal data for vision/audio models
#[derive(Debug, Clone)]
pub struct MultiModalData {
    /// Image data (list of images)
    pub images: Vec<ImageData>,

    /// Audio data (list of audio clips)
    pub audio: Vec<AudioData>,
}

/// Image data for vision models
#[derive(Debug, Clone)]
pub struct ImageData {
    /// Raw image bytes
    pub data: Vec<u8>,

    /// Image format (e.g., "png", "jpeg")
    pub format: String,

    /// Image width
    pub width: u32,

    /// Image height
    pub height: u32,
}

/// Audio data for audio models
#[derive(Debug, Clone)]
pub struct AudioData {
    /// Raw audio bytes
    pub data: Vec<u8>,

    /// Audio format (e.g., "wav", "mp3")
    pub format: String,

    /// Sample rate
    pub sample_rate: u32,
}

/// A sequence group (beam search or parallel sampling)
#[derive(Debug)]
pub struct SequenceGroup {
    /// Request ID this group belongs to
    pub request_id: RequestId,

    /// Sequences in this group
    pub sequences: Vec<Sequence>,

    /// Sampling parameters
    pub sampling_params: crate::config::SamplingConfig,

    /// Arrival time
    pub arrival_time: Instant,

    /// Current state
    pub state: SequenceGroupState,
}

impl SequenceGroup {
    /// Create a new sequence group
    pub fn new(request: &Request) -> Self {
        let sequence = Sequence::new(request.prompt_token_ids.clone());
        Self {
            request_id: request.id,
            sequences: vec![sequence],
            sampling_params: request.sampling_params.clone(),
            arrival_time: request.arrival_time,
            state: SequenceGroupState::Prefill,
        }
    }

    /// Get the number of unfinished sequences
    pub fn num_unfinished(&self) -> usize {
        self.sequences
            .iter()
            .filter(|s| !s.is_finished())
            .count()
    }

    /// Check if all sequences are finished
    pub fn is_finished(&self) -> bool {
        self.sequences.iter().all(|s| s.is_finished())
    }

    /// Get the prompt length
    pub fn prompt_len(&self) -> usize {
        self.sequences
            .first()
            .map(|s| s.prompt_len)
            .unwrap_or(0)
    }

    /// Get the total number of tokens across all sequences
    pub fn total_tokens(&self) -> usize {
        self.sequences.iter().map(|s| s.len()).sum()
    }
}

/// Sequence group state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceGroupState {
    /// Processing prefill (prompt)
    Prefill,
    /// Processing decode (generation)
    Decode,
    /// Finished
    Finished,
}

/// A single sequence (one beam or sample)
#[derive(Debug)]
pub struct Sequence {
    /// Unique sequence ID
    pub id: SequenceId,

    /// Token IDs (prompt + generated)
    pub token_ids: Vec<TokenId>,

    /// Length of the original prompt
    pub prompt_len: usize,

    /// Output tokens (with metadata)
    pub output_tokens: Vec<Token>,

    /// Cumulative log probability
    pub cumulative_logprob: f32,

    /// Block table (for PagedAttention)
    pub block_table: Vec<usize>,

    /// Physical token count
    pub physical_token_count: usize,

    /// Sequence status
    pub status: SequenceStatus,

    /// Finish reason (if finished)
    pub finish_reason: Option<FinishReason>,
}

impl Sequence {
    /// Create a new sequence from prompt tokens
    pub fn new(prompt_tokens: Vec<TokenId>) -> Self {
        let prompt_len = prompt_tokens.len();
        Self {
            id: SequenceId::new(),
            token_ids: prompt_tokens,
            prompt_len,
            output_tokens: Vec::new(),
            cumulative_logprob: 0.0,
            block_table: Vec::new(),
            physical_token_count: 0,
            status: SequenceStatus::Running,
            finish_reason: None,
        }
    }

    /// Get the total length (prompt + generated)
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    /// Check if the sequence is empty
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }

    /// Get the number of generated tokens
    pub fn num_generated(&self) -> usize {
        self.token_ids.len().saturating_sub(self.prompt_len)
    }

    /// Append a token
    pub fn append_token(&mut self, token: Token) {
        self.token_ids.push(token.id);
        if let Some(logprob) = token.logprob {
            self.cumulative_logprob += logprob;
        }
        self.output_tokens.push(token);
    }

    /// Check if the sequence is finished
    pub fn is_finished(&self) -> bool {
        self.status == SequenceStatus::Finished
    }

    /// Mark the sequence as finished
    pub fn finish(&mut self, reason: FinishReason) {
        self.status = SequenceStatus::Finished;
        self.finish_reason = Some(reason);
    }

    /// Fork this sequence (for beam search)
    pub fn fork(&self) -> Self {
        Self {
            id: SequenceId::new(),
            token_ids: self.token_ids.clone(),
            prompt_len: self.prompt_len,
            output_tokens: self.output_tokens.clone(),
            cumulative_logprob: self.cumulative_logprob,
            block_table: self.block_table.clone(),
            physical_token_count: self.physical_token_count,
            status: self.status,
            finish_reason: self.finish_reason,
        }
    }
}

/// Sequence status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    /// Running
    Running,
    /// Waiting (preempted)
    Waiting,
    /// Finished
    Finished,
}

/// Output for a single request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestOutput {
    /// Request ID
    pub request_id: RequestId,

    /// Generated outputs (one per sequence)
    pub outputs: Vec<GenerationOutput>,

    /// Whether the request is finished
    pub finished: bool,

    /// Prompt (if echo is enabled)
    pub prompt: Option<String>,

    /// Prompt token IDs (if echo is enabled)
    pub prompt_token_ids: Option<Vec<TokenId>>,

    /// Prompt logprobs (if requested)
    pub prompt_logprobs: Option<Vec<Option<HashMap<TokenId, f32>>>>,

    /// Metrics
    pub metrics: Option<RequestMetrics>,
}

/// Output for a single generated sequence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationOutput {
    /// Sequence index
    pub index: usize,

    /// Generated text
    pub text: String,

    /// Generated token IDs
    pub token_ids: Vec<TokenId>,

    /// Cumulative log probability
    pub cumulative_logprob: f32,

    /// Token-level log probabilities
    pub logprobs: Option<Vec<TokenLogprobs>>,

    /// Finish reason
    pub finish_reason: Option<FinishReason>,

    /// Stop reason (matched stop string)
    pub stop_reason: Option<String>,
}

/// Token-level log probabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLogprobs {
    /// Token ID
    pub token_id: TokenId,

    /// Log probability
    pub logprob: f32,

    /// Top alternatives
    pub top_logprobs: Option<HashMap<TokenId, f32>>,
}

/// Request metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetrics {
    /// Time to first token (seconds)
    pub time_to_first_token: f64,

    /// Total time (seconds)
    pub total_time: f64,

    /// Number of prompt tokens
    pub prompt_tokens: usize,

    /// Number of generated tokens
    pub generated_tokens: usize,

    /// Tokens per second
    pub tokens_per_second: f64,
}

/// Scheduler output (what to execute)
#[derive(Debug, Default)]
pub struct SchedulerOutput {
    /// Sequence groups to run
    pub scheduled_groups: Vec<ScheduledSequenceGroup>,

    /// Total number of tokens in this batch
    pub num_batched_tokens: usize,

    /// Blocks to swap in (from CPU to GPU)
    pub blocks_to_swap_in: HashMap<usize, usize>,

    /// Blocks to swap out (from GPU to CPU)
    pub blocks_to_swap_out: HashMap<usize, usize>,

    /// Blocks to copy (CoW)
    pub blocks_to_copy: HashMap<usize, Vec<usize>>,

    /// Preempted sequence groups
    pub preempted: Vec<SequenceGroup>,

    /// Ignored sequence groups (couldn't fit)
    pub ignored: Vec<SequenceGroup>,
}

/// A scheduled sequence group with metadata
#[derive(Debug)]
pub struct ScheduledSequenceGroup {
    /// The sequence group
    pub seq_group: SequenceGroup,

    /// Number of tokens to compute for each sequence
    pub token_chunk_size: usize,
}

/// Batch for model execution
#[derive(Debug)]
pub struct ExecutionBatch {
    /// Input token IDs (flattened)
    pub input_tokens: Vec<TokenId>,

    /// Position IDs
    pub positions: Vec<usize>,

    /// Sequence lengths
    pub seq_lens: Vec<usize>,

    /// Block tables
    pub block_tables: Vec<Vec<usize>>,

    /// Context lengths
    pub context_lens: Vec<usize>,

    /// Slot mappings for PagedAttention
    pub slot_mapping: Vec<usize>,

    /// Whether this is a prefill batch
    pub is_prefill: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_id() {
        let id1 = RequestId::new();
        let id2 = RequestId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_sequence_id() {
        let id1 = SequenceId::new();
        let id2 = SequenceId::new();
        assert_ne!(id1.as_u64(), id2.as_u64());
    }

    #[test]
    fn test_sequence_operations() {
        let prompt = vec![1, 2, 3, 4, 5];
        let mut seq = Sequence::new(prompt);

        assert_eq!(seq.len(), 5);
        assert_eq!(seq.prompt_len, 5);
        assert_eq!(seq.num_generated(), 0);

        seq.append_token(Token::new(6));
        seq.append_token(Token::new(7));

        assert_eq!(seq.len(), 7);
        assert_eq!(seq.num_generated(), 2);
        assert!(!seq.is_finished());

        seq.finish(FinishReason::Stop);
        assert!(seq.is_finished());
    }

    #[test]
    fn test_request_status() {
        assert!(!RequestStatus::Pending.is_finished());
        assert!(!RequestStatus::Running.is_finished());
        assert!(RequestStatus::Completed.is_finished());
        assert!(RequestStatus::Failed.is_finished());
    }
}
