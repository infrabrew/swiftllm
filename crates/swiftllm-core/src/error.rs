//! Error types for SwiftLLM

use thiserror::Error;

/// Result type alias for SwiftLLM operations
pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for SwiftLLM
#[derive(Error, Debug)]
pub enum Error {
    /// Model loading error
    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    /// Model not found
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Invalid model configuration
    #[error("Invalid model configuration: {0}")]
    InvalidConfig(String),

    /// Unsupported model architecture
    #[error("Unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),

    /// Memory allocation error
    #[error("Memory allocation failed: {0}")]
    MemoryAllocation(String),

    /// Out of memory
    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    /// Block allocation error
    #[error("Block allocation failed: {0}")]
    BlockAllocation(String),

    /// KV cache error
    #[error("KV cache error: {0}")]
    KvCache(String),

    /// Scheduler error
    #[error("Scheduler error: {0}")]
    Scheduler(String),

    /// Request queue full
    #[error("Request queue is full")]
    QueueFull,

    /// Request not found
    #[error("Request not found: {0}")]
    RequestNotFound(String),

    /// Request cancelled
    #[error("Request was cancelled: {0}")]
    RequestCancelled(String),

    /// Request timeout
    #[error("Request timed out: {0}")]
    RequestTimeout(String),

    /// Tokenizer error
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    /// CUDA error
    #[error("CUDA error: {0}")]
    Cuda(String),

    /// Device error
    #[error("Device error: {0}")]
    Device(String),

    /// Tensor operation error
    #[error("Tensor operation error: {0}")]
    Tensor(String),

    /// Shape mismatch
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Data type mismatch
    #[error("Data type mismatch: expected {expected}, got {actual}")]
    DTypeMismatch { expected: String, actual: String },

    /// Execution error
    #[error("Execution error: {0}")]
    Execution(String),

    /// Sampling error
    #[error("Sampling error: {0}")]
    Sampling(String),

    /// Invalid sampling parameters
    #[error("Invalid sampling parameters: {0}")]
    InvalidSamplingParams(String),

    /// Speculative decoding error
    #[error("Speculative decoding error: {0}")]
    SpeculativeDecoding(String),

    /// Tensor parallelism error
    #[error("Tensor parallelism error: {0}")]
    TensorParallel(String),

    /// Communication error
    #[error("Communication error: {0}")]
    Communication(String),

    /// File I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),

    /// Not implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Other error with context
    #[error("{context}: {source}")]
    WithContext {
        context: String,
        source: Box<Error>,
    },
}

impl Error {
    /// Add context to an error
    pub fn with_context(self, context: impl Into<String>) -> Self {
        Error::WithContext {
            context: context.into(),
            source: Box::new(self),
        }
    }

    /// Create an internal error
    pub fn internal(msg: impl Into<String>) -> Self {
        Error::Internal(msg.into())
    }

    /// Create a not implemented error
    pub fn not_implemented(feature: impl Into<String>) -> Self {
        Error::NotImplemented(feature.into())
    }

    /// Check if this is an out-of-memory error
    pub fn is_oom(&self) -> bool {
        matches!(self, Error::OutOfMemory(_))
    }

    /// Check if this is a cancellation
    pub fn is_cancelled(&self) -> bool {
        matches!(self, Error::RequestCancelled(_))
    }

    /// Check if this is a timeout
    pub fn is_timeout(&self) -> bool {
        matches!(self, Error::RequestTimeout(_))
    }
}

/// Extension trait for adding context to Results
pub trait ResultExt<T> {
    /// Add context to an error result
    fn context(self, context: impl Into<String>) -> Result<T>;

    /// Add lazy context to an error result
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;
}

impl<T> ResultExt<T> for Result<T> {
    fn context(self, context: impl Into<String>) -> Result<T> {
        self.map_err(|e| e.with_context(context))
    }

    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| e.with_context(f()))
    }
}

/// Extension trait for converting other error types
pub trait IntoError<T> {
    fn into_error(self, msg: impl Into<String>) -> Result<T>;
}

impl<T, E: std::error::Error> IntoError<T> for std::result::Result<T, E> {
    fn into_error(self, msg: impl Into<String>) -> Result<T> {
        self.map_err(|e| Error::Internal(format!("{}: {}", msg.into(), e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context() {
        let err = Error::ModelLoad("file not found".to_string());
        let with_ctx = err.with_context("loading llama model");

        match with_ctx {
            Error::WithContext { context, source } => {
                assert_eq!(context, "loading llama model");
                assert!(matches!(*source, Error::ModelLoad(_)));
            }
            _ => panic!("Expected WithContext error"),
        }
    }

    #[test]
    fn test_error_checks() {
        assert!(Error::OutOfMemory("GPU".to_string()).is_oom());
        assert!(Error::RequestCancelled("user".to_string()).is_cancelled());
        assert!(Error::RequestTimeout("5s".to_string()).is_timeout());
    }
}
