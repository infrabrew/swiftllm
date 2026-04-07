//! Streaming utilities for Server-Sent Events (SSE) responses
//!
//! Provides SSE formatting for OpenAI-compatible streaming chat/completion responses.

use axum::response::sse::{Event, Sse};
use futures::stream::Stream;
use std::convert::Infallible;
use std::pin::Pin;
use std::task::{Context, Poll};

/// SSE stream wrapper that formats items as SSE data events
pub struct SseStream<S> {
    inner: S,
}

impl<S> SseStream<S> {
    /// Create a new SSE stream
    pub fn new(stream: S) -> Self {
        Self { inner: stream }
    }
}

impl<S, T, E> Stream for SseStream<S>
where
    S: Stream<Item = Result<T, E>> + Unpin,
{
    type Item = Result<T, E>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

/// Create a streaming chat completion chunk in OpenAI format
pub fn chat_completion_chunk(
    id: &str,
    model: &str,
    content: &str,
    finish_reason: Option<&str>,
) -> String {
    let chunk = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": { "content": content },
            "finish_reason": finish_reason,
        }]
    });
    chunk.to_string()
}

/// Create the final [DONE] sentinel for SSE streaming
pub fn stream_done() -> String {
    "[DONE]".to_string()
}

/// Format a string as an SSE data event
pub fn sse_event(data: &str) -> String {
    format!("data: {}\n\n", data)
}
