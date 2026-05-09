// ==============================================================================
// PROJECT:   SWIFTLLM
// FILE:      streaming.rs
// PATH:      /crates/swiftllm-server/src/streaming.rs
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

//! Streaming utilities for Server-Sent Events (SSE) responses
//!
//! Provides SSE formatting for OpenAI-compatible streaming chat/completion responses.

use futures::stream::Stream;
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

// ------------------------------------------------------------------------------
// END OF FILE: streaming.rs
// REPO PATH:   /swiftllm/crates/swiftllm-server/src/streaming.rs
// (c) 2026 SWIFTLLM | Apache 2.0 License
// ------------------------------------------------------------------------------
