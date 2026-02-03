//! Streaming utilities for SSE responses

use futures::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};

/// SSE stream wrapper
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
