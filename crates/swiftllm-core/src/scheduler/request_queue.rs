//! Request Queue Management
//!
//! This module provides a thread-safe request queue with support for
//! timeouts, priorities, and various queuing policies.

use crate::error::{Error, Result};
use crate::types::{Request, RequestId, RequestStatus};
use parking_lot::{Condvar, Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for the request queue
#[derive(Debug, Clone)]
pub struct RequestQueueConfig {
    /// Maximum queue size (0 = unlimited)
    pub max_size: usize,

    /// Request timeout in seconds (0 = no timeout)
    pub timeout_secs: u64,

    /// Enable priority scheduling
    pub enable_priority: bool,

    /// Maximum wait time before request is dropped
    pub max_wait_secs: u64,
}

impl Default for RequestQueueConfig {
    fn default() -> Self {
        Self {
            max_size: 10000,
            timeout_secs: 300,
            enable_priority: false,
            max_wait_secs: 600,
        }
    }
}

/// Entry in the request queue
#[derive(Debug)]
struct QueueEntry {
    /// The request
    request: Request,

    /// Time added to queue
    enqueue_time: Instant,
}

impl QueueEntry {
    fn new(request: Request) -> Self {
        Self {
            request,
            enqueue_time: Instant::now(),
        }
    }

    fn wait_time(&self) -> Duration {
        self.enqueue_time.elapsed()
    }
}

/// Thread-safe request queue
pub struct RequestQueue {
    /// Configuration
    config: RequestQueueConfig,

    /// Queue storage
    queue: RwLock<VecDeque<QueueEntry>>,

    /// High priority queue
    priority_queue: RwLock<VecDeque<QueueEntry>>,

    /// Request lookup (ID -> position)
    request_lookup: RwLock<HashMap<RequestId, QueuePosition>>,

    /// Current queue size
    size: AtomicUsize,

    /// Whether queue is accepting new requests
    accepting: AtomicBool,

    /// Condition variable for waiting
    condvar: Condvar,

    /// Mutex for condition variable
    condvar_mutex: Mutex<()>,
}

/// Position of a request in the queue
#[derive(Debug, Clone, Copy)]
enum QueuePosition {
    Normal(usize),
    Priority(usize),
}

impl RequestQueue {
    /// Create a new request queue
    pub fn new(config: RequestQueueConfig) -> Self {
        Self {
            config,
            queue: RwLock::new(VecDeque::new()),
            priority_queue: RwLock::new(VecDeque::new()),
            request_lookup: RwLock::new(HashMap::new()),
            size: AtomicUsize::new(0),
            accepting: AtomicBool::new(true),
            condvar: Condvar::new(),
            condvar_mutex: Mutex::new(()),
        }
    }

    /// Enqueue a request
    pub fn enqueue(&self, request: Request) -> Result<()> {
        if !self.accepting.load(Ordering::SeqCst) {
            return Err(Error::QueueFull);
        }

        let current_size = self.size.load(Ordering::SeqCst);
        if self.config.max_size > 0 && current_size >= self.config.max_size {
            return Err(Error::QueueFull);
        }

        let request_id = request.id;
        let priority = request.priority;
        let entry = QueueEntry::new(request);

        if self.config.enable_priority && priority > 0 {
            // High priority request
            let mut pqueue = self.priority_queue.write();
            let pos = pqueue.len();
            pqueue.push_back(entry);
            self.request_lookup
                .write()
                .insert(request_id, QueuePosition::Priority(pos));
        } else {
            // Normal priority
            let mut queue = self.queue.write();
            let pos = queue.len();
            queue.push_back(entry);
            self.request_lookup
                .write()
                .insert(request_id, QueuePosition::Normal(pos));
        }

        self.size.fetch_add(1, Ordering::SeqCst);

        // Notify waiters
        self.condvar.notify_one();

        Ok(())
    }

    /// Dequeue the next request
    pub fn dequeue(&self) -> Option<Request> {
        // Try priority queue first
        {
            let mut pqueue = self.priority_queue.write();
            if let Some(entry) = pqueue.pop_front() {
                self.request_lookup.write().remove(&entry.request.id);
                self.size.fetch_sub(1, Ordering::SeqCst);
                return Some(entry.request);
            }
        }

        // Then normal queue
        {
            let mut queue = self.queue.write();
            if let Some(entry) = queue.pop_front() {
                self.request_lookup.write().remove(&entry.request.id);
                self.size.fetch_sub(1, Ordering::SeqCst);
                return Some(entry.request);
            }
        }

        None
    }

    /// Dequeue with timeout
    pub fn dequeue_timeout(&self, timeout: Duration) -> Option<Request> {
        let deadline = Instant::now() + timeout;

        loop {
            // Try to dequeue
            if let Some(request) = self.dequeue() {
                return Some(request);
            }

            // Wait for notification or timeout
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return None;
            }

            let mut guard = self.condvar_mutex.lock();
            let result = self.condvar.wait_for(&mut guard, remaining);

            if result.timed_out() {
                return self.dequeue();
            }
        }
    }

    /// Try to dequeue a batch of requests
    pub fn dequeue_batch(&self, max_count: usize) -> Vec<Request> {
        let mut batch = Vec::with_capacity(max_count);

        // Priority requests first
        {
            let mut pqueue = self.priority_queue.write();
            while batch.len() < max_count {
                if let Some(entry) = pqueue.pop_front() {
                    self.request_lookup.write().remove(&entry.request.id);
                    batch.push(entry.request);
                } else {
                    break;
                }
            }
        }

        // Then normal requests
        {
            let mut queue = self.queue.write();
            while batch.len() < max_count {
                if let Some(entry) = queue.pop_front() {
                    self.request_lookup.write().remove(&entry.request.id);
                    batch.push(entry.request);
                } else {
                    break;
                }
            }
        }

        self.size.fetch_sub(batch.len(), Ordering::SeqCst);
        batch
    }

    /// Remove a specific request from the queue
    pub fn remove(&self, request_id: RequestId) -> Option<Request> {
        let position = self.request_lookup.write().remove(&request_id)?;

        match position {
            QueuePosition::Priority(pos) => {
                let mut pqueue = self.priority_queue.write();
                if pos < pqueue.len() {
                    let entry = pqueue.remove(pos)?;
                    self.size.fetch_sub(1, Ordering::SeqCst);
                    return Some(entry.request);
                }
            }
            QueuePosition::Normal(pos) => {
                let mut queue = self.queue.write();
                if pos < queue.len() {
                    let entry = queue.remove(pos)?;
                    self.size.fetch_sub(1, Ordering::SeqCst);
                    return Some(entry.request);
                }
            }
        }

        None
    }

    /// Check if a request is in the queue
    pub fn contains(&self, request_id: RequestId) -> bool {
        self.request_lookup.read().contains_key(&request_id)
    }

    /// Get the current queue size
    pub fn len(&self) -> usize {
        self.size.load(Ordering::SeqCst)
    }

    /// Check if the queue is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the number of priority requests
    pub fn priority_len(&self) -> usize {
        self.priority_queue.read().len()
    }

    /// Stop accepting new requests
    pub fn stop_accepting(&self) {
        self.accepting.store(false, Ordering::SeqCst);
    }

    /// Start accepting new requests
    pub fn start_accepting(&self) {
        self.accepting.store(true, Ordering::SeqCst);
    }

    /// Check if accepting requests
    pub fn is_accepting(&self) -> bool {
        self.accepting.load(Ordering::SeqCst)
    }

    /// Drain timed out requests
    pub fn drain_timeouts(&self) -> Vec<Request> {
        let timeout = Duration::from_secs(self.config.max_wait_secs);
        let mut timed_out = Vec::new();

        // Check priority queue
        {
            let mut pqueue = self.priority_queue.write();
            let mut i = 0;
            while i < pqueue.len() {
                if pqueue[i].wait_time() > timeout {
                    if let Some(entry) = pqueue.remove(i) {
                        self.request_lookup.write().remove(&entry.request.id);
                        timed_out.push(entry.request);
                    }
                } else {
                    i += 1;
                }
            }
        }

        // Check normal queue
        {
            let mut queue = self.queue.write();
            let mut i = 0;
            while i < queue.len() {
                if queue[i].wait_time() > timeout {
                    if let Some(entry) = queue.remove(i) {
                        self.request_lookup.write().remove(&entry.request.id);
                        timed_out.push(entry.request);
                    }
                } else {
                    i += 1;
                }
            }
        }

        self.size.fetch_sub(timed_out.len(), Ordering::SeqCst);
        timed_out
    }

    /// Clear the queue
    pub fn clear(&self) {
        self.queue.write().clear();
        self.priority_queue.write().clear();
        self.request_lookup.write().clear();
        self.size.store(0, Ordering::SeqCst);
    }

    /// Get queue statistics
    pub fn stats(&self) -> QueueStats {
        let queue = self.queue.read();
        let pqueue = self.priority_queue.read();

        let total = queue.len() + pqueue.len();
        let priority = pqueue.len();

        let avg_wait_time = if total > 0 {
            let total_wait: Duration = queue
                .iter()
                .chain(pqueue.iter())
                .map(|e| e.wait_time())
                .sum();
            total_wait / total as u32
        } else {
            Duration::ZERO
        };

        let max_wait_time = queue
            .iter()
            .chain(pqueue.iter())
            .map(|e| e.wait_time())
            .max()
            .unwrap_or(Duration::ZERO);

        QueueStats {
            total_size: total,
            priority_size: priority,
            normal_size: queue.len(),
            avg_wait_time,
            max_wait_time,
            is_accepting: self.is_accepting(),
        }
    }
}

/// Queue statistics
#[derive(Debug, Clone)]
pub struct QueueStats {
    /// Total queue size
    pub total_size: usize,

    /// Priority queue size
    pub priority_size: usize,

    /// Normal queue size
    pub normal_size: usize,

    /// Average wait time
    pub avg_wait_time: Duration,

    /// Maximum wait time
    pub max_wait_time: Duration,

    /// Whether accepting new requests
    pub is_accepting: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enqueue_dequeue() {
        let queue = RequestQueue::new(RequestQueueConfig::default());

        let r1 = Request::new(vec![1, 2, 3]);
        let r2 = Request::new(vec![4, 5, 6]);
        let r1_id = r1.id;

        queue.enqueue(r1).unwrap();
        queue.enqueue(r2).unwrap();

        assert_eq!(queue.len(), 2);

        let dequeued = queue.dequeue().unwrap();
        assert_eq!(dequeued.id, r1_id);
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_priority_queue() {
        let config = RequestQueueConfig {
            enable_priority: true,
            ..Default::default()
        };
        let queue = RequestQueue::new(config);

        let r1 = Request::new(vec![1, 2, 3]);
        let mut r2 = Request::new(vec![4, 5, 6]);
        r2.priority = 1; // High priority

        let r2_id = r2.id;

        queue.enqueue(r1).unwrap();
        queue.enqueue(r2).unwrap();

        // Priority request should come first
        let dequeued = queue.dequeue().unwrap();
        assert_eq!(dequeued.id, r2_id);
    }

    #[test]
    fn test_remove_request() {
        let queue = RequestQueue::new(RequestQueueConfig::default());

        let r1 = Request::new(vec![1, 2, 3]);
        let r1_id = r1.id;

        queue.enqueue(r1).unwrap();
        assert!(queue.contains(r1_id));

        let removed = queue.remove(r1_id).unwrap();
        assert_eq!(removed.id, r1_id);
        assert!(!queue.contains(r1_id));
    }

    #[test]
    fn test_queue_full() {
        let config = RequestQueueConfig {
            max_size: 2,
            ..Default::default()
        };
        let queue = RequestQueue::new(config);

        queue.enqueue(Request::new(vec![1])).unwrap();
        queue.enqueue(Request::new(vec![2])).unwrap();

        let result = queue.enqueue(Request::new(vec![3]));
        assert!(result.is_err());
    }

    #[test]
    fn test_dequeue_batch() {
        let queue = RequestQueue::new(RequestQueueConfig::default());

        for i in 0..10 {
            queue.enqueue(Request::new(vec![i as u32])).unwrap();
        }

        let batch = queue.dequeue_batch(5);
        assert_eq!(batch.len(), 5);
        assert_eq!(queue.len(), 5);
    }

    #[test]
    fn test_stop_accepting() {
        let queue = RequestQueue::new(RequestQueueConfig::default());

        queue.enqueue(Request::new(vec![1])).unwrap();
        queue.stop_accepting();

        let result = queue.enqueue(Request::new(vec![2]));
        assert!(result.is_err());

        queue.start_accepting();
        queue.enqueue(Request::new(vec![3])).unwrap();
    }
}
