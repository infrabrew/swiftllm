//! Training metrics and logging

use std::collections::VecDeque;
use std::time::Instant;

/// Training metrics tracker
#[derive(Debug)]
pub struct TrainingMetrics {
    /// Training losses (per step)
    train_losses: VecDeque<f64>,

    /// Validation losses (per eval)
    eval_losses: Vec<f64>,

    /// Learning rate history
    lr_history: Vec<f64>,

    /// Throughput samples (tokens/sec)
    throughput_samples: VecDeque<f64>,

    /// Step count
    step: usize,

    /// Epoch count
    epoch: usize,

    /// Training start time
    start_time: Instant,

    /// Tokens processed in current logging window
    tokens_in_window: usize,

    /// Window start time
    window_start: Instant,

    /// Maximum window size for rolling averages
    window_size: usize,

    /// Total tokens processed
    total_tokens: usize,
}

impl TrainingMetrics {
    /// Create a new metrics tracker
    pub fn new(window_size: usize) -> Self {
        let now = Instant::now();
        Self {
            train_losses: VecDeque::with_capacity(window_size),
            eval_losses: Vec::new(),
            lr_history: Vec::new(),
            throughput_samples: VecDeque::with_capacity(window_size),
            step: 0,
            epoch: 0,
            start_time: now,
            tokens_in_window: 0,
            window_start: now,
            window_size,
            total_tokens: 0,
        }
    }

    /// Record a training step
    pub fn record_step(&mut self, loss: f64, lr: f64, num_tokens: usize) {
        self.step += 1;
        self.total_tokens += num_tokens;

        if self.train_losses.len() >= self.window_size {
            self.train_losses.pop_front();
        }
        self.train_losses.push_back(loss);
        self.lr_history.push(lr);

        // Update throughput
        self.tokens_in_window += num_tokens;
        let elapsed = self.window_start.elapsed().as_secs_f64();
        if elapsed >= 1.0 {
            let tps = self.tokens_in_window as f64 / elapsed;
            if self.throughput_samples.len() >= self.window_size {
                self.throughput_samples.pop_front();
            }
            self.throughput_samples.push_back(tps);
            self.tokens_in_window = 0;
            self.window_start = Instant::now();
        }
    }

    /// Record an evaluation result
    pub fn record_eval(&mut self, loss: f64) {
        self.eval_losses.push(loss);
    }

    /// Advance to the next epoch
    pub fn next_epoch(&mut self) {
        self.epoch += 1;
    }

    /// Get current step
    pub fn step(&self) -> usize { self.step }

    /// Get current epoch
    pub fn epoch(&self) -> usize { self.epoch }

    /// Get rolling average training loss
    pub fn avg_train_loss(&self) -> f64 {
        if self.train_losses.is_empty() { return 0.0; }
        self.train_losses.iter().sum::<f64>() / self.train_losses.len() as f64
    }

    /// Get the last training loss
    pub fn last_train_loss(&self) -> f64 {
        self.train_losses.back().copied().unwrap_or(0.0)
    }

    /// Get the last evaluation loss
    pub fn last_eval_loss(&self) -> Option<f64> {
        self.eval_losses.last().copied()
    }

    /// Get perplexity from loss
    pub fn perplexity(&self) -> f64 {
        self.avg_train_loss().exp()
    }

    /// Get eval perplexity
    pub fn eval_perplexity(&self) -> Option<f64> {
        self.last_eval_loss().map(|l| l.exp())
    }

    /// Get current learning rate
    pub fn current_lr(&self) -> f64 {
        self.lr_history.last().copied().unwrap_or(0.0)
    }

    /// Get average throughput (tokens/sec)
    pub fn avg_throughput(&self) -> f64 {
        if self.throughput_samples.is_empty() {
            if self.total_tokens == 0 { return 0.0; }
            let elapsed = self.start_time.elapsed().as_secs_f64();
            return self.total_tokens as f64 / elapsed.max(0.001);
        }
        self.throughput_samples.iter().sum::<f64>() / self.throughput_samples.len() as f64
    }

    /// Get total training time in seconds
    pub fn elapsed_secs(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Get total tokens processed
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Format a log line for the current step
    pub fn log_line(&self) -> String {
        let mut parts = vec![
            format!("step: {}", self.step),
            format!("epoch: {}", self.epoch),
            format!("loss: {:.4}", self.last_train_loss()),
            format!("ppl: {:.2}", self.perplexity()),
            format!("lr: {:.2e}", self.current_lr()),
            format!("tok/s: {:.0}", self.avg_throughput()),
        ];

        if let Some(eval_loss) = self.last_eval_loss() {
            parts.push(format!("eval_loss: {:.4}", eval_loss));
        }

        parts.join(" | ")
    }

    /// Get a summary snapshot
    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            step: self.step,
            epoch: self.epoch,
            train_loss: self.avg_train_loss(),
            eval_loss: self.last_eval_loss(),
            perplexity: self.perplexity(),
            learning_rate: self.current_lr(),
            throughput: self.avg_throughput(),
            total_tokens: self.total_tokens,
            elapsed_secs: self.elapsed_secs(),
        }
    }
}

/// Snapshot of metrics at a point in time
#[derive(Debug, Clone, serde::Serialize)]
pub struct MetricsSummary {
    /// Step number
    pub step: usize,
    /// Epoch number
    pub epoch: usize,
    /// Average training loss
    pub train_loss: f64,
    /// Evaluation loss (if available)
    pub eval_loss: Option<f64>,
    /// Perplexity
    pub perplexity: f64,
    /// Current learning rate
    pub learning_rate: f64,
    /// Tokens per second
    pub throughput: f64,
    /// Total tokens processed
    pub total_tokens: usize,
    /// Total elapsed time in seconds
    pub elapsed_secs: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_tracking() {
        let mut metrics = TrainingMetrics::new(100);

        metrics.record_step(2.5, 1e-4, 1024);
        metrics.record_step(2.3, 1e-4, 1024);
        metrics.record_step(2.1, 1e-4, 1024);

        assert_eq!(metrics.step(), 3);
        assert!((metrics.avg_train_loss() - 2.3).abs() < 0.01);
        assert_eq!(metrics.total_tokens(), 3072);
    }

    #[test]
    fn test_perplexity() {
        let mut metrics = TrainingMetrics::new(100);
        metrics.record_step(1.0, 1e-4, 100);

        // e^1.0 ≈ 2.718
        assert!((metrics.perplexity() - std::f64::consts::E).abs() < 0.01);
    }

    #[test]
    fn test_eval_metrics() {
        let mut metrics = TrainingMetrics::new(100);
        assert!(metrics.last_eval_loss().is_none());

        metrics.record_eval(1.5);
        assert_eq!(metrics.last_eval_loss(), Some(1.5));
    }

    #[test]
    fn test_log_line() {
        let mut metrics = TrainingMetrics::new(100);
        metrics.record_step(2.0, 5e-5, 512);
        let line = metrics.log_line();
        assert!(line.contains("step: 1"));
        assert!(line.contains("loss:"));
    }
}
