//! Optimizers and learning rate schedulers for training

use std::collections::HashMap;

/// Trait for optimizers
pub trait Optimizer: Send + Sync {
    /// Get the optimizer name
    fn name(&self) -> &str;

    /// Get the current learning rate
    fn learning_rate(&self) -> f64;

    /// Set the learning rate
    fn set_learning_rate(&mut self, lr: f64);

    /// Perform one optimization step on a parameter
    /// `param`: current parameter value (modified in-place)
    /// `grad`: gradient for this parameter
    /// `param_name`: identifier for per-parameter state
    fn step(&mut self, param: &mut [f32], grad: &[f32], param_name: &str);

    /// Reset optimizer state
    fn reset(&mut self);
}

/// AdamW optimizer
pub struct AdamW {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    step_count: u64,
    /// First moment estimates per parameter
    m: HashMap<String, Vec<f32>>,
    /// Second moment estimates per parameter
    v: HashMap<String, Vec<f32>>,
}

impl AdamW {
    /// Create a new AdamW optimizer
    pub fn new(lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step_count: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    /// Create with default hyperparameters
    pub fn default_for_lr(lr: f64) -> Self {
        Self::new(lr, 0.9, 0.999, 1e-8, 0.01)
    }
}

impl Optimizer for AdamW {
    fn name(&self) -> &str { "AdamW" }

    fn learning_rate(&self) -> f64 { self.lr }

    fn set_learning_rate(&mut self, lr: f64) { self.lr = lr; }

    fn step(&mut self, param: &mut [f32], grad: &[f32], param_name: &str) {
        assert_eq!(param.len(), grad.len());
        let n = param.len();

        self.step_count += 1;
        let t = self.step_count as f64;

        // Initialize state if needed
        let m = self.m.entry(param_name.to_string())
            .or_insert_with(|| vec![0.0; n]);
        let v = self.v.entry(param_name.to_string())
            .or_insert_with(|| vec![0.0; n]);

        // Resize if needed (shouldn't happen normally)
        if m.len() != n { m.resize(n, 0.0); }
        if v.len() != n { v.resize(n, 0.0); }

        // Bias correction — use powf to avoid i32 overflow for t > 2^31
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        let lr = self.lr;
        let beta1 = self.beta1 as f32;
        let beta2 = self.beta2 as f32;
        let eps = self.eps as f32;
        let wd = self.weight_decay as f32;

        for i in 0..n {
            let g = grad[i];

            // Update moments
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;

            // Bias-corrected estimates
            let m_hat = m[i] / bc1 as f32;
            let v_hat = v[i] / bc2 as f32;

            // AdamW: decoupled weight decay applied directly to parameters
            param[i] -= lr as f32 * (m_hat / (v_hat.sqrt() + eps) + wd * param[i]);
        }
    }

    fn reset(&mut self) {
        self.step_count = 0;
        self.m.clear();
        self.v.clear();
    }
}

/// SGD optimizer with momentum
pub struct SGD {
    lr: f64,
    momentum: f64,
    weight_decay: f64,
    nesterov: bool,
    /// Velocity per parameter
    velocity: HashMap<String, Vec<f32>>,
}

impl SGD {
    /// Create a new SGD optimizer
    pub fn new(lr: f64, momentum: f64, weight_decay: f64, nesterov: bool) -> Self {
        Self {
            lr,
            momentum,
            weight_decay,
            nesterov,
            velocity: HashMap::new(),
        }
    }
}

impl Optimizer for SGD {
    fn name(&self) -> &str { "SGD" }

    fn learning_rate(&self) -> f64 { self.lr }

    fn set_learning_rate(&mut self, lr: f64) { self.lr = lr; }

    fn step(&mut self, param: &mut [f32], grad: &[f32], param_name: &str) {
        assert_eq!(param.len(), grad.len());
        let n = param.len();

        let v = self.velocity.entry(param_name.to_string())
            .or_insert_with(|| vec![0.0; n]);
        if v.len() != n { v.resize(n, 0.0); }

        let lr = self.lr as f32;
        let mu = self.momentum as f32;
        let wd = self.weight_decay as f32;

        for i in 0..n {
            let mut g = grad[i];

            // L2 weight decay
            if wd != 0.0 {
                g += wd * param[i];
            }

            // Momentum update
            v[i] = mu * v[i] + g;

            if self.nesterov {
                param[i] -= lr * (g + mu * v[i]);
            } else {
                param[i] -= lr * v[i];
            }
        }
    }

    fn reset(&mut self) {
        self.velocity.clear();
    }
}

/// Clip gradient by global norm — returns the original norm
pub fn clip_grad_norm(grads: &mut [f32], max_norm: f32) -> f32 {
    if max_norm <= 0.0 {
        return 0.0;
    }
    let total_norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for g in grads.iter_mut() {
            *g *= scale;
        }
    }
    total_norm
}

/// Learning rate scheduler
pub struct LearningRateScheduler {
    /// Base learning rate
    base_lr: f64,

    /// Minimum learning rate
    min_lr: f64,

    /// Total training steps
    total_steps: usize,

    /// Warmup steps
    warmup_steps: usize,

    /// Scheduler type
    scheduler_type: SchedulerType,

    /// Current step
    current_step: usize,
}

/// Scheduler type
#[derive(Debug, Clone, Copy)]
pub enum SchedulerType {
    /// Linear decay after warmup
    Linear,
    /// Cosine annealing after warmup
    Cosine,
    /// Constant after warmup
    Constant,
}

impl LearningRateScheduler {
    /// Create a new learning rate scheduler
    pub fn new(
        base_lr: f64,
        total_steps: usize,
        warmup_steps: usize,
        scheduler_type: SchedulerType,
    ) -> Self {
        Self {
            base_lr,
            min_lr: 0.0,
            total_steps,
            warmup_steps,
            scheduler_type,
            current_step: 0,
        }
    }

    /// Set minimum learning rate
    pub fn with_min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }

    /// Get the learning rate for the current step
    pub fn get_lr(&self) -> f64 {
        if self.current_step < self.warmup_steps {
            // Linear warmup
            let progress = self.current_step as f64 / self.warmup_steps.max(1) as f64;
            return self.base_lr * progress;
        }

        let remaining = self.total_steps.saturating_sub(self.warmup_steps);
        let elapsed = self.current_step.saturating_sub(self.warmup_steps);
        let progress = elapsed as f64 / remaining.max(1) as f64;

        match self.scheduler_type {
            SchedulerType::Linear => {
                self.base_lr + (self.min_lr - self.base_lr) * progress
            }
            SchedulerType::Cosine => {
                self.min_lr + 0.5 * (self.base_lr - self.min_lr)
                    * (1.0 + (std::f64::consts::PI * progress).cos())
            }
            SchedulerType::Constant => self.base_lr,
        }
    }

    /// Advance one step and return the new learning rate
    pub fn step(&mut self) -> f64 {
        self.current_step += 1;
        self.get_lr()
    }

    /// Get the current step
    pub fn current_step(&self) -> usize {
        self.current_step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adamw_step() {
        let mut optimizer = AdamW::default_for_lr(0.01);
        let mut param = vec![1.0, 2.0, 3.0];
        let grad = vec![0.1, 0.2, 0.3];

        optimizer.step(&mut param, &grad, "test");

        // Parameters should have changed
        assert!(param[0] < 1.0);
        assert!(param[1] < 2.0);
    }

    #[test]
    fn test_sgd_step() {
        let mut optimizer = SGD::new(0.01, 0.9, 0.0, false);
        let mut param = vec![1.0, 2.0, 3.0];
        let grad = vec![0.1, 0.2, 0.3];

        optimizer.step(&mut param, &grad, "test");

        assert!((param[0] - 0.999).abs() < 1e-6); // 1.0 - 0.01 * 0.1
    }

    #[test]
    fn test_lr_scheduler_warmup() {
        let mut scheduler = LearningRateScheduler::new(1e-3, 1000, 100, SchedulerType::Cosine);

        // Step 0: lr should be 0
        assert!(scheduler.get_lr().abs() < 1e-9);

        // Step 50: should be ~half of base lr (linear warmup)
        for _ in 0..50 { scheduler.step(); }
        assert!((scheduler.get_lr() - 5e-4).abs() < 1e-5);

        // Step 100: should be base lr
        for _ in 0..50 { scheduler.step(); }
        assert!((scheduler.get_lr() - 1e-3).abs() < 1e-5);
    }

    #[test]
    fn test_lr_scheduler_cosine() {
        let mut scheduler = LearningRateScheduler::new(1e-3, 100, 0, SchedulerType::Cosine);

        // At step 0, lr = base_lr
        assert!((scheduler.get_lr() - 1e-3).abs() < 1e-6);

        // At halfway, lr should be ~half
        for _ in 0..50 { scheduler.step(); }
        assert!((scheduler.get_lr() - 5e-4).abs() < 1e-4);
    }
}
