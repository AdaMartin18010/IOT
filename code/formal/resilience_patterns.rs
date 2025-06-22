//! Resilience Patterns Implementation
//! 
//! This module provides an implementation of resilience patterns for microservices
//! including circuit breakers, retries, fallbacks, timeouts, and bulkheads.
//! 
//! The implementation aligns with the formal TLA+ model in ResiliencePatterns.tla
//! and provides runtime verification of key resilience properties.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use std::future::Future;
use std::pin::Pin;
use rand::Rng;
use thiserror::Error;

/// Errors that can occur in resilience operations
#[derive(Error, Debug)]
pub enum ResilienceError {
    #[error("circuit is open for service {0}")]
    CircuitOpen(String),
    
    #[error("request timed out after {0:?}")]
    Timeout(Duration),
    
    #[error("max retries ({0}) exceeded")]
    MaxRetriesExceeded(usize),
    
    #[error("all fallbacks failed")]
    AllFallbacksFailed,
    
    #[error("bulkhead full for service {0}")]
    BulkheadFull(String),
    
    #[error("service error: {0}")]
    ServiceError(String),
}

/// Circuit breaker states aligned with TLA+ model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Configuration for circuit breaker
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub max_failures: usize,
    pub reset_timeout: Duration,
    pub half_open_allowed_calls: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            max_failures: 3,
            reset_timeout: Duration::from_secs(30),
            half_open_allowed_calls: 1,
        }
    }
}

/// Configuration for retry mechanism
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: usize,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub jitter_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            jitter_factor: 0.2,
        }
    }
}

/// Configuration for timeout mechanism
#[derive(Debug, Clone)]
pub struct TimeoutConfig {
    pub timeout: Duration,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(5),
        }
    }
}

/// Configuration for bulkhead pattern
#[derive(Debug, Clone)]
pub struct BulkheadConfig {
    pub max_concurrent_calls: usize,
    pub max_queue_size: usize,
}

impl Default for BulkheadConfig {
    fn default() -> Self {
        Self {
            max_concurrent_calls: 10,
            max_queue_size: 100,
        }
    }
}

/// Combined resilience configuration
#[derive(Debug, Clone)]
pub struct ResilienceConfig {
    pub circuit_breaker: CircuitBreakerConfig,
    pub retry: RetryConfig,
    pub timeout: TimeoutConfig,
    pub bulkhead: BulkheadConfig,
}

impl Default for ResilienceConfig {
    fn default() -> Self {
        Self {
            circuit_breaker: CircuitBreakerConfig::default(),
            retry: RetryConfig::default(),
            timeout: TimeoutConfig::default(),
            bulkhead: BulkheadConfig::default(),
        }
    }
}

/// State of a specific circuit breaker
#[derive(Debug)]
struct CircuitBreakerState {
    service_id: String,
    state: CircuitState,
    failure_count: usize,
    last_failure: Instant,
    config: CircuitBreakerConfig,
    half_open_calls: usize,
}

impl CircuitBreakerState {
    fn new(service_id: &str, config: CircuitBreakerConfig) -> Self {
        Self {
            service_id: service_id.to_string(),
            state: CircuitState::Closed,
            failure_count: 0,
            last_failure: Instant::now(),
            config,
            half_open_calls: 0,
        }
    }
    
    fn record_success(&mut self) {
        match self.state {
            CircuitState::Closed => {
                self.failure_count = 0;
            }
            CircuitState::HalfOpen => {
                self.half_open_calls += 1;
                if self.half_open_calls >= self.config.half_open_allowed_calls {
                    self.state = CircuitState::Closed;
                    self.failure_count = 0;
                    self.half_open_calls = 0;
                }
            }
            CircuitState::Open => {
                // Shouldn't happen, but handle it anyway
                if self.last_failure.elapsed() >= self.config.reset_timeout {
                    self.state = CircuitState::HalfOpen;
                    self.half_open_calls = 1;
                }
            }
        }
    }
    
    fn record_failure(&mut self) {
        self.last_failure = Instant::now();
        
        match self.state {
            CircuitState::Closed => {
                self.failure_count += 1;
                if self.failure_count >= self.config.max_failures {
                    self.state = CircuitState::Open;
                }
            }
            CircuitState::HalfOpen => {
                self.state = CircuitState::Open;
                self.half_open_calls = 0;
            }
            CircuitState::Open => {
                // Reset the timer on continued failures
                self.last_failure = Instant::now();
            }
        }
    }
    
    fn allow_request(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                if self.last_failure.elapsed() >= self.config.reset_timeout {
                    self.state = CircuitState::HalfOpen;
                    self.half_open_calls = 0;
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => {
                self.half_open_calls < self.config.half_open_allowed_calls
            }
        }
    }
    
    // INVARIANT: Circuit breaker opens after max failures
    fn verify_circuit_breaker_state_invariant(&self) -> bool {
        !(self.failure_count >= self.config.max_failures && self.state != CircuitState::Open)
    }
}

/// State of bulkhead for a service
#[derive(Debug)]
struct BulkheadState {
    service_id: String,
    active_calls: usize,
    queue_size: usize,
    config: BulkheadConfig,
}

impl BulkheadState {
    fn new(service_id: &str, config: BulkheadConfig) -> Self {
        Self {
            service_id: service_id.to_string(),
            active_calls: 0,
            queue_size: 0,
            config,
        }
    }
    
    fn try_acquire(&mut self) -> bool {
        if self.active_calls < self.config.max_concurrent_calls {
            self.active_calls += 1;
            true
        } else if self.queue_size < self.config.max_queue_size {
            self.queue_size += 1;
            true
        } else {
            false
        }
    }
    
    fn release(&mut self) {
        if self.active_calls > 0 {
            self.active_calls -= 1;
            if self.queue_size > 0 {
                self.queue_size -= 1;
                self.active_calls += 1;
            }
        }
    }
}

/// Main resilience facade that provides all resilience patterns
#[derive(Debug, Clone)]
pub struct ResilienceFacade {
    // Shared state across all instances
    state: Arc<ResilienceState>,
}

/// Shared resilience state
#[derive(Debug)]
struct ResilienceState {
    circuit_breakers: RwLock<HashMap<String, Mutex<CircuitBreakerState>>>,
    bulkheads: RwLock<HashMap<String, Mutex<BulkheadState>>>,
    config: ResilienceConfig,
    pending_requests: RwLock<HashSet<String>>,
}

impl ResilienceFacade {
    /// Create a new resilience facade with default configuration
    pub fn new() -> Self {
        Self::with_config(ResilienceConfig::default())
    }
    
    /// Create a new resilience facade with custom configuration
    pub fn with_config(config: ResilienceConfig) -> Self {
        Self {
            state: Arc::new(ResilienceState {
                circuit_breakers: RwLock::new(HashMap::new()),
                bulkheads: RwLock::new(HashMap::new()),
                config,
                pending_requests: RwLock::new(HashSet::new()),
            }),
        }
    }
    
    /// Execute an operation with all configured resilience patterns
    pub async fn execute<F, Fut, T, E>(
        &self,
        service_id: &str,
        operation_id: &str,
        operation: F,
        fallback: Option<F>,
    ) -> Result<T, ResilienceError>
    where
        F: Fn() -> Fut + Clone + Send + Sync + 'static,
        Fut: Future<Output = Result<T, E>> + Send + 'static,
        T: Send + 'static,
        E: std::error::Error + Send + 'static,
    {
        // Track the request
        let request_id = format!("{}:{}", service_id, operation_id);
        self.add_pending_request(&request_id);
        
        // Apply circuit breaker
        self.check_circuit_breaker(service_id)?;
        
        // Apply bulkhead
        let bulkhead_guard = self.acquire_bulkhead(service_id)?;
        
        // Apply retry with timeout
        let retry_result = self.retry_with_timeout(service_id, operation.clone()).await;
        
        // Remove from pending
        self.remove_pending_request(&request_id);
        
        // Handle result
        match retry_result {
            Ok(result) => {
                // Record success
                self.record_success(service_id);
                drop(bulkhead_guard);
                Ok(result)
            }
            Err(err) => {
                // Record failure
                self.record_failure(service_id);
                drop(bulkhead_guard);
                
                // Try fallback if available
                if let Some(fallback_fn) = fallback {
                    match self.execute_with_timeout(fallback_fn).await {
                        Ok(result) => Ok(result),
                        Err(_) => Err(ResilienceError::AllFallbacksFailed),
                    }
                } else {
                    Err(err)
                }
            }
        }
    }
    
    /// Add a pending request
    fn add_pending_request(&self, request_id: &str) {
        let mut requests = self.state.pending_requests.write().unwrap();
        requests.insert(request_id.to_string());
    }
    
    /// Remove a pending request
    fn remove_pending_request(&self, request_id: &str) {
        let mut requests = self.state.pending_requests.write().unwrap();
        requests.remove(request_id);
    }
    
    /// Check if circuit breaker allows the request
    fn check_circuit_breaker(&self, service_id: &str) -> Result<(), ResilienceError> {
        let circuit_breaker = self.get_or_create_circuit_breaker(service_id);
        let mut cb = circuit_breaker.lock().unwrap();
        
        // Verify circuit breaker invariant before state change
        debug_assert!(
            cb.verify_circuit_breaker_state_invariant(),
            "Circuit breaker invariant violated before state change"
        );
        
        if cb.allow_request() {
            Ok(())
        } else {
            Err(ResilienceError::CircuitOpen(service_id.to_string()))
        }
    }
    
    /// Get or create circuit breaker for service
    fn get_or_create_circuit_breaker(&self, service_id: &str) -> Arc<Mutex<CircuitBreakerState>> {
        let circuit_breakers = self.state.circuit_breakers.read().unwrap();
        
        if let Some(cb) = circuit_breakers.get(service_id) {
            Arc::clone(cb)
        } else {
            drop(circuit_breakers);
            let mut circuit_breakers = self.state.circuit_breakers.write().unwrap();
            
            // Double-check after getting write lock
            if let Some(cb) = circuit_breakers.get(service_id) {
                return Arc::clone(cb);
            }
            
            let cb = Arc::new(Mutex::new(CircuitBreakerState::new(
                service_id,
                self.state.config.circuit_breaker.clone(),
            )));
            
            circuit_breakers.insert(service_id.to_string(), cb.clone());
            cb
        }
    }
    
    /// Acquire bulkhead permit
    fn acquire_bulkhead(&self, service_id: &str) -> Result<BulkheadGuard, ResilienceError> {
        let bulkhead = self.get_or_create_bulkhead(service_id);
        let mut bh = bulkhead.lock().unwrap();
        
        if bh.try_acquire() {
            Ok(BulkheadGuard {
                service_id: service_id.to_string(),
                state: Arc::clone(&self.state),
            })
        } else {
            Err(ResilienceError::BulkheadFull(service_id.to_string()))
        }
    }
    
    /// Get or create bulkhead for service
    fn get_or_create_bulkhead(&self, service_id: &str) -> Arc<Mutex<BulkheadState>> {
        let bulkheads = self.state.bulkheads.read().unwrap();
        
        if let Some(bh) = bulkheads.get(service_id) {
            Arc::clone(bh)
        } else {
            drop(bulkheads);
            let mut bulkheads = self.state.bulkheads.write().unwrap();
            
            // Double-check after getting write lock
            if let Some(bh) = bulkheads.get(service_id) {
                return Arc::clone(bh);
            }
            
            let bh = Arc::new(Mutex::new(BulkheadState::new(
                service_id,
                self.state.config.bulkhead.clone(),
            )));
            
            bulkheads.insert(service_id.to_string(), bh.clone());
            bh
        }
    }
    
    /// Record success for a service
    fn record_success(&self, service_id: &str) {
        let circuit_breaker = self.get_or_create_circuit_breaker(service_id);
        let mut cb = circuit_breaker.lock().unwrap();
        cb.record_success();
        
        // Verify circuit breaker invariant after state change
        debug_assert!(
            cb.verify_circuit_breaker_state_invariant(),
            "Circuit breaker invariant violated after success"
        );
    }
    
    /// Record failure for a service
    fn record_failure(&self, service_id: &str) {
        let circuit_breaker = self.get_or_create_circuit_breaker(service_id);
        let mut cb = circuit_breaker.lock().unwrap();
        cb.record_failure();
        
        // Verify circuit breaker invariant after state change
        debug_assert!(
            cb.verify_circuit_breaker_state_invariant(),
            "Circuit breaker invariant violated after failure"
        );
    }
    
    /// Execute with retry and timeout
    async fn retry_with_timeout<F, Fut, T, E>(
        &self,
        service_id: &str,
        operation: F,
    ) -> Result<T, ResilienceError>
    where
        F: Fn() -> Fut + Clone,
        Fut: Future<Output = Result<T, E>>,
        E: std::error::Error,
    {
        let mut attempts = 0;
        let max_retries = self.state.config.retry.max_retries;
        let base_delay = self.state.config.retry.base_delay;
        let max_delay = self.state.config.retry.max_delay;
        let jitter_factor = self.state.config.retry.jitter_factor;
        
        loop {
            match self.execute_with_timeout(operation.clone()).await {
                Ok(result) => return Ok(result),
                Err(err) => {
                    attempts += 1;
                    if attempts > max_retries {
                        return Err(ResilienceError::MaxRetriesExceeded(max_retries));
                    }
                    
                    // Calculate exponential backoff with jitter
                    let delay_millis = std::cmp::min(
                        base_delay.as_millis() * 2_u128.pow(attempts as u32),
                        max_delay.as_millis(),
                    ) as u64;
                    
                    let jitter = (delay_millis as f64 * jitter_factor * rand::thread_rng().gen::<f64>()) as u64;
                    let delay = std::cmp::min(delay_millis + jitter, max_delay.as_millis() as u64);
                    
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                    
                    // Check circuit breaker again before retry
                    self.check_circuit_breaker(service_id)?;
                }
            }
        }
    }
    
    /// Execute with timeout
    async fn execute_with_timeout<F, Fut, T, E>(&self, operation: F) -> Result<T, ResilienceError>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T, E>>,
        E: std::error::Error,
    {
        let timeout_duration = self.state.config.timeout.timeout;
        
        match tokio::time::timeout(timeout_duration, operation()).await {
            Ok(result) => result.map_err(|e| ResilienceError::ServiceError(e.to_string())),
            Err(_) => Err(ResilienceError::Timeout(timeout_duration)),
        }
    }
    
    /// Get current circuit breaker state
    pub fn get_circuit_state(&self, service_id: &str) -> Option<CircuitState> {
        let circuit_breakers = self.state.circuit_breakers.read().unwrap();
        circuit_breakers.get(service_id).map(|cb| {
            let cb = cb.lock().unwrap();
            cb.state
        })
    }
    
    /// Get failure count for a service
    pub fn get_failure_count(&self, service_id: &str) -> Option<usize> {
        let circuit_breakers = self.state.circuit_breakers.read().unwrap();
        circuit_breakers.get(service_id).map(|cb| {
            let cb = cb.lock().unwrap();
            cb.failure_count
        })
    }
    
    /// Verify that all circuit breakers satisfy the invariant
    pub fn verify_all_circuit_breaker_invariants(&self) -> bool {
        let circuit_breakers = self.state.circuit_breakers.read().unwrap();
        circuit_breakers.values().all(|cb| {
            let cb = cb.lock().unwrap();
            cb.verify_circuit_breaker_state_invariant()
        })
    }
}

/// Bulkhead guard to track bulkhead resource usage
struct BulkheadGuard {
    service_id: String,
    state: Arc<ResilienceState>,
}

impl Drop for BulkheadGuard {
    fn drop(&mut self) {
        let bulkheads = self.state.bulkheads.read().unwrap();
        if let Some(bh) = bulkheads.get(&self.service_id) {
            let mut bh = bh.lock().unwrap();
            bh.release();
        }
    }
}

/// Helper function to inject failures for testing
pub fn inject_failure(facade: &ResilienceFacade, service_id: &str) {
    facade.record_failure(service_id);
}

/// Helper function to inject latency for testing
pub async fn inject_latency(delay: Duration) {
    tokio::time::sleep(delay).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    
    #[tokio::test]
    async fn test_circuit_breaker() {
        let facade = ResilienceFacade::with_config(ResilienceConfig {
            circuit_breaker: CircuitBreakerConfig {
                max_failures: 3,
                reset_timeout: Duration::from_millis(100),
                half_open_allowed_calls: 1,
            },
            ..Default::default()
        });
        
        let service_id = "test_service";
        
        // Should work initially
        assert!(facade.check_circuit_breaker(service_id).is_ok());
        
        // Record failures
        for _ in 0..3 {
            facade.record_failure(service_id);
        }
        
        // Should be open now
        assert!(matches!(
            facade.check_circuit_breaker(service_id),
            Err(ResilienceError::CircuitOpen(_))
        ));
        
        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(200)).await;
        
        // Should be half-open and allow a request
        assert!(facade.check_circuit_breaker(service_id).is_ok());
        
        // Record success to close circuit
        facade.record_success(service_id);
        
        // Should be closed
        assert_eq!(facade.get_circuit_state(service_id), Some(CircuitState::Closed));
    }
    
    #[tokio::test]
    async fn test_bulkhead() {
        let facade = ResilienceFacade::with_config(ResilienceConfig {
            bulkhead: BulkheadConfig {
                max_concurrent_calls: 2,
                max_queue_size: 1,
            },
            ..Default::default()
        });
        
        let service_id = "test_service";
        
        // Acquire permits
        let guard1 = facade.acquire_bulkhead(service_id).unwrap();
        let guard2 = facade.acquire_bulkhead(service_id).unwrap();
        let guard3 = facade.acquire_bulkhead(service_id).unwrap();
        
        // Should be full now
        assert!(matches!(
            facade.acquire_bulkhead(service_id),
            Err(ResilienceError::BulkheadFull(_))
        ));
        
        // Release one permit
        drop(guard1);
        
        // Should be able to acquire again
        let _guard4 = facade.acquire_bulkhead(service_id).unwrap();
        
        // Clean up
        drop(guard2);
        drop(guard3);
        drop(_guard4);
    }
    
    #[tokio::test]
    async fn test_retry() {
        let facade = ResilienceFacade::with_config(ResilienceConfig {
            retry: RetryConfig {
                max_retries: 2,
                base_delay: Duration::from_millis(10),
                max_delay: Duration::from_millis(100),
                jitter_factor: 0.1,
            },
            ..Default::default()
        });
        
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();
        
        // Operation that fails twice then succeeds
        let result = facade
            .retry_with_timeout("test_service", move || {
                let counter = counter_clone.clone();
                async move {
                    let count = counter.fetch_add(1, Ordering::SeqCst);
                    if count < 2 {
                        Err("deliberate failure")
                    } else {
                        Ok("success")
                    }
                }
            })
            .await;
        
        assert!(result.is_ok());
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }
    
    #[tokio::test]
    async fn test_timeout() {
        let facade = ResilienceFacade::with_config(ResilienceConfig {
            timeout: TimeoutConfig {
                timeout: Duration::from_millis(50),
            },
            ..Default::default()
        });
        
        // Fast operation succeeds
        let fast_result = facade
            .execute_with_timeout(|| async { Ok::<_, std::io::Error>(()) })
            .await;
        
        assert!(fast_result.is_ok());
        
        // Slow operation times out
        let slow_result = facade
            .execute_with_timeout(|| async {
                tokio::time::sleep(Duration::from_millis(200)).await;
                Ok::<_, std::io::Error>(())
            })
            .await;
        
        assert!(matches!(slow_result, Err(ResilienceError::Timeout(_))));
    }
    
    #[tokio::test]
    async fn test_fallback() {
        let facade = ResilienceFacade::new();
        
        let primary = || async { Err::<&str, _>("primary error") };
        let fallback = || async { Ok("fallback result") };
        
        let result = facade
            .execute("test_service", "op", primary, Some(fallback))
            .await;
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "fallback result");
    }
    
    #[test]
    fn test_circuit_breaker_invariant() {
        let facade = ResilienceFacade::new();
        let service_id = "test_service";
        
        // Initial state should satisfy invariant
        assert!(facade.verify_all_circuit_breaker_invariants());
        
        // Record failures up to threshold
        for _ in 0..3 {
            facade.record_failure(service_id);
        }
        
        // Should still satisfy invariant after failures
        assert!(facade.verify_all_circuit_breaker_invariants());
    }
}
