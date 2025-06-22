//! Resilience Patterns Usage Example
//!
//! This example demonstrates how to use the resilience patterns
//! in a microservices environment to handle failures gracefully.

use std::time::Duration;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// Import our resilience patterns library
use crate::formal::resilience_patterns::{
    ResilienceFacade, 
    ResilienceConfig,
    CircuitBreakerConfig,
    RetryConfig,
    TimeoutConfig,
    BulkheadConfig,
    ResilienceError
};

// Mock service for demonstration
struct ServiceRegistry {
    services: HashMap<String, ServiceInfo>,
    failure_rates: Arc<Mutex<HashMap<String, f64>>>,
    latency_ms: Arc<Mutex<HashMap<String, u64>>>,
}

struct ServiceInfo {
    name: String,
    endpoint: String,
}

impl ServiceRegistry {
    fn new() -> Self {
        let mut services = HashMap::new();
        services.insert("auth".to_string(), ServiceInfo { 
            name: "Authentication Service".to_string(), 
            endpoint: "http://auth-service:8081".to_string() 
        });
        services.insert("user".to_string(), ServiceInfo { 
            name: "User Service".to_string(), 
            endpoint: "http://user-service:8082".to_string() 
        });
        services.insert("payment".to_string(), ServiceInfo { 
            name: "Payment Service".to_string(), 
            endpoint: "http://payment-service:8083".to_string() 
        });
        
        let mut failure_rates = HashMap::new();
        failure_rates.insert("auth".to_string(), 0.0);
        failure_rates.insert("user".to_string(), 0.0);
        failure_rates.insert("payment".to_string(), 0.0);
        
        let mut latency_ms = HashMap::new();
        latency_ms.insert("auth".to_string(), 50);
        latency_ms.insert("user".to_string(), 100);
        latency_ms.insert("payment".to_string(), 150);
        
        Self { 
            services,
            failure_rates: Arc::new(Mutex::new(failure_rates)),
            latency_ms: Arc::new(Mutex::new(latency_ms)),
        }
    }
    
    fn get_service(&self, service_id: &str) -> Option<&ServiceInfo> {
        self.services.get(service_id)
    }
    
    // Simulates service call with configurable failure rates and latency
    async fn call_service(&self, service_id: &str, operation: &str) -> Result<String, String> {
        let failure_rate = {
            let failure_rates = self.failure_rates.lock().unwrap();
            failure_rates.get(service_id).cloned().unwrap_or(0.0)
        };
        
        let latency = {
            let latency_ms = self.latency_ms.lock().unwrap();
            latency_ms.get(service_id).cloned().unwrap_or(50)
        };
        
        // Simulate network latency
        tokio::time::sleep(Duration::from_millis(latency)).await;
        
        // Simulate failures based on configured rate
        if rand::random::<f64>() < failure_rate {
            return Err(format!("Service {} failed during operation {}", service_id, operation));
        }
        
        // Successful response
        Ok(format!("Response from {} - operation: {}", service_id, operation))
    }
    
    fn set_failure_rate(&self, service_id: &str, rate: f64) {
        let mut failure_rates = self.failure_rates.lock().unwrap();
        if let Some(current_rate) = failure_rates.get_mut(service_id) {
            *current_rate = rate;
        }
    }
    
    fn set_latency(&self, service_id: &str, latency_ms: u64) {
        let mut latencies = self.latency_ms.lock().unwrap();
        if let Some(current_latency) = latencies.get_mut(service_id) {
            *current_latency = latency_ms;
        }
    }
}

struct ApiGateway {
    resilience: ResilienceFacade,
    service_registry: Arc<ServiceRegistry>,
}

impl ApiGateway {
    fn new(service_registry: Arc<ServiceRegistry>) -> Self {
        // Create a custom resilience configuration
        let config = ResilienceConfig {
            circuit_breaker: CircuitBreakerConfig {
                max_failures: 5,
                reset_timeout: Duration::from_secs(10),
                half_open_allowed_calls: 2,
            },
            retry: RetryConfig {
                max_retries: 3,
                base_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(2),
                jitter_factor: 0.1,
            },
            timeout: TimeoutConfig {
                timeout: Duration::from_millis(500),
            },
            bulkhead: BulkheadConfig {
                max_concurrent_calls: 20,
                max_queue_size: 50,
            },
        };
        
        Self {
            resilience: ResilienceFacade::with_config(config),
            service_registry,
        }
    }
    
    async fn authenticate_user(&self, user_id: &str, password: &str) -> Result<String, ResilienceError> {
        let service_registry = self.service_registry.clone();
        
        // Define main operation
        let authenticate_op = move || {
            let service_registry = service_registry.clone();
            let user_id = user_id.to_string();
            let password = password.to_string();
            
            async move {
                service_registry.call_service("auth", &format!("authenticate:{}", user_id)).await
            }
        };
        
        // Define fallback operation
        let fallback_op = move || {
            // Simplified fallback just returns a default response
            async move {
                Ok("Using cached authentication (fallback)".to_string())
            }
        };
        
        // Execute with resilience patterns
        self.resilience.execute("auth", "authenticate", authenticate_op, Some(fallback_op)).await
    }
    
    async fn get_user_profile(&self, user_id: &str) -> Result<String, ResilienceError> {
        let service_registry = self.service_registry.clone();
        
        let get_profile_op = move || {
            let service_registry = service_registry.clone();
            let user_id = user_id.to_string();
            
            async move {
                service_registry.call_service("user", &format!("get_profile:{}", user_id)).await
            }
        };
        
        // No fallback for this operation
        self.resilience.execute("user", "get_profile", get_profile_op, None).await
    }
    
    async fn process_payment(&self, user_id: &str, amount: f64) -> Result<String, ResilienceError> {
        let service_registry = self.service_registry.clone();
        
        let payment_op = move || {
            let service_registry = service_registry.clone();
            let user_id = user_id.to_string();
            
            async move {
                service_registry.call_service("payment", &format!("payment:{:.2}", amount)).await
            }
        };
        
        let fallback_op = move || {
            // Fallback that queues the payment for later processing
            async move {
                Ok(format!("Payment of ${:.2} for user {} queued for later processing", amount, user_id))
            }
        };
        
        self.resilience.execute("payment", "process_payment", payment_op, Some(fallback_op)).await
    }
    
    fn get_service_status(&self, service_id: &str) -> String {
        match self.resilience.get_circuit_state(service_id) {
            Some(state) => format!("Service {} circuit state: {:?}", service_id, state),
            None => format!("Service {} has no circuit breaker state yet", service_id),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create service registry and API gateway
    let service_registry = Arc::new(ServiceRegistry::new());
    let api_gateway = ApiGateway::new(service_registry.clone());
    
    println!("== Initial Service Status ==");
    println!("{}", api_gateway.get_service_status("auth"));
    println!("{}", api_gateway.get_service_status("user"));
    println!("{}", api_gateway.get_service_status("payment"));
    
    println!("\n== Normal Operation ==");
    let auth_result = api_gateway.authenticate_user("user123", "password").await;
    println!("Auth result: {:?}", auth_result);
    
    let profile_result = api_gateway.get_user_profile("user123").await;
    println!("Profile result: {:?}", profile_result);
    
    let payment_result = api_gateway.process_payment("user123", 99.99).await;
    println!("Payment result: {:?}", payment_result);
    
    // Simulate auth service having high failure rate
    println!("\n== Auth Service Degradation ==");
    service_registry.set_failure_rate("auth", 0.8); // 80% failure rate
    
    for i in 1..=8 {
        let result = api_gateway.authenticate_user(&format!("user{}", i), "password").await;
        println!("Auth attempt {}: {:?}", i, result);
    }
    
    println!("\n== Circuit Breaker Status ==");
    println!("{}", api_gateway.get_service_status("auth"));
    
    // Wait for circuit breaker to transition to half-open
    println!("\n== Waiting for reset timeout ==");
    tokio::time::sleep(Duration::from_secs(10)).await;
    println!("{}", api_gateway.get_service_status("auth"));
    
    // Fix the auth service
    service_registry.set_failure_rate("auth", 0.0);
    
    println!("\n== Service Recovery ==");
    for i in 1..=3 {
        let result = api_gateway.authenticate_user(&format!("recovery{}", i), "password").await;
        println!("Auth attempt {}: {:?}", i, result);
        println!("{}", api_gateway.get_service_status("auth"));
    }
    
    // Simulate payment service timeout
    println!("\n== Payment Service Timeout ==");
    service_registry.set_latency("payment", 700); // 700ms > timeout of 500ms
    
    for i in 1..=3 {
        let result = api_gateway.process_payment(&format!("user{}", i), 100.0 * i as f64).await;
        println!("Payment attempt {}: {:?}", i, result);
    }
    
    // Restore normal operation
    service_registry.set_latency("payment", 150);
    
    println!("\n== All Services Normal ==");
    let auth_result = api_gateway.authenticate_user("user999", "password").await;
    println!("Auth result: {:?}", auth_result);
    
    let profile_result = api_gateway.get_user_profile("user999").await;
    println!("Profile result: {:?}", profile_result);
    
    let payment_result = api_gateway.process_payment("user999", 999.99).await;
    println!("Payment result: {:?}", payment_result);
    
    Ok(())
} 