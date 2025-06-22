//! Microservice Discovery Implementation
//!
//! This module provides an implementation of the service discovery mechanism
//! described in the ServiceDiscovery.tla TLA+ specification. It includes
//! service registry, health status tracking, and discovery operations.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use thiserror::Error;
use serde::{Serialize, Deserialize};
use tokio::sync::Mutex as AsyncMutex;

/// Errors that can occur in service discovery operations
#[derive(Error, Debug)]
pub enum DiscoveryError {
    #[error("no healthy instances available for service {0}")]
    NoHealthyInstances(String),
    
    #[error("service {0} not found in registry")]
    ServiceNotFound(String),
    
    #[error("instance {0} not found")]
    InstanceNotFound(String),
    
    #[error("maximum instances reached for service {0}")]
    MaxInstancesReached(String),
    
    #[error("registry error: {0}")]
    RegistryError(String),
}

/// Service instance ID
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstanceId {
    pub service: String,
    pub id: String,
}

/// Service instance health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Unhealthy,
}

/// Service instance metadata
pub type Metadata = HashMap<String, String>;

/// Service instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instance {
    pub id: InstanceId,
    pub host: String,
    pub port: u16,
    pub metadata: Metadata,
}

/// Configuration for service discovery
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    pub max_instances: usize,
    pub health_check_interval: Duration,
    pub health_check_timeout: Duration,
    pub max_imbalance: usize,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            max_instances: 10,
            health_check_interval: Duration::from_secs(15),
            health_check_timeout: Duration::from_secs(5),
            max_imbalance: 100,
        }
    }
}

/// Instance health state with tracking information
#[derive(Debug, Clone)]
struct InstanceHealth {
    status: HealthStatus,
    last_checked: Instant,
    request_count: usize,
}

impl Default for InstanceHealth {
    fn default() -> Self {
        Self {
            status: HealthStatus::Healthy,
            last_checked: Instant::now(),
            request_count: 0,
        }
    }
}

/// Main service discovery facade
#[derive(Debug, Clone)]
pub struct ServiceDiscovery {
    state: Arc<DiscoveryState>,
}

/// Internal state for service discovery
#[derive(Debug)]
struct DiscoveryState {
    registry: RwLock<HashMap<String, HashSet<Instance>>>,
    health: RwLock<HashMap<InstanceId, InstanceHealth>>,
    config: DiscoveryConfig,
    health_check_lock: AsyncMutex<()>,
}

impl ServiceDiscovery {
    /// Create a new service discovery instance with default configuration
    pub fn new() -> Self {
        Self::with_config(DiscoveryConfig::default())
    }
    
    /// Create a new service discovery instance with custom configuration
    pub fn with_config(config: DiscoveryConfig) -> Self {
        Self {
            state: Arc::new(DiscoveryState {
                registry: RwLock::new(HashMap::new()),
                health: RwLock::new(HashMap::new()),
                config,
                health_check_lock: AsyncMutex::new(()),
            }),
        }
    }
    
    /// Register a service instance
    pub fn register(&self, instance: Instance) -> Result<(), DiscoveryError> {
        let mut registry = self.state.registry.write().unwrap();
        
        let instances = registry.entry(instance.id.service.clone()).or_insert_with(HashSet::new);
        
        // Check max instances limit
        if instances.len() >= self.state.config.max_instances {
            return Err(DiscoveryError::MaxInstancesReached(instance.id.service.clone()));
        }
        
        // Add to registry
        instances.insert(instance.clone());
        
        // Initialize health state
        let mut health = self.state.health.write().unwrap();
        health.insert(instance.id.clone(), InstanceHealth::default());
        
        Ok(())
    }
    
    /// Deregister a service instance
    pub fn deregister(&self, instance_id: &InstanceId) -> Result<(), DiscoveryError> {
        let mut registry = self.state.registry.write().unwrap();
        
        let instances = registry.get_mut(&instance_id.service).ok_or_else(|| {
            DiscoveryError::ServiceNotFound(instance_id.service.clone())
        })?;
        
        // Find and remove the instance
        let instance = instances.iter().find(|i| i.id == *instance_id).cloned();
        if let Some(instance) = instance {
            instances.remove(&instance);
            
            // Clean up health state
            let mut health = self.state.health.write().unwrap();
            health.remove(instance_id);
            
            Ok(())
        } else {
            Err(DiscoveryError::InstanceNotFound(format!("{:?}", instance_id)))
        }
    }
    
    /// Discover healthy instances for a service
    pub fn discover(&self, service: &str) -> Result<Vec<Instance>, DiscoveryError> {
        let registry = self.state.registry.read().unwrap();
        let health = self.state.health.read().unwrap();
        
        let instances = registry.get(service).ok_or_else(|| {
            DiscoveryError::ServiceNotFound(service.to_string())
        })?;
        
        // Filter to healthy instances
        let healthy_instances: Vec<Instance> = instances.iter()
            .filter(|i| {
                if let Some(health) = health.get(&i.id) {
                    health.status == HealthStatus::Healthy
                } else {
                    false
                }
            })
            .cloned()
            .collect();
        
        if healthy_instances.is_empty() {
            return Err(DiscoveryError::NoHealthyInstances(service.to_string()));
        }
        
        Ok(healthy_instances)
    }
    
    /// Discover a single healthy instance for a service, with load balancing
    pub fn discover_one(&self, service: &str) -> Result<Instance, DiscoveryError> {
        let registry = self.state.registry.read().unwrap();
        let mut health = self.state.health.write().unwrap();
        
        let instances = registry.get(service).ok_or_else(|| {
            DiscoveryError::ServiceNotFound(service.to_string())
        })?;
        
        // Find healthy instance with lowest request count
        let mut best_instance: Option<&Instance> = None;
        let mut min_requests = usize::MAX;
        
        for instance in instances {
            if let Some(health_data) = health.get(&instance.id) {
                if health_data.status == HealthStatus::Healthy && health_data.request_count < min_requests {
                    min_requests = health_data.request_count;
                    best_instance = Some(instance);
                }
            }
        }
        
        if let Some(instance) = best_instance {
            // Increment request count
            if let Some(health_data) = health.get_mut(&instance.id) {
                health_data.request_count += 1;
            }
            
            return Ok(instance.clone());
        }
        
        Err(DiscoveryError::NoHealthyInstances(service.to_string()))
    }
    
    /// Update health status of an instance
    pub fn update_health(&self, instance_id: &InstanceId, status: HealthStatus) -> Result<(), DiscoveryError> {
        let mut health = self.state.health.write().unwrap();
        
        if let Some(health_data) = health.get_mut(instance_id) {
            health_data.status = status;
            health_data.last_checked = Instant::now();
            Ok(())
        } else {
            Err(DiscoveryError::InstanceNotFound(format!("{:?}", instance_id)))
        }
    }
    
    /// Get all registered services
    pub fn get_services(&self) -> Vec<String> {
        let registry = self.state.registry.read().unwrap();
        registry.keys().cloned().collect()
    }
    
    /// Get all instances for a service
    pub fn get_instances(&self, service: &str) -> Result<Vec<Instance>, DiscoveryError> {
        let registry = self.state.registry.read().unwrap();
        
        let instances = registry.get(service).ok_or_else(|| {
            DiscoveryError::ServiceNotFound(service.to_string())
        })?;
        
        Ok(instances.iter().cloned().collect())
    }
    
    /// Check health of all registered instances
    pub async fn check_all_health<F>(&self, health_check: F) -> Result<(), DiscoveryError>
    where
        F: Fn(&Instance) -> impl std::future::Future<Output = bool> + Clone,
    {
        // Acquire lock to prevent concurrent health checks
        let _lock = self.state.health_check_lock.lock().await;
        
        let instances = {
            let registry = self.state.registry.read().unwrap();
            let mut all_instances = Vec::new();
            
            for instances in registry.values() {
                for instance in instances {
                    all_instances.push(instance.clone());
                }
            }
            
            all_instances
        };
        
        // Check health of each instance
        for instance in instances {
            let is_healthy = health_check(&instance).await;
            let status = if is_healthy { HealthStatus::Healthy } else { HealthStatus::Unhealthy };
            self.update_health(&instance.id, status)?;
        }
        
        Ok(())
    }
    
    /// Reset request counts to rebalance load
    pub fn reset_request_counts(&self) {
        let mut health = self.state.health.write().unwrap();
        
        for health_data in health.values_mut() {
            health_data.request_count = 0;
        }
    }
    
    /// Check if load is balanced across instances of a service
    pub fn is_balanced(&self, service: &str) -> Result<bool, DiscoveryError> {
        let registry = self.state.registry.read().unwrap();
        let health = self.state.health.read().unwrap();
        
        let instances = registry.get(service).ok_or_else(|| {
            DiscoveryError::ServiceNotFound(service.to_string())
        })?;
        
        if instances.len() <= 1 {
            return Ok(true);
        }
        
        let mut min_requests = usize::MAX;
        let mut max_requests = 0;
        
        for instance in instances {
            if let Some(health_data) = health.get(&instance.id) {
                if health_data.status == HealthStatus::Healthy {
                    min_requests = min_requests.min(health_data.request_count);
                    max_requests = max_requests.max(health_data.request_count);
                }
            }
        }
        
        let imbalance = max_requests.saturating_sub(min_requests);
        Ok(imbalance <= self.state.config.max_imbalance)
    }
}

/// Helper types for implementing health check functions
pub mod health_checkers {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;
    
    /// HTTP health check function
    pub fn http_health_check(path: &'static str, timeout: Duration) -> impl Fn(&Instance) -> impl Future<Output = bool> + Clone {
        move |instance| {
            let url = format!("http://{}:{}{}", instance.host, instance.port, path);
            let client_timeout = timeout;
            
            async move {
                let client = reqwest::Client::builder()
                    .timeout(client_timeout)
                    .build()
                    .unwrap_or_default();
                
                match client.get(&url).send().await {
                    Ok(response) => response.status().is_success(),
                    Err(_) => false,
                }
            }
        }
    }
    
    /// TCP health check function
    pub fn tcp_health_check(timeout: Duration) -> impl Fn(&Instance) -> impl Future<Output = bool> + Clone {
        move |instance| {
            let addr = format!("{}:{}", instance.host, instance.port);
            let client_timeout = timeout;
            
            async move {
                match tokio::time::timeout(client_timeout, tokio::net::TcpStream::connect(addr)).await {
                    Ok(Ok(_)) => true,
                    _ => false,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_instance(service: &str, id: &str, host: &str, port: u16) -> Instance {
        Instance {
            id: InstanceId {
                service: service.to_string(),
                id: id.to_string(),
            },
            host: host.to_string(),
            port,
            metadata: HashMap::new(),
        }
    }
    
    #[test]
    fn test_register_deregister() {
        let discovery = ServiceDiscovery::new();
        let instance = create_instance("test-service", "inst-1", "localhost", 8080);
        
        // Register
        assert!(discovery.register(instance.clone()).is_ok());
        
        // Get instances
        let instances = discovery.get_instances("test-service").unwrap();
        assert_eq!(instances.len(), 1);
        
        // Deregister
        assert!(discovery.deregister(&instance.id).is_ok());
        
        // Verify deregistered
        let result = discovery.get_instances("test-service");
        assert_eq!(result.unwrap().len(), 0);
    }
    
    #[test]
    fn test_max_instances_limit() {
        let config = DiscoveryConfig {
            max_instances: 2,
            ..Default::default()
        };
        
        let discovery = ServiceDiscovery::with_config(config);
        
        // Register first instance
        let instance1 = create_instance("limited-service", "inst-1", "localhost", 8080);
        assert!(discovery.register(instance1).is_ok());
        
        // Register second instance
        let instance2 = create_instance("limited-service", "inst-2", "localhost", 8081);
        assert!(discovery.register(instance2).is_ok());
        
        // Try to register third instance, should fail
        let instance3 = create_instance("limited-service", "inst-3", "localhost", 8082);
        let result = discovery.register(instance3);
        assert!(matches!(result, Err(DiscoveryError::MaxInstancesReached(_))));
    }
    
    #[test]
    fn test_health_update() {
        let discovery = ServiceDiscovery::new();
        let instance = create_instance("health-service", "inst-1", "localhost", 8080);
        
        // Register
        assert!(discovery.register(instance.clone()).is_ok());
        
        // Update health to unhealthy
        assert!(discovery.update_health(&instance.id, HealthStatus::Unhealthy).is_ok());
        
        // Try to discover, should fail with no healthy instances
        let result = discovery.discover("health-service");
        assert!(matches!(result, Err(DiscoveryError::NoHealthyInstances(_))));
        
        // Update health back to healthy
        assert!(discovery.update_health(&instance.id, HealthStatus::Healthy).is_ok());
        
        // Now should be able to discover
        let discovered = discovery.discover("health-service").unwrap();
        assert_eq!(discovered.len(), 1);
    }
    
    #[test]
    fn test_load_balancing() {
        let discovery = ServiceDiscovery::new();
        
        // Register two instances
        let instance1 = create_instance("balanced-service", "inst-1", "localhost", 8080);
        let instance2 = create_instance("balanced-service", "inst-2", "localhost", 8081);
        
        assert!(discovery.register(instance1.clone()).is_ok());
        assert!(discovery.register(instance2.clone()).is_ok());
        
        // Discover multiple times to accumulate request counts
        for _ in 0..10 {
            let _ = discovery.discover_one("balanced-service").unwrap();
        }
        
        // Should be balanced if max_imbalance is high enough
        assert!(discovery.is_balanced("balanced-service").unwrap());
        
        // Reset counts
        discovery.reset_request_counts();
        
        // After reset, counts should be 0
        let health = discovery.state.health.read().unwrap();
        for health_data in health.values() {
            assert_eq!(health_data.request_count, 0);
        }
    }
    
    #[tokio::test]
    async fn test_health_check_all() {
        let discovery = ServiceDiscovery::new();
        
        // Register instances
        let instance1 = create_instance("check-service", "inst-1", "good-host", 8080);
        let instance2 = create_instance("check-service", "inst-2", "bad-host", 8081);
        
        assert!(discovery.register(instance1.clone()).is_ok());
        assert!(discovery.register(instance2.clone()).is_ok());
        
        // Mock health check function
        let health_check = |instance: &Instance| async move {
            instance.host == "good-host"
        };
        
        // Run health check
        assert!(discovery.check_all_health(health_check).await.is_ok());
        
        // Verify health status
        let health = discovery.state.health.read().unwrap();
        assert_eq!(health.get(&instance1.id).unwrap().status, HealthStatus::Healthy);
        assert_eq!(health.get(&instance2.id).unwrap().status, HealthStatus::Unhealthy);
    }
} 