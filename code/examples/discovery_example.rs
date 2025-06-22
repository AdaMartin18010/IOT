//! Service Discovery Usage Example
//!
//! This example demonstrates how to use the service discovery mechanism
//! in a microservices environment for registration, discovery, and load balancing.

use std::time::Duration;
use tokio::time::sleep;

use crate::formal::MicroserviceDiscovery::{
    ServiceDiscovery,
    DiscoveryConfig,
    Instance,
    InstanceId,
    HealthStatus,
    health_checkers,
};

// Mock service for demonstration
struct ApiGateway {
    service_discovery: ServiceDiscovery,
}

impl ApiGateway {
    fn new(service_discovery: ServiceDiscovery) -> Self {
        Self { service_discovery }
    }
    
    async fn call_service(&self, service_name: &str, endpoint: &str) -> Result<String, String> {
        // Discover a healthy service instance using load balancing
        let instance = self.service_discovery.discover_one(service_name)
            .map_err(|e| e.to_string())?;
        
        // In a real implementation, this would make an actual HTTP/RPC call
        let response = format!(
            "Called service {} at http://{}:{}{} (instance id: {})",
            service_name, instance.host, instance.port, endpoint, instance.id.id
        );
        
        println!("{}", response);
        Ok(response)
    }
    
    async fn get_all_services(&self) -> Vec<String> {
        let services = self.service_discovery.get_services();
        
        println!("Available services: {:?}", services);
        services
    }
    
    async fn get_service_instances(&self, service_name: &str) -> Result<Vec<Instance>, String> {
        let instances = self.service_discovery.get_instances(service_name)
            .map_err(|e| e.to_string())?;
        
        println!("Service {} has {} instances", service_name, instances.len());
        Ok(instances)
    }
}

// Create a new service instance for registration
fn create_instance(service: &str, id: &str, host: &str, port: u16) -> Instance {
    Instance {
        id: InstanceId {
            service: service.to_string(),
            id: id.to_string(),
        },
        host: host.to_string(),
        port,
        metadata: [
            ("version".to_string(), "1.0".to_string()),
            ("environment".to_string(), "development".to_string()),
        ].into_iter().collect(),
    }
}

// Simulated health check function
async fn simulate_health_check(discovery: &ServiceDiscovery) {
    println!("Running health checks...");
    
    // Define a health check function that considers all "api" instances healthy
    // and marks "auth" instances with port > 9000 as unhealthy
    let health_checker = |instance: &Instance| async move {
        if instance.id.service == "auth" && instance.port > 9000 {
            println!("Instance {}:{} failed health check", instance.host, instance.port);
            false
        } else {
            println!("Instance {}:{} passed health check", instance.host, instance.port);
            true
        }
    };
    
    // Run health checks on all instances
    if let Err(e) = discovery.check_all_health(health_checker).await {
        println!("Health check error: {}", e);
    } else {
        println!("Health checks completed successfully");
    }
}

// Example showing load balancing through repeated service discovery
async fn demonstrate_load_balancing(gateway: &ApiGateway) {
    println!("\n=== Load Balancing Demonstration ===");
    
    // Make multiple calls to the same service to show load balancing
    for i in 1..=6 {
        println!("Call {}", i);
        if let Err(e) = gateway.call_service("api", "/products").await {
            println!("Error: {}", e);
        }
    }
}

// Example showing failure handling when services are unhealthy
async fn demonstrate_failure_handling(gateway: &ApiGateway, discovery: &ServiceDiscovery) {
    println!("\n=== Failure Handling Demonstration ===");
    
    // Mark all auth service instances as unhealthy
    let instances = discovery.get_instances("auth").unwrap();
    for instance in instances {
        discovery.update_health(&instance.id, HealthStatus::Unhealthy).unwrap();
        println!("Marked {} as unhealthy", instance.id.id);
    }
    
    // Try to call the auth service
    match gateway.call_service("auth", "/validate").await {
        Ok(response) => println!("Response: {}", response),
        Err(e) => println!("Expected error: {}", e),
    }
    
    // Restore one instance to healthy
    let instances = discovery.get_instances("auth").unwrap();
    if let Some(instance) = instances.first() {
        discovery.update_health(&instance.id, HealthStatus::Healthy).unwrap();
        println!("Restored {} to healthy status", instance.id.id);
    }
    
    // Now the call should succeed
    match gateway.call_service("auth", "/validate").await {
        Ok(response) => println!("Response after recovery: {}", response),
        Err(e) => println!("Error: {}", e),
    }
}

// Example showing service registration and deregistration
async fn demonstrate_registration(discovery: &ServiceDiscovery) {
    println!("\n=== Registration Demonstration ===");
    
    // Register a new service
    let instance = create_instance("cache", "cache-1", "cache-server", 6379);
    match discovery.register(instance.clone()) {
        Ok(()) => println!("Registered cache service instance successfully"),
        Err(e) => println!("Registration error: {}", e),
    }
    
    // List services
    let services = discovery.get_services();
    println!("Services after registration: {:?}", services);
    
    // Deregister the service
    match discovery.deregister(&instance.id) {
        Ok(()) => println!("Deregistered cache service instance successfully"),
        Err(e) => println!("Deregistration error: {}", e),
    }
    
    // List services again
    let services = discovery.get_services();
    println!("Services after deregistration: {:?}", services);
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a discovery service with custom configuration
    let config = DiscoveryConfig {
        max_instances: 5,
        health_check_interval: Duration::from_secs(5),
        health_check_timeout: Duration::from_secs(2),
        max_imbalance: 2,
    };
    let discovery = ServiceDiscovery::with_config(config);
    
    println!("=== Microservice Discovery Example ===");
    
    // Register some service instances
    let api_instance1 = create_instance("api", "api-1", "api-server-1", 8080);
    let api_instance2 = create_instance("api", "api-2", "api-server-2", 8080);
    let api_instance3 = create_instance("api", "api-3", "api-server-3", 8080);
    
    let auth_instance1 = create_instance("auth", "auth-1", "auth-server-1", 9000);
    let auth_instance2 = create_instance("auth", "auth-2", "auth-server-2", 9001);
    
    println!("Registering service instances...");
    discovery.register(api_instance1)?;
    discovery.register(api_instance2)?;
    discovery.register(api_instance3)?;
    discovery.register(auth_instance1)?;
    discovery.register(auth_instance2)?;
    
    // Create an API gateway that uses the discovery service
    let gateway = ApiGateway::new(discovery.clone());
    
    // List available services
    let services = gateway.get_all_services().await;
    
    // Get instances for each service
    for service in services {
        let _ = gateway.get_service_instances(&service).await;
    }
    
    // Run health checks
    simulate_health_check(&discovery).await;
    
    // Demonstrate different features
    demonstrate_load_balancing(&gateway).await;
    demonstrate_failure_handling(&gateway, &discovery).await;
    demonstrate_registration(&discovery).await;
    
    println!("\n=== Example completed ===");
    Ok(())
} 