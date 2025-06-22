//! # 微服务弹性模式库
//! 
//! 基于形式化方法实现的微服务弹性模式库，提供了电路熔断器、重试、超时、回退和隔板等弹性模式，
//! 用于构建高可用性分布式系统。
//! 
//! ## 快速开始
//! 
//! ```no_run
//! use std::time::Duration;
//! use resilience_patterns::{ResilienceFacade, ResilienceError};
//! 
//! async fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     // 创建弹性门面
//!     let resilience = ResilienceFacade::new();
//!     
//!     // 执行受弹性保护的操作
//!     let result = resilience.execute(
//!         "my_service",      // 服务ID
//!         "get_data",        // 操作ID
//!         || async {
//!             // 主要操作实现
//!             Ok("Data retrieved successfully")
//!         },
//!         Some(|| async {
//!             // 回退实现
//!             Ok("Fallback data")
//!         }),
//!     ).await?;
//!     
//!     println!("Result: {}", result);
//!     Ok(())
//! }
//! ```

// 重导出主要组件
pub use crate::formal::resilience_patterns::{
    ResilienceFacade,
    ResilienceConfig,
    CircuitBreakerConfig,
    RetryConfig,
    TimeoutConfig,
    BulkheadConfig,
    CircuitState,
    ResilienceError,
    inject_failure,
    inject_latency,
};

// 重导出服务发现模块组件
pub use crate::formal::MicroserviceDiscovery::{
    ServiceDiscovery,
    DiscoveryConfig,
    Instance,
    InstanceId,
    HealthStatus,
    DiscoveryError,
    health_checkers,
};

// 模块声明
pub mod formal {
    pub mod resilience_patterns;
    pub mod MicroserviceDiscovery;
}

pub mod examples {
    pub mod resilience_example;
    pub mod discovery_example;
}

// 提供构建弹性门面的便捷函数
/// 使用默认配置创建新的弹性门面
pub fn new_resilience_facade() -> ResilienceFacade {
    ResilienceFacade::new()
}

/// 使用自定义配置创建新的弹性门面
pub fn with_config(config: ResilienceConfig) -> ResilienceFacade {
    ResilienceFacade::with_config(config)
}

/// 创建服务弹性配置的便捷方法
pub fn resilience_config_builder() -> ResilienceConfigBuilder {
    ResilienceConfigBuilder::default()
}

/// 弹性配置构建器，用于链式构建配置
pub struct ResilienceConfigBuilder {
    config: ResilienceConfig,
}

impl Default for ResilienceConfigBuilder {
    fn default() -> Self {
        Self {
            config: ResilienceConfig::default(),
        }
    }
}

impl ResilienceConfigBuilder {
    /// 设置电路熔断器配置
    pub fn circuit_breaker(mut self, max_failures: usize, reset_timeout: Duration, half_open_calls: usize) -> Self {
        self.config.circuit_breaker = CircuitBreakerConfig {
            max_failures,
            reset_timeout,
            half_open_allowed_calls: half_open_calls,
        };
        self
    }
    
    /// 设置重试配置
    pub fn retry(mut self, max_retries: usize, base_delay: Duration, max_delay: Duration, jitter: f64) -> Self {
        self.config.retry = RetryConfig {
            max_retries,
            base_delay,
            max_delay,
            jitter_factor: jitter,
        };
        self
    }
    
    /// 设置超时配置
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = TimeoutConfig { timeout };
        self
    }
    
    /// 设置隔板配置
    pub fn bulkhead(mut self, max_concurrent_calls: usize, max_queue_size: usize) -> Self {
        self.config.bulkhead = BulkheadConfig {
            max_concurrent_calls,
            max_queue_size,
        };
        self
    }
    
    /// 生成最终配置
    pub fn build(self) -> ResilienceConfig {
        self.config
    }
}

// 服务发现相关便捷函数

/// 使用默认配置创建新的服务发现实例
pub fn new_service_discovery() -> ServiceDiscovery {
    ServiceDiscovery::new()
}

/// 使用自定义配置创建新的服务发现实例
pub fn service_discovery_with_config(config: DiscoveryConfig) -> ServiceDiscovery {
    ServiceDiscovery::with_config(config)
}

/// 服务发现配置构建器
pub struct DiscoveryConfigBuilder {
    config: DiscoveryConfig,
}

impl Default for DiscoveryConfigBuilder {
    fn default() -> Self {
        Self {
            config: DiscoveryConfig::default(),
        }
    }
}

/// 创建服务发现配置的便捷方法
pub fn discovery_config_builder() -> DiscoveryConfigBuilder {
    DiscoveryConfigBuilder::default()
}

impl DiscoveryConfigBuilder {
    /// 设置最大实例数
    pub fn max_instances(mut self, max_instances: usize) -> Self {
        self.config.max_instances = max_instances;
        self
    }
    
    /// 设置健康检查间隔
    pub fn health_check_interval(mut self, interval: Duration) -> Self {
        self.config.health_check_interval = interval;
        self
    }
    
    /// 设置健康检查超时
    pub fn health_check_timeout(mut self, timeout: Duration) -> Self {
        self.config.health_check_timeout = timeout;
        self
    }
    
    /// 设置最大负载不均衡度
    pub fn max_imbalance(mut self, max_imbalance: usize) -> Self {
        self.config.max_imbalance = max_imbalance;
        self
    }
    
    /// 生成最终配置
    pub fn build(self) -> DiscoveryConfig {
        self.config
    }
} 