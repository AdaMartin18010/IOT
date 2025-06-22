# 微服务弹性模式库

一个基于形式化方法实现的微服务弹性模式库，使用TLA+规范和Rust实现。提供了电路熔断器、重试、超时、回退和隔板等弹性模式，以及服务发现与负载均衡机制，用于构建高可用性分布式系统。

## 项目背景

微服务架构虽然带来了很多优势，但也引入了系统复杂性和新的故障模式。在分布式系统中，故障是不可避免的，因此我们需要弹性模式和服务发现机制来处理这些故障。

本项目基于以下原则设计：

1. **形式化优先**：使用TLA+定义模式的形式规范，确保行为正确性
2. **验证重要性**：提供运行时验证机制，确保实现符合规范
3. **可组合性**：允许弹性模式组合使用，形成完整的防御机制
4. **可配置性**：所有模式都高度可配置，适应不同场景

## 功能特点

### 弹性模式

#### 电路熔断器

- 防止级联故障，通过快速失败保护系统资源
- 支持三种状态：关闭(正常)、开启(快速失败)和半开(恢复测试)
- 可配置失败阈值和重置超时

#### 重试机制

- 处理瞬态故障，自动重试失败的操作
- 支持指数退避策略，避免重试风暴
- 可配置重试次数、延迟和抖动因子

#### 超时控制

- 防止操作长时间阻塞，确保系统响应性
- 基于Tokio的高效异步超时实现
- 超时发生时自动取消操作

#### 回退机制

- 当主要操作失败时提供备用解决方案
- 支持自定义回退逻辑
- 确保系统在退化状态下仍能提供服务

#### 隔板模式

- 限制并发调用，防止资源耗尽
- 提供请求队列和最大并发限制配置
- 使用RAII模式确保资源正确释放

### 服务发现

#### 服务注册与注销

- 动态注册和注销服务实例
- 支持实例元数据和版本信息
- 自动管理实例生命周期

#### 健康检查

- 支持多种健康检查策略（HTTP、TCP、自定义）
- 自动排除不健康的实例
- 可配置的健康检查间隔和超时

#### 负载均衡

- 内置负载均衡算法（最少连接、轮询）
- 自动平衡请求分布
- 根据服务健康状态动态调整

#### 服务路由

- 支持基于实例元数据的路由规则
- 环境隔离与版本控制
- 灵活的服务发现策略

## 快速开始

### 安装

将以下依赖添加到你的 `Cargo.toml` 文件：

```toml
[dependencies]
resilience_patterns = "0.1.0"
tokio = { version = "1.28", features = ["full"] }
```

### 弹性模式使用

```rust
use resilience_patterns::{ResilienceFacade, ResilienceError};
use std::time::Duration;

async fn example() -> Result<(), Box<dyn std::error::Error>> {
    // 创建弹性门面
    let resilience = ResilienceFacade::new();
    
    // 执行受弹性保护的操作
    let result = resilience.execute(
        "my_service",      // 服务ID
        "get_data",        // 操作ID
        || async {
            // 主要操作实现
            Ok("Data retrieved successfully")
        },
        Some(|| async {
            // 回退实现
            Ok("Fallback data")
        }),
    ).await?;
    
    println!("Result: {}", result);
    Ok(())
}
```

### 服务发现使用

```rust
use resilience_patterns::{ServiceDiscovery, Instance, InstanceId, HealthStatus};
use std::collections::HashMap;

async fn discovery_example() -> Result<(), Box<dyn std::error::Error>> {
    // 创建服务发现实例
    let discovery = ServiceDiscovery::new();
    
    // 注册服务实例
    let instance = Instance {
        id: InstanceId {
            service: "user-service".to_string(),
            id: "user-1".to_string(),
        },
        host: "localhost".to_string(),
        port: 8080,
        metadata: HashMap::new(),
    };
    discovery.register(instance)?;
    
    // 发现服务实例
    let instances = discovery.discover("user-service")?;
    println!("Found {} instances", instances.len());
    
    // 使用负载均衡选择单个实例
    let instance = discovery.discover_one("user-service")?;
    println!("Selected instance: {}:{}", instance.host, instance.port);
    
    // 健康检查
    discovery.check_all_health(|instance| async move {
        // 实现健康检查逻辑
        true
    }).await?;
    
    Ok(())
}
```

## 自定义配置

### 弹性模式配置

```rust
use resilience_patterns::{
    ResilienceFacade, 
    ResilienceConfig,
    CircuitBreakerConfig,
    RetryConfig,
    TimeoutConfig,
    BulkheadConfig
};
use std::time::Duration;

// 创建自定义配置
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

// 使用自定义配置创建弹性门面
let resilience = ResilienceFacade::with_config(config);
```

### 服务发现配置

```rust
use resilience_patterns::{ServiceDiscovery, DiscoveryConfig};
use std::time::Duration;

// 创建自定义配置
let config = DiscoveryConfig {
    max_instances: 10,
    health_check_interval: Duration::from_secs(15),
    health_check_timeout: Duration::from_secs(5),
    max_imbalance: 100,
};

// 使用自定义配置创建服务发现实例
let discovery = ServiceDiscovery::with_config(config);
```

## 设计原则

### 形式规范映射

实现直接映射了TLA+形式规范中定义的状态和操作：

```text
TLA+规范: CircuitState ∈ {"Closed", "Open", "HalfOpen"}
Rust实现: enum CircuitState { Closed, Open, HalfOpen }
```

### 不变量验证

实现通过断言和验证确保关键不变量：

```rust
// 电路熔断器不变量检查
debug_assert!(
    cb.verify_circuit_breaker_state_invariant(),
    "Circuit breaker invariant violated"
);
```

### 组合模式

弹性模式可以通过 `execute` 方法组合使用：

```text
execute = check_circuit_breaker → acquire_bulkhead → retry_with_timeout → fallback
```

## 项目结构

- `src/`
  - `formal/` - TLA+形式规范和Rust实现
    - `resilience_patterns.rs` - 弹性模式实现
    - `MicroserviceDiscovery.rs` - 服务发现实现
  - `examples/` - 使用示例
- `docs/` - 文档
  - `resilience_patterns.md` - 弹性模式设计与实现
  - `service_discovery.md` - 服务发现机制设计与实现

## 测试和验证

项目提供了全面的测试套件，覆盖所有模式的正常和异常情况：

```rust
cargo test
```

## 贡献

欢迎贡献！请参考以下步骤：

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 许可证

MIT License

## 相关资源

- [弹性模式文档](docs/resilience_patterns.md)
- [服务发现文档](docs/service_discovery.md)
- [TLA+ 规范](code/formal/ResiliencePatterns.tla)
- [服务发现规范](code/formal/ServiceDiscovery.tla)
- [弹性模式示例](code/examples/resilience_example.rs)
- [服务发现示例](code/examples/discovery_example.rs)
