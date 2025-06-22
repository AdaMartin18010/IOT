# 弹性模式：从形式规范到实现

本文档描述了微服务弹性模式的TLA+形式规范与Rust实现之间的对应关系，并解释了如何通过运行时验证确保实现符合形式规范中定义的关键属性和不变量。

## 形式规范与实现映射

### 电路熔断器 (Circuit Breaker)

| TLA+ 规范 | Rust 实现 |
|----------|----------|
| `CircuitState` 枚举 | `CircuitState` 枚举 |
| `CircuitBreaker` 状态转换 | `CircuitBreakerState` 结构体中的方法 |
| `RecordSuccess()` 操作 | `record_success()` 方法 |
| `RecordFailure()` 操作 | `record_failure()` 方法 |
| `AllowRequest()` 操作 | `allow_request()` 方法 |
| `CircuitBreakerOpens` 不变量 | `verify_circuit_breaker_state_invariant()` 方法 |

### 重试 (Retry)

| TLA+ 规范 | Rust 实现 |
|----------|----------|
| `RetryAttempt` 变量 | `attempts` 计数器 |
| `MaxRetries` 常量 | `RetryConfig.max_retries` 配置 |
| `ExponentialBackoff` 函数 | 退避计算逻辑在 `retry_with_timeout` 方法中 |
| `RetryLimitInvariant` 不变量 | 通过 `MAX_RETRIES_EXCEEDED` 错误强制执行 |

### 隔板 (Bulkhead)

| TLA+ 规范 | Rust 实现 |
|----------|----------|
| `ActiveCalls` 变量 | `BulkheadState.active_calls` 字段 |
| `QueueSize` 变量 | `BulkheadState.queue_size` 字段 |
| `MaxConcurrentCalls` 常量 | `BulkheadConfig.max_concurrent_calls` 配置 |
| `MaxQueueSize` 常量 | `BulkheadConfig.max_queue_size` 配置 |
| `TryAcquire()` 操作 | `try_acquire()` 方法 |
| `Release()` 操作 | `release()` 方法 |
| `BulkheadLimitInvariant` 不变量 | 通过方法实现逻辑保证 |

### 超时 (Timeout)

| TLA+ 规范 | Rust 实现 |
|----------|----------|
| `RequestTimeout` 常量 | `TimeoutConfig.timeout` 配置 |
| `ElapsedTime` 变量 | 通过 `tokio::time::timeout` 隐式处理 |
| `TimeoutExpired` 谓词 | 通过 `tokio::time::timeout` 的结果处理 |
| `TimeoutSafetyInvariant` | 通过错误处理保证 |

### 回退 (Fallback)

| TLA+ 规范 | Rust 实现 |
|----------|----------|
| `PrimaryOperation()` | 主操作闭包 |
| `FallbackOperation()` | 回退操作闭包 |
| `FallbackSuccessfulInvariant` | 通过错误处理保证 |

## 关键不变量验证

实现通过以下机制验证TLA+规范中定义的关键不变量：

1. **电路熔断器不变量**：
   - 通过 `verify_circuit_breaker_state_invariant()` 方法实现
   - 在每次状态变更前后使用 `debug_assert!` 验证不变量

2. **重试限制不变量**：
   - 通过显式计数和限制检查实现
   - 通过错误返回机制确保不会超过最大重试次数

3. **隔板容量不变量**：
   - 通过 `BulkheadGuard` RAII模式确保资源正确释放
   - 通过原子计数和检查确保不超过容量限制

4. **超时安全不变量**：
   - 利用 `tokio::time::timeout` 提供的超时保证
   - 确保所有可能长时间运行的操作都受到超时控制

5. **回退成功不变量**：
   - 通过错误处理和回退逻辑确保在主操作失败时能正确应用回退

## 运行时验证方法

实现使用以下方法进行运行时验证：

1. **Debug断言**：
   - 使用 `debug_assert!` 在关键点验证不变量
   - 在开发和测试环境中提供早期检测

2. **统计收集**：
   - 通过 `get_circuit_state` 和 `get_failure_count` 等方法提供可观察性
   - 允许外部系统监控弹性模式的状态和行为

3. **单元测试**：
   - 测试套件覆盖所有弹性模式的正常和异常行为
   - 包括边缘情况和故障情景测试

4. **模拟退化**：
   - 示例程序通过 `ServiceRegistry` 提供可控的故障注入
   - 可以模拟延迟、错误率和其他故障模式

## 模型-实现一致性

为确保Rust实现与TLA+模型保持一致，采用以下策略：

1. **结构对应**：
   - 实现类和方法与TLA+规范中的操作和变量直接对应
   - 状态转换逻辑与规范中定义的转换规则保持一致

2. **不变量检查**：
   - 关键不变量通过运行时检查机制实现
   - 测试集验证不变量在各种条件下都能保持

3. **配置映射**：
   - 所有TLA+中的常量都通过配置对象暴露
   - 允许调整参数同时保持行为模型不变

4. **错误处理**：
   - 错误类型与TLA+规范中定义的异常情况对应
   - 提供详细的错误信息以帮助排查问题

## 使用指南

1. **创建和配置**：

   ```rust
   let resilience = ResilienceFacade::with_config(ResilienceConfig {
       circuit_breaker: CircuitBreakerConfig {
           max_failures: 5,
           reset_timeout: Duration::from_secs(10),
           half_open_allowed_calls: 2,
       },
       // 其他配置...
   });
   ```

2. **执行操作**：

   ```rust
   let result = resilience.execute(
       "service_id", 
       "operation_id",
       || async { /* 主要操作 */ },
       Some(|| async { /* 回退操作 */ }),
   ).await;
   ```

3. **监控状态**：

   ```rust
   let state = resilience.get_circuit_state("service_id");
   let failures = resilience.get_failure_count("service_id");
   ```

4. **验证不变量**：

   ```rust
   let invariants_satisfied = resilience.verify_all_circuit_breaker_invariants();
   assert!(invariants_satisfied);
   ```

## 结论

微服务弹性模式的Rust实现直接映射了TLA+形式规范中的关键概念和行为。通过运行时验证和全面的测试，可以确保实现满足规范中定义的安全和活性属性。这种基于形式方法的实现方法提供了更高的可靠性和可预测性，特别适合构建需要高可用性的分布式系统。
