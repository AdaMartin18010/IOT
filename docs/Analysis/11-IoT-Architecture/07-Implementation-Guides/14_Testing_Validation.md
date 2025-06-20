# 14. IoT测试验证与质量保证

## 14.1 测试策略概述

### 14.1.1 测试金字塔

**定义14.1（测试金字塔）**：\( TestPyramid = UnitTests + IntegrationTests + E2ETests \)

### 14.1.2 测试层次

- **单元测试**：70% - 组件级别测试
- **集成测试**：20% - 服务间交互测试
- **端到端测试**：10% - 完整业务流程测试

## 14.2 单元测试

### 14.2.1 Rust单元测试

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    fn test_device_registration() {
        let mut manager = DeviceManager::new();
        let device_info = DeviceInfo {
            name: "test-device".to_string(),
            device_type: "sensor".to_string(),
            capabilities: vec!["temperature".to_string()],
        };
        
        let result = manager.register_device(device_info);
        assert!(result.is_ok());
        
        let device_id = result.unwrap();
        assert!(manager.devices.contains_key(&device_id));
    }

    #[test]
    fn test_data_validation() {
        let validator = DataValidator::new();
        let valid_data = SensorData {
            device_id: "test-device".to_string(),
            timestamp: Utc::now(),
            temperature: 25.0,
            humidity: 60.0,
        };
        
        let result = validator.validate(&valid_data);
        assert!(result.is_ok());
        
        let invalid_data = SensorData {
            device_id: "".to_string(),
            timestamp: Utc::now(),
            temperature: -100.0, // 无效温度
            humidity: 150.0,     // 无效湿度
        };
        
        let result = validator.validate(&invalid_data);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_async_device_operations() {
        let mut device = IoTDevice::new("test-device");
        device.connect().await.unwrap();
        
        let data = device.read_sensor_data().await.unwrap();
        assert!(data.temperature >= -50.0 && data.temperature <= 100.0);
        
        device.disconnect().await.unwrap();
    }
}
```

### 14.2.2 Golang单元测试

```go
package iot

import (
    "testing"
    "time"
)

func TestDeviceRegistration(t *testing.T) {
    manager := NewDeviceManager()
    deviceInfo := DeviceInfo{
        Name:        "test-device",
        DeviceType:  "sensor",
        Capabilities: []string{"temperature"},
    }
    
    deviceID, err := manager.RegisterDevice(deviceInfo)
    if err != nil {
        t.Fatalf("设备注册失败: %v", err)
    }
    
    if deviceID == "" {
        t.Error("设备ID不应为空")
    }
    
    device, exists := manager.GetDevice(deviceID)
    if !exists {
        t.Error("设备应存在于管理器中")
    }
    
    if device.Name != deviceInfo.Name {
        t.Errorf("设备名称不匹配: 期望 %s, 实际 %s", deviceInfo.Name, device.Name)
    }
}

func TestDataValidation(t *testing.T) {
    validator := NewDataValidator()
    
    // 测试有效数据
    validData := SensorData{
        DeviceID:   "test-device",
        Timestamp:  time.Now(),
        Temperature: 25.0,
        Humidity:   60.0,
    }
    
    if err := validator.Validate(validData); err != nil {
        t.Errorf("有效数据验证失败: %v", err)
    }
    
    // 测试无效数据
    invalidData := SensorData{
        DeviceID:   "",
        Timestamp:  time.Now(),
        Temperature: -100.0,
        Humidity:   150.0,
    }
    
    if err := validator.Validate(invalidData); err == nil {
        t.Error("无效数据验证应失败")
    }
}

func TestAsyncDeviceOperations(t *testing.T) {
    device := NewIoTDevice("test-device")
    
    // 测试连接
    if err := device.Connect(); err != nil {
        t.Fatalf("设备连接失败: %v", err)
    }
    defer device.Disconnect()
    
    // 测试数据读取
    data, err := device.ReadSensorData()
    if err != nil {
        t.Fatalf("传感器数据读取失败: %v", err)
    }
    
    if data.Temperature < -50 || data.Temperature > 100 {
        t.Errorf("温度值超出合理范围: %f", data.Temperature)
    }
}
```

## 14.3 集成测试

### 14.3.1 服务集成测试

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    use tokio::test;
    use testcontainers::*;

    #[tokio::test]
    async fn test_device_to_cloud_integration() {
        // 启动测试数据库
        let docker = clients::Cli::default();
        let postgres = images::postgres::Postgres::default()
            .with_db_name("iot_test");
        let node = docker.run(postgres);
        let connection_string = format!(
            "postgresql://postgres:postgres@localhost:{}/iot_test",
            node.get_host_port_ipv4(5432)
        );
        
        // 初始化服务
        let data_service = DataService::new(&connection_string).await.unwrap();
        let device_service = DeviceService::new(&connection_string).await.unwrap();
        
        // 测试设备注册和数据流
        let device_info = DeviceInfo {
            name: "integration-test-device".to_string(),
            device_type: "sensor".to_string(),
            capabilities: vec!["temperature".to_string()],
        };
        
        let device_id = device_service.register_device(device_info).await.unwrap();
        
        let sensor_data = SensorData {
            device_id: device_id.clone(),
            timestamp: Utc::now(),
            temperature: 25.0,
            humidity: 60.0,
        };
        
        data_service.store_data(sensor_data).await.unwrap();
        
        let stored_data = data_service.get_device_data(&device_id).await.unwrap();
        assert_eq!(stored_data.len(), 1);
        assert_eq!(stored_data[0].temperature, 25.0);
    }

    #[tokio::test]
    async fn test_event_driven_integration() {
        let event_bus = EventBus::new(100);
        let mut device_handler = DeviceEventHandler::new();
        let mut alert_handler = AlertEventHandler::new();
        
        // 订阅事件
        let mut device_rx = event_bus.subscribe().await.unwrap();
        let mut alert_rx = event_bus.subscribe().await.unwrap();
        
        // 发布设备连接事件
        let connect_event = IoTEvent::DeviceConnected(DeviceConnectedEvent {
            device_id: "test-device".to_string(),
            timestamp: Utc::now(),
            connection_info: ConnectionInfo::default(),
        });
        
        event_bus.publish(connect_event).await.unwrap();
        
        // 验证事件处理
        let received_event = device_rx.recv().await.unwrap();
        assert!(matches!(received_event, IoTEvent::DeviceConnected(_)));
    }
}
```

### 14.3.2 API集成测试

```rust
use actix_web::{test, web, App};
use serde_json::json;

#[tokio::test]
async fn test_device_api_integration() {
    let app = test::init_service(
        App::new()
            .service(web::resource("/devices/{id}").to(get_device))
            .service(web::resource("/devices").to(create_device))
    ).await;
    
    // 测试设备创建
    let device_data = json!({
        "name": "api-test-device",
        "device_type": "sensor",
        "capabilities": ["temperature"]
    });
    
    let req = test::TestRequest::post()
        .uri("/devices")
        .set_json(&device_data)
        .to_request();
    
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
    
    let device: DeviceInfo = test::read_body_json(resp).await;
    assert_eq!(device.name, "api-test-device");
    
    // 测试设备查询
    let req = test::TestRequest::get()
        .uri(&format!("/devices/{}", device.id))
        .to_request();
    
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
    
    let retrieved_device: DeviceInfo = test::read_body_json(resp).await;
    assert_eq!(retrieved_device.id, device.id);
}
```

## 14.4 性能测试

### 14.4.1 负载测试

```rust
use tokio::time::{sleep, Duration};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::test]
async fn test_system_load_performance() {
    let device_manager = Arc::new(Mutex::new(DeviceManager::new()));
    let mut handles = vec![];
    
    // 模拟1000个并发设备连接
    for i in 0..1000 {
        let manager_clone = device_manager.clone();
        let handle = tokio::spawn(async move {
            let device_info = DeviceInfo {
                name: format!("load-test-device-{}", i),
                device_type: "sensor".to_string(),
                capabilities: vec!["temperature".to_string()],
            };
            
            let mut manager = manager_clone.lock().await;
            manager.register_device(device_info).await
        });
        handles.push(handle);
    }
    
    // 等待所有设备注册完成
    let start_time = std::time::Instant::now();
    for handle in handles {
        handle.await.unwrap().unwrap();
    }
    let duration = start_time.elapsed();
    
    // 验证性能指标
    assert!(duration.as_millis() < 5000); // 5秒内完成1000个设备注册
    
    let manager = device_manager.lock().await;
    assert_eq!(manager.devices.len(), 1000);
}

#[tokio::test]
async fn test_data_throughput_performance() {
    let data_service = DataService::new("test_db").await.unwrap();
    let mut handles = vec![];
    
    // 模拟高并发数据写入
    for i in 0..10000 {
        let service_clone = data_service.clone();
        let handle = tokio::spawn(async move {
            let sensor_data = SensorData {
                device_id: format!("device-{}", i % 100),
                timestamp: Utc::now(),
                temperature: 20.0 + (i as f64 * 0.1),
                humidity: 50.0 + (i as f64 * 0.05),
            };
            service_clone.store_data(sensor_data).await
        });
        handles.push(handle);
    }
    
    let start_time = std::time::Instant::now();
    for handle in handles {
        handle.await.unwrap().unwrap();
    }
    let duration = start_time.elapsed();
    
    // 验证吞吐量
    let throughput = 10000.0 / duration.as_secs_f64();
    assert!(throughput > 1000.0); // 每秒处理1000+条数据
}
```

### 14.4.2 压力测试

```rust
#[tokio::test]
async fn test_system_stress_limits() {
    let event_bus = EventBus::new(1000);
    let mut handles = vec![];
    
    // 持续发送事件直到系统饱和
    let start_time = std::time::Instant::now();
    let mut event_count = 0;
    
    loop {
        if start_time.elapsed().as_secs() > 60 {
            break; // 测试1分钟
        }
        
        let event = IoTEvent::DataReceived(DataReceivedEvent {
            device_id: format!("stress-device-{}", event_count % 100),
            timestamp: Utc::now(),
            data: SensorData::random(),
        });
        
        match event_bus.publish(event).await {
            Ok(_) => {
                event_count += 1;
            }
            Err(_) => {
                // 系统饱和，停止测试
                break;
            }
        }
        
        sleep(Duration::from_millis(1)).await;
    }
    
    println!("系统在1分钟内处理了 {} 个事件", event_count);
    assert!(event_count > 10000); // 至少处理10000个事件
}
```

## 14.5 安全测试

### 14.5.1 认证测试

```rust
#[cfg(test)]
mod security_tests {
    use super::*;

    #[tokio::test]
    async fn test_authentication_mechanisms() {
        let auth_service = AuthenticationService::new();
        
        // 测试有效凭据
        let valid_credentials = Credentials {
            username: "test_user".to_string(),
            password: "secure_password".to_string(),
        };
        
        let token = auth_service.authenticate(&valid_credentials).await.unwrap();
        assert!(!token.is_empty());
        
        // 测试无效凭据
        let invalid_credentials = Credentials {
            username: "test_user".to_string(),
            password: "wrong_password".to_string(),
        };
        
        let result = auth_service.authenticate(&invalid_credentials).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_authorization_checks() {
        let auth_service = AuthenticationService::new();
        let device_service = DeviceService::new();
        
        // 创建用户和权限
        let user = User {
            id: "user1".to_string(),
            roles: vec!["device_operator".to_string()],
        };
        
        let device = DeviceInfo {
            id: "device1".to_string(),
            name: "test-device".to_string(),
            device_type: "sensor".to_string(),
            owner: "user1".to_string(),
        };
        
        // 测试授权访问
        let has_access = auth_service.check_permission(&user, "read", &device).await.unwrap();
        assert!(has_access);
        
        // 测试未授权访问
        let unauthorized_user = User {
            id: "user2".to_string(),
            roles: vec!["guest".to_string()],
        };
        
        let has_access = auth_service.check_permission(&unauthorized_user, "write", &device).await.unwrap();
        assert!(!has_access);
    }
}
```

### 14.5.2 数据安全测试

```rust
#[tokio::test]
async fn test_data_encryption() {
    let crypto_service = CryptoService::new();
    let test_data = "sensitive_sensor_data".as_bytes();
    
    // 测试数据加密
    let encrypted_data = crypto_service.encrypt(test_data).await.unwrap();
    assert_ne!(encrypted_data, test_data);
    
    // 测试数据解密
    let decrypted_data = crypto_service.decrypt(&encrypted_data).await.unwrap();
    assert_eq!(decrypted_data, test_data);
}

#[tokio::test]
async fn test_sql_injection_prevention() {
    let data_service = DataService::new("test_db").await.unwrap();
    
    // 测试SQL注入攻击
    let malicious_input = "'; DROP TABLE devices; --";
    
    let result = data_service.search_devices(malicious_input).await;
    // 应该安全处理，不执行恶意SQL
    assert!(result.is_ok());
}
```

## 14.6 自动化测试

### 14.6.1 持续集成测试

```yaml
# .github/workflows/test.yml
name: IoT Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: iot_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
    
    - name: Run unit tests
      run: cargo test --lib
    
    - name: Run integration tests
      run: cargo test --test integration_tests
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/iot_test
    
    - name: Run performance tests
      run: cargo test --test performance_tests
    
    - name: Run security tests
      run: cargo test --test security_tests
```

### 14.6.2 测试覆盖率

```rust
// 在Cargo.toml中添加
[profile.test]
opt-level = 0
debug = true

[profile.release]
opt-level = 3
debug = false

// 使用tarpaulin进行覆盖率测试
// cargo install cargo-tarpaulin
// cargo tarpaulin --out Html
```

## 14.7 测试工具与框架

### 14.7.1 模拟与存根

```rust
use mockall::predicate::*;
use mockall::*;

#[automock]
pub trait DeviceInterface {
    async fn read_sensor(&self) -> Result<SensorData>;
    async fn write_actuator(&self, data: &ActuatorData) -> Result<()>;
}

#[tokio::test]
async fn test_device_interface_mock() {
    let mut mock_device = MockDeviceInterface::new();
    
    // 设置模拟行为
    mock_device
        .expect_read_sensor()
        .times(1)
        .returning(|| Ok(SensorData {
            device_id: "mock-device".to_string(),
            timestamp: Utc::now(),
            temperature: 25.0,
            humidity: 60.0,
        }));
    
    mock_device
        .expect_write_actuator()
        .times(1)
        .returning(|_| Ok(()));
    
    // 测试模拟设备
    let sensor_data = mock_device.read_sensor().await.unwrap();
    assert_eq!(sensor_data.temperature, 25.0);
    
    let actuator_data = ActuatorData {
        device_id: "mock-device".to_string(),
        command: "set_temperature".to_string(),
        value: 26.0,
    };
    
    mock_device.write_actuator(&actuator_data).await.unwrap();
}
```

## 14.8 测试架构图

### 14.8.1 测试金字塔

```mermaid
graph TD
    subgraph "端到端测试 (10%)"
        E2E1[用户界面测试]
        E2E2[业务流程测试]
        E2E3[性能测试]
    end
    
    subgraph "集成测试 (20%)"
        INT1[API集成测试]
        INT2[服务间测试]
        INT3[数据库集成测试]
    end
    
    subgraph "单元测试 (70%)"
        UNIT1[组件测试]
        UNIT2[函数测试]
        UNIT3[类测试]
    end
    
    E2E1 --> INT1
    E2E2 --> INT2
    E2E3 --> INT3
    INT1 --> UNIT1
    INT2 --> UNIT2
    INT3 --> UNIT3
```

### 14.8.2 测试流程

```mermaid
graph LR
    subgraph "开发阶段"
        DEV1[代码编写]
        DEV2[单元测试]
        DEV3[代码审查]
    end
    
    subgraph "集成阶段"
        INT1[分支合并]
        INT2[集成测试]
        INT3[安全测试]
    end
    
    subgraph "部署阶段"
        DEP1[性能测试]
        DEP2[端到端测试]
        DEP3[生产部署]
    end
    
    DEV1 --> DEV2
    DEV2 --> DEV3
    DEV3 --> INT1
    INT1 --> INT2
    INT2 --> INT3
    INT3 --> DEP1
    DEP1 --> DEP2
    DEP2 --> DEP3
```

## 14.9 跨主题引用

- 基础理论与行业标准详见[1. IoT基础理论与行业标准](01_Foundation.md)
- 性能与可靠性详见[5. IoT性能与可靠性](05_Performance_Reliability.md)
- 安全与隐私详见[4. IoT安全与隐私](04_Security_Privacy.md)
- 实现指南与最佳实践详见[11. IoT实现指南与最佳实践](11_Implementation_Guide.md)

## 14.10 参考与扩展阅读

- [Rust测试指南](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [Golang测试](https://golang.org/pkg/testing/)
- [测试驱动开发](https://en.wikipedia.org/wiki/Test-driven_development)
- [持续集成最佳实践](https://martinfowler.com/articles/continuousIntegration.html)
