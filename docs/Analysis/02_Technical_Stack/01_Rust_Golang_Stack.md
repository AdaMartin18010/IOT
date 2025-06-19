# 01 Rust/Golang技术栈

## 3.1 核心技术栈概览

### 3.1.1 技术栈分层架构

```mermaid
graph TD
    A[应用层] --> B[框架层]
    B --> C[运行时层]
    C --> D[系统层]
    D --> E[硬件层]
    
    A1[Rust应用] --> A
    A2[Go应用] --> A
    B1[Tokio/async-std] --> B
    B2[Gin/Echo] --> B
    C1[Rust Runtime] --> C
    C2[Go Runtime] --> C
    D1[Linux/RTOS] --> D
    E1[ARM/x86] --> E
```

### 3.1.2 技术选型对比

| 技术领域 | Rust方案 | Go方案 | 适用场景 |
|----------|----------|--------|----------|
| 异步运行时 | Tokio, async-std | goroutines | 高并发I/O |
| Web框架 | Actix-web, Axum | Gin, Echo | API服务 |
| 数据库 | SQLx, Diesel | GORM, sqlx | 数据持久化 |
| 消息队列 | lapin, redis | RabbitMQ, Redis | 异步通信 |
| 序列化 | serde | encoding/json | 数据交换 |
| 配置管理 | config, toml | viper | 配置管理 |

## 3.2 Rust技术栈详解

### 3.2.1 核心依赖配置

```toml
# Cargo.toml
[package]
name = "iot-platform"
version = "0.1.0"
edition = "2021"

[dependencies]
# 异步运行时
tokio = { version = "1.35", features = ["full"] }
async-std = "1.35"

# 网络通信
tokio-mqtt = "0.8"
rumqttc = "0.24"
coap = "0.3"
reqwest = { version = "0.11", features = ["json"] }

# 序列化
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "1.3"

# 数据库
sqlx = { version = "0.7", features = ["sqlite", "runtime-tokio-rustls"] }
rusqlite = "0.29"
sled = "0.34"

# 加密和安全
ring = "0.17"
rustls = "0.21"
webpki-roots = "0.25"

# 配置管理
config = "0.14"
toml = "0.8"

# 日志
tracing = "0.1"
tracing-subscriber = "0.3"
log = "0.4"

# 硬件抽象
embedded-hal = "0.2"
cortex-m = "0.7"
cortex-m-rt = "0.7"

# 传感器支持
embedded-sensors = "0.1"
dht-sensor = "0.1"

# 时间处理
chrono = { version = "0.4", features = ["serde"] }
time = "0.3"

# 消息队列
lapin = "2.3"
redis = { version = "0.24", features = ["tokio-comp"] }

# 缓存
moka = "0.12"
```

### 3.2.2 异步运行时示例

```rust
use tokio::time::{sleep, Duration};
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::main]
async fn main() {
    // 创建异步任务
    let device_manager = Arc::new(RwLock::new(DeviceManager::new()));
    
    // 并发执行多个任务
    let tasks = vec![
        tokio::spawn(device_monitoring_task(device_manager.clone())),
        tokio::spawn(data_processing_task(device_manager.clone())),
        tokio::spawn(communication_task(device_manager.clone())),
    ];
    
    // 等待所有任务完成
    for task in tasks {
        task.await.unwrap();
    }
}

async fn device_monitoring_task(device_manager: Arc<RwLock<DeviceManager>>) {
    loop {
        {
            let mut manager = device_manager.write().await;
            manager.check_device_status().await;
        }
        sleep(Duration::from_secs(30)).await;
    }
}

async fn data_processing_task(device_manager: Arc<RwLock<DeviceManager>>) {
    loop {
        {
            let manager = device_manager.read().await;
            manager.process_sensor_data().await;
        }
        sleep(Duration::from_secs(5)).await;
    }
}

async fn communication_task(device_manager: Arc<RwLock<DeviceManager>>) {
    loop {
        {
            let manager = device_manager.read().await;
            manager.sync_with_cloud().await;
        }
        sleep(Duration::from_secs(60)).await;
    }
}
```

### 3.2.3 MQTT通信示例

```rust
use rumqttc::{AsyncClient, EventLoop, MqttOptions, QoS};
use tokio::time::{sleep, Duration};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct SensorData {
    device_id: String,
    sensor_type: String,
    value: f64,
    timestamp: i64,
}

async fn mqtt_communication() -> Result<(), Box<dyn std::error::Error>> {
    let mut mqtt_options = MqttOptions::new("iot-device-001", "mqtt.example.com", 1883);
    mqtt_options.set_keep_alive(Duration::from_secs(5));
    
    let (client, mut eventloop) = AsyncClient::new(mqtt_options, 10);
    
    // 订阅主题
    client.subscribe("device/+/sensor", QoS::AtLeastOnce).await?;
    
    // 发布数据
    let sensor_data = SensorData {
        device_id: "device-001".to_string(),
        sensor_type: "temperature".to_string(),
        value: 25.5,
        timestamp: chrono::Utc::now().timestamp(),
    };
    
    let payload = serde_json::to_vec(&sensor_data)?;
    client.publish("device/001/sensor", QoS::AtLeastOnce, false, payload).await?;
    
    // 处理接收到的消息
    loop {
        match eventloop.poll().await {
            Ok(notification) => {
                match notification {
                    rumqttc::Event::Incoming(rumqttc::Packet::Publish(msg)) => {
                        if let Ok(data) = serde_json::from_slice::<SensorData>(&msg.payload) {
                            println!("Received: {:?}", data);
                        }
                    }
                    _ => {}
                }
            }
            Err(e) => {
                println!("Error: {:?}", e);
                sleep(Duration::from_secs(1)).await;
            }
        }
    }
}
```

## 3.3 Go技术栈详解

### 3.3.1 核心依赖配置

```go
// go.mod
module iot-platform

go 1.21

require (
    // Web框架
    github.com/gin-gonic/gin v1.9.1
    github.com/labstack/echo/v4 v4.11.1
    
    // 数据库
    gorm.io/gorm v1.25.4
    gorm.io/driver/postgres v1.5.2
    gorm.io/driver/sqlite v1.5.2
    
    // 消息队列
    github.com/streadway/amqp v1.0.0
    github.com/go-redis/redis/v8 v8.11.5
    
    // MQTT
    github.com/eclipse/paho.mqtt.golang v1.4.2
    
    // 配置管理
    github.com/spf13/viper v1.16.0
    
    // 日志
    go.uber.org/zap v1.24.0
    
    // 时间处理
    github.com/jinzhu/now v1.1.5
    
    // 加密
    golang.org/x/crypto v0.13.0
    
    // 序列化
    github.com/json-iterator/go v1.1.12
)
```

### 3.3.2 并发处理示例

```go
package main

import (
    "context"
    "log"
    "sync"
    "time"
    
    "github.com/gin-gonic/gin"
    "go.uber.org/zap"
)

type DeviceManager struct {
    devices map[string]*Device
    mu      sync.RWMutex
    logger  *zap.Logger
}

func NewDeviceManager() *DeviceManager {
    logger, _ := zap.NewProduction()
    return &DeviceManager{
        devices: make(map[string]*Device),
        logger:  logger,
    }
}

func (dm *DeviceManager) StartMonitoring(ctx context.Context) {
    // 启动设备监控协程
    go dm.deviceMonitoringTask(ctx)
    
    // 启动数据处理协程
    go dm.dataProcessingTask(ctx)
    
    // 启动通信协程
    go dm.communicationTask(ctx)
}

func (dm *DeviceManager) deviceMonitoringTask(ctx context.Context) {
    ticker := time.NewTicker(30 * time.Second)
    defer ticker.Stop()
    
    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            dm.checkDeviceStatus()
        }
    }
}

func (dm *DeviceManager) dataProcessingTask(ctx context.Context) {
    ticker := time.NewTicker(5 * time.Second)
    defer ticker.Stop()
    
    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            dm.processSensorData()
        }
    }
}

func (dm *DeviceManager) communicationTask(ctx context.Context) {
    ticker := time.NewTicker(60 * time.Second)
    defer ticker.Stop()
    
    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            dm.syncWithCloud()
        }
    }
}

func (dm *DeviceManager) checkDeviceStatus() {
    dm.mu.RLock()
    defer dm.mu.RUnlock()
    
    for deviceID, device := range dm.devices {
        if time.Since(device.LastSeen) > 5*time.Minute {
            dm.logger.Warn("Device offline", zap.String("device_id", deviceID))
        }
    }
}

func (dm *DeviceManager) processSensorData() {
    // 处理传感器数据
    dm.logger.Info("Processing sensor data")
}

func (dm *DeviceManager) syncWithCloud() {
    // 与云端同步
    dm.logger.Info("Syncing with cloud")
}
```

### 3.3.3 MQTT通信示例

```go
package main

import (
    "encoding/json"
    "log"
    "time"
    
    MQTT "github.com/eclipse/paho.mqtt.golang"
)

type SensorData struct {
    DeviceID   string  `json:"device_id"`
    SensorType string  `json:"sensor_type"`
    Value      float64 `json:"value"`
    Timestamp  int64   `json:"timestamp"`
}

func mqttCommunication() error {
    // MQTT客户端配置
    opts := MQTT.NewClientOptions()
    opts.AddBroker("tcp://mqtt.example.com:1883")
    opts.SetClientID("iot-device-001")
    opts.SetKeepAlive(5 * time.Second)
    
    // 创建客户端
    client := MQTT.NewClient(opts)
    if token := client.Connect(); token.Wait() && token.Error() != nil {
        return token.Error()
    }
    
    // 订阅主题
    if token := client.Subscribe("device/+/sensor", 1, func(client MQTT.Client, msg MQTT.Message) {
        var sensorData SensorData
        if err := json.Unmarshal(msg.Payload(), &sensorData); err != nil {
            log.Printf("Error unmarshaling message: %v", err)
            return
        }
        log.Printf("Received: %+v", sensorData)
    }); token.Wait() && token.Error() != nil {
        return token.Error()
    }
    
    // 发布数据
    sensorData := SensorData{
        DeviceID:   "device-001",
        SensorType: "temperature",
        Value:      25.5,
        Timestamp:  time.Now().Unix(),
    }
    
    payload, err := json.Marshal(sensorData)
    if err != nil {
        return err
    }
    
    if token := client.Publish("device/001/sensor", 1, false, payload); token.Wait() && token.Error() != nil {
        return token.Error()
    }
    
    // 保持连接
    select {}
}
```

## 3.4 性能优化策略

### 3.4.1 Rust性能优化

```rust
// 使用内存池减少分配
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Clone)]
struct DataProcessor {
    pool: Arc<rayon::ThreadPool>,
    sender: mpsc::Sender<SensorData>,
}

impl DataProcessor {
    pub fn new() -> Self {
        let pool = Arc::new(rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap());
        
        let (sender, mut receiver) = mpsc::channel(1000);
        
        // 启动处理协程
        let pool_clone = pool.clone();
        tokio::spawn(async move {
            while let Some(data) = receiver.recv().await {
                pool_clone.install(|| {
                    // 在专用线程池中处理数据
                    process_sensor_data(data);
                });
            }
        });
        
        Self { pool, sender }
    }
    
    pub async fn process(&self, data: SensorData) {
        let _ = self.sender.send(data).await;
    }
}

fn process_sensor_data(data: SensorData) {
    // 数据处理逻辑
    println!("Processing: {:?}", data);
}
```

### 3.4.2 Go性能优化

```go
package main

import (
    "context"
    "sync"
    "time"
    
    "github.com/panjf2000/ants/v2"
)

type DataProcessor struct {
    pool   *ants.Pool
    buffer chan SensorData
}

func NewDataProcessor() (*DataProcessor, error) {
    // 创建协程池
    pool, err := ants.NewPool(10, ants.WithPreAlloc(true))
    if err != nil {
        return nil, err
    }
    
    buffer := make(chan SensorData, 1000)
    
    dp := &DataProcessor{
        pool:   pool,
        buffer: buffer,
    }
    
    // 启动处理协程
    go dp.processLoop()
    
    return dp, nil
}

func (dp *DataProcessor) processLoop() {
    for data := range dp.buffer {
        dp.pool.Submit(func() {
            dp.processSensorData(data)
        })
    }
}

func (dp *DataProcessor) processSensorData(data SensorData) {
    // 数据处理逻辑
    log.Printf("Processing: %+v", data)
}

func (dp *DataProcessor) Process(data SensorData) {
    select {
    case dp.buffer <- data:
    default:
        // 缓冲区满，丢弃数据
        log.Printf("Buffer full, dropping data: %+v", data)
    }
}
```

## 3.5 部署与配置

### 3.5.1 Docker部署

```dockerfile
# Rust应用Dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bullseye-slim
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
    
WORKDIR /app
COPY --from=builder /app/target/release/iot-platform .
COPY config/ config/

EXPOSE 8080
CMD ["./iot-platform"]
```

```dockerfile
# Go应用Dockerfile
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o iot-platform .

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/iot-platform .
COPY config/ config/

EXPOSE 8080
CMD ["./iot-platform"]
```

### 3.5.2 配置文件示例

```toml
# config.toml (Rust)
[server]
host = "0.0.0.0"
port = 8080
workers = 4

[database]
url = "postgresql://user:pass@localhost/iot_db"
max_connections = 10

[mqtt]
broker = "mqtt.example.com"
port = 1883
client_id = "iot-platform"
username = "user"
password = "pass"

[redis]
url = "redis://localhost:6379"
pool_size = 10

[logging]
level = "info"
format = "json"
```

```yaml
# config.yaml (Go)
server:
  host: "0.0.0.0"
  port: 8080
  workers: 4

database:
  url: "postgresql://user:pass@localhost/iot_db"
  max_connections: 10

mqtt:
  broker: "mqtt.example.com"
  port: 1883
  client_id: "iot-platform"
  username: "user"
  password: "pass"

redis:
  url: "redis://localhost:6379"
  pool_size: 10

logging:
  level: "info"
  format: "json"
```

---

> 本文档详细描述了IOT平台的核心技术栈，包含Rust和Go两种实现方案，涵盖异步处理、通信协议、性能优化和部署配置。
