# IoT Things 实施工具支持

## 1. 开发环境配置

### 1.1 核心开发工具

#### 1.1.1 Rust开发环境

```bash
#!/bin/bash
# Rust IoT开发环境配置脚本

echo "配置Rust IoT开发环境..."

# 安装Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 安装必要的工具链
rustup toolchain install stable
rustup toolchain install nightly
rustup default stable

# 安装开发工具
cargo install cargo-watch
cargo install cargo-audit
cargo install cargo-tarpaulin
cargo install cargo-doc
cargo install cargo-expand

# 安装IoT相关crate
cargo install tokio-console
cargo install async-std
cargo install serde_json
cargo install reqwest
cargo install sqlx-cli

echo "Rust开发环境配置完成！"
```

#### 1.1.2 Go开发环境

```bash
#!/bin/bash
# Go IoT开发环境配置脚本

echo "配置Go IoT开发环境..."

# 安装Go
wget https://golang.org/dl/go1.21.0.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.0.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin

# 配置Go模块
go env -w GO111MODULE=on
go env -w GOPROXY=https://goproxy.cn,direct

# 安装开发工具
go install golang.org/x/tools/cmd/goimports@latest
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
go install github.com/go-delve/delve/cmd/dlv@latest
go install github.com/cosmtrek/air@latest

# 安装IoT相关包
go get github.com/eclipse/paho.mqtt.golang
go get github.com/plgd-dev/go-coap/v3
go get github.com/edgexfoundry/go-mod-core-contracts
go get github.com/edgexfoundry/device-sdk-go

echo "Go开发环境配置完成！"
```

### 1.2 IDE配置

#### 1.2.1 VSCode配置

```json
{
  "workbench.colorTheme": "Dark+ (default dark)",
  "editor.fontSize": 14,
  "editor.tabSize": 2,
  "editor.insertSpaces": true,
  "files.autoSave": "onFocusChange",
  
  "rust-analyzer.checkOnSave.command": "clippy",
  "rust-analyzer.cargo.buildScripts.enable": true,
  "rust-analyzer.procMacro.enable": true,
  
  "go.useLanguageServer": true,
  "go.lintOnSave": "package",
  "go.vetOnSave": "package",
  
  "extensions.recommendations": [
    "rust-lang.rust-analyzer",
    "golang.go",
    "ms-vscode.vscode-json",
    "redhat.vscode-yaml",
    "ms-vscode.vscode-docker",
    "ms-kubernetes-tools.vscode-kubernetes-tools"
  ]
}
```

## 2. 测试工具链

### 2.1 单元测试框架

#### 2.1.1 Rust测试配置

```rust
// tests/unit_tests.rs
use iot_things::models::IoTThing;
use iot_things::services::ThingService;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_thing_creation() {
        let thing = IoTThing::new(
            "test_thing_001",
            "测试设备",
            "Sensor",
            "TemperatureSensor"
        );
        
        assert_eq!(thing.id(), "test_thing_001");
        assert_eq!(thing.name(), "测试设备");
        assert_eq!(thing.thing_type(), "Sensor");
    }
    
    #[test]
    fn test_thing_state_transition() {
        let mut thing = IoTThing::new(
            "test_thing_002",
            "测试设备2",
            "Sensor",
            "TemperatureSensor"
        );
        
        assert_eq!(thing.state(), "Initializing");
        
        thing.activate();
        assert_eq!(thing.state(), "Active");
        
        thing.deactivate();
        assert_eq!(thing.state(), "Inactive");
    }
    
    #[tokio::test]
    async fn test_thing_service() {
        let service = ThingService::new();
        
        let thing = IoTThing::new(
            "test_thing_003",
            "测试设备3",
            "Sensor",
            "TemperatureSensor"
        );
        
        let result = service.register_thing(thing).await;
        assert!(result.is_ok());
    }
}
```

#### 2.1.2 Go测试配置

```go
// internal/services/thing_service_test.go
package services

import (
    "testing"
    "context"
    
    "iot-things/internal/models"
)

func TestThingService_CreateThing(t *testing.T) {
    service := NewThingService()
    
    thing := &models.IoTThing{
        ID:         "test_thing_001",
        Name:       "测试设备",
        ThingType:  "Sensor",
        Category:   "TemperatureSensor",
        State:      "Initializing",
    }
    
    result, err := service.CreateThing(context.Background(), thing)
    if err != nil {
        t.Fatalf("创建Thing失败: %v", err)
    }
    
    if result.ID != thing.ID {
        t.Errorf("期望ID: %s, 实际ID: %s", thing.ID, result.ID)
    }
}

func TestThingService_UpdateState(t *testing.T) {
    service := NewThingService()
    
    thing := &models.IoTThing{
        ID:         "test_thing_002",
        Name:       "测试设备2",
        ThingType:  "Sensor",
        Category:   "TemperatureSensor",
        State:      "Initializing",
    }
    
    // 创建Thing
    _, err := service.CreateThing(context.Background(), thing)
    if err != nil {
        t.Fatalf("创建Thing失败: %v", err)
    }
    
    // 更新状态
    err = service.UpdateState(context.Background(), thing.ID, "Active")
    if err != nil {
        t.Fatalf("更新状态失败: %v", err)
    }
    
    // 验证状态
    updatedThing, err := service.GetThing(context.Background(), thing.ID)
    if err != nil {
        t.Fatalf("获取Thing失败: %v", err)
    }
    
    if updatedThing.State != "Active" {
        t.Errorf("期望状态: Active, 实际状态: %s", updatedThing.State)
    }
}
```

### 2.2 集成测试

#### 2.2.1 Docker测试环境

```yaml
# docker-compose.test.yml
version: '3.8'

services:
  # 数据库服务
  postgres-test:
    image: postgres:15
    environment:
      POSTGRES_DB: iot_things_test
      POSTGRES_USER: test_user
      POSTGRES_PASSWORD: test_password
    ports:
      - "5433:5432"
    volumes:
      - postgres_test_data:/var/lib/postgresql/data

  # Redis缓存
  redis-test:
    image: redis:7-alpine
    ports:
      - "6380:6379"
    volumes:
      - redis_test_data:/data

  # MQTT代理
  mqtt-test:
    image: eclipse-mosquitto:2.0
    ports:
      - "1884:1883"
      - "9002:9001"
    volumes:
      - ./test/mosquitto.conf:/mosquitto/config/mosquitto.conf

  # 测试应用
  app-test:
    build:
      context: .
      dockerfile: Dockerfile.test
    environment:
      DATABASE_URL: postgresql://test_user:test_password@postgres-test:5432/iot_things_test
      REDIS_URL: redis://redis-test:6379
      MQTT_BROKER: mqtt-test:1883
    depends_on:
      - postgres-test
      - redis-test
      - mqtt-test
    volumes:
      - ./test:/app/test
      - ./coverage:/app/coverage

volumes:
  postgres_test_data:
  redis_test_data:
```

## 3. 部署工具

### 3.1 Docker部署

#### 3.1.1 Dockerfile配置

```dockerfile
# Dockerfile
FROM rust:1.75 as builder

WORKDIR /app

# 复制依赖文件
COPY Cargo.toml Cargo.lock ./

# 创建虚拟项目以缓存依赖
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src

# 复制源代码
COPY src ./src

# 构建应用
RUN cargo build --release

# 运行时镜像
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 复制二进制文件
COPY --from=builder /app/target/release/iot-things .

# 复制配置文件
COPY config ./config

# 创建非root用户
RUN useradd -r -s /bin/false iot
RUN chown -R iot:iot /app
USER iot

EXPOSE 8080

CMD ["./iot-things"]
```

#### 3.1.2 Docker Compose配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  # IoT Things服务
  iot-things:
    build: .
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgresql://iot_user:iot_password@postgres:5432/iot_things
      - REDIS_URL=redis://redis:6379
      - MQTT_BROKER=mqtt:1883
      - LOG_LEVEL=info
    depends_on:
      - postgres
      - redis
      - mqtt
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  # PostgreSQL数据库
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: iot_things
      POSTGRES_USER: iot_user
      POSTGRES_PASSWORD: iot_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  # Redis缓存
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped

  # MQTT代理
  mqtt:
    image: eclipse-mosquitto:2.0
    volumes:
      - ./mosquitto.conf:/mosquitto/config/mosquitto.conf
      - mqtt_data:/mosquitto/data
      - mqtt_logs:/mosquitto/log
    ports:
      - "1883:1883"
      - "9001:9001"
    restart: unless-stopped

  # 监控服务
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    restart: unless-stopped

  # 可视化服务
  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  mqtt_data:
  mqtt_logs:
  prometheus_data:
  grafana_data:
```

### 3.2 Kubernetes部署

#### 3.2.1 Kubernetes配置

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: iot-things
  labels:
    name: iot-things
```

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: iot-things-config
  namespace: iot-things
data:
  database_url: "postgresql://iot_user:iot_password@postgres:5432/iot_things"
  redis_url: "redis://redis:6379"
  mqtt_broker: "mqtt:1883"
  log_level: "info"
```

```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: iot-things-secret
  namespace: iot-things
type: Opaque
data:
  database_password: aW90X3Bhc3N3b3Jk  # base64 encoded
  jwt_secret: anN0X3NlY3JldA==  # base64 encoded
```

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iot-things
  namespace: iot-things
spec:
  replicas: 3
  selector:
    matchLabels:
      app: iot-things
  template:
    metadata:
      labels:
        app: iot-things
    spec:
      containers:
      - name: iot-things
        image: iot-things:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: iot-things-config
              key: database_url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: iot-things-config
              key: redis_url
        - name: MQTT_BROKER
          valueFrom:
            configMapKeyRef:
              name: iot-things-config
              key: mqtt_broker
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: iot-things-service
  namespace: iot-things
spec:
  selector:
    app: iot-things
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: iot-things-ingress
  namespace: iot-things
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: iot-things.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: iot-things-service
            port:
              number: 80
```

## 4. 监控和日志

### 4.1 监控配置

#### 4.1.1 Prometheus配置

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "iot_things_rules.yml"

scrape_configs:
  - job_name: 'iot-things'
    static_configs:
      - targets: ['iot-things:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'mqtt'
    static_configs:
      - targets: ['mqtt-exporter:9210']
```

#### 4.1.2 Grafana仪表板

```json
{
  "dashboard": {
    "title": "IoT Things 监控仪表板",
    "panels": [
      {
        "title": "设备状态分布",
        "type": "pie",
        "targets": [
          {
            "expr": "iot_things_device_state_total",
            "legendFormat": "{{state}}"
          }
        ]
      },
      {
        "title": "消息吞吐量",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(iot_things_messages_total[5m])",
            "legendFormat": "消息/秒"
          }
        ]
      },
      {
        "title": "响应时间",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(iot_things_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95%响应时间"
          }
        ]
      }
    ]
  }
}
```

### 4.2 日志配置

#### 4.2.1 结构化日志

```rust
// src/logging.rs
use tracing::{info, error, warn, debug};
use tracing_subscriber::{fmt, EnvFilter};

pub fn init_logging() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));
    
    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_file(true)
        .with_line_number(true)
        .init();
}

pub fn log_thing_event(thing_id: &str, event: &str, details: &str) {
    info!(
        thing_id = %thing_id,
        event = %event,
        details = %details,
        "Thing事件"
    );
}

pub fn log_message_sent(sender: &str, receiver: &str, message_type: &str) {
    info!(
        sender = %sender,
        receiver = %receiver,
        message_type = %message_type,
        "消息发送"
    );
}
```

## 5. 自动化脚本

### 5.1 构建脚本

```bash
#!/bin/bash
# 自动化构建脚本

set -e

echo "开始构建IoT Things项目..."

# 检查依赖
echo "检查依赖..."
cargo check
go mod tidy

# 运行测试
echo "运行测试..."
cargo test
go test ./...

# 代码质量检查
echo "代码质量检查..."
cargo clippy -- -D warnings
golangci-lint run

# 构建
echo "构建项目..."
cargo build --release
go build -o bin/iot-things-go cmd/server/main.go

# 生成文档
echo "生成文档..."
cargo doc --no-deps
godoc -http=:6060 &

echo "构建完成！"
```

### 5.2 部署脚本

```bash
#!/bin/bash
# 自动化部署脚本

set -e

# 配置变量
PROJECT_NAME="iot-things"
NAMESPACE="iot-things"
REGISTRY="your-registry.com"
VERSION=$(git describe --tags --always)

echo "开始部署 $PROJECT_NAME v$VERSION..."

# 1. 构建镜像
echo "构建Docker镜像..."
docker build -t $REGISTRY/$PROJECT_NAME:$VERSION .
docker tag $REGISTRY/$PROJECT_NAME:$VERSION $REGISTRY/$PROJECT_NAME:latest

# 2. 推送镜像
echo "推送镜像到仓库..."
docker push $REGISTRY/$PROJECT_NAME:$VERSION
docker push $REGISTRY/$PROJECT_NAME:latest

# 3. 部署到Kubernetes
echo "部署到Kubernetes..."

# 创建命名空间
kubectl apply -f k8s/namespace.yaml

# 应用配置
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# 更新部署镜像
kubectl set image deployment/$PROJECT_NAME $PROJECT_NAME=$REGISTRY/$PROJECT_NAME:$VERSION -n $NAMESPACE

# 应用服务配置
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# 4. 等待部署完成
echo "等待部署完成..."
kubectl rollout status deployment/$PROJECT_NAME -n $NAMESPACE

# 5. 健康检查
echo "执行健康检查..."
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE

echo "部署完成！"
echo "服务地址: http://iot-things.example.com"
```

## 6. 总结

本实施工具支持文档提供了：

1. **开发环境配置**：Rust、Go开发环境，IDE配置
2. **测试工具链**：单元测试、集成测试、Docker测试环境
3. **部署工具**：Docker、Kubernetes部署配置
4. **监控和日志**：Prometheus、Grafana监控，结构化日志
5. **自动化脚本**：构建和部署自动化脚本

这些工具为IoT Things系统的开发、测试、部署和运维提供了完整的支持。
