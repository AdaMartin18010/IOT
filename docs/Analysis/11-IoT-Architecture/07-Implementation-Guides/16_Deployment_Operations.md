# 16. IoT部署运维与DevOps

## 16.1 部署策略概述

### 16.1.1 部署模式

**定义16.1（部署模式）**：\( DeploymentPattern = \{BlueGreen, Rolling, Canary, A/B\} \)

### 16.1.2 部署环境

- **开发环境**：功能开发和单元测试
- **测试环境**：集成测试和性能测试
- **预生产环境**：生产环境模拟
- **生产环境**：实际业务运行

## 16.2 容器化部署

### 16.2.1 Docker容器化

```dockerfile
# IoT Edge Node Dockerfile
FROM rust:1.70 as builder

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY Cargo.toml Cargo.lock ./

# 构建依赖
RUN cargo build --release --bin iot-edge-node

# 复制源代码
COPY src/ ./src/

# 构建应用
RUN cargo build --release --bin iot-edge-node

# 运行阶段
FROM debian:bullseye-slim

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/*

# 创建非root用户
RUN useradd -r -s /bin/false iotuser

# 设置工作目录
WORKDIR /app

# 复制二进制文件
COPY --from=builder /app/target/release/iot-edge-node .

# 复制配置文件
COPY config/ ./config/

# 设置权限
RUN chown -R iotuser:iotuser /app

# 切换到非root用户
USER iotuser

# 暴露端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 启动命令
CMD ["./iot-edge-node"]
```

### 16.2.2 Docker Compose配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  iot-edge-node:
    build: .
    container_name: iot-edge-node
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgresql://iotuser:iotpass@postgres:5432/iotdb
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=info
    depends_on:
      - postgres
      - redis
    volumes:
      - ./config:/app/config
      - iot-data:/app/data
    restart: unless-stopped
    networks:
      - iot-network

  postgres:
    image: postgres:13
    container_name: iot-postgres
    environment:
      - POSTGRES_DB=iotdb
      - POSTGRES_USER=iotuser
      - POSTGRES_PASSWORD=iotpass
    volumes:
      - postgres-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - iot-network

  redis:
    image: redis:6-alpine
    container_name: iot-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - iot-network

  prometheus:
    image: prom/prometheus
    container_name: iot-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    networks:
      - iot-network

  grafana:
    image: grafana/grafana
    container_name: iot-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    networks:
      - iot-network

volumes:
  iot-data:
  postgres-data:
  redis-data:
  prometheus-data:
  grafana-data:

networks:
  iot-network:
    driver: bridge
```

## 16.3 Kubernetes部署

### 16.3.1 Kubernetes配置

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iot-edge-node
  labels:
    app: iot-edge-node
spec:
  replicas: 3
  selector:
    matchLabels:
      app: iot-edge-node
  template:
    metadata:
      labels:
        app: iot-edge-node
    spec:
      containers:
      - name: iot-edge-node
        image: iot-edge-node:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: iot-secrets
              key: database-url
        - name: REDIS_URL
          value: "redis://iot-redis:6379"
        - name: LOG_LEVEL
          value: "info"
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
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: data-volume
          mountPath: /app/data
      volumes:
      - name: config-volume
        configMap:
          name: iot-config
      - name: data-volume
        persistentVolumeClaim:
          claimName: iot-data-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: iot-edge-service
spec:
  selector:
    app: iot-edge-node
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: iot-config
data:
  app.toml: |
    [server]
    host = "0.0.0.0"
    port = 8080
    
    [database]
    url = "postgresql://iotuser:iotpass@iot-postgres:5432/iotdb"
    
    [redis]
    url = "redis://iot-redis:6379"

---
apiVersion: v1
kind: Secret
metadata:
  name: iot-secrets
type: Opaque
data:
  database-url: cG9zdGdyZXNxbDovL2lvdHVzZXI6aW90cGFzc0Bpb3QtcG9zdGdyZXM6NTQzMi9pb3RkYg==

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: iot-data-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

### 16.3.2 Helm Chart

```yaml
# Chart.yaml
apiVersion: v2
name: iot-edge
description: IoT Edge Node Helm Chart
version: 1.0.0
appVersion: "1.0.0"

# values.yaml
replicaCount: 3

image:
  repository: iot-edge-node
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: LoadBalancer
  port: 80

resources:
  requests:
    memory: 256Mi
    cpu: 250m
  limits:
    memory: 512Mi
    cpu: 500m

config:
  database:
    url: "postgresql://iotuser:iotpass@iot-postgres:5432/iotdb"
  redis:
    url: "redis://iot-redis:6379"
  logLevel: "info"

persistence:
  enabled: true
  size: 10Gi

# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "iot-edge.fullname" . }}
  labels:
    {{- include "iot-edge.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "iot-edge.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "iot-edge.selectorLabels" . | nindent 8 }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: 8080
              protocol: TCP
          env:
            - name: DATABASE_URL
              value: {{ .Values.config.database.url | quote }}
            - name: REDIS_URL
              value: {{ .Values.config.redis.url | quote }}
            - name: LOG_LEVEL
              value: {{ .Values.config.logLevel | quote }}
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
```

## 16.4 CI/CD流水线

### 16.4.1 GitHub Actions

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

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
    - uses: actions/checkout@v3
    
    - name: Setup Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
    
    - name: Run tests
      run: cargo test --verbose
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/iot_test
    
    - name: Run clippy
      run: cargo clippy -- -D warnings
    
    - name: Run security audit
      run: cargo audit

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'latest'
    
    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > kubeconfig.yaml
        export KUBECONFIG=kubeconfig.yaml
    
    - name: Deploy to Kubernetes
      run: |
        kubectl apply -f k8s/
        kubectl rollout status deployment/iot-edge-node
```

### 16.4.2 GitLab CI/CD

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"

test:
  stage: test
  image: rust:1.70
  services:
    - postgres:13
  variables:
    POSTGRES_DB: iot_test
    POSTGRES_USER: postgres
    POSTGRES_PASSWORD: postgres
    DATABASE_URL: postgresql://postgres:postgres@postgres:5432/iot_test
  script:
    - cargo test --verbose
    - cargo clippy -- -D warnings
    - cargo audit
  only:
    - main
    - develop
    - merge_requests

build:
  stage: build
  image: docker:20.10.16
  services:
    - docker:20.10.16-dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - |
      if [ "$CI_COMMIT_BRANCH" = "main" ]; then
        docker tag $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA $CI_REGISTRY_IMAGE:latest
        docker push $CI_REGISTRY_IMAGE:latest
      fi
  only:
    - main
    - develop

deploy:staging:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context $KUBE_CONTEXT_STAGING
    - kubectl set image deployment/iot-edge-node iot-edge-node=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - kubectl rollout status deployment/iot-edge-node
  environment:
    name: staging
  only:
    - develop

deploy:production:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context $KUBE_CONTEXT_PRODUCTION
    - kubectl set image deployment/iot-edge-node iot-edge-node=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - kubectl rollout status deployment/iot-edge-node
  environment:
    name: production
  when: manual
  only:
    - main
```

## 16.5 运维自动化

### 16.5.1 自动化脚本

```rust
use std::process::Command;
use tokio::time::{sleep, Duration};

pub struct DevOpsAutomation {
    kubectl_path: String,
    helm_path: String,
}

impl DevOpsAutomation {
    pub fn new() -> Self {
        Self {
            kubectl_path: "kubectl".to_string(),
            helm_path: "helm".to_string(),
        }
    }
    
    pub async fn deploy_application(&self, image_tag: &str, environment: &str) -> Result<(), Error> {
        println!("开始部署应用到环境: {}", environment);
        
        // 更新镜像标签
        self.update_image_tag(image_tag).await?;
        
        // 执行滚动更新
        self.rollout_deployment().await?;
        
        // 等待部署完成
        self.wait_for_deployment().await?;
        
        // 运行健康检查
        self.run_health_checks().await?;
        
        println!("部署完成");
        Ok(())
    }
    
    async fn update_image_tag(&self, image_tag: &str) -> Result<(), Error> {
        let output = Command::new(&self.kubectl_path)
            .args(&["set", "image", "deployment/iot-edge-node", 
                   &format!("iot-edge-node=iot-edge-node:{}", image_tag)])
            .output()?;
        
        if !output.status.success() {
            return Err(Error::DeploymentFailed(
                String::from_utf8_lossy(&output.stderr).to_string()
            ));
        }
        
        Ok(())
    }
    
    async fn rollout_deployment(&self) -> Result<(), Error> {
        let output = Command::new(&self.kubectl_path)
            .args(&["rollout", "restart", "deployment/iot-edge-node"])
            .output()?;
        
        if !output.status.success() {
            return Err(Error::DeploymentFailed(
                String::from_utf8_lossy(&output.stderr).to_string()
            ));
        }
        
        Ok(())
    }
    
    async fn wait_for_deployment(&self) -> Result<(), Error> {
        let output = Command::new(&self.kubectl_path)
            .args(&["rollout", "status", "deployment/iot-edge-node"])
            .output()?;
        
        if !output.status.success() {
            return Err(Error::DeploymentFailed(
                String::from_utf8_lossy(&output.stderr).to_string()
            ));
        }
        
        Ok(())
    }
    
    async fn run_health_checks(&self) -> Result<(), Error> {
        // 等待服务启动
        sleep(Duration::from_secs(30)).await;
        
        // 检查健康端点
        let output = Command::new("curl")
            .args(&["-f", "http://localhost:8080/health"])
            .output()?;
        
        if !output.status.success() {
            return Err(Error::HealthCheckFailed);
        }
        
        Ok(())
    }
    
    pub async fn scale_application(&self, replicas: u32) -> Result<(), Error> {
        let output = Command::new(&self.kubectl_path)
            .args(&["scale", "deployment/iot-edge-node", 
                   &format!("--replicas={}", replicas)])
            .output()?;
        
        if !output.status.success() {
            return Err(Error::ScalingFailed(
                String::from_utf8_lossy(&output.stderr).to_string()
            ));
        }
        
        println!("应用已扩展到 {} 个副本", replicas);
        Ok(())
    }
    
    pub async fn backup_database(&self) -> Result<(), Error> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let backup_file = format!("iot_backup_{}.sql", timestamp);
        
        let output = Command::new("pg_dump")
            .args(&[
                "-h", "localhost",
                "-U", "iotuser",
                "-d", "iotdb",
                "-f", &backup_file
            ])
            .env("PGPASSWORD", "iotpass")
            .output()?;
        
        if !output.status.success() {
            return Err(Error::BackupFailed(
                String::from_utf8_lossy(&output.stderr).to_string()
            ));
        }
        
        println!("数据库备份完成: {}", backup_file);
        Ok(())
    }
}
```

### 16.5.2 监控自动化

```rust
pub struct MonitoringAutomation {
    prometheus_client: reqwest::Client,
    alert_manager_client: reqwest::Client,
}

impl MonitoringAutomation {
    pub fn new() -> Self {
        Self {
            prometheus_client: reqwest::Client::new(),
            alert_manager_client: reqwest::Client::new(),
        }
    }
    
    pub async fn check_system_health(&self) -> Result<SystemHealth, Error> {
        let metrics = self.get_system_metrics().await?;
        let alerts = self.get_active_alerts().await?;
        
        let health = SystemHealth {
            status: self.calculate_health_status(&metrics, &alerts),
            metrics,
            alerts,
            timestamp: Utc::now(),
        };
        
        Ok(health)
    }
    
    async fn get_system_metrics(&self) -> Result<SystemMetrics, Error> {
        let response = self.prometheus_client
            .get("http://prometheus:9090/api/v1/query")
            .query(&[("query", "up")])
            .send()
            .await?;
        
        let metrics: PrometheusResponse = response.json().await?;
        Ok(SystemMetrics::from_prometheus(metrics))
    }
    
    async fn get_active_alerts(&self) -> Result<Vec<Alert>, Error> {
        let response = self.alert_manager_client
            .get("http://alertmanager:9093/api/v1/alerts")
            .send()
            .await?;
        
        let alerts: Vec<Alert> = response.json().await?;
        Ok(alerts.into_iter().filter(|a| a.state == "active").collect())
    }
    
    fn calculate_health_status(&self, metrics: &SystemMetrics, alerts: &[Alert]) -> HealthStatus {
        if alerts.iter().any(|a| a.severity == "critical") {
            HealthStatus::Critical
        } else if alerts.iter().any(|a| a.severity == "warning") {
            HealthStatus::Warning
        } else if metrics.cpu_usage > 80.0 || metrics.memory_usage > 80.0 {
            HealthStatus::Warning
        } else {
            HealthStatus::Healthy
        }
    }
    
    pub async fn auto_scale_based_on_metrics(&self) -> Result<(), Error> {
        let metrics = self.get_system_metrics().await?;
        
        if metrics.cpu_usage > 80.0 {
            // 自动扩容
            self.scale_up().await?;
        } else if metrics.cpu_usage < 30.0 {
            // 自动缩容
            self.scale_down().await?;
        }
        
        Ok(())
    }
    
    async fn scale_up(&self) -> Result<(), Error> {
        let current_replicas = self.get_current_replicas().await?;
        let new_replicas = (current_replicas as f64 * 1.5).ceil() as u32;
        
        self.update_replicas(new_replicas).await?;
        println!("自动扩容到 {} 个副本", new_replicas);
        Ok(())
    }
    
    async fn scale_down(&self) -> Result<(), Error> {
        let current_replicas = self.get_current_replicas().await?;
        let new_replicas = std::cmp::max(1, (current_replicas as f64 * 0.7).ceil() as u32);
        
        self.update_replicas(new_replicas).await?;
        println!("自动缩容到 {} 个副本", new_replicas);
        Ok(())
    }
}
```

## 16.6 故障处理

### 16.6.1 故障检测与恢复

```rust
pub struct FaultTolerance {
    health_checker: HealthChecker,
    auto_recovery: AutoRecovery,
}

impl FaultTolerance {
    pub async fn monitor_and_recover(&self) -> Result<(), Error> {
        loop {
            let health_status = self.health_checker.check_all_services().await?;
            
            for (service, status) in health_status {
                if status == HealthStatus::Unhealthy {
                    println!("检测到服务故障: {}", service);
                    self.auto_recovery.recover_service(&service).await?;
                }
            }
            
            sleep(Duration::from_secs(30)).await;
        }
    }
}

pub struct AutoRecovery {
    kubectl_client: KubectlClient,
}

impl AutoRecovery {
    pub async fn recover_service(&self, service_name: &str) -> Result<(), Error> {
        println!("开始恢复服务: {}", service_name);
        
        // 重启Pod
        self.restart_pod(service_name).await?;
        
        // 等待服务恢复
        self.wait_for_service_recovery(service_name).await?;
        
        // 验证服务健康状态
        self.verify_service_health(service_name).await?;
        
        println!("服务恢复完成: {}", service_name);
        Ok(())
    }
    
    async fn restart_pod(&self, service_name: &str) -> Result<(), Error> {
        let output = Command::new("kubectl")
            .args(&["delete", "pod", "-l", &format!("app={}", service_name)])
            .output()?;
        
        if !output.status.success() {
            return Err(Error::PodRestartFailed);
        }
        
        Ok(())
    }
    
    async fn wait_for_service_recovery(&self, service_name: &str) -> Result<(), Error> {
        let mut attempts = 0;
        const max_attempts: u32 = 10;
        
        while attempts < max_attempts {
            let health_status = self.check_service_health(service_name).await?;
            if health_status == HealthStatus::Healthy {
                return Ok(());
            }
            
            sleep(Duration::from_secs(10)).await;
            attempts += 1;
        }
        
        Err(Error::ServiceRecoveryTimeout)
    }
}
```

## 16.7 容量规划

### 16.7.1 资源需求分析

```rust
pub struct CapacityPlanning {
    metrics_analyzer: MetricsAnalyzer,
    resource_predictor: ResourcePredictor,
}

impl CapacityPlanning {
    pub async fn analyze_capacity_needs(&self) -> Result<CapacityRecommendation, Error> {
        // 分析历史使用模式
        let historical_metrics = self.metrics_analyzer.get_historical_metrics().await?;
        
        // 预测未来需求
        let predicted_demand = self.resource_predictor.predict_demand(&historical_metrics).await?;
        
        // 计算资源需求
        let cpu_requirement = self.calculate_cpu_requirement(&predicted_demand).await?;
        let memory_requirement = self.calculate_memory_requirement(&predicted_demand).await?;
        let storage_requirement = self.calculate_storage_requirement(&predicted_demand).await?;
        
        Ok(CapacityRecommendation {
            cpu_cores: cpu_requirement,
            memory_gb: memory_requirement,
            storage_gb: storage_requirement,
            replicas: self.calculate_optimal_replicas(&predicted_demand).await?,
            scaling_policy: self.generate_scaling_policy(&predicted_demand).await?,
        })
    }
    
    async fn calculate_cpu_requirement(&self, demand: &DemandPrediction) -> Result<f64, Error> {
        let peak_cpu_usage = demand.peak_cpu_usage;
        let safety_margin = 1.2; // 20%安全边际
        
        Ok(peak_cpu_usage * safety_margin)
    }
    
    async fn calculate_memory_requirement(&self, demand: &DemandPrediction) -> Result<f64, Error> {
        let peak_memory_usage = demand.peak_memory_usage;
        let safety_margin = 1.15; // 15%安全边际
        
        Ok(peak_memory_usage * safety_margin)
    }
    
    async fn calculate_storage_requirement(&self, demand: &DemandPrediction) -> Result<f64, Error> {
        let data_growth_rate = demand.data_growth_rate_gb_per_month;
        let retention_period_months = 12; // 保留12个月
        
        Ok(data_growth_rate * retention_period_months as f64)
    }
}
```

## 16.8 部署架构图

### 16.8.1 CI/CD流水线

```mermaid
graph LR
    subgraph "开发阶段"
        DEV1[代码提交]
        DEV2[代码审查]
        DEV3[单元测试]
    end
    
    subgraph "构建阶段"
        BUILD1[Docker构建]
        BUILD2[镜像推送]
        BUILD3[安全扫描]
    end
    
    subgraph "测试阶段"
        TEST1[集成测试]
        TEST2[性能测试]
        TEST3[安全测试]
    end
    
    subgraph "部署阶段"
        DEP1[预生产部署]
        DEP2[生产部署]
        DEP3[监控验证]
    end
    
    DEV1 --> DEV2
    DEV2 --> DEV3
    DEV3 --> BUILD1
    BUILD1 --> BUILD2
    BUILD2 --> BUILD3
    BUILD3 --> TEST1
    TEST1 --> TEST2
    TEST2 --> TEST3
    TEST3 --> DEP1
    DEP1 --> DEP2
    DEP2 --> DEP3
```

### 16.8.2 生产环境架构

```mermaid
graph TB
    subgraph "负载均衡层"
        LB1[负载均衡器]
        LB2[API网关]
    end
    
    subgraph "应用层"
        APP1[IoT Edge Node 1]
        APP2[IoT Edge Node 2]
        APP3[IoT Edge Node 3]
    end
    
    subgraph "数据层"
        DB1[主数据库]
        DB2[从数据库]
        DB3[缓存]
    end
    
    subgraph "监控层"
        MON1[Prometheus]
        MON2[Grafana]
        MON3[AlertManager]
    end
    
    LB1 --> APP1
    LB1 --> APP2
    LB1 --> APP3
    LB2 --> APP1
    LB2 --> APP2
    LB2 --> APP3
    
    APP1 --> DB1
    APP2 --> DB1
    APP3 --> DB1
    DB1 --> DB2
    APP1 --> DB3
    APP2 --> DB3
    APP3 --> DB3
    
    APP1 --> MON1
    APP2 --> MON1
    APP3 --> MON1
    MON1 --> MON2
    MON1 --> MON3
```

## 16.9 跨主题引用

- 基础理论与行业标准详见[1. IoT基础理论与行业标准](01_Foundation.md)
- 性能与可靠性详见[5. IoT性能与可靠性](05_Performance_Reliability.md)
- 监控可观测性详见[15. IoT监控可观测性与运维](15_Monitoring_Observability.md)
- 实现指南与最佳实践详见[11. IoT实现指南与最佳实践](11_Implementation_Guide.md)

## 16.10 参考与扩展阅读

- [Kubernetes官方文档](https://kubernetes.io/docs/)
- [Docker最佳实践](https://docs.docker.com/develop/dev-best-practices/)
- [Helm包管理器](https://helm.sh/)
- [GitOps实践](https://www.gitops.tech/)
