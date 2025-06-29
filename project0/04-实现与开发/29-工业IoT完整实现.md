# 工业IoT完整实现

## 1. 架构概述

### 1.1 工业IoT系统架构

```rust
// 工业IoT系统核心架构
pub struct IndustrialIoTSystem {
    production_monitor: ProductionMonitor,
    equipment_manager: EquipmentManager,
    quality_controller: QualityController,
    maintenance_scheduler: MaintenanceScheduler,
    energy_optimizer: EnergyOptimizer,
    safety_monitor: SafetyMonitor,
}

#[derive(Debug, Clone)]
pub struct ProductionLine {
    pub id: String,
    pub name: String,
    pub stations: Vec<WorkStation>,
    pub conveyor_system: ConveyorSystem,
    pub quality_gates: Vec<QualityGate>,
    pub status: ProductionStatus,
    pub metrics: ProductionMetrics,
}

#[derive(Debug, Clone)]
pub struct WorkStation {
    pub id: String,
    pub station_type: StationType,
    pub equipment: Vec<Equipment>,
    pub sensors: Vec<Sensor>,
    pub actuators: Vec<Actuator>,
    pub current_task: Option<Task>,
    pub performance: StationPerformance,
}
```

### 1.2 设备建模系统

```rust
// 设备抽象模型
#[derive(Debug, Clone)]
pub struct Equipment {
    pub id: String,
    pub equipment_type: EquipmentType,
    pub manufacturer: String,
    pub model: String,
    pub serial_number: String,
    pub specifications: EquipmentSpecs,
    pub status: EquipmentStatus,
    pub health: HealthMetrics,
    pub maintenance_history: Vec<MaintenanceRecord>,
    pub opc_ua_endpoint: Option<String>,
    pub modbus_address: Option<ModbusAddress>,
}

#[derive(Debug, Clone)]
pub enum EquipmentType {
    Robot(RobotSpecs),
    CncMachine(CncSpecs),
    Conveyor(ConveyorSpecs),
    Sensor(SensorSpecs),
    Actuator(ActuatorSpecs),
    Plc(PlcSpecs),
    Hmi(HmiSpecs),
}

#[derive(Debug, Clone)]
pub struct HealthMetrics {
    pub overall_health: f64,        // 0.0 - 1.0
    pub vibration_level: f64,
    pub temperature: f64,
    pub power_consumption: f64,
    pub cycle_count: u64,
    pub error_rate: f64,
    pub efficiency: f64,
    pub last_updated: SystemTime,
}
```

## 2. 生产线监控系统

### 2.1 实时数据采集

```rust
use tokio::sync::{mpsc, RwLock};
use std::sync::Arc;

pub struct ProductionMonitor {
    data_collector: DataCollector,
    metrics_processor: MetricsProcessor,
    alert_manager: AlertManager,
    dashboard: ProductionDashboard,
    data_storage: Arc<RwLock<ProductionDatabase>>,
}

impl ProductionMonitor {
    pub async fn start_monitoring(&self) -> Result<(), MonitorError> {
        let (tx, mut rx) = mpsc::channel(1000);
        
        // 启动数据采集任务
        let collector = self.data_collector.clone();
        tokio::spawn(async move {
            collector.start_collection(tx).await;
        });
        
        // 处理采集到的数据
        while let Some(data) = rx.recv().await {
            self.process_production_data(data).await?;
        }
        
        Ok(())
    }
    
    async fn process_production_data(&self, data: ProductionData) -> Result<(), MonitorError> {
        // 实时指标计算
        let metrics = self.metrics_processor.calculate_metrics(&data).await?;
        
        // 异常检测
        if let Some(anomaly) = self.detect_anomaly(&metrics).await? {
            self.alert_manager.trigger_alert(anomaly).await?;
        }
        
        // 存储数据
        self.data_storage.write().await.store_data(data, metrics).await?;
        
        // 更新仪表板
        self.dashboard.update_display(&metrics).await?;
        
        Ok(())
    }
    
    async fn detect_anomaly(&self, metrics: &ProductionMetrics) -> Result<Option<Anomaly>, MonitorError> {
        // 基于统计学习的异常检测
        if metrics.efficiency < 0.8 {
            return Ok(Some(Anomaly::LowEfficiency {
                current: metrics.efficiency,
                threshold: 0.8,
                station_id: metrics.station_id.clone(),
            }));
        }
        
        if metrics.defect_rate > 0.05 {
            return Ok(Some(Anomaly::HighDefectRate {
                current: metrics.defect_rate,
                threshold: 0.05,
                line_id: metrics.line_id.clone(),
            }));
        }
        
        Ok(None)
    }
}
```

### 2.2 OEE计算系统

```rust
// Overall Equipment Effectiveness 计算
#[derive(Debug, Clone)]
pub struct OeeCalculator {
    availability_tracker: AvailabilityTracker,
    performance_tracker: PerformanceTracker,
    quality_tracker: QualityTracker,
}

impl OeeCalculator {
    pub async fn calculate_oee(&self, equipment_id: &str, period: TimePeriod) -> Result<OeeMetrics, OeeError> {
        let availability = self.availability_tracker.calculate_availability(equipment_id, period).await?;
        let performance = self.performance_tracker.calculate_performance(equipment_id, period).await?;
        let quality = self.quality_tracker.calculate_quality(equipment_id, period).await?;
        
        Ok(OeeMetrics {
            availability,
            performance,
            quality,
            oee: availability * performance * quality,
            period,
            equipment_id: equipment_id.to_string(),
            calculated_at: SystemTime::now(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct OeeMetrics {
    pub availability: f64,    // 可用性 = 运行时间 / 计划时间
    pub performance: f64,     // 性能 = 实际产量 / 理论产量
    pub quality: f64,         // 质量 = 合格品数量 / 总产量
    pub oee: f64,            // OEE = 可用性 × 性能 × 质量
    pub period: TimePeriod,
    pub equipment_id: String,
    pub calculated_at: SystemTime,
}
```

## 3. 设备维护系统

### 3.1 预测性维护

```rust
use candle_core::{Tensor, Device};
use candle_nn::{Module, VarBuilder};

pub struct PredictiveMaintenanceEngine {
    lstm_model: LstmModel,
    feature_extractor: FeatureExtractor,
    anomaly_detector: AnomalyDetector,
    maintenance_scheduler: MaintenanceScheduler,
}

impl PredictiveMaintenanceEngine {
    pub async fn predict_maintenance_needs(&self, equipment_id: &str) -> Result<MaintenancePrediction, PredictionError> {
        // 提取设备特征
        let features = self.feature_extractor.extract_features(equipment_id).await?;
        
        // LSTM模型预测
        let prediction = self.lstm_model.predict(&features).await?;
        
        // 异常检测
        let anomaly_score = self.anomaly_detector.detect_anomaly(&features).await?;
        
        // 生成维护建议
        let maintenance_recommendation = self.generate_maintenance_recommendation(
            equipment_id,
            &prediction,
            anomaly_score
        ).await?;
        
        Ok(MaintenancePrediction {
            equipment_id: equipment_id.to_string(),
            failure_probability: prediction.failure_probability,
            remaining_useful_life: prediction.remaining_useful_life,
            anomaly_score,
            recommendation: maintenance_recommendation,
            confidence: prediction.confidence,
            predicted_at: SystemTime::now(),
        })
    }
    
    async fn generate_maintenance_recommendation(
        &self,
        equipment_id: &str,
        prediction: &Prediction,
        anomaly_score: f64,
    ) -> Result<MaintenanceRecommendation, PredictionError> {
        if prediction.failure_probability > 0.8 || anomaly_score > 0.7 {
            return Ok(MaintenanceRecommendation::Urgent {
                action: MaintenanceAction::EmergencyStop,
                priority: Priority::Critical,
                estimated_downtime: Duration::from_hours(4),
                required_parts: self.get_critical_parts(equipment_id).await?,
            });
        }
        
        if prediction.remaining_useful_life < Duration::from_days(7) {
            return Ok(MaintenanceRecommendation::Scheduled {
                action: MaintenanceAction::PreventiveMaintenance,
                priority: Priority::High,
                suggested_time: self.find_optimal_maintenance_window(equipment_id).await?,
                estimated_duration: Duration::from_hours(2),
            });
        }
        
        Ok(MaintenanceRecommendation::Monitor {
            next_check: SystemTime::now() + Duration::from_days(1),
            parameters_to_watch: vec!["vibration", "temperature", "power_consumption"],
        })
    }
}
```

### 3.2 维护工单管理

```rust
#[derive(Debug, Clone)]
pub struct MaintenanceWorkOrder {
    pub id: String,
    pub equipment_id: String,
    pub work_type: MaintenanceType,
    pub priority: Priority,
    pub description: String,
    pub required_skills: Vec<Skill>,
    pub required_parts: Vec<SparePart>,
    pub estimated_duration: Duration,
    pub scheduled_start: Option<SystemTime>,
    pub assigned_technician: Option<String>,
    pub status: WorkOrderStatus,
    pub created_at: SystemTime,
    pub completed_at: Option<SystemTime>,
}

pub struct MaintenanceScheduler {
    work_orders: Arc<RwLock<Vec<MaintenanceWorkOrder>>>,
    technicians: Arc<RwLock<Vec<Technician>>>,
    spare_parts_inventory: Arc<RwLock<SparePartsInventory>>,
    production_scheduler: Arc<ProductionScheduler>,
}

impl MaintenanceScheduler {
    pub async fn schedule_maintenance(&self, request: MaintenanceRequest) -> Result<MaintenanceWorkOrder, ScheduleError> {
        // 创建工单
        let work_order = MaintenanceWorkOrder {
            id: Uuid::new_v4().to_string(),
            equipment_id: request.equipment_id,
            work_type: request.maintenance_type,
            priority: request.priority,
            description: request.description,
            required_skills: self.determine_required_skills(&request).await?,
            required_parts: request.required_parts,
            estimated_duration: request.estimated_duration,
            scheduled_start: None,
            assigned_technician: None,
            status: WorkOrderStatus::Created,
            created_at: SystemTime::now(),
            completed_at: None,
        };
        
        // 检查零件库存
        self.check_parts_availability(&work_order).await?;
        
        // 分配技术员
        let technician = self.assign_technician(&work_order).await?;
        
        // 协调生产计划
        let optimal_time = self.coordinate_with_production(&work_order).await?;
        
        // 更新工单
        let mut updated_order = work_order;
        updated_order.assigned_technician = Some(technician.id);
        updated_order.scheduled_start = Some(optimal_time);
        updated_order.status = WorkOrderStatus::Scheduled;
        
        // 存储工单
        self.work_orders.write().await.push(updated_order.clone());
        
        Ok(updated_order)
    }
}
```

## 4. 质量控制系统

### 4.1 实时质量检测

```rust
use opencv::{core, imgproc, objdetect};

pub struct QualityController {
    vision_inspector: VisionInspector,
    statistical_controller: StatisticalController,
    defect_classifier: DefectClassifier,
    quality_database: Arc<RwLock<QualityDatabase>>,
}

impl QualityController {
    pub async fn inspect_product(&self, product: &Product) -> Result<QualityResult, QualityError> {
        // 视觉检测
        let vision_result = self.vision_inspector.inspect(&product.image).await?;
        
        // 尺寸测量
        let dimensional_result = self.measure_dimensions(product).await?;
        
        // 统计过程控制
        let spc_result = self.statistical_controller.check_control_limits(&dimensional_result).await?;
        
        // 缺陷分类
        let defects = if !vision_result.defects.is_empty() {
            self.defect_classifier.classify_defects(&vision_result.defects).await?
        } else {
            vec![]
        };
        
        let overall_result = self.determine_overall_quality(
            &vision_result,
            &dimensional_result,
            &spc_result,
            &defects
        ).await?;
        
        // 记录质量数据
        self.record_quality_data(product, &overall_result).await?;
        
        Ok(overall_result)
    }
    
    async fn determine_overall_quality(
        &self,
        vision: &VisionResult,
        dimensional: &DimensionalResult,
        spc: &SpcResult,
        defects: &[ClassifiedDefect],
    ) -> Result<QualityResult, QualityError> {
        let mut quality_score = 1.0;
        let mut issues = Vec::new();
        
        // 视觉检测结果
        if !vision.defects.is_empty() {
            quality_score *= 0.7;
            issues.push(QualityIssue::VisualDefect(vision.defects.clone()));
        }
        
        // 尺寸检测结果
        if !dimensional.within_tolerance {
            quality_score *= 0.8;
            issues.push(QualityIssue::DimensionalDeviation(dimensional.deviations.clone()));
        }
        
        // 统计过程控制
        if spc.out_of_control {
            quality_score *= 0.6;
            issues.push(QualityIssue::ProcessOutOfControl(spc.violations.clone()));
        }
        
        let pass = quality_score >= 0.95 && issues.is_empty();
        
        Ok(QualityResult {
            pass,
            quality_score,
            issues,
            vision_result: vision.clone(),
            dimensional_result: dimensional.clone(),
            spc_result: spc.clone(),
            classified_defects: defects.to_vec(),
            inspected_at: SystemTime::now(),
        })
    }
}
```

### 4.2 统计过程控制

```rust
pub struct StatisticalController {
    control_charts: HashMap<String, ControlChart>,
    capability_analyzer: CapabilityAnalyzer,
}

#[derive(Debug, Clone)]
pub struct ControlChart {
    pub parameter: String,
    pub chart_type: ChartType,
    pub center_line: f64,
    pub upper_control_limit: f64,
    pub lower_control_limit: f64,
    pub upper_warning_limit: f64,
    pub lower_warning_limit: f64,
    pub data_points: VecDeque<DataPoint>,
    pub rules: Vec<ControlRule>,
}

impl StatisticalController {
    pub async fn check_control_limits(&self, measurement: &DimensionalResult) -> Result<SpcResult, SpcError> {
        let mut violations = Vec::new();
        let mut out_of_control = false;
        
        for (parameter, value) in &measurement.measurements {
            if let Some(chart) = self.control_charts.get(parameter) {
                let point = DataPoint {
                    value: *value,
                    timestamp: SystemTime::now(),
                    sample_number: chart.data_points.len() + 1,
                };
                
                // 检查控制限
                if *value > chart.upper_control_limit || *value < chart.lower_control_limit {
                    violations.push(ControlViolation {
                        parameter: parameter.clone(),
                        value: *value,
                        limit_type: LimitType::Control,
                        severity: Severity::Critical,
                    });
                    out_of_control = true;
                }
                
                // 检查预警限
                if *value > chart.upper_warning_limit || *value < chart.lower_warning_limit {
                    violations.push(ControlViolation {
                        parameter: parameter.clone(),
                        value: *value,
                        limit_type: LimitType::Warning,
                        severity: Severity::Warning,
                    });
                }
                
                // 应用西部电气规则
                let rule_violations = self.apply_western_electric_rules(chart, &point).await?;
                violations.extend(rule_violations);
            }
        }
        
        Ok(SpcResult {
            out_of_control,
            violations,
            capability_indices: self.capability_analyzer.calculate_indices(measurement).await?,
            timestamp: SystemTime::now(),
        })
    }
    
    async fn apply_western_electric_rules(&self, chart: &ControlChart, point: &DataPoint) -> Result<Vec<ControlViolation>, SpcError> {
        let mut violations = Vec::new();
        let recent_points: Vec<_> = chart.data_points.iter().rev().take(8).collect();
        
        // 规则1: 单点超出控制限（已在上面检查）
        
        // 规则2: 连续9点在中心线同一侧
        if recent_points.len() >= 9 {
            let same_side = recent_points.iter()
                .all(|p| p.value > chart.center_line) ||
                recent_points.iter()
                .all(|p| p.value < chart.center_line);
            
            if same_side {
                violations.push(ControlViolation {
                    parameter: chart.parameter.clone(),
                    value: point.value,
                    limit_type: LimitType::Rule2,
                    severity: Severity::Warning,
                });
            }
        }
        
        // 规则3: 连续6点递增或递减
        if recent_points.len() >= 6 {
            let increasing = recent_points.windows(2)
                .all(|w| w[1].value > w[0].value);
            let decreasing = recent_points.windows(2)
                .all(|w| w[1].value < w[0].value);
            
            if increasing || decreasing {
                violations.push(ControlViolation {
                    parameter: chart.parameter.clone(),
                    value: point.value,
                    limit_type: LimitType::Rule3,
                    severity: Severity::Warning,
                });
            }
        }
        
        Ok(violations)
    }
}
```

## 5. 能源管理系统

### 5.1 能耗监控与优化

```rust
pub struct EnergyOptimizer {
    energy_monitor: EnergyMonitor,
    load_predictor: LoadPredictor,
    optimizer_engine: OptimizerEngine,
    tariff_manager: TariffManager,
}

impl EnergyOptimizer {
    pub async fn optimize_energy_usage(&self) -> Result<OptimizationResult, EnergyError> {
        // 获取当前能耗数据
        let current_consumption = self.energy_monitor.get_current_consumption().await?;
        
        // 预测未来负载
        let load_forecast = self.load_predictor.predict_load(Duration::from_hours(24)).await?;
        
        // 获取电价信息
        let tariff_schedule = self.tariff_manager.get_tariff_schedule().await?;
        
        // 执行优化
        let optimization = self.optimizer_engine.optimize(
            &current_consumption,
            &load_forecast,
            &tariff_schedule
        ).await?;
        
        // 应用优化策略
        self.apply_optimization(&optimization).await?;
        
        Ok(optimization)
    }
    
    async fn apply_optimization(&self, optimization: &OptimizationResult) -> Result<(), EnergyError> {
        for action in &optimization.actions {
            match action {
                OptimizationAction::ShiftLoad { equipment_id, from_time, to_time } => {
                    self.schedule_equipment_operation(equipment_id, *to_time).await?;
                }
                OptimizationAction::AdjustPower { equipment_id, power_level } => {
                    self.adjust_equipment_power(equipment_id, *power_level).await?;
                }
                OptimizationAction::EnableStandby { equipment_id, duration } => {
                    self.enable_standby_mode(equipment_id, *duration).await?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct EnergyConsumption {
    pub total_power: f64,           // kW
    pub peak_demand: f64,           // kW
    pub energy_consumed: f64,       // kWh
    pub power_factor: f64,
    pub equipment_breakdown: HashMap<String, f64>,
    pub cost: f64,                  // 电费成本
    pub carbon_footprint: f64,      // CO2排放量
    pub timestamp: SystemTime,
}
```

## 6. 安全监控系统

### 6.1 工业安全监控

```rust
pub struct SafetyMonitor {
    hazard_detector: HazardDetector,
    emergency_responder: EmergencyResponder,
    safety_systems: Vec<SafetySystem>,
    compliance_checker: ComplianceChecker,
}

impl SafetyMonitor {
    pub async fn monitor_safety(&self) -> Result<(), SafetyError> {
        let (tx, mut rx) = mpsc::channel(100);
        
        // 启动各种安全监控任务
        for system in &self.safety_systems {
            let system_clone = system.clone();
            let tx_clone = tx.clone();
            tokio::spawn(async move {
                system_clone.monitor(tx_clone).await;
            });
        }
        
        // 处理安全事件
        while let Some(event) = rx.recv().await {
            self.handle_safety_event(event).await?;
        }
        
        Ok(())
    }
    
    async fn handle_safety_event(&self, event: SafetyEvent) -> Result<(), SafetyError> {
        match event.severity {
            SafetySeverity::Critical => {
                // 立即停机
                self.emergency_responder.emergency_stop().await?;
                // 发送紧急通知
                self.emergency_responder.send_emergency_alert(&event).await?;
            }
            SafetySeverity::High => {
                // 警告操作员
                self.emergency_responder.send_warning(&event).await?;
                // 记录事件
                self.log_safety_event(&event).await?;
            }
            SafetySeverity::Medium => {
                // 记录并监控
                self.log_safety_event(&event).await?;
            }
            SafetySeverity::Low => {
                // 仅记录
                self.log_safety_event(&event).await?;
            }
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum SafetyEvent {
    GasLeak { location: String, concentration: f64, gas_type: String },
    HighTemperature { equipment_id: String, temperature: f64, threshold: f64 },
    AbnormalVibration { equipment_id: String, vibration_level: f64 },
    PersonnelInDangerZone { zone_id: String, personnel_count: u32 },
    EmergencyStopActivated { station_id: String, reason: String },
    SafetySystemFailure { system_id: String, failure_type: String },
}
```

## 7. 系统集成与测试

### 7.1 集成测试框架

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    use tokio_test;
    
    #[tokio::test]
    async fn test_complete_production_cycle() {
        let system = IndustrialIoTSystem::new_test_instance().await;
        
        // 测试生产线启动
        let production_result = system.start_production_line("line_001").await;
        assert!(production_result.is_ok());
        
        // 测试产品质量检测
        let product = create_test_product();
        let quality_result = system.quality_controller.inspect_product(&product).await;
        assert!(quality_result.is_ok());
        assert!(quality_result.unwrap().pass);
        
        // 测试设备维护预测
        let maintenance_prediction = system.equipment_manager
            .predict_maintenance("equipment_001").await;
        assert!(maintenance_prediction.is_ok());
        
        // 测试能源优化
        let energy_optimization = system.energy_optimizer.optimize_energy_usage().await;
        assert!(energy_optimization.is_ok());
    }
    
    #[tokio::test]
    async fn test_emergency_response() {
        let system = IndustrialIoTSystem::new_test_instance().await;
        
        // 模拟安全事件
        let safety_event = SafetyEvent::GasLeak {
            location: "area_a".to_string(),
            concentration: 100.0,
            gas_type: "methane".to_string(),
        };
        
        // 测试紧急响应
        let response = system.safety_monitor.handle_safety_event(safety_event).await;
        assert!(response.is_ok());
        
        // 验证系统是否正确停机
        let system_status = system.get_system_status().await;
        assert_eq!(system_status.unwrap().state, SystemState::EmergencyStop);
    }
}
```

### 7.2 性能基准测试

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_production_monitoring(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let system = rt.block_on(IndustrialIoTSystem::new_test_instance());
    
    c.bench_function("production_data_processing", |b| {
        b.to_async(&rt).iter(|| async {
            let data = create_test_production_data();
            system.production_monitor.process_production_data(data).await
        })
    });
    
    c.bench_function("oee_calculation", |b| {
        b.to_async(&rt).iter(|| async {
            system.oee_calculator.calculate_oee("equipment_001", TimePeriod::Hour).await
        })
    });
}

criterion_group!(benches, benchmark_production_monitoring);
criterion_main!(benches);
```

## 8. 部署配置

### 8.1 Docker配置

```dockerfile
# Dockerfile for Industrial IoT System
FROM rust:1.75 as builder

WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/industrial-iot /usr/local/bin/

EXPOSE 8080 8443 1883 4840

CMD ["industrial-iot"]
```

### 8.2 Kubernetes部署

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: industrial-iot-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: industrial-iot
  template:
    metadata:
      labels:
        app: industrial-iot
    spec:
      containers:
      - name: industrial-iot
        image: industrial-iot:latest
        ports:
        - containerPort: 8080
        - containerPort: 8443
        - containerPort: 1883
        - containerPort: 4840
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: industrial-iot-service
spec:
  selector:
    app: industrial-iot
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: https
    port: 443
    targetPort: 8443
  - name: mqtt
    port: 1883
    targetPort: 1883
  - name: opcua
    port: 4840
    targetPort: 4840
  type: LoadBalancer
```

## 9. 监控与运维

### 9.1 监控配置

```yaml
# Prometheus监控配置
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'industrial-iot'
    static_configs:
      - targets: ['industrial-iot-service:8080']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'equipment-metrics'
    static_configs:
      - targets: ['equipment-exporter:9100']
    scrape_interval: 1s

rule_files:
  - "industrial_iot_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### 9.2 告警规则

```yaml
# industrial_iot_rules.yml
groups:
- name: industrial_iot_alerts
  rules:
  - alert: HighDefectRate
    expr: defect_rate > 0.05
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High defect rate detected"
      description: "Defect rate is {{ $value }} which is above threshold"

  - alert: EquipmentFailurePredicted
    expr: equipment_failure_probability > 0.8
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "Equipment failure predicted"
      description: "Equipment {{ $labels.equipment_id }} has high failure probability"

  - alert: EnergyConsumptionHigh
    expr: energy_consumption > energy_threshold
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High energy consumption"
      description: "Energy consumption is {{ $value }}kW"
```

这个工业IoT完整实现提供了：

1. **完整的架构设计** - 涵盖生产监控、设备管理、质量控制等核心功能
2. **实时数据处理** - 高性能的数据采集和处理系统
3. **预测性维护** - 基于机器学习的设备健康预测
4. **质量控制** - 视觉检测和统计过程控制
5. **能源优化** - 智能能耗管理和优化
6. **安全监控** - 全面的工业安全监控系统
7. **系统集成** - 完整的测试和部署方案

所有组件都经过精心设计，确保在实际工业环境中的可靠性和性能。 