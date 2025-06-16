# IoT 业务建模 (IoT Business Modeling)

## 目录

1. [业务领域概念建模](#1-业务领域概念建模)
2. [数据建模与存储设计](#2-数据建模与存储设计)
3. [流程建模与工作流](#3-流程建模与工作流)
4. [规则引擎与决策系统](#4-规则引擎与决策系统)
5. [事件驱动架构](#5-事件驱动架构)
6. [业务价值分析](#6-业务价值分析)

## 1. 业务领域概念建模

### 1.1 核心业务概念

**定义 1.1 (IoT 业务域)**
IoT 业务域是一个四元组：

$$\mathcal{B}_{IoT} = (\mathcal{D}, \mathcal{S}, \mathcal{R}, \mathcal{E})$$

其中：
- $\mathcal{D}$ 是设备集合
- $\mathcal{S}$ 是传感器数据集合
- $\mathcal{R}$ 是规则集合
- $\mathcal{E}$ 是事件集合

**定理 1.1 (业务域完整性)**
IoT 业务域对于描述完整的 IoT 业务是完备的。

**证明：** 通过业务需求分析：

1. **设备管理**：$\mathcal{D}$ 覆盖所有设备相关业务
2. **数据处理**：$\mathcal{S}$ 覆盖所有数据相关业务
3. **业务规则**：$\mathcal{R}$ 覆盖所有规则相关业务
4. **事件处理**：$\mathcal{E}$ 覆盖所有事件相关业务

### 1.2 设备聚合根

**定义 1.2 (设备聚合根)**
设备聚合根是 IoT 系统的核心业务实体：

$$\mathcal{D}_{agg} = (id, name, type, config, status, capabilities)$$

**设备模型实现 1.1**

```rust
#[derive(Debug, Clone)]
pub struct Device {
    pub id: DeviceId,
    pub name: String,
    pub device_type: DeviceType,
    pub model: String,
    pub manufacturer: String,
    pub firmware_version: String,
    pub location: Location,
    pub status: DeviceStatus,
    pub capabilities: Vec<Capability>,
    pub configuration: DeviceConfiguration,
    pub last_seen: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct DeviceConfiguration {
    pub sampling_rate: Duration,
    pub threshold_values: HashMap<String, f64>,
    pub communication_interval: Duration,
    pub power_mode: PowerMode,
    pub security_settings: SecuritySettings,
}

impl Device {
    pub fn is_online(&self) -> bool {
        self.status == DeviceStatus::Online &&
        self.last_seen > Utc::now() - Duration::from_secs(300)
    }
    
    pub fn can_communicate(&self) -> bool {
        self.is_online() && self.capabilities.contains(&Capability::Communication)
    }
    
    pub fn check_threshold(&self, sensor_type: &str, value: f64) -> Option<ThresholdAlert> {
        if let Some(threshold) = self.configuration.threshold_values.get(sensor_type) {
            if value > *threshold {
                Some(ThresholdAlert {
                    device_id: self.id.clone(),
                    sensor_type: sensor_type.to_string(),
                    value,
                    threshold: *threshold,
                    timestamp: Utc::now(),
                })
            } else {
                None
            }
        } else {
            None
        }
    }
}
```

### 1.3 传感器数据聚合根

**定义 1.3 (传感器数据聚合根)**
传感器数据聚合根是数据业务的核心实体：

$$\mathcal{S}_{agg} = (id, device_id, type, value, quality, metadata)$$

**传感器数据模型 1.1**

```rust
#[derive(Debug, Clone)]
pub struct SensorData {
    pub id: SensorDataId,
    pub device_id: DeviceId,
    pub sensor_type: SensorType,
    pub value: f64,
    pub unit: String,
    pub timestamp: DateTime<Utc>,
    pub quality: DataQuality,
    pub metadata: SensorMetadata,
}

#[derive(Debug, Clone)]
pub struct SensorMetadata {
    pub accuracy: f64,
    pub precision: f64,
    pub calibration_date: Option<DateTime<Utc>>,
    pub environmental_conditions: HashMap<String, f64>,
}

impl SensorData {
    pub fn is_valid(&self) -> bool {
        self.quality == DataQuality::Good && 
        self.value.is_finite() &&
        !self.value.is_nan()
    }
    
    pub fn is_outlier(&self, historical_data: &[SensorData]) -> bool {
        if historical_data.len() < 10 {
            return false;
        }
        
        let values: Vec<f64> = historical_data.iter()
            .map(|d| d.value)
            .collect();
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();
        
        (self.value - mean).abs() > 3.0 * std_dev
    }
}
```

## 2. 数据建模与存储设计

### 2.1 数据库设计原则

**定义 2.1 (IoT 数据模型)**
IoT 数据模型遵循以下原则：

$$\mathcal{M}_{data} = \{\text{时间序列}, \text{高可用}, \text{可扩展}, \text{一致性}\}$$

**定理 2.1 (数据模型最优性)**
分层存储模型对于 IoT 数据是最优的。

**证明：** 通过存储需求分析：

1. **热数据**：存储在内存或 SSD 中，支持快速查询
2. **温数据**：存储在 SSD 中，支持中等频率查询
3. **冷数据**：存储在对象存储中，支持归档查询

### 2.2 时间序列数据设计

**定义 2.2 (时间序列数据)**
时间序列数据是 IoT 系统的核心数据模型：

$$\mathcal{TS} = \{(t_i, v_i) | i \in \mathbb{N}, t_i < t_{i+1}\}$$

**时间序列存储设计 2.1**

```sql
-- 传感器数据表（原始数据）
CREATE TABLE sensor_data (
    id UUID PRIMARY KEY,
    device_id UUID NOT NULL,
    sensor_type VARCHAR(50) NOT NULL,
    value DECIMAL(15, 6) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    quality VARCHAR(20) NOT NULL,
    accuracy DECIMAL(8, 4),
    precision DECIMAL(8, 4),
    calibration_date TIMESTAMP WITH TIME ZONE,
    environmental_conditions JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    FOREIGN KEY (device_id) REFERENCES devices(id)
);

-- 传感器数据聚合表（按小时）
CREATE TABLE sensor_data_hourly (
    id UUID PRIMARY KEY,
    device_id UUID NOT NULL,
    sensor_type VARCHAR(50) NOT NULL,
    hour_start TIMESTAMP WITH TIME ZONE NOT NULL,
    min_value DECIMAL(15, 6) NOT NULL,
    max_value DECIMAL(15, 6) NOT NULL,
    avg_value DECIMAL(15, 6) NOT NULL,
    count INTEGER NOT NULL,
    FOREIGN KEY (device_id) REFERENCES devices(id),
    UNIQUE(device_id, sensor_type, hour_start)
);

-- 传感器数据聚合表（按天）
CREATE TABLE sensor_data_daily (
    id UUID PRIMARY KEY,
    device_id UUID NOT NULL,
    sensor_type VARCHAR(50) NOT NULL,
    day_start DATE NOT NULL,
    min_value DECIMAL(15, 6) NOT NULL,
    max_value DECIMAL(15, 6) NOT NULL,
    avg_value DECIMAL(15, 6) NOT NULL,
    count INTEGER NOT NULL,
    FOREIGN KEY (device_id) REFERENCES devices(id),
    UNIQUE(device_id, sensor_type, day_start)
);
```

### 2.3 数据访问模式

**定义 2.3 (数据访问模式)**
IoT 系统的数据访问模式：

$$\mathcal{A}_{pattern} = \{\text{实时查询}, \text{历史分析}, \text{聚合统计}, \text{趋势预测}\}$$

**数据访问实现 2.1**

```rust
#[derive(Debug, Clone)]
pub struct DataAccessLayer {
    pub real_time_db: RealTimeDatabase,
    pub historical_db: HistoricalDatabase,
    pub analytics_db: AnalyticsDatabase,
}

impl DataAccessLayer {
    pub async fn get_latest_data(&self, device_id: &DeviceId, sensor_type: &str) -> Result<SensorData, DataError> {
        self.real_time_db.get_latest(device_id, sensor_type).await
    }
    
    pub async fn get_historical_data(
        &self,
        device_id: &DeviceId,
        sensor_type: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<SensorData>, DataError> {
        self.historical_db.query_range(device_id, sensor_type, start_time, end_time).await
    }
    
    pub async fn get_aggregated_data(
        &self,
        device_id: &DeviceId,
        sensor_type: &str,
        aggregation: AggregationType,
        time_range: TimeRange,
    ) -> Result<AggregatedData, DataError> {
        self.analytics_db.get_aggregated(device_id, sensor_type, aggregation, time_range).await
    }
    
    pub async fn predict_trend(
        &self,
        device_id: &DeviceId,
        sensor_type: &str,
        prediction_horizon: Duration,
    ) -> Result<PredictionResult, DataError> {
        self.analytics_db.predict_trend(device_id, sensor_type, prediction_horizon).await
    }
}
```

## 3. 流程建模与工作流

### 3.1 业务流程定义

**定义 3.1 (IoT 业务流程)**
IoT 业务流程是一个有向图：

$$\mathcal{W} = (V, E, \lambda, \mu)$$

其中：
- $V$ 是活动节点集合
- $E$ 是边集合
- $\lambda: V \rightarrow \mathcal{A}$ 是活动映射
- $\mu: E \rightarrow \mathcal{C}$ 是条件映射

**定理 3.1 (流程可达性)**
所有业务流程都是可达的。

**证明：** 通过图论分析：

1. **连通性**：每个节点都有入边和出边
2. **可达性**：存在从起始节点到任意节点的路径
3. **终止性**：每个流程都有明确的终止条件

### 3.2 设备注册流程

**定义 3.2 (设备注册流程)**
设备注册流程包含以下步骤：

$$\mathcal{R}_{device} = \{\text{验证}, \text{配置}, \text{激活}, \text{监控}\}$$

**设备注册实现 3.1**

```rust
#[derive(Debug, Clone)]
pub struct DeviceRegistrationWorkflow {
    pub device_repository: DeviceRepository,
    pub security_service: SecurityService,
    pub configuration_service: ConfigurationService,
    pub monitoring_service: MonitoringService,
}

impl DeviceRegistrationWorkflow {
    pub async fn register_device(&mut self, device_info: DeviceInfo) -> Result<DeviceId, RegistrationError> {
        // 步骤 1: 验证设备信息
        let verified_info = self.security_service.verify_device(&device_info).await?;
        
        // 步骤 2: 创建设备记录
        let device_id = self.device_repository.create_device(verified_info).await?;
        
        // 步骤 3: 配置设备
        let configuration = self.configuration_service.generate_configuration(&device_id).await?;
        self.device_repository.update_configuration(&device_id, configuration).await?;
        
        // 步骤 4: 激活设备
        self.device_repository.activate_device(&device_id).await?;
        
        // 步骤 5: 启动监控
        self.monitoring_service.start_monitoring(&device_id).await?;
        
        Ok(device_id)
    }
    
    pub async fn handle_registration_event(&mut self, event: RegistrationEvent) -> Result<(), WorkflowError> {
        match event {
            RegistrationEvent::DeviceConnected { device_id } => {
                self.device_repository.update_status(&device_id, DeviceStatus::Online).await?;
                self.monitoring_service.record_connection(&device_id).await?;
            }
            RegistrationEvent::DeviceDisconnected { device_id } => {
                self.device_repository.update_status(&device_id, DeviceStatus::Offline).await?;
                self.monitoring_service.record_disconnection(&device_id).await?;
            }
            RegistrationEvent::ConfigurationUpdated { device_id, config } => {
                self.device_repository.update_configuration(&device_id, config).await?;
            }
        }
        Ok(())
    }
}
```

### 3.3 数据处理流程

**定义 3.3 (数据处理流程)**
数据处理流程包含以下阶段：

$$\mathcal{P}_{data} = \{\text{采集}, \text{验证}, \text{转换}, \text{存储}, \text{分析}\}$$

**数据处理实现 3.1**

```rust
#[derive(Debug, Clone)]
pub struct DataProcessingWorkflow {
    pub data_collector: DataCollector,
    pub data_validator: DataValidator,
    pub data_transformer: DataTransformer,
    pub data_storage: DataStorage,
    pub data_analyzer: DataAnalyzer,
}

impl DataProcessingWorkflow {
    pub async fn process_sensor_data(&mut self, raw_data: RawSensorData) -> Result<ProcessedData, ProcessingError> {
        // 步骤 1: 数据采集
        let collected_data = self.data_collector.collect(raw_data).await?;
        
        // 步骤 2: 数据验证
        let validated_data = self.data_validator.validate(collected_data).await?;
        
        // 步骤 3: 数据转换
        let transformed_data = self.data_transformer.transform(validated_data).await?;
        
        // 步骤 4: 数据存储
        let stored_data = self.data_storage.store(transformed_data).await?;
        
        // 步骤 5: 数据分析
        let analysis_result = self.data_analyzer.analyze(&stored_data).await?;
        
        Ok(ProcessedData {
            original: raw_data,
            processed: stored_data,
            analysis: analysis_result,
        })
    }
    
    pub async fn handle_processing_event(&mut self, event: ProcessingEvent) -> Result<(), WorkflowError> {
        match event {
            ProcessingEvent::DataReceived { data } => {
                self.process_sensor_data(data).await?;
            }
            ProcessingEvent::ValidationFailed { data, error } => {
                self.handle_validation_failure(data, error).await?;
            }
            ProcessingEvent::TransformationFailed { data, error } => {
                self.handle_transformation_failure(data, error).await?;
            }
        }
        Ok(())
    }
}
```

## 4. 规则引擎与决策系统

### 4.1 规则引擎架构

**定义 4.1 (规则引擎)**
规则引擎是一个三元组：

$$\mathcal{R}_{engine} = (\mathcal{R}, \mathcal{C}, \mathcal{E})$$

其中：
- $\mathcal{R}$ 是规则集合
- $\mathcal{C}$ 是条件评估器
- $\mathcal{E}$ 是动作执行器

**定理 4.1 (规则引擎完备性)**
规则引擎对于 IoT 决策是完备的。

**证明：** 通过决策逻辑分析：

1. **条件完备性**：覆盖所有可能的条件组合
2. **动作完备性**：覆盖所有可能的动作类型
3. **执行完备性**：确保所有规则都能正确执行

### 4.2 规则定义与执行

**定义 4.2 (业务规则)**
业务规则是一个五元组：

$$\mathcal{R}_{rule} = (id, conditions, actions, priority, enabled)$$

**规则引擎实现 4.1**

```rust
#[derive(Debug, Clone)]
pub struct Rule {
    pub id: RuleId,
    pub name: String,
    pub description: String,
    pub conditions: Vec<Condition>,
    pub actions: Vec<Action>,
    pub priority: u32,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub enum Condition {
    Threshold {
        device_id: DeviceId,
        sensor_type: String,
        operator: ComparisonOperator,
        value: f64,
    },
    TimeRange {
        start_time: TimeOfDay,
        end_time: TimeOfDay,
        days_of_week: Vec<DayOfWeek>,
    },
    DeviceStatus {
        device_id: DeviceId,
        status: DeviceStatus,
    },
    Composite {
        conditions: Vec<Condition>,
        operator: LogicalOperator,
    },
}

#[derive(Debug, Clone)]
pub enum Action {
    SendAlert {
        alert_type: AlertType,
        recipients: Vec<String>,
        message_template: String,
    },
    ControlDevice {
        device_id: DeviceId,
        command: DeviceCommand,
    },
    StoreData {
        data_type: String,
        destination: String,
    },
    TriggerWorkflow {
        workflow_id: String,
        parameters: HashMap<String, String>,
    },
}

impl Rule {
    pub async fn evaluate(&self, context: &RuleContext) -> Result<bool, RuleError> {
        for condition in &self.conditions {
            if !self.evaluate_condition(condition, context).await? {
                return Ok(false);
            }
        }
        Ok(true)
    }
    
    pub async fn execute(&self, context: &RuleContext) -> Result<Vec<ActionResult>, RuleError> {
        let mut results = Vec::new();
        
        for action in &self.actions {
            let result = self.execute_action(action, context).await?;
            results.push(result);
        }
        
        Ok(results)
    }
    
    async fn evaluate_condition(&self, condition: &Condition, context: &RuleContext) -> Result<bool, RuleError> {
        match condition {
            Condition::Threshold { device_id, sensor_type, operator, value } => {
                if let Some(sensor_data) = context.get_latest_sensor_data(device_id, sensor_type) {
                    match operator {
                        ComparisonOperator::GreaterThan => Ok(sensor_data.value > *value),
                        ComparisonOperator::LessThan => Ok(sensor_data.value < *value),
                        ComparisonOperator::Equals => Ok(sensor_data.value == *value),
                        ComparisonOperator::NotEquals => Ok(sensor_data.value != *value),
                    }
                } else {
                    Ok(false)
                }
            }
            Condition::TimeRange { start_time, end_time, days_of_week } => {
                let now = Utc::now();
                let current_time = now.time();
                let current_day = now.weekday();
                
                let day_matches = days_of_week.contains(&current_day);
                let time_matches = current_time >= *start_time && current_time <= *end_time;
                
                Ok(day_matches && time_matches)
            }
            Condition::DeviceStatus { device_id, status } => {
                if let Some(device) = context.get_device(device_id) {
                    Ok(device.status == *status)
                } else {
                    Ok(false)
                }
            }
            Condition::Composite { conditions, operator } => {
                let results: Vec<bool> = futures::future::join_all(
                    conditions.iter().map(|c| self.evaluate_condition(c, context))
                ).await
                    .into_iter()
                    .collect::<Result<Vec<bool>, RuleError>>()?;
                
                match operator {
                    LogicalOperator::And => Ok(results.iter().all(|&r| r)),
                    LogicalOperator::Or => Ok(results.iter().any(|&r| r)),
                }
            }
        }
    }
}
```

## 5. 事件驱动架构

### 5.1 事件系统设计

**定义 5.1 (事件系统)**
事件系统是一个四元组：

$$\mathcal{E}_{system} = (\mathcal{E}, \mathcal{H}, \mathcal{P}, \mathcal{Q})$$

其中：
- $\mathcal{E}$ 是事件集合
- $\mathcal{H}$ 是事件处理器集合
- $\mathcal{P}$ 是事件发布者集合
- $\mathcal{Q}$ 是事件队列集合

**事件系统实现 5.1**

```rust
#[derive(Debug, Clone)]
pub struct EventSystem {
    pub event_bus: EventBus,
    pub event_handlers: HashMap<EventType, Vec<EventHandler>>,
    pub event_processors: Vec<EventProcessor>,
    pub event_queue: EventQueue,
}

#[derive(Debug, Clone)]
pub enum IoTEvent {
    DeviceRegistered { device_id: DeviceId, timestamp: DateTime<Utc> },
    DeviceConnected { device_id: DeviceId, timestamp: DateTime<Utc> },
    DeviceDisconnected { device_id: DeviceId, timestamp: DateTime<Utc> },
    SensorDataReceived { device_id: DeviceId, data: SensorData },
    ThresholdExceeded { device_id: DeviceId, sensor_type: String, value: f64, threshold: f64 },
    RuleTriggered { rule_id: RuleId, device_id: DeviceId, actions: Vec<Action> },
    AlertGenerated { alert_id: AlertId, device_id: DeviceId, message: String },
}

impl EventSystem {
    pub async fn publish_event(&mut self, event: IoTEvent) -> Result<(), PublishError> {
        // 1. 事件验证
        self.validate_event(&event)?;
        
        // 2. 事件路由
        let handlers = self.event_handlers.get(&event.event_type())
            .ok_or(PublishError::NoHandlers)?;
        
        // 3. 异步处理
        for handler in handlers {
            let event_clone = event.clone();
            tokio::spawn(async move {
                handler.handle(event_clone).await;
            });
        }
        
        // 4. 事件持久化
        self.event_queue.enqueue(event).await?;
        
        Ok(())
    }
    
    pub async fn process_event_stream(&mut self, stream: EventStream) -> Result<(), ProcessingError> {
        let mut stream_processor = StreamProcessor::new();
        
        while let Some(event) = stream.next().await {
            // 1. 事件转换
            let processed_event = stream_processor.process(event).await?;
            
            // 2. 发布处理结果
            self.publish_event(processed_event).await?;
        }
        
        Ok(())
    }
    
    pub async fn handle_device_event(&mut self, event: DeviceEvent) -> Result<(), EventError> {
        match event {
            DeviceEvent::Connected { device_id } => {
                self.publish_event(IoTEvent::DeviceConnected { 
                    device_id, 
                    timestamp: Utc::now() 
                }).await?;
            }
            DeviceEvent::Disconnected { device_id } => {
                self.publish_event(IoTEvent::DeviceDisconnected { 
                    device_id, 
                    timestamp: Utc::now() 
                }).await?;
            }
            DeviceEvent::DataReceived { device_id, data } => {
                self.publish_event(IoTEvent::SensorDataReceived { 
                    device_id, 
                    data 
                }).await?;
            }
        }
        Ok(())
    }
}
```

## 6. 业务价值分析

### 6.1 价值模型

**定义 6.1 (IoT 业务价值)**
IoT 业务价值可以用以下指标衡量：

$$\mathcal{V}_{IoT} = (\text{效率提升}, \text{成本降低}, \text{质量改善}, \text{创新价值})$$

**定理 6.1 (价值最大化)**
通过优化业务流程和规则引擎，可以实现业务价值最大化。

**证明：** 通过价值分析：

1. **效率提升**：自动化流程减少人工干预
2. **成本降低**：预测性维护减少故障成本
3. **质量改善**：实时监控提高产品质量
4. **创新价值**：数据驱动决策创造新价值

### 6.2 业务指标

**定义 6.2 (业务指标)**
IoT 业务的关键指标：

$$\mathcal{KPI}_{IoT} = \{\text{设备可用性}, \text{数据质量}, \text{响应时间}, \text{规则执行率}\}$$

**业务指标实现 6.1**

```rust
#[derive(Debug, Clone)]
pub struct BusinessMetrics {
    pub device_availability: f64,
    pub data_quality_score: f64,
    pub average_response_time: Duration,
    pub rule_execution_rate: f64,
}

#[derive(Debug, Clone)]
pub struct MetricsCollector {
    pub device_repository: DeviceRepository,
    pub data_quality_service: DataQualityService,
    pub performance_monitor: PerformanceMonitor,
    pub rule_engine: RuleEngine,
}

impl MetricsCollector {
    pub async fn calculate_device_availability(&self) -> Result<f64, MetricsError> {
        let total_devices = self.device_repository.count_devices().await?;
        let online_devices = self.device_repository.count_online_devices().await?;
        
        if total_devices == 0 {
            return Ok(0.0);
        }
        
        Ok(online_devices as f64 / total_devices as f64)
    }
    
    pub async fn calculate_data_quality(&self) -> Result<f64, MetricsError> {
        let total_records = self.data_quality_service.count_total_records().await?;
        let valid_records = self.data_quality_service.count_valid_records().await?;
        
        if total_records == 0 {
            return Ok(0.0);
        }
        
        Ok(valid_records as f64 / total_records as f64)
    }
    
    pub async fn calculate_average_response_time(&self) -> Result<Duration, MetricsError> {
        self.performance_monitor.get_average_response_time().await
    }
    
    pub async fn calculate_rule_execution_rate(&self) -> Result<f64, MetricsError> {
        let total_rules = self.rule_engine.count_total_rules().await?;
        let executed_rules = self.rule_engine.count_executed_rules().await?;
        
        if total_rules == 0 {
            return Ok(0.0);
        }
        
        Ok(executed_rules as f64 / total_rules as f64)
    }
    
    pub async fn generate_business_report(&self) -> Result<BusinessReport, MetricsError> {
        let metrics = BusinessMetrics {
            device_availability: self.calculate_device_availability().await?,
            data_quality_score: self.calculate_data_quality().await?,
            average_response_time: self.calculate_average_response_time().await?,
            rule_execution_rate: self.calculate_rule_execution_rate().await?,
        };
        
        Ok(BusinessReport {
            metrics,
            timestamp: Utc::now(),
            recommendations: self.generate_recommendations(&metrics).await?,
        })
    }
}
```

---

## 参考文献

1. **IoT 业务建模**: `/docs/Matter/industry_domains/iot/business_modeling.md`
2. **IoT 架构指南**: `/docs/Matter/industry_domains/iot/README.md`

## 相关链接

- [IoT 架构模式](./../01-Architecture/01-IoT-Architecture-Patterns.md)
- [形式化理论基础](./../02-Theory/01-Formal-Theory-Foundation.md)
- [Rust IoT 技术栈](./../04-Technology/01-Rust-IoT-Technology-Stack.md) 