# IoT 设备管理架构分析 (IoT Device Management Architecture Analysis)

## 1. 设备管理基础

### 1.1 设备模型

**定义 1.1 (IoT设备)**
IoT设备是一个六元组 $\mathcal{D} = (I, S, C, P, M, T)$，其中：

- $I$ 是设备标识符
- $S$ 是设备状态
- $C$ 是设备能力
- $P$ 是设备属性
- $M$ 是设备元数据
- $T$ 是时间戳

**定理 1.1 (设备状态一致性)**
设备状态在任何时刻都满足一致性约束：
$$\forall t \in T : S(t) \in \text{ValidStates}$$

**证明：** 通过状态机验证：

1. **初始状态**：设备启动时处于有效状态
2. **状态转移**：所有转移都保持状态有效性
3. **状态保持**：状态在时间演化中保持有效

**算法 1.1 (设备模型实现)**

```rust
/// IoT设备
#[derive(Debug, Clone)]
pub struct IoTDevice {
    pub id: DeviceId,
    pub name: String,
    pub device_type: DeviceType,
    pub model: String,
    pub manufacturer: String,
    pub firmware_version: SemanticVersion,
    pub location: Location,
    pub status: DeviceStatus,
    pub capabilities: Vec<Capability>,
    pub properties: HashMap<String, PropertyValue>,
    pub metadata: HashMap<String, Value>,
    pub last_seen: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// 设备状态
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceStatus {
    Online,
    Offline,
    Maintenance,
    Error,
    Updating,
    Configuring,
}

/// 设备能力
#[derive(Debug, Clone)]
pub struct Capability {
    pub name: String,
    pub version: String,
    pub parameters: Vec<Parameter>,
    pub supported_operations: Vec<Operation>,
}

/// 设备属性
#[derive(Debug, Clone)]
pub enum PropertyValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Vec<PropertyValue>),
    Object(HashMap<String, PropertyValue>),
}

/// 设备管理器
pub struct DeviceManager {
    devices: HashMap<DeviceId, IoTDevice>,
    device_registry: DeviceRegistry,
    status_monitor: StatusMonitor,
    configuration_manager: ConfigurationManager,
}

impl DeviceManager {
    /// 注册设备
    pub async fn register_device(&mut self, device: IoTDevice) -> Result<(), Error> {
        // 验证设备信息
        self.validate_device(&device)?;
        
        // 检查设备唯一性
        if self.devices.contains_key(&device.id) {
            return Err(Error::DeviceAlreadyExists);
        }
        
        // 注册设备
        self.devices.insert(device.id.clone(), device.clone());
        self.device_registry.add_device(&device).await?;
        
        // 初始化监控
        self.status_monitor.start_monitoring(&device.id).await?;
        
        Ok(())
    }
    
    /// 验证设备
    fn validate_device(&self, device: &IoTDevice) -> Result<(), Error> {
        // 检查必要字段
        if device.name.is_empty() {
            return Err(Error::InvalidDeviceName);
        }
        
        if device.device_type == DeviceType::Unknown {
            return Err(Error::InvalidDeviceType);
        }
        
        // 检查固件版本
        if !self.is_valid_firmware_version(&device.firmware_version) {
            return Err(Error::InvalidFirmwareVersion);
        }
        
        Ok(())
    }
    
    /// 获取设备状态
    pub async fn get_device_status(&self, device_id: &DeviceId) -> Result<DeviceStatus, Error> {
        let device = self.devices.get(device_id)
            .ok_or(Error::DeviceNotFound)?;
        
        Ok(device.status.clone())
    }
    
    /// 更新设备状态
    pub async fn update_device_status(&mut self, device_id: &DeviceId, status: DeviceStatus) -> Result<(), Error> {
        let device = self.devices.get_mut(device_id)
            .ok_or(Error::DeviceNotFound)?;
        
        // 验证状态转移
        if !self.is_valid_status_transition(&device.status, &status) {
            return Err(Error::InvalidStatusTransition);
        }
        
        device.status = status;
        device.updated_at = Utc::now();
        
        // 通知状态变化
        self.notify_status_change(device_id, &device.status).await?;
        
        Ok(())
    }
    
    /// 验证状态转移
    fn is_valid_status_transition(&self, from: &DeviceStatus, to: &DeviceStatus) -> bool {
        match (from, to) {
            (DeviceStatus::Offline, DeviceStatus::Online) => true,
            (DeviceStatus::Online, DeviceStatus::Offline) => true,
            (DeviceStatus::Online, DeviceStatus::Maintenance) => true,
            (DeviceStatus::Maintenance, DeviceStatus::Online) => true,
            (DeviceStatus::Online, DeviceStatus::Error) => true,
            (DeviceStatus::Error, DeviceStatus::Online) => true,
            (DeviceStatus::Online, DeviceStatus::Updating) => true,
            (DeviceStatus::Updating, DeviceStatus::Online) => true,
            (DeviceStatus::Online, DeviceStatus::Configuring) => true,
            (DeviceStatus::Configuring, DeviceStatus::Online) => true,
            _ => false,
        }
    }
}
```

### 1.2 设备生命周期管理

**定义 1.2 (设备生命周期)**
设备生命周期包含以下阶段：

1. **注册阶段**：设备注册到系统
2. **配置阶段**：设备参数配置
3. **运行阶段**：设备正常运行
4. **维护阶段**：设备维护和更新
5. **退役阶段**：设备从系统移除

**算法 1.2 (生命周期管理)**

```rust
/// 设备生命周期管理器
pub struct DeviceLifecycleManager {
    lifecycle_states: HashMap<DeviceId, LifecycleState>,
    state_machine: StateMachine,
    event_handler: EventHandler,
}

/// 生命周期状态
#[derive(Debug, Clone, PartialEq)]
pub enum LifecycleState {
    Registered,
    Configured,
    Running,
    Maintenance,
    Retired,
}

/// 状态机
pub struct StateMachine {
    transitions: HashMap<(LifecycleState, LifecycleEvent), LifecycleState>,
}

impl DeviceLifecycleManager {
    /// 处理生命周期事件
    pub async fn handle_lifecycle_event(&mut self, device_id: &DeviceId, event: LifecycleEvent) -> Result<(), Error> {
        let current_state = self.lifecycle_states.get(device_id)
            .ok_or(Error::DeviceNotFound)?;
        
        // 查找状态转移
        let next_state = self.state_machine.get_transition(current_state, &event)
            .ok_or(Error::InvalidTransition)?;
        
        // 执行状态转移
        self.execute_transition(device_id, current_state, &next_state, &event).await?;
        
        // 更新状态
        self.lifecycle_states.insert(device_id.clone(), next_state);
        
        Ok(())
    }
    
    /// 执行状态转移
    async fn execute_transition(&self, device_id: &DeviceId, from: &LifecycleState, to: &LifecycleState, event: &LifecycleEvent) -> Result<(), Error> {
        match (from, to, event) {
            (LifecycleState::Registered, LifecycleState::Configured, LifecycleEvent::Configured) => {
                self.configure_device(device_id).await?;
            },
            (LifecycleState::Configured, LifecycleState::Running, LifecycleEvent::Started) => {
                self.start_device(device_id).await?;
            },
            (LifecycleState::Running, LifecycleState::Maintenance, LifecycleEvent::MaintenanceRequired) => {
                self.enter_maintenance(device_id).await?;
            },
            (LifecycleState::Maintenance, LifecycleState::Running, LifecycleEvent::MaintenanceCompleted) => {
                self.exit_maintenance(device_id).await?;
            },
            (LifecycleState::Running, LifecycleState::Retired, LifecycleEvent::Retired) => {
                self.retire_device(device_id).await?;
            },
            _ => return Err(Error::InvalidTransition),
        }
        
        Ok(())
    }
    
    /// 配置设备
    async fn configure_device(&self, device_id: &DeviceId) -> Result<(), Error> {
        // 加载设备配置
        let configuration = self.load_device_configuration(device_id).await?;
        
        // 应用配置
        self.apply_configuration(device_id, &configuration).await?;
        
        // 验证配置
        self.validate_configuration(device_id, &configuration).await?;
        
        Ok(())
    }
    
    /// 启动设备
    async fn start_device(&self, device_id: &DeviceId) -> Result<(), Error> {
        // 初始化设备
        self.initialize_device(device_id).await?;
        
        // 启动监控
        self.start_monitoring(device_id).await?;
        
        // 通知设备启动
        self.notify_device_started(device_id).await?;
        
        Ok(())
    }
    
    /// 进入维护模式
    async fn enter_maintenance(&self, device_id: &DeviceId) -> Result<(), Error> {
        // 停止正常操作
        self.stop_normal_operations(device_id).await?;
        
        // 保存当前状态
        self.save_device_state(device_id).await?;
        
        // 通知维护开始
        self.notify_maintenance_started(device_id).await?;
        
        Ok(())
    }
    
    /// 退出维护模式
    async fn exit_maintenance(&self, device_id: &DeviceId) -> Result<(), Error> {
        // 恢复设备状态
        self.restore_device_state(device_id).await?;
        
        // 重新启动操作
        self.resume_normal_operations(device_id).await?;
        
        // 通知维护完成
        self.notify_maintenance_completed(device_id).await?;
        
        Ok(())
    }
    
    /// 退役设备
    async fn retire_device(&self, device_id: &DeviceId) -> Result<(), Error> {
        // 停止所有操作
        self.stop_all_operations(device_id).await?;
        
        // 备份设备数据
        self.backup_device_data(device_id).await?;
        
        // 清理资源
        self.cleanup_resources(device_id).await?;
        
        // 通知设备退役
        self.notify_device_retired(device_id).await?;
        
        Ok(())
    }
}
```

## 2. 设备配置管理

### 2.1 配置模型

**定义 2.1 (设备配置)**
设备配置是一个四元组 $\mathcal{C} = (P, V, C, M)$，其中：

- $P$ 是参数集
- $V$ 是参数值
- $C$ 是约束条件
- $M$ 是元数据

**定理 2.1 (配置一致性)**
设备配置满足一致性约束：
$$\forall p \in P : V(p) \in C(p)$$

**算法 2.1 (配置管理)**

```rust
/// 设备配置
#[derive(Debug, Clone)]
pub struct DeviceConfiguration {
    pub device_id: DeviceId,
    pub parameters: HashMap<String, ConfigurationParameter>,
    pub version: u32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// 配置参数
#[derive(Debug, Clone)]
pub struct ConfigurationParameter {
    pub name: String,
    pub value: ParameterValue,
    pub data_type: DataType,
    pub constraints: Vec<Constraint>,
    pub description: Option<String>,
    pub default_value: Option<ParameterValue>,
}

/// 参数值
#[derive(Debug, Clone)]
pub enum ParameterValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Vec<ParameterValue>),
    Object(HashMap<String, ParameterValue>),
}

/// 约束条件
#[derive(Debug, Clone)]
pub enum Constraint {
    Range { min: f64, max: f64 },
    Enum { values: Vec<String> },
    Pattern { regex: String },
    Required,
    Optional,
}

/// 配置管理器
pub struct ConfigurationManager {
    configurations: HashMap<DeviceId, DeviceConfiguration>,
    configuration_store: ConfigurationStore,
    validator: ConfigurationValidator,
    notifier: ConfigurationNotifier,
}

impl ConfigurationManager {
    /// 获取设备配置
    pub async fn get_configuration(&self, device_id: &DeviceId) -> Result<DeviceConfiguration, Error> {
        self.configurations.get(device_id)
            .cloned()
            .ok_or(Error::ConfigurationNotFound)
    }
    
    /// 更新设备配置
    pub async fn update_configuration(&mut self, device_id: &DeviceId, updates: HashMap<String, ParameterValue>) -> Result<(), Error> {
        let mut configuration = self.get_configuration(device_id).await?;
        
        // 验证更新
        for (param_name, new_value) in &updates {
            self.validate_parameter_update(&configuration, param_name, new_value).await?;
        }
        
        // 应用更新
        for (param_name, new_value) in updates {
            if let Some(parameter) = configuration.parameters.get_mut(&param_name) {
                parameter.value = new_value;
            }
        }
        
        configuration.version += 1;
        configuration.updated_at = Utc::now();
        
        // 保存配置
        self.configurations.insert(device_id.clone(), configuration.clone());
        self.configuration_store.save_configuration(&configuration).await?;
        
        // 通知配置变化
        self.notifier.notify_configuration_changed(device_id, &configuration).await?;
        
        Ok(())
    }
    
    /// 验证参数更新
    async fn validate_parameter_update(&self, configuration: &DeviceConfiguration, param_name: &str, new_value: &ParameterValue) -> Result<(), Error> {
        let parameter = configuration.parameters.get(param_name)
            .ok_or(Error::ParameterNotFound)?;
        
        // 检查数据类型
        if !self.is_compatible_type(&parameter.data_type, new_value) {
            return Err(Error::TypeMismatch);
        }
        
        // 检查约束
        for constraint in &parameter.constraints {
            if !self.satisfies_constraint(new_value, constraint) {
                return Err(Error::ConstraintViolation);
            }
        }
        
        Ok(())
    }
    
    /// 检查类型兼容性
    fn is_compatible_type(&self, data_type: &DataType, value: &ParameterValue) -> bool {
        match (data_type, value) {
            (DataType::String, ParameterValue::String(_)) => true,
            (DataType::Integer, ParameterValue::Integer(_)) => true,
            (DataType::Float, ParameterValue::Float(_)) => true,
            (DataType::Boolean, ParameterValue::Boolean(_)) => true,
            (DataType::Array, ParameterValue::Array(_)) => true,
            (DataType::Object, ParameterValue::Object(_)) => true,
            _ => false,
        }
    }
    
    /// 检查约束满足
    fn satisfies_constraint(&self, value: &ParameterValue, constraint: &Constraint) -> bool {
        match constraint {
            Constraint::Range { min, max } => {
                if let ParameterValue::Float(v) = value {
                    *v >= *min && *v <= *max
                } else if let ParameterValue::Integer(v) = value {
                    *v as f64 >= *min && *v as f64 <= *max
                } else {
                    false
                }
            },
            Constraint::Enum { values } => {
                if let ParameterValue::String(v) = value {
                    values.contains(v)
                } else {
                    false
                }
            },
            Constraint::Pattern { regex } => {
                if let ParameterValue::String(v) = value {
                    Regex::new(regex).unwrap().is_match(v)
                } else {
                    false
                }
            },
            Constraint::Required => true, // 已通过存在性检查
            Constraint::Optional => true,
        }
    }
    
    /// 批量配置更新
    pub async fn batch_update_configuration(&mut self, updates: HashMap<DeviceId, HashMap<String, ParameterValue>>) -> Result<(), Error> {
        let mut results = Vec::new();
        
        for (device_id, device_updates) in updates {
            let result = self.update_configuration(&device_id, device_updates).await;
            results.push((device_id, result));
        }
        
        // 检查是否有失败
        let failures: Vec<_> = results.iter()
            .filter(|(_, result)| result.is_err())
            .collect();
        
        if !failures.is_empty() {
            return Err(Error::BatchUpdateFailed { failures: failures.len() });
        }
        
        Ok(())
    }
}
```

### 2.2 配置同步机制

**定义 2.2 (配置同步)**
配置同步确保设备配置与云端配置保持一致。

**算法 2.2 (配置同步)**

```rust
/// 配置同步器
pub struct ConfigurationSynchronizer {
    sync_strategy: SyncStrategy,
    conflict_resolver: ConflictResolver,
    sync_history: SyncHistory,
}

/// 同步策略
#[derive(Debug, Clone)]
pub enum SyncStrategy {
    /// 云端优先
    CloudFirst,
    /// 设备优先
    DeviceFirst,
    /// 时间戳优先
    TimestampFirst,
    /// 手动解决
    ManualResolution,
}

/// 配置同步器实现
impl ConfigurationSynchronizer {
    /// 同步配置
    pub async fn sync_configuration(&mut self, device_id: &DeviceId, cloud_config: &DeviceConfiguration, device_config: &DeviceConfiguration) -> Result<DeviceConfiguration, Error> {
        // 检查是否需要同步
        if self.configurations_are_synchronized(cloud_config, device_config) {
            return Ok(cloud_config.clone());
        }
        
        // 检测冲突
        let conflicts = self.detect_conflicts(cloud_config, device_config);
        
        if conflicts.is_empty() {
            // 无冲突，直接同步
            self.perform_sync(device_id, cloud_config).await?;
            Ok(cloud_config.clone())
        } else {
            // 有冲突，需要解决
            let resolved_config = self.resolve_conflicts(conflicts, cloud_config, device_config).await?;
            self.perform_sync(device_id, &resolved_config).await?;
            Ok(resolved_config)
        }
    }
    
    /// 检查配置是否同步
    fn configurations_are_synchronized(&self, cloud_config: &DeviceConfiguration, device_config: &DeviceConfiguration) -> bool {
        cloud_config.version == device_config.version &&
        cloud_config.updated_at == device_config.updated_at
    }
    
    /// 检测冲突
    fn detect_conflicts(&self, cloud_config: &DeviceConfiguration, device_config: &DeviceConfiguration) -> Vec<ConfigurationConflict> {
        let mut conflicts = Vec::new();
        
        for (param_name, cloud_param) in &cloud_config.parameters {
            if let Some(device_param) = device_config.parameters.get(param_name) {
                if cloud_param.value != device_param.value {
                    conflicts.push(ConfigurationConflict {
                        parameter_name: param_name.clone(),
                        cloud_value: cloud_param.value.clone(),
                        device_value: device_param.value.clone(),
                        cloud_timestamp: cloud_config.updated_at,
                        device_timestamp: device_config.updated_at,
                    });
                }
            }
        }
        
        conflicts
    }
    
    /// 解决冲突
    async fn resolve_conflicts(&self, conflicts: Vec<ConfigurationConflict>, cloud_config: &DeviceConfiguration, device_config: &DeviceConfiguration) -> Result<DeviceConfiguration, Error> {
        let mut resolved_config = cloud_config.clone();
        
        for conflict in conflicts {
            let resolved_value = match self.sync_strategy {
                SyncStrategy::CloudFirst => conflict.cloud_value,
                SyncStrategy::DeviceFirst => conflict.device_value,
                SyncStrategy::TimestampFirst => {
                    if conflict.cloud_timestamp > conflict.device_timestamp {
                        conflict.cloud_value
                    } else {
                        conflict.device_value
                    }
                },
                SyncStrategy::ManualResolution => {
                    self.conflict_resolver.resolve_conflict(&conflict).await?
                },
            };
            
            if let Some(parameter) = resolved_config.parameters.get_mut(&conflict.parameter_name) {
                parameter.value = resolved_value;
            }
        }
        
        resolved_config.version += 1;
        resolved_config.updated_at = Utc::now();
        
        Ok(resolved_config)
    }
    
    /// 执行同步
    async fn perform_sync(&mut self, device_id: &DeviceId, config: &DeviceConfiguration) -> Result<(), Error> {
        // 更新云端配置
        self.update_cloud_configuration(device_id, config).await?;
        
        // 推送配置到设备
        self.push_configuration_to_device(device_id, config).await?;
        
        // 记录同步历史
        self.sync_history.record_sync(device_id, config).await?;
        
        Ok(())
    }
}
```

## 3. 设备状态监控

### 3.1 状态监控模型

**定义 3.1 (状态监控)**
状态监控是一个五元组 $\mathcal{M} = (D, S, T, A, R)$，其中：

- $D$ 是设备集
- $S$ 是状态集
- $T$ 是时间序列
- $A$ 是告警规则
- $R$ 是报告机制

**算法 3.1 (状态监控)**

```rust
/// 状态监控器
pub struct StatusMonitor {
    monitored_devices: HashMap<DeviceId, DeviceMonitor>,
    alert_rules: Vec<AlertRule>,
    alert_manager: AlertManager,
    metrics_collector: MetricsCollector,
}

/// 设备监控器
pub struct DeviceMonitor {
    device_id: DeviceId,
    current_status: DeviceStatus,
    status_history: Vec<StatusRecord>,
    health_metrics: HealthMetrics,
    last_check: DateTime<Utc>,
}

/// 状态记录
#[derive(Debug, Clone)]
pub struct StatusRecord {
    pub timestamp: DateTime<Utc>,
    pub status: DeviceStatus,
    pub metrics: HashMap<String, f64>,
    pub events: Vec<DeviceEvent>,
}

/// 健康指标
#[derive(Debug, Clone)]
pub struct HealthMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_usage: f64,
    pub network_usage: f64,
    pub battery_level: Option<f64>,
    pub temperature: Option<f64>,
    pub signal_strength: Option<f64>,
}

/// 告警规则
#[derive(Debug, Clone)]
pub struct AlertRule {
    pub id: String,
    pub name: String,
    pub condition: AlertCondition,
    pub severity: AlertSeverity,
    pub actions: Vec<AlertAction>,
}

/// 告警条件
#[derive(Debug, Clone)]
pub enum AlertCondition {
    StatusChanged { from: DeviceStatus, to: DeviceStatus },
    MetricThreshold { metric: String, operator: ComparisonOperator, threshold: f64 },
    HealthDegraded { min_health_score: f64 },
    OfflineDuration { max_duration: Duration },
}

impl StatusMonitor {
    /// 监控设备状态
    pub async fn monitor_device(&mut self, device_id: &DeviceId) -> Result<(), Error> {
        let monitor = self.monitored_devices.get_mut(device_id)
            .ok_or(Error::DeviceNotFound)?;
        
        // 收集当前状态
        let current_status = self.collect_device_status(device_id).await?;
        let health_metrics = self.collect_health_metrics(device_id).await?;
        let events = self.collect_device_events(device_id).await?;
        
        // 创建状态记录
        let status_record = StatusRecord {
            timestamp: Utc::now(),
            status: current_status.clone(),
            metrics: health_metrics.to_hash_map(),
            events,
        };
        
        // 更新监控器
        monitor.status_history.push(status_record);
        monitor.current_status = current_status;
        monitor.health_metrics = health_metrics;
        monitor.last_check = Utc::now();
        
        // 检查告警规则
        self.check_alert_rules(device_id, &monitor).await?;
        
        Ok(())
    }
    
    /// 收集设备状态
    async fn collect_device_status(&self, device_id: &DeviceId) -> Result<DeviceStatus, Error> {
        // 这里应该实现实际的设备状态收集逻辑
        // 例如：通过设备API、网络ping、心跳检测等
        
        // 模拟状态收集
        if self.is_device_responding(device_id).await? {
            Ok(DeviceStatus::Online)
        } else {
            Ok(DeviceStatus::Offline)
        }
    }
    
    /// 收集健康指标
    async fn collect_health_metrics(&self, device_id: &DeviceId) -> Result<HealthMetrics, Error> {
        // 这里应该实现实际的指标收集逻辑
        // 例如：通过SNMP、设备API、监控代理等
        
        // 模拟指标收集
        Ok(HealthMetrics {
            cpu_usage: self.get_cpu_usage(device_id).await?,
            memory_usage: self.get_memory_usage(device_id).await?,
            disk_usage: self.get_disk_usage(device_id).await?,
            network_usage: self.get_network_usage(device_id).await?,
            battery_level: self.get_battery_level(device_id).await?,
            temperature: self.get_temperature(device_id).await?,
            signal_strength: self.get_signal_strength(device_id).await?,
        })
    }
    
    /// 检查告警规则
    async fn check_alert_rules(&self, device_id: &DeviceId, monitor: &DeviceMonitor) -> Result<(), Error> {
        for rule in &self.alert_rules {
            if self.evaluate_alert_condition(rule, device_id, monitor).await? {
                let alert = Alert {
                    id: Uuid::new_v4().to_string(),
                    rule_id: rule.id.clone(),
                    device_id: device_id.clone(),
                    severity: rule.severity.clone(),
                    message: self.generate_alert_message(rule, monitor),
                    timestamp: Utc::now(),
                };
                
                self.alert_manager.trigger_alert(alert).await?;
            }
        }
        
        Ok(())
    }
    
    /// 评估告警条件
    async fn evaluate_alert_condition(&self, rule: &AlertRule, device_id: &DeviceId, monitor: &DeviceMonitor) -> Result<bool, Error> {
        match &rule.condition {
            AlertCondition::StatusChanged { from, to } => {
                Ok(monitor.current_status == *to)
            },
            AlertCondition::MetricThreshold { metric, operator, threshold } => {
                let current_value = monitor.health_metrics.get_metric(metric)?;
                Ok(self.compare_values(current_value, *operator, *threshold))
            },
            AlertCondition::HealthDegraded { min_health_score } => {
                let health_score = self.calculate_health_score(&monitor.health_metrics);
                Ok(health_score < *min_health_score)
            },
            AlertCondition::OfflineDuration { max_duration } => {
                if monitor.current_status == DeviceStatus::Offline {
                    let offline_duration = Utc::now() - monitor.last_check;
                    Ok(offline_duration > *max_duration)
                } else {
                    Ok(false)
                }
            },
        }
    }
    
    /// 计算健康分数
    fn calculate_health_score(&self, metrics: &HealthMetrics) -> f64 {
        let mut score = 100.0;
        
        // CPU使用率影响
        if metrics.cpu_usage > 80.0 {
            score -= (metrics.cpu_usage - 80.0) * 0.5;
        }
        
        // 内存使用率影响
        if metrics.memory_usage > 85.0 {
            score -= (metrics.memory_usage - 85.0) * 0.3;
        }
        
        // 磁盘使用率影响
        if metrics.disk_usage > 90.0 {
            score -= (metrics.disk_usage - 90.0) * 0.2;
        }
        
        // 电池电量影响
        if let Some(battery) = metrics.battery_level {
            if battery < 20.0 {
                score -= (20.0 - battery) * 0.5;
            }
        }
        
        // 温度影响
        if let Some(temp) = metrics.temperature {
            if temp > 70.0 {
                score -= (temp - 70.0) * 0.3;
            }
        }
        
        score.max(0.0)
    }
}
```

## 4. 总结

本文档建立了完整的IoT设备管理架构分析框架，包括：

1. **设备管理基础**：提供了设备模型和生命周期管理
2. **设备配置管理**：实现了配置同步和冲突解决
3. **设备状态监控**：提供了健康监控和告警机制

这些架构组件为IoT系统的设备管理提供了完整的解决方案。

---

**参考文献：**

- [IoT系统架构分析](../02-System/01-IoT_System_Architecture.md)
- [通信协议分析](../../02-Technology/01-Protocol/01-Communication_Protocols.md)
