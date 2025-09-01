# 实现模式设计文档

## 概述

本文档详细阐述IoT系统中常用的实现模式，包括创建型模式、结构型模式、行为型模式等，为构建高质量、可维护的IoT组件提供实现指导。

## 1. 创建型模式 (Creational Patterns)

### 1.1 工厂模式 (Factory Pattern)

```rust
// 抽象工厂接口
pub trait SensorFactory {
    type Sensor: Sensor;
    fn create_sensor(&self, config: &SensorConfig) -> Result<Self::Sensor, FactoryError>;
}

// 具体工厂实现
pub struct TemperatureSensorFactory;
impl SensorFactory for TemperatureSensorFactory {
    type Sensor = TemperatureSensor;
    
    fn create_sensor(&self, config: &SensorConfig) -> Result<Self::Sensor, FactoryError> {
        Ok(TemperatureSensor::new(
            config.device_id.clone(),
            config.sampling_rate,
            config.calibration_data.clone(),
        )?)
    }
}

pub struct HumiditySensorFactory;
impl SensorFactory for HumiditySensorFactory {
    type Sensor = HumiditySensor;
    
    fn create_sensor(&self, config: &SensorConfig) -> Result<Self::Sensor, FactoryError> {
        Ok(HumiditySensor::new(
            config.device_id.clone(),
            config.sampling_rate,
            config.calibration_data.clone(),
        )?)
    }
}

// 工厂注册表
pub struct SensorFactoryRegistry {
    factories: HashMap<SensorType, Box<dyn SensorFactory<Sensor = Box<dyn Sensor>>>>,
}

impl SensorFactoryRegistry {
    pub fn register<T: SensorFactory + 'static>(&mut self, sensor_type: SensorType, factory: T) 
    where
        T::Sensor: Sensor + 'static,
    {
        self.factories.insert(sensor_type, Box::new(factory));
    }
    
    pub fn create_sensor(&self, sensor_type: &SensorType, config: &SensorConfig) -> Result<Box<dyn Sensor>, FactoryError> {
        let factory = self.factories.get(sensor_type)
            .ok_or_else(|| FactoryError::UnsupportedSensorType(sensor_type.clone()))?;
        
        factory.create_sensor(config)
    }
}
```

### 1.2 建造者模式 (Builder Pattern)

```rust
// IoT设备配置建造者
pub struct IoTDeviceBuilder {
    device_id: Option<DeviceId>,
    device_name: Option<String>,
    device_type: Option<DeviceType>,
    network_config: Option<NetworkConfig>,
    sensor_configs: Vec<SensorConfig>,
    security_config: Option<SecurityConfig>,
    power_config: Option<PowerConfig>,
}

impl IoTDeviceBuilder {
    pub fn new() -> Self {
        Self {
            device_id: None,
            device_name: None,
            device_type: None,
            network_config: None,
            sensor_configs: Vec::new(),
            security_config: None,
            power_config: None,
        }
    }
    
    pub fn with_device_id(mut self, device_id: DeviceId) -> Self {
        self.device_id = Some(device_id);
        self
    }
    
    pub fn with_device_name(mut self, name: String) -> Self {
        self.device_name = Some(name);
        self
    }
    
    pub fn with_device_type(mut self, device_type: DeviceType) -> Self {
        self.device_type = Some(device_type);
        self
    }
    
    pub fn with_network_config(mut self, config: NetworkConfig) -> Self {
        self.network_config = Some(config);
        self
    }
    
    pub fn add_sensor(mut self, sensor_config: SensorConfig) -> Self {
        self.sensor_configs.push(sensor_config);
        self
    }
    
    pub fn with_security_config(mut self, config: SecurityConfig) -> Self {
        self.security_config = Some(config);
        self
    }
    
    pub fn with_power_config(mut self, config: PowerConfig) -> Self {
        self.power_config = Some(config);
        self
    }
    
    pub fn build(self) -> Result<IoTDevice, BuildError> {
        let device_id = self.device_id.ok_or(BuildError::MissingDeviceId)?;
        let device_name = self.device_name.ok_or(BuildError::MissingDeviceName)?;
        let device_type = self.device_type.ok_or(BuildError::MissingDeviceType)?;
        let network_config = self.network_config.ok_or(BuildError::MissingNetworkConfig)?;
        let security_config = self.security_config.ok_or(BuildError::MissingSecurityConfig)?;
        let power_config = self.power_config.ok_or(BuildError::MissingPowerConfig)?;
        
        Ok(IoTDevice {
            id: device_id,
            name: device_name,
            device_type,
            network_config,
            sensor_configs: self.sensor_configs,
            security_config,
            power_config,
            status: DeviceStatus::Offline,
            created_at: Utc::now(),
        })
    }
}

// 使用示例
let device = IoTDeviceBuilder::new()
    .with_device_id(DeviceId::generate())
    .with_device_name("Temperature Sensor 001".to_string())
    .with_device_type(DeviceType::TemperatureSensor)
    .with_network_config(NetworkConfig::wifi("SSID", "password"))
    .add_sensor(SensorConfig::temperature(1.0, 100.0))
    .with_security_config(SecurityConfig::tls_only())
    .with_power_config(PowerConfig::battery_powered())
    .build()?;
```

### 1.3 单例模式 (Singleton Pattern)

```rust
// 线程安全的单例模式
pub struct IoTConfigurationManager {
    config: Arc<RwLock<Configuration>>,
    watchers: Arc<Mutex<Vec<Box<dyn ConfigWatcher>>>>,
}

impl IoTConfigurationManager {
    // 使用OnceCell确保只初始化一次
    static INSTANCE: OnceCell<Arc<IoTConfigurationManager>> = OnceCell::new();
    
    pub fn instance() -> Arc<Self> {
        INSTANCE.get_or_init(|| {
            Arc::new(Self::new())
        }).clone()
    }
    
    fn new() -> Self {
        Self {
            config: Arc::new(RwLock::new(Configuration::default())),
            watchers: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    pub async fn load_config(&self, config_path: &Path) -> Result<(), ConfigError> {
        let config_data = tokio::fs::read_to_string(config_path).await?;
        let config: Configuration = toml::from_str(&config_data)?;
        
        {
            let mut current_config = self.config.write().await;
            *current_config = config;
        }
        
        // 通知所有观察者
        self.notify_watchers().await;
        
        Ok(())
    }
    
    pub async fn get_config(&self) -> Configuration {
        self.config.read().await.clone()
    }
    
    pub async fn update_config<F>(&self, updater: F) -> Result<(), ConfigError>
    where
        F: FnOnce(&mut Configuration) -> Result<(), ConfigError>,
    {
        {
            let mut config = self.config.write().await;
            updater(&mut config)?;
        }
        
        self.notify_watchers().await;
        Ok(())
    }
    
    async fn notify_watchers(&self) {
        let watchers = self.watchers.lock().unwrap();
        for watcher in watchers.iter() {
            watcher.on_config_changed().await;
        }
    }
}
```

## 2. 结构型模式 (Structural Patterns)

### 2.1 适配器模式 (Adapter Pattern)

```rust
// 目标接口
pub trait DataProcessor {
    async fn process(&self, data: &RawData) -> Result<ProcessedData, ProcessingError>;
}

// 需要适配的第三方库接口
pub struct LegacyDataProcessor {
    // 第三方库的具体实现
}

impl LegacyDataProcessor {
    pub fn legacy_process(&self, input: &[u8]) -> Result<Vec<u8>, LegacyError> {
        // 第三方库的处理逻辑
        Ok(input.to_vec())
    }
}

// 适配器实现
pub struct LegacyDataProcessorAdapter {
    legacy_processor: LegacyDataProcessor,
    data_converter: DataConverter,
}

impl LegacyDataProcessorAdapter {
    pub fn new(legacy_processor: LegacyDataProcessor) -> Self {
        Self {
            legacy_processor,
            data_converter: DataConverter::new(),
        }
    }
}

impl DataProcessor for LegacyDataProcessorAdapter {
    async fn process(&self, data: &RawData) -> Result<ProcessedData, ProcessingError> {
        // 1. 转换数据格式
        let legacy_input = self.data_converter.to_legacy_format(data)?;
        
        // 2. 调用第三方库
        let legacy_output = self.legacy_processor.legacy_process(&legacy_input)
            .map_err(|e| ProcessingError::LegacyProcessingError(e))?;
        
        // 3. 转换回标准格式
        let processed_data = self.data_converter.from_legacy_format(&legacy_output)?;
        
        Ok(processed_data)
    }
}

// 数据转换器
pub struct DataConverter;

impl DataConverter {
    pub fn new() -> Self {
        Self
    }
    
    pub fn to_legacy_format(&self, data: &RawData) -> Result<Vec<u8>, ConversionError> {
        // 转换逻辑
        Ok(data.to_bytes())
    }
    
    pub fn from_legacy_format(&self, data: &[u8]) -> Result<ProcessedData, ConversionError> {
        // 转换逻辑
        Ok(ProcessedData::from_bytes(data))
    }
}
```

### 2.2 装饰器模式 (Decorator Pattern)

```rust
// 基础组件接口
pub trait DataTransmission {
    async fn transmit(&self, data: &Data) -> Result<(), TransmissionError>;
}

// 基础传输实现
pub struct BasicTransmission {
    endpoint: String,
}

impl DataTransmission for BasicTransmission {
    async fn transmit(&self, data: &Data) -> Result<(), TransmissionError> {
        // 基础传输逻辑
        println!("Transmitting data to {}", self.endpoint);
        Ok(())
    }
}

// 装饰器基类
pub struct TransmissionDecorator {
    inner: Box<dyn DataTransmission>,
}

impl TransmissionDecorator {
    pub fn new(inner: Box<dyn DataTransmission>) -> Self {
        Self { inner }
    }
}

impl DataTransmission for TransmissionDecorator {
    async fn transmit(&self, data: &Data) -> Result<(), TransmissionError> {
        self.inner.transmit(data).await
    }
}

// 加密装饰器
pub struct EncryptionDecorator {
    inner: Box<dyn DataTransmission>,
    encryption_key: Vec<u8>,
}

impl EncryptionDecorator {
    pub fn new(inner: Box<dyn DataTransmission>, key: Vec<u8>) -> Self {
        Self {
            inner,
            encryption_key: key,
        }
    }
}

impl DataTransmission for EncryptionDecorator {
    async fn transmit(&self, data: &Data) -> Result<(), TransmissionError> {
        // 1. 加密数据
        let encrypted_data = self.encrypt_data(data)?;
        
        // 2. 传输加密后的数据
        self.inner.transmit(&encrypted_data).await
    }
}

impl EncryptionDecorator {
    fn encrypt_data(&self, data: &Data) -> Result<Data, TransmissionError> {
        // 加密逻辑
        Ok(Data::encrypted(data.to_bytes(), &self.encryption_key))
    }
}

// 压缩装饰器
pub struct CompressionDecorator {
    inner: Box<dyn DataTransmission>,
    compression_level: u32,
}

impl CompressionDecorator {
    pub fn new(inner: Box<dyn DataTransmission>, level: u32) -> Self {
        Self {
            inner,
            compression_level: level,
        }
    }
}

impl DataTransmission for CompressionDecorator {
    async fn transmit(&self, data: &Data) -> Result<(), TransmissionError> {
        // 1. 压缩数据
        let compressed_data = self.compress_data(data)?;
        
        // 2. 传输压缩后的数据
        self.inner.transmit(&compressed_data).await
    }
}

impl CompressionDecorator {
    fn compress_data(&self, data: &Data) -> Result<Data, TransmissionError> {
        // 压缩逻辑
        Ok(Data::compressed(data.to_bytes(), self.compression_level))
    }
}

// 使用示例
let transmission = Box::new(
    CompressionDecorator::new(
        Box::new(
            EncryptionDecorator::new(
                Box::new(BasicTransmission::new("https://api.iot.com".to_string())),
                encryption_key,
            )
        ),
        6, // 压缩级别
    )
);

transmission.transmit(&data).await?;
```

### 2.3 外观模式 (Facade Pattern)

```rust
// 复杂的IoT系统子系统
pub struct DeviceManagementSystem {
    device_registry: DeviceRegistry,
    device_monitor: DeviceMonitor,
    device_configurator: DeviceConfigurator,
}

pub struct DataProcessingSystem {
    data_ingestion: DataIngestion,
    data_processor: DataProcessor,
    data_storage: DataStorage,
}

pub struct SecuritySystem {
    authentication: AuthenticationService,
    authorization: AuthorizationService,
    encryption: EncryptionService,
}

// 外观类
pub struct IoTSystemFacade {
    device_management: DeviceManagementSystem,
    data_processing: DataProcessingSystem,
    security: SecuritySystem,
}

impl IoTSystemFacade {
    pub fn new() -> Result<Self, SystemError> {
        Ok(Self {
            device_management: DeviceManagementSystem::new()?,
            data_processing: DataProcessingSystem::new()?,
            security: SecuritySystem::new()?,
        })
    }
    
    // 简化的设备注册接口
    pub async fn register_device(&self, device_info: DeviceInfo) -> Result<DeviceId, SystemError> {
        // 1. 安全验证
        self.security.authentication.authenticate_device(&device_info)?;
        
        // 2. 注册设备
        let device_id = self.device_management.device_registry.register(device_info.clone())?;
        
        // 3. 配置设备
        self.device_management.device_configurator.configure(&device_id, &device_info.config)?;
        
        // 4. 开始监控
        self.device_management.device_monitor.start_monitoring(&device_id)?;
        
        Ok(device_id)
    }
    
    // 简化的数据处理接口
    pub async fn process_device_data(&self, device_id: &DeviceId, raw_data: &RawData) -> Result<ProcessedData, SystemError> {
        // 1. 验证设备权限
        self.security.authorization.authorize_data_access(device_id)?;
        
        // 2. 摄取数据
        let ingested_data = self.data_processing.data_ingestion.ingest(raw_data)?;
        
        // 3. 处理数据
        let processed_data = self.data_processing.data_processor.process(&ingested_data)?;
        
        // 4. 存储结果
        self.data_processing.data_storage.store(&processed_data)?;
        
        Ok(processed_data)
    }
    
    // 简化的系统状态查询
    pub async fn get_system_status(&self) -> Result<SystemStatus, SystemError> {
        let device_status = self.device_management.device_monitor.get_status().await?;
        let data_status = self.data_processing.data_storage.get_status().await?;
        let security_status = self.security.authentication.get_status().await?;
        
        Ok(SystemStatus {
            device_count: device_status.total_devices,
            active_devices: device_status.active_devices,
            data_volume: data_status.total_volume,
            security_level: security_status.current_level,
            overall_health: self.calculate_overall_health(&device_status, &data_status, &security_status),
        })
    }
    
    fn calculate_overall_health(&self, device: &DeviceStatus, data: &DataStatus, security: &SecurityStatus) -> HealthLevel {
        // 健康度计算逻辑
        if device.active_devices > 0 && data.total_volume > 0 && security.current_level >= SecurityLevel::High {
            HealthLevel::Healthy
        } else {
            HealthLevel::Degraded
        }
    }
}
```

## 3. 行为型模式 (Behavioral Patterns)

### 3.1 观察者模式 (Observer Pattern)

```rust
// 事件发布者
pub trait EventPublisher {
    fn subscribe(&mut self, observer: Box<dyn EventObserver>);
    fn unsubscribe(&mut self, observer_id: ObserverId);
    async fn notify(&self, event: &Event);
}

// 事件观察者
pub trait EventObserver: Send + Sync {
    fn get_id(&self) -> ObserverId;
    async fn on_event(&self, event: &Event);
}

// IoT事件管理器
pub struct IoTEventManager {
    observers: HashMap<ObserverId, Box<dyn EventObserver>>,
    event_queue: Arc<Mutex<VecDeque<Event>>>,
}

impl IoTEventManager {
    pub fn new() -> Self {
        Self {
            observers: HashMap::new(),
            event_queue: Arc::new(Mutex::new(VecDeque::new())),
        }
    }
    
    pub async fn start_event_loop(&self) {
        let queue = self.event_queue.clone();
        let observers = self.observers.clone();
        
        tokio::spawn(async move {
            loop {
                if let Some(event) = queue.lock().unwrap().pop_front() {
                    for observer in observers.values() {
                        observer.on_event(&event).await;
                    }
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        });
    }
    
    pub fn publish_event(&self, event: Event) {
        self.event_queue.lock().unwrap().push_back(event);
    }
}

impl EventPublisher for IoTEventManager {
    fn subscribe(&mut self, observer: Box<dyn EventObserver>) {
        let id = observer.get_id();
        self.observers.insert(id, observer);
    }
    
    fn unsubscribe(&mut self, observer_id: ObserverId) {
        self.observers.remove(&observer_id);
    }
    
    async fn notify(&self, event: &Event) {
        for observer in self.observers.values() {
            observer.on_event(event).await;
        }
    }
}

// 具体观察者实现
pub struct DeviceStatusObserver {
    id: ObserverId,
    notification_service: Arc<dyn NotificationService>,
}

impl DeviceStatusObserver {
    pub fn new(notification_service: Arc<dyn NotificationService>) -> Self {
        Self {
            id: ObserverId::generate(),
            notification_service,
        }
    }
}

impl EventObserver for DeviceStatusObserver {
    fn get_id(&self) -> ObserverId {
        self.id
    }
    
    async fn on_event(&self, event: &Event) {
        match event.event_type {
            EventType::DeviceOffline => {
                let notification = Notification {
                    title: "Device Offline".to_string(),
                    message: format!("Device {} is offline", event.device_id),
                    severity: NotificationSeverity::Warning,
                };
                
                self.notification_service.send_notification(notification).await;
            }
            EventType::DeviceOnline => {
                let notification = Notification {
                    title: "Device Online".to_string(),
                    message: format!("Device {} is back online", event.device_id),
                    severity: NotificationSeverity::Info,
                };
                
                self.notification_service.send_notification(notification).await;
            }
            _ => {}
        }
    }
}
```

### 3.2 策略模式 (Strategy Pattern)

```rust
// 数据处理策略接口
pub trait DataProcessingStrategy {
    async fn process(&self, data: &RawData) -> Result<ProcessedData, ProcessingError>;
    fn get_strategy_name(&self) -> &str;
}

// 实时处理策略
pub struct RealTimeProcessingStrategy {
    max_latency: Duration,
}

impl DataProcessingStrategy for RealTimeProcessingStrategy {
    async fn process(&self, data: &RawData) -> Result<ProcessedData, ProcessingError> {
        let start_time = Instant::now();
        
        // 快速处理逻辑
        let processed = self.fast_process(data)?;
        
        let processing_time = start_time.elapsed();
        if processing_time > self.max_latency {
            return Err(ProcessingError::LatencyExceeded(processing_time));
        }
        
        Ok(processed)
    }
    
    fn get_strategy_name(&self) -> &str {
        "RealTime"
    }
}

// 批量处理策略
pub struct BatchProcessingStrategy {
    batch_size: usize,
    processing_algorithm: Box<dyn ProcessingAlgorithm>,
}

impl DataProcessingStrategy for BatchProcessingStrategy {
    async fn process(&self, data: &RawData) -> Result<ProcessedData, ProcessingError> {
        // 批量处理逻辑
        let batches = self.create_batches(data)?;
        let mut results = Vec::new();
        
        for batch in batches {
            let batch_result = self.processing_algorithm.process_batch(&batch).await?;
            results.push(batch_result);
        }
        
        Ok(self.merge_results(results))
    }
    
    fn get_strategy_name(&self) -> &str {
        "Batch"
    }
}

// 流处理策略
pub struct StreamProcessingStrategy {
    window_size: Duration,
    aggregation_function: Box<dyn AggregationFunction>,
}

impl DataProcessingStrategy for StreamProcessingStrategy {
    async fn process(&self, data: &RawData) -> Result<ProcessedData, ProcessingError> {
        // 流处理逻辑
        let windowed_data = self.create_time_window(data)?;
        let aggregated = self.aggregation_function.aggregate(&windowed_data).await?;
        
        Ok(ProcessedData::from_aggregated(aggregated))
    }
    
    fn get_strategy_name(&self) -> &str {
        "Stream"
    }
}

// 策略上下文
pub struct DataProcessor {
    strategy: Box<dyn DataProcessingStrategy>,
    metrics_collector: Arc<dyn MetricsCollector>,
}

impl DataProcessor {
    pub fn new(strategy: Box<dyn DataProcessingStrategy>) -> Self {
        Self {
            strategy,
            metrics_collector: Arc::new(DefaultMetricsCollector::new()),
        }
    }
    
    pub fn set_strategy(&mut self, strategy: Box<dyn DataProcessingStrategy>) {
        self.strategy = strategy;
    }
    
    pub async fn process_data(&self, data: &RawData) -> Result<ProcessedData, ProcessingError> {
        let start_time = Instant::now();
        
        let result = self.strategy.process(data).await?;
        
        let processing_time = start_time.elapsed();
        self.metrics_collector.record_processing_time(
            self.strategy.get_strategy_name(),
            processing_time,
        ).await;
        
        Ok(result)
    }
}
```

### 3.3 命令模式 (Command Pattern)

```rust
// 命令接口
pub trait Command {
    async fn execute(&self) -> Result<CommandResult, CommandError>;
    fn get_command_type(&self) -> CommandType;
    fn get_timestamp(&self) -> DateTime<Utc>;
}

// 具体命令实现
pub struct RegisterDeviceCommand {
    device_info: DeviceInfo,
    timestamp: DateTime<Utc>,
    device_registry: Arc<dyn DeviceRegistry>,
}

impl RegisterDeviceCommand {
    pub fn new(device_info: DeviceInfo, device_registry: Arc<dyn DeviceRegistry>) -> Self {
        Self {
            device_info,
            timestamp: Utc::now(),
            device_registry,
        }
    }
}

impl Command for RegisterDeviceCommand {
    async fn execute(&self) -> Result<CommandResult, CommandError> {
        let device_id = self.device_registry.register(self.device_info.clone()).await?;
        
        Ok(CommandResult::DeviceRegistered { device_id })
    }
    
    fn get_command_type(&self) -> CommandType {
        CommandType::RegisterDevice
    }
    
    fn get_timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
}

pub struct UpdateDeviceConfigCommand {
    device_id: DeviceId,
    new_config: DeviceConfiguration,
    timestamp: DateTime<Utc>,
    device_configurator: Arc<dyn DeviceConfigurator>,
}

impl UpdateDeviceConfigCommand {
    pub fn new(
        device_id: DeviceId,
        new_config: DeviceConfiguration,
        device_configurator: Arc<dyn DeviceConfigurator>,
    ) -> Self {
        Self {
            device_id,
            new_config,
            timestamp: Utc::now(),
            device_configurator,
        }
    }
}

impl Command for UpdateDeviceConfigCommand {
    async fn execute(&self) -> Result<CommandResult, CommandError> {
        self.device_configurator.update_config(&self.device_id, &self.new_config).await?;
        
        Ok(CommandResult::DeviceConfigUpdated {
            device_id: self.device_id.clone(),
        })
    }
    
    fn get_command_type(&self) -> CommandType {
        CommandType::UpdateDeviceConfig
    }
    
    fn get_timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
}

// 命令执行器
pub struct CommandExecutor {
    command_history: Vec<Box<dyn Command>>,
    max_history_size: usize,
}

impl CommandExecutor {
    pub fn new(max_history_size: usize) -> Self {
        Self {
            command_history: Vec::new(),
            max_history_size,
        }
    }
    
    pub async fn execute_command(&mut self, command: Box<dyn Command>) -> Result<CommandResult, CommandError> {
        let result = command.execute().await?;
        
        // 添加到历史记录
        self.command_history.push(command);
        
        // 限制历史记录大小
        if self.command_history.len() > self.max_history_size {
            self.command_history.remove(0);
        }
        
        Ok(result)
    }
    
    pub fn get_command_history(&self) -> &[Box<dyn Command>] {
        &self.command_history
    }
}

// 宏命令（组合命令）
pub struct MacroCommand {
    commands: Vec<Box<dyn Command>>,
    timestamp: DateTime<Utc>,
}

impl MacroCommand {
    pub fn new(commands: Vec<Box<dyn Command>>) -> Self {
        Self {
            commands,
            timestamp: Utc::now(),
        }
    }
}

impl Command for MacroCommand {
    async fn execute(&self) -> Result<CommandResult, CommandError> {
        let mut results = Vec::new();
        
        for command in &self.commands {
            let result = command.execute().await?;
            results.push(result);
        }
        
        Ok(CommandResult::MacroCommandCompleted { results })
    }
    
    fn get_command_type(&self) -> CommandType {
        CommandType::MacroCommand
    }
    
    fn get_timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
}
```

## 4. 总结

实现模式为IoT系统提供了具体的代码组织方式：

1. **创建型模式**：工厂、建造者、单例等，用于对象创建和管理
2. **结构型模式**：适配器、装饰器、外观等，用于对象组合和接口适配
3. **行为型模式**：观察者、策略、命令等，用于对象间通信和行为封装

选择合适的实现模式能够提高代码的可维护性、可扩展性和可测试性。
