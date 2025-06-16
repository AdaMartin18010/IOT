# IoT设计模式分析

## 1. IoT设计模式形式化模型

### 1.1 设计模式定义

**定义 1.1** (设计模式)
设计模式是一个五元组 $\mathcal{P} = (N, I, S, C, E)$，其中：

- $N$ 是模式名称
- $I$ 是意图描述
- $S$ 是结构定义
- $C$ 是协作关系
- $E$ 是效果评估

**定义 1.2** (模式分类)
设计模式分类定义为：
$$\mathcal{C} = \{C_{cre}, C_{str}, C_{beh}, C_{con}, C_{par}, C_{dis}\}$$

其中：
- $C_{cre}$: 创建型模式
- $C_{str}$: 结构型模式
- $C_{beh}$: 行为型模式
- $C_{con}$: 并发型模式
- $C_{par}$: 并行型模式
- $C_{dis}$: 分布式模式

**定理 1.1** (模式组合)
如果模式 $P_1$ 和 $P_2$ 兼容，则组合模式 $P_1 \circ P_2$ 也是有效的设计模式。

### 1.2 IoT特定模式

**定义 1.3** (IoT模式)
IoT特定模式是一个六元组 $\mathcal{I} = (D, S, C, A, R, T)$，其中：

- $D$ 是设备管理模式
- $S$ 是传感器模式
- $C$ 是通信模式
- $A$ 是聚合模式
- $R$ 是路由模式
- $T$ 是时间模式

## 2. 创建型模式

### 2.1 设备工厂模式

**定义 2.1** (设备工厂)
设备工厂模式定义为：
$$F: T \times C \rightarrow D$$

其中 $T$ 是设备类型，$C$ 是配置参数，$D$ 是设备实例。

```rust
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// 设备类型枚举
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceType {
    Sensor(SensorType),
    Actuator(ActuatorType),
    Gateway,
    EdgeNode,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SensorType {
    Temperature,
    Humidity,
    Pressure,
    Light,
    Motion,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ActuatorType {
    Relay,
    Motor,
    Valve,
    Light,
    Display,
}

/// 设备配置
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    pub device_id: String,
    pub device_type: DeviceType,
    pub parameters: HashMap<String, String>,
    pub location: Location,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Location {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: Option<f64>,
}

/// 设备特征
pub trait Device: Send + Sync {
    fn get_id(&self) -> &str;
    fn get_type(&self) -> &DeviceType;
    fn initialize(&mut self) -> Result<(), DeviceError>;
    fn shutdown(&mut self) -> Result<(), DeviceError>;
    fn get_status(&self) -> DeviceStatus;
}

#[derive(Debug, Clone)]
pub struct DeviceStatus {
    pub online: bool,
    pub battery_level: f64,
    pub last_seen: std::time::Instant,
    pub error_count: u32,
}

/// 温度传感器
pub struct TemperatureSensor {
    id: String,
    config: DeviceConfig,
    status: DeviceStatus,
    current_temperature: f64,
}

impl TemperatureSensor {
    pub fn new(config: DeviceConfig) -> Self {
        Self {
            id: config.device_id.clone(),
            config,
            status: DeviceStatus {
                online: false,
                battery_level: 100.0,
                last_seen: std::time::Instant::now(),
                error_count: 0,
            },
            current_temperature: 0.0,
        }
    }
    
    pub fn read_temperature(&mut self) -> Result<f64, DeviceError> {
        // 模拟温度读取
        self.current_temperature = 20.0 + (rand::random::<f64>() - 0.5) * 10.0;
        self.status.last_seen = std::time::Instant::now();
        Ok(self.current_temperature)
    }
}

impl Device for TemperatureSensor {
    fn get_id(&self) -> &str {
        &self.id
    }
    
    fn get_type(&self) -> &DeviceType {
        &self.config.device_type
    }
    
    fn initialize(&mut self) -> Result<(), DeviceError> {
        self.status.online = true;
        self.status.last_seen = std::time::Instant::now();
        Ok(())
    }
    
    fn shutdown(&mut self) -> Result<(), DeviceError> {
        self.status.online = false;
        Ok(())
    }
    
    fn get_status(&self) -> DeviceStatus {
        self.status.clone()
    }
}

/// 继电器执行器
pub struct RelayActuator {
    id: String,
    config: DeviceConfig,
    status: DeviceStatus,
    is_on: bool,
}

impl RelayActuator {
    pub fn new(config: DeviceConfig) -> Self {
        Self {
            id: config.device_id.clone(),
            config,
            status: DeviceStatus {
                online: false,
                battery_level: 100.0,
                last_seen: std::time::Instant::now(),
                error_count: 0,
            },
            is_on: false,
        }
    }
    
    pub fn turn_on(&mut self) -> Result<(), DeviceError> {
        self.is_on = true;
        self.status.last_seen = std::time::Instant::now();
        Ok(())
    }
    
    pub fn turn_off(&mut self) -> Result<(), DeviceError> {
        self.is_on = false;
        self.status.last_seen = std::time::Instant::now();
        Ok(())
    }
    
    pub fn get_state(&self) -> bool {
        self.is_on
    }
}

impl Device for RelayActuator {
    fn get_id(&self) -> &str {
        &self.id
    }
    
    fn get_type(&self) -> &DeviceType {
        &self.config.device_type
    }
    
    fn initialize(&mut self) -> Result<(), DeviceError> {
        self.status.online = true;
        self.status.last_seen = std::time::Instant::now();
        Ok(())
    }
    
    fn shutdown(&mut self) -> Result<(), DeviceError> {
        self.status.online = false;
        Ok(())
    }
    
    fn get_status(&self) -> DeviceStatus {
        self.status.clone()
    }
}

/// 设备工厂
pub struct DeviceFactory;

impl DeviceFactory {
    /// 创建设备实例
    pub fn create_device(config: DeviceConfig) -> Result<Box<dyn Device>, DeviceError> {
        match config.device_type {
            DeviceType::Sensor(SensorType::Temperature) => {
                Ok(Box::new(TemperatureSensor::new(config)))
            }
            DeviceType::Actuator(ActuatorType::Relay) => {
                Ok(Box::new(RelayActuator::new(config)))
            }
            _ => Err(DeviceError::UnsupportedDeviceType),
        }
    }
    
    /// 批量创建设备
    pub fn create_devices(configs: Vec<DeviceConfig>) -> Result<Vec<Box<dyn Device>>, DeviceError> {
        let mut devices = Vec::new();
        
        for config in configs {
            let device = Self::create_device(config)?;
            devices.push(device);
        }
        
        Ok(devices)
    }
}

#[derive(Debug)]
pub enum DeviceError {
    UnsupportedDeviceType,
    InitializationFailed,
    CommunicationError,
    ConfigurationError,
}
```

### 2.2 单例模式

```rust
use std::sync::{Arc, Mutex, Once};
use std::collections::HashMap;

/// IoT系统管理器单例
pub struct IoTSystemManager {
    devices: HashMap<String, Box<dyn Device>>,
    event_handlers: Vec<Box<dyn EventHandler>>,
    system_status: SystemStatus,
}

#[derive(Debug, Clone)]
pub struct SystemStatus {
    pub total_devices: usize,
    pub online_devices: usize,
    pub system_health: f64,
    pub last_update: std::time::Instant,
}

#[async_trait::async_trait]
pub trait EventHandler: Send + Sync {
    async fn handle_event(&self, event: &SystemEvent) -> Result<(), EventError>;
    fn event_type(&self) -> &str;
}

#[derive(Debug, Clone)]
pub enum SystemEvent {
    DeviceConnected(String),
    DeviceDisconnected(String),
    DataReceived(String, Vec<u8>),
    Alert(String, String),
}

#[derive(Debug)]
pub enum EventError {
    HandlerError,
    ProcessingError,
}

impl IoTSystemManager {
    fn new() -> Self {
        Self {
            devices: HashMap::new(),
            event_handlers: Vec::new(),
            system_status: SystemStatus {
                total_devices: 0,
                online_devices: 0,
                system_health: 100.0,
                last_update: std::time::Instant::now(),
            },
        }
    }
    
    /// 注册设备
    pub fn register_device(&mut self, device: Box<dyn Device>) -> Result<(), ManagerError> {
        let device_id = device.get_id().to_string();
        self.devices.insert(device_id.clone(), device);
        
        self.system_status.total_devices = self.devices.len();
        self.system_status.last_update = std::time::Instant::now();
        
        // 发布设备连接事件
        let event = SystemEvent::DeviceConnected(device_id);
        self.publish_event(event).await?;
        
        Ok(())
    }
    
    /// 获取设备
    pub fn get_device(&self, device_id: &str) -> Option<&Box<dyn Device>> {
        self.devices.get(device_id)
    }
    
    /// 注册事件处理器
    pub fn register_event_handler(&mut self, handler: Box<dyn EventHandler>) {
        self.event_handlers.push(handler);
    }
    
    /// 发布事件
    async fn publish_event(&self, event: SystemEvent) -> Result<(), ManagerError> {
        for handler in &self.event_handlers {
            if let Err(e) = handler.handle_event(&event).await {
                eprintln!("事件处理错误: {:?}", e);
            }
        }
        Ok(())
    }
    
    /// 获取系统状态
    pub fn get_system_status(&self) -> SystemStatus {
        let online_devices = self.devices.values()
            .filter(|device| device.get_status().online)
            .count();
        
        SystemStatus {
            total_devices: self.devices.len(),
            online_devices,
            system_health: self.calculate_system_health(),
            last_update: std::time::Instant::now(),
        }
    }
    
    fn calculate_system_health(&self) -> f64 {
        if self.devices.is_empty() {
            return 100.0;
        }
        
        let total_errors: u32 = self.devices.values()
            .map(|device| device.get_status().error_count)
            .sum();
        
        let health_score = 100.0 - (total_errors as f64 * 10.0).min(100.0);
        health_score.max(0.0)
    }
}

/// 单例实现
pub struct IoTSystemManagerSingleton {
    instance: Arc<Mutex<Option<IoTSystemManager>>>,
    init_once: Once,
}

impl IoTSystemManagerSingleton {
    pub fn new() -> Self {
        Self {
            instance: Arc::new(Mutex::new(None)),
            init_once: Once::new(),
        }
    }
    
    /// 获取单例实例
    pub fn get_instance(&self) -> Arc<Mutex<IoTSystemManager>> {
        self.init_once.call_once(|| {
            let mut instance_guard = self.instance.lock().unwrap();
            *instance_guard = Some(IoTSystemManager::new());
        });
        
        Arc::new(Mutex::new(self.instance.lock().unwrap().as_ref().unwrap().clone()))
    }
}

#[derive(Debug)]
pub enum ManagerError {
    DeviceNotFound,
    EventPublishError,
    SystemError,
}
```

## 3. 结构型模式

### 3.1 适配器模式

```rust
use std::collections::HashMap;

/// 旧版设备接口
pub trait LegacyDevice {
    fn read_data(&self) -> String;
    fn write_data(&self, data: &str) -> bool;
    fn get_device_info(&self) -> HashMap<String, String>;
}

/// 新版设备接口
pub trait ModernDevice {
    async fn read_data(&self) -> Result<Vec<u8>, DeviceError>;
    async fn write_data(&self, data: &[u8]) -> Result<(), DeviceError>;
    async fn get_device_info(&self) -> Result<DeviceInfo, DeviceError>;
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub device_id: String,
    pub device_type: String,
    pub capabilities: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// 旧版温度传感器
pub struct LegacyTemperatureSensor {
    device_id: String,
    current_temperature: f64,
}

impl LegacyTemperatureSensor {
    pub fn new(device_id: String) -> Self {
        Self {
            device_id,
            current_temperature: 20.0,
        }
    }
}

impl LegacyDevice for LegacyTemperatureSensor {
    fn read_data(&self) -> String {
        format!("{{\"temperature\": {:.2}}}", self.current_temperature)
    }
    
    fn write_data(&self, _data: &str) -> bool {
        // 旧版设备不支持写入
        false
    }
    
    fn get_device_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("device_id".to_string(), self.device_id.clone());
        info.insert("device_type".to_string(), "legacy_temperature_sensor".to_string());
        info.insert("protocol".to_string(), "legacy".to_string());
        info
    }
}

/// 适配器：将旧版设备适配到新版接口
pub struct LegacyDeviceAdapter {
    legacy_device: Box<dyn LegacyDevice>,
}

impl LegacyDeviceAdapter {
    pub fn new(legacy_device: Box<dyn LegacyDevice>) -> Self {
        Self { legacy_device }
    }
}

#[async_trait::async_trait]
impl ModernDevice for LegacyDeviceAdapter {
    async fn read_data(&self) -> Result<Vec<u8>, DeviceError> {
        let data = self.legacy_device.read_data();
        Ok(data.into_bytes())
    }
    
    async fn write_data(&self, data: &[u8]) -> Result<(), DeviceError> {
        let data_str = String::from_utf8_lossy(data);
        if self.legacy_device.write_data(&data_str) {
            Ok(())
        } else {
            Err(DeviceError::CommunicationError)
        }
    }
    
    async fn get_device_info(&self) -> Result<DeviceInfo, DeviceError> {
        let legacy_info = self.legacy_device.get_device_info();
        
        let device_info = DeviceInfo {
            device_id: legacy_info.get("device_id").unwrap_or(&"unknown".to_string()).clone(),
            device_type: legacy_info.get("device_type").unwrap_or(&"unknown".to_string()).clone(),
            capabilities: vec!["read".to_string()],
            metadata: legacy_info,
        };
        
        Ok(device_info)
    }
}
```

### 3.2 装饰器模式

```rust
use std::time::{Duration, Instant};

/// 基础设备接口
pub trait BaseDevice {
    fn read(&self) -> Result<Vec<u8>, DeviceError>;
    fn write(&self, data: &[u8]) -> Result<(), DeviceError>;
    fn get_id(&self) -> &str;
}

/// 基础传感器
pub struct BaseSensor {
    id: String,
    data: Vec<u8>,
}

impl BaseSensor {
    pub fn new(id: String) -> Self {
        Self {
            id,
            data: vec![0; 10],
        }
    }
}

impl BaseDevice for BaseSensor {
    fn read(&self) -> Result<Vec<u8>, DeviceError> {
        Ok(self.data.clone())
    }
    
    fn write(&self, data: &[u8]) -> Result<(), DeviceError> {
        // 基础传感器不支持写入
        Err(DeviceError::CommunicationError)
    }
    
    fn get_id(&self) -> &str {
        &self.id
    }
}

/// 缓存装饰器
pub struct CachedDevice {
    device: Box<dyn BaseDevice>,
    cache: HashMap<String, (Vec<u8>, Instant)>,
    cache_duration: Duration,
}

impl CachedDevice {
    pub fn new(device: Box<dyn BaseDevice>, cache_duration: Duration) -> Self {
        Self {
            device,
            cache: HashMap::new(),
            cache_duration,
        }
    }
    
    fn is_cache_valid(&self, key: &str) -> bool {
        if let Some((_, timestamp)) = self.cache.get(key) {
            timestamp.elapsed() < self.cache_duration
        } else {
            false
        }
    }
}

impl BaseDevice for CachedDevice {
    fn read(&self) -> Result<Vec<u8>, DeviceError> {
        let cache_key = format!("read_{}", self.device.get_id());
        
        if self.is_cache_valid(&cache_key) {
            if let Some((data, _)) = self.cache.get(&cache_key) {
                return Ok(data.clone());
            }
        }
        
        // 从设备读取数据
        let data = self.device.read()?;
        
        // 更新缓存
        let mut cache = self.cache.clone();
        cache.insert(cache_key, (data.clone(), Instant::now()));
        
        Ok(data)
    }
    
    fn write(&self, data: &[u8]) -> Result<(), DeviceError> {
        // 写入时清除相关缓存
        let cache_key = format!("read_{}", self.device.get_id());
        let mut cache = self.cache.clone();
        cache.remove(&cache_key);
        
        self.device.write(data)
    }
    
    fn get_id(&self) -> &str {
        self.device.get_id()
    }
}

/// 重试装饰器
pub struct RetryDevice {
    device: Box<dyn BaseDevice>,
    max_retries: u32,
    retry_delay: Duration,
}

impl RetryDevice {
    pub fn new(device: Box<dyn BaseDevice>, max_retries: u32, retry_delay: Duration) -> Self {
        Self {
            device,
            max_retries,
            retry_delay,
        }
    }
}

impl BaseDevice for RetryDevice {
    fn read(&self) -> Result<Vec<u8>, DeviceError> {
        let mut last_error = None;
        
        for attempt in 0..=self.max_retries {
            match self.device.read() {
                Ok(data) => return Ok(data),
                Err(e) => {
                    last_error = Some(e);
                    if attempt < self.max_retries {
                        std::thread::sleep(self.retry_delay);
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or(DeviceError::CommunicationError))
    }
    
    fn write(&self, data: &[u8]) -> Result<(), DeviceError> {
        let mut last_error = None;
        
        for attempt in 0..=self.max_retries {
            match self.device.write(data) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    last_error = Some(e);
                    if attempt < self.max_retries {
                        std::thread::sleep(self.retry_delay);
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or(DeviceError::CommunicationError))
    }
    
    fn get_id(&self) -> &str {
        self.device.get_id()
    }
}

/// 日志装饰器
pub struct LoggedDevice {
    device: Box<dyn BaseDevice>,
    logger: Box<dyn DeviceLogger>,
}

#[async_trait::async_trait]
pub trait DeviceLogger: Send + Sync {
    async fn log(&self, level: LogLevel, message: &str);
}

#[derive(Debug, Clone)]
pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
}

impl LoggedDevice {
    pub fn new(device: Box<dyn BaseDevice>, logger: Box<dyn DeviceLogger>) -> Self {
        Self { device, logger }
    }
}

impl BaseDevice for LoggedDevice {
    fn read(&self) -> Result<Vec<u8>, DeviceError> {
        let device_id = self.device.get_id();
        
        // 记录读取开始
        let logger = self.logger.clone();
        tokio::spawn(async move {
            logger.log(LogLevel::Debug, &format!("开始读取设备 {}", device_id)).await;
        });
        
        let result = self.device.read();
        
        // 记录读取结果
        let logger = self.logger.clone();
        let device_id = device_id.to_string();
        tokio::spawn(async move {
            match &result {
                Ok(data) => {
                    logger.log(LogLevel::Info, &format!("设备 {} 读取成功，数据长度: {}", device_id, data.len())).await;
                }
                Err(e) => {
                    logger.log(LogLevel::Error, &format!("设备 {} 读取失败: {:?}", device_id, e)).await;
                }
            }
        });
        
        result
    }
    
    fn write(&self, data: &[u8]) -> Result<(), DeviceError> {
        let device_id = self.device.get_id();
        
        // 记录写入开始
        let logger = self.logger.clone();
        let device_id_clone = device_id.to_string();
        tokio::spawn(async move {
            logger.log(LogLevel::Debug, &format!("开始写入设备 {}", device_id_clone)).await;
        });
        
        let result = self.device.write(data);
        
        // 记录写入结果
        let logger = self.logger.clone();
        let device_id = device_id.to_string();
        tokio::spawn(async move {
            match &result {
                Ok(()) => {
                    logger.log(LogLevel::Info, &format!("设备 {} 写入成功", device_id)).await;
                }
                Err(e) => {
                    logger.log(LogLevel::Error, &format!("设备 {} 写入失败: {:?}", device_id, e)).await;
                }
            }
        });
        
        result
    }
    
    fn get_id(&self) -> &str {
        self.device.get_id()
    }
}
```

## 4. 行为型模式

### 4.1 观察者模式

```rust
use std::collections::HashMap;
use tokio::sync::mpsc;
use serde::{Deserialize, Serialize};

/// 事件类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IoTEvent {
    DeviceConnected { device_id: String, timestamp: u64 },
    DeviceDisconnected { device_id: String, timestamp: u64 },
    DataReceived { device_id: String, data: Vec<u8>, timestamp: u64 },
    Alert { device_id: String, message: String, severity: AlertSeverity, timestamp: u64 },
    SystemStatus { status: SystemStatus, timestamp: u64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// 事件观察者
#[async_trait::async_trait]
pub trait EventObserver: Send + Sync {
    async fn on_event(&self, event: &IoTEvent) -> Result<(), ObserverError>;
    fn observer_id(&self) -> &str;
    fn event_types(&self) -> Vec<String>;
}

/// 事件总线
pub struct EventBus {
    observers: HashMap<String, Box<dyn EventObserver>>,
    event_sender: mpsc::Sender<IoTEvent>,
    event_receiver: mpsc::Receiver<IoTEvent>,
}

impl EventBus {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel(1000);
        Self {
            observers: HashMap::new(),
            event_sender: tx,
            event_receiver: rx,
        }
    }
    
    /// 注册观察者
    pub fn register_observer(&mut self, observer: Box<dyn EventObserver>) {
        let observer_id = observer.observer_id().to_string();
        self.observers.insert(observer_id, observer);
    }
    
    /// 取消注册观察者
    pub fn unregister_observer(&mut self, observer_id: &str) {
        self.observers.remove(observer_id);
    }
    
    /// 发布事件
    pub async fn publish_event(&self, event: IoTEvent) -> Result<(), EventBusError> {
        self.event_sender.send(event).await
            .map_err(|_| EventBusError::SendError)
    }
    
    /// 运行事件总线
    pub async fn run(&mut self) -> Result<(), EventBusError> {
        while let Some(event) = self.event_receiver.recv().await {
            self.notify_observers(&event).await?;
        }
        Ok(())
    }
    
    /// 通知观察者
    async fn notify_observers(&self, event: &IoTEvent) -> Result<(), EventBusError> {
        let mut tasks = Vec::new();
        
        for observer in self.observers.values() {
            let event_types = observer.event_types();
            let event_type = self.get_event_type(event);
            
            if event_types.contains(&event_type) {
                let observer = observer.clone();
                let event = event.clone();
                
                let task = tokio::spawn(async move {
                    if let Err(e) = observer.on_event(&event).await {
                        eprintln!("观察者 {} 处理事件失败: {:?}", observer.observer_id(), e);
                    }
                });
                
                tasks.push(task);
            }
        }
        
        // 等待所有观察者处理完成
        for task in tasks {
            if let Err(e) = task.await {
                eprintln!("观察者任务失败: {:?}", e);
            }
        }
        
        Ok(())
    }
    
    fn get_event_type(&self, event: &IoTEvent) -> String {
        match event {
            IoTEvent::DeviceConnected { .. } => "device_connected".to_string(),
            IoTEvent::DeviceDisconnected { .. } => "device_disconnected".to_string(),
            IoTEvent::DataReceived { .. } => "data_received".to_string(),
            IoTEvent::Alert { .. } => "alert".to_string(),
            IoTEvent::SystemStatus { .. } => "system_status".to_string(),
        }
    }
}

/// 日志观察者
pub struct LoggingObserver {
    observer_id: String,
    log_file: String,
}

impl LoggingObserver {
    pub fn new(observer_id: String, log_file: String) -> Self {
        Self {
            observer_id,
            log_file,
        }
    }
}

#[async_trait::async_trait]
impl EventObserver for LoggingObserver {
    async fn on_event(&self, event: &IoTEvent) -> Result<(), ObserverError> {
        let log_entry = format!("[{}] {:?}\n", 
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S"), 
            event
        );
        
        // 这里应该写入日志文件
        println!("日志观察者 {}: {}", self.observer_id, log_entry.trim());
        
        Ok(())
    }
    
    fn observer_id(&self) -> &str {
        &self.observer_id
    }
    
    fn event_types(&self) -> Vec<String> {
        vec![
            "device_connected".to_string(),
            "device_disconnected".to_string(),
            "data_received".to_string(),
            "alert".to_string(),
            "system_status".to_string(),
        ]
    }
}

/// 告警观察者
pub struct AlertingObserver {
    observer_id: String,
    alert_rules: Vec<AlertRule>,
}

#[derive(Debug, Clone)]
pub struct AlertRule {
    pub rule_id: String,
    pub condition: AlertCondition,
    pub action: AlertAction,
}

#[derive(Debug, Clone)]
pub enum AlertCondition {
    DeviceOffline { device_id: String, duration: u64 },
    DataThreshold { device_id: String, field: String, operator: String, value: f64 },
    ErrorRate { device_id: String, threshold: f64 },
}

#[derive(Debug, Clone)]
pub enum AlertAction {
    SendEmail { recipients: Vec<String>, template: String },
    SendSMS { phone_numbers: Vec<String>, message: String },
    Webhook { url: String, payload: HashMap<String, String> },
}

impl AlertingObserver {
    pub fn new(observer_id: String) -> Self {
        Self {
            observer_id,
            alert_rules: Vec::new(),
        }
    }
    
    pub fn add_rule(&mut self, rule: AlertRule) {
        self.alert_rules.push(rule);
    }
}

#[async_trait::async_trait]
impl EventObserver for AlertingObserver {
    async fn on_event(&self, event: &IoTEvent) -> Result<(), ObserverError> {
        for rule in &self.alert_rules {
            if self.evaluate_condition(&rule.condition, event).await? {
                self.execute_action(&rule.action, event).await?;
            }
        }
        Ok(())
    }
    
    fn observer_id(&self) -> &str {
        &self.observer_id
    }
    
    fn event_types(&self) -> Vec<String> {
        vec![
            "device_disconnected".to_string(),
            "data_received".to_string(),
            "alert".to_string(),
        ]
    }
    
    async fn evaluate_condition(&self, condition: &AlertCondition, event: &IoTEvent) -> Result<bool, ObserverError> {
        match condition {
            AlertCondition::DeviceOffline { device_id, duration } => {
                // 检查设备离线时间
                if let IoTEvent::DeviceDisconnected { device_id: event_device_id, timestamp } = event {
                    if device_id == event_device_id {
                        let current_time = chrono::Utc::now().timestamp() as u64;
                        return Ok(current_time - timestamp > *duration);
                    }
                }
                Ok(false)
            }
            AlertCondition::DataThreshold { device_id, field, operator, value } => {
                // 检查数据阈值
                if let IoTEvent::DataReceived { device_id: event_device_id, data, .. } = event {
                    if device_id == event_device_id {
                        // 解析数据并检查阈值
                        // 这里简化处理
                        return Ok(data.len() as f64 > *value);
                    }
                }
                Ok(false)
            }
            AlertCondition::ErrorRate { device_id, threshold } => {
                // 检查错误率
                // 这里需要维护错误统计
                Ok(false)
            }
        }
    }
    
    async fn execute_action(&self, action: &AlertAction, event: &IoTEvent) -> Result<(), ObserverError> {
        match action {
            AlertAction::SendEmail { recipients, template } => {
                println!("发送邮件到 {:?}: {}", recipients, template);
            }
            AlertAction::SendSMS { phone_numbers, message } => {
                println!("发送短信到 {:?}: {}", phone_numbers, message);
            }
            AlertAction::Webhook { url, payload } => {
                println!("调用Webhook {}: {:?}", url, payload);
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum ObserverError {
    ProcessingError,
    CommunicationError,
    ConfigurationError,
}

#[derive(Debug)]
pub enum EventBusError {
    SendError,
    ReceiveError,
    ObserverError,
}
```

## 5. 总结

本文档提供了IoT设计模式的完整分析，包括：

1. **形式化模型**：设计模式的数学定义和分类
2. **创建型模式**：设备工厂和单例模式
3. **结构型模式**：适配器和装饰器模式
4. **行为型模式**：观察者模式
5. **Rust实现**：完整的代码示例

IoT设计模式提供了：
- 代码复用和可维护性
- 系统灵活性和扩展性
- 错误处理和容错能力
- 性能优化和资源管理

这些模式为IoT系统的设计和实现提供了最佳实践指导。 