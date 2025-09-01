# 架构模式设计文档

## 概述

本文档详细阐述IoT系统中常用的架构模式，包括分层架构、微服务架构、事件驱动架构、CQRS模式等，为构建可扩展、可维护的IoT系统提供架构指导。

## 1. 分层架构模式 (Layered Architecture)

### 1.1 模式概述

分层架构将系统划分为多个水平层，每层都有明确的职责，层与层之间通过接口通信。

```rust
// IoT系统分层架构示例
pub mod presentation {
    // 表示层 - 用户界面和API
    pub struct IoTController {
        service: Arc<dyn IoTService>,
    }
    
    impl IoTController {
        pub async fn handle_device_data(&self, request: DeviceDataRequest) -> Result<DeviceDataResponse, ControllerError> {
            let command = self.map_to_command(request);
            let result = self.service.process_device_data(command).await?;
            Ok(self.map_to_response(result))
        }
    }
}

pub mod application {
    // 应用层 - 业务逻辑协调
    pub struct IoTService {
        device_repository: Arc<dyn DeviceRepository>,
        data_processor: Arc<dyn DataProcessor>,
        notification_service: Arc<dyn NotificationService>,
    }
    
    impl IoTService {
        pub async fn process_device_data(&self, command: ProcessDeviceDataCommand) -> Result<ProcessedData, ServiceError> {
            // 1. 验证设备
            let device = self.device_repository.find_by_id(&command.device_id).await?;
            self.validate_device(&device)?;
            
            // 2. 处理数据
            let processed_data = self.data_processor.process(&command.data).await?;
            
            // 3. 保存结果
            self.device_repository.save_data(&processed_data).await?;
            
            // 4. 发送通知
            if processed_data.requires_notification() {
                self.notification_service.send_alert(&processed_data).await?;
            }
            
            Ok(processed_data)
        }
    }
}

pub mod domain {
    // 领域层 - 核心业务逻辑
    pub struct Device {
        pub id: DeviceId,
        pub name: String,
        pub device_type: DeviceType,
        pub status: DeviceStatus,
        pub configuration: DeviceConfiguration,
    }
    
    impl Device {
        pub fn is_online(&self) -> bool {
            matches!(self.status, DeviceStatus::Online)
        }
        
        pub fn can_process_data(&self, data_type: &DataType) -> bool {
            self.device_type.supports_data_type(data_type)
        }
    }
    
    pub struct DataProcessor {
        pub algorithms: Vec<Box<dyn ProcessingAlgorithm>>,
    }
    
    impl DataProcessor {
        pub fn process(&self, data: &RawData) -> Result<ProcessedData, ProcessingError> {
            let mut processed = ProcessedData::from(data);
            
            for algorithm in &self.algorithms {
                processed = algorithm.apply(processed)?;
            }
            
            Ok(processed)
        }
    }
}

pub mod infrastructure {
    // 基础设施层 - 数据访问和外部服务
    pub struct DatabaseDeviceRepository {
        connection_pool: Arc<ConnectionPool>,
    }
    
    impl DeviceRepository for DatabaseDeviceRepository {
        async fn find_by_id(&self, id: &DeviceId) -> Result<Device, RepositoryError> {
            let connection = self.connection_pool.get().await?;
            let device_data = connection.query_device_by_id(id).await?;
            Ok(Device::from(device_data))
        }
        
        async fn save_data(&self, data: &ProcessedData) -> Result<(), RepositoryError> {
            let connection = self.connection_pool.get().await?;
            connection.insert_processed_data(data).await?;
            Ok(())
        }
    }
}
```

### 1.2 分层架构优势

```rust
// 依赖注入确保层间解耦
pub struct LayeredArchitectureBuilder {
    presentation_layer: Option<Box<dyn PresentationLayer>>,
    application_layer: Option<Box<dyn ApplicationLayer>>,
    domain_layer: Option<Box<dyn DomainLayer>>,
    infrastructure_layer: Option<Box<dyn InfrastructureLayer>>,
}

impl LayeredArchitectureBuilder {
    pub fn build(self) -> Result<LayeredArchitecture, ArchitectureError> {
        Ok(LayeredArchitecture {
            presentation: self.presentation_layer.ok_or(ArchitectureError::MissingLayer("presentation"))?,
            application: self.application_layer.ok_or(ArchitectureError::MissingLayer("application"))?,
            domain: self.domain_layer.ok_or(ArchitectureError::MissingLayer("domain"))?,
            infrastructure: self.infrastructure_layer.ok_or(ArchitectureError::MissingLayer("infrastructure"))?,
        })
    }
}
```

## 2. 微服务架构模式 (Microservices Architecture)

### 2.1 服务分解策略

```rust
// IoT微服务分解示例
pub mod device_management_service {
    pub struct DeviceManagementService {
        device_repository: Arc<dyn DeviceRepository>,
        device_registry: Arc<dyn DeviceRegistry>,
        event_publisher: Arc<dyn EventPublisher>,
    }
    
    impl DeviceManagementService {
        pub async fn register_device(&self, registration: DeviceRegistration) -> Result<DeviceId, ServiceError> {
            // 1. 验证设备信息
            self.validate_device_registration(&registration)?;
            
            // 2. 创建设备记录
            let device = Device::from_registration(registration);
            let device_id = self.device_repository.save(device).await?;
            
            // 3. 注册到设备注册表
            self.device_registry.register(device_id.clone()).await?;
            
            // 4. 发布设备注册事件
            self.event_publisher.publish(DeviceRegisteredEvent {
                device_id: device_id.clone(),
                timestamp: Utc::now(),
            }).await?;
            
            Ok(device_id)
        }
    }
}

pub mod data_processing_service {
    pub struct DataProcessingService {
        data_ingestion: Arc<dyn DataIngestion>,
        processing_pipeline: Arc<dyn ProcessingPipeline>,
        result_storage: Arc<dyn ResultStorage>,
    }
    
    impl DataProcessingService {
        pub async fn process_sensor_data(&self, data: SensorData) -> Result<ProcessingResult, ServiceError> {
            // 1. 数据摄取
            let ingested_data = self.data_ingestion.ingest(data).await?;
            
            // 2. 处理管道
            let processed_result = self.processing_pipeline.process(ingested_data).await?;
            
            // 3. 存储结果
            self.result_storage.store(processed_result.clone()).await?;
            
            Ok(processed_result)
        }
    }
}

pub mod notification_service {
    pub struct NotificationService {
        notification_channels: HashMap<NotificationType, Box<dyn NotificationChannel>>,
        user_preferences: Arc<dyn UserPreferencesRepository>,
    }
    
    impl NotificationService {
        pub async fn send_notification(&self, notification: Notification) -> Result<(), ServiceError> {
            let user_preferences = self.user_preferences.get(&notification.user_id).await?;
            
            for channel_type in user_preferences.enabled_channels() {
                if let Some(channel) = self.notification_channels.get(&channel_type) {
                    channel.send(&notification).await?;
                }
            }
            
            Ok(())
        }
    }
}
```

### 2.2 服务间通信

```rust
// 服务间通信抽象
pub trait ServiceCommunication {
    async fn send_request<T: Serialize, R: DeserializeOwned>(
        &self,
        service: &str,
        endpoint: &str,
        request: T,
    ) -> Result<R, CommunicationError>;
    
    async fn publish_event<T: Serialize>(&self, event: T) -> Result<(), CommunicationError>;
}

// HTTP通信实现
pub struct HttpServiceCommunication {
    client: reqwest::Client,
    service_registry: Arc<dyn ServiceRegistry>,
}

impl ServiceCommunication for HttpServiceCommunication {
    async fn send_request<T: Serialize, R: DeserializeOwned>(
        &self,
        service: &str,
        endpoint: &str,
        request: T,
    ) -> Result<R, CommunicationError> {
        let service_url = self.service_registry.get_service_url(service).await?;
        let url = format!("{}/{}", service_url, endpoint);
        
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?;
        
        if response.status().is_success() {
            let result: R = response.json().await?;
            Ok(result)
        } else {
            Err(CommunicationError::ServiceError(response.status()))
        }
    }
}

// 消息队列通信实现
pub struct MessageQueueCommunication {
    publisher: Arc<dyn MessagePublisher>,
    subscriber: Arc<dyn MessageSubscriber>,
}

impl ServiceCommunication for MessageQueueCommunication {
    async fn publish_event<T: Serialize>(&self, event: T) -> Result<(), CommunicationError> {
        let message = Message::new(event);
        self.publisher.publish(message).await?;
        Ok(())
    }
}
```

## 3. 事件驱动架构模式 (Event-Driven Architecture)

### 3.1 事件模型设计

```rust
// 事件基础结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainEvent {
    pub event_id: EventId,
    pub event_type: String,
    pub aggregate_id: String,
    pub aggregate_type: String,
    pub event_data: serde_json::Value,
    pub metadata: EventMetadata,
    pub timestamp: DateTime<Utc>,
    pub version: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMetadata {
    pub correlation_id: Option<String>,
    pub causation_id: Option<String>,
    pub user_id: Option<String>,
    pub source: String,
    pub tags: HashMap<String, String>,
}

// IoT领域事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IoTEvent {
    DeviceRegistered {
        device_id: DeviceId,
        device_info: DeviceInfo,
        registration_time: DateTime<Utc>,
    },
    DeviceDataReceived {
        device_id: DeviceId,
        data: SensorData,
        received_time: DateTime<Utc>,
    },
    DataProcessingCompleted {
        device_id: DeviceId,
        processing_result: ProcessingResult,
        processing_time: DateTime<Utc>,
    },
    AlertTriggered {
        device_id: DeviceId,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: String,
        triggered_time: DateTime<Utc>,
    },
}
```

### 3.2 事件发布与订阅

```rust
// 事件发布器
pub trait EventPublisher {
    async fn publish<T: Serialize>(&self, event: T) -> Result<(), PublishError>;
    async fn publish_batch<T: Serialize>(&self, events: Vec<T>) -> Result<(), PublishError>;
}

pub struct EventStorePublisher {
    event_store: Arc<dyn EventStore>,
    event_router: Arc<dyn EventRouter>,
}

impl EventPublisher for EventStorePublisher {
    async fn publish<T: Serialize>(&self, event: T) -> Result<(), PublishError> {
        // 1. 存储事件
        let domain_event = self.create_domain_event(event).await?;
        self.event_store.append_event(&domain_event).await?;
        
        // 2. 路由事件
        self.event_router.route_event(&domain_event).await?;
        
        Ok(())
    }
}

// 事件订阅器
pub trait EventSubscriber {
    async fn subscribe<T: DeserializeOwned + Send + 'static>(
        &self,
        event_type: &str,
        handler: Box<dyn EventHandler<T>>,
    ) -> Result<(), SubscriptionError>;
}

pub struct EventBusSubscriber {
    event_bus: Arc<dyn EventBus>,
    subscription_manager: Arc<dyn SubscriptionManager>,
}

impl EventSubscriber for EventBusSubscriber {
    async fn subscribe<T: DeserializeOwned + Send + 'static>(
        &self,
        event_type: &str,
        handler: Box<dyn EventHandler<T>>,
    ) -> Result<(), SubscriptionError> {
        let subscription = EventSubscription {
            event_type: event_type.to_string(),
            handler: Box::new(handler),
        };
        
        self.subscription_manager.register(subscription).await?;
        self.event_bus.subscribe(event_type, handler).await?;
        
        Ok(())
    }
}
```

### 3.3 事件溯源 (Event Sourcing)

```rust
// 事件存储
pub trait EventStore {
    async fn append_event(&self, event: &DomainEvent) -> Result<(), EventStoreError>;
    async fn get_events(&self, aggregate_id: &str) -> Result<Vec<DomainEvent>, EventStoreError>;
    async fn get_events_since(&self, aggregate_id: &str, version: u64) -> Result<Vec<DomainEvent>, EventStoreError>;
}

// 聚合根
pub trait AggregateRoot {
    fn aggregate_id(&self) -> &str;
    fn version(&self) -> u64;
    fn uncommitted_events(&self) -> Vec<DomainEvent>;
    fn mark_events_as_committed(&mut self);
    fn apply_event(&mut self, event: &DomainEvent);
}

// IoT设备聚合
pub struct DeviceAggregate {
    id: DeviceId,
    name: String,
    status: DeviceStatus,
    configuration: DeviceConfiguration,
    version: u64,
    uncommitted_events: Vec<DomainEvent>,
}

impl AggregateRoot for DeviceAggregate {
    fn aggregate_id(&self) -> &str {
        &self.id.0
    }
    
    fn version(&self) -> u64 {
        self.version
    }
    
    fn uncommitted_events(&self) -> Vec<DomainEvent> {
        self.uncommitted_events.clone()
    }
    
    fn mark_events_as_committed(&mut self) {
        self.uncommitted_events.clear();
    }
    
    fn apply_event(&mut self, event: &DomainEvent) {
        match event.event_type.as_str() {
            "DeviceRegistered" => {
                // 应用设备注册事件
                self.version = event.version;
            }
            "DeviceStatusChanged" => {
                // 应用设备状态变更事件
                self.version = event.version;
            }
            _ => {}
        }
    }
}

impl DeviceAggregate {
    pub fn register_device(&mut self, device_info: DeviceInfo) -> Result<(), DomainError> {
        if self.status != DeviceStatus::Unknown {
            return Err(DomainError::DeviceAlreadyRegistered);
        }
        
        let event = DomainEvent {
            event_id: EventId::generate(),
            event_type: "DeviceRegistered".to_string(),
            aggregate_id: self.id.0.clone(),
            aggregate_type: "Device".to_string(),
            event_data: serde_json::to_value(device_info)?,
            metadata: EventMetadata::default(),
            timestamp: Utc::now(),
            version: self.version + 1,
        };
        
        self.apply_event(&event);
        self.uncommitted_events.push(event);
        
        Ok(())
    }
}
```

## 4. CQRS模式 (Command Query Responsibility Segregation)

### 4.1 命令查询分离

```rust
// 命令模型
pub trait Command {
    fn command_type(&self) -> &str;
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RegisterDeviceCommand {
    pub device_id: DeviceId,
    pub device_info: DeviceInfo,
    pub user_id: UserId,
}

impl Command for RegisterDeviceCommand {
    fn command_type(&self) -> &str {
        "RegisterDevice"
    }
}

// 查询模型
pub trait Query {
    fn query_type(&self) -> &str;
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetDeviceQuery {
    pub device_id: DeviceId,
}

impl Query for GetDeviceQuery {
    fn query_type(&self) -> &str {
        "GetDevice"
    }
}

// 命令处理器
pub trait CommandHandler<C: Command> {
    async fn handle(&self, command: C) -> Result<CommandResult, CommandError>;
}

pub struct RegisterDeviceCommandHandler {
    device_repository: Arc<dyn DeviceRepository>,
    event_publisher: Arc<dyn EventPublisher>,
}

impl CommandHandler<RegisterDeviceCommand> for RegisterDeviceCommandHandler {
    async fn handle(&self, command: RegisterDeviceCommand) -> Result<CommandResult, CommandError> {
        // 1. 验证命令
        self.validate_command(&command)?;
        
        // 2. 执行业务逻辑
        let device = Device::from_command(&command);
        self.device_repository.save(device).await?;
        
        // 3. 发布事件
        let event = DeviceRegisteredEvent {
            device_id: command.device_id,
            device_info: command.device_info,
            registration_time: Utc::now(),
        };
        self.event_publisher.publish(event).await?;
        
        Ok(CommandResult::Success)
    }
}

// 查询处理器
pub trait QueryHandler<Q: Query, R> {
    async fn handle(&self, query: Q) -> Result<R, QueryError>;
}

pub struct GetDeviceQueryHandler {
    device_read_model: Arc<dyn DeviceReadModel>,
}

impl QueryHandler<GetDeviceQuery, DeviceReadModel> for GetDeviceQueryHandler {
    async fn handle(&self, query: GetDeviceQuery) -> Result<DeviceReadModel, QueryError> {
        self.device_read_model.get_by_id(&query.device_id).await
    }
}
```

### 4.2 读写模型分离

```rust
// 写模型（命令端）
pub struct DeviceWriteModel {
    id: DeviceId,
    name: String,
    status: DeviceStatus,
    configuration: DeviceConfiguration,
    version: u64,
}

// 读模型（查询端）
pub struct DeviceReadModel {
    pub id: DeviceId,
    pub name: String,
    pub status: DeviceStatus,
    pub last_seen: DateTime<Utc>,
    pub data_count: u64,
    pub alert_count: u64,
    pub performance_metrics: PerformanceMetrics,
}

// 读模型存储
pub trait DeviceReadModelRepository {
    async fn get_by_id(&self, id: &DeviceId) -> Result<DeviceReadModel, RepositoryError>;
    async fn get_by_status(&self, status: DeviceStatus) -> Result<Vec<DeviceReadModel>, RepositoryError>;
    async fn get_performance_metrics(&self, id: &DeviceId) -> Result<PerformanceMetrics, RepositoryError>;
}

// 投影处理器（从事件更新读模型）
pub struct DeviceProjectionHandler {
    read_model_repository: Arc<dyn DeviceReadModelRepository>,
}

impl EventHandler<DeviceRegisteredEvent> for DeviceProjectionHandler {
    async fn handle(&self, event: &DeviceRegisteredEvent) -> Result<(), HandlerError> {
        let read_model = DeviceReadModel {
            id: event.device_id.clone(),
            name: event.device_info.name.clone(),
            status: DeviceStatus::Online,
            last_seen: event.registration_time,
            data_count: 0,
            alert_count: 0,
            performance_metrics: PerformanceMetrics::default(),
        };
        
        self.read_model_repository.save(read_model).await?;
        Ok(())
    }
}
```

## 5. 六边形架构模式 (Hexagonal Architecture)

### 5.1 端口适配器模式

```rust
// 端口定义（接口）
pub trait DeviceRepository {
    async fn save(&self, device: Device) -> Result<DeviceId, RepositoryError>;
    async fn find_by_id(&self, id: &DeviceId) -> Result<Option<Device>, RepositoryError>;
    async fn find_by_status(&self, status: DeviceStatus) -> Result<Vec<Device>, RepositoryError>;
}

pub trait NotificationService {
    async fn send_notification(&self, notification: Notification) -> Result<(), NotificationError>;
}

// 适配器实现
pub struct DatabaseDeviceRepository {
    connection_pool: Arc<ConnectionPool>,
}

impl DeviceRepository for DatabaseDeviceRepository {
    async fn save(&self, device: Device) -> Result<DeviceId, RepositoryError> {
        let connection = self.connection_pool.get().await?;
        let device_id = connection.insert_device(&device).await?;
        Ok(device_id)
    }
    
    async fn find_by_id(&self, id: &DeviceId) -> Result<Option<Device>, RepositoryError> {
        let connection = self.connection_pool.get().await?;
        let device_data = connection.query_device_by_id(id).await?;
        Ok(device_data.map(Device::from))
    }
    
    async fn find_by_status(&self, status: DeviceStatus) -> Result<Vec<Device>, RepositoryError> {
        let connection = self.connection_pool.get().await?;
        let devices_data = connection.query_devices_by_status(status).await?;
        Ok(devices_data.into_iter().map(Device::from).collect())
    }
}

pub struct EmailNotificationService {
    email_client: Arc<dyn EmailClient>,
}

impl NotificationService for EmailNotificationService {
    async fn send_notification(&self, notification: Notification) -> Result<(), NotificationError> {
        let email = Email {
            to: notification.recipient,
            subject: notification.subject,
            body: notification.message,
        };
        
        self.email_client.send_email(email).await?;
        Ok(())
    }
}

// 应用核心（不依赖外部）
pub struct IoTApplication {
    device_repository: Arc<dyn DeviceRepository>,
    notification_service: Arc<dyn NotificationService>,
}

impl IoTApplication {
    pub fn new(
        device_repository: Arc<dyn DeviceRepository>,
        notification_service: Arc<dyn NotificationService>,
    ) -> Self {
        Self {
            device_repository,
            notification_service,
        }
    }
    
    pub async fn register_device(&self, device_info: DeviceInfo) -> Result<DeviceId, ApplicationError> {
        // 业务逻辑不依赖具体的外部实现
        let device = Device::new(device_info);
        let device_id = self.device_repository.save(device).await?;
        
        let notification = Notification {
            recipient: "admin@iot.com".to_string(),
            subject: "New Device Registered".to_string(),
            message: format!("Device {} has been registered", device_id),
        };
        
        self.notification_service.send_notification(notification).await?;
        
        Ok(device_id)
    }
}
```

## 6. 总结

架构模式为IoT系统提供了结构化的设计指导：

1. **分层架构**：清晰的职责分离，便于维护和测试
2. **微服务架构**：服务独立部署，技术栈多样化
3. **事件驱动架构**：松耦合，高可扩展性
4. **CQRS模式**：读写分离，性能优化
5. **六边形架构**：端口适配器，核心业务隔离

选择合适的架构模式需要考虑系统规模、团队结构、技术栈和业务需求等因素。
