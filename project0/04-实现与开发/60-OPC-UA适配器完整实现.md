# OPC-UA适配器完整实现

## 1. 总体架构设计

### 1.1 核心组件架构

```rust
// src/adapters/opcua/mod.rs
use opcua_client::{ClientBuilder, DataValue, NodeId, Session};
use opcua_core::prelude::*;
use tokio::sync::{mpsc, RwLock};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct OpcUaConfig {
    pub endpoint_url: String,
    pub security_policy: String,
    pub security_mode: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub certificate_path: Option<String>,
    pub private_key_path: Option<String>,
    pub connection_timeout: u64,
    pub session_timeout: u64,
    pub subscription_publishing_interval: f64,
}

#[derive(Debug)]
pub struct OpcUaAdapter {
    config: OpcUaConfig,
    session: Arc<RwLock<Option<Session>>>,
    subscriptions: Arc<RwLock<HashMap<u32, SubscriptionInfo>>>,
    event_sender: mpsc::UnboundedSender<SemanticEvent>,
    node_mappings: Arc<RwLock<HashMap<NodeId, SemanticNodeInfo>>>,
}

#[derive(Debug, Clone)]
struct SubscriptionInfo {
    subscription_id: u32,
    monitored_items: Vec<MonitoredItemInfo>,
    publishing_interval: f64,
}

#[derive(Debug, Clone)]
struct MonitoredItemInfo {
    monitored_item_id: u32,
    node_id: NodeId,
    semantic_path: String,
    data_type: String,
}

#[derive(Debug, Clone)]
pub struct SemanticNodeInfo {
    pub node_id: NodeId,
    pub browse_name: String,
    pub display_name: String,
    pub semantic_type: String,
    pub unit: Option<String>,
    pub range: Option<(f64, f64)>,
    pub access_level: u8,
}
```

### 1.2 连接管理实现

```rust
impl OpcUaAdapter {
    pub async fn new(
        config: OpcUaConfig,
        event_sender: mpsc::UnboundedSender<SemanticEvent>,
    ) -> Result<Self, AdapterError> {
        let adapter = Self {
            config,
            session: Arc::new(RwLock::new(None)),
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            event_sender,
            node_mappings: Arc::new(RwLock::new(HashMap::new())),
        };
        
        adapter.initialize_connection().await?;
        Ok(adapter)
    }

    async fn initialize_connection(&self) -> Result<(), AdapterError> {
        let client = ClientBuilder::new()
            .application_name("IoT Semantic Gateway")
            .application_uri("urn:iot-semantic-gateway")
            .product_uri("urn:iot-semantic-gateway")
            .trust_server_certs(true)
            .create_sample_keypair(true)
            .session_retry_limit(3)
            .session_retry_interval(5000)
            .session_timeout(self.config.session_timeout * 1000)
            .client()?;

        let session = client.connect_to_endpoint(
            &self.config.endpoint_url,
            SecurityPolicy::from_str(&self.config.security_policy)?,
            MessageSecurityMode::from_str(&self.config.security_mode)?,
            UserTokenPolicy::anonymous(),
        ).await?;

        *self.session.write().await = Some(session);
        
        // 启动连接监控
        self.start_connection_monitor().await;
        
        // 执行节点发现
        self.discover_nodes().await?;
        
        Ok(())
    }

    async fn start_connection_monitor(&self) {
        let session_clone = Arc::clone(&self.session);
        let config_clone = self.config.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                std::time::Duration::from_secs(30)
            );
            
            loop {
                interval.tick().await;
                
                let session_guard = session_clone.read().await;
                if let Some(session) = session_guard.as_ref() {
                    if !session.is_connected() {
                        drop(session_guard);
                        // 重连逻辑
                        Self::reconnect(Arc::clone(&session_clone), &config_clone).await;
                    }
                }
            }
        });
    }

    async fn reconnect(
        session_lock: Arc<RwLock<Option<Session>>>,
        config: &OpcUaConfig,
    ) {
        for attempt in 1..=5 {
            match Self::create_session(config).await {
                Ok(new_session) => {
                    *session_lock.write().await = Some(new_session);
                    tracing::info!("OPC-UA重连成功，尝试次数: {}", attempt);
                    break;
                }
                Err(e) => {
                    tracing::warn!("OPC-UA重连失败，尝试次数: {}, 错误: {:?}", attempt, e);
                    tokio::time::sleep(std::time::Duration::from_secs(attempt * 2)).await;
                }
            }
        }
    }
}
```

## 2. 语义映射实现

### 2.1 节点语义发现

```rust
impl OpcUaAdapter {
    async fn discover_nodes(&self) -> Result<(), AdapterError> {
        let session_guard = self.session.read().await;
        let session = session_guard.as_ref().ok_or(AdapterError::NoConnection)?;
        
        // 从根节点开始浏览
        let root_node = NodeId::from(ObjectId::RootFolder);
        self.browse_recursive(session, root_node, 0).await?;
        
        Ok(())
    }

    async fn browse_recursive(
        &self,
        session: &Session,
        node_id: NodeId,
        depth: usize,
    ) -> Result<(), AdapterError> {
        if depth > 10 { // 限制递归深度
            return Ok(());
        }

        let browse_result = session.browse(
            &node_id,
            BrowseDirection::Forward,
            ReferenceTypeId::Organizes,
            true,
            NodeClassMask::all(),
            ResultMask::all(),
        ).await?;

        for reference in browse_result {
            let target_node = reference.node_id.node_id;
            
            // 读取节点属性
            let attributes = self.read_node_attributes(session, &target_node).await?;
            
            // 语义分析
            let semantic_info = self.analyze_node_semantics(&attributes).await?;
            
            // 存储映射
            self.node_mappings.write().await.insert(
                target_node.clone(),
                semantic_info,
            );
            
            // 如果是对象或变量，继续递归
            if reference.node_class == NodeClass::Object ||
               reference.node_class == NodeClass::Variable {
                self.browse_recursive(session, target_node, depth + 1).await?;
            }
        }
        
        Ok(())
    }

    async fn read_node_attributes(
        &self,
        session: &Session,
        node_id: &NodeId,
    ) -> Result<NodeAttributes, AdapterError> {
        let read_request = vec![
            ReadValueId::from((node_id, AttributeId::BrowseName)),
            ReadValueId::from((node_id, AttributeId::DisplayName)),
            ReadValueId::from((node_id, AttributeId::Description)),
            ReadValueId::from((node_id, AttributeId::DataType)),
            ReadValueId::from((node_id, AttributeId::AccessLevel)),
            ReadValueId::from((node_id, AttributeId::UserAccessLevel)),
        ];

        let results = session.read(&read_request).await?;
        
        Ok(NodeAttributes {
            browse_name: Self::extract_string_value(&results[0])?,
            display_name: Self::extract_string_value(&results[1])?,
            description: Self::extract_string_value(&results[2]).ok(),
            data_type: Self::extract_node_id_value(&results[3]).ok(),
            access_level: Self::extract_byte_value(&results[4]).unwrap_or(0),
        })
    }

    async fn analyze_node_semantics(
        &self,
        attributes: &NodeAttributes,
    ) -> Result<SemanticNodeInfo, AdapterError> {
        // 语义类型推断
        let semantic_type = self.infer_semantic_type(attributes).await?;
        
        // 单位识别
        let unit = self.extract_unit_information(attributes).await;
        
        // 范围分析
        let range = self.analyze_value_range(attributes).await;
        
        Ok(SemanticNodeInfo {
            node_id: attributes.node_id.clone(),
            browse_name: attributes.browse_name.clone(),
            display_name: attributes.display_name.clone(),
            semantic_type,
            unit,
            range,
            access_level: attributes.access_level,
        })
    }

    async fn infer_semantic_type(
        &self,
        attributes: &NodeAttributes,
    ) -> Result<String, AdapterError> {
        // 基于节点名称的语义推断
        let name_lower = attributes.browse_name.to_lowercase();
        
        let semantic_type = match true {
            _ if name_lower.contains("temperature") => "sosa:Temperature",
            _ if name_lower.contains("pressure") => "sosa:Pressure", 
            _ if name_lower.contains("humidity") => "sosa:Humidity",
            _ if name_lower.contains("speed") => "sosa:Speed",
            _ if name_lower.contains("level") => "sosa:Level",
            _ if name_lower.contains("flow") => "sosa:FlowRate",
            _ if name_lower.contains("power") => "saref:Power",
            _ if name_lower.contains("energy") => "saref:Energy",
            _ if name_lower.contains("voltage") => "saref:Voltage",
            _ if name_lower.contains("current") => "saref:Current",
            _ if name_lower.contains("alarm") || name_lower.contains("alert") => "sosa:Alert",
            _ if name_lower.contains("status") => "saref:State",
            _ if name_lower.contains("position") => "geo:Point",
            _ => "sosa:ObservableProperty",
        };
        
        Ok(semantic_type.to_string())
    }
}
```

## 3. 数据订阅与发布

### 3.1 订阅管理

```rust
impl OpcUaAdapter {
    pub async fn create_subscription(
        &self,
        nodes: Vec<NodeId>,
        publishing_interval: f64,
    ) -> Result<u32, AdapterError> {
        let session_guard = self.session.read().await;
        let session = session_guard.as_ref().ok_or(AdapterError::NoConnection)?;
        
        // 创建订阅
        let subscription_id = session.create_subscription(
            publishing_interval,
            10, // lifetime_count
            3,  // max_keep_alive_count  
            0,  // max_notifications_per_publish
            true, // publishing_enabled
            0, // priority
        ).await?;
        
        // 创建监控项
        let mut monitored_items = Vec::new();
        for (i, node_id) in nodes.iter().enumerate() {
            let item_id = session.create_monitored_item(
                subscription_id,
                TimestampsToReturn::Both,
                &ReadValueId::from((node_id, AttributeId::Value)),
                MonitoringMode::Reporting,
                &MonitoringParameters::default(),
            ).await?;
            
            let semantic_info = self.node_mappings.read().await
                .get(node_id)
                .cloned()
                .unwrap_or_else(|| SemanticNodeInfo {
                    node_id: node_id.clone(),
                    browse_name: format!("Node_{}", i),
                    display_name: format!("Unknown Node {}", i),
                    semantic_type: "sosa:ObservableProperty".to_string(),
                    unit: None,
                    range: None,
                    access_level: 1,
                });
                
            monitored_items.push(MonitoredItemInfo {
                monitored_item_id: item_id,
                node_id: node_id.clone(),
                semantic_path: format!("/{}/{}", 
                    semantic_info.semantic_type, 
                    semantic_info.browse_name),
                data_type: "Unknown".to_string(),
            });
        }
        
        // 存储订阅信息
        let subscription_info = SubscriptionInfo {
            subscription_id,
            monitored_items,
            publishing_interval,
        };
        
        self.subscriptions.write().await.insert(subscription_id, subscription_info);
        
        // 启动数据处理
        self.start_data_processing(subscription_id).await;
        
        Ok(subscription_id)
    }

    async fn start_data_processing(&self, subscription_id: u32) {
        let session_clone = Arc::clone(&self.session);
        let subscriptions_clone = Arc::clone(&self.subscriptions);
        let event_sender_clone = self.event_sender.clone();
        
        tokio::spawn(async move {
            loop {
                let session_guard = session_clone.read().await;
                if let Some(session) = session_guard.as_ref() {
                    match session.publish().await {
                        Ok(publish_response) => {
                            for notification_data in publish_response.notification_message.notification_data {
                                Self::process_notification_data(
                                    &notification_data,
                                    subscription_id,
                                    &subscriptions_clone,
                                    &event_sender_clone,
                                ).await;
                            }
                        }
                        Err(e) => {
                            tracing::error!("OPC-UA发布失败: {:?}", e);
                            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        }
                    }
                }
                
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        });
    }

    async fn process_notification_data(
        notification_data: &NotificationData,
        subscription_id: u32,
        subscriptions: &Arc<RwLock<HashMap<u32, SubscriptionInfo>>>,
        event_sender: &mpsc::UnboundedSender<SemanticEvent>,
    ) {
        let subscriptions_guard = subscriptions.read().await;
        let subscription_info = match subscriptions_guard.get(&subscription_id) {
            Some(info) => info,
            None => return,
        };

        if let NotificationData::DataChangeNotification(data_change) = notification_data {
            for monitored_item in &data_change.monitored_items {
                if let Some(item_info) = subscription_info.monitored_items.iter()
                    .find(|item| item.monitored_item_id == monitored_item.client_handle) {
                    
                    let semantic_event = SemanticEvent {
                        event_id: uuid::Uuid::new_v4().to_string(),
                        timestamp: chrono::Utc::now(),
                        source: "opcua".to_string(),
                        semantic_path: item_info.semantic_path.clone(),
                        value: Self::convert_opc_value(&monitored_item.value),
                        quality: monitored_item.value.status.map(|s| s.name().to_string()),
                        metadata: Self::create_metadata(&item_info),
                    };
                    
                    if let Err(e) = event_sender.send(semantic_event) {
                        tracing::error!("发送语义事件失败: {:?}", e);
                    }
                }
            }
        }
    }
}
```

## 4. 错误处理与监控

### 4.1 错误类型定义

```rust
#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("OPC-UA连接错误: {0}")]
    ConnectionError(#[from] opcua_client::ClientError),
    
    #[error("会话错误: {0}")]
    SessionError(String),
    
    #[error("订阅错误: {0}")]
    SubscriptionError(String),
    
    #[error("语义映射错误: {0}")]
    SemanticMappingError(String),
    
    #[error("配置错误: {0}")]
    ConfigError(String),
    
    #[error("没有可用连接")]
    NoConnection,
}
```

### 4.2 健康检查

```rust
impl OpcUaAdapter {
    pub async fn health_check(&self) -> HealthStatus {
        let mut status = HealthStatus::new("opcua-adapter");
        
        // 检查连接状态
        let session_guard = self.session.read().await;
        match session_guard.as_ref() {
            Some(session) if session.is_connected() => {
                status.add_check("connection", true, "已连接");
            }
            _ => {
                status.add_check("connection", false, "未连接");
                return status;
            }
        }
        
        // 检查订阅状态
        let subscriptions_guard = self.subscriptions.read().await;
        let subscription_count = subscriptions_guard.len();
        status.add_check(
            "subscriptions", 
            subscription_count > 0, 
            &format!("活跃订阅数: {}", subscription_count)
        );
        
        // 检查节点映射
        let mappings_guard = self.node_mappings.read().await;
        let mapping_count = mappings_guard.len();
        status.add_check(
            "node_mappings",
            mapping_count > 0,
            &format!("已映射节点数: {}", mapping_count)
        );
        
        status
    }
}
```

## 5. 配置文件

### 5.1 适配器配置

```yaml
# config/opcua_adapter.yaml
opcua:
  adapters:
    - name: "factory_opcua_server"
      endpoint_url: "opc.tcp://192.168.1.100:4840"
      security_policy: "None"
      security_mode: "None"
      connection_timeout: 10
      session_timeout: 300
      subscription_publishing_interval: 1000.0
      
      # 语义映射配置
      semantic_mappings:
        - node_pattern: ".*Temperature.*"
          semantic_type: "sosa:Temperature"
          unit: "°C"
          
        - node_pattern: ".*Pressure.*"
          semantic_type: "sosa:Pressure" 
          unit: "Pa"
          
        - node_pattern: ".*Speed.*"
          semantic_type: "sosa:Speed"
          unit: "m/s"
      
      # 订阅配置
      subscriptions:
        - name: "sensor_data"
          nodes: 
            - "ns=2;s=Temperature1"
            - "ns=2;s=Pressure1"
            - "ns=2;s=Speed1"
          publishing_interval: 1000.0
          
        - name: "alarm_data"
          nodes:
            - "ns=2;s=HighTempAlarm"
            - "ns=2;s=LowPressureAlarm"
          publishing_interval: 500.0

    - name: "building_automation"
      endpoint_url: "opc.tcp://192.168.1.200:4840"
      security_policy: "Basic256Sha256"
      security_mode: "SignAndEncrypt"
      username: "admin"
      password: "password123"
      certificate_path: "/etc/certs/client.der"
      private_key_path: "/etc/certs/private.pem"
```

## 6. 使用示例

### 6.1 基本使用

```rust
use crate::adapters::opcua::{OpcUaAdapter, OpcUaConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (event_sender, mut event_receiver) = mpsc::unbounded_channel();
    
    let config = OpcUaConfig {
        endpoint_url: "opc.tcp://localhost:4840".to_string(),
        security_policy: "None".to_string(),
        security_mode: "None".to_string(),
        username: None,
        password: None,
        certificate_path: None,
        private_key_path: None,
        connection_timeout: 10,
        session_timeout: 300,
        subscription_publishing_interval: 1000.0,
    };
    
    // 创建适配器
    let adapter = OpcUaAdapter::new(config, event_sender).await?;
    
    // 创建订阅
    let nodes = vec![
        NodeId::from("ns=2;s=Temperature"),
        NodeId::from("ns=2;s=Pressure"),
    ];
    
    let subscription_id = adapter.create_subscription(nodes, 1000.0).await?;
    
    // 处理事件
    tokio::spawn(async move {
        while let Some(event) = event_receiver.recv().await {
            println!("收到语义事件: {:?}", event);
        }
    });
    
    // 健康检查
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
        loop {
            interval.tick().await;
            let health = adapter.health_check().await;
            println!("健康状态: {:?}", health);
        }
    });
    
    // 保持运行
    tokio::signal::ctrl_c().await?;
    Ok(())
}
```

这个OPC-UA适配器实现提供了完整的连接管理、语义映射、数据订阅和错误处理功能，是IoT语义互操作平台的核心组件之一。
