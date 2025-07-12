# oneM2M适配器完整实现

## 1. 总体架构设计

### 1.1 核心组件架构

```rust
// src/adapters/onem2m/mod.rs
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct OneM2MConfig {
    pub cse_base_url: String,
    pub cse_id: String,
    pub ae_id: String,
    pub originator: String,
    pub request_identifier: String,
    pub resource_type: String,
    pub content_format: String,
    pub accept_format: String,
    pub api_key: Option<String>,
    pub timeout: u64,
    pub retry_count: u32,
    pub polling_interval: u64,
}

#[derive(Debug)]
pub struct OneM2MAdapter {
    config: OneM2MConfig,
    http_client: Client,
    resource_cache: Arc<RwLock<HashMap<String, ResourceInfo>>>,
    subscriptions: Arc<RwLock<HashMap<String, SubscriptionInfo>>>,
    event_sender: mpsc::UnboundedSender<SemanticEvent>,
    notification_handler: Arc<RwLock<Option<NotificationHandler>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceInfo {
    pub resource_id: String,
    pub resource_type: String,
    pub resource_name: String,
    pub parent_id: String,
    pub creation_time: String,
    pub last_modified_time: String,
    pub semantic_type: String,
    pub labels: Vec<String>,
    pub attributes: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
struct SubscriptionInfo {
    pub subscription_id: String,
    pub resource_uri: String,
    pub notification_uri: String,
    pub event_criteria: Vec<String>,
    pub semantic_mapping: SemanticMapping,
}

#[derive(Debug, Clone)]
struct SemanticMapping {
    pub source_path: String,
    pub target_ontology: String,
    pub transformation_rules: Vec<TransformationRule>,
}

#[derive(Debug, Clone)]
struct TransformationRule {
    pub condition: String,
    pub action: String,
    pub parameters: HashMap<String, String>,
}
```

### 1.2 oneM2M资源模型实现

```rust
// oneM2M标准资源类型
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "ty")]
pub enum OneM2MResource {
    #[serde(rename = "1")]
    AccessControlPolicy(AccessControlPolicyResource),
    #[serde(rename = "2")]
    ApplicationEntity(ApplicationEntityResource),
    #[serde(rename = "3")]
    Container(ContainerResource),
    #[serde(rename = "4")]
    ContentInstance(ContentInstanceResource),
    #[serde(rename = "5")]
    CSEBase(CSEBaseResource),
    #[serde(rename = "9")]
    Group(GroupResource),
    #[serde(rename = "23")]
    Subscription(SubscriptionResource),
    #[serde(rename = "58")]
    FlexContainer(FlexContainerResource),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerResource {
    pub rn: String,           // resourceName
    pub ri: String,           // resourceID
    pub pi: String,           // parentID
    pub ty: u32,              // resourceType
    pub ct: String,           // creationTime
    pub lt: String,           // lastModifiedTime
    pub lbl: Vec<String>,     // labels
    pub acpi: Vec<String>,    // accessControlPolicyIDs
    pub et: String,           // expirationTime
    pub st: u32,              // stateTag
    pub mni: Option<u32>,     // maxNrOfInstances
    pub mbs: Option<u32>,     // maxByteSize
    pub mia: Option<u32>,     // maxInstanceAge
    pub cni: u32,             // currentNrOfInstances
    pub cbs: u32,             // currentByteSize
    pub or: Option<String>,   // ontologyRef
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentInstanceResource {
    pub rn: String,
    pub ri: String,
    pub pi: String,
    pub ty: u32,
    pub ct: String,
    pub lt: String,
    pub lbl: Vec<String>,
    pub st: u32,
    pub cnf: String,          // contentInfo
    pub cs: u32,              // contentSize
    pub con: serde_json::Value, // content
    pub or: Option<String>,   // ontologyRef
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionResource {
    pub rn: String,
    pub ri: String,
    pub pi: String,
    pub ty: u32,
    pub ct: String,
    pub lt: String,
    pub et: String,
    pub nu: Vec<String>,      // notificationURI
    pub enc: EventNotificationCriteria,
    pub exc: Option<u32>,     // expirationCounter
    pub gpi: Option<String>,  // groupID
    pub nfu: Option<String>,  // notificationForwardingURI
    pub bn: Option<BatchNotify>,
    pub rl: Option<RateLimitInfo>,
    pub psn: Option<bool>,    // pendingNotification
    pub pn: Option<u32>,      // notificationPersistence
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventNotificationCriteria {
    pub crb: Option<String>,  // createdBefore
    pub cra: Option<String>,  // createdAfter
    pub ms: Option<String>,   // modifiedSince
    pub us: Option<String>,   // unmodifiedSince
    pub sts: Option<u32>,     // stateTagSmaller
    pub stb: Option<u32>,     // stateTagBigger
    pub exb: Option<String>,  // expireBefore
    pub exa: Option<String>,  // expireAfter
    pub lbl: Option<Vec<String>>, // labels
    pub ty: Option<Vec<u32>>, // resourceType
    pub sza: Option<u32>,     // sizeAbove
    pub szb: Option<u32>,     // sizeBelow
    pub catr: Option<Vec<String>>, // contentType
    pub patr: Option<HashMap<String, String>>, // attribute
}
```

## 2. HTTP客户端实现

### 2.1 RESTful API客户端

```rust
impl OneM2MAdapter {
    pub async fn new(
        config: OneM2MConfig,
        event_sender: mpsc::UnboundedSender<SemanticEvent>,
    ) -> Result<Self, AdapterError> {
        let http_client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout))
            .build()?;

        let adapter = Self {
            config,
            http_client,
            resource_cache: Arc::new(RwLock::new(HashMap::new())),
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            event_sender,
            notification_handler: Arc::new(RwLock::new(None)),
        };

        // 初始化连接和发现资源
        adapter.initialize().await?;
        
        Ok(adapter)
    }

    async fn initialize(&self) -> Result<(), AdapterError> {
        // 验证CSE连接
        self.verify_cse_connection().await?;
        
        // 发现资源树
        self.discover_resource_tree().await?;
        
        // 启动通知处理器
        self.start_notification_handler().await?;
        
        Ok(())
    }

    async fn verify_cse_connection(&self) -> Result<(), AdapterError> {
        let url = format!("{}/{}", self.config.cse_base_url, self.config.cse_id);
        
        let response = self.http_client
            .get(&url)
            .header("X-M2M-Origin", &self.config.originator)
            .header("X-M2M-RI", &self.config.request_identifier)
            .header("Accept", &self.config.accept_format)
            .send()
            .await?;

        if response.status().is_success() {
            tracing::info!("oneM2M CSE连接验证成功: {}", url);
            Ok(())
        } else {
            Err(AdapterError::ConnectionError(format!(
                "CSE连接失败: {} - {}", 
                response.status(), 
                response.text().await.unwrap_or_default()
            )))
        }
    }

    pub async fn create_resource(
        &self,
        parent_uri: &str,
        resource: &OneM2MResource,
    ) -> Result<String, AdapterError> {
        let url = format!("{}{}", self.config.cse_base_url, parent_uri);
        
        let payload = serde_json::to_string(resource)?;
        
        let response = self.http_client
            .post(&url)
            .header("X-M2M-Origin", &self.config.originator)
            .header("X-M2M-RI", &self.generate_request_id())
            .header("Content-Type", &self.config.content_format)
            .header("Accept", &self.config.accept_format)
            .body(payload)
            .send()
            .await?;

        if response.status().is_success() {
            let created_resource: serde_json::Value = response.json().await?;
            let resource_uri = created_resource["m2m:uri"]
                .as_str()
                .ok_or(AdapterError::ParseError("无法解析资源URI".to_string()))?;
            
            // 更新缓存
            self.update_resource_cache(&created_resource).await?;
            
            Ok(resource_uri.to_string())
        } else {
            Err(AdapterError::RequestError(format!(
                "创建资源失败: {} - {}",
                response.status(),
                response.text().await.unwrap_or_default()
            )))
        }
    }

    pub async fn retrieve_resource(
        &self,
        resource_uri: &str,
    ) -> Result<OneM2MResource, AdapterError> {
        let url = format!("{}{}", self.config.cse_base_url, resource_uri);
        
        let response = self.http_client
            .get(&url)
            .header("X-M2M-Origin", &self.config.originator)
            .header("X-M2M-RI", &self.generate_request_id())
            .header("Accept", &self.config.accept_format)
            .send()
            .await?;

        if response.status().is_success() {
            let resource_data: serde_json::Value = response.json().await?;
            let resource = self.parse_resource(&resource_data)?;
            
            // 更新缓存
            self.update_resource_cache(&resource_data).await?;
            
            Ok(resource)
        } else {
            Err(AdapterError::RequestError(format!(
                "获取资源失败: {} - {}",
                response.status(),
                response.text().await.unwrap_or_default()
            )))
        }
    }

    pub async fn update_resource(
        &self,
        resource_uri: &str,
        resource: &OneM2MResource,
    ) -> Result<(), AdapterError> {
        let url = format!("{}{}", self.config.cse_base_url, resource_uri);
        
        let payload = serde_json::to_string(resource)?;
        
        let response = self.http_client
            .put(&url)
            .header("X-M2M-Origin", &self.config.originator)
            .header("X-M2M-RI", &self.generate_request_id())
            .header("Content-Type", &self.config.content_format)
            .header("Accept", &self.config.accept_format)
            .body(payload)
            .send()
            .await?;

        if response.status().is_success() {
            let updated_resource: serde_json::Value = response.json().await?;
            self.update_resource_cache(&updated_resource).await?;
            Ok(())
        } else {
            Err(AdapterError::RequestError(format!(
                "更新资源失败: {} - {}",
                response.status(),
                response.text().await.unwrap_or_default()
            )))
        }
    }

    pub async fn delete_resource(&self, resource_uri: &str) -> Result<(), AdapterError> {
        let url = format!("{}{}", self.config.cse_base_url, resource_uri);
        
        let response = self.http_client
            .delete(&url)
            .header("X-M2M-Origin", &self.config.originator)
            .header("X-M2M-RI", &self.generate_request_id())
            .send()
            .await?;

        if response.status().is_success() {
            // 从缓存中移除
            self.remove_from_cache(resource_uri).await;
            Ok(())
        } else {
            Err(AdapterError::RequestError(format!(
                "删除资源失败: {} - {}",
                response.status(),
                response.text().await.unwrap_or_default()
            )))
        }
    }
}
```

## 3. 订阅和通知处理

### 3.1 订阅管理

```rust
impl OneM2MAdapter {
    pub async fn create_subscription(
        &self,
        parent_uri: &str,
        notification_uri: &str,
        event_criteria: EventNotificationCriteria,
    ) -> Result<String, AdapterError> {
        let subscription_name = format!("sub_{}", self.generate_request_id());
        
        let subscription = SubscriptionResource {
            rn: subscription_name.clone(),
            ri: String::new(), // 服务器生成
            pi: String::new(), // 服务器设置
            ty: 23, // Subscription resource type
            ct: String::new(), // 服务器生成
            lt: String::new(), // 服务器生成
            et: self.calculate_expiration_time(),
            nu: vec![notification_uri.to_string()],
            enc: event_criteria,
            exc: Some(100), // 通知计数器
            gpi: None,
            nfu: None,
            bn: None,
            rl: None,
            psn: Some(true),
            pn: Some(1),
        };

        let subscription_resource = OneM2MResource::Subscription(subscription);
        
        let subscription_uri = self.create_resource(parent_uri, &subscription_resource).await?;
        
        // 存储订阅信息
        let subscription_info = SubscriptionInfo {
            subscription_id: subscription_name,
            resource_uri: subscription_uri.clone(),
            notification_uri: notification_uri.to_string(),
            event_criteria: vec!["dataCreated".to_string(), "dataUpdated".to_string()],
            semantic_mapping: SemanticMapping {
                source_path: parent_uri.to_string(),
                target_ontology: "sosa:Observation".to_string(),
                transformation_rules: vec![],
            },
        };
        
        self.subscriptions.write().await.insert(
            subscription_uri.clone(),
            subscription_info,
        );
        
        Ok(subscription_uri)
    }

    async fn start_notification_handler(&self) -> Result<(), AdapterError> {
        use warp::Filter;
        
        let subscriptions_clone = Arc::clone(&self.subscriptions);
        let event_sender_clone = self.event_sender.clone();
        
        // 创建通知接收端点
        let notification_route = warp::path("notifications")
            .and(warp::post())
            .and(warp::body::json())
            .map(move |notification: serde_json::Value| {
                let subscriptions = Arc::clone(&subscriptions_clone);
                let sender = event_sender_clone.clone();
                
                tokio::spawn(async move {
                    if let Err(e) = Self::handle_notification(notification, subscriptions, sender).await {
                        tracing::error!("处理通知失败: {:?}", e);
                    }
                });
                
                warp::reply::with_status("OK", warp::http::StatusCode::OK)
            });

        // 启动通知服务器
        let routes = notification_route.with(warp::cors().allow_any_origin());
        
        tokio::spawn(async move {
            warp::serve(routes)
                .run(([0, 0, 0, 0], 8080))
                .await;
        });
        
        Ok(())
    }

    async fn handle_notification(
        notification: serde_json::Value,
        subscriptions: Arc<RwLock<HashMap<String, SubscriptionInfo>>>,
        event_sender: mpsc::UnboundedSender<SemanticEvent>,
    ) -> Result<(), AdapterError> {
        // 解析通知
        let notification_request = notification["m2m:sgn"].as_object()
            .ok_or(AdapterError::ParseError("无效的通知格式".to_string()))?;
        
        let subscription_reference = notification_request["sur"]
            .as_str()
            .ok_or(AdapterError::ParseError("缺少订阅引用".to_string()))?;
        
        let notification_event = &notification_request["nev"];
        
        // 查找订阅信息
        let subscriptions_guard = subscriptions.read().await;
        let subscription_info = subscriptions_guard.get(subscription_reference)
            .ok_or(AdapterError::NotFound(format!("订阅未找到: {}", subscription_reference)))?;
        
        // 转换为语义事件
        let semantic_event = Self::convert_to_semantic_event(
            notification_event,
            subscription_info,
        )?;
        
        // 发送事件
        event_sender.send(semantic_event)
            .map_err(|e| AdapterError::SendError(format!("发送事件失败: {:?}", e)))?;
        
        Ok(())
    }

    fn convert_to_semantic_event(
        notification_event: &serde_json::Value,
        subscription_info: &SubscriptionInfo,
    ) -> Result<SemanticEvent, AdapterError> {
        let representation = &notification_event["rep"];
        
        // 提取资源数据
        let resource_data = representation["m2m:cin"].as_object()
            .or_else(|| representation["m2m:cnt"].as_object())
            .or_else(|| representation["m2m:fcnt"].as_object())
            .ok_or(AdapterError::ParseError("无法解析资源数据".to_string()))?;
        
        // 创建语义事件
        Ok(SemanticEvent {
            event_id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            source: "onem2m".to_string(),
            semantic_path: subscription_info.semantic_mapping.source_path.clone(),
            value: resource_data.get("con").cloned().unwrap_or(serde_json::Value::Null),
            quality: None,
            metadata: Self::extract_metadata(resource_data),
        })
    }
}
```

## 4. 语义映射和转换

### 4.1 ontology映射实现

```rust
impl OneM2MAdapter {
    async fn discover_resource_tree(&self) -> Result<(), AdapterError> {
        let cse_uri = format!("/{}", self.config.cse_id);
        self.discover_resources_recursive(&cse_uri, 0).await
    }

    async fn discover_resources_recursive(
        &self,
        parent_uri: &str,
        depth: usize,
    ) -> Result<(), AdapterError> {
        if depth > 10 { // 限制递归深度
            return Ok(());
        }

        // 发现查询
        let discovery_filter = format!(
            "{}?fu=1&drt=2&ty=3", // 查找Container类型资源
            parent_uri
        );
        
        let url = format!("{}{}", self.config.cse_base_url, discovery_filter);
        
        let response = self.http_client
            .get(&url)
            .header("X-M2M-Origin", &self.config.originator)
            .header("X-M2M-RI", &self.generate_request_id())
            .header("Accept", &self.config.accept_format)
            .send()
            .await?;

        if response.status().is_success() {
            let discovery_result: serde_json::Value = response.json().await?;
            
            if let Some(uris) = discovery_result["m2m:uril"].as_array() {
                for uri_value in uris {
                    if let Some(uri) = uri_value.as_str() {
                        // 获取资源详情
                        if let Ok(resource) = self.retrieve_resource(uri).await {
                            // 语义分析
                            self.analyze_resource_semantics(&resource, uri).await?;
                            
                            // 递归发现子资源
                            self.discover_resources_recursive(uri, depth + 1).await?;
                        }
                    }
                }
            }
        }
        
        Ok(())
    }

    async fn analyze_resource_semantics(
        &self,
        resource: &OneM2MResource,
        resource_uri: &str,
    ) -> Result<(), AdapterError> {
        let semantic_type = match resource {
            OneM2MResource::Container(container) => {
                self.infer_container_semantics(container).await?
            }
            OneM2MResource::ContentInstance(content) => {
                self.infer_content_semantics(content).await?
            }
            OneM2MResource::FlexContainer(flex_container) => {
                self.infer_flex_container_semantics(flex_container).await?
            }
            _ => "sosa:FeatureOfInterest".to_string(),
        };

        // 存储语义信息
        let resource_info = ResourceInfo {
            resource_id: self.extract_resource_id(resource),
            resource_type: self.get_resource_type_name(resource),
            resource_name: self.extract_resource_name(resource),
            parent_id: self.extract_parent_id(resource),
            creation_time: self.extract_creation_time(resource),
            last_modified_time: self.extract_last_modified_time(resource),
            semantic_type,
            labels: self.extract_labels(resource),
            attributes: self.extract_attributes(resource),
        };

        self.resource_cache.write().await.insert(
            resource_uri.to_string(),
            resource_info,
        );

        Ok(())
    }

    async fn infer_container_semantics(
        &self,
        container: &ContainerResource,
    ) -> Result<String, AdapterError> {
        // 基于标签的语义推断
        for label in &container.lbl {
            let label_lower = label.to_lowercase();
            match true {
                _ if label_lower.contains("temperature") => return Ok("sosa:Temperature".to_string()),
                _ if label_lower.contains("sensor") => return Ok("sosa:Sensor".to_string()),
                _ if label_lower.contains("actuator") => return Ok("sosa:Actuator".to_string()),
                _ if label_lower.contains("observation") => return Ok("sosa:Observation".to_string()),
                _ if label_lower.contains("measurement") => return Ok("om:Measurement".to_string()),
                _ => continue,
            }
        }

        // 基于资源名称的推断
        let name_lower = container.rn.to_lowercase();
        let semantic_type = match true {
            _ if name_lower.contains("temp") => "sosa:Temperature",
            _ if name_lower.contains("hum") => "sosa:Humidity",
            _ if name_lower.contains("press") => "sosa:Pressure",
            _ if name_lower.contains("light") => "sosa:Illuminance",
            _ if name_lower.contains("motion") => "sosa:Motion",
            _ if name_lower.contains("energy") => "saref:Energy",
            _ if name_lower.contains("power") => "saref:Power",
            _ => "sosa:ObservableProperty",
        };

        Ok(semantic_type.to_string())
    }

    async fn infer_content_semantics(
        &self,
        content: &ContentInstanceResource,
    ) -> Result<String, AdapterError> {
        // 分析内容类型
        let content_info = &content.cnf;
        
        let semantic_type = match content_info.as_str() {
            "application/json" => {
                // 分析JSON内容结构
                if let Ok(json_content) = serde_json::from_value::<serde_json::Value>(content.con.clone()) {
                    self.analyze_json_content_semantics(&json_content).await?
                } else {
                    "sosa:Result".to_string()
                }
            }
            "text/plain" => "rdfs:Literal".to_string(),
            "application/xml" => "sosa:Result".to_string(),
            _ => "sosa:Result".to_string(),
        };

        Ok(semantic_type)
    }

    async fn analyze_json_content_semantics(
        &self,
        json_content: &serde_json::Value,
    ) -> Result<String, AdapterError> {
        if let Some(obj) = json_content.as_object() {
            // 查找常见的物联网数据字段
            for (key, _value) in obj {
                let key_lower = key.to_lowercase();
                match key_lower.as_str() {
                    "temperature" | "temp" => return Ok("sosa:Temperature".to_string()),
                    "humidity" | "hum" => return Ok("sosa:Humidity".to_string()),
                    "pressure" | "press" => return Ok("sosa:Pressure".to_string()),
                    "illuminance" | "light" | "lux" => return Ok("sosa:Illuminance".to_string()),
                    "motion" | "movement" => return Ok("sosa:Motion".to_string()),
                    "energy" => return Ok("saref:Energy".to_string()),
                    "power" => return Ok("saref:Power".to_string()),
                    "voltage" => return Ok("saref:Voltage".to_string()),
                    "current" => return Ok("saref:Current".to_string()),
                    _ => continue,
                }
            }
        }

        Ok("sosa:Observation".to_string())
    }
}
```

## 5. 配置和错误处理

### 5.1 配置文件

```yaml
# config/onem2m_adapter.yaml
onem2m:
  adapters:
    - name: "iot_platform_cse"
      cse_base_url: "http://192.168.1.100:8080/~/in-cse"
      cse_id: "in-cse"
      ae_id: "semantic-gateway"
      originator: "C-semantic-gateway"
      request_identifier: "req_001"
      resource_type: "application/json"
      content_format: "application/vnd.onem2m-res+json"
      accept_format: "application/vnd.onem2m-res+json"
      timeout: 30
      retry_count: 3
      polling_interval: 5000
      
      # 语义映射配置
      semantic_mappings:
        - resource_pattern: ".*temperature.*"
          semantic_type: "sosa:Temperature"
          unit: "celsius"
          
        - resource_pattern: ".*sensor.*"
          semantic_type: "sosa:Sensor"
          
        - resource_pattern: ".*actuator.*"
          semantic_type: "sosa:Actuator"
      
      # 订阅配置
      subscriptions:
        - name: "sensor_data_subscription"
          parent_uri: "/in-cse/sensors"
          notification_uri: "http://localhost:8080/notifications"
          event_criteria:
            - "dataCreated"
            - "dataUpdated"
          
        - name: "alarm_subscription"
          parent_uri: "/in-cse/alarms"
          notification_uri: "http://localhost:8080/notifications"
          event_criteria:
            - "dataCreated"

# 错误处理配置
error_handling:
  retry_policy:
    max_attempts: 3
    backoff_strategy: "exponential"
    initial_delay: 1000
    max_delay: 30000
  
  circuit_breaker:
    failure_threshold: 5
    recovery_timeout: 60000
    half_open_max_calls: 3
```

### 5.2 错误类型定义

```rust
#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("HTTP请求错误: {0}")]
    RequestError(#[from] reqwest::Error),
    
    #[error("连接错误: {0}")]
    ConnectionError(String),
    
    #[error("解析错误: {0}")]
    ParseError(String),
    
    #[error("序列化错误: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("资源未找到: {0}")]
    NotFound(String),
    
    #[error("认证失败: {0}")]
    AuthenticationError(String),
    
    #[error("权限不足: {0}")]
    AuthorizationError(String),
    
    #[error("发送事件失败: {0}")]
    SendError(String),
    
    #[error("配置错误: {0}")]
    ConfigError(String),
}
```

## 6. 使用示例

### 6.1 基本使用

```rust
use crate::adapters::onem2m::{OneM2MAdapter, OneM2MConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (event_sender, mut event_receiver) = mpsc::unbounded_channel();
    
    let config = OneM2MConfig {
        cse_base_url: "http://localhost:8080/~/in-cse".to_string(),
        cse_id: "in-cse".to_string(),
        ae_id: "semantic-gateway".to_string(),
        originator: "C-semantic-gateway".to_string(),
        request_identifier: "req_001".to_string(),
        resource_type: "application/json".to_string(),
        content_format: "application/vnd.onem2m-res+json".to_string(),
        accept_format: "application/vnd.onem2m-res+json".to_string(),
        api_key: None,
        timeout: 30,
        retry_count: 3,
        polling_interval: 5000,
    };
    
    // 创建适配器
    let adapter = OneM2MAdapter::new(config, event_sender).await?;
    
    // 创建订阅
    let subscription_uri = adapter.create_subscription(
        "/in-cse/sensors",
        "http://localhost:8080/notifications",
        EventNotificationCriteria {
            crb: None,
            cra: None,
            ms: None,
            us: None,
            sts: None,
            stb: None,
            exb: None,
            exa: None,
            lbl: Some(vec!["temperature".to_string()]),
            ty: Some(vec![4]), // ContentInstance
            sza: None,
            szb: None,
            catr: None,
            patr: None,
        },
    ).await?;
    
    println!("创建订阅成功: {}", subscription_uri);
    
    // 处理事件
    tokio::spawn(async move {
        while let Some(event) = event_receiver.recv().await {
            println!("收到oneM2M语义事件: {:?}", event);
        }
    });
    
    // 保持运行
    tokio::signal::ctrl_c().await?;
    Ok(())
}
```

这个oneM2M适配器实现提供了完整的资源管理、订阅通知、语义映射和错误处理功能，完全符合oneM2M标准规范。
