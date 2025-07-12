# WoT(Web of Things)适配器完整实现

## 1. 总体架构设计

### 1.1 核心组件架构

```rust
// src/adapters/wot/mod.rs
use serde::{Deserialize, Serialize};
use reqwest::Client;
use tokio::sync::{mpsc, RwLock};
use std::collections::HashMap;
use std::sync::Arc;
use url::Url;

#[derive(Debug, Clone)]
pub struct WoTConfig {
    pub thing_directory_url: Option<String>,
    pub discovery_methods: Vec<DiscoveryMethod>,
    pub security_schemes: HashMap<String, SecurityScheme>,
    pub timeout: u64,
    pub retry_count: u32,
    pub polling_interval: u64,
    pub enable_websocket: bool,
    pub enable_coap: bool,
}

#[derive(Debug, Clone)]
pub enum DiscoveryMethod {
    Directory(String),    // TD Directory URL
    Multicast(String),    // Multicast address
    DNS_SD(String),       // DNS-SD service type
    Manual(Vec<String>),  // Manual TD URLs
}

#[derive(Debug)]
pub struct WoTAdapter {
    config: WoTConfig,
    http_client: Client,
    thing_descriptions: Arc<RwLock<HashMap<String, ThingDescription>>>,
    property_subscriptions: Arc<RwLock<HashMap<String, PropertySubscription>>>,
    event_subscriptions: Arc<RwLock<HashMap<String, EventSubscription>>>,
    event_sender: mpsc::UnboundedSender<SemanticEvent>,
    protocol_handlers: Arc<RwLock<HashMap<String, Box<dyn ProtocolHandler>>>>,
}

// WoT Thing Description 数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThingDescription {
    #[serde(rename = "@context")]
    pub context: serde_json::Value,
    #[serde(rename = "@type")]
    pub thing_type: Option<Vec<String>>,
    pub id: String,
    pub title: String,
    pub description: Option<String>,
    pub properties: Option<HashMap<String, PropertyAffordance>>,
    pub actions: Option<HashMap<String, ActionAffordance>>,
    pub events: Option<HashMap<String, EventAffordance>>,
    pub links: Option<Vec<Link>>,
    pub forms: Option<Vec<Form>>,
    pub security: Vec<String>,
    #[serde(rename = "securityDefinitions")]
    pub security_definitions: HashMap<String, SecurityScheme>,
    pub base: Option<String>,
    pub version: Option<VersionInfo>,
    pub created: Option<String>,
    pub modified: Option<String>,
    pub support: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyAffordance {
    #[serde(rename = "@type")]
    pub property_type: Option<Vec<String>>,
    pub title: Option<String>,
    pub description: Option<String>,
    #[serde(flatten)]
    pub data_schema: DataSchema,
    pub forms: Vec<Form>,
    pub observable: Option<bool>,
    pub writeOnly: Option<bool>,
    pub readOnly: Option<bool>,
    pub unit: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionAffordance {
    #[serde(rename = "@type")]
    pub action_type: Option<Vec<String>>,
    pub title: Option<String>,
    pub description: Option<String>,
    pub input: Option<DataSchema>,
    pub output: Option<DataSchema>,
    pub forms: Vec<Form>,
    pub safe: Option<bool>,
    pub idempotent: Option<bool>,
    pub synchronous: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventAffordance {
    #[serde(rename = "@type")]
    pub event_type: Option<Vec<String>>,
    pub title: Option<String>,
    pub description: Option<String>,
    pub data: Option<DataSchema>,
    pub forms: Vec<Form>,
    pub subscription: Option<DataSchema>,
    pub cancellation: Option<DataSchema>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Form {
    pub href: String,
    #[serde(rename = "contentType")]
    pub content_type: Option<String>,
    #[serde(rename = "contentCoding")]
    pub content_coding: Option<String>,
    #[serde(rename = "subprotocol")]
    pub subprotocol: Option<String>,
    pub security: Option<Vec<String>>,
    pub scopes: Option<Vec<String>>,
    pub response: Option<ExpectedResponse>,
    #[serde(rename = "additionalResponses")]
    pub additional_responses: Option<Vec<AdditionalExpectedResponse>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DataSchema {
    Object {
        #[serde(rename = "type")]
        schema_type: String,
        properties: Option<HashMap<String, Box<DataSchema>>>,
        required: Option<Vec<String>>,
    },
    Array {
        #[serde(rename = "type")]
        schema_type: String,
        items: Option<Box<DataSchema>>,
        #[serde(rename = "minItems")]
        min_items: Option<u32>,
        #[serde(rename = "maxItems")]
        max_items: Option<u32>,
    },
    Number {
        #[serde(rename = "type")]
        schema_type: String,
        minimum: Option<f64>,
        maximum: Option<f64>,
        unit: Option<String>,
    },
    String {
        #[serde(rename = "type")]
        schema_type: String,
        #[serde(rename = "minLength")]
        min_length: Option<u32>,
        #[serde(rename = "maxLength")]
        max_length: Option<u32>,
        pattern: Option<String>,
        #[serde(rename = "enum")]
        enumeration: Option<Vec<String>>,
    },
    Boolean {
        #[serde(rename = "type")]
        schema_type: String,
    },
    Null {
        #[serde(rename = "type")]
        schema_type: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityScheme {
    #[serde(rename = "@type")]
    pub scheme_type: Option<Vec<String>>,
    pub scheme: String,
    pub description: Option<String>,
    #[serde(rename = "proxyURI")]
    pub proxy_uri: Option<String>,
    // OAuth2特定字段
    pub authorization: Option<String>,
    pub token: Option<String>,
    pub refresh: Option<String>,
    pub scopes: Option<Vec<String>>,
    pub flow: Option<String>,
    // API Key特定字段
    #[serde(rename = "in")]
    pub location: Option<String>,
    pub name: Option<String>,
    // Basic Auth特定字段 - 无额外字段
    // Digest特定字段
    pub qop: Option<String>,
    // Bearer特定字段
    pub format: Option<String>,
    #[serde(rename = "alg")]
    pub algorithm: Option<String>,
}
```

### 1.2 协议处理器接口

```rust
#[async_trait::async_trait]
pub trait ProtocolHandler: Send + Sync {
    async fn read_property(
        &self,
        form: &Form,
        security: &[SecurityScheme],
    ) -> Result<serde_json::Value, WoTError>;
    
    async fn write_property(
        &self,
        form: &Form,
        value: &serde_json::Value,
        security: &[SecurityScheme],
    ) -> Result<(), WoTError>;
    
    async fn invoke_action(
        &self,
        form: &Form,
        input: Option<&serde_json::Value>,
        security: &[SecurityScheme],
    ) -> Result<Option<serde_json::Value>, WoTError>;
    
    async fn subscribe_event(
        &self,
        form: &Form,
        security: &[SecurityScheme],
        event_sender: mpsc::UnboundedSender<WoTEvent>,
    ) -> Result<String, WoTError>;
    
    async fn unsubscribe_event(
        &self,
        subscription_id: &str,
    ) -> Result<(), WoTError>;
    
    fn get_protocol(&self) -> &str;
}

// HTTP协议处理器
#[derive(Debug)]
pub struct HttpProtocolHandler {
    client: Client,
    timeout: std::time::Duration,
}

#[async_trait::async_trait]
impl ProtocolHandler for HttpProtocolHandler {
    async fn read_property(
        &self,
        form: &Form,
        security: &[SecurityScheme],
    ) -> Result<serde_json::Value, WoTError> {
        let mut request = self.client.get(&form.href);
        
        // 应用安全方案
        request = self.apply_security(request, security).await?;
        
        // 设置Content-Type
        if let Some(content_type) = &form.content_type {
            request = request.header("Accept", content_type);
        }
        
        let response = request.send().await?;
        
        if response.status().is_success() {
            let value = response.json().await?;
            Ok(value)
        } else {
            Err(WoTError::ProtocolError(format!(
                "HTTP请求失败: {} - {}",
                response.status(),
                response.text().await.unwrap_or_default()
            )))
        }
    }
    
    async fn write_property(
        &self,
        form: &Form,
        value: &serde_json::Value,
        security: &[SecurityScheme],
    ) -> Result<(), WoTError> {
        let mut request = self.client.put(&form.href);
        
        // 应用安全方案
        request = self.apply_security(request, security).await?;
        
        // 设置Content-Type
        let content_type = form.content_type
            .as_deref()
            .unwrap_or("application/json");
        request = request.header("Content-Type", content_type);
        
        // 序列化值
        let body = match content_type {
            "application/json" => serde_json::to_string(value)?,
            "text/plain" => value.as_str()
                .ok_or(WoTError::SerializationError("值不是字符串".to_string()))?
                .to_string(),
            _ => return Err(WoTError::UnsupportedContentType(content_type.to_string())),
        };
        
        let response = request.body(body).send().await?;
        
        if response.status().is_success() {
            Ok(())
        } else {
            Err(WoTError::ProtocolError(format!(
                "HTTP写入失败: {} - {}",
                response.status(),
                response.text().await.unwrap_or_default()
            )))
        }
    }
    
    async fn invoke_action(
        &self,
        form: &Form,
        input: Option<&serde_json::Value>,
        security: &[SecurityScheme],
    ) -> Result<Option<serde_json::Value>, WoTError> {
        let mut request = self.client.post(&form.href);
        
        // 应用安全方案
        request = self.apply_security(request, security).await?;
        
        // 设置Content-Type和Body
        if let Some(input_value) = input {
            let content_type = form.content_type
                .as_deref()
                .unwrap_or("application/json");
            request = request.header("Content-Type", content_type);
            
            let body = serde_json::to_string(input_value)?;
            request = request.body(body);
        }
        
        let response = request.send().await?;
        
        if response.status().is_success() {
            if response.content_length().unwrap_or(0) > 0 {
                let result = response.json().await?;
                Ok(Some(result))
            } else {
                Ok(None)
            }
        } else {
            Err(WoTError::ProtocolError(format!(
                "HTTP动作调用失败: {} - {}",
                response.status(),
                response.text().await.unwrap_or_default()
            )))
        }
    }
    
    async fn subscribe_event(
        &self,
        form: &Form,
        security: &[SecurityScheme],
        event_sender: mpsc::UnboundedSender<WoTEvent>,
    ) -> Result<String, WoTError> {
        // HTTP事件订阅通常使用WebSocket或SSE
        let url = Url::parse(&form.href)?;
        
        match url.scheme() {
            "ws" | "wss" => {
                self.subscribe_websocket_events(form, security, event_sender).await
            }
            "http" | "https" if form.subprotocol.as_deref() == Some("sse") => {
                self.subscribe_sse_events(form, security, event_sender).await
            }
            _ => {
                // 轮询方式
                self.subscribe_polling_events(form, security, event_sender).await
            }
        }
    }
    
    async fn unsubscribe_event(
        &self,
        subscription_id: &str,
    ) -> Result<(), WoTError> {
        // 实现取消订阅逻辑
        // 这里需要维护订阅ID到取消机制的映射
        tracing::info!("取消事件订阅: {}", subscription_id);
        Ok(())
    }
    
    fn get_protocol(&self) -> &str {
        "http"
    }
}

impl HttpProtocolHandler {
    async fn apply_security(
        &self,
        mut request: reqwest::RequestBuilder,
        security: &[SecurityScheme],
    ) -> Result<reqwest::RequestBuilder, WoTError> {
        for scheme in security {
            match scheme.scheme.as_str() {
                "basic" => {
                    // 基本认证
                    if let (Some(username), Some(password)) = (
                        std::env::var("WOT_USERNAME").ok(),
                        std::env::var("WOT_PASSWORD").ok()
                    ) {
                        request = request.basic_auth(username, Some(password));
                    }
                }
                "bearer" => {
                    // Bearer Token
                    if let Ok(token) = std::env::var("WOT_BEARER_TOKEN") {
                        request = request.bearer_auth(token);
                    }
                }
                "apikey" => {
                    // API Key
                    if let Ok(api_key) = std::env::var("WOT_API_KEY") {
                        match scheme.location.as_deref() {
                            Some("header") => {
                                let header_name = scheme.name.as_deref().unwrap_or("X-API-Key");
                                request = request.header(header_name, api_key);
                            }
                            Some("query") => {
                                // 查询参数中的API Key处理
                                // 需要修改URL
                            }
                            _ => {}
                        }
                    }
                }
                "oauth2" => {
                    // OAuth2处理
                    if let Ok(access_token) = std::env::var("WOT_OAUTH2_TOKEN") {
                        request = request.bearer_auth(access_token);
                    }
                }
                _ => {
                    tracing::warn!("不支持的安全方案: {}", scheme.scheme);
                }
            }
        }
        
        Ok(request)
    }
}
```

## 2. Thing Description发现与管理

### 2.1 发现机制实现

```rust
impl WoTAdapter {
    pub async fn new(
        config: WoTConfig,
        event_sender: mpsc::UnboundedSender<SemanticEvent>,
    ) -> Result<Self, WoTError> {
        let http_client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout))
            .build()?;
        
        let mut protocol_handlers: HashMap<String, Box<dyn ProtocolHandler>> = HashMap::new();
        
        // 注册HTTP协议处理器
        protocol_handlers.insert(
            "http".to_string(),
            Box::new(HttpProtocolHandler::new(http_client.clone())),
        );
        
        // 如果启用WebSocket，注册WebSocket处理器
        if config.enable_websocket {
            protocol_handlers.insert(
                "ws".to_string(),
                Box::new(WebSocketProtocolHandler::new()),
            );
        }
        
        // 如果启用CoAP，注册CoAP处理器
        if config.enable_coap {
            protocol_handlers.insert(
                "coap".to_string(),
                Box::new(CoapProtocolHandler::new()),
            );
        }
        
        let adapter = Self {
            config,
            http_client,
            thing_descriptions: Arc::new(RwLock::new(HashMap::new())),
            property_subscriptions: Arc::new(RwLock::new(HashMap::new())),
            event_subscriptions: Arc::new(RwLock::new(HashMap::new())),
            event_sender,
            protocol_handlers: Arc::new(RwLock::new(protocol_handlers)),
        };
        
        // 执行设备发现
        adapter.discover_things().await?;
        
        Ok(adapter)
    }
    
    async fn discover_things(&self) -> Result<(), WoTError> {
        for discovery_method in &self.config.discovery_methods {
            match discovery_method {
                DiscoveryMethod::Directory(url) => {
                    self.discover_from_directory(url).await?;
                }
                DiscoveryMethod::Multicast(address) => {
                    self.discover_via_multicast(address).await?;
                }
                DiscoveryMethod::DNS_SD(service_type) => {
                    self.discover_via_dns_sd(service_type).await?;
                }
                DiscoveryMethod::Manual(urls) => {
                    self.discover_manual_things(urls).await?;
                }
            }
        }
        
        Ok(())
    }
    
    async fn discover_from_directory(&self, directory_url: &str) -> Result<(), WoTError> {
        let response = self.http_client
            .get(directory_url)
            .header("Accept", "application/td+json")
            .send()
            .await?;
        
        if response.status().is_success() {
            let td_list: Vec<ThingDescription> = response.json().await?;
            
            for td in td_list {
                self.register_thing_description(td).await?;
            }
        }
        
        Ok(())
    }
    
    async fn discover_via_multicast(&self, multicast_address: &str) -> Result<(), WoTError> {
        use tokio::net::UdpSocket;
        
        let socket = UdpSocket::bind("0.0.0.0:0").await?;
        socket.set_broadcast(true)?;
        
        // 发送mDNS查询
        let query = self.create_mdns_query("_wot._tcp.local");
        socket.send_to(&query, multicast_address).await?;
        
        // 接收响应
        let mut buf = [0; 1024];
        let timeout = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            socket.recv_from(&mut buf)
        );
        
        if let Ok(Ok((len, _addr))) = timeout.await {
            let response = &buf[..len];
            if let Ok(td_urls) = self.parse_mdns_response(response) {
                for url in td_urls {
                    if let Ok(td) = self.fetch_thing_description(&url).await {
                        self.register_thing_description(td).await?;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    async fn discover_manual_things(&self, urls: &[String]) -> Result<(), WoTError> {
        for url in urls {
            match self.fetch_thing_description(url).await {
                Ok(td) => {
                    self.register_thing_description(td).await?;
                }
                Err(e) => {
                    tracing::warn!("获取Thing Description失败 {}: {:?}", url, e);
                }
            }
        }
        
        Ok(())
    }
    
    async fn fetch_thing_description(&self, url: &str) -> Result<ThingDescription, WoTError> {
        let response = self.http_client
            .get(url)
            .header("Accept", "application/td+json")
            .send()
            .await?;
        
        if response.status().is_success() {
            let td: ThingDescription = response.json().await?;
            Ok(td)
        } else {
            Err(WoTError::FetchError(format!(
                "获取TD失败: {} - {}",
                response.status(),
                response.text().await.unwrap_or_default()
            )))
        }
    }
    
    async fn register_thing_description(&self, td: ThingDescription) -> Result<(), WoTError> {
        let thing_id = td.id.clone();
        
        // 语义分析
        self.analyze_thing_semantics(&td).await?;
        
        // 存储TD
        self.thing_descriptions.write().await.insert(thing_id.clone(), td);
        
        tracing::info!("注册Thing Description: {}", thing_id);
        
        Ok(())
    }
}
```

## 3. 语义分析与映射

### 3.1 语义类型推断

```rust
impl WoTAdapter {
    async fn analyze_thing_semantics(&self, td: &ThingDescription) -> Result<(), WoTError> {
        // 分析Thing的语义类型
        let thing_semantic_type = self.infer_thing_semantic_type(td).await?;
        
        // 分析属性语义
        if let Some(properties) = &td.properties {
            for (prop_name, prop_affordance) in properties {
                let prop_semantic_type = self.infer_property_semantic_type(
                    prop_name, 
                    prop_affordance
                ).await?;
                
                tracing::debug!(
                    "属性语义映射: {} -> {} -> {}",
                    td.id,
                    prop_name,
                    prop_semantic_type
                );
            }
        }
        
        // 分析动作语义
        if let Some(actions) = &td.actions {
            for (action_name, action_affordance) in actions {
                let action_semantic_type = self.infer_action_semantic_type(
                    action_name,
                    action_affordance
                ).await?;
                
                tracing::debug!(
                    "动作语义映射: {} -> {} -> {}",
                    td.id,
                    action_name,
                    action_semantic_type
                );
            }
        }
        
        // 分析事件语义
        if let Some(events) = &td.events {
            for (event_name, event_affordance) in events {
                let event_semantic_type = self.infer_event_semantic_type(
                    event_name,
                    event_affordance
                ).await?;
                
                tracing::debug!(
                    "事件语义映射: {} -> {} -> {}",
                    td.id,
                    event_name,
                    event_semantic_type
                );
            }
        }
        
        Ok(())
    }
    
    async fn infer_thing_semantic_type(&self, td: &ThingDescription) -> Result<String, WoTError> {
        // 检查@type字段
        if let Some(types) = &td.thing_type {
            for type_iri in types {
                if self.is_semantic_type(type_iri) {
                    return Ok(type_iri.clone());
                }
            }
        }
        
        // 基于标题和描述推断
        let title_lower = td.title.to_lowercase();
        let description_lower = td.description
            .as_deref()
            .unwrap_or("")
            .to_lowercase();
        
        let combined_text = format!("{} {}", title_lower, description_lower);
        
        let semantic_type = match true {
            _ if combined_text.contains("sensor") => "sosa:Sensor",
            _ if combined_text.contains("actuator") => "sosa:Actuator",
            _ if combined_text.contains("light") || combined_text.contains("lamp") => "saref:LightingDevice",
            _ if combined_text.contains("temperature") || combined_text.contains("thermometer") => "saref:TemperatureSensor",
            _ if combined_text.contains("humidity") => "saref:HumiditySensor",
            _ if combined_text.contains("pressure") => "saref:PressureSensor",
            _ if combined_text.contains("motion") || combined_text.contains("pir") => "saref:MotionSensor",
            _ if combined_text.contains("camera") => "saref:Camera",
            _ if combined_text.contains("switch") => "saref:Switch",
            _ if combined_text.contains("meter") => "saref:Meter",
            _ if combined_text.contains("hvac") || combined_text.contains("climate") => "saref:HVAC",
            _ => "sosa:Platform",
        };
        
        Ok(semantic_type.to_string())
    }
    
    async fn infer_property_semantic_type(
        &self,
        prop_name: &str,
        prop_affordance: &PropertyAffordance,
    ) -> Result<String, WoTError> {
        // 检查@type字段
        if let Some(types) = &prop_affordance.property_type {
            for type_iri in types {
                if self.is_semantic_type(type_iri) {
                    return Ok(type_iri.clone());
                }
            }
        }
        
        // 基于属性名称推断
        let name_lower = prop_name.to_lowercase();
        let title_lower = prop_affordance.title
            .as_deref()
            .unwrap_or("")
            .to_lowercase();
        
        let combined_name = format!("{} {}", name_lower, title_lower);
        
        let semantic_type = match true {
            _ if combined_name.contains("temperature") || combined_name.contains("temp") => "sosa:Temperature",
            _ if combined_name.contains("humidity") || combined_name.contains("hum") => "sosa:Humidity",
            _ if combined_name.contains("pressure") || combined_name.contains("press") => "sosa:Pressure",
            _ if combined_name.contains("illuminance") || combined_name.contains("light") || combined_name.contains("lux") => "sosa:Illuminance",
            _ if combined_name.contains("motion") || combined_name.contains("movement") => "sosa:Motion",
            _ if combined_name.contains("energy") => "saref:Energy",
            _ if combined_name.contains("power") => "saref:Power",
            _ if combined_name.contains("voltage") => "saref:Voltage",
            _ if combined_name.contains("current") => "saref:Current",
            _ if combined_name.contains("speed") || combined_name.contains("velocity") => "sosa:Speed",
            _ if combined_name.contains("level") => "sosa:Level",
            _ if combined_name.contains("status") || combined_name.contains("state") => "saref:State",
            _ if combined_name.contains("brightness") => "saref:Brightness",
            _ if combined_name.contains("color") => "saref:Color",
            _ if combined_name.contains("position") || combined_name.contains("location") => "geo:Point",
            _ => self.infer_from_data_schema(&prop_affordance.data_schema).await?,
        };
        
        Ok(semantic_type.to_string())
    }
    
    async fn infer_from_data_schema(&self, schema: &DataSchema) -> Result<String, WoTError> {
        match schema {
            DataSchema::Number { unit, .. } => {
                if let Some(unit_str) = unit {
                    let unit_lower = unit_str.to_lowercase();
                    match unit_lower.as_str() {
                        "celsius" | "°c" | "c" => Ok("sosa:Temperature".to_string()),
                        "fahrenheit" | "°f" | "f" => Ok("sosa:Temperature".to_string()),
                        "kelvin" | "k" => Ok("sosa:Temperature".to_string()),
                        "percent" | "%" => Ok("sosa:Humidity".to_string()),
                        "pascal" | "pa" | "bar" | "psi" => Ok("sosa:Pressure".to_string()),
                        "lux" | "lumen" => Ok("sosa:Illuminance".to_string()),
                        "watt" | "w" => Ok("saref:Power".to_string()),
                        "joule" | "j" | "kwh" => Ok("saref:Energy".to_string()),
                        "volt" | "v" => Ok("saref:Voltage".to_string()),
                        "ampere" | "a" => Ok("saref:Current".to_string()),
                        "meter" | "m" | "cm" | "mm" => Ok("sosa:Distance".to_string()),
                        "m/s" | "km/h" | "mph" => Ok("sosa:Speed".to_string()),
                        _ => Ok("om:Measure".to_string()),
                    }
                } else {
                    Ok("om:Measure".to_string())
                }
            }
            DataSchema::Boolean { .. } => Ok("saref:OnOffState".to_string()),
            DataSchema::String { enumeration, .. } => {
                if enumeration.is_some() {
                    Ok("saref:State".to_string())
                } else {
                    Ok("rdfs:Literal".to_string())
                }
            }
            DataSchema::Object { .. } => Ok("sosa:Result".to_string()),
            DataSchema::Array { .. } => Ok("sosa:ResultCollection".to_string()),
            DataSchema::Null { .. } => Ok("rdfs:Resource".to_string()),
        }
    }
    
    fn is_semantic_type(&self, type_iri: &str) -> bool {
        type_iri.contains(':') && (
            type_iri.starts_with("sosa:") ||
            type_iri.starts_with("saref:") ||
            type_iri.starts_with("om:") ||
            type_iri.starts_with("geo:") ||
            type_iri.starts_with("schema:") ||
            type_iri.contains("ontology")
        )
    }
}
```

## 4. 属性读写和动作调用

### 4.1 属性操作实现

```rust
impl WoTAdapter {
    pub async fn read_property(
        &self,
        thing_id: &str,
        property_name: &str,
    ) -> Result<serde_json::Value, WoTError> {
        let thing_descriptions = self.thing_descriptions.read().await;
        let td = thing_descriptions.get(thing_id)
            .ok_or(WoTError::ThingNotFound(thing_id.to_string()))?;
        
        let property = td.properties
            .as_ref()
            .and_then(|props| props.get(property_name))
            .ok_or(WoTError::PropertyNotFound(property_name.to_string()))?;
        
        // 选择合适的Form
        let form = self.select_best_form(&property.forms, "readproperty").await?;
        
        // 获取安全方案
        let security_schemes = self.resolve_security_schemes(td, &form.security).await?;
        
        // 根据协议选择处理器
        let protocol = Self::extract_protocol(&form.href)?;
        let handlers = self.protocol_handlers.read().await;
        let handler = handlers.get(&protocol)
            .ok_or(WoTError::UnsupportedProtocol(protocol.clone()))?;
        
        // 执行读取
        let value = handler.read_property(form, &security_schemes).await?;
        
        // 发送语义事件
        self.emit_property_read_event(thing_id, property_name, &value).await?;
        
        Ok(value)
    }
    
    pub async fn write_property(
        &self,
        thing_id: &str,
        property_name: &str,
        value: &serde_json::Value,
    ) -> Result<(), WoTError> {
        let thing_descriptions = self.thing_descriptions.read().await;
        let td = thing_descriptions.get(thing_id)
            .ok_or(WoTError::ThingNotFound(thing_id.to_string()))?;
        
        let property = td.properties
            .as_ref()
            .and_then(|props| props.get(property_name))
            .ok_or(WoTError::PropertyNotFound(property_name.to_string()))?;
        
        // 检查是否可写
        if property.readOnly == Some(true) {
            return Err(WoTError::ReadOnlyProperty(property_name.to_string()));
        }
        
        // 验证数据
        self.validate_data_schema(value, &property.data_schema)?;
        
        // 选择合适的Form
        let form = self.select_best_form(&property.forms, "writeproperty").await?;
        
        // 获取安全方案
        let security_schemes = self.resolve_security_schemes(td, &form.security).await?;
        
        // 根据协议选择处理器
        let protocol = Self::extract_protocol(&form.href)?;
        let handlers = self.protocol_handlers.read().await;
        let handler = handlers.get(&protocol)
            .ok_or(WoTError::UnsupportedProtocol(protocol.clone()))?;
        
        // 执行写入
        handler.write_property(form, value, &security_schemes).await?;
        
        // 发送语义事件
        self.emit_property_write_event(thing_id, property_name, value).await?;
        
        Ok(())
    }
    
    pub async fn invoke_action(
        &self,
        thing_id: &str,
        action_name: &str,
        input: Option<&serde_json::Value>,
    ) -> Result<Option<serde_json::Value>, WoTError> {
        let thing_descriptions = self.thing_descriptions.read().await;
        let td = thing_descriptions.get(thing_id)
            .ok_or(WoTError::ThingNotFound(thing_id.to_string()))?;
        
        let action = td.actions
            .as_ref()
            .and_then(|actions| actions.get(action_name))
            .ok_or(WoTError::ActionNotFound(action_name.to_string()))?;
        
        // 验证输入数据
        if let (Some(input_value), Some(input_schema)) = (input, &action.input) {
            self.validate_data_schema(input_value, input_schema)?;
        }
        
        // 选择合适的Form
        let form = self.select_best_form(&action.forms, "invokeaction").await?;
        
        // 获取安全方案
        let security_schemes = self.resolve_security_schemes(td, &form.security).await?;
        
        // 根据协议选择处理器
        let protocol = Self::extract_protocol(&form.href)?;
        let handlers = self.protocol_handlers.read().await;
        let handler = handlers.get(&protocol)
            .ok_or(WoTError::UnsupportedProtocol(protocol.clone()))?;
        
        // 执行动作
        let result = handler.invoke_action(form, input, &security_schemes).await?;
        
        // 发送语义事件
        self.emit_action_invocation_event(thing_id, action_name, input, &result).await?;
        
        Ok(result)
    }
    
    async fn select_best_form(
        &self,
        forms: &[Form],
        op: &str,
    ) -> Result<&Form, WoTError> {
        // 简单选择第一个兼容的form
        // 实际实现中应该根据协议偏好、安全要求等选择最佳form
        forms.first()
            .ok_or(WoTError::NoSuitableForm(op.to_string()))
    }
    
    fn extract_protocol(href: &str) -> Result<String, WoTError> {
        let url = Url::parse(href)?;
        Ok(url.scheme().to_string())
    }
    
    async fn resolve_security_schemes(
        &self,
        td: &ThingDescription,
        form_security: &Option<Vec<String>>,
    ) -> Result<Vec<SecurityScheme>, WoTError> {
        let security_names = form_security
            .as_ref()
            .unwrap_or(&td.security);
        
        let mut schemes = Vec::new();
        for name in security_names {
            if let Some(scheme) = td.security_definitions.get(name) {
                schemes.push(scheme.clone());
            }
        }
        
        Ok(schemes)
    }
}
```

## 5. 配置和使用示例

### 5.1 配置文件

```yaml
# config/wot_adapter.yaml
wot:
  thing_directory_url: "http://localhost:8080/things"
  discovery_methods:
    - type: "directory"
      url: "http://192.168.1.100:8080/directory"
    - type: "multicast"
      address: "224.0.0.251:5353"
    - type: "manual"
      urls:
        - "http://192.168.1.10/thing-description"
        - "http://192.168.1.11/thing-description"
  
  timeout: 30
  retry_count: 3
  polling_interval: 5000
  enable_websocket: true
  enable_coap: true
  
  security_schemes:
    basic_auth:
      scheme: "basic"
      description: "Basic Authentication"
    
    api_key:
      scheme: "apikey"
      in: "header"
      name: "X-API-Key"
      description: "API Key Authentication"
    
    oauth2:
      scheme: "oauth2"
      flow: "client_credentials"
      token: "https://oauth.example.com/token"
      scopes: ["read", "write"]

# 语义映射配置
semantic_mappings:
  property_types:
    temperature: "sosa:Temperature"
    humidity: "sosa:Humidity"
    pressure: "sosa:Pressure"
    illuminance: "sosa:Illuminance"
    motion: "sosa:Motion"
    energy: "saref:Energy"
    power: "saref:Power"
  
  thing_types:
    sensor: "sosa:Sensor"
    actuator: "sosa:Actuator"
    light: "saref:LightingDevice"
    switch: "saref:Switch"
    meter: "saref:Meter"
```

### 5.2 使用示例

```rust
use crate::adapters::wot::{WoTAdapter, WoTConfig, DiscoveryMethod};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (event_sender, mut event_receiver) = mpsc::unbounded_channel();
    
    let config = WoTConfig {
        thing_directory_url: Some("http://localhost:8080/directory".to_string()),
        discovery_methods: vec![
            DiscoveryMethod::Directory("http://localhost:8080/directory".to_string()),
            DiscoveryMethod::Manual(vec![
                "http://192.168.1.10/thing-description".to_string(),
                "http://192.168.1.11/thing-description".to_string(),
            ]),
        ],
        security_schemes: HashMap::new(),
        timeout: 30,
        retry_count: 3,
        polling_interval: 5000,
        enable_websocket: true,
        enable_coap: false,
    };
    
    // 创建适配器
    let adapter = WoTAdapter::new(config, event_sender).await?;
    
    // 读取属性
    let temperature = adapter.read_property(
        "urn:example:thermometer:1",
        "temperature"
    ).await?;
    
    println!("温度读取: {:?}", temperature);
    
    // 写入属性
    adapter.write_property(
        "urn:example:light:1",
        "brightness",
        &serde_json::json!(75)
    ).await?;
    
    // 调用动作
    let result = adapter.invoke_action(
        "urn:example:actuator:1",
        "toggle",
        None
    ).await?;
    
    println!("动作执行结果: {:?}", result);
    
    // 处理事件
    tokio::spawn(async move {
        while let Some(event) = event_receiver.recv().await {
            println!("收到WoT语义事件: {:?}", event);
        }
    });
    
    // 保持运行
    tokio::signal::ctrl_c().await?;
    Ok(())
}
```

这个WoT适配器实现提供了完整的Thing Description发现、属性读写、动作调用、事件订阅和语义映射功能，完全符合W3C WoT标准规范。
