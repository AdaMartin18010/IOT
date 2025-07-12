# Matter适配器完整实现

## 1. 核心架构

```rust
// src/adapters/matter/mod.rs
use tokio::sync::{mpsc, RwLock};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct MatterConfig {
    pub commissioner_node_id: u64,
    pub fabric_id: u64,
    pub vendor_id: u16,
    pub product_id: u16,
    pub discriminator: u16,
    pub passcode: u32,
    pub port: u16,
    pub interface: String,
    pub storage_path: String,
}

#[derive(Debug)]
pub struct MatterAdapter {
    config: MatterConfig,
    commissioning_client: Arc<RwLock<Option<CommissioningClient>>>,
    device_registry: Arc<RwLock<HashMap<u64, MatterDevice>>>,
    cluster_handlers: Arc<RwLock<HashMap<u32, Box<dyn ClusterHandler>>>>,
    event_sender: mpsc::UnboundedSender<SemanticEvent>,
}

#[derive(Debug, Clone)]
pub struct MatterDevice {
    pub node_id: u64,
    pub vendor_id: u16,
    pub product_id: u16,
    pub device_type: u32,
    pub endpoints: Vec<Endpoint>,
    pub semantic_type: String,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Endpoint {
    pub endpoint_id: u16,
    pub device_type: u32,
    pub clusters: Vec<ClusterInfo>,
}

#[derive(Debug, Clone)]
pub struct ClusterInfo {
    pub cluster_id: u32,
    pub revision: u16,
    pub attributes: Vec<AttributeInfo>,
    pub commands: Vec<CommandInfo>,
    pub events: Vec<EventInfo>,
}
```

## 2. 设备发现与配网

```rust
impl MatterAdapter {
    pub async fn new(
        config: MatterConfig,
        event_sender: mpsc::UnboundedSender<SemanticEvent>,
    ) -> Result<Self, MatterError> {
        let adapter = Self {
            config,
            commissioning_client: Arc::new(RwLock::new(None)),
            device_registry: Arc::new(RwLock::new(HashMap::new())),
            cluster_handlers: Arc::new(RwLock::new(HashMap::new())),
            event_sender,
        };
        
        adapter.initialize().await?;
        Ok(adapter)
    }
    
    async fn initialize(&self) -> Result<(), MatterError> {
        // 初始化Matter栈
        let client = CommissioningClient::new(&self.config).await?;
        *self.commissioning_client.write().await = Some(client);
        
        // 注册集群处理器
        self.register_cluster_handlers().await;
        
        // 发现已配网设备
        self.discover_commissioned_devices().await?;
        
        Ok(())
    }
    
    pub async fn commission_device(
        &self,
        setup_code: &str,
        device_address: &str,
    ) -> Result<u64, MatterError> {
        let client_guard = self.commissioning_client.read().await;
        let client = client_guard.as_ref()
            .ok_or(MatterError::NotInitialized)?;
        
        // 执行配网流程
        let node_id = client.commission_device(setup_code, device_address).await?;
        
        // 读取设备信息
        let device_info = self.read_device_information(node_id).await?;
        
        // 语义分析
        let semantic_type = self.infer_device_semantics(&device_info).await?;
        
        let matter_device = MatterDevice {
            node_id,
            vendor_id: device_info.vendor_id,
            product_id: device_info.product_id,
            device_type: device_info.device_type,
            endpoints: device_info.endpoints,
            semantic_type,
            capabilities: self.extract_capabilities(&device_info).await?,
        };
        
        // 注册设备
        self.device_registry.write().await.insert(node_id, matter_device);
        
        // 订阅设备事件
        self.subscribe_device_events(node_id).await?;
        
        Ok(node_id)
    }
    
    async fn infer_device_semantics(&self, device_info: &DeviceInfo) -> Result<String, MatterError> {
        // Matter设备类型到语义类型的映射
        let semantic_type = match device_info.device_type {
            0x0100 => "saref:LightingDevice",      // On/Off Light
            0x0101 => "saref:DimmableLight",       // Dimmable Light
            0x010C => "saref:ColorLight",          // Color Temperature Light
            0x010D => "saref:ColorLight",          // Extended Color Light
            0x0302 => "sosa:TemperatureSensor",    // Temperature Sensor
            0x0307 => "sosa:OccupancySensor",      // Occupancy Sensor
            0x0015 => "saref:ContactSensor",       // Contact Sensor
            0x0103 => "saref:Switch",              // On/Off Switch
            0x0104 => "saref:DimmerSwitch",        // Dimmer Switch
            0x010A => "saref:DoorLock",            // Door Lock
            0x0301 => "saref:Thermostat",          // Thermostat
            0x0300 => "saref:HVAC",                // Heating/Cooling Unit
            0x0840 => "saref:EnergyMeter",         // Electrical Sensor
            _ => "sosa:Platform",
        };
        
        Ok(semantic_type.to_string())
    }
}
```

## 3. 集群处理器

```rust
#[async_trait::async_trait]
pub trait ClusterHandler: Send + Sync {
    async fn handle_attribute_read(
        &self,
        node_id: u64,
        endpoint_id: u16,
        attribute_id: u32,
        value: &[u8],
    ) -> Result<SemanticEvent, MatterError>;
    
    async fn handle_command(
        &self,
        node_id: u64,
        endpoint_id: u16,
        command_id: u32,
        payload: &[u8],
    ) -> Result<(), MatterError>;
    
    async fn handle_event(
        &self,
        node_id: u64,
        endpoint_id: u16,
        event_id: u32,
        data: &[u8],
    ) -> Result<SemanticEvent, MatterError>;
    
    fn get_cluster_id(&self) -> u32;
}

// On/Off集群处理器
#[derive(Debug)]
pub struct OnOffClusterHandler;

#[async_trait::async_trait]
impl ClusterHandler for OnOffClusterHandler {
    async fn handle_attribute_read(
        &self,
        node_id: u64,
        endpoint_id: u16,
        attribute_id: u32,
        value: &[u8],
    ) -> Result<SemanticEvent, MatterError> {
        match attribute_id {
            0x0000 => { // OnOff attribute
                let on_off = value[0] != 0;
                Ok(SemanticEvent {
                    event_id: uuid::Uuid::new_v4().to_string(),
                    timestamp: chrono::Utc::now(),
                    source: "matter".to_string(),
                    semantic_path: format!("/matter/{}/endpoint/{}/onoff", node_id, endpoint_id),
                    value: serde_json::json!(on_off),
                    quality: Some("good".to_string()),
                    metadata: Self::create_metadata(node_id, endpoint_id, "OnOff"),
                })
            }
            _ => Err(MatterError::UnsupportedAttribute(attribute_id)),
        }
    }
    
    async fn handle_command(
        &self,
        node_id: u64,
        endpoint_id: u16,
        command_id: u32,
        _payload: &[u8],
    ) -> Result<(), MatterError> {
        match command_id {
            0x00 => tracing::info!("设备关闭: {}/{}", node_id, endpoint_id),
            0x01 => tracing::info!("设备开启: {}/{}", node_id, endpoint_id),
            0x02 => tracing::info!("设备切换: {}/{}", node_id, endpoint_id),
            _ => return Err(MatterError::UnsupportedCommand(command_id)),
        }
        Ok(())
    }
    
    async fn handle_event(
        &self,
        node_id: u64,
        endpoint_id: u16,
        event_id: u32,
        data: &[u8],
    ) -> Result<SemanticEvent, MatterError> {
        match event_id {
            0x00 => { // Startup event
                Ok(SemanticEvent {
                    event_id: uuid::Uuid::new_v4().to_string(),
                    timestamp: chrono::Utc::now(),
                    source: "matter".to_string(),
                    semantic_path: format!("/matter/{}/endpoint/{}/startup", node_id, endpoint_id),
                    value: serde_json::json!("startup"),
                    quality: Some("good".to_string()),
                    metadata: Self::create_metadata(node_id, endpoint_id, "Startup"),
                })
            }
            _ => Err(MatterError::UnsupportedEvent(event_id)),
        }
    }
    
    fn get_cluster_id(&self) -> u32 {
        0x0006 // On/Off Cluster
    }
}

// Level Control集群处理器
#[derive(Debug)]
pub struct LevelControlClusterHandler;

#[async_trait::async_trait]
impl ClusterHandler for LevelControlClusterHandler {
    async fn handle_attribute_read(
        &self,
        node_id: u64,
        endpoint_id: u16,
        attribute_id: u32,
        value: &[u8],
    ) -> Result<SemanticEvent, MatterError> {
        match attribute_id {
            0x0000 => { // CurrentLevel
                let level = value[0] as f32 / 254.0 * 100.0; // 转换为百分比
                Ok(SemanticEvent {
                    event_id: uuid::Uuid::new_v4().to_string(),
                    timestamp: chrono::Utc::now(),
                    source: "matter".to_string(),
                    semantic_path: format!("/matter/{}/endpoint/{}/level", node_id, endpoint_id),
                    value: serde_json::json!(level),
                    quality: Some("good".to_string()),
                    metadata: Self::create_metadata(node_id, endpoint_id, "Level"),
                })
            }
            _ => Err(MatterError::UnsupportedAttribute(attribute_id)),
        }
    }
    
    async fn handle_command(
        &self,
        node_id: u64,
        endpoint_id: u16,
        command_id: u32,
        payload: &[u8],
    ) -> Result<(), MatterError> {
        match command_id {
            0x00 => { // MoveToLevel
                let level = payload[0];
                tracing::info!("设置亮度: {}/{} -> {}", node_id, endpoint_id, level);
            }
            0x01 => tracing::info!("移动到指定亮度: {}/{}", node_id, endpoint_id),
            _ => return Err(MatterError::UnsupportedCommand(command_id)),
        }
        Ok(())
    }
    
    async fn handle_event(
        &self,
        _node_id: u64,
        _endpoint_id: u16,
        event_id: u32,
        _data: &[u8],
    ) -> Result<SemanticEvent, MatterError> {
        Err(MatterError::UnsupportedEvent(event_id))
    }
    
    fn get_cluster_id(&self) -> u32 {
        0x0008 // Level Control Cluster
    }
}
```

## 4. 属性读写和命令执行

```rust
impl MatterAdapter {
    pub async fn read_attribute(
        &self,
        node_id: u64,
        endpoint_id: u16,
        cluster_id: u32,
        attribute_id: u32,
    ) -> Result<serde_json::Value, MatterError> {
        let client_guard = self.commissioning_client.read().await;
        let client = client_guard.as_ref()
            .ok_or(MatterError::NotInitialized)?;
        
        // 执行属性读取
        let attribute_path = AttributePath {
            node_id,
            endpoint_id,
            cluster_id,
            attribute_id,
        };
        
        let raw_value = client.read_attribute(&attribute_path).await?;
        
        // 使用集群处理器处理数据
        let handlers = self.cluster_handlers.read().await;
        if let Some(handler) = handlers.get(&cluster_id) {
            let semantic_event = handler.handle_attribute_read(
                node_id,
                endpoint_id, 
                attribute_id,
                &raw_value
            ).await?;
            
            // 发送语义事件
            if let Err(e) = self.event_sender.send(semantic_event.clone()) {
                tracing::error!("发送语义事件失败: {:?}", e);
            }
            
            return Ok(semantic_event.value);
        }
        
        // 默认处理
        Ok(serde_json::json!(raw_value))
    }
    
    pub async fn write_attribute(
        &self,
        node_id: u64,
        endpoint_id: u16,
        cluster_id: u32,
        attribute_id: u32,
        value: &serde_json::Value,
    ) -> Result<(), MatterError> {
        let client_guard = self.commissioning_client.read().await;
        let client = client_guard.as_ref()
            .ok_or(MatterError::NotInitialized)?;
        
        // 转换值为Matter格式
        let matter_value = self.convert_to_matter_value(value, cluster_id, attribute_id)?;
        
        let attribute_path = AttributePath {
            node_id,
            endpoint_id,
            cluster_id,
            attribute_id,
        };
        
        client.write_attribute(&attribute_path, &matter_value).await?;
        
        // 发送写入事件
        let semantic_event = SemanticEvent {
            event_id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            source: "matter".to_string(),
            semantic_path: format!(
                "/matter/{}/endpoint/{}/cluster/{}/attribute/{}",
                node_id, endpoint_id, cluster_id, attribute_id
            ),
            value: value.clone(),
            quality: Some("written".to_string()),
            metadata: HashMap::new(),
        };
        
        if let Err(e) = self.event_sender.send(semantic_event) {
            tracing::error!("发送写入事件失败: {:?}", e);
        }
        
        Ok(())
    }
    
    pub async fn invoke_command(
        &self,
        node_id: u64,
        endpoint_id: u16,
        cluster_id: u32,
        command_id: u32,
        payload: Option<&serde_json::Value>,
    ) -> Result<Option<serde_json::Value>, MatterError> {
        let client_guard = self.commissioning_client.read().await;
        let client = client_guard.as_ref()
            .ok_or(MatterError::NotInitialized)?;
        
        // 转换载荷
        let matter_payload = if let Some(payload_value) = payload {
            self.convert_to_matter_payload(payload_value, cluster_id, command_id)?
        } else {
            Vec::new()
        };
        
        let command_path = CommandPath {
            node_id,
            endpoint_id,
            cluster_id,
            command_id,
        };
        
        let response = client.invoke_command(&command_path, &matter_payload).await?;
        
        // 通知集群处理器
        let handlers = self.cluster_handlers.read().await;
        if let Some(handler) = handlers.get(&cluster_id) {
            handler.handle_command(node_id, endpoint_id, command_id, &matter_payload).await?;
        }
        
        // 发送命令调用事件
        let semantic_event = SemanticEvent {
            event_id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            source: "matter".to_string(),
            semantic_path: format!(
                "/matter/{}/endpoint/{}/cluster/{}/command/{}",
                node_id, endpoint_id, cluster_id, command_id
            ),
            value: payload.cloned().unwrap_or(serde_json::Value::Null),
            quality: Some("invoked".to_string()),
            metadata: HashMap::new(),
        };
        
        if let Err(e) = self.event_sender.send(semantic_event) {
            tracing::error!("发送命令事件失败: {:?}", e);
        }
        
        // 处理响应
        if response.is_empty() {
            Ok(None)
        } else {
            Ok(Some(serde_json::json!(response)))
        }
    }
}
```

## 5. 配置和使用示例

### 5.1 配置文件

```yaml
# config/matter_adapter.yaml
matter:
  commissioner_node_id: 1
  fabric_id: 1
  vendor_id: 0xFFF1
  product_id: 0x8000
  discriminator: 3840
  passcode: 20202021
  port: 5540
  interface: "eth0"
  storage_path: "/var/lib/matter"
  
  # 集群映射
  cluster_semantics:
    0x0006: "saref:OnOffFunction"     # On/Off
    0x0008: "saref:LevelControlFunction"  # Level Control
    0x0300: "saref:ColorControlFunction"  # Color Control
    0x0402: "sosa:TemperatureSensor"      # Temperature Measurement
    0x0405: "sosa:HumiditySensor"         # Relative Humidity Measurement
    0x0406: "sosa:OccupancySensor"        # Occupancy Sensing
```

### 5.2 使用示例

```rust
use crate::adapters::matter::{MatterAdapter, MatterConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (event_sender, mut event_receiver) = mpsc::unbounded_channel();
    
    let config = MatterConfig {
        commissioner_node_id: 1,
        fabric_id: 1,
        vendor_id: 0xFFF1,
        product_id: 0x8000,
        discriminator: 3840,
        passcode: 20202021,
        port: 5540,
        interface: "eth0".to_string(),
        storage_path: "/var/lib/matter".to_string(),
    };
    
    // 创建适配器
    let adapter = MatterAdapter::new(config, event_sender).await?;
    
    // 配网新设备
    let node_id = adapter.commission_device(
        "MT:XDHM-YW2T-D2EP",  // QR码或配对码
        "192.168.1.100:5540"
    ).await?;
    
    println!("设备配网成功，节点ID: {}", node_id);
    
    // 读取属性
    let on_off_state = adapter.read_attribute(
        node_id,
        1,      // endpoint
        0x0006, // On/Off cluster
        0x0000  // OnOff attribute
    ).await?;
    
    println!("开关状态: {:?}", on_off_state);
    
    // 控制设备
    adapter.invoke_command(
        node_id,
        1,      // endpoint
        0x0006, // On/Off cluster
        0x01,   // On command
        None
    ).await?;
    
    // 处理事件
    tokio::spawn(async move {
        while let Some(event) = event_receiver.recv().await {
            println!("收到Matter语义事件: {:?}", event);
        }
    });
    
    tokio::signal::ctrl_c().await?;
    Ok(())
}
```

这个Matter适配器实现提供了完整的设备配网、属性读写、命令调用和事件处理功能，支持标准Matter集群的语义映射。
