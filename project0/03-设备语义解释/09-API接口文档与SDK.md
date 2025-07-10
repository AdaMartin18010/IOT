# IoT设备语义解释系统API接口文档与SDK

## 1. API概述

### 1.1 API设计原则

- **RESTful架构**: 遵循REST设计规范
- **版本控制**: 支持API版本管理
- **统一响应**: 标准化的响应格式
- **错误处理**: 详细的错误码和错误信息
- **安全认证**: OAuth 2.0 / JWT认证
- **限流控制**: API调用频率限制

### 1.2 基础URL

```text
生产环境: https://api.iot-semantic.com/v1
测试环境: https://api-test.iot-semantic.com/v1
开发环境: http://localhost:8080/api/v1
```

### 1.3 认证方式

```http
Authorization: Bearer <JWT_TOKEN>
Content-Type: application/json
X-API-Key: <API_KEY>
```

## 2. 核心API接口

### 2.1 设备注册与发现

#### 注册设备

```http
POST /devices/register
```

**请求参数:**

```json
{
  "device_id": "device_001",
  "device_type": "temperature_sensor",
  "manufacturer": "Acme Corp",
  "model": "TempSens-2000",
  "firmware_version": "v1.2.3",
  "protocols": ["mqtt", "coap"],
  "capabilities": [
    {
      "name": "temperature_measurement",
      "unit": "celsius",
      "range": {"min": -40, "max": 125},
      "accuracy": 0.1
    }
  ],
  "semantic_profile": {
    "ontology_uri": "http://purl.oclc.org/NET/UNIS/fiware/iot-lite#",
    "device_class": "iot-lite:TemperatureSensor",
    "measurement_type": "sosa:Temperature"
  },
  "location": {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "description": "New York Office Building A, Floor 5"
  },
  "metadata": {
    "installation_date": "2024-01-15T10:30:00Z",
    "maintenance_interval": "P1M"
  }
}
```

**响应:**

```json
{
  "status": "success",
  "data": {
    "device_id": "device_001",
    "registration_id": "reg_12345",
    "semantic_id": "sem_67890",
    "registration_time": "2024-01-20T14:30:00Z",
    "assigned_capabilities": [
      {
        "capability_id": "cap_001",
        "semantic_mapping": {
          "concept": "http://purl.oclc.org/NET/UNIS/fiware/iot-lite#Temperature",
          "property": "sosa:hasValue",
          "unit": "qudt:DegreeCelsius"
        }
      }
    ]
  },
  "message": "Device registered successfully"
}
```

#### 设备发现

```http
GET /devices/discover?query={semantic_query}&location={location}&radius={radius}
```

**查询参数:**

- `query`: 语义查询字符串
- `location`: 位置坐标 (lat,lng)
- `radius`: 搜索半径 (km)
- `capabilities`: 能力过滤
- `protocols`: 协议过滤

**响应:**

```json
{
  "status": "success",
  "data": {
    "total_count": 25,
    "devices": [
      {
        "device_id": "device_001",
        "device_type": "temperature_sensor",
        "semantic_match_score": 0.95,
        "capabilities": ["temperature_measurement"],
        "location": {
          "latitude": 40.7128,
          "longitude": -74.0060,
          "distance": 0.5
        },
        "status": "online",
        "last_seen": "2024-01-20T14:28:00Z"
      }
    ]
  }
}
```

### 2.2 语义映射与转换

#### 语义映射

```http
POST /semantic/mapping
```

**请求参数:**

```json
{
  "source_protocol": "mqtt",
  "target_protocol": "opcua",
  "source_message": {
    "topic": "sensors/temp001/data",
    "payload": {
      "temperature": 23.5,
      "humidity": 65.2,
      "timestamp": "2024-01-20T14:30:00Z"
    }
  },
  "mapping_options": {
    "preserve_semantics": true,
    "validate_output": true,
    "include_metadata": true
  }
}
```

**响应:**

```json
{
  "status": "success",
  "data": {
    "mapping_id": "map_12345",
    "target_message": {
      "node_id": "ns=2;s=TempSensor001",
      "values": [
        {
          "attribute_id": 13,
          "value": {
            "type": "Double",
            "body": 23.5
          },
          "semantic_annotation": {
            "concept": "sosa:Temperature",
            "unit": "qudt:DegreeCelsius"
          }
        }
      ]
    },
    "mapping_confidence": 0.98,
    "transformation_time": 2.3,
    "semantic_preservation_score": 0.96
  }
}
```

#### 批量语义转换

```http
POST /semantic/batch-transform
```

**请求参数:**

```json
{
  "transformations": [
    {
      "transform_id": "trans_001",
      "source_format": "mqtt_json",
      "target_format": "opcua_binary",
      "source_data": "...",
      "semantic_context": {
        "device_type": "temperature_sensor",
        "measurement_context": "environmental_monitoring"
      }
    }
  ],
  "options": {
    "parallel_processing": true,
    "error_handling": "continue_on_error",
    "result_format": "detailed"
  }
}
```

### 2.3 语义推理

#### 设备能力推理

```http
POST /reasoning/device-capabilities
```

**请求参数:**

```json
{
  "device_data": {
    "device_id": "unknown_device_001",
    "observed_behaviors": [
      {
        "timestamp": "2024-01-20T14:30:00Z",
        "action": "data_transmission",
        "protocol": "mqtt",
        "payload_structure": {
          "temperature": "numeric",
          "humidity": "numeric",
          "battery_level": "percentage"
        }
      }
    ],
    "network_metadata": {
      "mac_address": "00:1B:44:11:3A:B7",
      "signal_strength": -45,
      "connection_type": "wifi"
    }
  },
  "reasoning_options": {
    "confidence_threshold": 0.7,
    "include_similar_devices": true,
    "max_inference_depth": 3
  }
}
```

**响应:**

```json
{
  "status": "success",
  "data": {
    "inferred_capabilities": [
      {
        "capability": "temperature_sensing",
        "confidence": 0.95,
        "evidence": [
          "payload contains temperature field",
          "numeric range typical for temperature sensors",
          "periodic transmission pattern"
        ],
        "semantic_mapping": {
          "concept": "sosa:Temperature",
          "measurement_property": "sosa:hasValue"
        }
      },
      {
        "capability": "humidity_sensing",
        "confidence": 0.88,
        "evidence": ["humidity field in payload", "value range 0-100%"]
      }
    ],
    "device_classification": {
      "primary_type": "environmental_sensor",
      "sub_types": ["temperature_sensor", "humidity_sensor"],
      "confidence": 0.92
    },
    "recommended_ontology": "http://purl.oclc.org/NET/UNIS/fiware/iot-lite#"
  }
}
```

#### 语义查询推理

```http
POST /reasoning/semantic-query
```

**请求参数:**

```json
{
  "query": {
    "type": "sparql",
    "query_string": "SELECT ?device ?temperature WHERE { ?device rdf:type iot-lite:TemperatureSensor . ?device sosa:hasValue ?temperature . FILTER(?temperature > 25) }",
    "reasoning_rules": [
      "infer_device_types",
      "calculate_derived_properties",
      "apply_unit_conversions"
    ]
  },
  "context": {
    "time_range": {
      "start": "2024-01-20T00:00:00Z",
      "end": "2024-01-20T23:59:59Z"
    },
    "location_filter": {
      "type": "polygon",
      "coordinates": [[40.7, -74.1], [40.8, -74.1], [40.8, -74.0], [40.7, -74.0]]
    }
  }
}
```

### 2.4 设备健康监测

#### 健康状态查询

```http
GET /devices/{device_id}/health
```

**响应:**

```json
{
  "status": "success",
  "data": {
    "device_id": "device_001",
    "overall_health": "good",
    "health_score": 0.87,
    "last_check": "2024-01-20T14:30:00Z",
    "metrics": {
      "connectivity": {
        "status": "connected",
        "signal_strength": -42,
        "last_heartbeat": "2024-01-20T14:29:45Z"
      },
      "performance": {
        "response_time": 234,
        "throughput": 125.5,
        "error_rate": 0.002
      },
      "hardware": {
        "cpu_usage": 23.5,
        "memory_usage": 45.2,
        "temperature": 35.7,
        "battery_level": 78
      }
    },
    "anomalies": [
      {
        "type": "performance_degradation",
        "severity": "low",
        "description": "Response time increased by 15% in last hour",
        "detected_at": "2024-01-20T13:45:00Z"
      }
    ],
    "predictions": {
      "failure_probability": 0.03,
      "estimated_remaining_life": "P6M",
      "maintenance_recommendation": "routine_check_in_2_weeks"
    }
  }
}
```

#### 批量健康监测

```http
POST /devices/health/batch-check
```

**请求参数:**

```json
{
  "device_ids": ["device_001", "device_002", "device_003"],
  "health_checks": [
    "connectivity",
    "performance",
    "anomaly_detection",
    "predictive_analysis"
  ],
  "options": {
    "include_historical_data": true,
    "prediction_horizon": "P1M"
  }
}
```

### 2.5 配置管理

#### 更新设备配置

```http
PUT /devices/{device_id}/config
```

**请求参数:**

```json
{
  "semantic_config": {
    "ontology_preference": "http://purl.oclc.org/NET/UNIS/fiware/iot-lite#",
    "mapping_rules": [
      {
        "source_field": "temp",
        "target_concept": "sosa:Temperature",
        "transformation": "celsius_to_kelvin"
      }
    ],
    "inference_rules": [
      "infer_device_status",
      "calculate_energy_efficiency"
    ]
  },
  "communication_config": {
    "preferred_protocol": "mqtt",
    "backup_protocols": ["coap", "http"],
    "message_format": "json",
    "compression": "gzip"
  },
  "monitoring_config": {
    "health_check_interval": 300,
    "anomaly_detection": true,
    "predictive_maintenance": true
  }
}
```

## 3. Python SDK

### 3.1 SDK安装与初始化

```python
# 安装SDK
pip install iot-semantic-sdk

# 初始化客户端
from iot_semantic import SemanticClient

client = SemanticClient(
    api_url="https://api.iot-semantic.com/v1",
    api_key="your_api_key",
    auth_token="your_jwt_token"
)
```

### 3.2 设备管理SDK

```python
class DeviceManager:
    def __init__(self, client):
        self.client = client
    
    async def register_device(self, device_info):
        """注册设备"""
        response = await self.client.post('/devices/register', device_info)
        return DeviceRegistration(response.data)
    
    async def discover_devices(self, query=None, location=None, **filters):
        """发现设备"""
        params = {'query': query, 'location': location, **filters}
        response = await self.client.get('/devices/discover', params=params)
        return [Device(device) for device in response.data['devices']]
    
    async def get_device_health(self, device_id):
        """获取设备健康状态"""
        response = await self.client.get(f'/devices/{device_id}/health')
        return DeviceHealth(response.data)
    
    async def update_device_config(self, device_id, config):
        """更新设备配置"""
        response = await self.client.put(f'/devices/{device_id}/config', config)
        return ConfigurationResult(response.data)

# 使用示例
async def main():
    device_manager = DeviceManager(client)
    
    # 注册新设备
    device_info = {
        "device_id": "temp_sensor_001",
        "device_type": "temperature_sensor",
        "capabilities": [
            {
                "name": "temperature_measurement",
                "unit": "celsius",
                "range": {"min": -40, "max": 125}
            }
        ]
    }
    
    registration = await device_manager.register_device(device_info)
    print(f"Device registered: {registration.device_id}")
    
    # 发现附近的设备
    devices = await device_manager.discover_devices(
        query="temperature sensor",
        location="40.7128,-74.0060",
        radius=5
    )
    
    for device in devices:
        print(f"Found device: {device.device_id} - {device.device_type}")
```

### 3.3 语义处理SDK

```python
class SemanticProcessor:
    def __init__(self, client):
        self.client = client
    
    async def map_protocol_message(self, source_protocol, target_protocol, message):
        """协议消息映射"""
        mapping_request = {
            "source_protocol": source_protocol,
            "target_protocol": target_protocol,
            "source_message": message
        }
        
        response = await self.client.post('/semantic/mapping', mapping_request)
        return ProtocolMapping(response.data)
    
    async def infer_device_capabilities(self, device_data, options=None):
        """推理设备能力"""
        inference_request = {
            "device_data": device_data,
            "reasoning_options": options or {}
        }
        
        response = await self.client.post(
            '/reasoning/device-capabilities', 
            inference_request
        )
        return CapabilityInference(response.data)
    
    async def semantic_query(self, query, context=None):
        """语义查询"""
        query_request = {
            "query": query,
            "context": context or {}
        }
        
        response = await self.client.post('/reasoning/semantic-query', query_request)
        return QueryResult(response.data)
    
    async def batch_transform(self, transformations, options=None):
        """批量语义转换"""
        transform_request = {
            "transformations": transformations,
            "options": options or {}
        }
        
        response = await self.client.post(
            '/semantic/batch-transform', 
            transform_request
        )
        return BatchTransformResult(response.data)

# 使用示例
async def semantic_processing_example():
    semantic_processor = SemanticProcessor(client)
    
    # MQTT到OPC UA的消息映射
    mqtt_message = {
        "topic": "sensors/temp001/data",
        "payload": {
            "temperature": 23.5,
            "timestamp": "2024-01-20T14:30:00Z"
        }
    }
    
    mapping = await semantic_processor.map_protocol_message(
        "mqtt", "opcua", mqtt_message
    )
    
    print(f"Mapped message: {mapping.target_message}")
    print(f"Confidence: {mapping.mapping_confidence}")
    
    # 设备能力推理
    device_data = {
        "device_id": "unknown_device",
        "observed_behaviors": [
            {
                "action": "data_transmission",
                "payload_structure": {"temperature": "numeric"}
            }
        ]
    }
    
    capabilities = await semantic_processor.infer_device_capabilities(device_data)
    
    for cap in capabilities.inferred_capabilities:
        print(f"Capability: {cap.capability} (confidence: {cap.confidence})")
```

### 3.4 实时流处理SDK

```python
class RealTimeStreamProcessor:
    def __init__(self, client):
        self.client = client
        self.websocket = None
        self.message_handlers = {}
    
    async def connect_stream(self, stream_types=None):
        """连接实时数据流"""
        ws_url = self.client.get_websocket_url()
        self.websocket = await websockets.connect(ws_url)
        
        # 订阅流类型
        if stream_types:
            await self.subscribe_streams(stream_types)
        
        # 开始处理消息
        await self.start_message_processing()
    
    async def subscribe_streams(self, stream_types):
        """订阅数据流"""
        subscription = {
            "action": "subscribe",
            "streams": stream_types
        }
        await self.websocket.send(json.dumps(subscription))
    
    def add_message_handler(self, message_type, handler):
        """添加消息处理器"""
        self.message_handlers[message_type] = handler
    
    async def start_message_processing(self):
        """开始处理消息"""
        async for message in self.websocket:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                await handler(data)
    
    async def publish_semantic_data(self, data):
        """发布语义数据"""
        message = {
            "action": "publish",
            "type": "semantic_data",
            "data": data
        }
        await self.websocket.send(json.dumps(message))

# 使用示例
async def stream_processing_example():
    stream_processor = RealTimeStreamProcessor(client)
    
    # 定义消息处理器
    async def handle_device_data(message):
        device_id = message['data']['device_id']
        semantic_data = message['data']['semantic_data']
        print(f"Received semantic data from {device_id}: {semantic_data}")
    
    async def handle_anomaly_alert(message):
        alert = message['data']
        print(f"Anomaly detected: {alert['description']}")
    
    # 注册处理器
    stream_processor.add_message_handler('device_data', handle_device_data)
    stream_processor.add_message_handler('anomaly_alert', handle_anomaly_alert)
    
    # 连接并订阅流
    await stream_processor.connect_stream(['device_data', 'anomaly_alerts'])
```

### 3.5 配置管理SDK

```python
class ConfigurationManager:
    def __init__(self, client):
        self.client = client
    
    async def get_device_config(self, device_id):
        """获取设备配置"""
        response = await self.client.get(f'/devices/{device_id}/config')
        return DeviceConfiguration(response.data)
    
    async def update_device_config(self, device_id, config):
        """更新设备配置"""
        response = await self.client.put(f'/devices/{device_id}/config', config)
        return ConfigurationResult(response.data)
    
    async def get_global_config(self):
        """获取全局配置"""
        response = await self.client.get('/config/global')
        return GlobalConfiguration(response.data)
    
    async def update_semantic_rules(self, device_id, rules):
        """更新语义规则"""
        config_update = {
            "semantic_config": {
                "mapping_rules": rules.mapping_rules,
                "inference_rules": rules.inference_rules
            }
        }
        return await self.update_device_config(device_id, config_update)
    
    async def batch_config_update(self, device_configs):
        """批量配置更新"""
        response = await self.client.post('/config/batch-update', device_configs)
        return BatchConfigResult(response.data)

# 配置模板
class SemanticConfigTemplate:
    @staticmethod
    def temperature_sensor_config():
        return {
            "semantic_config": {
                "ontology_preference": "http://purl.oclc.org/NET/UNIS/fiware/iot-lite#",
                "mapping_rules": [
                    {
                        "source_field": "temp",
                        "target_concept": "sosa:Temperature",
                        "unit_conversion": "celsius_to_kelvin"
                    }
                ],
                "inference_rules": [
                    "infer_environmental_conditions",
                    "detect_temperature_anomalies"
                ]
            }
        }
    
    @staticmethod
    def humidity_sensor_config():
        return {
            "semantic_config": {
                "ontology_preference": "http://purl.oclc.org/NET/UNIS/fiware/iot-lite#",
                "mapping_rules": [
                    {
                        "source_field": "humidity",
                        "target_concept": "sosa:Humidity",
                        "validation_range": {"min": 0, "max": 100}
                    }
                ]
            }
        }
```

## 4. JavaScript/TypeScript SDK

### 4.1 SDK初始化

```typescript
// 安装: npm install @iot-semantic/sdk

import { SemanticClient, DeviceManager, SemanticProcessor } from '@iot-semantic/sdk';

const client = new SemanticClient({
    apiUrl: 'https://api.iot-semantic.com/v1',
    apiKey: 'your_api_key',
    authToken: 'your_jwt_token'
});

const deviceManager = new DeviceManager(client);
const semanticProcessor = new SemanticProcessor(client);
```

### 4.2 TypeScript类型定义

```typescript
interface DeviceInfo {
    device_id: string;
    device_type: string;
    manufacturer: string;
    model: string;
    firmware_version: string;
    protocols: string[];
    capabilities: Capability[];
    semantic_profile: SemanticProfile;
    location?: Location;
    metadata?: Record<string, any>;
}

interface Capability {
    name: string;
    unit?: string;
    range?: { min: number; max: number };
    accuracy?: number;
    semantic_mapping?: SemanticMapping;
}

interface SemanticProfile {
    ontology_uri: string;
    device_class: string;
    measurement_type?: string;
    relationships?: Record<string, string>;
}

interface SemanticMapping {
    concept: string;
    property: string;
    unit?: string;
    transformation?: string;
}

interface DeviceHealth {
    device_id: string;
    overall_health: 'excellent' | 'good' | 'fair' | 'poor' | 'critical';
    health_score: number;
    last_check: string;
    metrics: HealthMetrics;
    anomalies: Anomaly[];
    predictions: Predictions;
}

interface HealthMetrics {
    connectivity: ConnectivityMetrics;
    performance: PerformanceMetrics;
    hardware: HardwareMetrics;
}
```

### 4.3 React组件示例

```tsx
import React, { useState, useEffect } from 'react';
import { DeviceManager, DeviceHealth } from '@iot-semantic/sdk';

const DeviceHealthDashboard: React.FC = () => {
    const [devices, setDevices] = useState<DeviceHealth[]>([]);
    const [loading, setLoading] = useState(true);
    
    useEffect(() => {
        const loadDeviceHealth = async () => {
            try {
                const deviceIds = ['device_001', 'device_002', 'device_003'];
                const healthData = await Promise.all(
                    deviceIds.map(id => deviceManager.getDeviceHealth(id))
                );
                setDevices(healthData);
            } catch (error) {
                console.error('Failed to load device health:', error);
            } finally {
                setLoading(false);
            }
        };
        
        loadDeviceHealth();
        
        // 设置定时刷新
        const interval = setInterval(loadDeviceHealth, 30000);
        return () => clearInterval(interval);
    }, []);
    
    if (loading) return <div>Loading...</div>;
    
    return (
        <div className="device-health-dashboard">
            <h2>设备健康状态监控</h2>
            <div className="device-grid">
                {devices.map(device => (
                    <DeviceHealthCard key={device.device_id} device={device} />
                ))}
            </div>
        </div>
    );
};

const DeviceHealthCard: React.FC<{ device: DeviceHealth }> = ({ device }) => {
    const getHealthColor = (health: string) => {
        const colors = {
            excellent: '#4CAF50',
            good: '#8BC34A',
            fair: '#FFC107',
            poor: '#FF9800',
            critical: '#F44336'
        };
        return colors[health] || '#9E9E9E';
    };
    
    return (
        <div className="device-health-card">
            <div className="device-header">
                <h3>{device.device_id}</h3>
                <div 
                    className="health-indicator"
                    style={{ backgroundColor: getHealthColor(device.overall_health) }}
                >
                    {device.overall_health}
                </div>
            </div>
            
            <div className="health-score">
                <div className="score-label">健康分数</div>
                <div className="score-value">{(device.health_score * 100).toFixed(1)}%</div>
            </div>
            
            <div className="metrics">
                <div className="metric">
                    <span>连接状态:</span>
                    <span>{device.metrics.connectivity.status}</span>
                </div>
                <div className="metric">
                    <span>响应时间:</span>
                    <span>{device.metrics.performance.response_time}ms</span>
                </div>
                <div className="metric">
                    <span>错误率:</span>
                    <span>{(device.metrics.performance.error_rate * 100).toFixed(2)}%</span>
                </div>
            </div>
            
            {device.anomalies.length > 0 && (
                <div className="anomalies">
                    <h4>异常警报</h4>
                    {device.anomalies.map((anomaly, index) => (
                        <div key={index} className={`anomaly ${anomaly.severity}`}>
                            {anomaly.description}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};
```

## 5. Go SDK

### 5.1 Go SDK实现

```go
package iot_semantic

import (
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "net/http"
    "time"
)

type Client struct {
    baseURL    string
    apiKey     string
    authToken  string
    httpClient *http.Client
}

type DeviceInfo struct {
    DeviceID        string             `json:"device_id"`
    DeviceType      string             `json:"device_type"`
    Manufacturer    string             `json:"manufacturer"`
    Model           string             `json:"model"`
    FirmwareVersion string             `json:"firmware_version"`
    Protocols       []string           `json:"protocols"`
    Capabilities    []Capability       `json:"capabilities"`
    SemanticProfile SemanticProfile    `json:"semantic_profile"`
    Location        *Location          `json:"location,omitempty"`
    Metadata        map[string]interface{} `json:"metadata,omitempty"`
}

type DeviceHealth struct {
    DeviceID      string      `json:"device_id"`
    OverallHealth string      `json:"overall_health"`
    HealthScore   float64     `json:"health_score"`
    LastCheck     time.Time   `json:"last_check"`
    Metrics       HealthMetrics `json:"metrics"`
    Anomalies     []Anomaly   `json:"anomalies"`
    Predictions   Predictions `json:"predictions"`
}

func NewClient(baseURL, apiKey, authToken string) *Client {
    return &Client{
        baseURL:   baseURL,
        apiKey:    apiKey,
        authToken: authToken,
        httpClient: &http.Client{
            Timeout: 30 * time.Second,
        },
    }
}

func (c *Client) RegisterDevice(ctx context.Context, deviceInfo DeviceInfo) (*DeviceRegistration, error) {
    endpoint := "/devices/register"
    
    var response struct {
        Status  string              `json:"status"`
        Data    DeviceRegistration  `json:"data"`
        Message string              `json:"message"`
    }
    
    err := c.makeRequest(ctx, "POST", endpoint, deviceInfo, &response)
    if err != nil {
        return nil, err
    }
    
    return &response.Data, nil
}

func (c *Client) GetDeviceHealth(ctx context.Context, deviceID string) (*DeviceHealth, error) {
    endpoint := fmt.Sprintf("/devices/%s/health", deviceID)
    
    var response struct {
        Status string       `json:"status"`
        Data   DeviceHealth `json:"data"`
    }
    
    err := c.makeRequest(ctx, "GET", endpoint, nil, &response)
    if err != nil {
        return nil, err
    }
    
    return &response.Data, nil
}

func (c *Client) DiscoverDevices(ctx context.Context, query string, options DiscoveryOptions) ([]Device, error) {
    endpoint := "/devices/discover"
    
    params := map[string]string{
        "query": query,
    }
    if options.Location != "" {
        params["location"] = options.Location
    }
    if options.Radius > 0 {
        params["radius"] = fmt.Sprintf("%.2f", options.Radius)
    }
    
    var response struct {
        Status string `json:"status"`
        Data   struct {
            TotalCount int      `json:"total_count"`
            Devices    []Device `json:"devices"`
        } `json:"data"`
    }
    
    err := c.makeRequestWithParams(ctx, "GET", endpoint, params, &response)
    if err != nil {
        return nil, err
    }
    
    return response.Data.Devices, nil
}

func (c *Client) MapProtocolMessage(ctx context.Context, mapping ProtocolMapping) (*MappingResult, error) {
    endpoint := "/semantic/mapping"
    
    var response struct {
        Status string        `json:"status"`
        Data   MappingResult `json:"data"`
    }
    
    err := c.makeRequest(ctx, "POST", endpoint, mapping, &response)
    if err != nil {
        return nil, err
    }
    
    return &response.Data, nil
}

func (c *Client) makeRequest(ctx context.Context, method, endpoint string, payload interface{}, result interface{}) error {
    url := c.baseURL + endpoint
    
    var body []byte
    var err error
    
    if payload != nil {
        body, err = json.Marshal(payload)
        if err != nil {
            return fmt.Errorf("failed to marshal payload: %w", err)
        }
    }
    
    req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewBuffer(body))
    if err != nil {
        return fmt.Errorf("failed to create request: %w", err)
    }
    
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Authorization", "Bearer "+c.authToken)
    req.Header.Set("X-API-Key", c.apiKey)
    
    resp, err := c.httpClient.Do(req)
    if err != nil {
        return fmt.Errorf("request failed: %w", err)
    }
    defer resp.Body.Close()
    
    if resp.StatusCode >= 400 {
        var errorResp struct {
            Error   string `json:"error"`
            Message string `json:"message"`
        }
        if err := json.NewDecoder(resp.Body).Decode(&errorResp); err == nil {
            return fmt.Errorf("API error (%d): %s - %s", resp.StatusCode, errorResp.Error, errorResp.Message)
        }
        return fmt.Errorf("HTTP error: %d", resp.StatusCode)
    }
    
    if result != nil {
        err = json.NewDecoder(resp.Body).Decode(result)
        if err != nil {
            return fmt.Errorf("failed to decode response: %w", err)
        }
    }
    
    return nil
}

// 使用示例
func ExampleUsage() {
    client := NewClient(
        "https://api.iot-semantic.com/v1",
        "your_api_key",
        "your_jwt_token",
    )
    
    ctx := context.Background()
    
    // 注册设备
    deviceInfo := DeviceInfo{
        DeviceID:     "temp_sensor_001",
        DeviceType:   "temperature_sensor",
        Manufacturer: "Acme Corp",
        Model:        "TempSens-2000",
        Protocols:    []string{"mqtt", "coap"},
        Capabilities: []Capability{
            {
                Name: "temperature_measurement",
                Unit: "celsius",
                Range: &Range{Min: -40, Max: 125},
            },
        },
    }
    
    registration, err := client.RegisterDevice(ctx, deviceInfo)
    if err != nil {
        log.Fatalf("Failed to register device: %v", err)
    }
    
    fmt.Printf("Device registered: %s\n", registration.DeviceID)
    
    // 获取设备健康状态
    health, err := client.GetDeviceHealth(ctx, "temp_sensor_001")
    if err != nil {
        log.Fatalf("Failed to get device health: %v", err)
    }
    
    fmt.Printf("Device health: %s (score: %.2f)\n", 
        health.OverallHealth, health.HealthScore)
}
```

## 6. 错误处理与最佳实践

### 6.1 错误码定义

```python
class ErrorCodes:
    # 认证相关错误 (1000-1099)
    UNAUTHORIZED = 1001
    INVALID_TOKEN = 1002
    TOKEN_EXPIRED = 1003
    INSUFFICIENT_PERMISSIONS = 1004
    
    # 设备相关错误 (2000-2099)
    DEVICE_NOT_FOUND = 2001
    DEVICE_ALREADY_EXISTS = 2002
    INVALID_DEVICE_TYPE = 2003
    DEVICE_OFFLINE = 2004
    
    # 语义处理错误 (3000-3099)
    SEMANTIC_MAPPING_FAILED = 3001
    ONTOLOGY_NOT_FOUND = 3002
    INFERENCE_ERROR = 3003
    VALIDATION_FAILED = 3004
    
    # 协议相关错误 (4000-4099)
    UNSUPPORTED_PROTOCOL = 4001
    PROTOCOL_VERSION_MISMATCH = 4002
    MESSAGE_FORMAT_ERROR = 4003
    
    # 系统错误 (5000-5099)
    INTERNAL_SERVER_ERROR = 5001
    SERVICE_UNAVAILABLE = 5002
    RATE_LIMIT_EXCEEDED = 5003
    TIMEOUT_ERROR = 5004

class SemanticException(Exception):
    def __init__(self, error_code, message, details=None):
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
```

### 6.2 重试机制

```python
import asyncio
from functools import wraps

def with_retry(max_retries=3, backoff_factor=1.0, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    wait_time = backoff_factor * (2 ** attempt)
                    await asyncio.sleep(wait_time)
            
            raise last_exception
        return wrapper
    return decorator

# 使用示例
@with_retry(max_retries=3, backoff_factor=0.5)
async def register_device_with_retry(device_info):
    return await client.register_device(device_info)
```

### 6.3 最佳实践

```python
class BestPractices:
    """SDK使用最佳实践"""
    
    @staticmethod
    async def efficient_batch_processing(client, items, batch_size=100):
        """高效批处理"""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = await asyncio.gather(*[
                client.process_item(item) for item in batch
            ], return_exceptions=True)
            
            results.extend(batch_results)
        
        return results
    
    @staticmethod
    def implement_circuit_breaker(client, failure_threshold=5, timeout=60):
        """实现熔断器模式"""
        class CircuitBreaker:
            def __init__(self):
                self.failure_count = 0
                self.last_failure_time = None
                self.state = 'closed'  # closed, open, half-open
            
            async def call(self, func, *args, **kwargs):
                if self.state == 'open':
                    if time.time() - self.last_failure_time > timeout:
                        self.state = 'half-open'
                    else:
                        raise Exception("Circuit breaker is open")
                
                try:
                    result = await func(*args, **kwargs)
                    if self.state == 'half-open':
                        self.state = 'closed'
                        self.failure_count = 0
                    return result
                except Exception as e:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    if self.failure_count >= failure_threshold:
                        self.state = 'open'
                    
                    raise e
        
        return CircuitBreaker()
```

## 7. 测试与调试

### 7.1 单元测试示例

```python
import pytest
from unittest.mock import AsyncMock, patch

class TestSemanticSDK:
    @pytest.fixture
    def mock_client(self):
        client = AsyncMock()
        client.post = AsyncMock()
        client.get = AsyncMock()
        client.put = AsyncMock()
        return client
    
    @pytest.mark.asyncio
    async def test_device_registration(self, mock_client):
        # 准备测试数据
        device_info = {
            "device_id": "test_device_001",
            "device_type": "temperature_sensor"
        }
        
        expected_response = {
            "status": "success",
            "data": {
                "device_id": "test_device_001",
                "registration_id": "reg_12345"
            }
        }
        
        mock_client.post.return_value = MockResponse(expected_response)
        
        # 执行测试
        device_manager = DeviceManager(mock_client)
        result = await device_manager.register_device(device_info)
        
        # 验证结果
        assert result.device_id == "test_device_001"
        assert result.registration_id == "reg_12345"
        mock_client.post.assert_called_once_with('/devices/register', device_info)
    
    @pytest.mark.asyncio
    async def test_semantic_mapping(self, mock_client):
        # 测试语义映射
        mapping_request = {
            "source_protocol": "mqtt",
            "target_protocol": "opcua",
            "source_message": {"temp": 23.5}
        }
        
        expected_mapping = {
            "status": "success",
            "data": {
                "mapping_id": "map_12345",
                "target_message": {"temperature": 23.5},
                "mapping_confidence": 0.98
            }
        }
        
        mock_client.post.return_value = MockResponse(expected_mapping)
        
        semantic_processor = SemanticProcessor(mock_client)
        result = await semantic_processor.map_protocol_message(
            "mqtt", "opcua", {"temp": 23.5}
        )
        
        assert result.mapping_confidence == 0.98
        assert result.target_message["temperature"] == 23.5

class MockResponse:
    def __init__(self, data):
        self.data = data
```

### 7.2 集成测试

```python
class IntegrationTests:
    @pytest.mark.integration
    async def test_end_to_end_device_workflow(self):
        """端到端设备工作流测试"""
        client = SemanticClient(
            api_url="http://localhost:8080/api/v1",
            api_key="test_api_key",
            auth_token="test_token"
        )
        
        device_manager = DeviceManager(client)
        semantic_processor = SemanticProcessor(client)
        
        # 1. 注册设备
        device_info = {
            "device_id": "integration_test_device",
            "device_type": "temperature_sensor",
            "capabilities": [
                {
                    "name": "temperature_measurement",
                    "unit": "celsius"
                }
            ]
        }
        
        registration = await device_manager.register_device(device_info)
        assert registration.device_id == "integration_test_device"
        
        # 2. 发现设备
        discovered_devices = await device_manager.discover_devices(
            query="temperature sensor"
        )
        assert len(discovered_devices) > 0
        
        # 3. 语义映射
        mqtt_message = {
            "topic": "sensors/temp/data",
            "payload": {"temperature": 25.5}
        }
        
        mapping = await semantic_processor.map_protocol_message(
            "mqtt", "opcua", mqtt_message
        )
        assert mapping.mapping_confidence > 0.5
        
        # 4. 清理测试数据
        await device_manager.unregister_device("integration_test_device")
```

本API文档和SDK提供了完整的IoT设备语义解释系统接口实现，支持多种编程语言，包含详细的使用示例和最佳实践指南，可以直接用于生产环境开发。
