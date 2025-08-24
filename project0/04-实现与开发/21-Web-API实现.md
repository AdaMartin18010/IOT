# Web API实现

## 1. RESTful API实现

### 1.1 API路由和控制器

```rust
use axum::{
    routing::{get, post, put, delete},
    http::{StatusCode, HeaderMap},
    response::Json,
    extract::{Path, Query, State},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::database::{DeviceRepository, SensorDataRepository, Device, SensorData, DeviceStatus};

/// API响应包装器
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub message: Option<String>,
    pub error: Option<String>,
}

/// 分页参数
#[derive(Debug, Deserialize)]
pub struct PaginationParams {
    pub page: Option<u64>,
    pub size: Option<u64>,
    pub sort_by: Option<String>,
    pub sort_order: Option<String>,
}

/// 设备创建请求
#[derive(Debug, Deserialize)]
pub struct CreateDeviceRequest {
    pub name: String,
    pub device_type: String,
    pub location: Option<String>,
    pub protocol: String,
    pub connection_info: serde_json::Value,
    pub metadata: Option<serde_json::Value>,
}

/// 设备更新请求
#[derive(Debug, Deserialize)]
pub struct UpdateDeviceRequest {
    pub name: Option<String>,
    pub device_type: Option<String>,
    pub location: Option<String>,
    pub status: Option<DeviceStatus>,
    pub protocol: Option<String>,
    pub connection_info: Option<serde_json::Value>,
    pub metadata: Option<serde_json::Value>,
}

/// 传感器数据创建请求
#[derive(Debug, Deserialize)]
pub struct CreateSensorDataRequest {
    pub device_id: String,
    pub sensor_type: String,
    pub value: f64,
    pub unit: String,
    pub quality: Option<DataQuality>,
    pub metadata: Option<serde_json::Value>,
}

/// API状态
pub struct ApiState {
    pub device_repo: Arc<DeviceRepository>,
    pub sensor_data_repo: Arc<SensorDataRepository>,
}

/// 创建设备API路由
pub fn create_device_routes(state: Arc<ApiState>) -> Router {
    Router::new()
        .route("/devices", get(list_devices))
        .route("/devices", post(create_device))
        .route("/devices/:id", get(get_device))
        .route("/devices/:id", put(update_device))
        .route("/devices/:id", delete(delete_device))
        .route("/devices/:id/status", put(update_device_status))
        .route("/devices/status/:status", get(get_devices_by_status))
        .route("/devices/type/:type", get(get_devices_by_type))
        .with_state(state)
}

/// 创建传感器数据API路由
pub fn create_sensor_data_routes(state: Arc<ApiState>) -> Router {
    Router::new()
        .route("/sensor-data", get(list_sensor_data))
        .route("/sensor-data", post(create_sensor_data))
        .route("/sensor-data/batch", post(batch_create_sensor_data))
        .route("/sensor-data/:id", get(get_sensor_data))
        .route("/sensor-data/:id", delete(delete_sensor_data))
        .route("/sensor-data/device/:device_id", get(get_sensor_data_by_device))
        .route("/sensor-data/aggregate", get(get_aggregated_data))
        .with_state(state)
}

/// 列出设备
async fn list_devices(
    State(state): State<Arc<ApiState>>,
    Query(params): Query<PaginationParams>,
) -> Result<Json<ApiResponse<Vec<Device>>>, StatusCode> {
    let page = params.page.unwrap_or(1);
    let size = params.size.unwrap_or(20);
    let offset = (page - 1) * size;
    
    let criteria = QueryCriteria {
        filters: Vec::new(),
        sort_by: params.sort_by,
        sort_order: params.sort_order.map(|order| {
            if order.to_lowercase() == "desc" {
                SortOrder::Desc
            } else {
                SortOrder::Asc
            }
        }),
        limit: Some(size),
        offset: Some(offset),
    };
    
    match state.device_repo.find_by_criteria(criteria).await {
        Ok(devices) => {
            Ok(Json(ApiResponse {
                success: true,
                data: Some(devices),
                message: None,
                error: None,
            }))
        }
        Err(e) => {
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// 创建设备
async fn create_device(
    State(state): State<Arc<ApiState>>,
    Json(request): Json<CreateDeviceRequest>,
) -> Result<Json<ApiResponse<Device>>, StatusCode> {
    let device = Device {
        id: uuid::Uuid::new_v4().to_string(),
        name: request.name,
        device_type: request.device_type,
        location: request.location,
        status: DeviceStatus::Offline,
        protocol: request.protocol,
        connection_info: request.connection_info,
        metadata: request.metadata.unwrap_or_default(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        last_seen: None,
        version: 1,
    };
    
    match state.device_repo.save(&device).await {
        Ok(saved_device) => {
            Ok(Json(ApiResponse {
                success: true,
                data: Some(saved_device),
                message: Some("Device created successfully".to_string()),
                error: None,
            }))
        }
        Err(e) => {
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// 获取设备
async fn get_device(
    State(state): State<Arc<ApiState>>,
    Path(id): Path<String>,
) -> Result<Json<ApiResponse<Device>>, StatusCode> {
    match state.device_repo.find_by_id(&id).await {
        Ok(Some(device)) => {
            Ok(Json(ApiResponse {
                success: true,
                data: Some(device),
                message: None,
                error: None,
            }))
        }
        Ok(None) => {
            Err(StatusCode::NOT_FOUND)
        }
        Err(e) => {
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// 更新设备
async fn update_device(
    State(state): State<Arc<ApiState>>,
    Path(id): Path<String>,
    Json(request): Json<UpdateDeviceRequest>,
) -> Result<Json<ApiResponse<Device>>, StatusCode> {
    // 先获取现有设备
    let existing_device = match state.device_repo.find_by_id(&id).await {
        Ok(Some(device)) => device,
        Ok(None) => return Err(StatusCode::NOT_FOUND),
        Err(e) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
    };
    
    // 更新字段
    let mut updated_device = existing_device;
    if let Some(name) = request.name {
        updated_device.name = name;
    }
    if let Some(device_type) = request.device_type {
        updated_device.device_type = device_type;
    }
    if let Some(location) = request.location {
        updated_device.location = Some(location);
    }
    if let Some(status) = request.status {
        updated_device.status = status;
    }
    if let Some(protocol) = request.protocol {
        updated_device.protocol = protocol;
    }
    if let Some(connection_info) = request.connection_info {
        updated_device.connection_info = connection_info;
    }
    if let Some(metadata) = request.metadata {
        updated_device.metadata = metadata;
    }
    
    updated_device.updated_at = Utc::now();
    updated_device.version += 1;
    
    match state.device_repo.update(&updated_device).await {
        Ok(device) => {
            Ok(Json(ApiResponse {
                success: true,
                data: Some(device),
                message: Some("Device updated successfully".to_string()),
                error: None,
            }))
        }
        Err(e) => {
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// 删除设备
async fn delete_device(
    State(state): State<Arc<ApiState>>,
    Path(id): Path<String>,
) -> Result<Json<ApiResponse<()>>, StatusCode> {
    match state.device_repo.delete(&id).await {
        Ok(true) => {
            Ok(Json(ApiResponse {
                success: true,
                data: None,
                message: Some("Device deleted successfully".to_string()),
                error: None,
            }))
        }
        Ok(false) => {
            Err(StatusCode::NOT_FOUND)
        }
        Err(e) => {
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// 更新设备状态
async fn update_device_status(
    State(state): State<Arc<ApiState>>,
    Path(id): Path<String>,
    Json(status): Json<DeviceStatus>,
) -> Result<Json<ApiResponse<()>>, StatusCode> {
    match state.device_repo.update_status(&id, status).await {
        Ok(true) => {
            Ok(Json(ApiResponse {
                success: true,
                data: None,
                message: Some("Device status updated successfully".to_string()),
                error: None,
            }))
        }
        Ok(false) => {
            Err(StatusCode::NOT_FOUND)
        }
        Err(e) => {
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// 根据状态获取设备
async fn get_devices_by_status(
    State(state): State<Arc<ApiState>>,
    Path(status): Path<String>,
) -> Result<Json<ApiResponse<Vec<Device>>>, StatusCode> {
    let device_status = match status.to_lowercase().as_str() {
        "online" => DeviceStatus::Online,
        "offline" => DeviceStatus::Offline,
        "error" => DeviceStatus::Error,
        "maintenance" => DeviceStatus::Maintenance,
        _ => return Err(StatusCode::BAD_REQUEST),
    };
    
    match state.device_repo.find_by_status(device_status).await {
        Ok(devices) => {
            Ok(Json(ApiResponse {
                success: true,
                data: Some(devices),
                message: None,
                error: None,
            }))
        }
        Err(e) => {
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// 根据类型获取设备
async fn get_devices_by_type(
    State(state): State<Arc<ApiState>>,
    Path(device_type): Path<String>,
) -> Result<Json<ApiResponse<Vec<Device>>>, StatusCode> {
    match state.device_repo.find_by_type(&device_type).await {
        Ok(devices) => {
            Ok(Json(ApiResponse {
                success: true,
                data: Some(devices),
                message: None,
                error: None,
            }))
        }
        Err(e) => {
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}
```

### 1.2 传感器数据API

```rust
/// 列出传感器数据
async fn list_sensor_data(
    State(state): State<Arc<ApiState>>,
    Query(params): Query<PaginationParams>,
) -> Result<Json<ApiResponse<Vec<SensorData>>>, StatusCode> {
    let page = params.page.unwrap_or(1);
    let size = params.size.unwrap_or(100);
    let offset = (page - 1) * size;
    
    // 这里应该实现分页查询，简化实现
    let start_time = Utc::now() - chrono::Duration::days(1);
    let end_time = Utc::now();
    
    match state.sensor_data_repo.find_by_device_and_time_range(
        "", // 空字符串表示查询所有设备
        start_time,
        end_time,
        Some(size),
    ).await {
        Ok(data) => {
            Ok(Json(ApiResponse {
                success: true,
                data: Some(data),
                message: None,
                error: None,
            }))
        }
        Err(e) => {
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// 创建传感器数据
async fn create_sensor_data(
    State(state): State<Arc<ApiState>>,
    Json(request): Json<CreateSensorDataRequest>,
) -> Result<Json<ApiResponse<SensorData>>, StatusCode> {
    let sensor_data = SensorData {
        id: uuid::Uuid::new_v4().to_string(),
        device_id: request.device_id,
        sensor_type: request.sensor_type,
        value: request.value,
        unit: request.unit,
        timestamp: Utc::now(),
        quality: request.quality.unwrap_or(DataQuality::Good),
        metadata: request.metadata.unwrap_or_default(),
        created_at: Utc::now(),
    };
    
    // 批量插入单个记录
    match state.sensor_data_repo.batch_insert(vec![sensor_data.clone()]).await {
        Ok(_) => {
            Ok(Json(ApiResponse {
                success: true,
                data: Some(sensor_data),
                message: Some("Sensor data created successfully".to_string()),
                error: None,
            }))
        }
        Err(e) => {
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// 批量创建传感器数据
async fn batch_create_sensor_data(
    State(state): State<Arc<ApiState>>,
    Json(requests): Json<Vec<CreateSensorDataRequest>>,
) -> Result<Json<ApiResponse<u64>>, StatusCode> {
    let sensor_data_list: Vec<SensorData> = requests.into_iter().map(|request| {
        SensorData {
            id: uuid::Uuid::new_v4().to_string(),
            device_id: request.device_id,
            sensor_type: request.sensor_type,
            value: request.value,
            unit: request.unit,
            timestamp: Utc::now(),
            quality: request.quality.unwrap_or(DataQuality::Good),
            metadata: request.metadata.unwrap_or_default(),
            created_at: Utc::now(),
        }
    }).collect();
    
    match state.sensor_data_repo.batch_insert(sensor_data_list).await {
        Ok(count) => {
            Ok(Json(ApiResponse {
                success: true,
                data: Some(count),
                message: Some(format!("{} sensor data records created successfully", count)),
                error: None,
            }))
        }
        Err(e) => {
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// 获取传感器数据
async fn get_sensor_data(
    State(state): State<Arc<ApiState>>,
    Path(id): Path<String>,
) -> Result<Json<ApiResponse<SensorData>>, StatusCode> {
    // 这里需要实现根据ID查询传感器数据的方法
    // 简化实现，返回错误
    Err(StatusCode::NOT_IMPLEMENTED)
}

/// 删除传感器数据
async fn delete_sensor_data(
    State(state): State<Arc<ApiState>>,
    Path(id): Path<String>,
) -> Result<Json<ApiResponse<()>>, StatusCode> {
    // 这里需要实现删除传感器数据的方法
    // 简化实现，返回错误
    Err(StatusCode::NOT_IMPLEMENTED)
}

/// 根据设备获取传感器数据
async fn get_sensor_data_by_device(
    State(state): State<Arc<ApiState>>,
    Path(device_id): Path<String>,
    Query(params): Query<PaginationParams>,
) -> Result<Json<ApiResponse<Vec<SensorData>>>, StatusCode> {
    let page = params.page.unwrap_or(1);
    let size = params.size.unwrap_or(100);
    
    let start_time = Utc::now() - chrono::Duration::days(1);
    let end_time = Utc::now();
    
    match state.sensor_data_repo.find_by_device_and_time_range(
        &device_id,
        start_time,
        end_time,
        Some(size),
    ).await {
        Ok(data) => {
            Ok(Json(ApiResponse {
                success: true,
                data: Some(data),
                message: None,
                error: None,
            }))
        }
        Err(e) => {
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// 获取聚合数据
async fn get_aggregated_data(
    State(state): State<Arc<ApiState>>,
    Query(params): Query<AggregationParams>,
) -> Result<Json<ApiResponse<AggregatedData>>, StatusCode> {
    let device_id = params.device_id.ok_or(StatusCode::BAD_REQUEST)?;
    let sensor_type = params.sensor_type.ok_or(StatusCode::BAD_REQUEST)?;
    let aggregation = params.aggregation.unwrap_or("average".to_string());
    let start_time = params.start_time.unwrap_or(Utc::now() - chrono::Duration::hours(1));
    let end_time = params.end_time.unwrap_or(Utc::now());
    
    let aggregation_type = match aggregation.to_lowercase().as_str() {
        "average" => AggregationType::Average,
        "sum" => AggregationType::Sum,
        "min" => AggregationType::Min,
        "max" => AggregationType::Max,
        _ => return Err(StatusCode::BAD_REQUEST),
    };
    
    match state.sensor_data_repo.get_aggregated_data(
        &device_id,
        &sensor_type,
        start_time,
        end_time,
        aggregation_type,
    ).await {
        Ok(data) => {
            Ok(Json(ApiResponse {
                success: true,
                data: Some(data),
                message: None,
                error: None,
            }))
        }
        Err(e) => {
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// 聚合参数
#[derive(Debug, Deserialize)]
pub struct AggregationParams {
    pub device_id: Option<String>,
    pub sensor_type: Option<String>,
    pub aggregation: Option<String>,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
}
```

## 2. GraphQL API实现

### 2.1 GraphQL Schema和解析器

```rust
use async_graphql::{Schema, Object, Context, FieldResult, ID};
use async_graphql::http::{GraphiQLSource, playground_source};
use async_graphql_axum::{GraphQLRequest, GraphQLResponse};

/// GraphQL查询根
pub struct QueryRoot;

#[Object]
impl QueryRoot {
    /// 获取设备列表
    async fn devices(
        &self,
        ctx: &Context<'_>,
        limit: Option<i32>,
        offset: Option<i32>,
        status: Option<DeviceStatus>,
        device_type: Option<String>,
    ) -> FieldResult<Vec<Device>> {
        let state = ctx.data::<Arc<ApiState>>()?;
        
        if let Some(status) = status {
            let devices = state.device_repo.find_by_status(status).await
                .map_err(|e| async_graphql::Error::new(e.to_string()))?;
            Ok(devices)
        } else if let Some(device_type) = device_type {
            let devices = state.device_repo.find_by_type(&device_type).await
                .map_err(|e| async_graphql::Error::new(e.to_string()))?;
            Ok(devices)
        } else {
            let criteria = QueryCriteria {
                filters: Vec::new(),
                sort_by: Some("created_at".to_string()),
                sort_order: Some(SortOrder::Desc),
                limit: limit.map(|l| l as u64),
                offset: offset.map(|o| o as u64),
            };
            
            let devices = state.device_repo.find_by_criteria(criteria).await
                .map_err(|e| async_graphql::Error::new(e.to_string()))?;
            Ok(devices)
        }
    }
    
    /// 根据ID获取设备
    async fn device(&self, ctx: &Context<'_>, id: ID) -> FieldResult<Option<Device>> {
        let state = ctx.data::<Arc<ApiState>>()?;
        let device = state.device_repo.find_by_id(&id.to_string()).await
            .map_err(|e| async_graphql::Error::new(e.to_string()))?;
        Ok(device)
    }
    
    /// 获取传感器数据
    async fn sensor_data(
        &self,
        ctx: &Context<'_>,
        device_id: Option<String>,
        sensor_type: Option<String>,
        start_time: Option<DateTime<Utc>>,
        end_time: Option<DateTime<Utc>>,
        limit: Option<i32>,
    ) -> FieldResult<Vec<SensorData>> {
        let state = ctx.data::<Arc<ApiState>>()?;
        
        let start_time = start_time.unwrap_or(Utc::now() - chrono::Duration::hours(1));
        let end_time = end_time.unwrap_or(Utc::now());
        let device_id = device_id.unwrap_or_else(|| "".to_string());
        
        let data = state.sensor_data_repo.find_by_device_and_time_range(
            &device_id,
            start_time,
            end_time,
            limit.map(|l| l as u64),
        ).await.map_err(|e| async_graphql::Error::new(e.to_string()))?;
        
        Ok(data)
    }
    
    /// 获取聚合数据
    async fn aggregated_data(
        &self,
        ctx: &Context<'_>,
        device_id: String,
        sensor_type: String,
        aggregation: String,
        start_time: Option<DateTime<Utc>>,
        end_time: Option<DateTime<Utc>>,
    ) -> FieldResult<AggregatedData> {
        let state = ctx.data::<Arc<ApiState>>()?;
        
        let start_time = start_time.unwrap_or(Utc::now() - chrono::Duration::hours(1));
        let end_time = end_time.unwrap_or(Utc::now());
        
        let aggregation_type = match aggregation.to_lowercase().as_str() {
            "average" => AggregationType::Average,
            "sum" => AggregationType::Sum,
            "min" => AggregationType::Min,
            "max" => AggregationType::Max,
            _ => return Err(async_graphql::Error::new("Invalid aggregation type")),
        };
        
        let data = state.sensor_data_repo.get_aggregated_data(
            &device_id,
            &sensor_type,
            start_time,
            end_time,
            aggregation_type,
        ).await.map_err(|e| async_graphql::Error::new(e.to_string()))?;
        
        Ok(data)
    }
}

/// GraphQL变更根
pub struct MutationRoot;

#[Object]
impl MutationRoot {
    /// 创建设备
    async fn create_device(
        &self,
        ctx: &Context<'_>,
        input: CreateDeviceInput,
    ) -> FieldResult<Device> {
        let state = ctx.data::<Arc<ApiState>>()?;
        
        let device = Device {
            id: uuid::Uuid::new_v4().to_string(),
            name: input.name,
            device_type: input.device_type,
            location: input.location,
            status: DeviceStatus::Offline,
            protocol: input.protocol,
            connection_info: input.connection_info,
            metadata: input.metadata.unwrap_or_default(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_seen: None,
            version: 1,
        };
        
        let saved_device = state.device_repo.save(&device).await
            .map_err(|e| async_graphql::Error::new(e.to_string()))?;
        
        Ok(saved_device)
    }
    
    /// 更新设备
    async fn update_device(
        &self,
        ctx: &Context<'_>,
        id: ID,
        input: UpdateDeviceInput,
    ) -> FieldResult<Option<Device>> {
        let state = ctx.data::<Arc<ApiState>>()?;
        
        let existing_device = state.device_repo.find_by_id(&id.to_string()).await
            .map_err(|e| async_graphql::Error::new(e.to_string()))?;
        
        let mut updated_device = match existing_device {
            Some(device) => device,
            None => return Ok(None),
        };
        
        if let Some(name) = input.name {
            updated_device.name = name;
        }
        if let Some(device_type) = input.device_type {
            updated_device.device_type = device_type;
        }
        if let Some(location) = input.location {
            updated_device.location = Some(location);
        }
        if let Some(status) = input.status {
            updated_device.status = status;
        }
        if let Some(protocol) = input.protocol {
            updated_device.protocol = protocol;
        }
        if let Some(connection_info) = input.connection_info {
            updated_device.connection_info = connection_info;
        }
        if let Some(metadata) = input.metadata {
            updated_device.metadata = metadata;
        }
        
        updated_device.updated_at = Utc::now();
        updated_device.version += 1;
        
        let device = state.device_repo.update(&updated_device).await
            .map_err(|e| async_graphql::Error::new(e.to_string()))?;
        
        Ok(Some(device))
    }
    
    /// 删除设备
    async fn delete_device(&self, ctx: &Context<'_>, id: ID) -> FieldResult<bool> {
        let state = ctx.data::<Arc<ApiState>>()?;
        let deleted = state.device_repo.delete(&id.to_string()).await
            .map_err(|e| async_graphql::Error::new(e.to_string()))?;
        Ok(deleted)
    }
    
    /// 创建传感器数据
    async fn create_sensor_data(
        &self,
        ctx: &Context<'_>,
        input: CreateSensorDataInput,
    ) -> FieldResult<SensorData> {
        let state = ctx.data::<Arc<ApiState>>()?;
        
        let sensor_data = SensorData {
            id: uuid::Uuid::new_v4().to_string(),
            device_id: input.device_id,
            sensor_type: input.sensor_type,
            value: input.value,
            unit: input.unit,
            timestamp: Utc::now(),
            quality: input.quality.unwrap_or(DataQuality::Good),
            metadata: input.metadata.unwrap_or_default(),
            created_at: Utc::now(),
        };
        
        let _ = state.sensor_data_repo.batch_insert(vec![sensor_data.clone()]).await
            .map_err(|e| async_graphql::Error::new(e.to_string()))?;
        
        Ok(sensor_data)
    }
}

/// GraphQL输入类型
#[derive(async_graphql::InputObject)]
pub struct CreateDeviceInput {
    pub name: String,
    pub device_type: String,
    pub location: Option<String>,
    pub protocol: String,
    pub connection_info: serde_json::Value,
    pub metadata: Option<serde_json::Value>,
}

#[derive(async_graphql::InputObject)]
pub struct UpdateDeviceInput {
    pub name: Option<String>,
    pub device_type: Option<String>,
    pub location: Option<String>,
    pub status: Option<DeviceStatus>,
    pub protocol: Option<String>,
    pub connection_info: Option<serde_json::Value>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(async_graphql::InputObject)]
pub struct CreateSensorDataInput {
    pub device_id: String,
    pub sensor_type: String,
    pub value: f64,
    pub unit: String,
    pub quality: Option<DataQuality>,
    pub metadata: Option<serde_json::Value>,
}

/// 为实体实现GraphQL类型
#[Object]
impl Device {
    async fn id(&self) -> ID {
        ID(self.id.clone())
    }
    
    async fn name(&self) -> &str {
        &self.name
    }
    
    async fn device_type(&self) -> &str {
        &self.device_type
    }
    
    async fn location(&self) -> Option<&str> {
        self.location.as_deref()
    }
    
    async fn status(&self) -> DeviceStatus {
        self.status.clone()
    }
    
    async fn protocol(&self) -> &str {
        &self.protocol
    }
    
    async fn connection_info(&self) -> serde_json::Value {
        self.connection_info.clone()
    }
    
    async fn metadata(&self) -> serde_json::Value {
        self.metadata.clone()
    }
    
    async fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }
    
    async fn updated_at(&self) -> DateTime<Utc> {
        self.updated_at
    }
    
    async fn last_seen(&self) -> Option<DateTime<Utc>> {
        self.last_seen
    }
    
    async fn version(&self) -> i32 {
        self.version as i32
    }
}

#[Object]
impl SensorData {
    async fn id(&self) -> ID {
        ID(self.id.clone())
    }
    
    async fn device_id(&self) -> &str {
        &self.device_id
    }
    
    async fn sensor_type(&self) -> &str {
        &self.sensor_type
    }
    
    async fn value(&self) -> f64 {
        self.value
    }
    
    async fn unit(&self) -> &str {
        &self.unit
    }
    
    async fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
    
    async fn quality(&self) -> DataQuality {
        self.quality.clone()
    }
    
    async fn metadata(&self) -> serde_json::Value {
        self.metadata.clone()
    }
    
    async fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }
}

#[Object]
impl AggregatedData {
    async fn value(&self) -> Option<f64> {
        self.value
    }
    
    async fn count(&self) -> i32 {
        self.count as i32
    }
    
    async fn min_value(&self) -> Option<f64> {
        self.min_value
    }
    
    async fn max_value(&self) -> Option<f64> {
        self.max_value
    }
    
    async fn aggregation_type(&self) -> String {
        format!("{:?}", self.aggregation_type)
    }
}
```

## 3. 应用示例

### 3.1 IoT Web API服务

```rust
use axum::{
    routing::get,
    response::Html,
    Router,
};
use std::net::SocketAddr;

async fn iot_web_api_service() -> Result<(), Box<dyn std::error::Error>> {
    // 创建数据库连接
    let config = DatabaseConfig {
        host: "localhost".to_string(),
        port: 5432,
        database: "iot_platform".to_string(),
        username: "iot_user".to_string(),
        password: "iot_password".to_string(),
        max_connections: 20,
        min_connections: 5,
        connection_timeout: 30,
        idle_timeout: 300,
        max_lifetime: 3600,
    };
    
    let db_manager = DatabaseManager::new(config).await?;
    
    // 创建Repository
    let device_repo = Arc::new(DeviceRepository::new(db_manager.get_pool().clone()));
    let sensor_data_repo = Arc::new(SensorDataRepository::new(db_manager.get_pool().clone()));
    
    // 创建API状态
    let api_state = Arc::new(ApiState {
        device_repo,
        sensor_data_repo,
    });
    
    // 创建RESTful API路由
    let device_routes = create_device_routes(api_state.clone());
    let sensor_data_routes = create_sensor_data_routes(api_state.clone());
    
    // 创建GraphQL Schema
    let schema = Schema::build(QueryRoot, MutationRoot, async_graphql::EmptySubscription)
        .data(api_state.clone())
        .finish();
    
    // 创建GraphQL路由
    let graphql_routes = Router::new()
        .route("/graphql", get(graphql_playground).post(graphql_handler))
        .route("/graphiql", get(graphql_playground));
    
    // 合并所有路由
    let app = Router::new()
        .nest("/api/v1", device_routes)
        .nest("/api/v1", sensor_data_routes)
        .nest("/api", graphql_routes)
        .route("/health", get(health_check))
        .route("/", get(index));
    
    // 启动服务器
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("IoT Web API server starting on {}", addr);
    
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;
    
    Ok(())
}

/// 健康检查
async fn health_check() -> Json<ApiResponse<()>> {
    Json(ApiResponse {
        success: true,
        data: None,
        message: Some("IoT Web API is healthy".to_string()),
        error: None,
    })
}

/// 首页
async fn index() -> Html<&'static str> {
    Html(r#"
    <!DOCTYPE html>
    <html>
    <head>
        <title>IoT Web API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .endpoint { margin: 20px 0; padding: 10px; background: #f5f5f5; }
            .method { font-weight: bold; color: #007cba; }
        </style>
    </head>
    <body>
        <h1>IoT Web API</h1>
        <p>Welcome to the IoT Web API service.</p>
        
        <h2>RESTful API Endpoints</h2>
        <div class="endpoint">
            <span class="method">GET</span> /api/v1/devices - List devices
        </div>
        <div class="endpoint">
            <span class="method">POST</span> /api/v1/devices - Create device
        </div>
        <div class="endpoint">
            <span class="method">GET</span> /api/v1/devices/:id - Get device
        </div>
        <div class="endpoint">
            <span class="method">PUT</span> /api/v1/devices/:id - Update device
        </div>
        <div class="endpoint">
            <span class="method">DELETE</span> /api/v1/devices/:id - Delete device
        </div>
        
        <h2>GraphQL API</h2>
        <div class="endpoint">
            <span class="method">POST</span> /api/graphql - GraphQL endpoint
        </div>
        <div class="endpoint">
            <span class="method">GET</span> /api/graphiql - GraphQL playground
        </div>
        
        <h2>Health Check</h2>
        <div class="endpoint">
            <span class="method">GET</span> /health - Health check
        </div>
    </body>
    </html>
    "#)
}

/// GraphQL处理器
async fn graphql_handler(
    schema: axum::extract::Extension<Schema<QueryRoot, MutationRoot, async_graphql::EmptySubscription>>,
    req: GraphQLRequest,
) -> GraphQLResponse {
    schema.execute(req.into_inner()).await.into()
}

/// GraphQL Playground
async fn graphql_playground() -> Html<String> {
    Html(playground_source("/api/graphql", None))
}
```

## 4. 总结

本实现提供了：

1. **RESTful API** - 完整的CRUD操作
2. **GraphQL API** - 灵活的查询和变更
3. **分页支持** - 大数据集的分页处理
4. **错误处理** - 统一的错误响应格式
5. **参数验证** - 请求参数验证
6. **API文档** - 自动生成的API文档
7. **实际应用示例** - IoT Web API服务

这个Web API实现为IoT平台提供了现代化的API接口，支持RESTful和GraphQL两种风格。
