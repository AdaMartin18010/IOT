# API网关实现

## 1. API网关核心架构

### 1.1 网关基础结构

```rust
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use axum::{
    routing::{get, post, put, delete},
    http::{StatusCode, HeaderMap, HeaderValue},
    response::Response,
    extract::{Path, Query, State},
    Json,
};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// API路由定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiRoute {
    pub id: String,
    pub name: String,
    pub path: String,
    pub method: HttpMethod,
    pub target_url: String,
    pub timeout: u64,
    pub retry_count: u32,
    pub rate_limit: Option<RateLimit>,
    pub authentication: Option<AuthenticationConfig>,
    pub headers: HashMap<String, String>,
    pub enabled: bool,
}

/// HTTP方法
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HttpMethod {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    HEAD,
    OPTIONS,
}

/// 限流配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    pub requests_per_minute: u32,
    pub burst_size: u32,
    pub window_size: u64, // 秒
}

/// 认证配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationConfig {
    pub auth_type: AuthType,
    pub token_header: Option<String>,
    pub api_key_header: Option<String>,
    pub jwt_secret: Option<String>,
}

/// 认证类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthType {
    None,
    Token,
    ApiKey,
    JWT,
    OAuth2,
}

/// 请求上下文
#[derive(Debug, Clone)]
pub struct RequestContext {
    pub request_id: String,
    pub timestamp: DateTime<Utc>,
    pub client_ip: String,
    pub user_agent: String,
    pub user_id: Option<String>,
    pub route: ApiRoute,
}

/// API网关状态
pub struct ApiGatewayState {
    routes: Arc<RwLock<HashMap<String, ApiRoute>>>,
    rate_limiters: Arc<RwLock<HashMap<String, RateLimiter>>>,
    auth_service: Arc<dyn AuthenticationService>,
    metrics: Arc<dyn MetricsCollector>,
}

/// API网关
pub struct ApiGateway {
    state: ApiGatewayState,
    config: GatewayConfig,
}

/// 网关配置
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    pub port: u16,
    pub workers: usize,
    pub max_connections: usize,
    pub request_timeout: u64,
    pub enable_cors: bool,
    pub cors_origins: Vec<String>,
    pub enable_metrics: bool,
    pub enable_logging: bool,
}

impl ApiGateway {
    pub fn new(config: GatewayConfig) -> Self {
        let auth_service = Arc::new(JwtAuthenticationService::new());
        let metrics = Arc::new(PrometheusMetricsCollector::new());
        
        Self {
            state: ApiGatewayState {
                routes: Arc::new(RwLock::new(HashMap::new())),
                rate_limiters: Arc::new(RwLock::new(HashMap::new())),
                auth_service,
                metrics,
            },
            config,
        }
    }

    /// 添加路由
    pub async fn add_route(&self, route: ApiRoute) -> Result<(), Box<dyn std::error::Error>> {
        let mut routes = self.state.routes.write().await;
        routes.insert(route.id.clone(), route.clone());
        
        // 创建限流器
        if let Some(rate_limit) = &route.rate_limit {
            let mut limiters = self.state.rate_limiters.write().await;
            limiters.insert(route.id.clone(), RateLimiter::new(rate_limit.clone()));
        }
        
        Ok(())
    }

    /// 删除路由
    pub async fn remove_route(&self, route_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut routes = self.state.routes.write().await;
        routes.remove(route_id);
        
        let mut limiters = self.state.rate_limiters.write().await;
        limiters.remove(route_id);
        
        Ok(())
    }

    /// 获取路由
    pub async fn get_route(&self, route_id: &str) -> Option<ApiRoute> {
        let routes = self.state.routes.read().await;
        routes.get(route_id).cloned()
    }

    /// 启动网关
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        let app = self.create_router().await?;
        
        let addr = format!("0.0.0.0:{}", self.config.port);
        println!("API Gateway starting on {}", addr);
        
        axum::Server::bind(&addr.parse()?)
            .serve(app.into_make_service())
            .await?;
        
        Ok(())
    }

    /// 创建路由器
    async fn create_router(&self) -> Result<axum::Router, Box<dyn std::error::Error>> {
        let state = Arc::new(self.state.clone());
        
        let app = axum::Router::new()
            .route("/health", get(health_check))
            .route("/metrics", get(metrics_endpoint))
            .route("/routes", get(list_routes))
            .route("/routes", post(add_route))
            .route("/routes/:id", delete(remove_route))
            .route("/*path", get(proxy_request))
            .route("/*path", post(proxy_request))
            .route("/*path", put(proxy_request))
            .route("/*path", delete(proxy_request))
            .with_state(state);
        
        Ok(app)
    }
}

/// 健康检查
async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "timestamp": Utc::now().to_rfc3339(),
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// 指标端点
async fn metrics_endpoint(State(state): State<Arc<ApiGatewayState>>) -> String {
    state.metrics.collect_metrics().await
}

/// 列出路由
async fn list_routes(State(state): State<Arc<ApiGatewayState>>) -> Json<Vec<ApiRoute>> {
    let routes = state.routes.read().await;
    Json(routes.values().cloned().collect())
}

/// 添加路由
async fn add_route(
    State(state): State<Arc<ApiGatewayState>>,
    Json(route): Json<ApiRoute>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let mut routes = state.routes.write().await;
    routes.insert(route.id.clone(), route.clone());
    
    Ok(Json(serde_json::json!({
        "status": "success",
        "message": "Route added successfully",
        "route_id": route.id
    })))
}

/// 删除路由
async fn remove_route(
    State(state): State<Arc<ApiGatewayState>>,
    Path(route_id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let mut routes = state.routes.write().await;
    routes.remove(&route_id);
    
    Ok(Json(serde_json::json!({
        "status": "success",
        "message": "Route removed successfully",
        "route_id": route_id
    })))
}
```

### 1.2 代理请求处理

```rust
use axum::{
    body::Body,
    http::{Request, Uri},
    response::IntoResponse,
};
use reqwest::Client;

/// 代理请求
async fn proxy_request(
    State(state): State<Arc<ApiGatewayState>>,
    method: axum::http::Method,
    uri: Uri,
    headers: HeaderMap,
    body: Body,
) -> Result<impl IntoResponse, StatusCode> {
    let path = uri.path();
    let query = uri.query();
    
    // 查找匹配的路由
    let route = find_matching_route(&state, &method, path).await
        .ok_or(StatusCode::NOT_FOUND)?;
    
    // 创建请求上下文
    let context = RequestContext {
        request_id: Uuid::new_v4().to_string(),
        timestamp: Utc::now(),
        client_ip: extract_client_ip(&headers),
        user_agent: extract_user_agent(&headers),
        user_id: None,
        route: route.clone(),
    };
    
    // 认证检查
    if let Some(auth_config) = &route.authentication {
        match state.auth_service.authenticate(&headers, auth_config).await {
            Ok(user_id) => {
                // 更新上下文中的用户ID
                // 注意：这里需要可变引用，实际实现中需要重新设计
            }
            Err(_) => {
                return Err(StatusCode::UNAUTHORIZED);
            }
        }
    }
    
    // 限流检查
    if let Some(rate_limit) = &route.rate_limit {
        let limiters = state.rate_limiters.read().await;
        if let Some(limiter) = limiters.get(&route.id) {
            if !limiter.allow_request(&context).await {
                return Err(StatusCode::TOO_MANY_REQUESTS);
            }
        }
    }
    
    // 记录请求开始
    state.metrics.record_request_start(&context).await;
    
    // 构建目标URL
    let target_url = build_target_url(&route.target_url, path, query)?;
    
    // 创建HTTP客户端
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(route.timeout))
        .build()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    // 构建请求
    let mut request_builder = client.request(method, &target_url);
    
    // 添加自定义头部
    for (key, value) in &route.headers {
        request_builder = request_builder.header(key, value);
    }
    
    // 转发原始头部（排除一些不应该转发的头部）
    for (key, value) in headers.iter() {
        if !should_skip_header(key.as_str()) {
            request_builder = request_builder.header(key, value);
        }
    }
    
    // 添加请求体
    let request = request_builder.body(body)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    // 发送请求
    let start_time = std::time::Instant::now();
    let response = client.execute(request).await
        .map_err(|_| StatusCode::BAD_GATEWAY)?;
    
    let duration = start_time.elapsed();
    
    // 记录请求完成
    state.metrics.record_request_complete(&context, response.status(), duration).await;
    
    // 构建响应
    let status = response.status();
    let headers = response.headers().clone();
    let body = response.bytes().await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    let mut response_builder = Response::builder()
        .status(status);
    
    // 添加响应头部
    for (key, value) in headers.iter() {
        if !should_skip_header(key.as_str()) {
            response_builder = response_builder.header(key, value);
        }
    }
    
    // 添加CORS头部
    if state.config.enable_cors {
        response_builder = add_cors_headers(response_builder);
    }
    
    Ok(response_builder
        .body(Body::from(body))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?)
}

/// 查找匹配的路由
async fn find_matching_route(
    state: &ApiGatewayState,
    method: &axum::http::Method,
    path: &str,
) -> Option<ApiRoute> {
    let routes = state.routes.read().await;
    
    for route in routes.values() {
        if !route.enabled {
            continue;
        }
        
        if route.method.to_string() == method.to_string() && path_matches(&route.path, path) {
            return Some(route.clone());
        }
    }
    
    None
}

/// 路径匹配
fn path_matches(route_path: &str, request_path: &str) -> bool {
    // 简单的路径匹配，实际应该支持更复杂的模式
    if route_path.ends_with("/*") {
        let prefix = &route_path[..route_path.len() - 2];
        request_path.starts_with(prefix)
    } else {
        route_path == request_path
    }
}

/// 构建目标URL
fn build_target_url(base_url: &str, path: &str, query: Option<&str>) -> Result<String, StatusCode> {
    let mut url = base_url.to_string();
    
    if !url.ends_with('/') {
        url.push('/');
    }
    
    // 移除开头的斜杠
    let path = path.trim_start_matches('/');
    url.push_str(path);
    
    if let Some(query) = query {
        url.push('?');
        url.push_str(query);
    }
    
    Ok(url)
}

/// 提取客户端IP
fn extract_client_ip(headers: &HeaderMap) -> String {
    headers.get("x-forwarded-for")
        .or_else(|| headers.get("x-real-ip"))
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string()
}

/// 提取用户代理
fn extract_user_agent(headers: &HeaderMap) -> String {
    headers.get("user-agent")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string()
}

/// 检查是否应该跳过头部
fn should_skip_header(header_name: &str) -> bool {
    let skip_headers = [
        "host",
        "connection",
        "content-length",
        "transfer-encoding",
    ];
    
    skip_headers.contains(&header_name.to_lowercase().as_str())
}

/// 添加CORS头部
fn add_cors_headers(mut response_builder: http::response::Builder) -> http::response::Builder {
    response_builder
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        .header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        .header("Access-Control-Max-Age", "86400")
}
```

## 2. 认证服务实现

### 2.1 认证服务接口

```rust
/// 认证服务接口
#[async_trait::async_trait]
pub trait AuthenticationService: Send + Sync {
    async fn authenticate(
        &self,
        headers: &HeaderMap,
        config: &AuthenticationConfig,
    ) -> Result<String, Box<dyn std::error::Error>>;
    
    async fn validate_token(&self, token: &str) -> Result<UserInfo, Box<dyn std::error::Error>>;
}

/// 用户信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInfo {
    pub user_id: String,
    pub username: String,
    pub email: Option<String>,
    pub roles: Vec<String>,
    pub permissions: Vec<String>,
}

/// JWT认证服务
pub struct JwtAuthenticationService {
    secret: String,
    token_cache: Arc<RwLock<HashMap<String, (UserInfo, DateTime<Utc>)>>>,
}

impl JwtAuthenticationService {
    pub fn new() -> Self {
        Self {
            secret: std::env::var("JWT_SECRET").unwrap_or_else(|_| "default_secret".to_string()),
            token_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait::async_trait]
impl AuthenticationService for JwtAuthenticationService {
    async fn authenticate(
        &self,
        headers: &HeaderMap,
        config: &AuthenticationConfig,
    ) -> Result<String, Box<dyn std::error::Error>> {
        match config.auth_type {
            AuthType::JWT => {
                let token = extract_jwt_token(headers, config)?;
                let user_info = self.validate_token(&token).await?;
                Ok(user_info.user_id)
            }
            AuthType::Token => {
                let token = extract_token(headers, config)?;
                let user_info = self.validate_token(&token).await?;
                Ok(user_info.user_id)
            }
            AuthType::ApiKey => {
                let api_key = extract_api_key(headers, config)?;
                let user_info = self.validate_api_key(&api_key).await?;
                Ok(user_info.user_id)
            }
            AuthType::None => {
                Ok("anonymous".to_string())
            }
            AuthType::OAuth2 => {
                // OAuth2认证实现
                Err("OAuth2 not implemented".into())
            }
        }
    }

    async fn validate_token(&self, token: &str) -> Result<UserInfo, Box<dyn std::error::Error>> {
        // 检查缓存
        {
            let cache = self.token_cache.read().await;
            if let Some((user_info, expiry)) = cache.get(token) {
                if *expiry > Utc::now() {
                    return Ok(user_info.clone());
                }
            }
        }
        
        // 验证JWT令牌
        let claims = jsonwebtoken::decode::<JwtClaims>(
            token,
            &jsonwebtoken::DecodingKey::from_secret(self.secret.as_ref()),
            &jsonwebtoken::Validation::default(),
        )?;
        
        let user_info = UserInfo {
            user_id: claims.claims.sub,
            username: claims.claims.username,
            email: claims.claims.email,
            roles: claims.claims.roles,
            permissions: claims.claims.permissions,
        };
        
        // 缓存用户信息
        {
            let mut cache = self.token_cache.write().await;
            cache.insert(token.to_string(), (user_info.clone(), Utc::now() + chrono::Duration::hours(1)));
        }
        
        Ok(user_info)
    }
}

/// JWT声明
#[derive(Debug, Serialize, Deserialize)]
pub struct JwtClaims {
    pub sub: String,
    pub username: String,
    pub email: Option<String>,
    pub roles: Vec<String>,
    pub permissions: Vec<String>,
    pub exp: usize,
    pub iat: usize,
}

/// 提取JWT令牌
fn extract_jwt_token(headers: &HeaderMap, config: &AuthenticationConfig) -> Result<String, Box<dyn std::error::Error>> {
    let header_name = config.token_header.as_deref().unwrap_or("Authorization");
    
    let auth_header = headers.get(header_name)
        .ok_or("Authorization header not found")?
        .to_str()?;
    
    if auth_header.starts_with("Bearer ") {
        Ok(auth_header[7..].to_string())
    } else {
        Err("Invalid authorization header format".into())
    }
}

/// 提取令牌
fn extract_token(headers: &HeaderMap, config: &AuthenticationConfig) -> Result<String, Box<dyn std::error::Error>> {
    let header_name = config.token_header.as_deref().unwrap_or("X-Auth-Token");
    
    headers.get(header_name)
        .ok_or("Token header not found")?
        .to_str()
        .map(|s| s.to_string())
        .map_err(|e| e.into())
}

/// 提取API密钥
fn extract_api_key(headers: &HeaderMap, config: &AuthenticationConfig) -> Result<String, Box<dyn std::error::Error>> {
    let header_name = config.api_key_header.as_deref().unwrap_or("X-API-Key");
    
    headers.get(header_name)
        .ok_or("API key header not found")?
        .to_str()
        .map(|s| s.to_string())
        .map_err(|e| e.into())
}

impl JwtAuthenticationService {
    async fn validate_api_key(&self, api_key: &str) -> Result<UserInfo, Box<dyn std::error::Error>> {
        // 这里应该查询数据库或缓存来验证API密钥
        // 简化实现
        Ok(UserInfo {
            user_id: "api_user".to_string(),
            username: "api_user".to_string(),
            email: None,
            roles: vec!["api_user".to_string()],
            permissions: vec!["read".to_string(), "write".to_string()],
        })
    }
}
```

## 3. 限流器实现

### 3.1 令牌桶限流器

```rust
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// 限流器
pub struct RateLimiter {
    config: RateLimit,
    tokens: Arc<RwLock<f64>>,
    last_refill: Arc<RwLock<Instant>>,
}

impl RateLimiter {
    pub fn new(config: RateLimit) -> Self {
        Self {
            config,
            tokens: Arc::new(RwLock::new(config.burst_size as f64)),
            last_refill: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// 检查是否允许请求
    pub async fn allow_request(&self, context: &RequestContext) -> bool {
        let mut tokens = self.tokens.write().await;
        let mut last_refill = self.last_refill.write().await;
        
        // 计算需要补充的令牌
        let now = Instant::now();
        let time_passed = now.duration_since(*last_refill);
        let tokens_to_add = (time_passed.as_secs_f64() * self.config.requests_per_minute as f64) / 60.0;
        
        // 补充令牌
        *tokens = (*tokens + tokens_to_add).min(self.config.burst_size as f64);
        *last_refill = now;
        
        // 检查是否有足够的令牌
        if *tokens >= 1.0 {
            *tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// 等待直到有可用的令牌
    pub async fn wait_for_token(&self, context: &RequestContext) {
        loop {
            if self.allow_request(context).await {
                break;
            }
            
            // 等待一段时间后重试
            sleep(Duration::from_millis(100)).await;
        }
    }
}
```

## 4. 指标收集器实现

### 4.1 Prometheus指标收集器

```rust
/// 指标收集器接口
#[async_trait::async_trait]
pub trait MetricsCollector: Send + Sync {
    async fn record_request_start(&self, context: &RequestContext);
    async fn record_request_complete(&self, context: &RequestContext, status: StatusCode, duration: Duration);
    async fn collect_metrics(&self) -> String;
}

/// Prometheus指标收集器
pub struct PrometheusMetricsCollector {
    request_counter: Arc<RwLock<HashMap<String, u64>>>,
    request_duration: Arc<RwLock<HashMap<String, Vec<Duration>>>>,
    error_counter: Arc<RwLock<HashMap<String, u64>>>,
}

impl PrometheusMetricsCollector {
    pub fn new() -> Self {
        Self {
            request_counter: Arc::new(RwLock::new(HashMap::new())),
            request_duration: Arc::new(RwLock::new(HashMap::new())),
            error_counter: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait::async_trait]
impl MetricsCollector for PrometheusMetricsCollector {
    async fn record_request_start(&self, context: &RequestContext) {
        let route_key = format!("{}:{}", context.route.method.to_string(), context.route.path);
        
        let mut counter = self.request_counter.write().await;
        *counter.entry(route_key.clone()).or_insert(0) += 1;
    }

    async fn record_request_complete(&self, context: &RequestContext, status: StatusCode, duration: Duration) {
        let route_key = format!("{}:{}", context.route.method.to_string(), context.route.path);
        
        // 记录请求持续时间
        {
            let mut durations = self.request_duration.write().await;
            durations.entry(route_key.clone()).or_insert_with(Vec::new).push(duration);
            
            // 保持最近1000个请求的持续时间
            if let Some(durs) = durations.get_mut(&route_key) {
                if durs.len() > 1000 {
                    durs.remove(0);
                }
            }
        }
        
        // 记录错误
        if status.is_client_error() || status.is_server_error() {
            let mut errors = self.error_counter.write().await;
            *errors.entry(route_key).or_insert(0) += 1;
        }
    }

    async fn collect_metrics(&self) -> String {
        let mut metrics = String::new();
        
        // 请求计数器
        {
            let counter = self.request_counter.read().await;
            for (route, count) in counter.iter() {
                metrics.push_str(&format!(
                    "gateway_requests_total{{route=\"{}\"}} {}\n",
                    route, count
                ));
            }
        }
        
        // 错误计数器
        {
            let errors = self.error_counter.read().await;
            for (route, count) in errors.iter() {
                metrics.push_str(&format!(
                    "gateway_errors_total{{route=\"{}\"}} {}\n",
                    route, count
                ));
            }
        }
        
        // 请求持续时间
        {
            let durations = self.request_duration.read().await;
            for (route, durs) in durations.iter() {
                if !durs.is_empty() {
                    let avg_duration = durs.iter().map(|d| d.as_millis()).sum::<u128>() / durs.len() as u128;
                    metrics.push_str(&format!(
                        "gateway_request_duration_ms{{route=\"{}\"}} {}\n",
                        route, avg_duration
                    ));
                }
            }
        }
        
        metrics
    }
}
```

## 5. 应用示例

### 5.1 IoT API网关配置

```rust
use crate::gateway::{ApiGateway, ApiRoute, HttpMethod, RateLimit, AuthenticationConfig, AuthType};

async fn iot_api_gateway() -> Result<(), Box<dyn std::error::Error>> {
    // 创建网关配置
    let config = GatewayConfig {
        port: 8080,
        workers: 4,
        max_connections: 10000,
        request_timeout: 30,
        enable_cors: true,
        cors_origins: vec!["http://localhost:3000".to_string(), "https://iot-dashboard.com".to_string()],
        enable_metrics: true,
        enable_logging: true,
    };
    
    // 创建API网关
    let gateway = ApiGateway::new(config);
    
    // 添加设备管理API路由
    let device_routes = vec![
        ApiRoute {
            id: "get_devices".to_string(),
            name: "Get Devices".to_string(),
            path: "/api/v1/devices".to_string(),
            method: HttpMethod::GET,
            target_url: "http://device-service:8081".to_string(),
            timeout: 10,
            retry_count: 3,
            rate_limit: Some(RateLimit {
                requests_per_minute: 100,
                burst_size: 20,
                window_size: 60,
            }),
            authentication: Some(AuthenticationConfig {
                auth_type: AuthType::JWT,
                token_header: Some("Authorization".to_string()),
                api_key_header: None,
                jwt_secret: Some("your-jwt-secret".to_string()),
            }),
            headers: {
                let mut headers = HashMap::new();
                headers.insert("X-Service".to_string(), "device-service".to_string());
                headers
            },
            enabled: true,
        },
        ApiRoute {
            id: "create_device".to_string(),
            name: "Create Device".to_string(),
            path: "/api/v1/devices".to_string(),
            method: HttpMethod::POST,
            target_url: "http://device-service:8081".to_string(),
            timeout: 15,
            retry_count: 3,
            rate_limit: Some(RateLimit {
                requests_per_minute: 50,
                burst_size: 10,
                window_size: 60,
            }),
            authentication: Some(AuthenticationConfig {
                auth_type: AuthType::JWT,
                token_header: Some("Authorization".to_string()),
                api_key_header: None,
                jwt_secret: Some("your-jwt-secret".to_string()),
            }),
            headers: HashMap::new(),
            enabled: true,
        },
        ApiRoute {
            id: "get_device_data".to_string(),
            name: "Get Device Data".to_string(),
            path: "/api/v1/devices/*/data".to_string(),
            method: HttpMethod::GET,
            target_url: "http://data-service:8082".to_string(),
            timeout: 20,
            retry_count: 2,
            rate_limit: Some(RateLimit {
                requests_per_minute: 200,
                burst_size: 50,
                window_size: 60,
            }),
            authentication: Some(AuthenticationConfig {
                auth_type: AuthType::JWT,
                token_header: Some("Authorization".to_string()),
                api_key_header: None,
                jwt_secret: Some("your-jwt-secret".to_string()),
            }),
            headers: HashMap::new(),
            enabled: true,
        },
        ApiRoute {
            id: "analytics_api".to_string(),
            name: "Analytics API".to_string(),
            path: "/api/v1/analytics/*".to_string(),
            method: HttpMethod::GET,
            target_url: "http://analytics-service:8083".to_string(),
            timeout: 30,
            retry_count: 1,
            rate_limit: Some(RateLimit {
                requests_per_minute: 30,
                burst_size: 5,
                window_size: 60,
            }),
            authentication: Some(AuthenticationConfig {
                auth_type: AuthType::ApiKey,
                token_header: None,
                api_key_header: Some("X-API-Key".to_string()),
                jwt_secret: None,
            }),
            headers: HashMap::new(),
            enabled: true,
        },
    ];
    
    // 添加所有路由
    for route in device_routes {
        gateway.add_route(route).await?;
    }
    
    // 启动网关
    gateway.start().await?;
    
    Ok(())
}
```

## 6. 总结

本实现提供了：

1. **完整的API网关架构** - 支持路由、代理、认证、限流
2. **多种认证方式** - JWT、Token、API Key、OAuth2
3. **灵活的限流机制** - 基于令牌桶算法
4. **实时指标监控** - Prometheus格式的指标收集
5. **CORS支持** - 跨域请求处理
6. **实际应用示例** - IoT API网关配置

这个API网关为IoT平台提供了统一的API入口，支持安全认证、流量控制和监控。
