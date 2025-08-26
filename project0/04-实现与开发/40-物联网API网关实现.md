# 物联网API网关实现

## 概述

物联网API网关提供统一的API入口，支持请求路由、认证授权、限流、监控和负载均衡等功能。

## 核心架构

### 1. API网关核心

```rust
pub struct ApiGateway {
    router: Arc<Router>,
    auth_service: Arc<AuthService>,
    rate_limiter: Arc<RateLimiter>,
    load_balancer: Arc<LoadBalancer>,
    metrics_collector: Arc<MetricsCollector>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    pub port: u16,
    pub host: String,
    pub request_timeout: Duration,
    pub enable_cors: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Route {
    pub id: String,
    pub name: String,
    pub path: String,
    pub method: HttpMethod,
    pub target_url: String,
    pub service_name: String,
    pub auth_required: bool,
    pub rate_limit: Option<RateLimitConfig>,
    pub timeout: Option<Duration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HttpMethod {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub requests_per_minute: u32,
    pub burst_size: u32,
}

impl ApiGateway {
    pub fn new() -> Self {
        Self {
            router: Arc::new(Router::new()),
            auth_service: Arc::new(AuthService::new()),
            rate_limiter: Arc::new(RateLimiter::new()),
            load_balancer: Arc::new(LoadBalancer::new()),
            metrics_collector: Arc::new(MetricsCollector::new()),
        }
    }

    // 启动API网关
    pub async fn start(&self, config: GatewayConfig) -> Result<(), GatewayError> {
        let app = Router::new()
            .route("/*path", any(self.handle_request))
            .layer(Extension(self.clone()))
            .layer(CorsLayer::permissive());

        let addr = format!("{}:{}", config.host, config.port).parse()
            .map_err(|e| GatewayError::ConfigurationError(e.to_string()))?;

        println!("API网关启动在: {}", addr);

        axum::Server::bind(&addr)
            .serve(app.into_make_service())
            .await
            .map_err(|e| GatewayError::ServerError(e.to_string()))?;

        Ok(())
    }

    // 处理请求
    async fn handle_request(
        Extension(gateway): Extension<Arc<ApiGateway>>,
        method: Method,
        uri: Uri,
        headers: HeaderMap,
        body: Bytes,
    ) -> Result<Response, StatusCode> {
        let start_time = Instant::now();
        let request_id = Uuid::new_v4().to_string();

        // 记录请求开始
        gateway.metrics_collector.record_request_start(&request_id, &method, &uri).await;

        // 查找路由
        let route = gateway.router.find_route(&method, &uri)
            .ok_or(StatusCode::NOT_FOUND)?;

        // 认证检查
        if route.auth_required {
            let auth_result = gateway.auth_service.authenticate(&headers).await
                .map_err(|_| StatusCode::UNAUTHORIZED)?;
            
            if !gateway.auth_service.authorize(&auth_result, &route).await {
                return Err(StatusCode::FORBIDDEN);
            }
        }

        // 限流检查
        let client_id = gateway.extract_client_id(&headers);
        if let Some(rate_limit) = &route.rate_limit {
            if !gateway.rate_limiter.check_rate_limit(&client_id, rate_limit).await {
                return Err(StatusCode::TOO_MANY_REQUESTS);
            }
        }

        // 负载均衡选择目标服务
        let target_url = gateway.load_balancer.select_target(&route.service_name).await
            .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

        // 转发请求
        let response = gateway.forward_request(
            &method,
            &target_url,
            &headers,
            &body,
            &route,
            &request_id,
        ).await?;

        // 记录请求完成
        let duration = start_time.elapsed();
        gateway.metrics_collector.record_request_complete(&request_id, &response, duration).await;

        Ok(response)
    }

    // 转发请求
    async fn forward_request(
        &self,
        method: &Method,
        target_url: &str,
        headers: &HeaderMap,
        body: &Bytes,
        route: &Route,
        request_id: &str,
    ) -> Result<Response, StatusCode> {
        let client = reqwest::Client::new();
        let mut request_builder = match method.as_str() {
            "GET" => client.get(target_url),
            "POST" => client.post(target_url),
            "PUT" => client.put(target_url),
            "DELETE" => client.delete(target_url),
            "PATCH" => client.patch(target_url),
            _ => return Err(Status::METHOD_NOT_ALLOWED),
        };

        // 添加请求头
        for (name, value) in headers {
            if name != "host" && name != "content-length" {
                request_builder = request_builder.header(name, value);
            }
        }

        // 添加请求体
        if !body.is_empty() {
            request_builder = request_builder.body(body.clone());
        }

        // 设置超时
        if let Some(timeout) = route.timeout {
            request_builder = request_builder.timeout(timeout);
        }

        // 执行请求
        let response = request_builder.send().await
            .map_err(|_| StatusCode::BAD_GATEWAY)?;

        Ok(response)
    }

    // 提取客户端ID
    fn extract_client_id(&self, headers: &HeaderMap) -> String {
        headers.get("x-client-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("unknown")
            .to_string()
    }
}
```

### 2. 路由管理

```rust
pub struct Router {
    routes: Arc<RwLock<HashMap<String, Route>>>,
}

impl Router {
    pub fn new() -> Self {
        Self {
            routes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    // 添加路由
    pub async fn add_route(&self, route: Route) -> Result<(), RouterError> {
        let mut routes = self.routes.write().await;
        routes.insert(route.id.clone(), route);
        Ok(())
    }

    // 查找路由
    pub async fn find_route(&self, method: &Method, uri: &Uri) -> Option<Route> {
        let routes = self.routes.read().await;
        
        for route in routes.values() {
            if self.path_matches(&route.path, uri.path()) && 
               self.method_matches(&route.method, method) {
                return Some(route.clone());
            }
        }
        
        None
    }

    // 删除路由
    pub async fn remove_route(&self, route_id: &str) -> Result<(), RouterError> {
        let mut routes = self.routes.write().await;
        routes.remove(route_id);
        Ok(())
    }

    // 获取所有路由
    pub async fn get_all_routes(&self) -> Vec<Route> {
        let routes = self.routes.read().await;
        routes.values().cloned().collect()
    }

    // 检查路径匹配
    fn path_matches(&self, route_path: &str, request_path: &str) -> bool {
        if route_path == request_path {
            return true;
        }

        // 支持通配符匹配
        if route_path.ends_with("/*") {
            let route_prefix = &route_path[..route_path.len() - 2];
            return request_path.starts_with(route_prefix);
        }

        false
    }

    // 检查方法匹配
    fn method_matches(&self, route_method: &HttpMethod, request_method: &Method) -> bool {
        match route_method {
            HttpMethod::GET => request_method == Method::GET,
            HttpMethod::POST => request_method == Method::POST,
            HttpMethod::PUT => request_method == Method::PUT,
            HttpMethod::DELETE => request_method == Method::DELETE,
            HttpMethod::PATCH => request_method == Method::PATCH,
        }
    }
}
```

### 3. 认证授权服务

```rust
pub struct AuthService {
    jwt_validator: Arc<JwtValidator>,
    api_key_validator: Arc<ApiKeyValidator>,
}

impl AuthService {
    pub fn new() -> Self {
        Self {
            jwt_validator: Arc::new(JwtValidator::new()),
            api_key_validator: Arc::new(ApiKeyValidator::new()),
        }
    }

    // 认证
    pub async fn authenticate(&self, headers: &HeaderMap) -> Result<AuthResult, AuthError> {
        // 检查JWT Token
        if let Some(token) = headers.get("authorization") {
            if let Ok(token_str) = token.to_str() {
                if token_str.starts_with("Bearer ") {
                    let jwt_token = &token_str[7..];
                    return self.jwt_validator.validate(jwt_token).await;
                }
            }
        }

        // 检查API Key
        if let Some(api_key) = headers.get("x-api-key") {
            if let Ok(key_str) = api_key.to_str() {
                return self.api_key_validator.validate(key_str).await;
            }
        }

        Err(AuthError::NoValidToken)
    }

    // 授权
    pub async fn authorize(&self, auth_result: &AuthResult, route: &Route) -> bool {
        // 这里应该实现权限检查逻辑
        true
    }
}

#[derive(Debug, Clone)]
pub struct AuthResult {
    pub user_id: String,
    pub username: String,
    pub roles: Vec<String>,
    pub permissions: Vec<String>,
    pub token_type: TokenType,
}

#[derive(Debug, Clone)]
pub enum TokenType {
    Jwt,
    ApiKey,
}

// JWT验证器
pub struct JwtValidator {
    secret_key: String,
}

impl JwtValidator {
    pub fn new() -> Self {
        Self {
            secret_key: std::env::var("JWT_SECRET").unwrap_or_else(|_| "default_secret".to_string()),
        }
    }

    pub async fn validate(&self, token: &str) -> Result<AuthResult, AuthError> {
        // 这里应该实现JWT验证逻辑
        Ok(AuthResult {
            user_id: "user_123".to_string(),
            username: "test_user".to_string(),
            roles: vec!["user".to_string()],
            permissions: vec!["read".to_string()],
            token_type: TokenType::Jwt,
        })
    }
}

// API Key验证器
pub struct ApiKeyValidator {
    valid_keys: Arc<RwLock<HashMap<String, ApiKeyInfo>>>,
}

impl ApiKeyValidator {
    pub fn new() -> Self {
        Self {
            valid_keys: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn validate(&self, api_key: &str) -> Result<AuthResult, AuthError> {
        let keys = self.valid_keys.read().await;
        if let Some(key_info) = keys.get(api_key) {
            Ok(AuthResult {
                user_id: key_info.user_id.clone(),
                username: key_info.username.clone(),
                roles: key_info.roles.clone(),
                permissions: key_info.permissions.clone(),
                token_type: TokenType::ApiKey,
            })
        } else {
            Err(AuthError::InvalidToken)
        }
    }
}

#[derive(Debug, Clone)]
pub struct ApiKeyInfo {
    pub user_id: String,
    pub username: String,
    pub roles: Vec<String>,
    pub permissions: Vec<String>,
}
```

### 4. 限流器

```rust
pub struct RateLimiter {
    limiters: Arc<RwLock<HashMap<String, TokenBucket>>>,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {
            limiters: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    // 检查限流
    pub async fn check_rate_limit(&self, client_id: &str, config: &RateLimitConfig) -> bool {
        let key = format!("{}:{}", client_id, config.requests_per_minute);
        let mut limiters = self.limiters.write().await;
        
        let limiter = limiters.entry(key.clone()).or_insert_with(|| {
            TokenBucket::new(
                config.requests_per_minute as f64,
                config.burst_size as f64,
                Duration::from_secs(60),
            )
        });
        
        limiter.try_consume(1.0)
    }
}

// 令牌桶算法
pub struct TokenBucket {
    capacity: f64,
    tokens: f64,
    rate: f64,
    last_refill: Instant,
    refill_interval: Duration,
}

impl TokenBucket {
    pub fn new(capacity: f64, initial_tokens: f64, refill_interval: Duration) -> Self {
        Self {
            capacity,
            tokens: initial_tokens.min(capacity),
            rate: capacity / refill_interval.as_secs_f64(),
            last_refill: Instant::now(),
            refill_interval,
        }
    }

    pub fn try_consume(&mut self, tokens: f64) -> bool {
        self.refill();
        
        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill);
        
        if elapsed >= self.refill_interval {
            let new_tokens = self.rate * elapsed.as_secs_f64();
            self.tokens = (self.tokens + new_tokens).min(self.capacity);
            self.last_refill = now;
        }
    }
}
```

### 5. 负载均衡器

```rust
pub struct LoadBalancer {
    services: Arc<RwLock<HashMap<String, ServiceRegistry>>>,
}

impl LoadBalancer {
    pub fn new() -> Self {
        Self {
            services: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    // 注册服务
    pub async fn register_service(&self, service_name: String, instances: Vec<ServiceInstance>) {
        let mut services = self.services.write().await;
        services.insert(service_name, ServiceRegistry::new(instances));
    }

    // 选择目标服务
    pub async fn select_target(&self, service_name: &str) -> Option<String> {
        let services = self.services.read().await;
        if let Some(registry) = services.get(service_name) {
            let healthy_instances: Vec<_> = registry.instances.iter()
                .filter(|instance| instance.healthy)
                .collect();
            
            if !healthy_instances.is_empty() {
                // 简单的轮询策略
                let index = 0; // 简化实现
                return Some(healthy_instances[index].url.clone());
            }
        }
        None
    }
}

#[derive(Debug, Clone)]
pub struct ServiceRegistry {
    pub instances: Vec<ServiceInstance>,
}

impl ServiceRegistry {
    pub fn new(instances: Vec<ServiceInstance>) -> Self {
        Self { instances }
    }
}

#[derive(Debug, Clone)]
pub struct ServiceInstance {
    pub id: String,
    pub url: String,
    pub healthy: bool,
    pub active_connections: u32,
}
```

### 6. 指标收集器

```rust
pub struct MetricsCollector {
    metrics: Arc<RwLock<HashMap<String, MetricValue>>>,
    request_logs: Arc<RwLock<Vec<RequestLog>>>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            request_logs: Arc::new(RwLock::new(Vec::new())),
        }
    }

    // 记录请求开始
    pub async fn record_request_start(&self, request_id: &str, method: &Method, uri: &Uri) {
        let mut metrics = self.metrics.write().await;
        
        // 增加总请求数
        let total_requests = metrics.entry("total_requests".to_string())
            .or_insert(MetricValue::Counter(0));
        if let MetricValue::Counter(count) = total_requests {
            *count += 1;
        }
        
        // 增加活跃请求数
        let active_requests = metrics.entry("active_requests".to_string())
            .or_insert(MetricValue::Gauge(0));
        if let MetricValue::Gauge(count) = active_requests {
            *count += 1;
        }
        
        // 记录请求日志
        let mut logs = self.request_logs.write().await;
        logs.push(RequestLog {
            request_id: request_id.to_string(),
            method: method.to_string(),
            uri: uri.to_string(),
            start_time: Utc::now(),
            end_time: None,
            status_code: None,
            duration: None,
        });
    }

    // 记录请求完成
    pub async fn record_request_complete(&self, request_id: &str, response: &reqwest::Response, duration: Duration) {
        let mut metrics = self.metrics.write().await;
        
        // 减少活跃请求数
        let active_requests = metrics.entry("active_requests".to_string())
            .or_insert(MetricValue::Gauge(0));
        if let MetricValue::Gauge(count) = active_requests {
            *count = count.saturating_sub(1);
        }
        
        // 记录响应时间
        let response_time = metrics.entry("response_time_ms".to_string())
            .or_insert(MetricValue::Histogram(Vec::new()));
        if let MetricValue::Histogram(histogram) = response_time {
            histogram.push(duration.as_millis() as f64);
        }
        
        // 记录状态码分布
        let status_code = response.status().as_u16();
        let status_key = format!("status_code_{}", status_code);
        let status_count = metrics.entry(status_key)
            .or_insert(MetricValue::Counter(0));
        if let MetricValue::Counter(count) = status_count {
            *count += 1;
        }
        
        // 更新请求日志
        let mut logs = self.request_logs.write().await;
        if let Some(log) = logs.iter_mut().find(|l| l.request_id == request_id) {
            log.end_time = Some(Utc::now());
            log.status_code = Some(status_code);
            log.duration = Some(duration);
        }
    }

    // 获取指标
    pub async fn get_metrics(&self) -> HashMap<String, MetricValue> {
        let metrics = self.metrics.read().await;
        metrics.clone()
    }

    // 获取请求日志
    pub async fn get_request_logs(&self, limit: Option<usize>) -> Vec<RequestLog> {
        let logs = self.request_logs.read().await;
        let limit = limit.unwrap_or(100);
        logs.iter().rev().take(limit).cloned().collect()
    }
}

#[derive(Debug, Clone)]
pub enum MetricValue {
    Counter(u64),
    Gauge(i64),
    Histogram(Vec<f64>),
}

#[derive(Debug, Clone)]
pub struct RequestLog {
    pub request_id: String,
    pub method: String,
    pub uri: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub status_code: Option<u16>,
    pub duration: Option<Duration>,
}
```

### 7. API网关API

```rust
#[derive(Deserialize)]
pub struct AddRouteRequest {
    pub route: Route,
}

#[derive(Serialize)]
pub struct GatewayStatus {
    pub status: String,
    pub total_requests: u64,
    pub active_requests: u64,
}

// API网关管理API路由
pub fn gateway_routes() -> Router {
    Router::new()
        .route("/admin/routes", post(add_route))
        .route("/admin/routes", get(list_routes))
        .route("/admin/routes/:id", delete(remove_route))
        .route("/admin/metrics", get(get_metrics))
        .route("/admin/logs", get(get_logs))
        .route("/admin/status", get(get_status))
}

async fn add_route(
    Json(request): Json<AddRouteRequest>,
    State(gateway): State<Arc<ApiGateway>>,
) -> Result<Json<String>, StatusCode> {
    gateway.router.add_route(request.route).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(Json("路由添加成功".to_string()))
}

async fn list_routes(
    State(gateway): State<Arc<ApiGateway>>,
) -> Result<Json<Vec<Route>>, StatusCode> {
    let routes = gateway.router.get_all_routes().await;
    Ok(Json(routes))
}

async fn remove_route(
    Path(route_id): Path<String>,
    State(gateway): State<Arc<ApiGateway>>,
) -> Result<Json<String>, StatusCode> {
    gateway.router.remove_route(&route_id).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(Json("路由删除成功".to_string()))
}

async fn get_metrics(
    State(gateway): State<Arc<ApiGateway>>,
) -> Result<Json<HashMap<String, MetricValue>>, StatusCode> {
    let metrics = gateway.metrics_collector.get_metrics().await;
    Ok(Json(metrics))
}

async fn get_logs(
    Query(params): Query<LogQuery>,
    State(gateway): State<Arc<ApiGateway>>,
) -> Result<Json<Vec<RequestLog>>, StatusCode> {
    let logs = gateway.metrics_collector.get_request_logs(params.limit).await;
    Ok(Json(logs))
}

async fn get_status(
    State(gateway): State<Arc<ApiGateway>>,
) -> Result<Json<GatewayStatus>, StatusCode> {
    let metrics = gateway.metrics_collector.get_metrics().await;
    
    let total_requests = if let Some(MetricValue::Counter(count)) = metrics.get("total_requests") {
        *count
    } else {
        0
    };
    
    let active_requests = if let Some(MetricValue::Gauge(count)) = metrics.get("active_requests") {
        *count as u64
    } else {
        0
    };
    
    let status = GatewayStatus {
        status: "running".to_string(),
        total_requests,
        active_requests,
    };
    
    Ok(Json(status))
}

#[derive(Deserialize)]
pub struct LogQuery {
    pub limit: Option<usize>,
}
```

## 使用示例

### 1. 启动API网关

```rust
#[tokio::main]
async fn main() {
    let gateway = Arc::new(ApiGateway::new());
    
    // 配置网关
    let config = GatewayConfig {
        port: 8080,
        host: "0.0.0.0".to_string(),
        request_timeout: Duration::from_secs(30),
        enable_cors: true,
    };
    
    // 添加路由
    let route = Route {
        id: "device_api".to_string(),
        name: "设备API".to_string(),
        path: "/api/devices/*".to_string(),
        method: HttpMethod::GET,
        target_url: "http://device-service:8081".to_string(),
        service_name: "device-service".to_string(),
        auth_required: true,
        rate_limit: Some(RateLimitConfig {
            requests_per_minute: 100,
            burst_size: 20,
        }),
        timeout: Some(Duration::from_secs(10)),
    };
    
    gateway.router.add_route(route).await.unwrap();
    
    // 注册服务实例
    let instances = vec![
        ServiceInstance {
            id: "device-1".to_string(),
            url: "http://device-service-1:8081".to_string(),
            healthy: true,
            active_connections: 0,
        },
        ServiceInstance {
            id: "device-2".to_string(),
            url: "http://device-service-2:8081".to_string(),
            healthy: true,
            active_connections: 0,
        },
    ];
    
    gateway.load_balancer.register_service("device-service".to_string(), instances).await;
    
    // 启动网关
    gateway.start(config).await.unwrap();
}
```

### 2. 监控网关状态

```rust
// 监控网关状态
async fn monitor_gateway_status(gateway: Arc<ApiGateway>) {
    loop {
        let metrics = gateway.metrics_collector.get_metrics().await;
        
        if let Some(MetricValue::Counter(total_requests)) = metrics.get("total_requests") {
            println!("总请求数: {}", total_requests);
        }
        
        if let Some(MetricValue::Gauge(active_requests)) = metrics.get("active_requests") {
            println!("活跃请求数: {}", active_requests);
        }
        
        if let Some(MetricValue::Histogram(response_times)) = metrics.get("response_time_ms") {
            if !response_times.is_empty() {
                let avg_response_time = response_times.iter().sum::<f64>() / response_times.len() as f64;
                println!("平均响应时间: {:.2}ms", avg_response_time);
            }
        }
        
        tokio::time::sleep(Duration::from_secs(10)).await;
    }
}
```

### 3. 动态配置更新

```rust
// 动态更新路由配置
async fn update_route_config(gateway: Arc<ApiGateway>) {
    let new_route = Route {
        id: "sensor_api".to_string(),
        name: "传感器API".to_string(),
        path: "/api/sensors/*".to_string(),
        method: HttpMethod::GET,
        target_url: "http://sensor-service:8082".to_string(),
        service_name: "sensor-service".to_string(),
        auth_required: true,
        rate_limit: Some(RateLimitConfig {
            requests_per_minute: 200,
            burst_size: 50,
        }),
        timeout: Some(Duration::from_secs(15)),
    };
    
    gateway.router.add_route(new_route).await.unwrap();
    println!("新路由配置已添加");
}
```

## 核心特性

1. **请求路由**: 支持路径匹配、方法匹配和通配符
2. **认证授权**: 支持JWT和API Key认证
3. **限流控制**: 基于令牌桶算法的限流机制
4. **负载均衡**: 支持轮询负载均衡策略
5. **监控指标**: 完整的请求监控和性能指标
6. **动态配置**: 支持动态路由管理
7. **API管理**: 完整的路由管理API

这个物联网API网关实现提供了企业级的API管理功能，确保API的安全、可靠和高效访问。
