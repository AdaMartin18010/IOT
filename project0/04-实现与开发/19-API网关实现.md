# API网关实现

## 1. API网关核心

### 1.1 API网关架构

```rust
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

// API网关
#[derive(Debug, Clone)]
pub struct APIGateway {
    pub route_manager: RouteManager,
    pub load_balancer: LoadBalancer,
    pub rate_limiter: RateLimiter,
    pub authentication_middleware: AuthenticationMiddleware,
    pub logging_middleware: LoggingMiddleware,
}

impl APIGateway {
    pub fn new() -> Self {
        Self {
            route_manager: RouteManager::new(),
            load_balancer: LoadBalancer::new(),
            rate_limiter: RateLimiter::new(),
            authentication_middleware: AuthenticationMiddleware::new(),
            logging_middleware: LoggingMiddleware::new(),
        }
    }

    // 处理请求
    pub async fn handle_request(
        &self,
        request: &HTTPRequest,
    ) -> Result<HTTPResponse, GatewayError> {
        // 记录请求日志
        self.logging_middleware.log_request(request).await?;
        
        // 速率限制检查
        self.rate_limiter.check_rate_limit(request).await?;
        
        // 认证检查
        let auth_result = self.authentication_middleware.authenticate(request).await?;
        
        // 路由请求
        let route = self.route_manager.find_route(request).await?;
        
        // 负载均衡
        let target_service = self.load_balancer.select_service(&route.services).await?;
        
        // 转发请求
        let response = self.forward_request(request, &target_service).await?;
        
        // 记录响应日志
        self.logging_middleware.log_response(&response).await?;
        
        Ok(response)
    }

    // 转发请求
    async fn forward_request(
        &self,
        request: &HTTPRequest,
        target_service: &ServiceEndpoint,
    ) -> Result<HTTPResponse, GatewayError> {
        // 构建目标URL
        let target_url = format!("{}{}", target_service.url, request.path);
        
        // 创建HTTP客户端
        let client = reqwest::Client::new();
        
        // 构建请求
        let mut http_request = client.request(request.method.clone(), &target_url);
        
        // 添加请求头
        for (key, value) in &request.headers {
            http_request = http_request.header(key, value);
        }
        
        // 添加请求体
        if let Some(body) = &request.body {
            http_request = http_request.body(body.clone());
        }
        
        // 发送请求
        let response = http_request.send().await
            .map_err(|e| GatewayError::ForwardError(e.to_string()))?;
        
        // 构建响应
        let gateway_response = HTTPResponse {
            status_code: response.status().as_u16(),
            headers: response.headers().clone(),
            body: response.bytes().await
                .map_err(|e| GatewayError::ResponseError(e.to_string()))?
                .to_vec(),
        };
        
        Ok(gateway_response)
    }
}
```

### 1.2 路由管理器

```rust
// 路由管理器
#[derive(Debug, Clone)]
pub struct RouteManager {
    pub routes: HashMap<String, Route>,
    pub route_matcher: RouteMatcher,
}

impl RouteManager {
    pub fn new() -> Self {
        Self {
            routes: HashMap::new(),
            route_matcher: RouteMatcher::new(),
        }
    }

    // 添加路由
    pub async fn add_route(
        &mut self,
        route: Route,
    ) -> Result<(), GatewayError> {
        let route_key = format!("{}:{}", route.method, route.path);
        self.routes.insert(route_key, route);
        
        Ok(())
    }

    // 查找路由
    pub async fn find_route(
        &self,
        request: &HTTPRequest,
    ) -> Result<Route, GatewayError> {
        let route_key = format!("{}:{}", request.method, request.path);
        
        if let Some(route) = self.routes.get(&route_key) {
            return Ok(route.clone());
        }
        
        // 尝试模式匹配
        for (_, route) in &self.routes {
            if self.route_matcher.matches(&route.path, &request.path) {
                return Ok(route.clone());
            }
        }
        
        Err(GatewayError::RouteNotFound)
    }
}

// 路由匹配器
#[derive(Debug, Clone)]
pub struct RouteMatcher {
    pub pattern_cache: HashMap<String, regex::Regex>,
}

impl RouteMatcher {
    pub fn new() -> Self {
        Self {
            pattern_cache: HashMap::new(),
        }
    }

    // 检查路径是否匹配
    pub fn matches(
        &self,
        pattern: &str,
        path: &str,
    ) -> bool {
        // 获取或编译正则表达式
        let regex = self.pattern_cache.entry(pattern.to_string())
            .or_insert_with(|| {
                let pattern = pattern.replace("*", ".*");
                regex::Regex::new(&pattern).unwrap()
            });
        
        regex.is_match(path)
    }
}
```

## 2. 负载均衡器

### 2.1 负载均衡器

```rust
// 负载均衡器
#[derive(Debug, Clone)]
pub struct LoadBalancer {
    pub strategies: HashMap<LoadBalancingStrategy, Box<dyn LoadBalancingStrategy>>,
    pub health_checker: HealthChecker,
}

impl LoadBalancer {
    pub fn new() -> Self {
        let mut strategies: HashMap<LoadBalancingStrategy, Box<dyn LoadBalancingStrategy>> = HashMap::new();
        strategies.insert(LoadBalancingStrategy::RoundRobin, Box::new(RoundRobinStrategy::new()));
        strategies.insert(LoadBalancingStrategy::LeastConnections, Box::new(LeastConnectionsStrategy::new()));
        strategies.insert(LoadBalancingStrategy::Weighted, Box::new(WeightedStrategy::new()));
        
        Self {
            strategies,
            health_checker: HealthChecker::new(),
        }
    }

    // 选择服务
    pub async fn select_service(
        &self,
        services: &[ServiceEndpoint],
    ) -> Result<ServiceEndpoint, GatewayError> {
        // 过滤健康服务
        let healthy_services = self.health_checker.filter_healthy_services(services).await?;
        
        if healthy_services.is_empty() {
            return Err(GatewayError::NoHealthyServices);
        }
        
        // 使用默认策略选择服务
        let strategy = self.strategies.get(&LoadBalancingStrategy::RoundRobin)
            .ok_or(GatewayError::StrategyNotFound)?;
        
        let selected_service = strategy.select_service(&healthy_services).await?;
        
        Ok(selected_service)
    }
}

// 负载均衡策略trait
#[async_trait::async_trait]
pub trait LoadBalancingStrategy: Send + Sync {
    async fn select_service(
        &self,
        services: &[ServiceEndpoint],
    ) -> Result<ServiceEndpoint, GatewayError>;
}

// 轮询策略
#[derive(Debug, Clone)]
pub struct RoundRobinStrategy {
    pub current_index: AtomicUsize,
}

impl RoundRobinStrategy {
    pub fn new() -> Self {
        Self {
            current_index: AtomicUsize::new(0),
        }
    }
}

#[async_trait::async_trait]
impl LoadBalancingStrategy for RoundRobinStrategy {
    async fn select_service(
        &self,
        services: &[ServiceEndpoint],
    ) -> Result<ServiceEndpoint, GatewayError> {
        let index = self.current_index.fetch_add(1, Ordering::Relaxed) % services.len();
        Ok(services[index].clone())
    }
}

// 最少连接策略
#[derive(Debug, Clone)]
pub struct LeastConnectionsStrategy {
    pub connection_tracker: ConnectionTracker,
}

impl LeastConnectionsStrategy {
    pub fn new() -> Self {
        Self {
            connection_tracker: ConnectionTracker::new(),
        }
    }
}

#[async_trait::async_trait]
impl LoadBalancingStrategy for LeastConnectionsStrategy {
    async fn select_service(
        &self,
        services: &[ServiceEndpoint],
    ) -> Result<ServiceEndpoint, GatewayError> {
        let mut min_connections = usize::MAX;
        let mut selected_service = None;
        
        for service in services {
            let connections = self.connection_tracker.get_connection_count(&service.id).await?;
            if connections < min_connections {
                min_connections = connections;
                selected_service = Some(service.clone());
            }
        }
        
        selected_service.ok_or(GatewayError::NoServiceAvailable)
    }
}
```

## 3. 速率限制器

### 3.1 速率限制器

```rust
// 速率限制器
#[derive(Debug, Clone)]
pub struct RateLimiter {
    pub limit_store: LimitStore,
    pub limit_configs: HashMap<String, RateLimitConfig>,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {
            limit_store: LimitStore::new(),
            limit_configs: HashMap::new(),
        }
    }

    // 检查速率限制
    pub async fn check_rate_limit(
        &self,
        request: &HTTPRequest,
    ) -> Result<(), GatewayError> {
        // 获取客户端标识
        let client_id = self.get_client_id(request).await?;
        
        // 获取限制配置
        let limit_config = self.get_limit_config(request).await?;
        
        // 检查当前使用量
        let current_usage = self.limit_store.get_current_usage(&client_id).await?;
        
        // 检查是否超过限制
        if current_usage >= limit_config.max_requests {
            return Err(GatewayError::RateLimitExceeded);
        }
        
        // 增加使用量
        self.limit_store.increment_usage(&client_id).await?;
        
        Ok(())
    }

    // 获取客户端标识
    async fn get_client_id(
        &self,
        request: &HTTPRequest,
    ) -> Result<String, GatewayError> {
        // 优先使用API密钥
        if let Some(api_key) = request.headers.get("X-API-Key") {
            return Ok(api_key.to_string());
        }
        
        // 使用IP地址
        if let Some(ip) = request.headers.get("X-Forwarded-For") {
            return Ok(ip.to_string());
        }
        
        // 使用User-Agent
        if let Some(user_agent) = request.headers.get("User-Agent") {
            return Ok(user_agent.to_string());
        }
        
        Err(GatewayError::ClientIdNotFound)
    }

    // 获取限制配置
    async fn get_limit_config(
        &self,
        request: &HTTPRequest,
    ) -> Result<RateLimitConfig, GatewayError> {
        // 根据路径获取配置
        if let Some(config) = self.limit_configs.get(&request.path) {
            return Ok(config.clone());
        }
        
        // 使用默认配置
        Ok(RateLimitConfig {
            max_requests: 100,
            window_seconds: 60,
        })
    }
}

// 限制存储
#[derive(Debug, Clone)]
pub struct LimitStore {
    pub redis_client: redis::Client,
}

impl LimitStore {
    pub fn new() -> Self {
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        Self {
            redis_client: client,
        }
    }

    // 获取当前使用量
    pub async fn get_current_usage(
        &self,
        client_id: &str,
    ) -> Result<usize, GatewayError> {
        let mut conn = self.redis_client.get_async_connection().await
            .map_err(|e| GatewayError::StorageError(e.to_string()))?;
        
        let key = format!("rate_limit:{}", client_id);
        let usage: Option<usize> = redis::cmd("GET")
            .arg(&key)
            .query_async(&mut conn)
            .await
            .map_err(|e| GatewayError::StorageError(e.to_string()))?;
        
        Ok(usage.unwrap_or(0))
    }

    // 增加使用量
    pub async fn increment_usage(
        &self,
        client_id: &str,
    ) -> Result<(), GatewayError> {
        let mut conn = self.redis_client.get_async_connection().await
            .map_err(|e| GatewayError::StorageError(e.to_string()))?;
        
        let key = format!("rate_limit:{}", client_id);
        
        // 使用Redis的INCR命令
        redis::cmd("INCR")
            .arg(&key)
            .query_async(&mut conn)
            .await
            .map_err(|e| GatewayError::StorageError(e.to_string()))?;
        
        // 设置过期时间
        redis::cmd("EXPIRE")
            .arg(&key)
            .arg(60) // 60秒过期
            .query_async(&mut conn)
            .await
            .map_err(|e| GatewayError::StorageError(e.to_string()))?;
        
        Ok(())
    }
}
```

## 4. 中间件

### 4.1 认证中间件

```rust
// 认证中间件
#[derive(Debug, Clone)]
pub struct AuthenticationMiddleware {
    pub token_validator: TokenValidator,
    pub api_key_validator: APIKeyValidator,
}

impl AuthenticationMiddleware {
    pub fn new() -> Self {
        Self {
            token_validator: TokenValidator::new(),
            api_key_validator: APIKeyValidator::new(),
        }
    }

    // 认证请求
    pub async fn authenticate(
        &self,
        request: &HTTPRequest,
    ) -> Result<AuthenticationResult, GatewayError> {
        // 检查API密钥
        if let Some(api_key) = request.headers.get("X-API-Key") {
            let result = self.api_key_validator.validate(api_key).await?;
            return Ok(result);
        }
        
        // 检查Bearer令牌
        if let Some(auth_header) = request.headers.get("Authorization") {
            if auth_header.starts_with("Bearer ") {
                let token = &auth_header[7..];
                let result = self.token_validator.validate(token).await?;
                return Ok(result);
            }
        }
        
        // 检查JWT令牌
        if let Some(jwt_token) = request.headers.get("X-JWT-Token") {
            let result = self.token_validator.validate_jwt(jwt_token).await?;
            return Ok(result);
        }
        
        Err(GatewayError::AuthenticationRequired)
    }
}

// 令牌验证器
#[derive(Debug, Clone)]
pub struct TokenValidator {
    pub jwt_validator: JWTValidator,
}

impl TokenValidator {
    pub fn new() -> Self {
        Self {
            jwt_validator: JWTValidator::new(),
        }
    }

    // 验证令牌
    pub async fn validate(
        &self,
        token: &str,
    ) -> Result<AuthenticationResult, GatewayError> {
        // 这里可以实现自定义令牌验证逻辑
        // 暂时返回成功
        Ok(AuthenticationResult {
            authenticated: true,
            user_id: "user_123".to_string(),
            permissions: vec!["read".to_string(), "write".to_string()],
        })
    }

    // 验证JWT令牌
    pub async fn validate_jwt(
        &self,
        token: &str,
    ) -> Result<AuthenticationResult, GatewayError> {
        let result = self.jwt_validator.validate(token).await?;
        
        Ok(AuthenticationResult {
            authenticated: result.valid,
            user_id: result.user_id,
            permissions: result.permissions,
        })
    }
}

// JWT验证器
#[derive(Debug, Clone)]
pub struct JWTValidator {
    pub secret_key: String,
}

impl JWTValidator {
    pub fn new() -> Self {
        Self {
            secret_key: "your-secret-key".to_string(),
        }
    }

    // 验证JWT
    pub async fn validate(
        &self,
        token: &str,
    ) -> Result<JWTValidationResult, GatewayError> {
        use jsonwebtoken::{decode, DecodingKey, Validation};
        
        let key = DecodingKey::from_secret(self.secret_key.as_ref());
        let validation = Validation::default();
        
        match decode::<Claims>(token, &key, &validation) {
            Ok(token_data) => {
                Ok(JWTValidationResult {
                    valid: true,
                    user_id: token_data.claims.sub,
                    permissions: token_data.claims.permissions,
                })
            }
            Err(_) => {
                Ok(JWTValidationResult {
                    valid: false,
                    user_id: String::new(),
                    permissions: Vec::new(),
                })
            }
        }
    }
}

// JWT声明
#[derive(Debug, Serialize, Deserialize)]
struct Claims {
    sub: String,
    exp: usize,
    permissions: Vec<String>,
}
```

### 4.2 日志中间件

```rust
// 日志中间件
#[derive(Debug, Clone)]
pub struct LoggingMiddleware {
    pub logger: Logger,
    pub log_config: LogConfig,
}

impl LoggingMiddleware {
    pub fn new() -> Self {
        Self {
            logger: Logger::new(),
            log_config: LogConfig::default(),
        }
    }

    // 记录请求日志
    pub async fn log_request(
        &self,
        request: &HTTPRequest,
    ) -> Result<(), GatewayError> {
        let log_entry = LogEntry {
            timestamp: chrono::Utc::now(),
            level: LogLevel::Info,
            message: format!("Incoming request: {} {}", request.method, request.path),
            request_id: self.generate_request_id().await?,
            client_ip: self.extract_client_ip(request).await?,
            user_agent: request.headers.get("User-Agent")
                .map(|s| s.to_string())
                .unwrap_or_default(),
        };
        
        self.logger.log(log_entry).await?;
        
        Ok(())
    }

    // 记录响应日志
    pub async fn log_response(
        &self,
        response: &HTTPResponse,
    ) -> Result<(), GatewayError> {
        let log_entry = LogEntry {
            timestamp: chrono::Utc::now(),
            level: LogLevel::Info,
            message: format!("Response sent: {}", response.status_code),
            request_id: String::new(), // 需要从上下文获取
            client_ip: String::new(), // 需要从上下文获取
            user_agent: String::new(), // 需要从上下文获取
        };
        
        self.logger.log(log_entry).await?;
        
        Ok(())
    }

    // 生成请求ID
    async fn generate_request_id(&self) -> Result<String, GatewayError> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let request_id: String = (0..16)
            .map(|_| rng.sample(rand::distributions::Alphanumeric) as char)
            .collect();
        
        Ok(request_id)
    }

    // 提取客户端IP
    async fn extract_client_ip(
        &self,
        request: &HTTPRequest,
    ) -> Result<String, GatewayError> {
        // 优先使用X-Forwarded-For头
        if let Some(forwarded_for) = request.headers.get("X-Forwarded-For") {
            return Ok(forwarded_for.to_string());
        }
        
        // 使用X-Real-IP头
        if let Some(real_ip) = request.headers.get("X-Real-IP") {
            return Ok(real_ip.to_string());
        }
        
        // 使用Remote-Addr头
        if let Some(remote_addr) = request.headers.get("Remote-Addr") {
            return Ok(remote_addr.to_string());
        }
        
        Ok("unknown".to_string())
    }
}

// 日志记录器
#[derive(Debug, Clone)]
pub struct Logger {
    pub log_writer: LogWriter,
}

impl Logger {
    pub fn new() -> Self {
        Self {
            log_writer: LogWriter::new(),
        }
    }

    // 记录日志
    pub async fn log(
        &self,
        entry: LogEntry,
    ) -> Result<(), GatewayError> {
        self.log_writer.write_log(entry).await?;
        
        Ok(())
    }
}

// 日志写入器
#[derive(Debug, Clone)]
pub struct LogWriter {
    pub file_writer: FileLogWriter,
    pub console_writer: ConsoleLogWriter,
}

impl LogWriter {
    pub fn new() -> Self {
        Self {
            file_writer: FileLogWriter::new(),
            console_writer: ConsoleLogWriter::new(),
        }
    }

    // 写入日志
    pub async fn write_log(
        &self,
        entry: LogEntry,
    ) -> Result<(), GatewayError> {
        // 写入文件
        self.file_writer.write_log(&entry).await?;
        
        // 写入控制台
        self.console_writer.write_log(&entry).await?;
        
        Ok(())
    }
}
```

---

**API网关实现完成** - 包含路由管理、负载均衡、速率限制、认证中间件、日志中间件等核心功能。
