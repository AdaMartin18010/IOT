#!/bin/bash

# IoT技术实现全面展开计划 - 平台启动脚本
# 版本: 1.0.0
# 描述: 完整的IoT平台启动和管理脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示启动横幅
show_banner() {
    cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║                    IoT技术实现全面展开计划                    ║
║                                                              ║
║                    平台启动脚本 v1.0.0                       ║
║                                                              ║
║        基于形式化验证的完整IoT技术平台                        ║
║                                                              ║
║    支持: OPC-UA, oneM2M, WoT, Matter, AI, 区块链, 数字孪生  ║
╚══════════════════════════════════════════════════════════════╝
EOF
}

# 检查系统要求
check_system_requirements() {
    log_info "检查系统要求..."
    
    # 检查操作系统
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log_success "操作系统: Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        log_success "操作系统: macOS"
    else
        log_warning "操作系统: $OSTYPE (可能不完全支持)"
    fi
    
    # 检查CPU核心数
    CPU_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "unknown")
    if [[ "$CPU_CORES" != "unknown" ]] && [[ "$CPU_CORES" -ge 4 ]]; then
        log_success "CPU核心数: $CPU_CORES (满足要求)"
    else
        log_warning "CPU核心数: $CPU_CORES (建议4核心以上)"
    fi
    
    # 检查内存
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        MEMORY_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        MEMORY_GB=$((MEMORY_KB / 1024 / 1024))
    else
        MEMORY_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{print $0/1024/1024/1024}' || echo "unknown")
    fi
    
    if [[ "$MEMORY_GB" != "unknown" ]] && [[ "$MEMORY_GB" -ge 8 ]]; then
        log_success "内存: ${MEMORY_GB}GB (满足要求)"
    else
        log_warning "内存: ${MEMORY_GB}GB (建议8GB以上)"
    fi
    
    # 检查磁盘空间
    DISK_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if [[ "$DISK_SPACE" -ge 20 ]]; then
        log_success "可用磁盘空间: ${DISK_SPACE}GB (满足要求)"
    else
        log_warning "可用磁盘空间: ${DISK_SPACE}GB (建议20GB以上)"
    fi
}

# 检查软件依赖
check_software_dependencies() {
    log_info "检查软件依赖..."
    
    local missing_deps=()
    
    # 检查Rust
    if command -v rustc &> /dev/null; then
        RUST_VERSION=$(rustc --version | awk '{print $2}')
        log_success "Rust: $RUST_VERSION"
    else
        missing_deps+=("rust")
    fi
    
    # 检查Docker
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
        log_success "Docker: $DOCKER_VERSION"
    else
        missing_deps+=("docker")
    fi
    
    # 检查Docker Compose
    if command -v docker-compose &> /dev/null; then
        COMPOSE_VERSION=$(docker-compose --version | awk '{print $3}' | sed 's/,//')
        log_success "Docker Compose: $COMPOSE_VERSION"
    else
        missing_deps+=("docker-compose")
    fi
    
    # 检查Git
    if command -v git &> /dev/null; then
        GIT_VERSION=$(git --version | awk '{print $3}')
        log_success "Git: $GIT_VERSION"
    else
        missing_deps+=("git")
    fi
    
    # 检查curl
    if command -v curl &> /dev/null; then
        log_success "curl: 已安装"
    else
        missing_deps+=("curl")
    fi
    
    # 检查jq
    if command -v jq &> /dev/null; then
        log_success "jq: 已安装"
    else
        missing_deps+=("jq")
    fi
    
    # 如果有缺失的依赖，显示安装说明
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_warning "缺失的依赖: ${missing_deps[*]}"
        show_installation_guide "${missing_deps[@]}"
        return 1
    fi
    
    return 0
}

# 显示安装指南
show_installation_guide() {
    local deps=("$@")
    
    log_info "安装缺失的依赖:"
    
    for dep in "${deps[@]}"; do
        case $dep in
            "rust")
                echo "  Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
                ;;
            "docker")
                echo "  Docker: https://docs.docker.com/get-docker/"
                ;;
            "docker-compose")
                echo "  Docker Compose: https://docs.docker.com/compose/install/"
                ;;
            "git")
                echo "  Git: sudo apt-get install git (Ubuntu) 或 brew install git (macOS)"
                ;;
            "curl")
                echo "  curl: sudo apt-get install curl (Ubuntu) 或 brew install curl (macOS)"
                ;;
            "jq")
                echo "  jq: sudo apt-get install jq (Ubuntu) 或 brew install jq (macOS)"
                ;;
        esac
    done
}

# 安装Rust依赖
install_rust_dependencies() {
    log_info "安装Rust依赖..."
    
    # 更新Rust
    rustup update
    
    # 安装必要的工具
    cargo install cargo-watch
    cargo install cargo-audit
    cargo install cargo-tarpaulin
    
    log_success "Rust依赖安装完成"
}

# 构建项目
build_project() {
    log_info "构建IoT平台项目..."
    
    # 清理之前的构建
    cargo clean
    
    # 构建项目
    cargo build --release
    
    if [[ $? -eq 0 ]]; then
        log_success "项目构建成功"
    else
        log_error "项目构建失败"
        exit 1
    fi
}

# 启动Docker服务
start_docker_services() {
    log_info "启动Docker服务..."
    
    # 检查Docker是否运行
    if ! docker info &> /dev/null; then
        log_error "Docker未运行，请启动Docker服务"
        exit 1
    fi
    
    # 创建网络
    docker network create iot-platform-network 2>/dev/null || log_info "网络已存在"
    
    # 启动PostgreSQL
    docker run -d \
        --name iot-postgres \
        --network iot-platform-network \
        -e POSTGRES_DB=iot_platform \
        -e POSTGRES_USER=iot_user \
        -e POSTGRES_PASSWORD=iot_password \
        -p 5432:5432 \
        postgres:14-alpine
    
    # 启动Redis
    docker run -d \
        --name iot-redis \
        --network iot-platform-network \
        -p 6379:6379 \
        redis:7-alpine
    
    # 启动Prometheus
    docker run -d \
        --name iot-prometheus \
        --network iot-platform-network \
        -p 9090:9090 \
        -v $(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml \
        prom/prometheus
    
    # 启动Grafana
    docker run -d \
        --name iot-grafana \
        --network iot-platform-network \
        -p 3000:3000 \
        -e GF_SECURITY_ADMIN_PASSWORD=admin \
        grafana/grafana
    
    log_success "Docker服务启动完成"
}

# 配置环境
setup_environment() {
    log_info "配置环境..."
    
    # 创建配置文件目录
    mkdir -p config
    mkdir -p logs
    mkdir -p data
    
    # 创建配置文件
    if [[ ! -f config/config.toml ]]; then
        cat > config/config.toml << 'EOF'
[system]
name = "IoT-Platform"
version = "1.0.0"
environment = "development"

[database]
host = "localhost"
port = 5432
name = "iot_platform"
user = "iot_user"
password = "iot_password"

[redis]
host = "localhost"
port = 6379
password = ""

[opcua]
endpoint = "opc.tcp://localhost:4840"
security_policy = "Basic256Sha256"

[onem2m]
cse_base = "http://localhost:8080"
api_key = "your_api_key"

[wot]
thing_description_url = "http://localhost:8081"
http_binding_port = 8082

[matter]
fabric_id = "your_fabric_id"
cluster_endpoint = "localhost:5540"

[ai]
model_storage_path = "./data/ai/models"
inference_engine = "onnx"

[blockchain]
network_type = "ethereum"
node_url = "http://localhost:8545"
contract_address = "0x..."

[digital_twin]
sync_interval = 30
prediction_horizon = 3600

[monitoring]
metrics_port = 9090
alert_manager_url = "http://localhost:9093"
EOF
        log_success "配置文件创建完成"
    fi
    
    # 创建环境变量文件
    if [[ ! -f .env ]]; then
        cat > .env << 'EOF'
IOT_PLATFORM_ENV=development
DATABASE_URL=postgresql://iot_user:iot_password@localhost:5432/iot_platform
REDIS_URL=redis://localhost:6379
OPCUA_ENDPOINT=opc.tcp://localhost:4840
ONEM2M_CSE_BASE=http://localhost:8080
WOT_TD_URL=http://localhost:8081
MATTER_FABRIC_ID=your_fabric_id
AI_MODEL_PATH=./data/ai/models
BLOCKCHAIN_NODE_URL=http://localhost:8545
MONITORING_PORT=9090
EOF
        log_success "环境变量文件创建完成"
    fi
}

# 启动IoT平台服务
start_iot_services() {
    log_info "启动IoT平台服务..."
    
    # 启动核心服务
    nohup cargo run --bin iot-platform > logs/platform.log 2>&1 &
    echo $! > .platform.pid
    
    # 等待服务启动
    sleep 5
    
    # 检查服务状态
    if curl -s http://localhost:8080/health > /dev/null; then
        log_success "IoT平台服务启动成功"
    else
        log_error "IoT平台服务启动失败"
        exit 1
    fi
}

# 显示服务状态
show_service_status() {
    log_info "服务状态:"
    
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│                    服务状态概览                          │"
    echo "├─────────────────────────────────────────────────────────┤"
    
    # IoT平台服务
    if [[ -f .platform.pid ]] && kill -0 $(cat .platform.pid) 2>/dev/null; then
        echo "│ IoT平台服务    │ 🟢 运行中 (PID: $(cat .platform.pid))        │"
    else
        echo "│ IoT平台服务    │ 🔴 未运行                                    │"
    fi
    
    # Docker服务
    if docker ps --filter "name=iot-" --format "table {{.Names}}\t{{.Status}}" | grep -q iot-; then
        echo "│ Docker服务     │ 🟢 运行中                                    │"
        docker ps --filter "name=iot-" --format "│                 │ %s │"
    else
        echo "│ Docker服务     │ 🔴 未运行                                    │"
    fi
    
    echo "├─────────────────────────────────────────────────────────┤"
    echo "│                    访问地址                              │"
    echo "├─────────────────────────────────────────────────────────┤"
    echo "│ IoT平台API      │ http://localhost:8080                    │"
    echo "│ Grafana仪表板   │ http://localhost:3000 (admin/admin)      │"
    echo "│ Prometheus指标  │ http://localhost:9090                    │"
    echo "│ PostgreSQL数据库│ localhost:5432                           │"
    echo "│ Redis缓存      │ localhost:6379                           │"
    echo "└─────────────────────────────────────────────────────────┘"
}

# 停止服务
stop_services() {
    log_info "停止IoT平台服务..."
    
    # 停止IoT平台服务
    if [[ -f .platform.pid ]]; then
        kill $(cat .platform.pid) 2>/dev/null || true
        rm -f .platform.pid
    fi
    
    # 停止Docker服务
    docker stop iot-postgres iot-redis iot-prometheus iot-grafana 2>/dev/null || true
    
    log_success "服务已停止"
}

# 清理资源
cleanup() {
    log_info "清理资源..."
    
    # 停止服务
    stop_services
    
    # 删除Docker容器
    docker rm iot-postgres iot-redis iot-prometheus iot-grafana 2>/dev/null || true
    
    # 删除网络
    docker network rm iot-platform-network 2>/dev/null || true
    
    log_success "资源清理完成"
}

# 显示帮助信息
show_help() {
    cat << 'EOF'
IoT技术实现全面展开计划 - 平台启动脚本

用法: $0 [选项]

选项:
    start       启动完整的IoT平台
    stop        停止所有服务
    restart     重启所有服务
    status      显示服务状态
    build       构建项目
    setup       配置环境
    cleanup     清理所有资源
    help        显示此帮助信息

示例:
    $0 start     # 启动完整平台
    $0 status    # 查看服务状态
    $0 stop      # 停止所有服务
    $0 cleanup   # 清理所有资源

EOF
}

# 主函数
main() {
    local action=${1:-start}
    
    case $action in
        "start")
            show_banner
            check_system_requirements
            check_software_dependencies || exit 1
            install_rust_dependencies
            build_project
            setup_environment
            start_docker_services
            start_iot_services
            show_service_status
            log_success "IoT平台启动完成！"
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            stop_services
            sleep 2
            $0 start
            ;;
        "status")
            show_service_status
            ;;
        "build")
            build_project
            ;;
        "setup")
            setup_environment
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "未知选项: $action"
            show_help
            exit 1
            ;;
    esac
}

# 信号处理
trap 'log_info "收到中断信号，正在清理..."; cleanup; exit 0' INT TERM

# 执行主函数
main "$@"
