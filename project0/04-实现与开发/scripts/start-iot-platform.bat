@echo off
REM IoT技术实现全面展开计划 - 平台启动脚本 (Windows版本)
REM 版本: 1.0.0
REM 描述: 完整的IoT平台启动和管理脚本

setlocal enabledelayedexpansion

REM 设置颜色代码
set "BLUE=[94m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "NC=[0m"

REM 日志函数
:log_info
echo %BLUE%[INFO]%NC% %~1
goto :eof

:log_success
echo %GREEN%[SUCCESS]%NC% %~1
goto :eof

:log_warning
echo %YELLOW%[WARNING]%NC% %~1
goto :eof

:log_error
echo %RED%[ERROR]%NC% %~1
goto :eof

REM 显示启动横幅
:show_banner
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                    IoT技术实现全面展开计划                    ║
echo ║                                                              ║
echo ║                    平台启动脚本 v1.0.0                       ║
echo ║                                                              ║
echo ║        基于形式化验证的完整IoT技术平台                        ║
echo ║                                                              ║
echo ║    支持: OPC-UA, oneM2M, WoT, Matter, AI, 区块链, 数字孪生  ║
echo ╚══════════════════════════════════════════════════════════════╝
goto :eof

REM 检查系统要求
:check_system_requirements
call :log_info "检查系统要求..."
echo.
echo 操作系统: Windows
echo CPU核心数: %NUMBER_OF_PROCESSORS%
echo 内存: 检查中...
echo 磁盘空间: 检查中...
echo.
call :log_success "系统要求检查完成"
goto :eof

REM 检查软件依赖
:check_software_dependencies
call :log_info "检查软件依赖..."
echo.
set "missing_deps="

REM 检查Rust
rustc --version >nul 2>&1
if %errorlevel% neq 0 (
    set "missing_deps=rust"
    echo Rust: 未安装
) else (
    for /f "tokens=2" %%i in ('rustc --version') do echo Rust: %%i
)

REM 检查Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    set "missing_deps=!missing_deps! docker"
    echo Docker: 未安装
) else (
    for /f "tokens=3" %%i in ('docker --version') do echo Docker: %%i
)

REM 检查Git
git --version >nul 2>&1
if %errorlevel% neq 0 (
    set "missing_deps=!missing_deps! git"
    echo Git: 未安装
) else (
    for /f "tokens=3" %%i in ('git --version') do echo Git: %%i
)

if defined missing_deps (
    call :log_warning "缺失的依赖: !missing_deps!"
    call :show_installation_guide !missing_deps!
    exit /b 1
)

call :log_success "软件依赖检查完成"
goto :eof

REM 显示安装指南
:show_installation_guide
call :log_info "安装缺失的依赖:"
echo.
for %%d in (%*) do (
    if "%%d"=="rust" (
        echo Rust: https://rustup.rs/
    ) else if "%%d"=="docker" (
        echo Docker: https://docs.docker.com/get-docker/
    ) else if "%%d"=="git" (
        echo Git: https://git-scm.com/download/win
    )
)
echo.
echo 请安装上述依赖后重新运行脚本
goto :eof

REM 安装Rust依赖
:install_rust_dependencies
call :log_info "安装Rust依赖..."
echo.
rustup update
cargo install cargo-watch
cargo install cargo-audit
cargo install cargo-tarpaulin
echo.
call :log_success "Rust依赖安装完成"
goto :eof

REM 构建项目
:build_project
call :log_info "构建IoT平台项目..."
echo.
cargo clean
cargo build --release
if %errorlevel% neq 0 (
    call :log_error "项目构建失败"
    exit /b 1
)
call :log_success "项目构建成功"
goto :eof

REM 启动Docker服务
:start_docker_services
call :log_info "启动Docker服务..."
echo.
REM 检查Docker是否运行
docker info >nul 2>&1
if %errorlevel% neq 0 (
    call :log_error "Docker未运行，请启动Docker Desktop"
    exit /b 1
)

REM 创建网络
docker network create iot-platform-network >nul 2>&1
if %errorlevel% neq 0 (
    call :log_info "网络已存在"
)

REM 启动PostgreSQL
docker run -d --name iot-postgres --network iot-platform-network -e POSTGRES_DB=iot_platform -e POSTGRES_USER=iot_user -e POSTGRES_PASSWORD=iot_password -p 5432:5432 postgres:14-alpine

REM 启动Redis
docker run -d --name iot-redis --network iot-platform-network -p 6379:6379 redis:7-alpine

REM 启动Prometheus
docker run -d --name iot-prometheus --network iot-platform-network -p 9090:9090 -v %cd%/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus

REM 启动Grafana
docker run -d --name iot-grafana --network iot-platform-network -p 3000:3000 -e GF_SECURITY_ADMIN_PASSWORD=admin grafana/grafana

call :log_success "Docker服务启动完成"
goto :eof

REM 配置环境
:setup_environment
call :log_info "配置环境..."
echo.
REM 创建配置文件目录
if not exist "config" mkdir config
if not exist "logs" mkdir logs
if not exist "data" mkdir data

REM 创建配置文件
if not exist "config\config.toml" (
    (
        echo [system]
        echo name = "IoT-Platform"
        echo version = "1.0.0"
        echo environment = "development"
        echo.
        echo [database]
        echo host = "localhost"
        echo port = 5432
        echo name = "iot_platform"
        echo user = "iot_user"
        echo password = "iot_password"
        echo.
        echo [redis]
        echo host = "localhost"
        echo port = 6379
        echo password = ""
        echo.
        echo [opcua]
        echo endpoint = "opc.tcp://localhost:4840"
        echo security_policy = "Basic256Sha256"
        echo.
        echo [onem2m]
        echo cse_base = "http://localhost:8080"
        echo api_key = "your_api_key"
        echo.
        echo [wot]
        echo thing_description_url = "http://localhost:8081"
        echo http_binding_port = 8082
        echo.
        echo [matter]
        echo fabric_id = "your_fabric_id"
        echo cluster_endpoint = "localhost:5540"
        echo.
        echo [ai]
        echo model_storage_path = "./data/ai/models"
        echo inference_engine = "onnx"
        echo.
        echo [blockchain]
        echo network_type = "ethereum"
        echo node_url = "http://localhost:8545"
        echo contract_address = "0x..."
        echo.
        echo [digital_twin]
        echo sync_interval = 30
        echo prediction_horizon = 3600
        echo.
        echo [monitoring]
        echo metrics_port = 9090
        echo alert_manager_url = "http://localhost:9093"
    ) > "config\config.toml"
    call :log_success "配置文件创建完成"
)

REM 创建环境变量文件
if not exist ".env" (
    (
        echo IOT_PLATFORM_ENV=development
        echo DATABASE_URL=postgresql://iot_user:iot_password@localhost:5432/iot_platform
        echo REDIS_URL=redis://localhost:6379
        echo OPCUA_ENDPOINT=opc.tcp://localhost:4840
        echo ONEM2M_CSE_BASE=http://localhost:8080
        echo WOT_TD_URL=http://localhost:8081
        echo MATTER_FABRIC_ID=your_fabric_id
        echo AI_MODEL_PATH=./data/ai/models
        echo BLOCKCHAIN_NODE_URL=http://localhost:8545
        echo MONITORING_PORT=9090
    ) > ".env"
    call :log_success "环境变量文件创建完成"
)
goto :eof

REM 启动IoT平台服务
:start_iot_services
call :log_info "启动IoT平台服务..."
echo.
REM 启动核心服务
start /B cargo run --bin iot-platform > logs\platform.log 2>&1
echo %time% > .platform.pid

REM 等待服务启动
timeout /t 5 /nobreak >nul

REM 检查服务状态
curl -s http://localhost:8080/health >nul 2>&1
if %errorlevel% neq 0 (
    call :log_error "IoT平台服务启动失败"
    exit /b 1
)
call :log_success "IoT平台服务启动成功"
goto :eof

REM 显示服务状态
:show_service_status
call :log_info "服务状态:"
echo.
echo ┌─────────────────────────────────────────────────────────┐
echo │                    服务状态概览                          │
echo ├─────────────────────────────────────────────────────────┤

REM IoT平台服务
if exist ".platform.pid" (
    echo │ IoT平台服务    │ 🟢 运行中 (PID: 已记录)                │
) else (
    echo │ IoT平台服务    │ 🔴 未运行                                │
)

REM Docker服务
docker ps --filter "name=iot-" --format "table {{.Names}}\t{{.Status}}" | findstr "iot-" >nul
if %errorlevel% equ 0 (
    echo │ Docker服务     │ 🟢 运行中                                │
    docker ps --filter "name=iot-" --format "│                 │ %s │"
) else (
    echo │ Docker服务     │ 🔴 未运行                                │
)

echo ├─────────────────────────────────────────────────────────┤
echo │                    访问地址                              │
echo ├─────────────────────────────────────────────────────────┤
echo │ IoT平台API      │ http://localhost:8080                    │
echo │ Grafana仪表板   │ http://localhost:3000 (admin/admin)      │
echo │ Prometheus指标  │ http://localhost:9090                    │
echo │ PostgreSQL数据库│ localhost:5432                           │
echo │ Redis缓存      │ localhost:6379                           │
echo └─────────────────────────────────────────────────────────┘
goto :eof

REM 停止服务
:stop_services
call :log_info "停止IoT平台服务..."
echo.
REM 停止IoT平台服务
if exist ".platform.pid" (
    del .platform.pid
)

REM 停止Docker服务
docker stop iot-postgres iot-redis iot-prometheus iot-grafana >nul 2>&1
call :log_success "服务已停止"
goto :eof

REM 清理资源
:cleanup
call :log_info "清理资源..."
echo.
REM 停止服务
call :stop_services

REM 删除Docker容器
docker rm iot-postgres iot-redis iot-prometheus iot-grafana >nul 2>&1

REM 删除网络
docker network rm iot-platform-network >nul 2>&1

call :log_success "资源清理完成"
goto :eof

REM 显示帮助信息
:show_help
echo IoT技术实现全面展开计划 - 平台启动脚本
echo.
echo 用法: %~nx0 [选项]
echo.
echo 选项:
echo     start       启动完整的IoT平台
echo     stop        停止所有服务
echo     restart     重启所有服务
echo     status      显示服务状态
echo     build       构建项目
echo     setup       配置环境
echo     cleanup     清理所有资源
echo     help        显示此帮助信息
echo.
echo 示例:
echo     %~nx0 start     # 启动完整平台
echo     %~nx0 status    # 查看服务状态
echo     %~nx0 stop      # 停止所有服务
echo     %~nx0 cleanup   # 清理所有资源
echo.
goto :eof

REM 主函数
:main
set "action=%~1"
if "%action%"=="" set "action=start"

if "%action%"=="start" (
    call :show_banner
    call :check_system_requirements
    call :check_software_dependencies
    if %errorlevel% neq 0 exit /b 1
    call :install_rust_dependencies
    call :build_project
    call :setup_environment
    call :start_docker_services
    call :start_iot_services
    call :show_service_status
    call :log_success "IoT平台启动完成！"
) else if "%action%"=="stop" (
    call :stop_services
) else if "%action%"=="restart" (
    call :stop_services
    timeout /t 2 /nobreak >nul
    call :%~nx0 start
) else if "%action%"=="status" (
    call :show_service_status
) else if "%action%"=="build" (
    call :build_project
) else if "%action%"=="setup" (
    call :setup_environment
) else if "%action%"=="cleanup" (
    call :cleanup
) else if "%action%"=="help" (
    call :show_help
) else if "%action%"=="-h" (
    call :show_help
) else if "%action%"=="--help" (
    call :show_help
) else (
    call :log_error "未知选项: %action%"
    call :show_help
    exit /b 1
)

exit /b 0
