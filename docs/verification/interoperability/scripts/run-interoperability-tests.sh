#!/bin/bash

# IoT跨标准互操作性测试执行脚本
# 作者: IoT形式化验证团队
# 版本: 1.0.0

set -e  # 遇到错误立即退出

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

# 配置变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
TEST_RESULTS_DIR="$PROJECT_ROOT/test-results"
LOG_DIR="$PROJECT_ROOT/logs"
CONFIG_FILE="$PROJECT_ROOT/config.yml"

# 创建必要的目录
mkdir -p "$TEST_RESULTS_DIR" "$LOG_DIR"

# 测试开始时间
START_TIME=$(date +%s)
LOG_FILE="$LOG_DIR/interoperability-test-$(date +%Y%m%d-%H%M%S).log"

# 记录日志
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

log_info "开始IoT跨标准互操作性测试"
log_info "测试时间: $(date)"
log_info "测试结果目录: $TEST_RESULTS_DIR"
log_info "日志文件: $LOG_FILE"

# 检查环境
check_environment() {
    log_info "检查测试环境..."
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    
    # 检查配置文件
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "配置文件不存在: $CONFIG_FILE"
        exit 1
    fi
    
    log_success "环境检查通过"
}

# 启动测试环境
start_test_environment() {
    log_info "启动测试环境..."
    
    cd "$PROJECT_ROOT"
    
    # 启动基础服务
    log_info "启动PostgreSQL和Redis..."
    docker-compose -f docker-compose.test.yml up -d postgres redis
    
    # 等待服务就绪
    log_info "等待服务就绪..."
    sleep 10
    
    # 检查服务状态
    if ! docker-compose -f docker-compose.test.yml ps | grep -q "Up"; then
        log_error "测试环境启动失败"
        exit 1
    fi
    
    log_success "测试环境启动成功"
}

# 运行TSN + OPC-UA互操作性测试
run_tsn_opcua_tests() {
    log_info "开始TSN + OPC-UA互操作性测试..."
    
    local test_results="$TEST_RESULTS_DIR/tsn-opcua-results.json"
    
    # 启动TSN和OPC-UA服务
    docker-compose -f docker-compose.test.yml up -d tsn-service opcua-service
    
    # 等待服务就绪
    sleep 15
    
    # 运行测试用例
    log_info "执行基础连接测试..."
    python3 "$SCRIPT_DIR/test_tsn_opcua_connection.py" > "$test_results" 2>&1
    
    if [ $? -eq 0 ]; then
        log_success "TSN + OPC-UA测试完成"
    else
        log_warning "TSN + OPC-UA测试发现问题"
    fi
    
    # 停止服务
    docker-compose -f docker-compose.test.yml stop tsn-service opcua-service
}

# 运行oneM2M + WoT互操作性测试
run_onem2m_wot_tests() {
    log_info "开始oneM2M + WoT互操作性测试..."
    
    local test_results="$TEST_RESULTS_DIR/onem2m-wot-results.json"
    
    # 启动oneM2M和WoT服务
    docker-compose -f docker-compose.test.yml up -d onem2m-service wot-service
    
    # 等待服务就绪
    sleep 15
    
    # 运行测试用例
    log_info "执行语义互操作测试..."
    python3 "$SCRIPT_DIR/test_onem2m_wot_semantics.py" > "$test_results" 2>&1
    
    if [ $? -eq 0 ]; then
        log_success "oneM2M + WoT测试完成"
    else
        log_warning "oneM2M + WoT测试发现问题"
    fi
    
    # 停止服务
    docker-compose -f docker-compose.test.yml stop onem2m-service wot-service
}

# 运行Matter + TSN互操作性测试
run_matter_tsn_tests() {
    log_info "开始Matter + TSN互操作性测试..."
    
    local test_results="$TEST_RESULTS_DIR/matter-tsn-results.json"
    
    # 启动Matter和TSN服务
    docker-compose -f docker-compose.test.yml up -d matter-service tsn-service
    
    # 等待服务就绪
    sleep 15
    
    # 运行测试用例
    log_info "执行设备发现测试..."
    python3 "$SCRIPT_DIR/test_matter_tsn_discovery.py" > "$test_results" 2>&1
    
    if [ $? -eq 0 ]; then
        log_success "Matter + TSN测试完成"
    else
        log_warning "Matter + TSN测试发现问题"
    fi
    
    # 停止服务
    docker-compose -f docker-compose.test.yml stop matter-service tsn-service
}

# 运行OPC-UA + oneM2M互操作性测试
run_opcua_onem2m_tests() {
    log_info "开始OPC-UA + oneM2M互操作性测试..."
    
    local test_results="$TEST_RESULTS_DIR/opcua-onem2m-results.json"
    
    # 启动OPC-UA和oneM2M服务
    docker-compose -f docker-compose.test.yml up -d opcua-service onem2m-service
    
    # 等待服务就绪
    sleep 15
    
    # 运行测试用例
    log_info "执行数据模型转换测试..."
    python3 "$SCRIPT_DIR/test_opcua_onem2m_conversion.py" > "$test_results" 2>&1
    
    if [ $? -eq 0 ]; then
        log_success "OPC-UA + oneM2M测试完成"
    else
        log_warning "OPC-UA + oneM2M测试发现问题"
    fi
    
    # 停止服务
    docker-compose -f docker-compose.test.yml stop opcua-service onem2m-service
}

# 运行WoT + Matter互操作性测试
run_wot_matter_tests() {
    log_info "开始WoT + Matter互操作性测试..."
    
    local test_results="$TEST_RESULTS_DIR/wot-matter-results.json"
    
    # 启动WoT和Matter服务
    docker-compose -f docker-compose.test.yml up -d wot-service matter-service
    
    # 等待服务就绪
    sleep 15
    
    # 运行测试用例
    log_info "执行Thing模型映射测试..."
    python3 "$SCRIPT_DIR/test_wot_matter_mapping.py" > "$test_results" 2>&1
    
    if [ $? -eq 0 ]; then
        log_success "WoT + Matter测试完成"
    else
        log_warning "WoT + Matter测试发现问题"
    fi
    
    # 停止服务
    docker-compose -f docker-compose.test.yml stop wot-service matter-service
}

# 运行性能基准测试
run_performance_tests() {
    log_info "开始性能基准测试..."
    
    local performance_results="$TEST_RESULTS_DIR/performance-results.json"
    
    # 启动所有服务进行性能测试
    docker-compose -f docker-compose.test.yml up -d
    
    # 等待服务就绪
    sleep 30
    
    # 运行性能测试
    log_info "执行负载测试..."
    python3 "$SCRIPT_DIR/run_performance_benchmarks.py" > "$performance_results" 2>&1
    
    if [ $? -eq 0 ]; then
        log_success "性能基准测试完成"
    else
        log_warning "性能基准测试发现问题"
    fi
    
    # 停止所有服务
    docker-compose -f docker-compose.test.yml down
}

# 生成测试报告
generate_test_report() {
    log_info "生成测试报告..."
    
    local report_file="$TEST_RESULTS_DIR/interoperability-test-report-$(date +%Y%m%d-%H%M%S).md"
    
    # 计算测试统计
    local total_tests=0
    local passed_tests=0
    local failed_tests=0
    
    # 统计各标准组合的测试结果
    for result_file in "$TEST_RESULTS_DIR"/*-results.json; do
        if [ -f "$result_file" ]; then
            total_tests=$((total_tests + 1))
            if grep -q '"status": "passed"' "$result_file"; then
                passed_tests=$((passed_tests + 1))
            else
                failed_tests=$((failed_tests + 1))
            fi
        fi
    done
    
    # 生成Markdown报告
    cat > "$report_file" << EOF
# IoT跨标准互操作性测试报告

## 测试摘要
- **测试时间**: $(date)
- **测试标准**: TSN, OPC-UA, oneM2M, WoT, Matter
- **测试总数**: $total_tests
- **通过数量**: $passed_tests
- **失败数量**: $failed_tests
- **通过率**: $((passed_tests * 100 / total_tests))%

## 测试结果详情

### TSN + OPC-UA 互操作性
$(if [ -f "$TEST_RESULTS_DIR/tsn-opcua-results.json" ]; then
    echo "- 状态: $(grep -o '"status": "[^"]*"' "$TEST_RESULTS_DIR/tsn-opcua-results.json" | cut -d'"' -f4)"
    echo "- 详情: 查看 [tsn-opcua-results.json]($TEST_RESULTS_DIR/tsn-opcua-results.json)"
else
    echo "- 状态: 未执行"
fi)

### oneM2M + WoT 互操作性
$(if [ -f "$TEST_RESULTS_DIR/onem2m-wot-results.json" ]; then
    echo "- 状态: $(grep -o '"status": "[^"]*"' "$TEST_RESULTS_DIR/onem2m-wot-results.json" | cut -d'"' -f4)"
    echo "- 详情: 查看 [onem2m-wot-results.json]($TEST_RESULTS_DIR/onem2m-wot-results.json)"
else
    echo "- 状态: 未执行"
fi)

### Matter + TSN 互操作性
$(if [ -f "$TEST_RESULTS_DIR/matter-tsn-results.json" ]; then
    echo "- 状态: $(grep -o '"status": "[^"]*"' "$TEST_RESULTS_DIR/matter-tsn-results.json" | cut -d'"' -f4)"
    echo "- 详情: 查看 [matter-tsn-results.json]($TEST_RESULTS_DIR/matter-tsn-results.json)"
else
    echo "- 状态: 未执行"
fi)

### OPC-UA + oneM2M 互操作性
$(if [ -f "$TEST_RESULTS_DIR/opcua-onem2m-results.json" ]; then
    echo "- 状态: $(grep -o '"status": "[^"]*"' "$TEST_RESULTS_DIR/opcua-onem2m-results.json" | cut -d'"' -f4)"
    echo "- 详情: 查看 [opcua-onem2m-results.json]($TEST_RESULTS_DIR/opcua-onem2m-results.json)"
else
    echo "- 状态: 未执行"
fi)

### WoT + Matter 互操作性
$(if [ -f "$TEST_RESULTS_DIR/wot-matter-results.json" ]; then
    echo "- 状态: $(grep -o '"status": "[^"]*"' "$TEST_RESULTS_DIR/wot-matter-results.json" | cut -d'"' -f4)"
    echo "- 详情: 查看 [wot-matter-results.json]($TEST_RESULTS_DIR/wot-matter-results.json)"
else
    echo "- 状态: 未执行"
fi)

## 性能基准测试
$(if [ -f "$TEST_RESULTS_DIR/performance-results.json" ]; then
    echo "- 状态: 已完成"
    echo "- 详情: 查看 [performance-results.json]($TEST_RESULTS_DIR/performance-results.json)"
else
    echo "- 状态: 未执行"
fi)

## 测试环境信息
- **操作系统**: $(uname -s)
- **Docker版本**: $(docker --version)
- **Docker Compose版本**: $(docker-compose --version)
- **测试脚本版本**: 1.0.0

## 建议和改进
$(if [ $failed_tests -gt 0 ]; then
    echo "- 建议检查失败的测试用例，分析根本原因"
    echo "- 优化测试环境配置，确保服务稳定性"
    echo "- 增加重试机制，提高测试成功率"
else
    echo "- 所有测试通过，系统互操作性良好"
    echo "- 建议定期执行测试，监控系统变化"
    echo "- 考虑增加更多边界条件测试"
fi)

---
*报告生成时间: $(date)*
*测试执行脚本: $0*
EOF
    
    log_success "测试报告已生成: $report_file"
}

# 发送通知
send_notifications() {
    log_info "发送测试完成通知..."
    
    # 这里可以集成Slack、Teams、邮件等通知系统
    # 示例：发送到Slack
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        local message="IoT跨标准互操作性测试完成\n通过率: $((passed_tests * 100 / total_tests))%\n查看报告: $report_file"
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" \
            "$SLACK_WEBHOOK_URL"
    fi
    
    log_success "通知发送完成"
}

# 清理资源
cleanup() {
    log_info "清理测试资源..."
    
    # 停止所有容器
    cd "$PROJECT_ROOT"
    docker-compose -f docker-compose.test.yml down --volumes --remove-orphans
    
    # 清理临时文件
    rm -rf /tmp/iot-test-*
    
    log_success "资源清理完成"
}

# 主函数
main() {
    log_info "=== IoT跨标准互操作性测试开始 ==="
    
    # 设置错误处理
    trap 'log_error "测试过程中发生错误，退出码: $?"; cleanup; exit 1' ERR
    trap 'log_info "收到中断信号，正在清理..."; cleanup; exit 0' INT TERM
    
    # 执行测试流程
    check_environment
    start_test_environment
    
    # 运行各标准组合的互操作性测试
    run_tsn_opcua_tests
    run_onem2m_wot_tests
    run_matter_tsn_tests
    run_opcua_onem2m_tests
    run_wot_matter_tests
    
    # 运行性能基准测试
    run_performance_tests
    
    # 生成报告和通知
    generate_test_report
    send_notifications
    
    # 计算总耗时
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    log_success "=== IoT跨标准互操作性测试完成 ==="
    log_info "总耗时: ${DURATION}秒"
    log_info "测试报告: $report_file"
    log_info "详细日志: $LOG_FILE"
    
    # 清理资源
    cleanup
    
    exit 0
}

# 脚本入口点
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
