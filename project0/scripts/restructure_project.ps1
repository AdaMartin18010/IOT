# IoT项目重构执行脚本
# 用于重新组织整个项目结构和文档

param(
    [string]$ProjectRoot = ".",
    [switch]$DryRun = $false,
    [switch]$Backup = $true
)

Write-Host "IoT项目重构开始..." -ForegroundColor Green

# 创建备份
if ($Backup) {
    $backupDir = "backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Write-Host "创建备份到: $backupDir" -ForegroundColor Yellow
    Copy-Item -Path $ProjectRoot -Destination $backupDir -Recurse -Force
}

# 新项目结构定义
$newStructure = @{
    "00-项目概述" = @(
        "01-项目愿景与目标.md",
        "02-核心价值主张.md", 
        "03-项目路线图.md",
        "04-成功标准与评估.md",
        "05-项目治理.md",
        "06-贡献指南.md"
    )
    "01-国际标准体系" = @{
        "01-核心互操作标准" = @(
            "01-OPC-UA-1.05-深度解析.md",
            "02-oneM2M-R4-深度解析.md",
            "03-W3C-WoT-1.1-深度解析.md",
            "04-Matter-1.2-深度解析.md"
        )
        "02-语义建模标准" = @(
            "01-W3C-SSN-SOSA-1.1.md",
            "02-Schema.org-22.0.md",
            "03-JSON-LD-1.1.md",
            "04-RDF-OWL-2.md"
        )
        "03-行业特定标准" = @(
            "01-FIWARE-NGSI-LD-1.6.md",
            "02-Digital-Twin-DTDL-2.1.md",
            "03-HL7-FHIR-R5.md",
            "04-ISO-20078-车联网.md"
        )
        "04-标准间关系映射" = @(
            "01-标准兼容性矩阵.md",
            "02-语义映射关系图.md",
            "03-协议转换映射.md",
            "04-数据格式转换.md"
        )
    }
    "02-语义互操作理论" = @{
        "01-语义互操作基础" = @(
            "01-语义互操作定义.md",
            "02-语义层次模型.md",
            "03-语义映射理论.md",
            "04-语义一致性理论.md"
        )
        "02-跨标准语义理论" = @(
            "01-跨标准映射理论.md",
            "02-语义网关理论.md",
            "03-语义注册理论.md",
            "04-动态语义适配理论.md"
        )
        "03-形式化语义模型" = @(
            "01-设备语义模型.md",
            "02-服务语义模型.md",
            "03-交互语义模型.md",
            "04-上下文语义模型.md"
        )
        "04-语义质量保证" = @(
            "01-语义完整性评估.md",
            "02-语义准确性验证.md",
            "03-语义时效性分析.md",
            "04-语义一致性检查.md"
        )
    }
    "03-技术架构设计" = @{
        "01-语义网关架构" = @(
            "01-整体架构设计.md",
            "02-协议适配层.md",
            "03-语义映射层.md",
            "04-服务编排层.md",
            "05-QoS管理层.md"
        )
        "02-标准适配器" = @(
            "01-OPC-UA适配器.md",
            "02-oneM2M适配器.md",
            "03-WoT适配器.md",
            "04-Matter适配器.md"
        )
        "03-语义中间件" = @(
            "01-语义注册中心.md",
            "02-本体管理系统.md",
            "03-映射引擎.md",
            "04-推理引擎.md"
        )
        "04-安全与隐私" = @(
            "01-语义级安全.md",
            "02-数据隐私保护.md",
            "03-访问控制策略.md",
            "04-审计与合规.md"
        )
    }
    "04-实现与开发" = @{
        "01-Rust实现" = @{
            "01-语义网关核心" = @(
                "src/gateway/",
                "src/adapters/",
                "src/mappers/",
                "src/middleware/",
                "Cargo.toml",
                "README.md"
            )
            "02-OPC-UA实现" = @()
            "03-oneM2M实现" = @()
            "04-WoT实现" = @()
        }
        "02-Go实现" = @{
            "01-云原生服务" = @()
            "02-API网关" = @()
            "03-服务编排" = @()
            "04-监控系统" = @()
        }
        "03-Python实现" = @{
            "01-语义分析" = @()
            "02-机器学习" = @()
            "03-数据转换" = @()
            "04-测试工具" = @()
        }
        "04-前端实现" = @{
            "01-管理界面" = @()
            "02-可视化工具" = @()
            "03-配置工具" = @()
            "04-监控面板" = @()
        }
    }
    "05-形式化验证" = @{
        "01-语义一致性验证" = @(
            "01-本体一致性检查.md",
            "02-语义冲突检测.md",
            "03-映射正确性验证.md",
            "04-互操作性测试.md"
        )
        "02-系统行为验证" = @(
            "01-协议正确性验证.md",
            "02-死锁检测.md",
            "03-性能分析.md",
            "04-可靠性验证.md"
        )
        "03-TLA+规范" = @(
            "01-语义网关规范.tla",
            "02-协议转换规范.tla",
            "03-语义映射规范.tla",
            "04-服务编排规范.tla"
        )
        "04-验证工具" = @(
            "01-语义验证工具/",
            "02-互操作性测试工具/",
            "03-性能基准测试/",
            "04-安全验证工具/"
        )
    }
    "06-行业应用" = @{
        "01-工业IoT" = @(
            "01-智能制造场景.md",
            "02-预测性维护.md",
            "03-质量控制.md",
            "04-能源管理.md"
        )
        "02-智慧城市" = @(
            "01-交通管理.md",
            "02-环境监控.md",
            "03-能源管理.md",
            "04-公共安全.md"
        )
        "03-智能家居" = @(
            "01-设备互联.md",
            "02-场景控制.md",
            "03-能源优化.md",
            "04-安全监控.md"
        )
        "04-医疗IoT" = @(
            "01-设备互操作.md",
            "02-数据共享.md",
            "03-远程医疗.md",
            "04-健康监测.md"
        )
    }
    "07-测试与部署" = @{
        "01-测试策略" = @(
            "01-单元测试.md",
            "02-集成测试.md",
            "03-系统测试.md",
            "04-验收测试.md"
        )
        "02-测试环境" = @(
            "01-开发环境.md",
            "02-测试环境.md",
            "03-预生产环境.md",
            "04-生产环境.md"
        )
        "03-部署方案" = @(
            "01-Docker部署.md",
            "02-Kubernetes部署.md",
            "03-云原生部署.md",
            "04-边缘部署.md"
        )
        "04-运维监控" = @(
            "01-监控体系.md",
            "02-日志管理.md",
            "03-告警系统.md",
            "04-故障处理.md"
        )
    }
    "08-文档与规范" = @{
        "01-技术文档" = @(
            "01-架构设计文档.md",
            "02-API文档.md",
            "03-部署文档.md",
            "04-运维文档.md"
        )
        "02-用户文档" = @(
            "01-用户手册.md",
            "02-管理员手册.md",
            "03-开发者指南.md",
            "04-故障排除指南.md"
        )
        "03-标准规范" = @(
            "01-编码规范.md",
            "02-接口规范.md",
            "03-数据规范.md",
            "04-安全规范.md"
        )
        "04-培训材料" = @(
            "01-技术培训/",
            "02-用户培训/",
            "03-运维培训/",
            "04-认证考试/"
        )
    }
    "09-项目管理" = @{
        "01-项目计划" = @(
            "01-总体计划.md",
            "02-阶段计划.md",
            "03-里程碑计划.md",
            "04-资源计划.md"
        )
        "02-风险管理" = @(
            "01-风险识别.md",
            "02-风险评估.md",
            "03-风险应对.md",
            "04-风险监控.md"
        )
        "03-质量管理" = @(
            "01-质量计划.md",
            "02-质量保证.md",
            "03-质量控制.md",
            "04-质量改进.md"
        )
        "04-变更管理" = @(
            "01-变更控制.md",
            "02-版本管理.md",
            "03-配置管理.md",
            "04-发布管理.md"
        )
    }
    "10-附录" = @(
        "01-术语表.md",
        "02-缩写表.md",
        "03-参考标准.md",
        "04-工具清单.md",
        "05-资源链接.md"
    )
}

# 创建目录结构函数
function Create-DirectoryStructure {
    param($structure, $basePath)
    
    foreach ($item in $structure.GetEnumerator()) {
        $path = Join-Path $basePath $item.Key
        
        if ($DryRun) {
            Write-Host "将创建目录: $path" -ForegroundColor Cyan
        } else {
            if (!(Test-Path $path)) {
                New-Item -ItemType Directory -Path $path -Force | Out-Null
                Write-Host "创建目录: $path" -ForegroundColor Green
            }
        }
        
        if ($item.Value -is [hashtable]) {
            Create-DirectoryStructure -structure $item.Value -basePath $path
        } elseif ($item.Value -is [array]) {
            foreach ($file in $item.Value) {
                $filePath = Join-Path $path $file
                if ($DryRun) {
                    Write-Host "将创建文件: $filePath" -ForegroundColor Cyan
                } else {
                    if (!(Test-Path $filePath)) {
                        New-Item -ItemType File -Path $filePath -Force | Out-Null
                        Write-Host "创建文件: $filePath" -ForegroundColor Green
                    }
                }
            }
        }
    }
}

# 迁移现有文档函数
function Migrate-ExistingDocuments {
    Write-Host "开始迁移现有文档..." -ForegroundColor Yellow
    
    # 迁移策略映射
    $migrationMap = @{
        "docs/Analysis/01-Core-Architecture/IoT-Layered-Architecture-Formal-Analysis.md" = "03-技术架构设计/01-语义网关架构/01-整体架构设计.md"
        "docs/Analysis/03-Algorithms/IoT-Data-Stream-Processing-Formal-Analysis.md" = "02-语义互操作理论/03-形式化语义模型/03-交互语义模型.md"
        "docs/Analysis/04-Technology/Rust-Golang-Technology-Stack-Formal-Analysis.md" = "04-实现与开发/01-Rust实现/01-语义网关核心/README.md"
        "docs/Analysis/05-Specialized-Research/IoT-Edge-Intelligence-Federated-Learning-Analysis.md" = "06-行业应用/01-工业IoT/02-预测性维护.md"
        "docs/Analysis/06-Security-Architecture/IoMT-Security-Architecture-Formal-Analysis.md" = "03-技术架构设计/04-安全与隐私/01-语义级安全.md"
        "docs/Analysis/07-Advanced-Communication/IoT-LPWAN-Optimization-Analysis.md" = "01-国际标准体系/01-核心互操作标准/02-oneM2M-R4-深度解析.md"
        "docs/Analysis/08-Industry-Applications/Smart-City-IoT-Platform-Analysis.md" = "06-行业应用/02-智慧城市/01-交通管理.md"
        "docs/Matter/code/navigate/" = "04-实现与开发/04-前端实现/01-管理界面/"
        "docs/Theory/Advanced_Control_Theory_Extended.md" = "02-语义互操作理论/01-语义互操作基础/02-语义层次模型.md"
    }
    
    foreach ($mapping in $migrationMap.GetEnumerator()) {
        $sourcePath = Join-Path $ProjectRoot $mapping.Key
        $targetPath = Join-Path $ProjectRoot $mapping.Value
        
        if (Test-Path $sourcePath) {
            if ($DryRun) {
                Write-Host "将迁移: $sourcePath -> $targetPath" -ForegroundColor Cyan
            } else {
                $targetDir = Split-Path $targetPath -Parent
                if (!(Test-Path $targetDir)) {
                    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
                }
                Copy-Item -Path $sourcePath -Destination $targetPath -Force
                Write-Host "迁移: $sourcePath -> $targetPath" -ForegroundColor Green
            }
        }
    }
}

# 创建文档模板函数
function Create-DocumentTemplates {
    Write-Host "创建文档模板..." -ForegroundColor Yellow
    
    $templateContent = @"
# 文档标题

## 概述
简要描述本文档的目的和内容。

## 版本信息
- **版本**: 1.0.0
- **创建日期**: $(Get-Date -Format 'yyyy-MM-dd')
- **最后更新**: $(Get-Date -Format 'yyyy-MM-dd')
- **作者**: 
- **状态**: 草稿

## 目录
1. [章节1](#章节1)
2. [章节2](#章节2)
3. [章节3](#章节3)

## 章节1
内容描述...

## 章节2
内容描述...

## 章节3
内容描述...

## 参考文献
1. 参考文献1
2. 参考文献2

## 附录
附录内容...
"@

    # 为每个主要目录创建模板
    $templateDirs = @(
        "00-项目概述",
        "01-国际标准体系/01-核心互操作标准",
        "02-语义互操作理论/01-语义互操作基础",
        "03-技术架构设计/01-语义网关架构"
    )
    
    foreach ($dir in $templateDirs) {
        $templatePath = Join-Path $ProjectRoot $dir "文档模板.md"
        if ($DryRun) {
            Write-Host "将创建模板: $templatePath" -ForegroundColor Cyan
        } else {
            if (!(Test-Path $templatePath)) {
                $templateContent | Out-File -FilePath $templatePath -Encoding UTF8
                Write-Host "创建模板: $templatePath" -ForegroundColor Green
            }
        }
    }
}

# 创建配置文件函数
function Create-ConfigurationFiles {
    Write-Host "创建配置文件..." -ForegroundColor Yellow
    
    # Cargo.toml
    $cargoToml = @"
[package]
name = "iot-semantic-gateway"
version = "0.1.0"
edition = "2021"
authors = ["IoT Team"]
description = "IoT Semantic Interoperability Gateway"

[dependencies]
tokio = { version = "1.28", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
reqwest = { version = "0.11", features = ["json"] }
tracing = "0.1"
tracing-subscriber = "0.3"

[dev-dependencies]
tokio-test = "0.4"
"@

    # go.mod
    $goMod = @"
module iot-semantic-gateway

go 1.21

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/spf13/viper v1.16.0
    go.uber.org/zap v1.24.0
)
"@

    # requirements.txt
    $requirementsTxt = @"
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
requests==2.31.0
pytest==7.4.3
pytest-asyncio==0.21.1
"@

    # package.json
    $packageJson = @"
{
  "name": "iot-semantic-gateway-ui",
  "version": "0.1.0",
  "description": "IoT Semantic Gateway User Interface",
  "main": "index.js",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "vue": "^3.3.8",
    "vue-router": "^4.2.5",
    "pinia": "^2.1.7"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^4.5.0",
    "vite": "^5.0.0"
  }
}
"@

    $configs = @{
        "Cargo.toml" = $cargoToml
        "go.mod" = $goMod
        "requirements.txt" = $requirementsTxt
        "package.json" = $packageJson
    }
    
    foreach ($config in $configs.GetEnumerator()) {
        $configPath = Join-Path $ProjectRoot $config.Key
        if ($DryRun) {
            Write-Host "将创建配置: $configPath" -ForegroundColor Cyan
        } else {
            if (!(Test-Path $configPath)) {
                $config.Value | Out-File -FilePath $configPath -Encoding UTF8
                Write-Host "创建配置: $configPath" -ForegroundColor Green
            }
        }
    }
}

# 主执行流程
try {
    Write-Host "开始IoT项目重构..." -ForegroundColor Green
    Write-Host "项目根目录: $ProjectRoot" -ForegroundColor Yellow
    Write-Host "干运行模式: $DryRun" -ForegroundColor Yellow
    
    # 1. 创建新目录结构
    Write-Host "`n步骤1: 创建新目录结构" -ForegroundColor Magenta
    Create-DirectoryStructure -structure $newStructure -basePath $ProjectRoot
    
    # 2. 迁移现有文档
    Write-Host "`n步骤2: 迁移现有文档" -ForegroundColor Magenta
    Migrate-ExistingDocuments
    
    # 3. 创建文档模板
    Write-Host "`n步骤3: 创建文档模板" -ForegroundColor Magenta
    Create-DocumentTemplates
    
    # 4. 创建配置文件
    Write-Host "`n步骤4: 创建配置文件" -ForegroundColor Magenta
    Create-ConfigurationFiles
    
    Write-Host "`nIoT项目重构完成!" -ForegroundColor Green
    
    if ($DryRun) {
        Write-Host "`n注意: 这是干运行模式，没有实际创建文件。" -ForegroundColor Yellow
        Write-Host "要实际执行重构，请运行: .\restructure_project.ps1" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "重构过程中发生错误: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} 