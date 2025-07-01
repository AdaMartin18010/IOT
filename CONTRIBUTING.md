# 贡献指南

感谢您对IoT语义平台项目的关注！我们欢迎所有形式的贡献，包括但不限于代码提交、文档改进、问题报告和功能建议。

## 🤝 如何贡献

### 报告问题

如果您发现了bug或有功能建议，请：

1. 首先搜索现有的 [Issues](https://github.com/iot-semantic-platform/iot-system/issues) 确认问题未被报告
2. 使用相应的问题模板创建新的Issue
3. 提供详细的问题描述、复现步骤和环境信息
4. 如果是bug报告，请包含错误日志和截图

### 提交代码

1. **Fork项目**: 点击右上角的Fork按钮
2. **克隆仓库**:

   ```bash
   git clone https://github.com/your-username/iot-system.git
   cd iot-system
   ```

3. **创建分支**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **进行开发**: 按照下面的开发规范进行开发
5. **提交更改**:

   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

6. **推送分支**:

   ```bash
   git push origin feature/your-feature-name
   ```

7. **创建PR**: 在GitHub上创建Pull Request

## 📝 开发规范

### 代码风格

#### Rust代码规范

- 使用 `cargo fmt` 格式化代码
- 使用 `cargo clippy` 进行代码检查
- 遵循 [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- 函数和变量使用snake_case命名
- 结构体和枚举使用PascalCase命名
- 常量使用SCREAMING_SNAKE_CASE命名

```rust
// 好的示例
pub struct DeviceManager {
    active_devices: HashMap<String, Device>,
    max_connections: usize,
}

impl DeviceManager {
    pub fn new(max_connections: usize) -> Self {
        Self {
            active_devices: HashMap::new(),
            max_connections,
        }
    }
    
    pub async fn register_device(&mut self, device: Device) -> Result<(), DeviceError> {
        // 实现逻辑
    }
}
```

#### 文档注释

- 所有公共API必须有文档注释
- 使用标准的Rust文档格式
- 包含使用示例

```rust
/// 设备管理器，负责IoT设备的注册、管理和监控
/// 
/// # Examples
/// 
/// ```
/// use iot_system::DeviceManager;
/// 
/// let mut manager = DeviceManager::new(1000);
/// let device = Device::new("sensor-001", DeviceType::Sensor);
/// manager.register_device(device).await?;
/// ```
pub struct DeviceManager {
    // ...
}
```

### 提交信息规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```text
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### 类型说明

- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

#### 示例

```text
feat(gateway): add MQTT protocol support

Add support for MQTT v5.0 protocol in the gateway service.
This includes connection handling, message parsing, and QoS management.

Closes #123
```

### 测试要求

#### 单元测试

- 所有新功能必须包含单元测试
- 测试覆盖率不低于80%
- 使用有意义的测试名称

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_device_registration_success() {
        let mut manager = DeviceManager::new(10);
        let device = create_test_device();
        
        let result = manager.register_device(device.clone()).await;
        
        assert!(result.is_ok());
        assert_eq!(manager.device_count(), 1);
        assert!(manager.get_device(&device.id).is_some());
    }
    
    #[tokio::test]
    async fn test_device_registration_exceeds_limit() {
        let mut manager = DeviceManager::new(1);
        let device1 = create_test_device_with_id("device-1");
        let device2 = create_test_device_with_id("device-2");
        
        manager.register_device(device1).await.unwrap();
        let result = manager.register_device(device2).await;
        
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), DeviceError::MaxConnectionsReached);
    }
}
```

#### 集成测试

- 重要功能需要集成测试
- 使用testcontainers进行数据库和外部服务测试

```rust
#[tokio::test]
async fn test_gateway_database_integration() {
    let docker = Cli::default();
    let postgres_container = docker.run(Postgres::default());
    
    let database_url = format!(
        "postgres://postgres:postgres@localhost:{}/postgres",
        postgres_container.get_host_port_ipv4(5432)
    );
    
    let gateway = Gateway::new_with_database(&database_url).await.unwrap();
    
    // 测试逻辑
}
```

### 性能要求

- 关键路径的性能不能回退
- 新功能需要包含性能基准测试
- 使用criterion进行基准测试

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_message_processing(c: &mut Criterion) {
    c.bench_function("process_message", |b| {
        let gateway = create_test_gateway();
        let message = create_test_message();
        
        b.iter(|| {
            black_box(gateway.process_message(black_box(message.clone())))
        })
    });
}

criterion_group!(benches, benchmark_message_processing);
criterion_main!(benches);
```

## 🔧 开发环境设置

### 环境要求

- Rust 1.70+
- Docker 20.10+
- PostgreSQL 13+
- Redis 6+

### 设置步骤

1. **安装Rust**:

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

2. **安装开发工具**:

   ```bash
   rustup component add rustfmt clippy
   cargo install cargo-watch cargo-tarpaulin
   ```

3. **启动开发服务**:

   ```bash
   docker-compose -f docker-compose.dev.yml up -d
   ```

4. **运行测试**:

   ```bash
   cargo test
   ```

5. **启动开发服务器**:

   ```bash
   cargo watch -x run
   ```

### 开发工具

#### VS Code扩展推荐

- rust-analyzer
- CodeLLDB
- Better TOML
- Docker
- GitLens

#### 配置文件

创建 `.vscode/settings.json`:

```json
{
    "rust-analyzer.cargo.features": "all",
    "rust-analyzer.checkOnSave.command": "clippy",
    "editor.formatOnSave": true,
    "[rust]": {
        "editor.defaultFormatter": "rust-lang.rust-analyzer"
    }
}
```

## 📋 Pull Request检查清单

在提交PR之前，请确保：

- [ ] 代码通过所有测试 (`cargo test`)
- [ ] 代码通过格式检查 (`cargo fmt --check`)
- [ ] 代码通过Clippy检查 (`cargo clippy -- -D warnings`)
- [ ] 新功能包含适当的测试
- [ ] 文档已更新（如果需要）
- [ ] 提交信息遵循规范
- [ ] PR描述清晰，包含变更说明

### PR模板

```markdown
## 变更描述
简要描述此PR的变更内容

## 变更类型
- [ ] Bug修复
- [ ] 新功能
- [ ] 破坏性变更
- [ ] 文档更新
- [ ] 性能改进
- [ ] 代码重构

## 测试
- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 手动测试完成

## 检查清单
- [ ] 代码遵循项目规范
- [ ] 自我审查完成
- [ ] 注释清晰易懂
- [ ] 文档已更新

## 相关Issue
Closes #(issue number)
```

## 🏷️ 发布流程

### 版本号规范

使用 [Semantic Versioning](https://semver.org/):

- `MAJOR.MINOR.PATCH`
- `MAJOR`: 破坏性变更
- `MINOR`: 新功能，向后兼容
- `PATCH`: Bug修复，向后兼容

### 发布步骤

1. 更新版本号和CHANGELOG
2. 创建发布分支
3. 运行完整测试套件
4. 创建Git标签
5. 发布到crates.io（如果适用）
6. 更新文档

## 🌟 贡献者认可

我们使用 [All Contributors](https://allcontributors.org/) 来认可所有贡献者的努力。

贡献类型包括：

- 💻 代码
- 📖 文档
- 🐛 Bug报告
- 💡 想法和规划
- 🤔 答疑解惑
- 🎨 设计
- 📢 推广

## 📞 获得帮助

如果您在贡献过程中遇到问题：

1. 查看现有的 [Issues](https://github.com/iot-semantic-platform/iot-system/issues)
2. 在 [Discussions](https://github.com/iot-semantic-platform/iot-system/discussions) 中提问
3. 加入我们的 [Discord社区](https://discord.gg/iot-semantic-platform)
4. 发送邮件至 [contributors@iot-semantic-platform.org](mailto:contributors@iot-semantic-platform.org)

## 📄 行为准则

我们致力于创建一个开放、友好、多元化和包容的社区环境。请阅读我们的 [行为准则](CODE_OF_CONDUCT.md)。

## 🙏 感谢

感谢您考虑为IoT语义平台项目做出贡献！每一个贡献都很宝贵，无论大小。让我们一起构建更好的IoT生态系统！

---

-*最后更新: 2024年12月*
