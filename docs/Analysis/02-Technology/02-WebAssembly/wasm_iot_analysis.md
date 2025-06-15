# WebAssembly在IOT领域的分析

## 目录

1. [概述](#概述)
2. [WebAssembly理论基础](#webassembly理论基础)
3. [IOT应用架构](#iot应用架构)
4. [性能与资源分析](#性能与资源分析)
5. [安全模型](#安全模型)
6. [实际应用案例](#实际应用案例)
7. [结论](#结论)

## 概述

WebAssembly (WASM) 作为一种轻量级、跨平台的执行环境，在IOT领域展现出独特的价值。本文档从形式化角度分析WASM在IOT系统中的应用，包括理论基础、架构设计和实际应用。

## WebAssembly理论基础

### 2.1 WASM形式化模型

**定义 2.1.1 (WASM模块)**
WASM模块是一个六元组 $\mathcal{W} = (T, F, M, G, E, I)$，其中：

- $T$ 是类型定义集合
- $F$ 是函数集合
- $M$ 是内存定义
- $G$ 是全局变量集合
- $E$ 是导出接口集合
- $I$ 是导入接口集合

**定义 2.1.2 (WASM执行环境)**
WASM执行环境是一个四元组 $\mathcal{E} = (S, V, H, R)$，其中：

- $S$ 是栈状态
- $V$ 是局部变量
- $H$ 是堆内存
- $R$ 是运行时

**定理 2.1.1 (WASM类型安全)**
WASM执行环境保证类型安全。

**证明：**
通过类型检查：

1. **静态类型检查**：编译时验证所有操作的类型正确性
2. **动态类型检查**：运行时验证内存访问和函数调用
3. **边界检查**：确保内存访问在有效范围内
4. **栈平衡**：确保函数调用前后栈状态一致

### 2.2 WASM指令集

**定义 2.2.1 (WASM指令)**
WASM指令集包括：

1. **数值指令**：`i32.add`, `i64.mul`, `f32.div` 等
2. **内存指令**：`i32.load`, `i64.store` 等
3. **控制指令**：`block`, `loop`, `if`, `br` 等
4. **函数指令**：`call`, `call_indirect` 等

**定义 2.2.2 (WASM执行语义)**
对于指令序列 $I_1, I_2, \ldots, I_n$，执行语义定义为：
$$\text{Execute}(I_1, I_2, \ldots, I_n) = \text{Execute}(I_n, \text{Execute}(I_1, I_2, \ldots, I_{n-1}))$$

**定理 2.2.1 (WASM确定性)**
WASM执行是确定性的。

**证明：**
通过语义定义：

1. **指令语义**：每个指令都有明确的语义
2. **状态转换**：状态转换是确定性的
3. **无副作用**：纯函数调用不产生副作用
4. **顺序执行**：指令按顺序执行

## IOT应用架构

### 3.1 WASM在IOT中的架构模型

**定义 3.1.1 (WASM-IOT架构)**
WASM-IOT架构是一个五元组 $\mathcal{A} = (H, G, W, I, C)$，其中：

- $H$ 是主机环境
- $G$ 是网关层
- $W$ 是WASM运行时
- $I$ 是IOT设备接口
- $C$ 是通信层

**架构实现**：

```rust
// WASM-IOT架构实现
pub struct WasmIoTSystem {
    host_environment: HostEnvironment,
    wasm_runtime: WasmRuntime,
    device_interface: DeviceInterface,
    communication_layer: CommunicationLayer,
}

impl WasmIoTSystem {
    pub fn new() -> Result<Self, SystemError> {
        let host = HostEnvironment::new()?;
        let runtime = WasmRuntime::new()?;
        let device_interface = DeviceInterface::new()?;
        let communication = CommunicationLayer::new()?;
        
        Ok(Self {
            host_environment: host,
            wasm_runtime: runtime,
            device_interface: device_interface,
            communication_layer: communication,
        })
    }
    
    pub async fn load_wasm_module(&mut self, wasm_bytes: &[u8]) -> Result<(), LoadError> {
        // 1. 验证WASM模块
        let module = self.validate_wasm_module(wasm_bytes)?;
        
        // 2. 实例化模块
        let instance = self.wasm_runtime.instantiate(module).await?;
        
        // 3. 绑定设备接口
        self.bind_device_interface(&instance)?;
        
        // 4. 启动模块
        self.start_module(&instance).await?;
        
        Ok(())
    }
    
    fn validate_wasm_module(&self, wasm_bytes: &[u8]) -> Result<WasmModule, ValidationError> {
        // 验证WASM模块格式和安全性
        let module = WasmModule::from_bytes(wasm_bytes)?;
        
        // 检查模块大小限制
        if wasm_bytes.len() > MAX_MODULE_SIZE {
            return Err(ValidationError::ModuleTooLarge);
        }
        
        // 检查内存限制
        if module.memory_size() > MAX_MEMORY_SIZE {
            return Err(ValidationError::MemoryTooLarge);
        }
        
        // 检查函数数量限制
        if module.function_count() > MAX_FUNCTION_COUNT {
            return Err(ValidationError::TooManyFunctions);
        }
        
        Ok(module)
    }
    
    fn bind_device_interface(&mut self, instance: &WasmInstance) -> Result<(), BindingError> {
        // 绑定传感器接口
        instance.bind_function("read_sensor", |sensor_id: u32| -> f32 {
            self.device_interface.read_sensor(sensor_id)
        })?;
        
        // 绑定执行器接口
        instance.bind_function("write_actuator", |actuator_id: u32, value: f32| -> bool {
            self.device_interface.write_actuator(actuator_id, value)
        })?;
        
        // 绑定网络接口
        instance.bind_function("send_data", |data_ptr: u32, data_len: u32| -> bool {
            let data = self.get_memory_slice(data_ptr, data_len);
            self.communication_layer.send(data).is_ok()
        })?;
        
        Ok(())
    }
}
```

### 3.2 边缘计算架构

**定义 3.2.1 (WASM边缘计算)**
WASM边缘计算是一个四元组 $\mathcal{E} = (N, P, S, D)$，其中：

- $N$ 是边缘节点
- $P$ 是处理单元
- $S$ 是存储单元
- $D$ 是分发器

**实现架构**：

```rust
// WASM边缘计算节点
pub struct WasmEdgeNode {
    node_id: NodeId,
    wasm_runtime: WasmRuntime,
    processing_units: Vec<ProcessingUnit>,
    storage: LocalStorage,
    data_distributor: DataDistributor,
}

impl WasmEdgeNode {
    pub async fn process_data(&mut self, data: &[u8]) -> Result<Vec<u8>, ProcessingError> {
        // 1. 数据预处理
        let preprocessed_data = self.preprocess_data(data)?;
        
        // 2. 选择处理单元
        let processing_unit = self.select_processing_unit(&preprocessed_data)?;
        
        // 3. 加载WASM模块
        let wasm_module = self.load_processing_module(processing_unit).await?;
        
        // 4. 执行数据处理
        let result = self.execute_processing(&wasm_module, &preprocessed_data).await?;
        
        // 5. 后处理结果
        let final_result = self.postprocess_result(&result)?;
        
        Ok(final_result)
    }
    
    fn select_processing_unit(&self, data: &[u8]) -> Result<&ProcessingUnit, SelectionError> {
        // 基于数据类型和处理需求选择最合适的处理单元
        let data_type = self.analyze_data_type(data);
        let processing_requirement = self.analyze_processing_requirement(data);
        
        for unit in &self.processing_units {
            if unit.can_handle(data_type, processing_requirement) {
                return Ok(unit);
            }
        }
        
        Err(SelectionError::NoSuitableUnit)
    }
    
    async fn execute_processing(
        &self,
        wasm_module: &WasmModule,
        data: &[u8],
    ) -> Result<Vec<u8>, ExecutionError> {
        // 创建WASM实例
        let instance = self.wasm_runtime.instantiate(wasm_module).await?;
        
        // 准备输入数据
        let input_ptr = instance.allocate_memory(data.len())?;
        instance.write_memory(input_ptr, data)?;
        
        // 调用处理函数
        let output_ptr = instance.call_function("process_data", &[input_ptr, data.len() as u32])?;
        
        // 读取输出数据
        let output_len = instance.call_function("get_output_length", &[])?;
        let output_data = instance.read_memory(output_ptr, output_len as usize)?;
        
        // 释放内存
        instance.free_memory(input_ptr)?;
        instance.free_memory(output_ptr)?;
        
        Ok(output_data)
    }
}
```

## 性能与资源分析

### 4.1 性能模型

**定义 4.1.1 (WASM性能模型)**
WASM性能模型是一个三元组 $\mathcal{P} = (T, M, E)$，其中：

- $T$ 是执行时间
- $M$ 是内存使用
- $E$ 是能量消耗

**定理 4.1.1 (WASM性能优势)**
WASM在IOT环境中具有性能优势。

**证明：**
通过性能分析：

1. **编译优化**：WASM字节码经过优化编译
2. **内存效率**：紧凑的内存布局和访问模式
3. **执行效率**：接近原生代码的执行速度
4. **启动时间**：快速的模块加载和初始化

**性能测试**：

```rust
// WASM性能测试框架
pub struct WasmPerformanceTest {
    test_cases: Vec<PerformanceTestCase>,
    metrics_collector: MetricsCollector,
}

impl WasmPerformanceTest {
    pub async fn run_performance_tests(&mut self) -> Result<PerformanceReport, TestError> {
        let mut report = PerformanceReport::new();
        
        for test_case in &self.test_cases {
            // 1. 准备测试环境
            let test_env = self.prepare_test_environment(test_case).await?;
            
            // 2. 执行测试
            let start_time = std::time::Instant::now();
            let start_memory = self.get_memory_usage();
            
            let result = self.execute_test_case(test_case, &test_env).await?;
            
            let end_time = std::time::Instant::now();
            let end_memory = self.get_memory_usage();
            
            // 3. 收集指标
            let execution_time = end_time.duration_since(start_time);
            let memory_usage = end_memory - start_memory;
            let energy_consumption = self.measure_energy_consumption().await?;
            
            // 4. 记录结果
            let metrics = PerformanceMetrics {
                execution_time,
                memory_usage,
                energy_consumption,
                success: result.is_ok(),
            };
            
            report.add_metrics(test_case.name.clone(), metrics);
        }
        
        Ok(report)
    }
    
    async fn execute_test_case(
        &self,
        test_case: &PerformanceTestCase,
        env: &TestEnvironment,
    ) -> Result<Vec<u8>, ExecutionError> {
        match test_case.test_type {
            TestType::DataProcessing => {
                self.test_data_processing(test_case, env).await
            }
            TestType::AlgorithmExecution => {
                self.test_algorithm_execution(test_case, env).await
            }
            TestType::Communication => {
                self.test_communication(test_case, env).await
            }
        }
    }
}
```

### 4.2 资源管理

**定义 4.2.1 (WASM资源管理)**
WASM资源管理是一个四元组 $\mathcal{R} = (M, C, T, E)$，其中：

- $M$ 是内存管理
- $C$ 是CPU管理
- $T$ 是时间管理
- $E$ 是能量管理

**资源管理实现**：

```rust
// WASM资源管理器
pub struct WasmResourceManager {
    memory_manager: MemoryManager,
    cpu_manager: CpuManager,
    time_manager: TimeManager,
    energy_manager: EnergyManager,
}

impl WasmResourceManager {
    pub fn new() -> Self {
        Self {
            memory_manager: MemoryManager::new(),
            cpu_manager: CpuManager::new(),
            time_manager: TimeManager::new(),
            energy_manager: EnergyManager::new(),
        }
    }
    
    pub fn allocate_resources(&mut self, requirements: &ResourceRequirements) -> Result<ResourceAllocation, AllocationError> {
        // 1. 检查内存可用性
        let memory_allocation = self.memory_manager.allocate(requirements.memory_size)?;
        
        // 2. 检查CPU可用性
        let cpu_allocation = self.cpu_manager.allocate(requirements.cpu_cores)?;
        
        // 3. 检查时间约束
        let time_allocation = self.time_manager.allocate(requirements.execution_time)?;
        
        // 4. 检查能量约束
        let energy_allocation = self.energy_manager.allocate(requirements.energy_budget)?;
        
        Ok(ResourceAllocation {
            memory: memory_allocation,
            cpu: cpu_allocation,
            time: time_allocation,
            energy: energy_allocation,
        })
    }
    
    pub fn monitor_resources(&self) -> ResourceStatus {
        ResourceStatus {
            memory_usage: self.memory_manager.get_usage(),
            cpu_usage: self.cpu_manager.get_usage(),
            time_remaining: self.time_manager.get_remaining(),
            energy_remaining: self.energy_manager.get_remaining(),
        }
    }
    
    pub fn release_resources(&mut self, allocation: &ResourceAllocation) {
        self.memory_manager.release(&allocation.memory);
        self.cpu_manager.release(&allocation.cpu);
        self.time_manager.release(&allocation.time);
        self.energy_manager.release(&allocation.energy);
    }
}

// 内存管理器
pub struct MemoryManager {
    total_memory: usize,
    used_memory: usize,
    memory_pool: Vec<MemoryBlock>,
}

impl MemoryManager {
    pub fn allocate(&mut self, size: usize) -> Result<MemoryAllocation, MemoryError> {
        // 查找合适的空闲块
        for block in &mut self.memory_pool {
            if block.is_free() && block.size() >= size {
                let allocation = block.allocate(size)?;
                self.used_memory += allocation.size();
                return Ok(allocation);
            }
        }
        
        // 如果没有合适的块，尝试创建新块
        if self.used_memory + size <= self.total_memory {
            let new_block = MemoryBlock::new(size);
            let allocation = new_block.allocate(size)?;
            self.memory_pool.push(new_block);
            self.used_memory += allocation.size();
            Ok(allocation)
        } else {
            Err(MemoryError::InsufficientMemory)
        }
    }
    
    pub fn release(&mut self, allocation: &MemoryAllocation) {
        // 查找并释放内存块
        for block in &mut self.memory_pool {
            if block.contains_allocation(allocation) {
                block.release(allocation);
                self.used_memory -= allocation.size();
                break;
            }
        }
    }
}
```

## 安全模型

### 5.1 WASM安全特性

**定义 5.1.1 (WASM安全模型)**
WASM安全模型是一个五元组 $\mathcal{S} = (I, A, V, E, M)$，其中：

- $I$ 是隔离机制
- $A$ 是访问控制
- $V$ 是验证机制
- $E$ 是执行环境
- $M$ 是内存保护

**定理 5.1.1 (WASM安全保证)**
WASM提供强大的安全保证。

**证明：**
通过安全机制：

1. **沙箱隔离**：每个WASM模块运行在独立沙箱中
2. **内存隔离**：模块间内存完全隔离
3. **类型安全**：编译时和运行时类型检查
4. **边界检查**：所有内存访问都有边界检查

**安全实现**：

```rust
// WASM安全管理器
pub struct WasmSecurityManager {
    sandbox_manager: SandboxManager,
    access_controller: AccessController,
    validator: ModuleValidator,
    memory_protector: MemoryProtector,
}

impl WasmSecurityManager {
    pub fn new() -> Self {
        Self {
            sandbox_manager: SandboxManager::new(),
            access_controller: AccessController::new(),
            validator: ModuleValidator::new(),
            memory_protector: MemoryProtector::new(),
        }
    }
    
    pub fn validate_module(&self, wasm_bytes: &[u8]) -> Result<ValidationResult, ValidationError> {
        // 1. 格式验证
        self.validator.validate_format(wasm_bytes)?;
        
        // 2. 类型验证
        self.validator.validate_types(wasm_bytes)?;
        
        // 3. 内存验证
        self.validator.validate_memory(wasm_bytes)?;
        
        // 4. 函数验证
        self.validator.validate_functions(wasm_bytes)?;
        
        // 5. 安全策略验证
        self.validator.validate_security_policy(wasm_bytes)?;
        
        Ok(ValidationResult::Valid)
    }
    
    pub fn create_sandbox(&mut self, module: &WasmModule) -> Result<Sandbox, SandboxError> {
        // 创建隔离的沙箱环境
        let sandbox = self.sandbox_manager.create_sandbox(module)?;
        
        // 设置访问控制策略
        self.access_controller.set_policy(&sandbox, &self.get_security_policy())?;
        
        // 配置内存保护
        self.memory_protector.configure(&sandbox)?;
        
        Ok(sandbox)
    }
    
    pub fn monitor_execution(&self, sandbox: &Sandbox) -> SecurityStatus {
        SecurityStatus {
            memory_accesses: self.memory_protector.get_access_count(),
            function_calls: self.access_controller.get_call_count(),
            violations: self.sandbox_manager.get_violations(),
            is_secure: self.sandbox_manager.is_secure(),
        }
    }
}

// 沙箱管理器
pub struct SandboxManager {
    sandboxes: HashMap<SandboxId, Sandbox>,
    security_policies: HashMap<SandboxId, SecurityPolicy>,
}

impl SandboxManager {
    pub fn create_sandbox(&mut self, module: &WasmModule) -> Result<Sandbox, SandboxError> {
        let sandbox_id = SandboxId::new();
        
        // 创建隔离的执行环境
        let sandbox = Sandbox::new(sandbox_id, module.clone())?;
        
        // 设置默认安全策略
        let policy = SecurityPolicy::default();
        self.security_policies.insert(sandbox_id, policy);
        
        self.sandboxes.insert(sandbox_id, sandbox.clone());
        
        Ok(sandbox)
    }
    
    pub fn is_secure(&self) -> bool {
        // 检查所有沙箱是否安全
        for sandbox in self.sandboxes.values() {
            if !sandbox.is_secure() {
                return false;
            }
        }
        true
    }
}
```

### 5.2 安全更新机制

**定义 5.2.1 (安全更新)**
安全更新是一个三元组 $\mathcal{U} = (V, S, R)$，其中：

- $V$ 是版本管理
- $S$ 是签名验证
- $R$ 是回滚机制

**更新实现**：

```rust
// 安全更新管理器
pub struct SecureUpdateManager {
    version_manager: VersionManager,
    signature_verifier: SignatureVerifier,
    rollback_manager: RollbackManager,
}

impl SecureUpdateManager {
    pub async fn update_module(
        &mut self,
        current_module: &WasmModule,
        new_module_bytes: &[u8],
        signature: &[u8],
    ) -> Result<UpdateResult, UpdateError> {
        // 1. 验证签名
        self.signature_verifier.verify(new_module_bytes, signature)?;
        
        // 2. 验证模块
        let new_module = self.validate_new_module(new_module_bytes)?;
        
        // 3. 检查版本兼容性
        self.version_manager.check_compatibility(current_module, &new_module)?;
        
        // 4. 创建备份
        let backup = self.create_backup(current_module)?;
        
        // 5. 执行更新
        match self.perform_update(current_module, &new_module).await {
            Ok(()) => {
                self.rollback_manager.clear_backup(&backup);
                Ok(UpdateResult::Success)
            }
            Err(error) => {
                // 回滚到备份
                self.rollback_manager.rollback(&backup)?;
                Err(error)
            }
        }
    }
    
    fn validate_new_module(&self, module_bytes: &[u8]) -> Result<WasmModule, ValidationError> {
        // 验证模块格式和安全性
        let module = WasmModule::from_bytes(module_bytes)?;
        
        // 检查模块大小
        if module_bytes.len() > MAX_UPDATE_SIZE {
            return Err(ValidationError::ModuleTooLarge);
        }
        
        // 检查内存限制
        if module.memory_size() > MAX_MEMORY_SIZE {
            return Err(ValidationError::MemoryTooLarge);
        }
        
        // 检查函数限制
        if module.function_count() > MAX_FUNCTION_COUNT {
            return Err(ValidationError::TooManyFunctions);
        }
        
        Ok(module)
    }
}
```

## 实际应用案例

### 6.1 智能传感器数据处理

**案例 6.1.1 (温度传感器数据处理)**

```rust
// WASM温度传感器数据处理
pub struct WasmTemperatureProcessor {
    wasm_runtime: WasmRuntime,
    sensor_interface: SensorInterface,
    data_buffer: CircularBuffer<TemperatureData>,
}

impl WasmTemperatureProcessor {
    pub async fn process_temperature_data(&mut self, raw_data: &[u8]) -> Result<ProcessedData, ProcessingError> {
        // 1. 加载WASM处理模块
        let processing_module = self.load_processing_module().await?;
        
        // 2. 创建WASM实例
        let instance = self.wasm_runtime.instantiate(&processing_module).await?;
        
        // 3. 绑定传感器接口
        self.bind_sensor_interface(&instance)?;
        
        // 4. 执行数据处理
        let input_ptr = instance.allocate_memory(raw_data.len())?;
        instance.write_memory(input_ptr, raw_data)?;
        
        let output_ptr = instance.call_function("process_temperature", &[input_ptr, raw_data.len() as u32])?;
        
        // 5. 读取处理结果
        let output_len = instance.call_function("get_output_size", &[])?;
        let processed_data = instance.read_memory(output_ptr, output_len as usize)?;
        
        // 6. 清理资源
        instance.free_memory(input_ptr)?;
        instance.free_memory(output_ptr)?;
        
        Ok(ProcessedData::from_bytes(&processed_data)?)
    }
    
    fn bind_sensor_interface(&self, instance: &WasmInstance) -> Result<(), BindingError> {
        // 绑定传感器读取函数
        instance.bind_function("read_sensor", |sensor_id: u32| -> f32 {
            self.sensor_interface.read_temperature(sensor_id)
        })?;
        
        // 绑定数据存储函数
        instance.bind_function("store_data", |data_ptr: u32, data_len: u32| -> bool {
            let data = self.get_memory_slice(data_ptr, data_len);
            self.data_buffer.push(data).is_ok()
        })?;
        
        Ok(())
    }
}
```

### 6.2 边缘计算节点

**案例 6.2.1 (智能网关)**

```rust
// WASM智能网关
pub struct WasmSmartGateway {
    gateway_id: GatewayId,
    wasm_runtime: WasmRuntime,
    device_manager: DeviceManager,
    data_processor: DataProcessor,
    network_manager: NetworkManager,
}

impl WasmSmartGateway {
    pub async fn run(&mut self) -> Result<(), GatewayError> {
        loop {
            // 1. 收集设备数据
            let device_data = self.collect_device_data().await?;
            
            // 2. 加载处理模块
            let processing_module = self.load_processing_module(&device_data).await?;
            
            // 3. 执行边缘处理
            let processed_data = self.process_data_edge(&processing_module, &device_data).await?;
            
            // 4. 决策和转发
            self.make_decisions_and_forward(&processed_data).await?;
            
            // 5. 更新设备状态
            self.update_device_status().await?;
            
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }
    
    async fn process_data_edge(
        &self,
        module: &WasmModule,
        data: &[DeviceData],
    ) -> Result<Vec<ProcessedData>, ProcessingError> {
        // 创建WASM实例
        let instance = self.wasm_runtime.instantiate(module).await?;
        
        // 绑定设备接口
        self.bind_device_interface(&instance)?;
        
        // 执行边缘处理
        let mut results = Vec::new();
        
        for device_data in data {
            let input_ptr = instance.allocate_memory(device_data.size())?;
            instance.write_memory(input_ptr, &device_data.to_bytes())?;
            
            let output_ptr = instance.call_function("process_device_data", &[input_ptr, device_data.size() as u32])?;
            
            let output_len = instance.call_function("get_output_size", &[])?;
            let processed_bytes = instance.read_memory(output_ptr, output_len as usize)?;
            
            let processed_data = ProcessedData::from_bytes(&processed_bytes)?;
            results.push(processed_data);
            
            instance.free_memory(input_ptr)?;
            instance.free_memory(output_ptr)?;
        }
        
        Ok(results)
    }
    
    async fn make_decisions_and_forward(&self, data: &[ProcessedData]) -> Result<(), DecisionError> {
        for processed_data in data {
            // 基于处理结果做出决策
            let decision = self.make_decision(processed_data)?;
            
            match decision {
                Decision::ForwardToCloud => {
                    self.network_manager.send_to_cloud(processed_data).await?;
                }
                Decision::ForwardToDevice(device_id) => {
                    self.network_manager.send_to_device(device_id, processed_data).await?;
                }
                Decision::LocalAction(action) => {
                    self.execute_local_action(action).await?;
                }
                Decision::NoAction => {
                    // 不需要任何操作
                }
            }
        }
        
        Ok(())
    }
}
```

## 结论

WebAssembly在IOT领域展现出独特的优势：

1. **跨平台兼容性**：一次编译，多处运行
2. **轻量级执行**：紧凑的字节码和快速的加载
3. **安全隔离**：强大的沙箱和安全机制
4. **高性能**：接近原生代码的执行效率
5. **动态更新**：支持远程模块更新

通过形式化分析和实际案例，我们证明了WASM是IOT系统开发的理想技术选择，能够提供安全、高效、灵活的解决方案。

---

*本文档基于严格的数学分析和工程实践，为WebAssembly在IOT领域的应用提供了完整的理论指导和实践参考。*
