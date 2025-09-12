use crate::automation::{
    AutomationTask, ResourceUsage, AutomationConfig, AutomationEvent, EventListener
};
use crate::core::ProofError;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Instant, Duration};
use tokio::sync::mpsc;
use serde::{Serialize, Deserialize};

/// 资源类型
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    /// CPU资源
    CPU,
    /// 内存资源
    Memory,
    /// 存储资源
    Storage,
    /// 网络资源
    Network,
    /// GPU资源
    GPU,
    /// 自定义资源
    Custom(String),
}

/// 资源状态
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceStatus {
    /// 可用
    Available,
    /// 使用中
    InUse,
    /// 维护中
    Maintenance,
    /// 故障
    Faulty,
    /// 已满
    Full,
}

/// 资源单元
#[derive(Debug, Clone)]
pub struct ResourceUnit {
    pub id: String,
    pub resource_type: ResourceType,
    pub capacity: f64,
    pub used: f64,
    pub status: ResourceStatus,
    pub allocated_tasks: Vec<String>,
    pub performance_metrics: ResourcePerformanceMetrics,
    pub last_updated: Instant,
}

/// 资源性能指标
#[derive(Debug, Clone, Default)]
pub struct ResourcePerformanceMetrics {
    pub utilization_rate: f64,
    pub response_time: Duration,
    pub throughput: f64,
    pub error_rate: f64,
    pub availability: f64,
}

/// 资源分配策略
#[derive(Debug, Clone, PartialEq)]
pub enum AllocationStrategy {
    /// 首次适应
    FirstFit,
    /// 最佳适应
    BestFit,
    /// 最差适应
    WorstFit,
    /// 轮转分配
    RoundRobin,
    /// 负载均衡
    LoadBalanced,
    /// 性能优先
    PerformanceFirst,
}

/// 资源管理器配置
#[derive(Debug, Clone)]
pub struct ResourceManagerConfig {
    pub max_cpu_usage: f64,
    pub max_memory_usage: f64,
    pub max_storage_usage: f64,
    pub allocation_strategy: AllocationStrategy,
    pub enable_auto_scaling: bool,
    pub enable_load_balancing: bool,
    pub resource_check_interval: Duration,
    pub performance_threshold: f64,
}

impl Default for ResourceManagerConfig {
    fn default() -> Self {
        Self {
            max_cpu_usage: 0.8,
            max_memory_usage: 0.8,
            max_storage_usage: 0.9,
            allocation_strategy: AllocationStrategy::LoadBalanced,
            enable_auto_scaling: true,
            enable_load_balancing: true,
            resource_check_interval: Duration::from_secs(30),
            performance_threshold: 0.7,
        }
    }
}

/// 资源管理器
pub struct ResourceManager {
    config: ResourceManagerConfig,
    resources: Arc<RwLock<HashMap<String, ResourceUnit>>>,
    resource_pools: Arc<RwLock<HashMap<ResourceType, Vec<String>>>>,
    allocation_history: Arc<Mutex<VecDeque<ResourceAllocation>>>,
    performance_monitor: Arc<Mutex<PerformanceMonitor>>,
    event_sender: mpsc::Sender<AutomationEvent>,
    event_listeners: Vec<Box<dyn EventListener + Send + Sync>>,
    stats: ResourceManagerStats,
}

/// 资源分配记录
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub task_id: String,
    pub resource_id: String,
    pub resource_type: ResourceType,
    pub allocated_amount: f64,
    pub allocation_time: Instant,
    pub deallocation_time: Option<Instant>,
}

/// 性能监控器
#[derive(Debug)]
pub struct PerformanceMonitor {
    pub resource_metrics: HashMap<String, Vec<ResourcePerformanceMetrics>>,
    pub performance_history: VecDeque<PerformanceSnapshot>,
    pub alert_thresholds: HashMap<ResourceType, f64>,
}

/// 性能快照
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    pub timestamp: Instant,
    pub overall_utilization: f64,
    pub resource_utilizations: HashMap<ResourceType, f64>,
    pub performance_score: f64,
}

/// 资源管理器统计
#[derive(Debug, Clone, Default)]
pub struct ResourceManagerStats {
    pub total_allocations: usize,
    pub successful_allocations: usize,
    pub failed_allocations: usize,
    pub total_deallocations: usize,
    pub average_allocation_time: Duration,
    pub resource_efficiency: f64,
}

impl ResourceManager {
    pub fn new(config: ResourceManagerConfig) -> (Self, mpsc::Receiver<AutomationEvent>) {
        let (event_sender, event_receiver) = mpsc::channel(100);
        
        let manager = Self {
            config,
            resources: Arc::new(RwLock::new(HashMap::new())),
            resource_pools: Arc::new(RwLock::new(HashMap::new())),
            allocation_history: Arc::new(Mutex::new(VecDeque::new())),
            performance_monitor: Arc::new(Mutex::new(PerformanceMonitor::new())),
            event_sender,
            event_listeners: Vec::new(),
            stats: ResourceManagerStats::default(),
        };

        (manager, event_receiver)
    }

    /// 注册资源
    pub async fn register_resource(&self, resource: ResourceUnit) -> Result<(), ProofError> {
        let mut resources = self.resources.write().await;
        let mut pools = self.resource_pools.write().await;
        
        // 添加到资源映射
        resources.insert(resource.id.clone(), resource.clone());
        
        // 添加到资源池
        pools.entry(resource.resource_type.clone())
            .or_insert_with(Vec::new)
            .push(resource.id.clone());
        
        Ok(())
    }

    /// 分配资源
    pub async fn allocate_resource(
        &self,
        task_id: &str,
        resource_type: &ResourceType,
        amount: f64,
    ) -> Result<String, ProofError> {
        let resources = self.resources.read().await;
        let pools = self.resource_pools.read().await;
        
        // 获取可用资源列表
        let available_resources = self.get_available_resources(&resources, resource_type, amount).await?;
        
        if available_resources.is_empty() {
            return Err(ProofError::InsufficientResources {
                task_id: task_id.to_string(),
                required: ResourceUsage::default(),
                available: ResourceUsage::default(),
            });
        }

        // 根据策略选择资源
        let selected_resource = self.select_resource_by_strategy(
            &available_resources,
            &self.config.allocation_strategy,
        )?;

        // 执行资源分配
        self.execute_allocation(task_id, &selected_resource, amount).await?;
        
        Ok(selected_resource.id.clone())
    }

    /// 获取可用资源
    async fn get_available_resources(
        &self,
        resources: &HashMap<String, ResourceUnit>,
        resource_type: &ResourceType,
        required_amount: f64,
    ) -> Result<Vec<&ResourceUnit>, ProofError> {
        let pools = self.resource_pools.read().await;
        
        let resource_ids = pools.get(resource_type)
            .ok_or_else(|| ProofError::ResourceNotFound {
                resource_type: format!("{:?}", resource_type),
            })?;

        let mut available = Vec::new();
        
        for resource_id in resource_ids {
            if let Some(resource) = resources.get(resource_id) {
                if resource.status == ResourceStatus::Available &&
                   (resource.capacity - resource.used) >= required_amount {
                    available.push(resource);
                }
            }
        }

        Ok(available)
    }

    /// 根据策略选择资源
    fn select_resource_by_strategy(
        &self,
        available_resources: &[&ResourceUnit],
        strategy: &AllocationStrategy,
    ) -> Result<&ResourceUnit, ProofError> {
        if available_resources.is_empty() {
            return Err(ProofError::NoAvailableResources);
        }

        match strategy {
            AllocationStrategy::FirstFit => Ok(available_resources[0]),
            AllocationStrategy::BestFit => {
                available_resources.iter()
                    .min_by(|a, b| {
                        (a.capacity - a.used).partial_cmp(&(b.capacity - b.used)).unwrap()
                    })
                    .ok_or(ProofError::NoAvailableResources)
            }
            AllocationStrategy::WorstFit => {
                available_resources.iter()
                    .max_by(|a, b| {
                        (a.capacity - a.used).partial_cmp(&(b.capacity - b.used)).unwrap()
                    })
                    .ok_or(ProofError::NoAvailableResources)
            }
            AllocationStrategy::RoundRobin => {
                // 简化的轮转实现
                Ok(available_resources[0])
            }
            AllocationStrategy::LoadBalanced => {
                available_resources.iter()
                    .min_by(|a, b| {
                        a.performance_metrics.utilization_rate
                            .partial_cmp(&b.performance_metrics.utilization_rate)
                            .unwrap()
                    })
                    .ok_or(ProofError::NoAvailableResources)
            }
            AllocationStrategy::PerformanceFirst => {
                available_resources.iter()
                    .max_by(|a, b| {
                        a.performance_metrics.availability
                            .partial_cmp(&b.performance_metrics.availability)
                            .unwrap()
                    })
                    .ok_or(ProofError::NoAvailableResources)
            }
        }
    }

    /// 执行资源分配
    async fn execute_allocation(
        &self,
        task_id: &str,
        resource: &ResourceUnit,
        amount: f64,
    ) -> Result<(), ProofError> {
        let mut resources = self.resources.write().await;
        
        if let Some(resource_unit) = resources.get_mut(&resource.id) {
            resource_unit.used += amount;
            resource_unit.allocated_tasks.push(task_id.to_string());
            
            if resource_unit.used >= resource_unit.capacity {
                resource_unit.status = ResourceStatus::Full;
            }
        }

        // 记录分配历史
        let allocation = ResourceAllocation {
            task_id: task_id.to_string(),
            resource_id: resource.id.clone(),
            resource_type: resource.resource_type.clone(),
            allocated_amount: amount,
            allocation_time: Instant::now(),
            deallocation_time: None,
        };

        let mut history = self.allocation_history.lock().unwrap();
        history.push_back(allocation);
        
        // 更新统计
        self.stats.total_allocations += 1;
        self.stats.successful_allocations += 1;

        Ok(())
    }

    /// 释放资源
    pub async fn deallocate_resource(
        &self,
        task_id: &str,
        resource_id: &str,
    ) -> Result<(), ProofError> {
        let mut resources = self.resources.write().await;
        
        if let Some(resource_unit) = resources.get_mut(resource_id) {
            // 查找任务分配的资源量
            if let Some(index) = resource_unit.allocated_tasks.iter().position(|id| id == task_id) {
                resource_unit.allocated_tasks.remove(index);
                
                // 更新资源使用量（简化处理）
                if resource_unit.used > 0.0 {
                    resource_unit.used = (resource_unit.used - 0.1).max(0.0);
                }
                
                if resource_unit.status == ResourceStatus::Full && resource_unit.used < resource_unit.capacity {
                    resource_unit.status = ResourceStatus::Available;
                }
            }
        }

        // 更新分配历史
        let mut history = self.allocation_history.lock().unwrap();
        if let Some(allocation) = history.iter_mut().find(|a| 
            a.task_id == task_id && a.resource_id == resource_id && a.deallocation_time.is_none()
        ) {
            allocation.deallocation_time = Some(Instant::now());
        }

        self.stats.total_deallocations += 1;

        Ok(())
    }

    /// 获取资源状态
    pub async fn get_resource_status(&self) -> HashMap<String, ResourceStatus> {
        let resources = self.resources.read().await;
        resources.iter()
            .map(|(id, resource)| (id.clone(), resource.status.clone()))
            .collect()
    }

    /// 获取资源使用情况
    pub async fn get_resource_usage(&self) -> HashMap<String, f64> {
        let resources = self.resources.read().await;
        resources.iter()
            .map(|(id, resource)| (id.clone(), resource.used / resource.capacity))
            .collect()
    }

    /// 性能监控
    pub async fn monitor_performance(&self) -> Result<(), ProofError> {
        let mut monitor = self.performance_monitor.lock().unwrap();
        let resources = self.resources.read().await;
        
        let mut snapshot = PerformanceSnapshot {
            timestamp: Instant::now(),
            overall_utilization: 0.0,
            resource_utilizations: HashMap::new(),
            performance_score: 0.0,
        };

        let mut total_utilization = 0.0;
        let mut resource_count = 0;

        for resource in resources.values() {
            let utilization = resource.used / resource.capacity;
            total_utilization += utilization;
            resource_count += 1;

            snapshot.resource_utilizations.insert(
                resource.resource_type.clone(),
                utilization,
            );

            // 检查性能阈值
            if utilization > self.config.performance_threshold {
                let _ = self.event_sender.try_send(AutomationEvent::ResourceHighUsage {
                    resource_id: resource.id.clone(),
                    utilization,
                    threshold: self.config.performance_threshold,
                    timestamp: Instant::now(),
                });
            }
        }

        if resource_count > 0 {
            snapshot.overall_utilization = total_utilization / resource_count as f64;
        }

        // 计算性能分数
        snapshot.performance_score = self.calculate_performance_score(&snapshot);
        
        monitor.performance_history.push_back(snapshot);
        
        // 保持历史记录数量
        if monitor.performance_history.len() > 100 {
            monitor.performance_history.pop_front();
        }

        Ok(())
    }

    /// 计算性能分数
    fn calculate_performance_score(&self, snapshot: &PerformanceSnapshot) -> f64 {
        let mut score = 0.0;
        let mut weight_sum = 0.0;

        for (resource_type, utilization) in &snapshot.resource_utilizations {
            let weight = match resource_type {
                ResourceType::CPU => 0.4,
                ResourceType::Memory => 0.3,
                ResourceType::Storage => 0.2,
                ResourceType::Network => 0.1,
                _ => 0.05,
            };

            score += (1.0 - utilization) * weight;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            score / weight_sum
        } else {
            0.0
        }
    }

    /// 自动扩缩容
    pub async fn auto_scale(&self) -> Result<(), ProofError> {
        if !self.config.enable_auto_scaling {
            return Ok(());
        }

        let monitor = self.performance_monitor.lock().unwrap();
        
        if let Some(latest_snapshot) = monitor.performance_history.back() {
            if latest_snapshot.overall_utilization > 0.8 {
                // 触发扩容
                self.scale_up().await?;
            } else if latest_snapshot.overall_utilization < 0.3 {
                // 触发缩容
                self.scale_down().await?;
            }
        }

        Ok(())
    }

    /// 扩容
    async fn scale_up(&self) -> Result<(), ProofError> {
        // 实现扩容逻辑
        let _ = self.event_sender.try_send(AutomationEvent::ResourceScaling {
            scaling_type: "scale_up".to_string(),
            timestamp: Instant::now(),
        });
        Ok(())
    }

    /// 缩容
    async fn scale_down(&self) -> Result<(), ProofError> {
        // 实现缩容逻辑
        let _ = self.event_sender.try_send(AutomationEvent::ResourceScaling {
            scaling_type: "scale_down".to_string(),
            timestamp: Instant::now(),
        });
        Ok(())
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> ResourceManagerStats {
        self.stats.clone()
    }

    /// 检查是否能接受任务
    pub fn can_accept_task(&self, task: &AutomationTask) -> bool {
        // 简化实现：检查活跃任务数
        true
    }

    /// 启动资源管理器
    pub fn start(&mut self) -> Result<(), ProofError> {
        Ok(())
    }

    /// 停止资源管理器
    pub fn stop(&mut self) -> Result<(), ProofError> {
        Ok(())
    }

    /// 添加事件监听器
    pub fn add_event_listener(&mut self, listener: Box<dyn EventListener + Send + Sync>) {
        self.event_listeners.push(listener);
    }
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            resource_metrics: HashMap::new(),
            performance_history: VecDeque::new(),
            alert_thresholds: HashMap::new(),
        }
    }

    /// 添加性能指标
    pub fn add_metrics(&mut self, resource_id: String, metrics: ResourcePerformanceMetrics) {
        self.resource_metrics.entry(resource_id)
            .or_insert_with(Vec::new)
            .push(metrics);
    }

    /// 获取性能趋势
    pub fn get_performance_trend(&self) -> Option<f64> {
        if self.performance_history.len() < 2 {
            return None;
        }

        let recent: Vec<_> = self.performance_history.iter().rev().take(5).collect();
        if recent.len() < 2 {
            return None;
        }

        let first_score = recent.last().unwrap().performance_score;
        let last_score = recent.first().unwrap().performance_score;
        
        Some(last_score - first_score)
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automation::ResourceUsage;

    fn create_test_resource(id: &str, resource_type: ResourceType, capacity: f64) -> ResourceUnit {
        ResourceUnit {
            id: id.to_string(),
            resource_type,
            capacity,
            used: 0.0,
            status: ResourceStatus::Available,
            allocated_tasks: Vec::new(),
            performance_metrics: ResourcePerformanceMetrics::default(),
            last_updated: Instant::now(),
        }
    }

    #[tokio::test]
    async fn test_resource_registration() {
        let config = ResourceManagerConfig::default();
        let (manager, _) = ResourceManager::new(config);
        
        let resource = create_test_resource("test_cpu", ResourceType::CPU, 100.0);
        let result = manager.register_resource(resource).await;
        
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_resource_allocation() {
        let config = ResourceManagerConfig::default();
        let (manager, _) = ResourceManager::new(config);
        
        let resource = create_test_resource("test_cpu", ResourceType::CPU, 100.0);
        manager.register_resource(resource).await.unwrap();
        
        let result = manager.allocate_resource("task1", &ResourceType::CPU, 50.0).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_resource_deallocation() {
        let config = ResourceManagerConfig::default();
        let (manager, _) = ResourceManager::new(config);
        
        let resource = create_test_resource("test_cpu", ResourceType::CPU, 100.0);
        manager.register_resource(resource).await.unwrap();
        
        manager.allocate_resource("task1", &ResourceType::CPU, 50.0).await.unwrap();
        let result = manager.deallocate_resource("task1", "test_cpu").await;
        
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_performance_monitoring() {
        let config = ResourceManagerConfig::default();
        let (manager, _) = ResourceManager::new(config);
        
        let result = manager.monitor_performance().await;
        assert!(result.is_ok());
    }
}
