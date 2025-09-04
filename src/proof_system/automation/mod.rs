pub mod engine;
pub mod scheduler;
pub mod resource_manager;
pub mod performance_monitor;

pub use engine::*;
pub use scheduler::*;
pub use resource_manager::*;
pub use performance_monitor::*;

use crate::core::{Proof, ProofStep, InferenceRule, ProofError, ProofStatus, Proposition, RuleLibrary};
use crate::strategies::{StrategyExecutionResult, StrategyPerformanceMetrics, StrategyConfig};
use std::collections::HashMap;
use std::time::{Instant, Duration};

/// 自动化任务状态
#[derive(Debug, Clone, PartialEq)]
pub enum AutomationTaskStatus {
    /// 等待中
    Pending,
    /// 运行中
    Running,
    /// 已完成
    Completed,
    /// 失败
    Failed,
    /// 已取消
    Cancelled,
    /// 暂停
    Paused,
}

/// 自动化任务优先级
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum TaskPriority {
    /// 低优先级
    Low = 1,
    /// 普通优先级
    Normal = 2,
    /// 高优先级
    High = 3,
    /// 紧急优先级
    Urgent = 4,
}

/// 自动化任务
#[derive(Debug, Clone)]
pub struct AutomationTask {
    /// 任务ID
    pub id: u64,
    /// 任务名称
    pub name: String,
    /// 任务描述
    pub description: String,
    /// 任务状态
    pub status: AutomationTaskStatus,
    /// 任务优先级
    pub priority: TaskPriority,
    /// 关联的证明
    pub proof_id: Option<u64>,
    /// 创建时间
    pub created_at: Instant,
    /// 开始时间
    pub started_at: Option<Instant>,
    /// 完成时间
    pub completed_at: Option<Instant>,
    /// 任务参数
    pub parameters: HashMap<String, String>,
    /// 任务结果
    pub result: Option<TaskResult>,
    /// 错误信息
    pub error: Option<String>,
}

/// 任务结果
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// 是否成功
    pub success: bool,
    /// 执行时间
    pub execution_time: Duration,
    /// 生成的步骤数
    pub steps_generated: usize,
    /// 应用的规则数
    pub rules_applied: usize,
    /// 性能指标
    pub performance_metrics: StrategyPerformanceMetrics,
}

/// 自动化配置
#[derive(Debug, Clone)]
pub struct AutomationConfig {
    /// 最大并发任务数
    pub max_concurrent_tasks: usize,
    /// 任务超时时间
    pub task_timeout: Duration,
    /// 是否启用自动重试
    pub enable_auto_retry: bool,
    /// 最大重试次数
    pub max_retry_count: usize,
    /// 是否启用任务优先级
    pub enable_task_priority: bool,
    /// 是否启用资源限制
    pub enable_resource_limits: bool,
    /// 是否启用性能监控
    pub enable_performance_monitoring: bool,
}

impl Default for AutomationConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: 10,
            task_timeout: Duration::from_secs(300), // 5分钟
            enable_auto_retry: true,
            max_retry_count: 3,
            enable_task_priority: true,
            enable_resource_limits: true,
            enable_performance_monitoring: true,
        }
    }
}

/// 资源使用情况
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// CPU使用率
    pub cpu_usage: f64,
    /// 内存使用量（MB）
    pub memory_usage_mb: u64,
    /// 磁盘使用量（MB）
    pub disk_usage_mb: u64,
    /// 网络使用量（MB）
    pub network_usage_mb: u64,
    /// 活跃任务数
    pub active_tasks: usize,
    /// 队列中的任务数
    pub queued_tasks: usize,
}

/// 性能指标
#[derive(Debug, Clone)]
pub struct AutomationPerformanceMetrics {
    /// 总任务数
    pub total_tasks: u64,
    /// 成功任务数
    pub successful_tasks: u64,
    /// 失败任务数
    pub failed_tasks: u64,
    /// 平均执行时间
    pub avg_execution_time: Duration,
    /// 平均步骤生成数
    pub avg_steps_generated: f64,
    /// 平均规则应用数
    pub avg_rules_applied: f64,
    /// 资源利用率
    pub resource_utilization: ResourceUsage,
}

impl Default for AutomationPerformanceMetrics {
    fn default() -> Self {
        Self {
            total_tasks: 0,
            successful_tasks: 0,
            failed_tasks: 0,
            avg_execution_time: Duration::from_secs(0),
            avg_steps_generated: 0.0,
            avg_rules_applied: 0.0,
            resource_utilization: ResourceUsage {
                cpu_usage: 0.0,
                memory_usage_mb: 0,
                disk_usage_mb: 0,
                network_usage_mb: 0,
                active_tasks: 0,
                queued_tasks: 0,
            },
        }
    }
}

/// 自动化事件
#[derive(Debug, Clone)]
pub enum AutomationEvent {
    /// 任务创建
    TaskCreated(u64),
    /// 任务开始
    TaskStarted(u64),
    /// 任务完成
    TaskCompleted(u64),
    /// 任务失败
    TaskFailed(u64, String),
    /// 任务取消
    TaskCancelled(u64),
    /// 任务暂停
    TaskPaused(u64),
    /// 资源不足
    ResourceLow(ResourceUsage),
    /// 性能警告
    PerformanceWarning(String),
    /// 系统错误
    SystemError(String),
}

/// 事件监听器特征
pub trait EventListener {
    /// 处理事件
    fn handle_event(&mut self, event: &AutomationEvent);
    
    /// 获取监听器名称
    fn name(&self) -> &str;
}

/// 自动化系统
pub struct AutomationSystem {
    config: AutomationConfig,
    engine: ProofAutomationEngine,
    scheduler: TaskScheduler,
    resource_manager: ResourceManager,
    performance_monitor: PerformanceMonitor,
    event_listeners: Vec<Box<dyn EventListener>>,
    performance_metrics: AutomationPerformanceMetrics,
}

impl AutomationSystem {
    /// 创建新的自动化系统
    pub fn new(
        rule_library: RuleLibrary,
        config: AutomationConfig,
    ) -> Self {
        let engine = ProofAutomationEngine::new(rule_library.clone(), config.clone());
        let scheduler = TaskScheduler::new(config.clone());
        let resource_manager = ResourceManager::new(config.clone());
        let performance_monitor = PerformanceMonitor::new(config.clone());
        
        Self {
            config,
            engine,
            scheduler,
            resource_manager,
            performance_monitor,
            event_listeners: Vec::new(),
            performance_metrics: AutomationPerformanceMetrics::default(),
        }
    }

    /// 添加事件监听器
    pub fn add_event_listener(&mut self, listener: Box<dyn EventListener>) {
        self.event_listeners.push(listener);
    }

    /// 提交自动化任务
    pub fn submit_task(&mut self, task: AutomationTask) -> Result<u64, ProofError> {
        // 检查资源可用性
        if !self.resource_manager.can_accept_task(&task) {
            return Err(ProofError::ResourceUnavailable("资源不足，无法接受新任务".to_string()));
        }
        
        // 提交到调度器
        let task_id = self.scheduler.submit_task(task.clone())?;
        
        // 通知事件监听器
        self.notify_event_listeners(&AutomationEvent::TaskCreated(task_id));
        
        Ok(task_id)
    }

    /// 启动自动化系统
    pub fn start(&mut self) -> Result<(), ProofError> {
        // 启动各个组件
        self.engine.start()?;
        self.scheduler.start()?;
        self.resource_manager.start()?;
        self.performance_monitor.start()?;
        
        Ok(())
    }

    /// 停止自动化系统
    pub fn stop(&mut self) -> Result<(), ProofError> {
        // 停止各个组件
        self.engine.stop()?;
        self.scheduler.stop()?;
        self.resource_manager.stop()?;
        self.performance_monitor.stop()?;
        
        Ok(())
    }

    /// 获取系统状态
    pub fn get_status(&self) -> SystemStatus {
        SystemStatus {
            engine_status: self.engine.get_status(),
            scheduler_status: self.scheduler.get_status(),
            resource_manager_status: self.resource_manager.get_status(),
            performance_monitor_status: self.performance_monitor.get_status(),
            active_tasks: self.scheduler.get_active_task_count(),
            queued_tasks: self.scheduler.get_queued_task_count(),
            resource_usage: self.resource_manager.get_current_usage(),
        }
    }

    /// 获取性能指标
    pub fn get_performance_metrics(&self) -> &AutomationPerformanceMetrics {
        &self.performance_metrics
    }

    /// 通知事件监听器
    fn notify_event_listeners(&mut self, event: &AutomationEvent) {
        for listener in &mut self.event_listeners {
            listener.handle_event(event);
        }
    }
}

/// 系统状态
#[derive(Debug, Clone)]
pub struct SystemStatus {
    /// 引擎状态
    pub engine_status: ComponentStatus,
    /// 调度器状态
    pub scheduler_status: ComponentStatus,
    /// 资源管理器状态
    pub resource_manager_status: ComponentStatus,
    /// 性能监控器状态
    pub performance_monitor_status: ComponentStatus,
    /// 活跃任务数
    pub active_tasks: usize,
    /// 队列中的任务数
    pub queued_tasks: usize,
    /// 资源使用情况
    pub resource_usage: ResourceUsage,
}

/// 组件状态
#[derive(Debug, Clone, PartialEq)]
pub enum ComponentStatus {
    /// 未启动
    NotStarted,
    /// 运行中
    Running,
    /// 已停止
    Stopped,
    /// 错误状态
    Error(String),
}

impl Default for ComponentStatus {
    fn default() -> Self {
        ComponentStatus::NotStarted
    }
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage_mb: 0,
            disk_usage_mb: 0,
            network_usage_mb: 0,
            active_tasks: 0,
            queued_tasks: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::RuleLibrary;

    #[test]
    fn test_automation_config_default() {
        let config = AutomationConfig::default();
        assert_eq!(config.max_concurrent_tasks, 10);
        assert_eq!(config.max_retry_count, 3);
        assert!(config.enable_auto_retry);
    }

    #[test]
    fn test_task_priority_ordering() {
        assert!(TaskPriority::Urgent > TaskPriority::High);
        assert!(TaskPriority::High > TaskPriority::Normal);
        assert!(TaskPriority::Normal > TaskPriority::Low);
    }

    #[test]
    fn test_automation_system_creation() {
        let rule_library = RuleLibrary::new();
        let config = AutomationConfig::default();
        let system = AutomationSystem::new(rule_library, config);
        
        assert_eq!(system.event_listeners.len(), 0);
    }

    #[test]
    fn test_resource_usage_default() {
        let usage = ResourceUsage::default();
        assert_eq!(usage.cpu_usage, 0.0);
        assert_eq!(usage.memory_usage_mb, 0);
        assert_eq!(usage.active_tasks, 0);
    }

    #[test]
    fn test_performance_metrics_default() {
        let metrics = AutomationPerformanceMetrics::default();
        assert_eq!(metrics.total_tasks, 0);
        assert_eq!(metrics.successful_tasks, 0);
        assert_eq!(metrics.failed_tasks, 0);
    }
}
