use crate::automation::{
    AutomationTask, AutomationTaskStatus, TaskPriority, TaskResult, AutomationConfig,
    ResourceUsage, AutomationEvent, EventListener
};
use crate::core::{Proof, ProofStep, InferenceRule, ProofError, ProofStatus};
use crate::strategies::{StrategyExecutionResult, StrategyPerformanceMetrics, StrategyConfig};
use std::collections::{HashMap, VecDeque, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Instant, Duration};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

/// 调度策略
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulingStrategy {
    /// 先进先出
    FIFO,
    /// 优先级优先
    Priority,
    /// 最短作业优先
    ShortestJobFirst,
    /// 轮转调度
    RoundRobin,
    /// 自适应调度
    Adaptive,
}

/// 任务依赖关系
#[derive(Debug, Clone)]
pub struct TaskDependency {
    pub task_id: String,
    pub depends_on: Vec<String>,
    pub required_resources: Vec<String>,
}

/// 调度队列
#[derive(Debug)]
pub struct TaskQueue {
    pub high_priority: VecDeque<AutomationTask>,
    pub normal_priority: VecDeque<AutomationTask>,
    pub low_priority: VecDeque<AutomationTask>,
}

impl TaskQueue {
    pub fn new() -> Self {
        Self {
            high_priority: VecDeque::new(),
            normal_priority: VecDeque::new(),
            low_priority: VecDeque::new(),
        }
    }

    pub fn push(&mut self, task: AutomationTask) {
        match task.priority {
            TaskPriority::High => self.high_priority.push_back(task),
            TaskPriority::Normal => self.normal_priority.push_back(task),
            TaskPriority::Low => self.low_priority.push_back(task),
        }
    }

    pub fn pop(&mut self) -> Option<AutomationTask> {
        // 优先处理高优先级任务
        if let Some(task) = self.high_priority.pop_front() {
            return Some(task);
        }
        if let Some(task) = self.normal_priority.pop_front() {
            return Some(task);
        }
        self.low_priority.pop_front()
    }

    pub fn is_empty(&self) -> bool {
        self.high_priority.is_empty() && 
        self.normal_priority.is_empty() && 
        self.low_priority.is_empty()
    }

    pub fn len(&self) -> usize {
        self.high_priority.len() + 
        self.normal_priority.len() + 
        self.low_priority.len()
    }
}

/// 任务调度器
pub struct TaskScheduler {
    config: AutomationConfig,
    queue: TaskQueue,
    running_tasks: HashMap<String, JoinHandle<TaskResult>>,
    completed_tasks: HashMap<String, TaskResult>,
    failed_tasks: HashMap<String, TaskResult>,
    dependencies: HashMap<String, TaskDependency>,
    resource_usage: ResourceUsage,
    event_sender: mpsc::Sender<AutomationEvent>,
    event_listeners: Vec<Box<dyn EventListener + Send + Sync>>,
    stats: SchedulerStats,
}

/// 调度器统计信息
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    pub total_tasks_processed: usize,
    pub successful_tasks: usize,
    pub failed_tasks: usize,
    pub average_task_duration: Duration,
    pub total_execution_time: Duration,
    pub queue_wait_time: Duration,
}

impl TaskScheduler {
    pub fn new(config: AutomationConfig) -> (Self, mpsc::Receiver<AutomationEvent>) {
        let (event_sender, event_receiver) = mpsc::channel(100);
        
        let scheduler = Self {
            config,
            queue: TaskQueue::new(),
            running_tasks: HashMap::new(),
            completed_tasks: HashMap::new(),
            failed_tasks: HashMap::new(),
            dependencies: HashMap::new(),
            resource_usage: ResourceUsage::default(),
            event_sender,
            event_listeners: Vec::new(),
            stats: SchedulerStats::default(),
        };

        (scheduler, event_receiver)
    }

    /// 添加任务到调度队列
    pub fn submit_task(&mut self, task: AutomationTask) -> Result<(), ProofError> {
        // 检查依赖关系
        if let Some(dependency) = &self.dependencies.get(&task.id) {
            for dep_id in &dependency.depends_on {
                if !self.completed_tasks.contains_key(dep_id) {
                    return Err(ProofError::DependencyNotMet {
                        task_id: task.id.clone(),
                        missing_dependency: dep_id.clone(),
                    });
                }
            }
        }

        // 检查资源可用性
        if !self.check_resource_availability(&task) {
            return Err(ProofError::InsufficientResources {
                task_id: task.id.clone(),
                required: task.estimated_resources.clone(),
                available: self.resource_usage.clone(),
            });
        }

        self.queue.push(task);
        self.stats.total_tasks_processed += 1;
        
        let _ = self.event_sender.try_send(AutomationEvent::TaskSubmitted {
            task_id: task.id.clone(),
            timestamp: Instant::now(),
        });

        Ok(())
    }

    /// 启动任务执行
    pub async fn start_execution(&mut self) -> Result<(), ProofError> {
        while !self.queue.is_empty() && self.can_start_new_task() {
            if let Some(task) = self.queue.pop() {
                self.start_task(task).await?;
            }
        }
        Ok(())
    }

    /// 启动单个任务
    async fn start_task(&mut self, task: AutomationTask) -> Result<(), ProofError> {
        let task_id = task.id.clone();
        let task_clone = task.clone();
        
        // 创建任务执行句柄
        let handle = tokio::spawn(async move {
            Self::execute_task(task_clone).await
        });

        self.running_tasks.insert(task_id.clone(), handle);
        
        // 更新资源使用
        self.allocate_resources(&task);
        
        let _ = self.event_sender.try_send(AutomationEvent::TaskStarted {
            task_id,
            timestamp: Instant::now(),
        });

        Ok(())
    }

    /// 执行任务
    async fn execute_task(task: AutomationTask) -> TaskResult {
        let start_time = Instant::now();
        
        // 这里应该调用实际的证明策略执行逻辑
        // 目前使用模拟实现
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        let duration = start_time.elapsed();
        
        TaskResult {
            task_id: task.id,
            status: AutomationTaskStatus::Completed,
            result: Some("任务执行成功".to_string()),
            error: None,
            execution_time: duration,
            resource_usage: ResourceUsage::default(),
            metadata: HashMap::new(),
        }
    }

    /// 检查是否可以启动新任务
    fn can_start_new_task(&self) -> bool {
        self.running_tasks.len() < self.config.max_concurrent_tasks
    }

    /// 检查资源可用性
    fn check_resource_availability(&self, task: &AutomationTask) -> bool {
        // 简化的资源检查逻辑
        true
    }

    /// 分配资源
    fn allocate_resources(&mut self, task: &AutomationTask) {
        // 简化的资源分配逻辑
    }

    /// 释放资源
    fn release_resources(&mut self, task: &AutomationTask) {
        // 简化的资源释放逻辑
    }

    /// 检查任务完成状态
    pub async fn check_completed_tasks(&mut self) -> Result<(), ProofError> {
        let mut completed_ids = Vec::new();
        
        for (task_id, handle) in &mut self.running_tasks {
            if handle.is_finished() {
                completed_ids.push(task_id.clone());
            }
        }

        for task_id in completed_ids {
            if let Some(handle) = self.running_tasks.remove(&task_id) {
                match handle.await {
                    Ok(result) => {
                        if result.status == AutomationTaskStatus::Completed {
                            self.completed_tasks.insert(task_id.clone(), result.clone());
                            self.stats.successful_tasks += 1;
                        } else {
                            self.failed_tasks.insert(task_id.clone(), result.clone());
                            self.stats.failed_tasks += 1;
                        }
                        
                        let _ = self.event_sender.try_send(AutomationEvent::TaskCompleted {
                            task_id: task_id.clone(),
                            result: result.clone(),
                            timestamp: Instant::now(),
                        });
                    }
                    Err(e) => {
                        let failed_result = TaskResult {
                            task_id: task_id.clone(),
                            status: AutomationTaskStatus::Failed,
                            result: None,
                            error: Some(format!("任务执行失败: {}", e)),
                            execution_time: Duration::ZERO,
                            resource_usage: ResourceUsage::default(),
                            metadata: HashMap::new(),
                        };
                        
                        self.failed_tasks.insert(task_id.clone(), failed_result.clone());
                        self.stats.failed_tasks += 1;
                        
                        let _ = self.event_sender.try_send(AutomationEvent::TaskFailed {
                            task_id,
                            error: failed_result.error.clone().unwrap_or_default(),
                            timestamp: Instant::now(),
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// 获取调度器状态
    pub fn get_status(&self) -> SchedulerStatus {
        SchedulerStatus {
            queue_length: self.queue.len(),
            running_tasks: self.running_tasks.len(),
            completed_tasks: self.completed_tasks.len(),
            failed_tasks: self.failed_tasks.len(),
            stats: self.stats.clone(),
        }
    }

    /// 添加事件监听器
    pub fn add_event_listener(&mut self, listener: Box<dyn EventListener + Send + Sync>) {
        self.event_listeners.push(listener);
    }

    /// 设置调度策略
    pub fn set_scheduling_strategy(&mut self, strategy: SchedulingStrategy) {
        // 实现策略切换逻辑
    }

    /// 获取队列统计
    pub fn get_queue_stats(&self) -> QueueStats {
        QueueStats {
            high_priority_count: self.queue.high_priority.len(),
            normal_priority_count: self.queue.normal_priority.len(),
            low_priority_count: self.queue.low_priority.len(),
            total_count: self.queue.len(),
        }
    }

    /// 清理已完成的任务
    pub fn cleanup_completed_tasks(&mut self) {
        // 保留最近的任务，清理旧任务
        if self.completed_tasks.len() > self.config.max_completed_tasks {
            let mut sorted_tasks: Vec<_> = self.completed_tasks.iter().collect();
            sorted_tasks.sort_by_key(|(_, result)| result.execution_time);
            
            let to_remove = sorted_tasks.len() - self.config.max_completed_tasks;
            for (task_id, _) in sorted_tasks.iter().take(to_remove) {
                self.completed_tasks.remove(*task_id);
            }
        }
    }
}

/// 调度器状态
#[derive(Debug, Clone)]
pub struct SchedulerStatus {
    pub queue_length: usize,
    pub running_tasks: usize,
    pub completed_tasks: usize,
    pub failed_tasks: usize,
    pub stats: SchedulerStats,
}

/// 队列统计
#[derive(Debug, Clone)]
pub struct QueueStats {
    pub high_priority_count: usize,
    pub normal_priority_count: usize,
    pub low_priority_count: usize,
    pub total_count: usize,
}

impl Default for TaskQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automation::{AutomationTask, TaskPriority, AutomationTaskStatus};

    fn create_test_task(id: &str, priority: TaskPriority) -> AutomationTask {
        AutomationTask {
            id: id.to_string(),
            name: format!("测试任务{}", id),
            description: "测试任务描述".to_string(),
            priority,
            status: AutomationTaskStatus::Pending,
            estimated_duration: Duration::from_secs(1),
            estimated_resources: ResourceUsage::default(),
            dependencies: Vec::new(),
            metadata: HashMap::new(),
            created_at: Instant::now(),
        }
    }

    #[tokio::test]
    async fn test_task_scheduler_creation() {
        let config = AutomationConfig::default();
        let (scheduler, _) = TaskScheduler::new(config);
        
        assert_eq!(scheduler.queue.len(), 0);
        assert_eq!(scheduler.running_tasks.len(), 0);
    }

    #[tokio::test]
    async fn test_task_submission() {
        let config = AutomationConfig::default();
        let (mut scheduler, _) = TaskScheduler::new(config);
        
        let task = create_test_task("1", TaskPriority::High);
        let result = scheduler.submit_task(task);
        
        assert!(result.is_ok());
        assert_eq!(scheduler.queue.len(), 1);
    }

    #[tokio::test]
    async fn test_priority_queue_ordering() {
        let config = AutomationConfig::default();
        let (mut scheduler, _) = TaskScheduler::new(config);
        
        let low_task = create_test_task("1", TaskPriority::Low);
        let high_task = create_test_task("2", TaskPriority::High);
        let normal_task = create_test_task("3", TaskPriority::Normal);
        
        scheduler.submit_task(low_task).unwrap();
        scheduler.submit_task(high_task).unwrap();
        scheduler.submit_task(normal_task).unwrap();
        
        // 高优先级任务应该首先被处理
        let first_task = scheduler.queue.pop();
        assert_eq!(first_task.unwrap().priority, TaskPriority::High);
    }

    #[tokio::test]
    async fn test_task_execution() {
        let config = AutomationConfig::default();
        let (mut scheduler, _) = TaskScheduler::new(config);
        
        let task = create_test_task("1", TaskPriority::Normal);
        scheduler.submit_task(task).unwrap();
        
        scheduler.start_execution().await.unwrap();
        
        // 等待任务完成
        tokio::time::sleep(Duration::from_millis(200)).await;
        
        scheduler.check_completed_tasks().await.unwrap();
        
        assert_eq!(scheduler.completed_tasks.len(), 1);
    }
}
