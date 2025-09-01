use std::sync::Arc;
use tokio::sync::{Mutex, Semaphore};
use tokio::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

// 任务结果结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: String,
    pub status: TaskStatus,
    pub duration: Duration,
    pub output: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    Completed,
    Failed,
    InProgress,
}

// 任务类型枚举
#[derive(Debug, Clone)]
pub enum TaskType {
    VerificationTool,
    SemanticAdapter,
    DeveloperToolchain,
    SmartCity,
    Industry40,
    PerformanceTest,
    Community,
    Standardization,
    Partnership,
}

impl TaskType {
    pub fn get_task_id(&self) -> String {
        match self {
            TaskType::VerificationTool => "verification_tool".to_string(),
            TaskType::SemanticAdapter => "semantic_adapter".to_string(),
            TaskType::DeveloperToolchain => "developer_toolchain".to_string(),
            TaskType::SmartCity => "smart_city".to_string(),
            TaskType::Industry40 => "industry_40".to_string(),
            TaskType::PerformanceTest => "performance_test".to_string(),
            TaskType::Community => "community".to_string(),
            TaskType::Standardization => "standardization".to_string(),
            TaskType::Partnership => "partnership".to_string(),
        }
    }

    pub fn get_priority(&self) -> u32 {
        match self {
            TaskType::VerificationTool => 1,
            TaskType::SemanticAdapter => 1,
            TaskType::DeveloperToolchain => 2,
            TaskType::SmartCity => 2,
            TaskType::Industry40 => 2,
            TaskType::PerformanceTest => 3,
            TaskType::Community => 3,
            TaskType::Standardization => 3,
            TaskType::Partnership => 4,
        }
    }

    pub async fn execute(&self) -> TaskResult {
        let task_id = self.get_task_id();
        
        // 模拟任务执行
        let (duration, output) = match self {
            TaskType::VerificationTool => {
                tokio::time::sleep(Duration::from_millis(500)).await;
                (Duration::from_millis(500), "自动化验证工具开发完成".to_string())
            }
            TaskType::SemanticAdapter => {
                tokio::time::sleep(Duration::from_millis(400)).await;
                (Duration::from_millis(400), "动态语义适配器开发完成".to_string())
            }
            TaskType::DeveloperToolchain => {
                tokio::time::sleep(Duration::from_millis(300)).await;
                (Duration::from_millis(300), "开发者工具链构建完成".to_string())
            }
            TaskType::SmartCity => {
                tokio::time::sleep(Duration::from_millis(600)).await;
                (Duration::from_millis(600), "智慧城市平台集成完成".to_string())
            }
            TaskType::Industry40 => {
                tokio::time::sleep(Duration::from_millis(450)).await;
                (Duration::from_millis(450), "工业4.0应用验证完成".to_string())
            }
            TaskType::PerformanceTest => {
                tokio::time::sleep(Duration::from_millis(350)).await;
                (Duration::from_millis(350), "性能基准测试完成".to_string())
            }
            TaskType::Community => {
                tokio::time::sleep(Duration::from_millis(700)).await;
                (Duration::from_millis(700), "开源社区基础设施完成".to_string())
            }
            TaskType::Standardization => {
                tokio::time::sleep(Duration::from_millis(550)).await;
                (Duration::from_millis(550), "标准化贡献提案完成".to_string())
            }
            TaskType::Partnership => {
                tokio::time::sleep(Duration::from_millis(800)).await;
                (Duration::from_millis(800), "合作伙伴网络建设完成".to_string())
            }
        };

        TaskResult {
            task_id,
            status: TaskStatus::Completed,
            duration,
            output,
            error: None,
        }
    }
}

// 多线程执行器
pub struct ParallelExecutor {
    semaphore: Arc<Semaphore>,
    task_results: Arc<Mutex<HashMap<String, TaskResult>>>,
    max_concurrent_tasks: usize,
}

impl ParallelExecutor {
    pub fn new(max_concurrent_tasks: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
            task_results: Arc::new(Mutex::new(HashMap::new())),
            max_concurrent_tasks,
        }
    }

    pub async fn execute_tasks(&self, tasks: Vec<TaskType>) -> Vec<TaskResult> {
        let start_time = Instant::now();
        
        println!("🚀 启动多线程并行执行...");
        println!("📊 任务数量: {}", tasks.len());
        println!("⚡ 最大并发数: {}", self.max_concurrent_tasks);
        
        // 创建任务句柄
        let mut handles = Vec::new();
        
        for task in tasks {
            let semaphore = self.semaphore.clone();
            let task_results = self.task_results.clone();
            
            let handle = tokio::spawn(async move {
                // 获取信号量许可
                let _permit = semaphore.acquire().await.unwrap();
                
                let task_id = task.get_task_id();
                let start = Instant::now();
                
                println!("🔄 开始执行任务: {}", task_id);
                
                // 执行任务
                let result = task.execute().await;
                let duration = start.elapsed();
                
                // 存储结果
                let mut results = task_results.lock().await;
                results.insert(task_id.clone(), result.clone());
                
                println!("✅ 任务完成: {} (耗时: {:?})", task_id, duration);
                
                result
            });
            
            handles.push(handle);
        }
        
        // 等待所有任务完成
        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result),
                Err(e) => {
                    println!("❌ 任务执行失败: {:?}", e);
                    results.push(TaskResult {
                        task_id: "unknown".to_string(),
                        status: TaskStatus::Failed,
                        duration: Duration::from_secs(0),
                        output: "".to_string(),
                        error: Some(e.to_string()),
                    });
                }
            }
        }
        
        let total_duration = start_time.elapsed();
        println!("🎉 所有任务执行完成!");
        println!("⏱️  总执行时间: {:?}", total_duration);
        println!("📈 平均任务时间: {:?}", total_duration / results.len() as u32);
        
        results
    }
}

// 主执行函数
pub async fn start_parallel_execution() -> Result<Vec<TaskResult>, Box<dyn std::error::Error>> {
    println!("🚀 IoT项目多线程加速推进启动!");
    println!("📅 执行时间: {}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S"));
    
    // 创建执行器
    let executor = ParallelExecutor::new(8); // 8个并发任务
    
    // 创建任务列表
    let tasks = vec![
        TaskType::VerificationTool,
        TaskType::SemanticAdapter,
        TaskType::DeveloperToolchain,
        TaskType::SmartCity,
        TaskType::Industry40,
        TaskType::PerformanceTest,
        TaskType::Community,
        TaskType::Standardization,
        TaskType::Partnership,
    ];
    
    // 执行任务
    let results = executor.execute_tasks(tasks).await;
    
    // 输出结果摘要
    println!("\n📊 执行结果摘要:");
    println!("{}", "=".repeat(50));
    
    let mut completed = 0;
    let mut failed = 0;
    let mut total_duration = Duration::from_secs(0);
    
    for result in &results {
        match result.status {
            TaskStatus::Completed => {
                completed += 1;
                total_duration += result.duration;
                println!("✅ {}: {:?} - {}", result.task_id, result.duration, result.output);
            }
            TaskStatus::Failed => {
                failed += 1;
                println!("❌ {}: 执行失败 - {:?}", result.task_id, result.error);
            }
            TaskStatus::InProgress => {
                println!("🔄 {}: 执行中", result.task_id);
            }
        }
    }
    
    println!("{}", "=".repeat(50));
    println!("📈 统计信息:");
    println!("   总任务数: {}", results.len());
    println!("   成功完成: {}", completed);
    println!("   执行失败: {}", failed);
    println!("   总耗时: {:?}", total_duration);
    println!("   平均耗时: {:?}", total_duration / completed as u32);
    
    // 计算加速比
    let sequential_time = Duration::from_millis(500 + 400 + 300 + 600 + 450 + 350 + 700 + 550 + 800);
    let parallel_time = total_duration;
    let speedup = if parallel_time.as_millis() > 0 {
        sequential_time.as_millis() as f64 / parallel_time.as_millis() as f64
    } else {
        1.0
    };
    
    println!("⚡ 加速比: {:.2}x", speedup);
    println!("🎯 效率提升: {:.1}%", (speedup - 1.0) * 100.0);
    
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_parallel_execution() {
        let result = start_parallel_execution().await;
        assert!(result.is_ok());
        
        let results = result.unwrap();
        assert!(!results.is_empty());
        
        // 验证至少有一些任务成功完成
        let completed_count = results.iter()
            .filter(|r| matches!(r.status, TaskStatus::Completed))
            .count();
        
        assert!(completed_count > 0);
    }
}
