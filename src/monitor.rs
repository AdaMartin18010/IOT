use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorMetrics {
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub failed_tasks: usize,
    pub in_progress_tasks: usize,
    pub start_time: Instant,
    pub current_time: Instant,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub thread_count: usize,
    pub throughput: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetrics {
    pub task_id: String,
    pub status: String,
    pub start_time: Instant,
    pub duration: Option<Duration>,
    pub progress: f64,
    pub priority: u32,
}

pub struct RealTimeMonitor {
    metrics: Arc<Mutex<MonitorMetrics>>,
    task_metrics: Arc<Mutex<HashMap<String, TaskMetrics>>>,
    update_interval: Duration,
}

impl RealTimeMonitor {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(MonitorMetrics {
                total_tasks: 0,
                completed_tasks: 0,
                failed_tasks: 0,
                in_progress_tasks: 0,
                start_time: Instant::now(),
                current_time: Instant::now(),
                cpu_usage: 0.0,
                memory_usage: 0.0,
                thread_count: num_cpus::get(),
                throughput: 0.0,
            })),
            task_metrics: Arc::new(Mutex::new(HashMap::new())),
            update_interval: Duration::from_millis(100),
        }
    }

    pub async fn start_monitoring(&self) {
        println!("📊 启动实时监控仪表板...");
        
        let metrics = self.metrics.clone();
        let task_metrics = self.task_metrics.clone();
        
        // 启动监控循环
        tokio::spawn(async move {
            loop {
                Self::update_metrics(&metrics, &task_metrics).await;
                Self::display_dashboard(&metrics, &task_metrics).await;
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        });
    }

    async fn update_metrics(
        metrics: &Arc<Mutex<MonitorMetrics>>,
        task_metrics: &Arc<Mutex<HashMap<String, TaskMetrics>>>
    ) {
        let mut metrics_guard = metrics.lock().await;
        let task_metrics_guard = task_metrics.lock().await;
        
        metrics_guard.current_time = Instant::now();
        metrics_guard.total_tasks = task_metrics_guard.len();
        
        let mut completed = 0;
        let mut failed = 0;
        let mut in_progress = 0;
        
        for task in task_metrics_guard.values() {
            match task.status.as_str() {
                "Completed" => completed += 1,
                "Failed" => failed += 1,
                "InProgress" => in_progress += 1,
                _ => {}
            }
        }
        
        metrics_guard.completed_tasks = completed;
        metrics_guard.failed_tasks = failed;
        metrics_guard.in_progress_tasks = in_progress;
        
        // 计算吞吐量
        let elapsed = metrics_guard.current_time.duration_since(metrics_guard.start_time);
        if elapsed.as_secs() > 0 {
            metrics_guard.throughput = completed as f64 / elapsed.as_secs() as f64;
        }
        
        // 模拟CPU和内存使用率
        metrics_guard.cpu_usage = (completed as f64 / metrics_guard.total_tasks as f64) * 100.0;
        metrics_guard.memory_usage = (task_metrics_guard.len() as f64 / 100.0) * 100.0;
    }

    async fn display_dashboard(
        metrics: &Arc<Mutex<MonitorMetrics>>,
        task_metrics: &Arc<Mutex<HashMap<String, TaskMetrics>>>
    ) {
        let metrics_guard = metrics.lock().await;
        let task_metrics_guard = task_metrics.lock().await;
        
        // 清屏
        print!("\x1B[2J\x1B[1;1H");
        
        println!("🚀 IoT项目多线程加速推进 - 实时监控仪表板");
        println!("=".repeat(80));
        println!("📅 监控时间: {}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));
        println!("⏱️  运行时间: {:?}", metrics_guard.current_time.duration_since(metrics_guard.start_time));
        println!();
        
        // 总体指标
        println!("📊 总体指标:");
        println!("   总任务数: {}", metrics_guard.total_tasks);
        println!("   已完成: {} ({:.1}%)", 
            metrics_guard.completed_tasks, 
            (metrics_guard.completed_tasks as f64 / metrics_guard.total_tasks as f64) * 100.0
        );
        println!("   执行中: {}", metrics_guard.in_progress_tasks);
        println!("   失败: {}", metrics_guard.failed_tasks);
        println!("   吞吐量: {:.2} 任务/秒", metrics_guard.throughput);
        println!();
        
        // 系统资源
        println!("💻 系统资源:");
        println!("   CPU使用率: {:.1}%", metrics_guard.cpu_usage);
        println!("   内存使用率: {:.1}%", metrics_guard.memory_usage);
        println!("   线程数: {}", metrics_guard.thread_count);
        println!();
        
        // 进度条
        let progress = if metrics_guard.total_tasks > 0 {
            metrics_guard.completed_tasks as f64 / metrics_guard.total_tasks as f64
        } else {
            0.0
        };
        
        println!("📈 总体进度:");
        Self::display_progress_bar(progress);
        println!("   {:.1}% 完成", progress * 100.0);
        println!();
        
        // 任务详情
        println!("🔍 任务详情:");
        println!("{:<20} {:<12} {:<10} {:<10} {:<10}", "任务ID", "状态", "优先级", "耗时", "进度");
        println!("-".repeat(80));
        
        for (task_id, task) in task_metrics_guard.iter() {
            let duration_str = if let Some(duration) = task.duration {
                format!("{:?}", duration)
            } else {
                "进行中".to_string()
            };
            
            println!("{:<20} {:<12} {:<10} {:<10} {:.1}%", 
                task_id, 
                task.status, 
                task.priority,
                duration_str,
                task.progress * 100.0
            );
        }
        
        println!();
        println!("=".repeat(80));
        println!("按 Ctrl+C 停止监控");
    }

    fn display_progress_bar(progress: f64) {
        let bar_width = 50;
        let filled_width = (progress * bar_width as f64) as usize;
        let empty_width = bar_width - filled_width;
        
        print!("   [");
        for _ in 0..filled_width {
            print!("█");
        }
        for _ in 0..empty_width {
            print!("░");
        }
        print!("]");
        println!();
    }

    pub async fn update_task_status(&self, task_id: String, status: String, progress: f64) {
        let mut task_metrics = self.task_metrics.lock().await;
        
        if let Some(task) = task_metrics.get_mut(&task_id) {
            task.status = status;
            task.progress = progress;
            
            if status == "Completed" || status == "Failed" {
                task.duration = Some(Instant::now().duration_since(task.start_time));
            }
        }
    }

    pub async fn add_task(&self, task_id: String, priority: u32) {
        let mut task_metrics = self.task_metrics.lock().await;
        
        task_metrics.insert(task_id.clone(), TaskMetrics {
            task_id,
            status: "InProgress".to_string(),
            start_time: Instant::now(),
            duration: None,
            progress: 0.0,
            priority,
        });
    }
}

// 性能分析器
pub struct PerformanceAnalyzer {
    start_time: Instant,
    task_times: Vec<Duration>,
    throughput_history: Vec<f64>,
}

impl PerformanceAnalyzer {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            task_times: Vec::new(),
            throughput_history: Vec::new(),
        }
    }

    pub fn record_task_completion(&mut self, duration: Duration) {
        self.task_times.push(duration);
    }

    pub fn calculate_statistics(&self) -> PerformanceStats {
        if self.task_times.is_empty() {
            return PerformanceStats {
                total_tasks: 0,
                average_time: Duration::from_secs(0),
                min_time: Duration::from_secs(0),
                max_time: Duration::from_secs(0),
                throughput: 0.0,
                efficiency: 0.0,
            };
        }

        let total_tasks = self.task_times.len();
        let total_time: Duration = self.task_times.iter().sum();
        let average_time = total_time / total_tasks as u32;
        let min_time = self.task_times.iter().min().unwrap().clone();
        let max_time = self.task_times.iter().max().unwrap().clone();
        
        let elapsed = Instant::now().duration_since(self.start_time);
        let throughput = if elapsed.as_secs() > 0 {
            total_tasks as f64 / elapsed.as_secs() as f64
        } else {
            0.0
        };

        // 计算并行效率
        let sequential_time: Duration = self.task_times.iter().sum();
        let parallel_time = elapsed;
        let efficiency = if parallel_time.as_millis() > 0 {
            sequential_time.as_millis() as f64 / parallel_time.as_millis() as f64
        } else {
            0.0
        };

        PerformanceStats {
            total_tasks,
            average_time,
            min_time,
            max_time,
            throughput,
            efficiency,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub total_tasks: usize,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub throughput: f64,
    pub efficiency: f64,
}

impl std::fmt::Display for PerformanceStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "📊 性能统计:")?;
        writeln!(f, "   总任务数: {}", self.total_tasks)?;
        writeln!(f, "   平均耗时: {:?}", self.average_time)?;
        writeln!(f, "   最短耗时: {:?}", self.min_time)?;
        writeln!(f, "   最长耗时: {:?}", self.max_time)?;
        writeln!(f, "   吞吐量: {:.2} 任务/秒", self.throughput)?;
        writeln!(f, "   并行效率: {:.2}x", self.efficiency)?;
        Ok(())
    }
}
