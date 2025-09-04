use crate::automation::{
    AutomationTask, TaskResult, AutomationConfig, AutomationEvent, EventListener,
    AutomationPerformanceMetrics, ResourceUsage
};
use crate::core::ProofError;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Instant, Duration};
use tokio::sync::mpsc;
use serde::{Serialize, Deserialize};

/// 性能指标类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetricType {
    /// 执行时间
    ExecutionTime,
    /// 吞吐量
    Throughput,
    /// 成功率
    SuccessRate,
    /// 资源使用率
    ResourceUtilization,
    /// 响应时间
    ResponseTime,
    /// 错误率
    ErrorRate,
    /// 队列长度
    QueueLength,
    /// 自定义指标
    Custom(String),
}

/// 性能指标
#[derive(Debug, Clone)]
pub struct PerformanceMetric {
    pub metric_type: MetricType,
    pub value: f64,
    pub unit: String,
    pub timestamp: Instant,
    pub metadata: HashMap<String, String>,
}

/// 性能阈值
#[derive(Debug, Clone)]
pub struct PerformanceThreshold {
    pub metric_type: MetricType,
    pub warning_threshold: f64,
    pub critical_threshold: f64,
    pub enabled: bool,
}

/// 性能告警级别
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum AlertLevel {
    /// 信息
    Info,
    /// 警告
    Warning,
    /// 严重
    Critical,
    /// 紧急
    Emergency,
}

/// 性能告警
#[derive(Debug, Clone)]
pub struct PerformanceAlert {
    pub id: String,
    pub level: AlertLevel,
    pub message: String,
    pub metric_type: MetricType,
    pub current_value: f64,
    pub threshold_value: f64,
    pub timestamp: Instant,
    pub acknowledged: bool,
    pub resolved: bool,
}

/// 性能分析结果
#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    pub overall_score: f64,
    pub bottleneck_metrics: Vec<PerformanceMetric>,
    pub improvement_suggestions: Vec<String>,
    pub trend_analysis: TrendAnalysis,
    pub resource_efficiency: f64,
}

/// 趋势分析
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    pub trend_direction: TrendDirection,
    pub trend_strength: f64,
    pub prediction: Option<f64>,
    pub confidence: f64,
}

/// 趋势方向
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    /// 上升
    Increasing,
    /// 下降
    Decreasing,
    /// 稳定
    Stable,
    /// 波动
    Fluctuating,
}

/// 性能监控器配置
#[derive(Debug, Clone)]
pub struct PerformanceMonitorConfig {
    pub collection_interval: Duration,
    pub retention_period: Duration,
    pub enable_real_time_monitoring: bool,
    pub enable_alerting: bool,
    pub enable_trend_analysis: bool,
    pub max_metrics_history: usize,
    pub alert_cooldown: Duration,
}

impl Default for PerformanceMonitorConfig {
    fn default() -> Self {
        Self {
            collection_interval: Duration::from_secs(10),
            retention_period: Duration::from_secs(3600), // 1小时
            enable_real_time_monitoring: true,
            enable_alerting: true,
            enable_trend_analysis: true,
            max_metrics_history: 1000,
            alert_cooldown: Duration::from_secs(300), // 5分钟
        }
    }
}

/// 性能监控器
pub struct PerformanceMonitor {
    config: PerformanceMonitorConfig,
    metrics_history: Arc<RwLock<HashMap<MetricType, VecDeque<PerformanceMetric>>>>,
    thresholds: Arc<RwLock<HashMap<MetricType, PerformanceThreshold>>>,
    alerts: Arc<Mutex<VecDeque<PerformanceAlert>>>,
    event_sender: mpsc::Sender<AutomationEvent>,
    event_listeners: Vec<Box<dyn EventListener + Send + Sync>>,
    stats: PerformanceMonitorStats,
    last_collection: Arc<Mutex<Instant>>,
}

/// 性能监控器统计
#[derive(Debug, Clone, Default)]
pub struct PerformanceMonitorStats {
    pub total_metrics_collected: usize,
    pub total_alerts_generated: usize,
    pub total_analyses_performed: usize,
    pub average_collection_time: Duration,
    pub last_collection_duration: Duration,
}

impl PerformanceMonitor {
    pub fn new(config: PerformanceMonitorConfig) -> (Self, mpsc::Receiver<AutomationEvent>) {
        let (event_sender, event_receiver) = mpsc::channel(100);
        
        let monitor = Self {
            config,
            metrics_history: Arc::new(RwLock::new(HashMap::new())),
            thresholds: Arc::new(RwLock::new(HashMap::new())),
            alerts: Arc::new(Mutex::new(VecDeque::new())),
            event_sender,
            event_listeners: Vec::new(),
            stats: PerformanceMonitorStats::default(),
            last_collection: Arc::new(Mutex::new(Instant::now())),
        };

        (monitor, event_receiver)
    }

    /// 收集性能指标
    pub async fn collect_metrics(&self, task_result: &TaskResult) -> Result<(), ProofError> {
        let start_time = Instant::now();
        
        // 收集任务执行指标
        let execution_metric = PerformanceMetric {
            metric_type: MetricType::ExecutionTime,
            value: task_result.execution_time.as_secs_f64(),
            unit: "seconds".to_string(),
            timestamp: Instant::now(),
            metadata: HashMap::new(),
        };

        self.add_metric(execution_metric).await?;

        // 收集资源使用指标
        if let Some(resource_usage) = &task_result.resource_usage {
            let resource_metric = PerformanceMetric {
                metric_type: MetricType::ResourceUtilization,
                value: self.calculate_resource_utilization(resource_usage),
                unit: "percentage".to_string(),
                timestamp: Instant::now(),
                metadata: HashMap::new(),
            };

            self.add_metric(resource_metric).await?;
        }

        // 收集成功率指标
        let success_metric = PerformanceMetric {
            metric_type: MetricType::SuccessRate,
            value: if task_result.status == crate::automation::AutomationTaskStatus::Completed { 1.0 } else { 0.0 },
            unit: "boolean".to_string(),
            timestamp: Instant::now(),
            metadata: HashMap::new(),
        };

        self.add_metric(success_metric).await?;

        // 更新统计
        let collection_duration = start_time.elapsed();
        self.stats.last_collection_duration = collection_duration;
        self.stats.total_metrics_collected += 3; // 3个指标

        // 检查阈值和生成告警
        if self.config.enable_alerting {
            self.check_thresholds_and_generate_alerts().await?;
        }

        // 清理旧指标
        self.cleanup_old_metrics().await?;

        Ok(())
    }

    /// 添加性能指标
    async fn add_metric(&self, metric: PerformanceMetric) -> Result<(), ProofError> {
        let mut history = self.metrics_history.write().await;
        
        let metric_queue = history.entry(metric.metric_type.clone())
            .or_insert_with(|| VecDeque::new());
        
        metric_queue.push_back(metric);
        
        // 限制历史记录数量
        if metric_queue.len() > self.config.max_metrics_history {
            metric_queue.pop_front();
        }

        Ok(())
    }

    /// 计算资源使用率
    fn calculate_resource_utilization(&self, resource_usage: &ResourceUsage) -> f64 {
        // 简化的资源使用率计算
        let cpu_usage = resource_usage.cpu_usage.unwrap_or(0.0);
        let memory_usage = resource_usage.memory_usage.unwrap_or(0.0);
        let storage_usage = resource_usage.storage_usage.unwrap_or(0.0);
        
        (cpu_usage + memory_usage + storage_usage) / 3.0
    }

    /// 设置性能阈值
    pub async fn set_threshold(
        &self,
        metric_type: MetricType,
        warning_threshold: f64,
        critical_threshold: f64,
    ) -> Result<(), ProofError> {
        let mut thresholds = self.thresholds.write().await;
        
        let threshold = PerformanceThreshold {
            metric_type: metric_type.clone(),
            warning_threshold,
            critical_threshold,
            enabled: true,
        };
        
        thresholds.insert(metric_type, threshold);
        
        Ok(())
    }

    /// 检查阈值并生成告警
    async fn check_thresholds_and_generate_alerts(&self) -> Result<(), ProofError> {
        let thresholds = self.thresholds.read().await;
        let history = self.metrics_history.read().await;
        
        for (metric_type, threshold) in thresholds.iter() {
            if !threshold.enabled {
                continue;
            }

            if let Some(metrics) = history.get(metric_type) {
                if let Some(latest_metric) = metrics.back() {
                    let value = latest_metric.value;
                    
                    // 检查严重阈值
                    if value >= threshold.critical_threshold {
                        self.generate_alert(
                            AlertLevel::Critical,
                            metric_type,
                            value,
                            threshold.critical_threshold,
                            &format!("指标 {} 达到严重阈值: {} >= {}", 
                                format!("{:?}", metric_type), value, threshold.critical_threshold),
                        ).await?;
                    }
                    // 检查警告阈值
                    else if value >= threshold.warning_threshold {
                        self.generate_alert(
                            AlertLevel::Warning,
                            metric_type,
                            value,
                            threshold.warning_threshold,
                            &format!("指标 {} 达到警告阈值: {} >= {}", 
                                format!("{:?}", metric_type), value, threshold.warning_threshold),
                        ).await?;
                    }
                }
            }
        }

        Ok(())
    }

    /// 生成告警
    async fn generate_alert(
        &self,
        level: AlertLevel,
        metric_type: &MetricType,
        current_value: f64,
        threshold_value: f64,
        message: &str,
    ) -> Result<(), ProofError> {
        let alert = PerformanceAlert {
            id: format!("alert_{}", Instant::now().elapsed().as_nanos()),
            level,
            message: message.to_string(),
            metric_type: metric_type.clone(),
            current_value,
            threshold_value,
            timestamp: Instant::now(),
            acknowledged: false,
            resolved: false,
        };

        let mut alerts = self.alerts.lock().unwrap();
        alerts.push_back(alert.clone());
        
        // 限制告警数量
        if alerts.len() > 100 {
            alerts.pop_front();
        }

        self.stats.total_alerts_generated += 1;

        // 发送告警事件
        let _ = self.event_sender.try_send(AutomationEvent::PerformanceAlert {
            alert_level: format!("{:?}", level),
            message: message.to_string(),
            timestamp: Instant::now(),
        });

        Ok(())
    }

    /// 清理旧指标
    async fn cleanup_old_metrics(&self) -> Result<(), ProofError> {
        let cutoff_time = Instant::now() - self.config.retention_period;
        let mut history = self.metrics_history.write().await;
        
        for metrics in history.values_mut() {
            while let Some(front) = metrics.front() {
                if front.timestamp < cutoff_time {
                    metrics.pop_front();
                } else {
                    break;
                }
            }
        }

        Ok(())
    }

    /// 执行性能分析
    pub async fn analyze_performance(&self) -> Result<PerformanceAnalysis, ProofError> {
        let start_time = Instant::now();
        
        let history = self.metrics_history.read().await;
        
        let mut overall_score = 0.0;
        let mut bottleneck_metrics = Vec::new();
        let mut improvement_suggestions = Vec::new();
        let mut resource_efficiency = 0.0;

        // 分析各种指标
        for (metric_type, metrics) in history.iter() {
            if let Some(analysis) = self.analyze_metric_type(metric_type, metrics).await? {
                overall_score += analysis.score;
                
                if analysis.is_bottleneck {
                    bottleneck_metrics.push(analysis.latest_metric);
                }
                
                improvement_suggestions.extend(analysis.suggestions);
                
                if matches!(metric_type, MetricType::ResourceUtilization) {
                    resource_efficiency = analysis.score;
                }
            }
        }

        // 计算平均分数
        if !history.is_empty() {
            overall_score /= history.len() as f64;
        }

        // 执行趋势分析
        let trend_analysis = self.perform_trend_analysis(&history).await?;

        let analysis = PerformanceAnalysis {
            overall_score,
            bottleneck_metrics,
            improvement_suggestions,
            trend_analysis,
            resource_efficiency,
        };

        self.stats.total_analyses_performed += 1;
        self.stats.average_collection_time = 
            (self.stats.average_collection_time + start_time.elapsed()) / 2;

        Ok(analysis)
    }

    /// 分析特定指标类型
    async fn analyze_metric_type(
        &self,
        metric_type: &MetricType,
        metrics: &VecDeque<PerformanceMetric>,
    ) -> Result<Option<MetricAnalysis>, ProofError> {
        if metrics.is_empty() {
            return Ok(None);
        }

        let latest_metric = metrics.back().unwrap();
        let mut score = 0.0;
        let mut is_bottleneck = false;
        let mut suggestions = Vec::new();

        match metric_type {
            MetricType::ExecutionTime => {
                // 执行时间越短越好
                score = (1.0 / (1.0 + latest_metric.value)).min(1.0);
                if latest_metric.value > 10.0 { // 超过10秒
                    is_bottleneck = true;
                    suggestions.push("优化算法或增加计算资源".to_string());
                }
            }
            MetricType::SuccessRate => {
                // 成功率越高越好
                score = latest_metric.value;
                if latest_metric.value < 0.9 { // 低于90%
                    is_bottleneck = true;
                    suggestions.push("检查错误原因，改进错误处理".to_string());
                }
            }
            MetricType::ResourceUtilization => {
                // 资源使用率适中最好
                let utilization = latest_metric.value;
                if utilization < 0.3 {
                    score = 0.5; // 资源浪费
                    suggestions.push("优化资源分配，减少资源浪费".to_string());
                } else if utilization > 0.8 {
                    score = 0.3; // 资源紧张
                    is_bottleneck = true;
                    suggestions.push("增加资源或优化资源使用".to_string());
                } else {
                    score = 1.0; // 理想状态
                }
            }
            _ => {
                score = 0.5; // 默认分数
            }
        }

        Ok(Some(MetricAnalysis {
            score,
            is_bottleneck,
            latest_metric: latest_metric.clone(),
            suggestions,
        }))
    }

    /// 执行趋势分析
    async fn perform_trend_analysis(
        &self,
        history: &HashMap<MetricType, VecDeque<PerformanceMetric>>,
    ) -> Result<TrendAnalysis, ProofError> {
        let mut overall_trend = 0.0;
        let mut trend_count = 0;

        for metrics in history.values() {
            if metrics.len() >= 2 {
                let trend = self.calculate_trend(metrics);
                overall_trend += trend;
                trend_count += 1;
            }
        }

        let average_trend = if trend_count > 0 { overall_trend / trend_count as f64 } else { 0.0 };
        
        let (direction, strength) = self.classify_trend(average_trend);
        let prediction = self.predict_future_value(history).await?;
        let confidence = self.calculate_confidence(history).await?;

        Ok(TrendAnalysis {
            trend_direction: direction,
            trend_strength: strength,
            prediction,
            confidence,
        })
    }

    /// 计算趋势
    fn calculate_trend(&self, metrics: &VecDeque<PerformanceMetric>) -> f64 {
        if metrics.len() < 2 {
            return 0.0;
        }

        let recent: Vec<_> = metrics.iter().rev().take(5).collect();
        if recent.len() < 2 {
            return 0.0;
        }

        let first_value = recent.last().unwrap().value;
        let last_value = recent.first().unwrap().value;
        
        last_value - first_value
    }

    /// 分类趋势
    fn classify_trend(&self, trend: f64) -> (TrendDirection, f64) {
        let abs_trend = trend.abs();
        let strength = (abs_trend / 10.0).min(1.0); // 标准化强度

        let direction = if abs_trend < 0.1 {
            TrendDirection::Stable
        } else if trend > 0.1 {
            TrendDirection::Increasing
        } else if trend < -0.1 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Fluctuating
        };

        (direction, strength)
    }

    /// 预测未来值
    async fn predict_future_value(
        &self,
        _history: &HashMap<MetricType, VecDeque<PerformanceMetric>>,
    ) -> Result<Option<f64>, ProofError> {
        // 简化的预测实现
        // 在实际应用中，这里可以使用更复杂的预测算法
        Ok(None)
    }

    /// 计算置信度
    async fn calculate_confidence(
        &self,
        history: &HashMap<MetricType, VecDeque<PerformanceMetric>>,
    ) -> Result<f64, ProofError> {
        let mut total_confidence = 0.0;
        let mut count = 0;

        for metrics in history.values() {
            if metrics.len() >= 5 {
                // 数据点越多，置信度越高
                let confidence = (metrics.len() as f64 / 100.0).min(1.0);
                total_confidence += confidence;
                count += 1;
            }
        }

        Ok(if count > 0 { total_confidence / count as f64 } else { 0.0 })
    }

    /// 获取告警列表
    pub fn get_alerts(&self) -> Vec<PerformanceAlert> {
        let alerts = self.alerts.lock().unwrap();
        alerts.iter().cloned().collect()
    }

    /// 确认告警
    pub fn acknowledge_alert(&self, alert_id: &str) -> Result<(), ProofError> {
        let mut alerts = self.alerts.lock().unwrap();
        
        if let Some(alert) = alerts.iter_mut().find(|a| a.id == alert_id) {
            alert.acknowledged = true;
        }

        Ok(())
    }

    /// 解决告警
    pub fn resolve_alert(&self, alert_id: &str) -> Result<(), ProofError> {
        let mut alerts = self.alerts.lock().unwrap();
        
        if let Some(alert) = alerts.iter_mut().find(|a| a.id == alert_id) {
            alert.resolved = true;
        }

        Ok(())
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> PerformanceMonitorStats {
        self.stats.clone()
    }

    /// 添加事件监听器
    pub fn add_event_listener(&mut self, listener: Box<dyn EventListener + Send + Sync>) {
        self.event_listeners.push(listener);
    }
}

/// 指标分析结果
#[derive(Debug)]
struct MetricAnalysis {
    score: f64,
    is_bottleneck: bool,
    latest_metric: PerformanceMetric,
    suggestions: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automation::{AutomationTaskStatus, ResourceUsage};

    fn create_test_task_result() -> TaskResult {
        TaskResult {
            task_id: "test_task".to_string(),
            status: AutomationTaskStatus::Completed,
            result: Some("成功".to_string()),
            error: None,
            execution_time: Duration::from_secs(5),
            resource_usage: ResourceUsage {
                cpu_usage: Some(0.6),
                memory_usage: Some(0.4),
                storage_usage: Some(0.2),
                network_usage: Some(0.1),
            },
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_performance_monitor_creation() {
        let config = PerformanceMonitorConfig::default();
        let (monitor, _) = PerformanceMonitor::new(config);
        
        assert_eq!(monitor.stats.total_metrics_collected, 0);
        assert_eq!(monitor.stats.total_alerts_generated, 0);
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let config = PerformanceMonitorConfig::default();
        let (monitor, _) = PerformanceMonitor::new(config);
        
        let task_result = create_test_task_result();
        let result = monitor.collect_metrics(&task_result).await;
        
        assert!(result.is_ok());
        assert_eq!(monitor.stats.total_metrics_collected, 3);
    }

    #[tokio::test]
    async fn test_threshold_setting() {
        let config = PerformanceMonitorConfig::default();
        let (monitor, _) = PerformanceMonitor::new(config);
        
        let result = monitor.set_threshold(
            MetricType::ExecutionTime,
            10.0,
            20.0,
        ).await;
        
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_performance_analysis() {
        let config = PerformanceMonitorConfig::default();
        let (monitor, _) = PerformanceMonitor::new(config);
        
        // 先收集一些指标
        let task_result = create_test_task_result();
        monitor.collect_metrics(&task_result).await.unwrap();
        
        let analysis = monitor.analyze_performance().await;
        assert!(analysis.is_ok());
        
        let analysis = analysis.unwrap();
        assert!(analysis.overall_score >= 0.0 && analysis.overall_score <= 1.0);
    }
}
