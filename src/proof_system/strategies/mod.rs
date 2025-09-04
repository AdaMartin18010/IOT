pub mod automated;
pub mod interactive;
pub mod hybrid;
pub mod selector;
pub mod optimizer;

pub use automated::*;
pub use interactive::*;
pub use hybrid::*;
pub use selector::*;
pub use optimizer::*;

use crate::core::{Proof, ProofStep, InferenceRule, ProofError, ProofStatus};
use std::collections::HashMap;

/// 证明策略执行结果
#[derive(Debug, Clone)]
pub struct StrategyExecutionResult {
    /// 是否成功
    pub success: bool,
    /// 生成的证明步骤
    pub generated_steps: Vec<ProofStep>,
    /// 应用的规则
    pub applied_rules: Vec<InferenceRule>,
    /// 执行时间（毫秒）
    pub execution_time_ms: u64,
    /// 错误信息
    pub error: Option<String>,
    /// 策略建议
    pub suggestions: Vec<String>,
}

/// 策略性能指标
#[derive(Debug, Clone)]
pub struct StrategyPerformanceMetrics {
    /// 成功率
    pub success_rate: f64,
    /// 平均执行时间
    pub avg_execution_time_ms: u64,
    /// 平均生成步骤数
    pub avg_steps_generated: f64,
    /// 规则应用效率
    pub rule_application_efficiency: f64,
    /// 内存使用量
    pub memory_usage_mb: u64,
}

/// 策略配置
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// 最大执行时间（毫秒）
    pub max_execution_time_ms: u64,
    /// 最大步骤数
    pub max_steps: usize,
    /// 是否启用并行执行
    pub enable_parallel: bool,
    /// 是否启用缓存
    pub enable_caching: bool,
    /// 策略特定参数
    pub parameters: HashMap<String, String>,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            max_execution_time_ms: 30000, // 30秒
            max_steps: 1000,
            enable_parallel: true,
            enable_caching: true,
            parameters: HashMap::new(),
        }
    }
}
