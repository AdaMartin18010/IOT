use crate::core::{Proof, ProofStep, InferenceRule, ProofError, ProofStatus, Proposition, RuleLibrary};
use crate::strategies::{StrategyExecutionResult, StrategyPerformanceMetrics, StrategyConfig};
use crate::strategies::selector::{StrategySelector, StrategyType, SelectionRule};
use std::collections::HashMap;
use std::time::Instant;

/// 优化目标
#[derive(Debug, Clone)]
pub enum OptimizationTarget {
    /// 最大化成功率
    MaximizeSuccessRate,
    /// 最小化执行时间
    MinimizeExecutionTime,
    /// 最小化步骤数
    MinimizeSteps,
    /// 平衡多个目标
    Balanced,
}

/// 优化配置
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// 优化目标
    pub target: OptimizationTarget,
    /// 最大迭代次数
    pub max_iterations: usize,
    /// 收敛阈值
    pub convergence_threshold: f64,
    /// 是否启用交叉验证
    pub enable_cross_validation: bool,
    /// 交叉验证折数
    pub cross_validation_folds: usize,
    /// 学习率
    pub learning_rate: f64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            target: OptimizationTarget::Balanced,
            max_iterations: 100,
            convergence_threshold: 0.001,
            enable_cross_validation: true,
            cross_validation_folds: 5,
            learning_rate: 0.1,
        }
    }
}

/// 策略优化器
pub struct StrategyOptimizer {
    config: OptimizationConfig,
    optimization_history: Vec<OptimizationStep>,
    best_configuration: Option<OptimizedConfiguration>,
}

/// 优化步骤
#[derive(Debug, Clone)]
pub struct OptimizationStep {
    /// 迭代次数
    pub iteration: usize,
    /// 当前配置
    pub configuration: OptimizedConfiguration,
    /// 性能指标
    pub performance: StrategyPerformanceMetrics,
    /// 改进幅度
    pub improvement: f64,
}

/// 优化后的配置
#[derive(Debug, Clone)]
pub struct OptimizedConfiguration {
    /// 策略配置
    pub strategy_configs: HashMap<StrategyType, StrategyConfig>,
    /// 选择规则权重
    pub rule_weights: HashMap<String, f64>,
    /// 策略选择优先级
    pub strategy_priorities: Vec<StrategyType>,
}

impl StrategyOptimizer {
    /// 创建新的策略优化器
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            config,
            optimization_history: Vec::new(),
            best_configuration: None,
        }
    }

    /// 优化策略选择器
    pub fn optimize_selector(&mut self, selector: &mut StrategySelector, training_proofs: &[Proof]) -> OptimizedConfiguration {
        let mut current_config = self.initialize_configuration(selector);
        let mut best_performance = 0.0;
        let mut iteration = 0;
        
        while iteration < self.config.max_iterations {
            // 评估当前配置
            let performance = self.evaluate_configuration(selector, &current_config, training_proofs);
            
            // 记录优化步骤
            let improvement = if iteration > 0 {
                performance.success_rate - best_performance
            } else {
                0.0
            };
            
            self.optimization_history.push(OptimizationStep {
                iteration,
                configuration: current_config.clone(),
                performance: performance.clone(),
                improvement,
            });
            
            // 检查是否收敛
            if iteration > 0 && improvement.abs() < self.config.convergence_threshold {
                break;
            }
            
            // 更新最佳配置
            if performance.success_rate > best_performance {
                best_performance = performance.success_rate;
                self.best_configuration = Some(current_config.clone());
            }
            
            // 生成新配置
            current_config = self.generate_next_configuration(&current_config, &performance);
            
            iteration += 1;
        }
        
        self.best_configuration.clone().unwrap_or_else(|| current_config)
    }

    /// 初始化配置
    fn initialize_configuration(&self, selector: &StrategySelector) -> OptimizedConfiguration {
        let mut strategy_configs = HashMap::new();
        let mut rule_weights = HashMap::new();
        let mut strategy_priorities = Vec::new();
        
        // 为每个策略类型设置默认配置
        for strategy_type in selector.get_available_strategies() {
            strategy_configs.insert(strategy_type.clone(), StrategyConfig::default());
            strategy_priorities.push(strategy_type);
        }
        
        // 设置默认规则权重
        rule_weights.insert("ComplexityBasedRule".to_string(), 0.4);
        rule_weights.insert("PerformanceBasedRule".to_string(), 0.3);
        rule_weights.insert("HistoryBasedRule".to_string(), 0.3);
        
        OptimizedConfiguration {
            strategy_configs,
            rule_weights,
            strategy_priorities,
        }
    }

    /// 评估配置性能
    fn evaluate_configuration(
        &self,
        selector: &mut StrategySelector,
        config: &OptimizedConfiguration,
        training_proofs: &[Proof],
    ) -> StrategyPerformanceMetrics {
        // 应用配置到选择器
        self.apply_configuration(selector, config);
        
        let mut total_success_rate = 0.0;
        let mut total_execution_time = 0;
        let mut total_steps = 0.0;
        let mut total_efficiency = 0.0;
        let mut proof_count = 0;
        
        for proof in training_proofs {
            let mut proof_copy = proof.clone();
            let start_time = Instant::now();
            
            // 执行策略选择
            if let Ok(result) = selector.execute_selected_strategy(&mut proof_copy) {
                total_success_rate += if result.success { 1.0 } else { 0.0 };
                total_execution_time += result.execution_time_ms;
                total_steps += result.generated_steps.len() as f64;
                total_efficiency += result.applied_rules.len() as f64 / (result.generated_steps.len() as f64 + 1.0);
                proof_count += 1;
            }
        }
        
        if proof_count == 0 {
            return StrategyPerformanceMetrics {
                success_rate: 0.0,
                avg_execution_time_ms: 0,
                avg_steps_generated: 0.0,
                rule_application_efficiency: 0.0,
                memory_usage_mb: 0,
            };
        }
        
        StrategyPerformanceMetrics {
            success_rate: total_success_rate / proof_count as f64,
            avg_execution_time_ms: total_execution_time / proof_count as u64,
            avg_steps_generated: total_steps / proof_count as f64,
            rule_application_efficiency: total_efficiency / proof_count as f64,
            memory_usage_mb: 0,
        }
    }

    /// 应用配置到选择器
    fn apply_configuration(&self, selector: &mut StrategySelector, config: &OptimizedConfiguration) {
        // 这里应该实现配置应用逻辑
        // 目前只是占位符
    }

    /// 生成下一个配置
    fn generate_next_configuration(
        &self,
        current_config: &OptimizedConfiguration,
        performance: &StrategyPerformanceMetrics,
    ) -> OptimizedConfiguration {
        let mut new_config = current_config.clone();
        
        // 基于性能调整配置
        match self.config.target {
            OptimizationTarget::MaximizeSuccessRate => {
                self.optimize_for_success_rate(&mut new_config, performance);
            }
            OptimizationTarget::MinimizeExecutionTime => {
                self.optimize_for_execution_time(&mut new_config, performance);
            }
            OptimizationTarget::MinimizeSteps => {
                self.optimize_for_minimize_steps(&mut new_config, performance);
            }
            OptimizationTarget::Balanced => {
                self.optimize_balanced(&mut new_config, performance);
            }
        }
        
        new_config
    }

    /// 为成功率优化
    fn optimize_for_success_rate(&self, config: &mut OptimizedConfiguration, performance: &StrategyPerformanceMetrics) {
        // 调整规则权重
        if performance.success_rate < 0.8 {
            // 增加复杂度规则的权重
            if let Some(weight) = config.rule_weights.get_mut("ComplexityBasedRule") {
                *weight += self.config.learning_rate;
            }
            
            // 减少性能规则的权重
            if let Some(weight) = config.rule_weights.get_mut("PerformanceBasedRule") {
                *weight -= self.config.learning_rate * 0.5;
            }
        }
        
        // 归一化权重
        self.normalize_weights(&mut config.rule_weights);
    }

    /// 为执行时间优化
    fn optimize_for_execution_time(&self, config: &mut OptimizedConfiguration, performance: &StrategyPerformanceMetrics) {
        // 调整策略配置
        for (_, strategy_config) in &mut config.strategy_configs {
            if performance.avg_execution_time_ms > 1000 {
                // 减少最大步骤数
                strategy_config.max_steps = (strategy_config.max_steps as f64 * 0.9) as usize;
                // 启用并行执行
                strategy_config.enable_parallel = true;
            }
        }
    }

    /// 为最小化步骤数优化
    fn optimize_for_minimize_steps(&self, config: &mut OptimizedConfiguration, performance: &StrategyPerformanceMetrics) {
        // 调整策略配置
        for (_, strategy_config) in &mut config.strategy_configs {
            if performance.avg_steps_generated > 10.0 {
                // 减少最大步骤数
                strategy_config.max_steps = (strategy_config.max_steps as f64 * 0.8) as usize;
                // 启用缓存
                strategy_config.enable_caching = true;
            }
        }
    }

    /// 平衡优化
    fn optimize_balanced(&self, config: &mut OptimizedConfiguration, performance: &StrategyPerformanceMetrics) {
        // 综合考虑多个指标
        self.optimize_for_success_rate(config, performance);
        self.optimize_for_execution_time(config, performance);
        self.optimize_for_minimize_steps(config, performance);
    }

    /// 归一化权重
    fn normalize_weights(&self, weights: &mut HashMap<String, f64>) {
        let total_weight: f64 = weights.values().sum();
        if total_weight > 0.0 {
            for weight in weights.values_mut() {
                *weight /= total_weight;
            }
        }
    }

    /// 获取优化历史
    pub fn get_optimization_history(&self) -> &[OptimizationStep] {
        &self.optimization_history
    }

    /// 获取最佳配置
    pub fn get_best_configuration(&self) -> Option<&OptimizedConfiguration> {
        self.best_configuration.as_ref()
    }

    /// 获取优化统计
    pub fn get_optimization_stats(&self) -> OptimizationStats {
        let total_iterations = self.optimization_history.len();
        let best_performance = self.optimization_history.iter()
            .map(|step| step.performance.success_rate)
            .fold(0.0, f64::max);
        
        let avg_improvement = if total_iterations > 1 {
            self.optimization_history.iter()
                .skip(1)
                .map(|step| step.improvement)
                .sum::<f64>() / (total_iterations - 1) as f64
        } else {
            0.0
        };
        
        OptimizationStats {
            total_iterations,
            best_performance,
            avg_improvement,
            convergence_reached: total_iterations < self.config.max_iterations,
        }
    }
}

/// 优化统计
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    /// 总迭代次数
    pub total_iterations: usize,
    /// 最佳性能
    pub best_performance: f64,
    /// 平均改进幅度
    pub avg_improvement: f64,
    /// 是否达到收敛
    pub convergence_reached: bool,
}

/// 自适应优化器
pub struct AdaptiveOptimizer {
    base_optimizer: StrategyOptimizer,
    adaptation_rate: f64,
    performance_window: Vec<StrategyPerformanceMetrics>,
    window_size: usize,
}

impl AdaptiveOptimizer {
    /// 创建新的自适应优化器
    pub fn new(config: OptimizationConfig, adaptation_rate: f64, window_size: usize) -> Self {
        Self {
            base_optimizer: StrategyOptimizer::new(config),
            adaptation_rate,
            performance_window: Vec::new(),
            window_size,
        }
    }

    /// 自适应优化
    pub fn adaptive_optimize(
        &mut self,
        selector: &mut StrategySelector,
        training_proofs: &[Proof],
        current_performance: StrategyPerformanceMetrics,
    ) -> OptimizedConfiguration {
        // 添加到性能窗口
        self.performance_window.push(current_performance);
        if self.performance_window.len() > self.window_size {
            self.performance_window.remove(0);
        }
        
        // 分析性能趋势
        let trend = self.analyze_performance_trend();
        
        // 根据趋势调整优化配置
        let adjusted_config = self.adjust_optimization_config(trend);
        
        // 执行优化
        self.base_optimizer.optimize_selector(selector, training_proofs)
    }

    /// 分析性能趋势
    fn analyze_performance_trend(&self) -> f64 {
        if self.performance_window.len() < 2 {
            return 0.0;
        }
        
        let recent_performance: f64 = self.performance_window.iter()
            .rev()
            .take(self.window_size / 2)
            .map(|p| p.success_rate)
            .sum();
        
        let older_performance: f64 = self.performance_window.iter()
            .take(self.window_size / 2)
            .map(|p| p.success_rate)
            .sum();
        
        let recent_count = (self.window_size / 2).min(self.performance_window.len() / 2);
        let older_count = (self.window_size / 2).min(self.performance_window.len() / 2);
        
        if recent_count == 0 || older_count == 0 {
            return 0.0;
        }
        
        (recent_performance / recent_count as f64) - (older_performance / older_count as f64)
    }

    /// 调整优化配置
    fn adjust_optimization_config(&self, trend: f64) -> OptimizationConfig {
        let mut config = self.base_optimizer.config.clone();
        
        if trend < -0.1 {
            // 性能下降，增加学习率
            config.learning_rate *= 1.5;
            config.max_iterations = (config.max_iterations as f64 * 1.2) as usize;
        } else if trend > 0.1 {
            // 性能提升，减少学习率
            config.learning_rate *= 0.8;
            config.max_iterations = (config.max_iterations as f64 * 0.9) as usize;
        }
        
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Proof, Proposition, PropositionType, RuleLibrary};

    #[test]
    fn test_optimizer_creation() {
        let config = OptimizationConfig::default();
        let optimizer = StrategyOptimizer::new(config);
        assert_eq!(optimizer.optimization_history.len(), 0);
    }

    #[test]
    fn test_optimization_config_default() {
        let config = OptimizationConfig::default();
        assert_eq!(config.max_iterations, 100);
        assert_eq!(config.learning_rate, 0.1);
    }

    #[test]
    fn test_adaptive_optimizer_creation() {
        let config = OptimizationConfig::default();
        let adaptive_optimizer = AdaptiveOptimizer::new(config, 0.1, 10);
        assert_eq!(adaptive_optimizer.window_size, 10);
    }

    #[test]
    fn test_weight_normalization() {
        let mut weights = HashMap::new();
        weights.insert("A".to_string(), 2.0);
        weights.insert("B".to_string(), 3.0);
        weights.insert("C".to_string(), 5.0);
        
        let optimizer = StrategyOptimizer::new(OptimizationConfig::default());
        optimizer.normalize_weights(&mut weights);
        
        let total: f64 = weights.values().sum();
        assert!((total - 1.0).abs() < 0.001);
    }
}
