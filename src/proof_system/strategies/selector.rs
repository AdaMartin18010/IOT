use crate::core::{Proof, ProofStep, InferenceRule, ProofError, ProofStatus, Proposition, RuleLibrary};
use crate::strategies::{StrategyExecutionResult, StrategyPerformanceMetrics, StrategyConfig};
use crate::strategies::automated::AutomatedProofStrategy;
use crate::strategies::interactive::InteractiveProofStrategy;
use crate::strategies::hybrid::{HybridProofStrategy, HybridMode};
use std::collections::HashMap;
use std::time::Instant;

/// 策略类型
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StrategyType {
    Automated,
    Interactive,
    Hybrid(HybridMode),
}

/// 策略特征
#[derive(Debug, Clone)]
pub struct StrategyCharacteristics {
    /// 策略类型
    pub strategy_type: StrategyType,
    /// 适用复杂度范围
    pub complexity_range: (f64, f64),
    /// 适用证明长度范围
    pub proof_length_range: (usize, usize),
    /// 适用命题类型
    pub supported_proposition_types: Vec<crate::core::PropositionType>,
    /// 适用规则类型
    pub supported_rule_types: Vec<crate::core::RuleType>,
    /// 预期成功率
    pub expected_success_rate: f64,
    /// 预期执行时间（毫秒）
    pub expected_execution_time_ms: u64,
}

/// 策略选择器
pub struct StrategySelector {
    strategies: HashMap<StrategyType, Box<dyn ProofStrategy>>,
    characteristics: HashMap<StrategyType, StrategyCharacteristics>,
    performance_history: HashMap<StrategyType, Vec<StrategyPerformanceMetrics>>,
    selection_rules: Vec<Box<dyn SelectionRule>>,
    learning_enabled: bool,
}

/// 证明策略特征
pub trait ProofStrategy {
    /// 执行策略
    fn execute(&mut self, proof: &mut Proof) -> StrategyExecutionResult;
    
    /// 获取策略类型
    fn strategy_type(&self) -> StrategyType;
    
    /// 获取策略特征
    fn characteristics(&self) -> StrategyCharacteristics;
    
    /// 检查是否适用于给定证明
    fn is_applicable(&self, proof: &Proof) -> bool;
    
    /// 获取适用性分数
    fn applicability_score(&self, proof: &Proof) -> f64;
}

/// 选择规则特征
pub trait SelectionRule {
    /// 应用选择规则
    fn apply(&self, proof: &Proof, strategies: &[StrategyType]) -> Vec<StrategyType>;
    
    /// 获取规则权重
    fn weight(&self) -> f64;
    
    /// 获取规则名称
    fn name(&self) -> &str;
}

impl StrategySelector {
    /// 创建新的策略选择器
    pub fn new() -> Self {
        let mut selector = Self {
            strategies: HashMap::new(),
            characteristics: HashMap::new(),
            performance_history: HashMap::new(),
            selection_rules: Vec::new(),
            learning_enabled: true,
        };
        
        // 添加默认选择规则
        selector.add_selection_rule(Box::new(ComplexityBasedRule::new()));
        selector.add_selection_rule(Box::new(PerformanceBasedRule::new()));
        selector.add_selection_rule(Box::new(HistoryBasedRule::new()));
        
        selector
    }

    /// 注册策略
    pub fn register_strategy(&mut self, strategy: Box<dyn ProofStrategy>) {
        let strategy_type = strategy.strategy_type();
        let characteristics = strategy.characteristics();
        
        self.strategies.insert(strategy_type.clone(), strategy);
        self.characteristics.insert(strategy_type, characteristics);
    }

    /// 添加选择规则
    pub fn add_selection_rule(&mut self, rule: Box<dyn SelectionRule>) {
        self.selection_rules.push(rule);
    }

    /// 选择最佳策略
    pub fn select_strategy(&self, proof: &Proof) -> Option<StrategyType> {
        let applicable_strategies: Vec<StrategyType> = self.strategies.keys()
            .filter(|strategy_type| {
                if let Some(strategy) = self.strategies.get(*strategy_type) {
                    strategy.is_applicable(proof)
                } else {
                    false
                }
            })
            .cloned()
            .collect();

        if applicable_strategies.is_empty() {
            return None;
        }

        // 应用所有选择规则
        let mut strategy_scores: HashMap<StrategyType, f64> = HashMap::new();
        
        for rule in &self.selection_rules {
            let selected = rule.apply(proof, &applicable_strategies);
            let weight = rule.weight();
            
            for strategy_type in selected {
                let score = strategy_scores.get(&strategy_type).unwrap_or(&0.0) + weight;
                strategy_scores.insert(strategy_type, score);
            }
        }

        // 选择得分最高的策略
        strategy_scores.into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(strategy_type, _)| strategy_type)
    }

    /// 执行选定的策略
    pub fn execute_selected_strategy(&mut self, proof: &mut Proof) -> StrategyExecutionResult {
        let start_time = Instant::now();
        
        // 选择策略
        let selected_strategy_type = self.select_strategy(proof)
            .ok_or_else(|| ProofError::StrategySelectionFailed("没有可用的策略".to_string()))?;
        
        // 执行策略
        let strategy = self.strategies.get_mut(&selected_strategy_type)
            .ok_or_else(|| ProofError::StrategyExecutionFailed("策略未找到".to_string()))?;
        
        let result = strategy.execute(proof);
        
        // 记录性能历史
        if self.learning_enabled {
            self.record_performance(selected_strategy_type, &result, start_time.elapsed().as_millis() as u64);
        }
        
        Ok(result)
    }

    /// 记录策略性能
    fn record_performance(&mut self, strategy_type: StrategyType, result: &StrategyExecutionResult, execution_time: u64) {
        let metrics = StrategyPerformanceMetrics {
            success_rate: if result.success { 1.0 } else { 0.0 },
            avg_execution_time_ms: execution_time,
            avg_steps_generated: result.generated_steps.len() as f64,
            rule_application_efficiency: if result.applied_rules.is_empty() { 0.0 } else { 1.0 },
            memory_usage_mb: 0, // 这里应该实现实际的内存使用监控
        };
        
        self.performance_history.entry(strategy_type)
            .or_insert_with(Vec::new)
            .push(metrics);
    }

    /// 获取策略性能统计
    pub fn get_strategy_performance(&self, strategy_type: &StrategyType) -> Option<StrategyPerformanceMetrics> {
        self.performance_history.get(strategy_type)
            .and_then(|history| {
                if history.is_empty() {
                    None
                } else {
                    let avg_success_rate = history.iter().map(|m| m.success_rate).sum::<f64>() / history.len() as f64;
                    let avg_execution_time = history.iter().map(|m| m.avg_execution_time_ms).sum::<u64>() / history.len() as u64;
                    let avg_steps = history.iter().map(|m| m.avg_steps_generated).sum::<f64>() / history.len() as f64;
                    let avg_efficiency = history.iter().map(|m| m.rule_application_efficiency).sum::<f64>() / history.len() as f64;
                    
                    Some(StrategyPerformanceMetrics {
                        success_rate: avg_success_rate,
                        avg_execution_time_ms: avg_execution_time,
                        avg_steps_generated: avg_steps,
                        rule_application_efficiency: avg_efficiency,
                        memory_usage_mb: 0,
                    })
                }
            })
    }

    /// 启用/禁用学习
    pub fn set_learning_enabled(&mut self, enabled: bool) {
        self.learning_enabled = enabled;
    }

    /// 获取所有可用策略
    pub fn get_available_strategies(&self) -> Vec<StrategyType> {
        self.strategies.keys().cloned().collect()
    }

    /// 获取策略特征
    pub fn get_strategy_characteristics(&self, strategy_type: &StrategyType) -> Option<&StrategyCharacteristics> {
        self.characteristics.get(strategy_type)
    }
}

/// 基于复杂度的选择规则
struct ComplexityBasedRule {
    weight: f64,
}

impl ComplexityBasedRule {
    fn new() -> Self {
        Self { weight: 0.4 }
    }
}

impl SelectionRule for ComplexityBasedRule {
    fn apply(&self, proof: &Proof, strategies: &[StrategyType]) -> Vec<StrategyType> {
        let complexity = Self::calculate_complexity(proof);
        let mut selected = Vec::new();
        
        for strategy_type in strategies {
            match strategy_type {
                StrategyType::Automated => {
                    if complexity < 10.0 {
                        selected.push(strategy_type.clone());
                    }
                }
                StrategyType::Interactive => {
                    if complexity >= 5.0 && complexity < 20.0 {
                        selected.push(strategy_type.clone());
                    }
                }
                StrategyType::Hybrid(HybridMode::AutoFirst) => {
                    if complexity < 15.0 {
                        selected.push(strategy_type.clone());
                    }
                }
                StrategyType::Hybrid(HybridMode::InteractiveFirst) => {
                    if complexity >= 10.0 {
                        selected.push(strategy_type.clone());
                    }
                }
                StrategyType::Hybrid(HybridMode::Parallel) => {
                    if complexity >= 15.0 {
                        selected.push(strategy_type.clone());
                    }
                }
                StrategyType::Hybrid(HybridMode::Adaptive) => {
                    selected.push(strategy_type.clone()); // 自适应策略总是可用
                }
            }
        }
        
        selected
    }

    fn weight(&self) -> f64 {
        self.weight
    }

    fn name(&self) -> &str {
        "ComplexityBasedRule"
    }

    fn calculate_complexity(proof: &Proof) -> f64 {
        let mut complexity = 0.0;
        complexity += proof.steps().len() as f64 * 0.1;
        complexity += proof.current_propositions().len() as f64 * 0.2;
        complexity
    }
}

/// 基于性能的选择规则
struct PerformanceBasedRule {
    weight: f64,
}

impl PerformanceBasedRule {
    fn new() -> Self {
        Self { weight: 0.3 }
    }
}

impl SelectionRule for PerformanceBasedRule {
    fn apply(&self, proof: &Proof, strategies: &[StrategyType]) -> Vec<StrategyType> {
        // 这里应该基于历史性能数据选择策略
        // 目前返回所有策略
        strategies.to_vec()
    }

    fn weight(&self) -> f64 {
        self.weight
    }

    fn name(&self) -> &str {
        "PerformanceBasedRule"
    }
}

/// 基于历史的选择规则
struct HistoryBasedRule {
    weight: f64,
}

impl HistoryBasedRule {
    fn new() -> Self {
        Self { weight: 0.3 }
    }
}

impl SelectionRule for HistoryBasedRule {
    fn apply(&self, proof: &Proof, strategies: &[StrategyType]) -> Vec<StrategyType> {
        // 这里应该基于历史成功记录选择策略
        // 目前返回所有策略
        strategies.to_vec()
    }

    fn weight(&self) -> f64 {
        self.weight
    }

    fn name(&self) -> &str {
        "HistoryBasedRule"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Proof, Proposition, PropositionType, RuleLibrary};

    #[test]
    fn test_strategy_selector_creation() {
        let selector = StrategySelector::new();
        assert!(!selector.get_available_strategies().is_empty());
    }

    #[test]
    fn test_complexity_based_rule() {
        let rule = ComplexityBasedRule::new();
        assert_eq!(rule.weight(), 0.4);
        assert_eq!(rule.name(), "ComplexityBasedRule");
    }

    #[test]
    fn test_performance_based_rule() {
        let rule = PerformanceBasedRule::new();
        assert_eq!(rule.weight(), 0.3);
        assert_eq!(rule.name(), "PerformanceBasedRule");
    }

    #[test]
    fn test_history_based_rule() {
        let rule = HistoryBasedRule::new();
        assert_eq!(rule.weight(), 0.3);
        assert_eq!(rule.name(), "HistoryBasedRule");
    }
}
