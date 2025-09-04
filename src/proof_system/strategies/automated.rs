use crate::core::{Proof, ProofStep, InferenceRule, ProofError, ProofStatus, Proposition, RuleLibrary};
use crate::strategies::{StrategyExecutionResult, StrategyPerformanceMetrics, StrategyConfig};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

/// 自动证明策略
pub struct AutomatedProofStrategy {
    config: StrategyConfig,
    rule_library: RuleLibrary,
    cache: HashMap<String, Vec<ProofStep>>,
    performance_metrics: StrategyPerformanceMetrics,
}

impl AutomatedProofStrategy {
    /// 创建新的自动证明策略
    pub fn new(rule_library: RuleLibrary, config: StrategyConfig) -> Self {
        Self {
            config,
            rule_library,
            cache: HashMap::new(),
            performance_metrics: StrategyPerformanceMetrics {
                success_rate: 0.0,
                avg_execution_time_ms: 0,
                avg_steps_generated: 0.0,
                rule_application_efficiency: 0.0,
                memory_usage_mb: 0,
            },
        }
    }

    /// 执行自动证明
    pub fn execute(&mut self, proof: &mut Proof) -> StrategyExecutionResult {
        let start_time = Instant::now();
        let mut generated_steps = Vec::new();
        let mut applied_rules = Vec::new();
        let mut error = None;

        // 检查缓存
        let cache_key = self.generate_cache_key(proof);
        if self.config.enable_caching {
            if let Some(cached_steps) = self.cache.get(&cache_key) {
                return StrategyExecutionResult {
                    success: true,
                    generated_steps: cached_steps.clone(),
                    applied_rules: vec![],
                    execution_time_ms: 0,
                    error: None,
                    suggestions: vec!["使用缓存结果".to_string()],
                };
            }
        }

        // 执行自动证明
        match self.execute_automated_proof(proof, &mut generated_steps, &mut applied_rules) {
            Ok(_) => {
                // 更新缓存
                if self.config.enable_caching {
                    self.cache.insert(cache_key, generated_steps.clone());
                }
                
                // 更新性能指标
                self.update_performance_metrics(start_time.elapsed().as_millis() as u64, generated_steps.len());
                
                StrategyExecutionResult {
                    success: true,
                    generated_steps,
                    applied_rules,
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    error: None,
                    suggestions: self.generate_suggestions(proof),
                }
            }
            Err(e) => {
                error = Some(e.to_string());
                StrategyExecutionResult {
                    success: false,
                    generated_steps,
                    applied_rules,
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    error,
                    suggestions: vec!["检查证明前提的完整性".to_string(), "验证推理规则的正确性".to_string()],
                }
            }
        }
    }

    /// 执行自动证明逻辑
    fn execute_automated_proof(
        &self,
        proof: &mut Proof,
        generated_steps: &mut Vec<ProofStep>,
        applied_rules: &mut Vec<InferenceRule>,
    ) -> Result<(), ProofError> {
        let mut step_count = 0;
        let max_steps = self.config.max_steps;
        
        while proof.status() != ProofStatus::Completed && step_count < max_steps {
            // 寻找可应用的规则
            if let Some((rule, step)) = self.find_applicable_rule(proof) {
                // 应用规则
                proof.add_step(step.clone());
                generated_steps.push(step);
                applied_rules.push(rule);
                step_count += 1;
            } else {
                // 无法找到可应用的规则，尝试启发式搜索
                if let Some((rule, step)) = self.heuristic_rule_search(proof) {
                    proof.add_step(step.clone());
                    generated_steps.push(step);
                    applied_rules.push(rule);
                    step_count += 1;
                } else {
                    // 无法继续，设置状态为暂停
                    proof.set_status(ProofStatus::Paused);
                    break;
                }
            }
        }

        if proof.status() == ProofStatus::Completed {
            Ok(())
        } else {
            Err(ProofError::IncompleteProof("自动证明未能在最大步骤数内完成".to_string()))
        }
    }

    /// 寻找可应用的规则
    fn find_applicable_rule(&self, proof: &Proof) -> Option<(InferenceRule, ProofStep)> {
        let current_propositions = proof.current_propositions();
        
        for rule in self.rule_library.get_all_rules() {
            if let Some(step) = self.try_apply_rule(rule.clone(), &current_propositions) {
                return Some((rule, step));
            }
        }
        None
    }

    /// 尝试应用规则
    fn try_apply_rule(&self, rule: InferenceRule, propositions: &[Proposition]) -> Option<ProofStep> {
        if rule.is_applicable(propositions) {
            let outputs = rule.apply(propositions);
            if !outputs.is_empty() {
                return Some(ProofStep::new()
                    .with_rule(rule.id())
                    .with_inputs(propositions.to_vec())
                    .with_outputs(outputs)
                    .with_justification(format!("应用规则: {}", rule.name())));
            }
        }
        None
    }

    /// 启发式规则搜索
    fn heuristic_rule_search(&self, proof: &Proof) -> Option<(InferenceRule, ProofStep)> {
        let current_propositions = proof.current_propositions();
        let mut best_rule = None;
        let mut best_score = 0.0;

        for rule in self.rule_library.get_all_rules() {
            let score = self.calculate_rule_score(&rule, &current_propositions);
            if score > best_score {
                best_score = score;
                best_rule = Some(rule);
            }
        }

        best_rule.and_then(|rule| {
            self.try_apply_rule(rule.clone(), &current_propositions)
                .map(|step| (rule, step))
        })
    }

    /// 计算规则分数
    fn calculate_rule_score(&self, rule: &InferenceRule, propositions: &[Proposition]) -> f64 {
        let mut score = 0.0;
        
        // 基础分数
        score += 1.0;
        
        // 匹配度分数
        let match_count = rule.patterns().iter()
            .filter(|pattern| self.matches_pattern(pattern, propositions))
            .count();
        score += match_count as f64 * 0.5;
        
        // 复杂度分数（偏好简单规则）
        score += 1.0 / (rule.complexity() as f64 + 1.0);
        
        score
    }

    /// 检查是否匹配模式
    fn matches_pattern(&self, pattern: &crate::core::RulePattern, propositions: &[Proposition]) -> bool {
        match pattern {
            crate::core::RulePattern::ExactMatch(prop) => {
                propositions.iter().any(|p| p == prop)
            }
            crate::core::RulePattern::TypeMatch(prop_type) => {
                propositions.iter().any(|p| p.proposition_type() == prop_type)
            }
            crate::core::RulePattern::AnyMatch => true,
        }
    }

    /// 生成缓存键
    fn generate_cache_key(&self, proof: &Proof) -> String {
        let premises_hash = format!("{:?}", proof.premises());
        let current_steps_hash = format!("{:?}", proof.steps());
        format!("{}:{}", premises_hash, current_steps_hash)
    }

    /// 生成策略建议
    fn generate_suggestions(&self, proof: &Proof) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        if proof.status() == ProofStatus::Completed {
            suggestions.push("证明已完成，建议进行验证检查".to_string());
        } else if proof.status() == ProofStatus::Paused {
            suggestions.push("证明已暂停，建议检查前提条件".to_string());
            suggestions.push("考虑使用交互式策略继续证明".to_string());
        }
        
        if proof.steps().len() > 100 {
            suggestions.push("证明步骤较多，建议优化策略".to_string());
        }
        
        suggestions
    }

    /// 更新性能指标
    fn update_performance_metrics(&mut self, execution_time: u64, steps_generated: usize) {
        let current_avg = self.performance_metrics.avg_execution_time_ms;
        let current_steps = self.performance_metrics.avg_steps_generated;
        
        // 简单的移动平均更新
        self.performance_metrics.avg_execution_time_ms = 
            (current_avg + execution_time) / 2;
        self.performance_metrics.avg_steps_generated = 
            (current_steps + steps_generated as f64) / 2.0;
    }

    /// 获取性能指标
    pub fn get_performance_metrics(&self) -> &StrategyPerformanceMetrics {
        &self.performance_metrics
    }

    /// 清除缓存
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// 设置配置
    pub fn set_config(&mut self, config: StrategyConfig) {
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Proof, Proposition, PropositionType, InferenceRule, RuleLibrary};

    #[test]
    fn test_automated_strategy_creation() {
        let rule_library = RuleLibrary::new();
        let config = StrategyConfig::default();
        let strategy = AutomatedProofStrategy::new(rule_library, config);
        
        assert_eq!(strategy.config.max_steps, 1000);
        assert!(strategy.config.enable_parallel);
    }

    #[test]
    fn test_cache_key_generation() {
        let rule_library = RuleLibrary::new();
        let config = StrategyConfig::default();
        let mut strategy = AutomatedProofStrategy::new(rule_library, config);
        
        let mut proof = Proof::new();
        proof.add_premise(Proposition::new("A", PropositionType::Axiom));
        
        let cache_key = strategy.generate_cache_key(&proof);
        assert!(!cache_key.is_empty());
    }

    #[test]
    fn test_performance_metrics_update() {
        let rule_library = RuleLibrary::new();
        let config = StrategyConfig::default();
        let mut strategy = AutomatedProofStrategy::new(rule_library, config);
        
        strategy.update_performance_metrics(100, 5);
        let metrics = strategy.get_performance_metrics();
        
        assert_eq!(metrics.avg_execution_time_ms, 100);
        assert_eq!(metrics.avg_steps_generated, 5.0);
    }
}
