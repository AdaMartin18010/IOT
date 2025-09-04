use crate::core::{Proof, ProofStep, InferenceRule, ProofError, ProofStatus, Proposition, RuleLibrary};
use crate::strategies::{StrategyExecutionResult, StrategyPerformanceMetrics, StrategyConfig};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// 用户交互类型
#[derive(Debug, Clone)]
pub enum UserInteraction {
    /// 选择推理规则
    RuleSelection(InferenceRule),
    /// 提供中间命题
    IntermediateProposition(Proposition),
    /// 指定证明方向
    ProofDirection(String),
    /// 请求提示
    RequestHint,
    /// 撤销上一步
    UndoLastStep,
    /// 重新开始
    Restart,
}

/// 交互式证明策略
pub struct InteractiveProofStrategy {
    config: StrategyConfig,
    rule_library: RuleLibrary,
    interaction_history: VecDeque<UserInteraction>,
    suggestion_engine: SuggestionEngine,
    performance_metrics: StrategyPerformanceMetrics,
}

/// 建议引擎
struct SuggestionEngine {
    rule_suggestions: HashMap<String, Vec<InferenceRule>>,
    direction_suggestions: HashMap<String, Vec<String>>,
    learning_data: HashMap<String, f64>,
}

impl InteractiveProofStrategy {
    /// 创建新的交互式证明策略
    pub fn new(rule_library: RuleLibrary, config: StrategyConfig) -> Self {
        Self {
            config,
            rule_library,
            interaction_history: VecDeque::new(),
            suggestion_engine: SuggestionEngine::new(),
            performance_metrics: StrategyPerformanceMetrics {
                success_rate: 0.0,
                avg_execution_time_ms: 0,
                avg_steps_generated: 0.0,
                rule_application_efficiency: 0.0,
                memory_usage_mb: 0,
            },
        }
    }

    /// 处理用户交互
    pub fn handle_interaction(&mut self, proof: &mut Proof, interaction: UserInteraction) -> StrategyExecutionResult {
        let start_time = Instant::now();
        let mut generated_steps = Vec::new();
        let mut applied_rules = Vec::new();
        let mut error = None;

        // 记录交互历史
        self.interaction_history.push_back(interaction.clone());

        // 处理交互
        match self.process_interaction(proof, &interaction, &mut generated_steps, &mut applied_rules) {
            Ok(_) => {
                // 更新建议引擎
                self.suggestion_engine.learn_from_interaction(&interaction, proof);
                
                // 更新性能指标
                self.update_performance_metrics(start_time.elapsed().as_millis() as u64, generated_steps.len());
                
                StrategyExecutionResult {
                    success: true,
                    generated_steps,
                    applied_rules,
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    error: None,
                    suggestions: self.generate_interactive_suggestions(proof, &interaction),
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
                    suggestions: vec!["检查输入的有效性".to_string(), "尝试不同的证明方向".to_string()],
                }
            }
        }
    }

    /// 处理交互逻辑
    fn process_interaction(
        &self,
        proof: &mut Proof,
        interaction: &UserInteraction,
        generated_steps: &mut Vec<ProofStep>,
        applied_rules: &mut Vec<InferenceRule>,
    ) -> Result<(), ProofError> {
        match interaction {
            UserInteraction::RuleSelection(rule) => {
                self.apply_selected_rule(proof, rule, generated_steps, applied_rules)
            }
            UserInteraction::IntermediateProposition(prop) => {
                self.add_intermediate_proposition(proof, prop, generated_steps)
            }
            UserInteraction::ProofDirection(direction) => {
                self.set_proof_direction(proof, direction)
            }
            UserInteraction::RequestHint => {
                Ok(()) // 提示在suggestions中提供
            }
            UserInteraction::UndoLastStep => {
                self.undo_last_step(proof, generated_steps, applied_rules)
            }
            UserInteraction::Restart => {
                self.restart_proof(proof, generated_steps, applied_rules)
            }
        }
    }

    /// 应用选定的规则
    fn apply_selected_rule(
        &self,
        proof: &mut Proof,
        rule: &InferenceRule,
        generated_steps: &mut Vec<ProofStep>,
        applied_rules: &mut Vec<InferenceRule>,
    ) -> Result<(), ProofError> {
        let current_propositions = proof.current_propositions();
        
        if rule.is_applicable(&current_propositions) {
            let outputs = rule.apply(&current_propositions);
            if !outputs.is_empty() {
                let step = ProofStep::new()
                    .with_rule(rule.id())
                    .with_inputs(current_propositions.to_vec())
                    .with_outputs(outputs)
                    .with_justification(format!("用户选择应用规则: {}", rule.name()));
                
                proof.add_step(step.clone());
                generated_steps.push(step);
                applied_rules.push(rule.clone());
                Ok(())
            } else {
                Err(ProofError::InvalidRuleApplication("规则应用后未产生新命题".to_string()))
            }
        } else {
            Err(ProofError::InvalidRuleApplication("选定的规则不适用于当前状态".to_string()))
        }
    }

    /// 添加中间命题
    fn add_intermediate_proposition(
        &self,
        proof: &mut Proof,
        prop: &Proposition,
        generated_steps: &mut Vec<ProofStep>,
    ) -> Result<(), ProofError> {
        // 检查命题是否已经存在
        if proof.current_propositions().iter().any(|p| p == prop) {
            return Err(ProofError::InvalidProposition("命题已存在".to_string()));
        }

        // 创建中间步骤
        let step = ProofStep::new()
            .with_outputs(vec![prop.clone()])
            .with_justification("用户提供的中间命题".to_string());
        
        proof.add_step(step.clone());
        generated_steps.push(step);
        Ok(())
    }

    /// 设置证明方向
    fn set_proof_direction(&self, proof: &mut Proof, direction: &str) -> Result<(), ProofError> {
        // 这里可以设置证明的元数据或状态
        // 目前只是记录方向
        Ok(())
    }

    /// 撤销上一步
    fn undo_last_step(
        &self,
        proof: &mut Proof,
        generated_steps: &mut Vec<ProofStep>,
        applied_rules: &mut Vec<InferenceRule>,
    ) -> Result<(), ProofError> {
        if let Some(step) = proof.remove_last_step() {
            // 从生成列表中移除
            if let Some(pos) = generated_steps.iter().position(|s| s.id() == step.id()) {
                generated_steps.remove(pos);
            }
            
            // 从应用规则列表中移除
            if let Some(pos) = applied_rules.iter().position(|r| r.id() == step.rule_id()) {
                applied_rules.remove(pos);
            }
            Ok(())
        } else {
            Err(ProofError::InvalidOperation("没有步骤可以撤销".to_string()))
        }
    }

    /// 重新开始证明
    fn restart_proof(
        &self,
        proof: &mut Proof,
        generated_steps: &mut Vec<ProofStep>,
        applied_rules: &mut Vec<InferenceRule>,
    ) -> Result<(), ProofError> {
        // 清除所有步骤，保留前提
        proof.clear_steps();
        generated_steps.clear();
        applied_rules.clear();
        proof.set_status(ProofStatus::InProgress);
        Ok(())
    }

    /// 生成交互式建议
    fn generate_interactive_suggestions(&self, proof: &Proof, interaction: &UserInteraction) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        match interaction {
            UserInteraction::RuleSelection(_) => {
                suggestions.push("规则应用成功，检查是否达到目标".to_string());
                suggestions.push("考虑下一步的证明方向".to_string());
            }
            UserInteraction::IntermediateProposition(_) => {
                suggestions.push("中间命题已添加，寻找连接路径".to_string());
                suggestions.push("考虑使用哪些规则连接现有命题".to_string());
            }
            UserInteraction::ProofDirection(_) => {
                suggestions.push("方向已设置，继续按此方向推进".to_string());
            }
            UserInteraction::RequestHint => {
                suggestions.extend(self.suggestion_engine.generate_hints(proof));
            }
            UserInteraction::UndoLastStep => {
                suggestions.push("步骤已撤销，重新考虑证明策略".to_string());
            }
            UserInteraction::Restart => {
                suggestions.push("证明已重新开始，重新规划证明路径".to_string());
            }
        }
        
        suggestions
    }

    /// 获取可用的规则建议
    pub fn get_rule_suggestions(&self, proof: &Proof) -> Vec<InferenceRule> {
        self.suggestion_engine.get_rule_suggestions(proof)
    }

    /// 获取证明方向建议
    pub fn get_direction_suggestions(&self, proof: &Proof) -> Vec<String> {
        self.suggestion_engine.get_direction_suggestions(proof)
    }

    /// 更新性能指标
    fn update_performance_metrics(&mut self, execution_time: u64, steps_generated: usize) {
        let current_avg = self.performance_metrics.avg_execution_time_ms;
        let current_steps = self.performance_metrics.avg_steps_generated;
        
        self.performance_metrics.avg_execution_time_ms = 
            (current_avg + execution_time) / 2;
        self.performance_metrics.avg_steps_generated = 
            (current_steps + steps_generated as f64) / 2.0;
    }

    /// 获取性能指标
    pub fn get_performance_metrics(&self) -> &StrategyPerformanceMetrics {
        &self.performance_metrics
    }

    /// 获取交互历史
    pub fn get_interaction_history(&self) -> &VecDeque<UserInteraction> {
        &self.interaction_history
    }
}

impl SuggestionEngine {
    fn new() -> Self {
        Self {
            rule_suggestions: HashMap::new(),
            direction_suggestions: HashMap::new(),
            learning_data: HashMap::new(),
        }
    }

    /// 从交互中学习
    fn learn_from_interaction(&mut self, interaction: &UserInteraction, proof: &Proof) {
        match interaction {
            UserInteraction::RuleSelection(rule) => {
                let key = format!("{:?}", proof.current_propositions());
                let score = self.learning_data.get(&key).unwrap_or(&0.0) + 1.0;
                self.learning_data.insert(key, score);
            }
            _ => {}
        }
    }

    /// 生成提示
    fn generate_hints(&self, proof: &Proof) -> Vec<String> {
        let mut hints = Vec::new();
        
        // 基于当前状态生成提示
        let current_props = proof.current_propositions();
        if current_props.len() < 2 {
            hints.push("需要更多前提条件来开始证明".to_string());
        } else {
            hints.push("考虑使用推理规则连接现有命题".to_string());
            hints.push("检查是否有直接可用的规则".to_string());
        }
        
        hints
    }

    /// 获取规则建议
    fn get_rule_suggestions(&self, proof: &Proof) -> Vec<InferenceRule> {
        // 这里应该实现基于当前状态的规则推荐算法
        // 目前返回空列表
        Vec::new()
    }

    /// 获取方向建议
    fn get_direction_suggestions(&self, proof: &Proof) -> Vec<String> {
        vec![
            "向前推理：从前提推导结论".to_string(),
            "向后推理：从目标反推前提".to_string(),
            "中间连接：寻找连接现有命题的路径".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Proof, Proposition, PropositionType, InferenceRule, RuleLibrary};

    #[test]
    fn test_interactive_strategy_creation() {
        let rule_library = RuleLibrary::new();
        let config = StrategyConfig::default();
        let strategy = InteractiveProofStrategy::new(rule_library, config);
        
        assert_eq!(strategy.interaction_history.len(), 0);
    }

    #[test]
    fn test_interaction_history_recording() {
        let rule_library = RuleLibrary::new();
        let config = StrategyConfig::default();
        let mut strategy = InteractiveProofStrategy::new(rule_library, config);
        
        let mut proof = Proof::new();
        let interaction = UserInteraction::RequestHint;
        
        let _result = strategy.handle_interaction(&mut proof, interaction);
        assert_eq!(strategy.interaction_history.len(), 1);
    }

    #[test]
    fn test_suggestion_engine_creation() {
        let engine = SuggestionEngine::new();
        assert!(engine.rule_suggestions.is_empty());
        assert!(engine.direction_suggestions.is_empty());
    }
}
