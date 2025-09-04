use crate::core::{Proof, ProofStep, InferenceRule, ProofError, ProofStatus, Proposition, RuleLibrary};
use crate::strategies::{StrategyExecutionResult, StrategyPerformanceMetrics, StrategyConfig};
use crate::strategies::automated::AutomatedProofStrategy;
use crate::strategies::interactive::{InteractiveProofStrategy, UserInteraction};
use std::collections::HashMap;
use std::time::Instant;

/// 混合策略模式
#[derive(Debug, Clone, PartialEq)]
pub enum HybridMode {
    /// 自动优先：先尝试自动，失败时转为交互
    AutoFirst,
    /// 交互优先：先提供建议，用户选择后自动执行
    InteractiveFirst,
    /// 并行执行：同时运行自动和交互策略
    Parallel,
    /// 自适应：根据证明复杂度自动选择模式
    Adaptive,
}

/// 混合证明策略
pub struct HybridProofStrategy {
    config: StrategyConfig,
    automated_strategy: AutomatedProofStrategy,
    interactive_strategy: InteractiveProofStrategy,
    mode: HybridMode,
    mode_history: Vec<HybridMode>,
    performance_tracker: PerformanceTracker,
}

/// 性能跟踪器
struct PerformanceTracker {
    mode_performance: HashMap<HybridMode, Vec<u64>>,
    success_rates: HashMap<HybridMode, f64>,
    mode_switches: Vec<(HybridMode, HybridMode, String)>,
}

impl HybridProofStrategy {
    /// 创建新的混合证明策略
    pub fn new(
        rule_library: RuleLibrary,
        config: StrategyConfig,
        mode: HybridMode,
    ) -> Self {
        let automated_strategy = AutomatedProofStrategy::new(rule_library.clone(), config.clone());
        let interactive_strategy = InteractiveProofStrategy::new(rule_library, config.clone());
        
        Self {
            config,
            automated_strategy,
            interactive_strategy,
            mode,
            mode_history: vec![mode.clone()],
            performance_tracker: PerformanceTracker::new(),
        }
    }

    /// 执行混合证明策略
    pub fn execute(&mut self, proof: &mut Proof) -> StrategyExecutionResult {
        let start_time = Instant::now();
        
        match self.mode {
            HybridMode::AutoFirst => self.execute_auto_first(proof, start_time),
            HybridMode::InteractiveFirst => self.execute_interactive_first(proof, start_time),
            HybridMode::Parallel => self.execute_parallel(proof, start_time),
            HybridMode::Adaptive => self.execute_adaptive(proof, start_time),
        }
    }

    /// 自动优先模式执行
    fn execute_auto_first(
        &mut self,
        proof: &mut Proof,
        start_time: Instant,
    ) -> StrategyExecutionResult {
        // 先尝试自动证明
        let auto_result = self.automated_strategy.execute(proof);
        
        if auto_result.success {
            // 自动证明成功
            self.record_mode_performance(self.mode.clone(), start_time.elapsed().as_millis() as u64, true);
            auto_result
        } else {
            // 自动证明失败，转为交互模式
            self.switch_mode(HybridMode::InteractiveFirst, "自动证明失败，转为交互模式".to_string());
            self.execute_interactive_first(proof, start_time)
        }
    }

    /// 交互优先模式执行
    fn execute_interactive_first(
        &mut self,
        proof: &mut Proof,
        start_time: Instant,
    ) -> StrategyExecutionResult {
        // 分析当前证明状态
        let suggestions = self.analyze_proof_state(proof);
        
        // 生成交互建议
        let interaction_suggestions = self.generate_interaction_suggestions(proof, &suggestions);
        
        // 如果建议足够明确，尝试自动执行
        if self.can_auto_execute(&suggestions) {
            // 尝试自动执行建议
            let auto_result = self.execute_suggestions_automatically(proof, &suggestions);
            if auto_result.success {
                self.record_mode_performance(self.mode.clone(), start_time.elapsed().as_millis() as u64, true);
                return auto_result;
            }
        }
        
        // 返回交互建议
        self.record_mode_performance(self.mode.clone(), start_time.elapsed().as_millis() as u64, false);
        StrategyExecutionResult {
            success: false,
            generated_steps: vec![],
            applied_rules: vec![],
            execution_time_ms: start_time.elapsed().as_millis() as u64,
            error: Some("需要用户交互".to_string()),
            suggestions: interaction_suggestions,
        }
    }

    /// 并行模式执行
    fn execute_parallel(
        &mut self,
        proof: &mut Proof,
        start_time: Instant,
    ) -> StrategyExecutionResult {
        // 创建证明的副本用于并行执行
        let mut auto_proof = proof.clone();
        let mut interactive_proof = proof.clone();
        
        // 并行执行自动和交互策略
        let auto_result = self.automated_strategy.execute(&mut auto_proof);
        let interactive_suggestions = self.interactive_strategy.get_rule_suggestions(&interactive_proof);
        
        // 选择最佳结果
        if auto_result.success {
            // 自动策略成功，应用结果
            *proof = auto_proof;
            self.record_mode_performance(self.mode.clone(), start_time.elapsed().as_millis() as u64, true);
            auto_result
        } else {
            // 自动策略失败，提供交互建议
            self.record_mode_performance(self.mode.clone(), start_time.elapsed().as_millis() as u64, false);
            StrategyExecutionResult {
                success: false,
                generated_steps: vec![],
                applied_rules: vec![],
                execution_time_ms: start_time.elapsed().as_millis() as u64,
                error: Some("并行执行完成，需要用户选择".to_string()),
                suggestions: vec![
                    "自动策略未能完成证明".to_string(),
                    "可用的交互选项已准备就绪".to_string(),
                ],
            }
        }
    }

    /// 自适应模式执行
    fn execute_adaptive(
        &mut self,
        proof: &mut Proof,
        start_time: Instant,
    ) -> StrategyExecutionResult {
        // 分析证明复杂度
        let complexity = self.analyze_proof_complexity(proof);
        
        // 根据复杂度选择策略
        let selected_mode = self.select_mode_by_complexity(complexity);
        self.switch_mode(selected_mode, format!("根据复杂度{}选择策略", complexity));
        
        // 执行选定的策略
        match selected_mode {
            HybridMode::AutoFirst => self.execute_auto_first(proof, start_time),
            HybridMode::InteractiveFirst => self.execute_interactive_first(proof, start_time),
            HybridMode::Parallel => self.execute_parallel(proof, start_time),
            HybridMode::Adaptive => unreachable!(), // 避免递归
        }
    }

    /// 分析证明状态
    fn analyze_proof_state(&self, proof: &Proof) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        let current_props = proof.current_propositions();
        let steps = proof.steps();
        
        // 分析当前状态
        if current_props.len() < 2 {
            suggestions.push("需要更多前提条件".to_string());
        }
        
        if steps.len() > 50 {
            suggestions.push("证明步骤较多，考虑简化路径".to_string());
        }
        
        // 检查是否有明显的推理路径
        if let Some(path) = self.find_obvious_inference_path(proof) {
            suggestions.push(format!("发现明显推理路径: {}", path));
        }
        
        suggestions
    }

    /// 生成交互建议
    fn generate_interaction_suggestions(
        &self,
        proof: &Proof,
        analysis: &[String],
    ) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        // 基于分析结果生成建议
        for analysis_item in analysis {
            match analysis_item.as_str() {
                "需要更多前提条件" => {
                    suggestions.push("请提供更多前提条件".to_string());
                    suggestions.push("或者指定证明的目标".to_string());
                }
                "证明步骤较多，考虑简化路径" => {
                    suggestions.push("考虑使用更直接的推理规则".to_string());
                    suggestions.push("或者重新规划证明路径".to_string());
                }
                _ => {
                    if analysis_item.starts_with("发现明显推理路径") {
                        suggestions.push("可以尝试自动执行此路径".to_string());
                    }
                }
            }
        }
        
        // 添加通用建议
        suggestions.push("选择推理规则".to_string());
        suggestions.push("提供中间命题".to_string());
        suggestions.push("指定证明方向".to_string());
        
        suggestions
    }

    /// 检查是否可以自动执行
    fn can_auto_execute(&self, suggestions: &[String]) -> bool {
        // 如果建议中包含明显的推理路径，可以尝试自动执行
        suggestions.iter().any(|s| s.contains("明显推理路径"))
    }

    /// 自动执行建议
    fn execute_suggestions_automatically(
        &mut self,
        proof: &mut Proof,
        suggestions: &[String],
    ) -> StrategyExecutionResult {
        // 尝试自动执行建议
        let result = self.automated_strategy.execute(proof);
        
        if result.success {
            result
        } else {
            // 自动执行失败，返回交互建议
            StrategyExecutionResult {
                success: false,
                generated_steps: vec![],
                applied_rules: vec![],
                execution_time_ms: 0,
                error: Some("自动执行建议失败".to_string()),
                suggestions: suggestions.to_vec(),
            }
        }
    }

    /// 分析证明复杂度
    fn analyze_proof_complexity(&self, proof: &Proof) -> f64 {
        let mut complexity = 0.0;
        
        // 基于步骤数量
        complexity += proof.steps().len() as f64 * 0.1;
        
        // 基于命题数量
        complexity += proof.current_propositions().len() as f64 * 0.2;
        
        // 基于规则复杂度
        let rule_complexity: f64 = proof.steps().iter()
            .map(|step| {
                // 这里应该获取规则的实际复杂度
                1.0
            })
            .sum();
        complexity += rule_complexity * 0.3;
        
        complexity
    }

    /// 根据复杂度选择策略
    fn select_mode_by_complexity(&self, complexity: f64) -> HybridMode {
        match complexity {
            c if c < 5.0 => HybridMode::AutoFirst,
            c if c < 15.0 => HybridMode::InteractiveFirst,
            c if c < 30.0 => HybridMode::Parallel,
            _ => HybridMode::InteractiveFirst,
        }
    }

    /// 寻找明显的推理路径
    fn find_obvious_inference_path(&self, proof: &Proof) -> Option<String> {
        let current_props = proof.current_propositions();
        
        // 检查是否有直接的推理规则可用
        for rule in self.automated_strategy.rule_library.get_all_rules() {
            if rule.is_applicable(&current_props) {
                return Some(format!("使用规则: {}", rule.name()));
            }
        }
        
        None
    }

    /// 切换策略模式
    fn switch_mode(&mut self, new_mode: HybridMode, reason: String) {
        let old_mode = self.mode.clone();
        self.mode = new_mode.clone();
        self.mode_history.push(new_mode);
        
        self.performance_tracker.record_mode_switch(old_mode, new_mode, reason);
    }

    /// 记录模式性能
    fn record_mode_performance(&mut self, mode: HybridMode, execution_time: u64, success: bool) {
        self.performance_tracker.record_performance(mode, execution_time, success);
    }

    /// 获取性能统计
    pub fn get_performance_stats(&self) -> HashMap<HybridMode, (f64, u64)> {
        self.performance_tracker.get_stats()
    }

    /// 设置策略模式
    pub fn set_mode(&mut self, mode: HybridMode) {
        self.mode = mode.clone();
        self.mode_history.push(mode);
    }

    /// 获取当前模式
    pub fn get_current_mode(&self) -> &HybridMode {
        &self.mode
    }

    /// 获取模式历史
    pub fn get_mode_history(&self) -> &[HybridMode] {
        &self.mode_history
    }
}

impl PerformanceTracker {
    fn new() -> Self {
        Self {
            mode_performance: HashMap::new(),
            success_rates: HashMap::new(),
            mode_switches: Vec::new(),
        }
    }

    fn record_performance(&mut self, mode: HybridMode, execution_time: u64, success: bool) {
        // 记录执行时间
        let times = self.mode_performance.entry(mode.clone()).or_insert_with(Vec::new);
        times.push(execution_time);
        
        // 更新成功率
        let current_rate = self.success_rates.get(&mode).unwrap_or(&0.0);
        let new_rate = if success {
            (*current_rate + 1.0) / 2.0
        } else {
            *current_rate / 2.0
        };
        self.success_rates.insert(mode, new_rate);
    }

    fn record_mode_switch(&mut self, from: HybridMode, to: HybridMode, reason: String) {
        self.mode_switches.push((from, to, reason));
    }

    fn get_stats(&self) -> HashMap<HybridMode, (f64, u64)> {
        let mut stats = HashMap::new();
        
        for (mode, times) in &self.mode_performance {
            let avg_time = if times.is_empty() {
                0
            } else {
                times.iter().sum::<u64>() / times.len() as u64
            };
            let success_rate = self.success_rates.get(mode).unwrap_or(&0.0);
            stats.insert(mode.clone(), (*success_rate, avg_time));
        }
        
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Proof, Proposition, PropositionType, RuleLibrary};

    #[test]
    fn test_hybrid_strategy_creation() {
        let rule_library = RuleLibrary::new();
        let config = StrategyConfig::default();
        let strategy = HybridProofStrategy::new(rule_library, config, HybridMode::AutoFirst);
        
        assert_eq!(*strategy.get_current_mode(), HybridMode::AutoFirst);
        assert_eq!(strategy.get_mode_history().len(), 1);
    }

    #[test]
    fn test_mode_switching() {
        let rule_library = RuleLibrary::new();
        let config = StrategyConfig::default();
        let mut strategy = HybridProofStrategy::new(rule_library, config, HybridMode::AutoFirst);
        
        strategy.set_mode(HybridMode::InteractiveFirst);
        assert_eq!(*strategy.get_current_mode(), HybridMode::InteractiveFirst);
        assert_eq!(strategy.get_mode_history().len(), 2);
    }

    #[test]
    fn test_complexity_analysis() {
        let rule_library = RuleLibrary::new();
        let config = StrategyConfig::default();
        let strategy = HybridProofStrategy::new(rule_library, config, HybridMode::AutoFirst);
        
        let mut proof = Proof::new();
        proof.add_premise(Proposition::new("A", PropositionType::Axiom));
        proof.add_premise(Proposition::new("B", PropositionType::Axiom));
        
        let complexity = strategy.analyze_proof_complexity(&proof);
        assert!(complexity > 0.0);
    }

    #[test]
    fn test_performance_tracker() {
        let tracker = PerformanceTracker::new();
        let stats = tracker.get_stats();
        assert!(stats.is_empty());
    }
}
