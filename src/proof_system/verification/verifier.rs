use crate::core::{Proof, ProofStep, InferenceRule, ProofError, ProofStatus, Proposition, RuleLibrary};
use crate::verification::{
    VerificationResult, VerificationError, VerificationWarning, VerificationLevel,
    VerificationConfig, VerificationStats, VerificationErrorType, VerificationWarningType,
    ErrorLocation, ErrorSeverity
};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// 证明验证器
pub struct ProofVerifier {
    config: VerificationConfig,
    rule_library: RuleLibrary,
    stats: VerificationStats,
    validation_rules: Vec<Box<dyn ValidationRule>>,
}

/// 验证规则特征
pub trait ValidationRule {
    /// 应用验证规则
    fn validate(&self, proof: &Proof, rule_library: &RuleLibrary) -> Vec<VerificationError>;
    
    /// 获取规则名称
    fn name(&self) -> &str;
    
    /// 获取规则描述
    fn description(&self) -> &str;
    
    /// 获取规则适用的验证级别
    fn applicable_levels(&self) -> Vec<VerificationLevel>;
}

impl ProofVerifier {
    /// 创建新的证明验证器
    pub fn new(rule_library: RuleLibrary, config: VerificationConfig) -> Self {
        let mut verifier = Self {
            config,
            rule_library,
            stats: VerificationStats::default(),
            validation_rules: Vec::new(),
        };
        
        // 添加默认验证规则
        verifier.add_validation_rule(Box::new(StructuralValidationRule::new()));
        verifier.add_validation_rule(Box::new(LogicalValidationRule::new()));
        verifier.add_validation_rule(Box::new(CompletenessValidationRule::new()));
        verifier.add_validation_rule(Box::new(ConsistencyValidationRule::new()));
        
        verifier
    }

    /// 添加验证规则
    pub fn add_validation_rule(&mut self, rule: Box<dyn ValidationRule>) {
        self.validation_rules.push(rule);
    }

    /// 验证证明
    pub fn verify(&mut self, proof: &Proof) -> VerificationResult {
        let start_time = Instant::now();
        
        // 检查验证时间限制
        if start_time.elapsed().as_millis() as u64 > self.config.max_verification_time_ms {
            return VerificationResult {
                passed: false,
                level: self.config.level.clone(),
                verification_time_ms: start_time.elapsed().as_millis() as u64,
                errors: vec![VerificationError {
                    error_type: VerificationErrorType::StructuralError,
                    location: ErrorLocation::default(),
                    description: "验证超时".to_string(),
                    severity: ErrorSeverity::Critical,
                    fix_suggestion: Some("增加验证时间限制或简化证明".to_string()),
                }],
                warnings: vec![],
                suggestions: vec![],
                detailed_report: None,
            };
        }
        
        let mut all_errors = Vec::new();
        let mut all_warnings = Vec::new();
        
        // 应用所有适用的验证规则
        for rule in &self.validation_rules {
            if self.is_rule_applicable(rule) {
                let errors = rule.validate(proof, &self.rule_library);
                all_errors.extend(errors);
            }
        }
        
        // 生成警告
        all_warnings.extend(self.generate_warnings(proof));
        
        // 生成建议
        let suggestions = self.generate_suggestions(proof, &all_errors, &all_warnings);
        
        // 生成详细报告
        let detailed_report = if self.config.enable_detailed_report {
            Some(self.generate_detailed_report(proof, &all_errors, &all_warnings, &suggestions))
        } else {
            None
        };
        
        // 判断是否通过验证
        let passed = all_errors.iter().all(|e| e.severity < ErrorSeverity::Error);
        
        // 更新统计信息
        self.update_stats(passed, start_time.elapsed().as_millis() as u64, &all_errors, &all_warnings);
        
        VerificationResult {
            passed,
            level: self.config.level.clone(),
            verification_time_ms: start_time.elapsed().as_millis() as u64,
            errors: all_errors,
            warnings: all_warnings,
            suggestions,
            detailed_report,
        }
    }

    /// 检查规则是否适用
    fn is_rule_applicable(&self, rule: &Box<dyn ValidationRule>) -> bool {
        rule.applicable_levels().contains(&self.config.level)
    }

    /// 生成警告
    fn generate_warnings(&self, proof: &Proof) -> Vec<VerificationWarning> {
        let mut warnings = Vec::new();
        
        // 检查证明长度
        if proof.steps().len() > 100 {
            warnings.push(VerificationWarning {
                warning_type: VerificationWarningType::PerformanceWarning,
                location: ErrorLocation::default(),
                description: "证明步骤过多，可能影响性能".to_string(),
                suggestion: "考虑简化证明或使用更高效的策略".to_string(),
            });
        }
        
        // 检查规则使用频率
        let rule_usage = self.analyze_rule_usage(proof);
        for (rule_id, count) in rule_usage {
            if count > 10 {
                warnings.push(VerificationWarning {
                    warning_type: VerificationWarningType::BestPracticeWarning,
                    location: ErrorLocation {
                        rule_id: Some(rule_id),
                        ..Default::default()
                    },
                    description: format!("规则使用频率过高: {}次", count),
                    suggestion: "考虑使用更直接的推理路径".to_string(),
                });
            }
        }
        
        warnings
    }

    /// 分析规则使用频率
    fn analyze_rule_usage(&self, proof: &Proof) -> HashMap<u64, usize> {
        let mut usage = HashMap::new();
        
        for step in proof.steps() {
            let count = usage.entry(step.rule_id()).or_insert(0);
            *count += 1;
        }
        
        usage
    }

    /// 生成建议
    fn generate_suggestions(
        &self,
        proof: &Proof,
        errors: &[VerificationError],
        warnings: &[VerificationWarning],
    ) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        // 基于错误生成建议
        for error in errors {
            if let Some(suggestion) = &error.fix_suggestion {
                suggestions.push(suggestion.clone());
            }
        }
        
        // 基于警告生成建议
        for warning in warnings {
            suggestions.push(warning.suggestion.clone());
        }
        
        // 基于证明状态生成建议
        if proof.status() == ProofStatus::Paused {
            suggestions.push("证明已暂停，检查前提条件和推理路径".to_string());
        }
        
        if proof.steps().is_empty() {
            suggestions.push("证明尚未开始，添加前提条件或选择推理规则".to_string());
        }
        
        suggestions
    }

    /// 生成详细报告
    fn generate_detailed_report(
        &self,
        proof: &Proof,
        errors: &[VerificationError],
        warnings: &[VerificationWarning],
        suggestions: &[String],
    ) -> String {
        let mut report = String::new();
        
        report.push_str("=== 证明验证详细报告 ===\n\n");
        
        // 证明基本信息
        report.push_str(&format!("证明ID: {}\n", proof.id()));
        report.push_str(&format!("证明状态: {:?}\n", proof.status()));
        report.push_str(&format!("前提数量: {}\n", proof.premises().len()));
        report.push_str(&format!("步骤数量: {}\n", proof.steps().len()));
        report.push_str(&format!("验证级别: {:?}\n\n", self.config.level));
        
        // 错误报告
        if !errors.is_empty() {
            report.push_str("=== 验证错误 ===\n");
            for (i, error) in errors.iter().enumerate() {
                report.push_str(&format!("{}. {:?}: {}\n", i + 1, error.error_type, error.description));
                if let Some(suggestion) = &error.fix_suggestion {
                    report.push_str(&format!("   建议: {}\n", suggestion));
                }
            }
            report.push_str("\n");
        }
        
        // 警告报告
        if !warnings.is_empty() {
            report.push_str("=== 验证警告 ===\n");
            for (i, warning) in warnings.iter().enumerate() {
                report.push_str(&format!("{}. {:?}: {}\n", i + 1, warning.warning_type, warning.description));
                report.push_str(&format!("   建议: {}\n", warning.suggestion));
            }
            report.push_str("\n");
        }
        
        // 建议报告
        if !suggestions.is_empty() {
            report.push_str("=== 改进建议 ===\n");
            for (i, suggestion) in suggestions.iter().enumerate() {
                report.push_str(&format!("{}. {}\n", i + 1, suggestion));
            }
            report.push_str("\n");
        }
        
        report
    }

    /// 更新统计信息
    fn update_stats(
        &mut self,
        passed: bool,
        verification_time: u64,
        errors: &[VerificationError],
        warnings: &[VerificationWarning],
    ) {
        self.stats.total_verifications += 1;
        
        if passed {
            self.stats.successful_verifications += 1;
        } else {
            self.stats.failed_verifications += 1;
        }
        
        // 更新平均验证时间
        let total_time = self.stats.avg_verification_time_ms * (self.stats.total_verifications - 1) + verification_time;
        self.stats.avg_verification_time_ms = total_time / self.stats.total_verifications;
        
        // 更新错误和警告统计
        self.stats.total_errors += errors.len() as u64;
        self.stats.total_warnings += warnings.len() as u64;
        
        // 更新最常见错误类型
        if !errors.is_empty() {
            let mut error_counts: HashMap<VerificationErrorType, usize> = HashMap::new();
            for error in errors {
                let count = error_counts.entry(error.error_type.clone()).or_insert(0);
                *count += 1;
            }
            
            if let Some((most_common_type, _)) = error_counts.iter().max_by_key(|(_, &count)| count) {
                self.stats.most_common_error_type = Some(most_common_type.clone());
            }
        }
    }

    /// 获取验证统计
    pub fn get_stats(&self) -> &VerificationStats {
        &self.stats
    }

    /// 重置统计信息
    pub fn reset_stats(&mut self) {
        self.stats = VerificationStats::default();
    }

    /// 设置验证配置
    pub fn set_config(&mut self, config: VerificationConfig) {
        self.config = config;
    }

    /// 获取验证配置
    pub fn get_config(&self) -> &VerificationConfig {
        &self.config
    }
}

/// 结构验证规则
struct StructuralValidationRule;

impl StructuralValidationRule {
    fn new() -> Self {
        Self
    }
}

impl ValidationRule for StructuralValidationRule {
    fn validate(&self, proof: &Proof, _rule_library: &RuleLibrary) -> Vec<VerificationError> {
        let mut errors = Vec::new();
        
        // 检查前提条件
        if proof.premises().is_empty() {
            errors.push(VerificationError {
                error_type: VerificationErrorType::StructuralError,
                location: ErrorLocation::default(),
                description: "证明缺少前提条件".to_string(),
                severity: ErrorSeverity::Error,
                fix_suggestion: Some("添加至少一个前提条件".to_string()),
            });
        }
        
        // 检查步骤依赖关系
        for (i, step) in proof.steps().iter().enumerate() {
            for dep_id in step.dependencies() {
                if *dep_id as usize >= i {
                    errors.push(VerificationError {
                        error_type: VerificationErrorType::StructuralError,
                        location: ErrorLocation {
                            step_id: Some(step.id()),
                            ..Default::default()
                        },
                        description: format!("步骤{}依赖了未来的步骤{}", i, dep_id),
                        severity: ErrorSeverity::Error,
                        fix_suggestion: Some("调整步骤顺序或修正依赖关系".to_string()),
                    });
                }
            }
        }
        
        errors
    }

    fn name(&self) -> &str {
        "StructuralValidationRule"
    }

    fn description(&self) -> &str {
        "验证证明的结构完整性"
    }

    fn applicable_levels(&self) -> Vec<VerificationLevel> {
        vec![VerificationLevel::Basic, VerificationLevel::Standard, VerificationLevel::Strict, VerificationLevel::Expert]
    }
}

/// 逻辑验证规则
struct LogicalValidationRule;

impl LogicalValidationRule {
    fn new() -> Self {
        Self
    }
}

impl ValidationRule for LogicalValidationRule {
    fn validate(&self, proof: &Proof, rule_library: &RuleLibrary) -> Vec<VerificationError> {
        let mut errors = Vec::new();
        
        // 检查每个步骤的逻辑正确性
        for step in proof.steps() {
            if let Some(rule) = rule_library.get_rule(step.rule_id()) {
                // 检查规则是否适用于输入
                if !rule.is_applicable(&step.inputs()) {
                    errors.push(VerificationError {
                        error_type: VerificationErrorType::RuleApplicationError,
                        location: ErrorLocation {
                            step_id: Some(step.id()),
                            rule_id: Some(step.rule_id()),
                            ..Default::default()
                        },
                        description: format!("规则{}不适用于给定的输入", rule.name()),
                        severity: ErrorSeverity::Error,
                        fix_suggestion: Some("检查输入条件或选择正确的规则".to_string()),
                    });
                }
                
                // 检查输出是否与规则一致
                let expected_outputs = rule.apply(&step.inputs());
                if step.outputs() != &expected_outputs {
                    errors.push(VerificationError {
                        error_type: VerificationErrorType::LogicalError,
                        location: ErrorLocation {
                            step_id: Some(step.id()),
                            rule_id: Some(step.rule_id()),
                            ..Default::default()
                        },
                        description: "步骤输出与规则预期不符".to_string(),
                        severity: ErrorSeverity::Error,
                        fix_suggestion: Some("检查规则应用或修正输出".to_string()),
                    });
                }
            } else {
                errors.push(VerificationError {
                    error_type: VerificationErrorType::RuleApplicationError,
                    location: ErrorLocation {
                        step_id: Some(step.id()),
                        rule_id: Some(step.rule_id()),
                        ..Default::default()
                    },
                    description: format!("未找到规则ID: {}", step.rule_id()),
                    severity: ErrorSeverity::Error,
                    fix_suggestion: Some("检查规则库或修正规则ID".to_string()),
                });
            }
        }
        
        errors
    }

    fn name(&self) -> &str {
        "LogicalValidationRule"
    }

    fn description(&self) -> &str {
        "验证证明的逻辑正确性"
    }

    fn applicable_levels(&self) -> Vec<VerificationLevel> {
        vec![VerificationLevel::Standard, VerificationLevel::Strict, VerificationLevel::Expert]
    }
}

/// 完整性验证规则
struct CompletenessValidationRule;

impl CompletenessValidationRule {
    fn new() -> Self {
        Self
    }
}

impl ValidationRule for CompletenessValidationRule {
    fn validate(&self, proof: &Proof, _rule_library: &RuleLibrary) -> Vec<VerificationError> {
        let mut errors = Vec::new();
        
        // 检查证明是否完成
        if proof.status() != ProofStatus::Completed {
            errors.push(VerificationError {
                error_type: VerificationErrorType::CompletenessError,
                location: ErrorLocation::default(),
                description: "证明尚未完成".to_string(),
                severity: ErrorSeverity::Warning,
                fix_suggestion: Some("继续完成证明或检查是否遗漏步骤".to_string()),
            });
        }
        
        // 检查是否有未使用的命题
        let used_propositions: HashSet<_> = proof.steps().iter()
            .flat_map(|step| step.inputs().iter().chain(step.outputs().iter()))
            .collect();
        
        let premise_propositions: HashSet<_> = proof.premises().iter().collect();
        let unused_premises: Vec<_> = premise_propositions.difference(&used_propositions).collect();
        
        if !unused_premises.is_empty() {
            errors.push(VerificationError {
                error_type: VerificationErrorType::CompletenessError,
                location: ErrorLocation::default(),
                description: format!("有{}个前提条件未被使用", unused_premises.len()),
                severity: ErrorSeverity::Warning,
                fix_suggestion: Some("检查是否遗漏了使用这些前提的步骤".to_string()),
            });
        }
        
        errors
    }

    fn name(&self) -> &str {
        "CompletenessValidationRule"
    }

    fn description(&self) -> &str {
        "验证证明的完整性"
    }

    fn applicable_levels(&self) -> Vec<VerificationLevel> {
        vec![VerificationLevel::Strict, VerificationLevel::Expert]
    }
}

/// 一致性验证规则
struct ConsistencyValidationRule;

impl ConsistencyValidationRule {
    fn new() -> Self {
        Self
    }
}

impl ValidationRule for ConsistencyValidationRule {
    fn validate(&self, proof: &Proof, _rule_library: &RuleLibrary) -> Vec<VerificationError> {
        let mut errors = Vec::new();
        
        // 检查命题的一致性
        let mut all_propositions = HashSet::new();
        let mut contradictions = Vec::new();
        
        // 收集所有命题
        for step in proof.steps() {
            for prop in step.outputs() {
                if !all_propositions.insert(prop.clone()) {
                    // 检查是否有矛盾的命题
                    if let Some(contradictory) = self.find_contradictory_proposition(prop, &all_propositions) {
                        contradictions.push((prop.clone(), contradictory.clone()));
                    }
                }
            }
        }
        
        // 报告矛盾
        for (prop1, prop2) in contradictions {
            errors.push(VerificationError {
                error_type: VerificationErrorType::ConsistencyError,
                location: ErrorLocation::default(),
                description: format!("发现矛盾命题: {} 和 {}", prop1.content(), prop2.content()),
                severity: ErrorSeverity::Critical,
                fix_suggestion: Some("检查推理过程，消除矛盾".to_string()),
            });
        }
        
        errors
    }

    fn name(&self) -> &str {
        "ConsistencyValidationRule"
    }

    fn description(&self) -> &str {
        "验证证明的一致性"
    }

    fn applicable_levels(&self) -> Vec<VerificationLevel> {
        vec![VerificationLevel::Strict, VerificationLevel::Expert]
    }

    fn find_contradictory_proposition(&self, prop: &Proposition, all_props: &HashSet<Proposition>) -> Option<Proposition> {
        // 这里应该实现更复杂的矛盾检测逻辑
        // 目前只是简单的示例
        None
    }
}

impl Default for ErrorLocation {
    fn default() -> Self {
        Self {
            step_id: None,
            rule_id: None,
            proposition_id: None,
            line_number: None,
            column_number: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Proof, Proposition, PropositionType, RuleLibrary};

    #[test]
    fn test_verifier_creation() {
        let rule_library = RuleLibrary::new();
        let config = VerificationConfig::default();
        let verifier = ProofVerifier::new(rule_library, config);
        
        assert_eq!(verifier.validation_rules.len(), 4);
    }

    #[test]
    fn test_structural_validation_rule() {
        let rule = StructuralValidationRule::new();
        assert_eq!(rule.name(), "StructuralValidationRule");
        assert_eq!(rule.applicable_levels().len(), 4);
    }

    #[test]
    fn test_logical_validation_rule() {
        let rule = LogicalValidationRule::new();
        assert_eq!(rule.name(), "LogicalValidationRule");
        assert_eq!(rule.applicable_levels().len(), 3);
    }

    #[test]
    fn test_completeness_validation_rule() {
        let rule = CompletenessValidationRule::new();
        assert_eq!(rule.name(), "CompletenessValidationRule");
        assert_eq!(rule.applicable_levels().len(), 2);
    }

    #[test]
    fn test_consistency_validation_rule() {
        let rule = ConsistencyValidationRule::new();
        assert_eq!(rule.name(), "ConsistencyValidationRule");
        assert_eq!(rule.applicable_levels().len(), 2);
    }
}
