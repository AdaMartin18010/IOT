use crate::core::{Proof, ProofStep, InferenceRule, ProofError, ProofStatus, Proposition, RuleLibrary};
use crate::verification::{
    VerificationResult, VerificationError, VerificationWarning, VerificationLevel,
    VerificationConfig, VerificationErrorType, VerificationWarningType,
    ErrorLocation, ErrorSeverity
};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// 证明检查器
pub struct ProofChecker {
    config: VerificationConfig,
    rule_library: RuleLibrary,
    check_rules: Vec<Box<dyn CheckRule>>,
    syntax_validator: SyntaxValidator,
    semantic_validator: SemanticValidator,
    style_checker: StyleChecker,
}

/// 检查规则特征
pub trait CheckRule {
    /// 应用检查规则
    fn check(&self, proof: &Proof, rule_library: &RuleLibrary) -> Vec<VerificationError>;
    
    /// 获取规则名称
    fn name(&self) -> &str;
    
    /// 获取规则描述
    fn description(&self) -> &str;
    
    /// 获取规则类型
    fn rule_type(&self) -> CheckRuleType;
}

/// 检查规则类型
#[derive(Debug, Clone, PartialEq)]
pub enum CheckRuleType {
    /// 语法检查
    Syntax,
    /// 语义检查
    Semantic,
    /// 风格检查
    Style,
    /// 规范性检查
    Normative,
}

impl ProofChecker {
    /// 创建新的证明检查器
    pub fn new(rule_library: RuleLibrary, config: VerificationConfig) -> Self {
        let mut checker = Self {
            config,
            rule_library,
            check_rules: Vec::new(),
            syntax_validator: SyntaxValidator::new(),
            semantic_validator: SemanticValidator::new(),
            style_checker: StyleChecker::new(),
        };
        
        // 添加默认检查规则
        checker.add_check_rule(Box::new(SyntaxCheckRule::new()));
        checker.add_check_rule(Box::new(SemanticCheckRule::new()));
        checker.add_check_rule(Box::new(StyleCheckRule::new()));
        checker.add_check_rule(Box::new(NormativeCheckRule::new()));
        
        checker
    }

    /// 添加检查规则
    pub fn add_check_rule(&mut self, rule: Box<dyn CheckRule>) {
        self.check_rules.push(rule);
    }

    /// 检查证明
    pub fn check(&self, proof: &Proof) -> VerificationResult {
        let start_time = Instant::now();
        
        let mut all_errors = Vec::new();
        let mut all_warnings = Vec::new();
        
        // 应用所有检查规则
        for rule in &self.check_rules {
            let errors = rule.check(proof, &self.rule_library);
            all_errors.extend(errors);
        }
        
        // 执行专门的验证器
        let syntax_errors = self.syntax_validator.validate(proof);
        let semantic_errors = self.semantic_validator.validate(proof, &self.rule_library);
        let style_warnings = self.style_checker.check(proof);
        
        all_errors.extend(syntax_errors);
        all_errors.extend(semantic_errors);
        all_warnings.extend(style_warnings);
        
        // 生成建议
        let suggestions = self.generate_suggestions(proof, &all_errors, &all_warnings);
        
        // 生成详细报告
        let detailed_report = if self.config.enable_detailed_report {
            Some(self.generate_detailed_report(proof, &all_errors, &all_warnings, &suggestions))
        } else {
            None
        };
        
        // 判断是否通过检查
        let passed = all_errors.iter().all(|e| e.severity < ErrorSeverity::Error);
        
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
        
        // 基于检查结果生成建议
        if proof.steps().is_empty() {
            suggestions.push("证明尚未开始，建议添加前提条件".to_string());
        }
        
        if proof.steps().len() > 50 {
            suggestions.push("证明步骤较多，建议检查是否有冗余步骤".to_string());
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
        
        report.push_str("=== 证明检查详细报告 ===\n\n");
        
        // 证明基本信息
        report.push_str(&format!("证明ID: {}\n", proof.id()));
        report.push_str(&format!("证明状态: {:?}\n", proof.status()));
        report.push_str(&format!("前提数量: {}\n", proof.premises().len()));
        report.push_str(&format!("步骤数量: {}\n", proof.steps().len()));
        report.push_str(&format!("检查级别: {:?}\n\n", self.config.level));
        
        // 错误报告
        if !errors.is_empty() {
            report.push_str("=== 检查错误 ===\n");
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
            report.push_str("=== 检查警告 ===\n");
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

    /// 获取检查规则
    pub fn get_check_rules(&self) -> &[Box<dyn CheckRule>] {
        &self.check_rules
    }

    /// 设置检查配置
    pub fn set_config(&mut self, config: VerificationConfig) {
        self.config = config;
    }
}

/// 语法验证器
struct SyntaxValidator;

impl SyntaxValidator {
    fn new() -> Self {
        Self
    }

    fn validate(&self, proof: &Proof) -> Vec<VerificationError> {
        let mut errors = Vec::new();
        
        // 检查证明ID的有效性
        if proof.id() == 0 {
            errors.push(VerificationError {
                error_type: VerificationErrorType::SyntaxError,
                location: ErrorLocation::default(),
                description: "证明ID无效".to_string(),
                severity: ErrorSeverity::Error,
                fix_suggestion: Some("设置有效的证明ID".to_string()),
            });
        }
        
        // 检查步骤ID的唯一性
        let mut step_ids = HashSet::new();
        for step in proof.steps() {
            if !step_ids.insert(step.id()) {
                errors.push(VerificationError {
                    error_type: VerificationErrorType::SyntaxError,
                    location: ErrorLocation {
                        step_id: Some(step.id()),
                        ..Default::default()
                    },
                    description: "步骤ID重复".to_string(),
                    severity: ErrorSeverity::Error,
                    fix_suggestion: Some("确保每个步骤有唯一的ID".to_string()),
                });
            }
        }
        
        // 检查命题内容的有效性
        for step in proof.steps() {
            for prop in step.inputs().iter().chain(step.outputs().iter()) {
                if prop.content().trim().is_empty() {
                    errors.push(VerificationError {
                        error_type: VerificationErrorType::SyntaxError,
                        location: ErrorLocation {
                            step_id: Some(step.id()),
                            ..Default::default()
                        },
                        description: "命题内容为空".to_string(),
                        severity: ErrorSeverity::Error,
                        fix_suggestion: Some("为命题提供有效内容".to_string()),
                    });
                }
            }
        }
        
        errors
    }
}

/// 语义验证器
struct SemanticValidator;

impl SemanticValidator {
    fn new() -> Self {
        Self
    }

    fn validate(&self, proof: &Proof, rule_library: &RuleLibrary) -> Vec<VerificationError> {
        let mut errors = Vec::new();
        
        // 检查规则引用的有效性
        for step in proof.steps() {
            if !rule_library.has_rule(step.rule_id()) {
                errors.push(VerificationError {
                    error_type: VerificationErrorType::RuleApplicationError,
                    location: ErrorLocation {
                        step_id: Some(step.id()),
                        rule_id: Some(step.rule_id()),
                        ..Default::default()
                    },
                    description: format!("引用了不存在的规则ID: {}", step.rule_id()),
                    severity: ErrorSeverity::Error,
                    fix_suggestion: Some("检查规则ID或添加缺失的规则".to_string()),
                });
            }
        }
        
        // 检查依赖关系的有效性
        for step in proof.steps() {
            for dep_id in step.dependencies() {
                if !proof.steps().iter().any(|s| s.id() == *dep_id) {
                    errors.push(VerificationError {
                        error_type: VerificationErrorType::StructuralError,
                        location: ErrorLocation {
                            step_id: Some(step.id()),
                            ..Default::default()
                        },
                        description: format!("步骤依赖了不存在的步骤ID: {}", dep_id),
                        severity: ErrorSeverity::Error,
                        fix_suggestion: Some("检查依赖关系或添加缺失的步骤".to_string()),
                    });
                }
            }
        }
        
        errors
    }
}

/// 风格检查器
struct StyleChecker;

impl StyleChecker {
    fn new() -> Self {
        Self
    }

    fn check(&self, proof: &Proof) -> Vec<VerificationWarning> {
        let mut warnings = Vec::new();
        
        // 检查证明长度
        if proof.steps().len() > 100 {
            warnings.push(VerificationWarning {
                warning_type: VerificationWarningType::StyleWarning,
                location: ErrorLocation::default(),
                description: "证明步骤过多，可能影响可读性".to_string(),
                suggestion: "考虑将证明分解为多个子证明".to_string(),
            });
        }
        
        // 检查步骤描述的质量
        for step in proof.steps() {
            if step.justification().trim().is_empty() {
                warnings.push(VerificationWarning {
                    warning_type: VerificationWarningType::StyleWarning,
                    location: ErrorLocation {
                        step_id: Some(step.id()),
                        ..Default::default()
                    },
                    description: "步骤缺少说明".to_string(),
                    suggestion: "为每个步骤添加清晰的说明".to_string(),
                });
            }
        }
        
        // 检查命题命名的规范性
        for step in proof.steps() {
            for prop in step.outputs() {
                if prop.content().len() < 3 {
                    warnings.push(VerificationWarning {
                        warning_type: VerificationWarningType::StyleWarning,
                        location: ErrorLocation {
                            step_id: Some(step.id()),
                            ..Default::default()
                        },
                        description: "命题内容过短，可能不够清晰".to_string(),
                        suggestion: "使用更清晰的命题描述".to_string(),
                    });
                }
            }
        }
        
        warnings
    }
}

/// 语法检查规则
struct SyntaxCheckRule;

impl SyntaxCheckRule {
    fn new() -> Self {
        Self
    }
}

impl CheckRule for SyntaxCheckRule {
    fn check(&self, proof: &Proof, _rule_library: &RuleLibrary) -> Vec<VerificationError> {
        let mut errors = Vec::new();
        
        // 检查基本语法结构
        if proof.premises().is_empty() && proof.steps().is_empty() {
            errors.push(VerificationError {
                error_type: VerificationErrorType::SyntaxError,
                location: ErrorLocation::default(),
                description: "证明缺少基本内容".to_string(),
                severity: ErrorSeverity::Error,
                fix_suggestion: Some("添加前提条件或证明步骤".to_string()),
            });
        }
        
        errors
    }

    fn name(&self) -> &str {
        "SyntaxCheckRule"
    }

    fn description(&self) -> &str {
        "检查证明的基本语法结构"
    }

    fn rule_type(&self) -> CheckRuleType {
        CheckRuleType::Syntax
    }
}

/// 语义检查规则
struct SemanticCheckRule;

impl SemanticCheckRule {
    fn new() -> Self {
        Self
    }
}

impl CheckRule for SemanticCheckRule {
    fn check(&self, proof: &Proof, rule_library: &RuleLibrary) -> Vec<VerificationError> {
        let mut errors = Vec::new();
        
        // 检查语义一致性
        for step in proof.steps() {
            if let Some(rule) = rule_library.get_rule(step.rule_id()) {
                // 检查输入输出的一致性
                if step.inputs().is_empty() && !step.outputs().is_empty() {
                    errors.push(VerificationError {
                        error_type: VerificationErrorType::LogicalError,
                        location: ErrorLocation {
                            step_id: Some(step.id()),
                            rule_id: Some(step.rule_id()),
                            ..Default::default()
                        },
                        description: "步骤有输出但无输入".to_string(),
                        severity: ErrorSeverity::Warning,
                        fix_suggestion: Some("检查步骤的输入输出关系".to_string()),
                    });
                }
            }
        }
        
        errors
    }

    fn name(&self) -> &str {
        "SemanticCheckRule"
    }

    fn description(&self) -> &str {
        "检查证明的语义一致性"
    }

    fn rule_type(&self) -> CheckRuleType {
        CheckRuleType::Semantic
    }
}

/// 风格检查规则
struct StyleCheckRule;

impl StyleCheckRule {
    fn new() -> Self {
        Self
    }
}

impl CheckRule for StyleCheckRule {
    fn check(&self, proof: &Proof, _rule_library: &RuleLibrary) -> Vec<VerificationError> {
        let mut errors = Vec::new();
        
        // 检查命名规范性
        for step in proof.steps() {
            if step.justification().contains("TODO") || step.justification().contains("FIXME") {
                errors.push(VerificationError {
                    error_type: VerificationErrorType::SyntaxError,
                    location: ErrorLocation {
                        step_id: Some(step.id()),
                        ..Default::default()
                    },
                    description: "步骤说明包含待办标记".to_string(),
                    severity: ErrorSeverity::Warning,
                    fix_suggestion: Some("完成步骤说明，移除待办标记".to_string()),
                });
            }
        }
        
        errors
    }

    fn name(&self) -> &str {
        "StyleCheckRule"
    }

    fn description(&self) -> &str {
        "检查证明的风格规范性"
    }

    fn rule_type(&self) -> CheckRuleType {
        CheckRuleType::Style
    }
}

/// 规范性检查规则
struct NormativeCheckRule;

impl NormativeCheckRule {
    fn new() -> Self {
        Self
    }
}

impl CheckRule for NormativeCheckRule {
    fn check(&self, proof: &Proof, _rule_library: &RuleLibrary) -> Vec<VerificationError> {
        let mut errors = Vec::new();
        
        // 检查是否符合证明规范
        if proof.steps().len() > 0 && proof.premises().len() == 0 {
            errors.push(VerificationError {
                error_type: VerificationErrorType::StructuralError,
                location: ErrorLocation::default(),
                description: "证明步骤存在但缺少前提条件".to_string(),
                severity: ErrorSeverity::Warning,
                fix_suggestion: Some("添加必要的前提条件".to_string()),
            });
        }
        
        errors
    }

    fn name(&self) -> &str {
        "NormativeCheckRule"
    }

    fn description(&self) -> &str {
        "检查证明是否符合规范要求"
    }

    fn rule_type(&self) -> CheckRuleType {
        CheckRuleType::Normative
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Proof, Proposition, PropositionType, RuleLibrary};

    #[test]
    fn test_checker_creation() {
        let rule_library = RuleLibrary::new();
        let config = VerificationConfig::default();
        let checker = ProofChecker::new(rule_library, config);
        
        assert_eq!(checker.check_rules.len(), 4);
    }

    #[test]
    fn test_syntax_check_rule() {
        let rule = SyntaxCheckRule::new();
        assert_eq!(rule.name(), "SyntaxCheckRule");
        assert_eq!(rule.rule_type(), CheckRuleType::Syntax);
    }

    #[test]
    fn test_semantic_check_rule() {
        let rule = SemanticCheckRule::new();
        assert_eq!(rule.name(), "SemanticCheckRule");
        assert_eq!(rule.rule_type(), CheckRuleType::Semantic);
    }

    #[test]
    fn test_style_check_rule() {
        let rule = StyleCheckRule::new();
        assert_eq!(rule.name(), "StyleCheckRule");
        assert_eq!(rule.rule_type(), CheckRuleType::Style);
    }

    #[test]
    fn test_normative_check_rule() {
        let rule = NormativeCheckRule::new();
        assert_eq!(rule.name(), "NormativeCheckRule");
        assert_eq!(rule.rule_type(), CheckRuleType::Normative);
    }
}
