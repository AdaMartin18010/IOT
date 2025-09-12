//! 证明步骤模块
//! 
//! 本模块定义了证明步骤的扩展功能，包括步骤验证、步骤转换等。

use super::{ProofStep, ProofStepType, StepId, ProofError};
use std::collections::HashMap;

/// 步骤验证器
pub struct StepValidator {
    /// 验证规则
    validation_rules: Vec<Box<dyn ValidationRule>>,
}

impl StepValidator {
    /// 创建新的步骤验证器
    pub fn new() -> Self {
        Self {
            validation_rules: Vec::new(),
        }
    }
    
    /// 添加验证规则
    pub fn add_validation_rule<T: ValidationRule + 'static>(&mut self, rule: T) {
        self.validation_rules.push(Box::new(rule));
    }
    
    /// 验证证明步骤
    pub fn validate_step(&self, step: &ProofStep) -> ValidationResult {
        let mut result = ValidationResult::new();
        
        // 应用所有验证规则
        for rule in &self.validation_rules {
            let rule_result = rule.validate(step);
            result.merge(rule_result);
        }
        
        result
    }
    
    /// 验证步骤序列
    pub fn validate_step_sequence(&self, steps: &[ProofStep]) -> ValidationResult {
        let mut result = ValidationResult::new();
        
        // 验证每个步骤
        for step in steps {
            let step_result = self.validate_step(step);
            result.merge(step_result);
        }
        
        // 验证步骤间的依赖关系
        if let Some(dependency_error) = self.validate_dependencies(steps) {
            result.add_error(dependency_error);
        }
        
        result
    }
    
    /// 验证步骤依赖关系
    fn validate_dependencies(&self, steps: &[ProofStep]) -> Option<String> {
        let step_ids: std::collections::HashSet<StepId> = steps.iter().map(|s| s.id).collect();
        
        for step in steps {
            for &dependency_id in &step.dependencies {
                if !step_ids.contains(&dependency_id) {
                    return Some(format!(
                        "步骤 {} 依赖的步骤 {} 不存在",
                        step.id, dependency_id
                    ));
                }
            }
        }
        
        None
    }
}

/// 验证规则特征
pub trait ValidationRule {
    /// 验证规则名称
    fn name(&self) -> &str;
    
    /// 验证规则描述
    fn description(&self) -> &str;
    
    /// 应用验证规则
    fn validate(&self, step: &ProofStep) -> ValidationResult;
}

/// 基本验证规则
pub struct BasicValidationRule;

impl ValidationRule for BasicValidationRule {
    fn name(&self) -> &str {
        "基本验证规则"
    }
    
    fn description(&self) -> &str {
        "验证步骤的基本属性，如ID、描述、类型等"
    }
    
    fn validate(&self, step: &ProofStep) -> ValidationResult {
        let mut result = ValidationResult::new();
        
        // 验证步骤ID
        if step.id == 0 {
            result.add_error("步骤ID不能为0".to_string());
        }
        
        // 验证步骤描述
        if step.description.is_empty() {
            result.add_error("步骤描述不能为空".to_string());
        }
        
        // 验证步骤类型
        if matches!(step.step_type, ProofStepType::Other(_)) {
            result.add_warning("使用了自定义步骤类型".to_string());
        }
        
        // 验证证明理由
        if step.justification.is_empty() {
            result.add_error("证明理由不能为空".to_string());
        }
        
        result
    }
}

/// 逻辑验证规则
pub struct LogicalValidationRule;

impl ValidationRule for LogicalValidationRule {
    fn name(&self) -> &str {
        "逻辑验证规则"
    }
    
    fn description(&self) -> &str {
        "验证步骤的逻辑一致性"
    }
    
    fn validate(&self, step: &ProofStep) -> ValidationResult {
        let mut result = ValidationResult::new();
        
        // 验证输入输出命题的一致性
        if !step.input_propositions.is_empty() && step.output_propositions.is_empty() {
            result.add_warning("步骤有输入但没有输出".to_string());
        }
        
        // 验证依赖步骤的合理性
        if step.dependencies.len() > 5 {
            result.add_warning("步骤依赖过多，可能影响可读性".to_string());
        }
        
        result
    }
}

/// 验证结果
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// 是否有效
    pub is_valid: bool,
    /// 错误信息
    pub errors: Vec<String>,
    /// 警告信息
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// 创建新的验证结果
    pub fn new() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }
    
    /// 添加错误
    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
        self.is_valid = false;
    }
    
    /// 添加警告
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
    
    /// 合并验证结果
    pub fn merge(&mut self, other: ValidationResult) {
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
        if !other.is_valid {
            self.is_valid = false;
        }
    }
    
    /// 获取错误数量
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }
    
    /// 获取警告数量
    pub fn warning_count(&self) -> usize {
        self.warnings.len()
    }
}

/// 步骤转换器
pub struct StepTransformer {
    /// 转换规则
    transformation_rules: Vec<Box<dyn TransformationRule>>,
}

impl StepTransformer {
    /// 创建新的步骤转换器
    pub fn new() -> Self {
        Self {
            transformation_rules: Vec::new(),
        }
    }
    
    /// 添加转换规则
    pub fn add_transformation_rule<T: TransformationRule + 'static>(&mut self, rule: T) {
        self.transformation_rules.push(Box::new(rule));
    }
    
    /// 转换证明步骤
    pub fn transform_step(&self, step: &ProofStep) -> Result<ProofStep, ProofError> {
        let mut transformed_step = step.clone();
        
        // 应用所有转换规则
        for rule in &self.transformation_rules {
            if rule.is_applicable(step) {
                transformed_step = rule.transform(&transformed_step)?;
            }
        }
        
        Ok(transformed_step)
    }
}

/// 转换规则特征
pub trait TransformationRule {
    /// 转换规则名称
    fn name(&self) -> &str;
    
    /// 转换规则描述
    fn description(&self) -> &str;
    
    /// 检查规则是否适用
    fn is_applicable(&self, step: &ProofStep) -> bool;
    
    /// 应用转换规则
    fn transform(&self, step: &ProofStep) -> Result<ProofStep, ProofError>;
}

/// 步骤标准化规则
pub struct StepNormalizationRule;

impl TransformationRule for StepNormalizationRule {
    fn name(&self) -> &str {
        "步骤标准化规则"
    }
    
    fn description(&self) -> &str {
        "标准化步骤的格式和结构"
    }
    
    fn is_applicable(&self, _step: &ProofStep) -> bool {
        // 总是适用
        true
    }
    
    fn transform(&self, step: &ProofStep) -> Result<ProofStep, ProofError> {
        let mut normalized_step = step.clone();
        
        // 标准化描述（去除首尾空格）
        normalized_step.description = step.description.trim().to_string();
        
        // 标准化证明理由
        normalized_step.justification = step.justification.trim().to_string();
        
        // 确保输入输出命题按ID排序
        normalized_step.input_propositions.sort_by(|a, b| a.id.cmp(&b.id));
        normalized_step.output_propositions.sort_by(|a, b| a.id.cmp(&b.id));
        
        // 确保依赖步骤按ID排序
        normalized_step.dependencies.sort();
        
        Ok(normalized_step)
    }
}

/// 步骤分析器
pub struct StepAnalyzer {
    /// 分析规则
    analysis_rules: Vec<Box<dyn AnalysisRule>>,
}

impl StepAnalyzer {
    /// 创建新的步骤分析器
    pub fn new() -> Self {
        Self {
            analysis_rules: Vec::new(),
        }
    }
    
    /// 添加分析规则
    pub fn add_analysis_rule<T: AnalysisRule + 'static>(&mut self, rule: T) {
        self.analysis_rules.push(Box::new(rule));
    }
    
    /// 分析证明步骤
    pub fn analyze_step(&self, step: &ProofStep) -> AnalysisResult {
        let mut result = AnalysisResult::new();
        
        // 应用所有分析规则
        for rule in &self.analysis_rules {
            let rule_result = rule.analyze(step);
            result.merge(rule_result);
        }
        
        result
    }
}

/// 分析规则特征
pub trait AnalysisRule {
    /// 分析规则名称
    fn name(&self) -> &str;
    
    /// 分析规则描述
    fn description(&self) -> &str;
    
    /// 应用分析规则
    fn analyze(&self, step: &ProofStep) -> AnalysisResult;
}

/// 复杂度分析规则
pub struct ComplexityAnalysisRule;

impl AnalysisRule for ComplexityAnalysisRule {
    fn name(&self) -> &str {
        "复杂度分析规则"
    }
    
    fn description(&self) -> &str {
        "分析步骤的复杂度"
    }
    
    fn analyze(&self, step: &ProofStep) -> AnalysisResult {
        let mut result = AnalysisResult::new();
        
        // 计算描述复杂度
        let description_complexity = step.description.len() as f64;
        if description_complexity > 200.0 {
            result.add_metric("描述复杂度".to_string(), "高".to_string());
            result.add_suggestion("考虑简化步骤描述".to_string());
        } else if description_complexity > 100.0 {
            result.add_metric("描述复杂度".to_string(), "中".to_string());
        } else {
            result.add_metric("描述复杂度".to_string(), "低".to_string());
        }
        
        // 计算依赖复杂度
        let dependency_complexity = step.dependencies.len() as f64;
        if dependency_complexity > 3.0 {
            result.add_metric("依赖复杂度".to_string(), "高".to_string());
            result.add_suggestion("考虑减少步骤依赖".to_string());
        } else if dependency_complexity > 1.0 {
            result.add_metric("依赖复杂度".to_string(), "中".to_string());
        } else {
            result.add_metric("依赖复杂度".to_string(), "低".to_string());
        }
        
        result
    }
}

/// 分析结果
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// 指标
    pub metrics: HashMap<String, String>,
    /// 建议
    pub suggestions: Vec<String>,
}

impl AnalysisResult {
    /// 创建新的分析结果
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            suggestions: Vec::new(),
        }
    }
    
    /// 添加指标
    pub fn add_metric(&mut self, key: String, value: String) {
        self.metrics.insert(key, value);
    }
    
    /// 添加建议
    pub fn add_suggestion(&mut self, suggestion: String) {
        self.suggestions.push(suggestion);
    }
    
    /// 合并分析结果
    pub fn merge(&mut self, other: AnalysisResult) {
        self.metrics.extend(other.metrics);
        self.suggestions.extend(other.suggestions);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::{ProofStep, ProofStepType, Proposition, PropositionType};
    use std::collections::HashMap;
    
    fn create_test_proposition(id: &str, content: &str) -> Proposition {
        Proposition {
            id: id.to_string(),
            content: content.to_string(),
            proposition_type: PropositionType::Theorem,
            metadata: HashMap::new(),
        }
    }
    
    fn create_test_step(id: StepId, description: &str) -> ProofStep {
        ProofStep::new(
            id,
            id as u32,
            description.to_string(),
            ProofStepType::RuleApplication,
        )
        .with_justification("测试理由".to_string())
    }
    
    #[test]
    fn test_step_validator() {
        let mut validator = StepValidator::new();
        validator.add_validation_rule(BasicValidationRule);
        
        let step = create_test_step(1, "测试步骤");
        let result = validator.validate_step(&step);
        
        assert!(result.is_valid);
        assert_eq!(result.error_count(), 0);
    }
    
    #[test]
    fn test_step_transformer() {
        let mut transformer = StepTransformer::new();
        transformer.add_transformation_rule(StepNormalizationRule);
        
        let step = create_test_step(1, "  测试步骤  ");
        let transformed = transformer.transform_step(&step).unwrap();
        
        assert_eq!(transformed.description, "测试步骤");
    }
    
    #[test]
    fn test_step_analyzer() {
        let mut analyzer = StepAnalyzer::new();
        analyzer.add_analysis_rule(ComplexityAnalysisRule);
        
        let step = create_test_step(1, "这是一个非常长的步骤描述，用来测试复杂度分析规则的功能和效果，这个描述应该足够长以触发复杂度分析规则的建议生成机制，确保测试能够正确验证分析器的功能");
        let result = analyzer.analyze_step(&step);
        
        assert!(!result.metrics.is_empty());
        assert!(!result.suggestions.is_empty());
    }
}
