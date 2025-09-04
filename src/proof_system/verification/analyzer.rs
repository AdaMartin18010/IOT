use crate::core::{Proof, ProofStep, InferenceRule, ProofError, ProofStatus, Proposition, RuleLibrary};
use crate::verification::{
    VerificationResult, VerificationError, VerificationWarning, VerificationLevel,
    VerificationConfig, VerificationErrorType, VerificationWarningType,
    ErrorLocation, ErrorSeverity
};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// 分析结果
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// 复杂度分析
    pub complexity_analysis: ComplexityAnalysis,
    /// 质量分析
    pub quality_analysis: QualityAnalysis,
    /// 性能分析
    pub performance_analysis: PerformanceAnalysis,
    /// 结构分析
    pub structure_analysis: StructureAnalysis,
    /// 分析时间（毫秒）
    pub analysis_time_ms: u64,
    /// 分析建议
    pub suggestions: Vec<String>,
}

/// 复杂度分析
#[derive(Debug, Clone)]
pub struct ComplexityAnalysis {
    /// 总体复杂度分数
    pub overall_complexity: f64,
    /// 步骤复杂度
    pub step_complexity: f64,
    /// 规则复杂度
    pub rule_complexity: f64,
    /// 依赖复杂度
    pub dependency_complexity: f64,
    /// 复杂度级别
    pub complexity_level: ComplexityLevel,
}

/// 复杂度级别
#[derive(Debug, Clone, PartialEq)]
pub enum ComplexityLevel {
    /// 简单
    Simple,
    /// 中等
    Medium,
    /// 复杂
    Complex,
    /// 极复杂
    VeryComplex,
}

/// 质量分析
#[derive(Debug, Clone)]
pub struct QualityAnalysis {
    /// 总体质量分数
    pub overall_quality: f64,
    /// 逻辑质量
    pub logical_quality: f64,
    /// 结构质量
    pub structural_quality: f64,
    /// 可读性质量
    pub readability_quality: f64,
    /// 质量级别
    pub quality_level: QualityLevel,
}

/// 质量级别
#[derive(Debug, Clone, PartialEq)]
pub enum QualityLevel {
    /// 优秀
    Excellent,
    /// 良好
    Good,
    /// 一般
    Fair,
    /// 需要改进
    NeedsImprovement,
}

/// 性能分析
#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    /// 执行效率
    pub execution_efficiency: f64,
    /// 内存使用效率
    pub memory_efficiency: f64,
    /// 规则应用效率
    pub rule_application_efficiency: f64,
    /// 缓存命中率
    pub cache_hit_rate: f64,
    /// 性能级别
    pub performance_level: PerformanceLevel,
}

/// 性能级别
#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceLevel {
    /// 优秀
    Excellent,
    /// 良好
    Good,
    /// 一般
    Fair,
    /// 需要优化
    NeedsOptimization,
}

/// 结构分析
#[derive(Debug, Clone)]
pub struct StructureAnalysis {
    /// 步骤数量
    pub step_count: usize,
    /// 前提数量
    pub premise_count: usize,
    /// 规则使用分布
    pub rule_usage_distribution: HashMap<u64, usize>,
    /// 依赖深度
    pub dependency_depth: usize,
    /// 分支数量
    pub branch_count: usize,
    /// 循环检测
    pub has_cycles: bool,
}

/// 证明分析器
pub struct ProofAnalyzer {
    config: VerificationConfig,
    rule_library: RuleLibrary,
    analysis_rules: Vec<Box<dyn AnalysisRule>>,
}

/// 分析规则特征
pub trait AnalysisRule {
    /// 应用分析规则
    fn analyze(&self, proof: &Proof, rule_library: &RuleLibrary) -> AnalysisResult;
    
    /// 获取规则名称
    fn name(&self) -> &str;
    
    /// 获取规则描述
    fn description(&self) -> &str;
    
    /// 获取规则类型
    fn rule_type(&self) -> AnalysisRuleType;
}

/// 分析规则类型
#[derive(Debug, Clone, PartialEq)]
pub enum AnalysisRuleType {
    /// 复杂度分析
    Complexity,
    /// 质量分析
    Quality,
    /// 性能分析
    Performance,
    /// 结构分析
    Structure,
}

impl ProofAnalyzer {
    /// 创建新的证明分析器
    pub fn new(rule_library: RuleLibrary, config: VerificationConfig) -> Self {
        let mut analyzer = Self {
            config,
            rule_library,
            analysis_rules: Vec::new(),
        };
        
        // 添加默认分析规则
        analyzer.add_analysis_rule(Box::new(ComplexityAnalysisRule::new()));
        analyzer.add_analysis_rule(Box::new(QualityAnalysisRule::new()));
        analyzer.add_analysis_rule(Box::new(PerformanceAnalysisRule::new()));
        analyzer.add_analysis_rule(Box::new(StructureAnalysisRule::new()));
        
        analyzer
    }

    /// 添加分析规则
    pub fn add_analysis_rule(&mut self, rule: Box<dyn AnalysisRule>) {
        self.analysis_rules.push(rule);
    }

    /// 分析证明
    pub fn analyze(&self, proof: &Proof) -> AnalysisResult {
        let start_time = Instant::now();
        
        // 执行各种分析
        let complexity_analysis = self.analyze_complexity(proof);
        let quality_analysis = self.analyze_quality(proof);
        let performance_analysis = self.analyze_performance(proof);
        let structure_analysis = self.analyze_structure(proof);
        
        // 生成分析建议
        let suggestions = self.generate_suggestions(
            proof,
            &complexity_analysis,
            &quality_analysis,
            &performance_analysis,
            &structure_analysis,
        );
        
        AnalysisResult {
            complexity_analysis,
            quality_analysis,
            performance_analysis,
            structure_analysis,
            analysis_time_ms: start_time.elapsed().as_millis() as u64,
            suggestions,
        }
    }

    /// 分析复杂度
    fn analyze_complexity(&self, proof: &Proof) -> ComplexityAnalysis {
        let step_complexity = (proof.steps().len() as f64).ln() * 0.3;
        let rule_complexity = 2.0; // 默认值
        let dependency_complexity = 1.0; // 默认值
        
        let overall_complexity = (step_complexity + rule_complexity + dependency_complexity) / 3.0;
        let complexity_level = self.determine_complexity_level(overall_complexity);
        
        ComplexityAnalysis {
            overall_complexity,
            step_complexity,
            rule_complexity,
            dependency_complexity,
            complexity_level,
        }
    }

    /// 分析质量
    fn analyze_quality(&self, proof: &Proof) -> QualityAnalysis {
        let logical_quality = 0.8; // 默认值
        let structural_quality = if proof.premises().is_empty() { 0.6 } else { 0.9 };
        let readability_quality = 0.7; // 默认值
        
        let overall_quality = (logical_quality + structural_quality + readability_quality) / 3.0;
        let quality_level = self.determine_quality_level(overall_quality);
        
        QualityAnalysis {
            overall_quality,
            logical_quality,
            structural_quality,
            readability_quality,
            quality_level,
        }
    }

    /// 分析性能
    fn analyze_performance(&self, proof: &Proof) -> PerformanceAnalysis {
        let step_count = proof.steps().len();
        let execution_efficiency = if step_count == 0 { 1.0 } else { 1.0 / (1.0 + step_count as f64 * 0.01) };
        let memory_efficiency = 0.8; // 默认值
        let rule_application_efficiency = 0.9; // 默认值
        let cache_hit_rate = 0.8; // 默认值
        
        let avg_efficiency = (execution_efficiency + memory_efficiency + rule_application_efficiency + cache_hit_rate) / 4.0;
        let performance_level = self.determine_performance_level(avg_efficiency);
        
        PerformanceAnalysis {
            execution_efficiency,
            memory_efficiency,
            rule_application_efficiency,
            cache_hit_rate,
            performance_level,
        }
    }

    /// 分析结构
    fn analyze_structure(&self, proof: &Proof) -> StructureAnalysis {
        let step_count = proof.steps().len();
        let premise_count = proof.premises().len();
        let mut rule_usage_distribution = HashMap::new();
        
        for step in proof.steps() {
            let count = rule_usage_distribution.entry(step.rule_id()).or_insert(0);
            *count += 1;
        }
        
        StructureAnalysis {
            step_count,
            premise_count,
            rule_usage_distribution,
            dependency_depth: 1,
            branch_count: 0,
            has_cycles: false,
        }
    }

    /// 确定复杂度级别
    fn determine_complexity_level(&self, complexity: f64) -> ComplexityLevel {
        match complexity {
            c if c < 2.0 => ComplexityLevel::Simple,
            c if c < 5.0 => ComplexityLevel::Medium,
            c if c < 10.0 => ComplexityLevel::Complex,
            _ => ComplexityLevel::VeryComplex,
        }
    }

    /// 确定质量级别
    fn determine_quality_level(&self, quality: f64) -> QualityLevel {
        match quality {
            q if q >= 0.9 => QualityLevel::Excellent,
            q if q >= 0.7 => QualityLevel::Good,
            q if q >= 0.5 => QualityLevel::Fair,
            _ => QualityLevel::NeedsImprovement,
        }
    }

    /// 确定性能级别
    fn determine_performance_level(&self, efficiency: f64) -> PerformanceLevel {
        match efficiency {
            e if e >= 0.9 => PerformanceLevel::Excellent,
            e if e >= 0.7 => PerformanceLevel::Good,
            e if e >= 0.5 => PerformanceLevel::Fair,
            _ => PerformanceLevel::NeedsOptimization,
        }
    }

    /// 生成分析建议
    fn generate_suggestions(
        &self,
        proof: &Proof,
        complexity: &ComplexityAnalysis,
        quality: &QualityAnalysis,
        performance: &PerformanceAnalysis,
        structure: &StructureAnalysis,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        // 基于复杂度生成建议
        match complexity.complexity_level {
            ComplexityLevel::VeryComplex => {
                suggestions.push("证明复杂度极高，建议分解为多个子证明".to_string());
            }
            ComplexityLevel::Complex => {
                suggestions.push("证明复杂度较高，建议优化推理路径".to_string());
            }
            _ => {}
        }
        
        // 基于质量生成建议
        match quality.quality_level {
            QualityLevel::NeedsImprovement => {
                suggestions.push("证明质量需要改进，建议重新审视推理过程".to_string());
            }
            _ => {}
        }
        
        // 基于结构生成建议
        if structure.step_count > 100 {
            suggestions.push("证明步骤过多，建议检查是否有冗余步骤".to_string());
        }
        
        suggestions
    }

    /// 获取分析规则
    pub fn get_analysis_rules(&self) -> &[Box<dyn AnalysisRule>] {
        &self.analysis_rules
    }

    /// 设置分析配置
    pub fn set_config(&mut self, config: VerificationConfig) {
        self.config = config;
    }
}

/// 复杂度分析规则
struct ComplexityAnalysisRule;

impl ComplexityAnalysisRule {
    fn new() -> Self {
        Self
    }
}

impl AnalysisRule for ComplexityAnalysisRule {
    fn analyze(&self, proof: &Proof, rule_library: &RuleLibrary) -> AnalysisResult {
        let complexity_analysis = ComplexityAnalysis {
            overall_complexity: 1.0,
            step_complexity: 0.5,
            rule_complexity: 1.5,
            dependency_complexity: 1.0,
            complexity_level: ComplexityLevel::Simple,
        };
        
        let quality_analysis = QualityAnalysis {
            overall_quality: 0.0,
            logical_quality: 0.0,
            structural_quality: 0.0,
            readability_quality: 0.0,
            quality_level: QualityLevel::NeedsImprovement,
        };
        
        let performance_analysis = PerformanceAnalysis {
            execution_efficiency: 0.0,
            memory_efficiency: 0.0,
            rule_application_efficiency: 0.0,
            cache_hit_rate: 0.0,
            performance_level: PerformanceLevel::NeedsOptimization,
        };
        
        let structure_analysis = StructureAnalysis {
            step_count: 0,
            premise_count: 0,
            rule_usage_distribution: HashMap::new(),
            dependency_depth: 0,
            branch_count: 0,
            has_cycles: false,
        };
        
        AnalysisResult {
            complexity_analysis,
            quality_analysis,
            performance_analysis,
            structure_analysis,
            analysis_time_ms: 0,
            suggestions: vec![],
        }
    }

    fn name(&self) -> &str {
        "ComplexityAnalysisRule"
    }

    fn description(&self) -> &str {
        "分析证明的复杂度特征"
    }

    fn rule_type(&self) -> AnalysisRuleType {
        AnalysisRuleType::Complexity
    }
}

/// 质量分析规则
struct QualityAnalysisRule;

impl QualityAnalysisRule {
    fn new() -> Self {
        Self
    }
}

impl AnalysisRule for QualityAnalysisRule {
    fn analyze(&self, proof: &Proof, rule_library: &RuleLibrary) -> AnalysisResult {
        let quality_analysis = QualityAnalysis {
            overall_quality: 0.8,
            logical_quality: 0.8,
            structural_quality: 0.9,
            readability_quality: 0.7,
            quality_level: QualityLevel::Good,
        };
        
        let complexity_analysis = ComplexityAnalysis {
            overall_complexity: 0.0,
            step_complexity: 0.0,
            rule_complexity: 0.0,
            dependency_complexity: 0.0,
            complexity_level: ComplexityLevel::Simple,
        };
        
        let performance_analysis = PerformanceAnalysis {
            execution_efficiency: 0.0,
            memory_efficiency: 0.0,
            rule_application_efficiency: 0.0,
            cache_hit_rate: 0.0,
            performance_level: PerformanceLevel::NeedsOptimization,
        };
        
        let structure_analysis = StructureAnalysis {
            step_count: 0,
            premise_count: 0,
            rule_usage_distribution: HashMap::new(),
            dependency_depth: 0,
            branch_count: 0,
            has_cycles: false,
        };
        
        AnalysisResult {
            complexity_analysis,
            quality_analysis,
            performance_analysis,
            structure_analysis,
            analysis_time_ms: 0,
            suggestions: vec![],
        }
    }

    fn name(&self) -> &str {
        "QualityAnalysisRule"
    }

    fn description(&self) -> &str {
        "分析证明的质量特征"
    }

    fn rule_type(&self) -> AnalysisRuleType {
        AnalysisRuleType::Quality
    }
}

/// 性能分析规则
struct PerformanceAnalysisRule;

impl PerformanceAnalysisRule {
    fn new() -> Self {
        Self
    }
}

impl AnalysisRule for PerformanceAnalysisRule {
    fn analyze(&self, proof: &Proof, rule_library: &RuleLibrary) -> AnalysisResult {
        let performance_analysis = PerformanceAnalysis {
            execution_efficiency: 0.8,
            memory_efficiency: 0.9,
            rule_application_efficiency: 0.8,
            cache_hit_rate: 0.7,
            performance_level: PerformanceLevel::Good,
        };
        
        let complexity_analysis = ComplexityAnalysis {
            overall_complexity: 0.0,
            step_complexity: 0.0,
            rule_complexity: 0.0,
            dependency_complexity: 0.0,
            complexity_level: ComplexityLevel::Simple,
        };
        
        let quality_analysis = QualityAnalysis {
            overall_quality: 0.0,
            logical_quality: 0.0,
            structural_quality: 0.0,
            readability_quality: 0.0,
            quality_level: QualityLevel::NeedsImprovement,
        };
        
        let structure_analysis = StructureAnalysis {
            step_count: 0,
            premise_count: 0,
            rule_usage_distribution: HashMap::new(),
            dependency_depth: 0,
            branch_count: 0,
            has_cycles: false,
        };
        
        AnalysisResult {
            complexity_analysis,
            quality_analysis,
            performance_analysis,
            structure_analysis,
            analysis_time_ms: 0,
            suggestions: vec![],
        }
    }

    fn name(&self) -> &str {
        "PerformanceAnalysisRule"
    }

    fn description(&self) -> &str {
        "分析证明的性能特征"
    }

    fn rule_type(&self) -> AnalysisRuleType {
        AnalysisRuleType::Performance
    }
}

/// 结构分析规则
struct StructureAnalysisRule;

impl StructureAnalysisRule {
    fn new() -> Self {
        Self
    }
}

impl AnalysisRule for StructureAnalysisRule {
    fn analyze(&self, proof: &Proof, _rule_library: &RuleLibrary) -> AnalysisResult {
        let structure_analysis = StructureAnalysis {
            step_count: proof.steps().len(),
            premise_count: proof.premises().len(),
            rule_usage_distribution: HashMap::new(),
            dependency_depth: 1,
            branch_count: 0,
            has_cycles: false,
        };
        
        let complexity_analysis = ComplexityAnalysis {
            overall_complexity: 0.0,
            step_complexity: 0.0,
            rule_complexity: 0.0,
            dependency_complexity: 0.0,
            complexity_level: ComplexityLevel::Simple,
        };
        
        let quality_analysis = QualityAnalysis {
            overall_quality: 0.0,
            logical_quality: 0.0,
            structural_quality: 0.0,
            readability_quality: 0.0,
            quality_level: QualityLevel::NeedsImprovement,
        };
        
        let performance_analysis = PerformanceAnalysis {
            execution_efficiency: 0.0,
            memory_efficiency: 0.0,
            rule_application_efficiency: 0.0,
            cache_hit_rate: 0.0,
            performance_level: PerformanceLevel::NeedsOptimization,
        };
        
        AnalysisResult {
            complexity_analysis,
            quality_analysis,
            performance_analysis,
            structure_analysis,
            analysis_time_ms: 0,
            suggestions: vec![],
        }
    }

    fn name(&self) -> &str {
        "StructureAnalysisRule"
    }

    fn description(&self) -> &str {
        "分析证明的结构特征"
    }

    fn rule_type(&self) -> AnalysisRuleType {
        AnalysisRuleType::Structure
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Proof, Proposition, PropositionType, RuleLibrary};

    #[test]
    fn test_analyzer_creation() {
        let rule_library = RuleLibrary::new();
        let config = VerificationConfig::default();
        let analyzer = ProofAnalyzer::new(rule_library, config);
        
        assert_eq!(analyzer.analysis_rules.len(), 4);
    }

    #[test]
    fn test_complexity_analysis_rule() {
        let rule = ComplexityAnalysisRule::new();
        assert_eq!(rule.name(), "ComplexityAnalysisRule");
        assert_eq!(rule.rule_type(), AnalysisRuleType::Complexity);
    }

    #[test]
    fn test_quality_analysis_rule() {
        let rule = QualityAnalysisRule::new();
        assert_eq!(rule.name(), "QualityAnalysisRule");
        assert_eq!(rule.rule_type(), AnalysisRuleType::Quality);
    }

    #[test]
    fn test_performance_analysis_rule() {
        let rule = PerformanceAnalysisRule::new();
        assert_eq!(rule.name(), "PerformanceAnalysisRule");
        assert_eq!(rule.rule_type(), AnalysisRuleType::Performance);
    }

    #[test]
    fn test_structure_analysis_rule() {
        let rule = StructureAnalysisRule::new();
        assert_eq!(rule.name(), "StructureAnalysisRule");
        assert_eq!(rule.rule_type(), AnalysisRuleType::Structure);
    }
}
