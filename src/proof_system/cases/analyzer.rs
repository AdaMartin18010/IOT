use crate::cases::{
    CaseType, DifficultyLevel, CaseCategory, CaseTag, CaseMetadata, CaseAnalysisResult,
    CaseRecord, CaseStatistics
};
use crate::core::{Proof, ProofStep, ProofError};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Instant, Duration};

/// 分析类型
#[derive(Debug, Clone, PartialEq)]
pub enum AnalysisType {
    /// 复杂度分析
    Complexity,
    /// 质量分析
    Quality,
    /// 学习价值分析
    LearningValue,
    /// 难度评估
    DifficultyAssessment,
    /// 改进建议
    ImprovementSuggestions,
    /// 相关案例分析
    RelatedCases,
    /// 综合分析
    Comprehensive,
}

/// 复杂度指标
#[derive(Debug, Clone)]
pub struct ComplexityMetrics {
    pub structural_complexity: f64,
    pub logical_complexity: f64,
    pub computational_complexity: f64,
    pub overall_complexity: f64,
    pub complexity_level: ComplexityLevel,
}

/// 复杂度级别
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum ComplexityLevel {
    /// 简单
    Simple,
    /// 中等
    Medium,
    /// 复杂
    Complex,
    /// 非常复杂
    VeryComplex,
}

/// 质量指标
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    pub correctness: f64,
    pub completeness: f64,
    pub clarity: f64,
    pub consistency: f64,
    pub maintainability: f64,
    pub overall_quality: f64,
    pub quality_level: QualityLevel,
}

/// 质量级别
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum QualityLevel {
    /// 差
    Poor,
    /// 一般
    Fair,
    /// 良好
    Good,
    /// 优秀
    Excellent,
}

/// 学习价值指标
#[derive(Debug, Clone)]
pub struct LearningValueMetrics {
    pub educational_value: f64,
    pub skill_development: f64,
    pub knowledge_transfer: f64,
    pub practical_applicability: f64,
    pub overall_learning_value: f64,
    pub learning_level: LearningLevel,
}

/// 学习级别
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum LearningLevel {
    /// 基础
    Basic,
    /// 进阶
    Intermediate,
    /// 高级
    Advanced,
    /// 专家
    Expert,
}

/// 分析配置
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    pub enable_complexity_analysis: bool,
    pub enable_quality_analysis: bool,
    pub enable_learning_analysis: bool,
    pub enable_improvement_suggestions: bool,
    pub enable_related_cases: bool,
    pub complexity_thresholds: HashMap<ComplexityLevel, f64>,
    pub quality_thresholds: HashMap<QualityLevel, f64>,
    pub learning_thresholds: HashMap<LearningLevel, f64>,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        let mut complexity_thresholds = HashMap::new();
        complexity_thresholds.insert(ComplexityLevel::Simple, 0.25);
        complexity_thresholds.insert(ComplexityLevel::Medium, 0.5);
        complexity_thresholds.insert(ComplexityLevel::Complex, 0.75);
        complexity_thresholds.insert(ComplexityLevel::VeryComplex, 1.0);
        
        let mut quality_thresholds = HashMap::new();
        quality_thresholds.insert(QualityLevel::Poor, 0.25);
        quality_thresholds.insert(QualityLevel::Fair, 0.5);
        quality_thresholds.insert(QualityLevel::Good, 0.75);
        quality_thresholds.insert(QualityLevel::Excellent, 1.0);
        
        let mut learning_thresholds = HashMap::new();
        learning_thresholds.insert(LearningLevel::Basic, 0.25);
        learning_thresholds.insert(LearningLevel::Intermediate, 0.5);
        learning_thresholds.insert(LearningLevel::Advanced, 0.75);
        learning_thresholds.insert(LearningLevel::Expert, 1.0);
        
        Self {
            enable_complexity_analysis: true,
            enable_quality_analysis: true,
            enable_learning_analysis: true,
            enable_improvement_suggestions: true,
            enable_related_cases: true,
            complexity_thresholds,
            quality_thresholds,
            learning_thresholds,
        }
    }
}

/// 案例分析器
pub struct CaseAnalyzer {
    config: AnalysisConfig,
    analysis_history: Arc<Mutex<VecDeque<AnalysisRecord>>>,
    performance_metrics: Arc<Mutex<AnalysisPerformanceMetrics>>,
    improvement_patterns: Arc<RwLock<HashMap<String, ImprovementPattern>>>,
    case_similarity_cache: Arc<RwLock<HashMap<String, HashMap<String, f64>>>>,
}

/// 分析记录
#[derive(Debug, Clone)]
pub struct AnalysisRecord {
    pub analysis_id: String,
    pub case_id: String,
    pub analysis_type: AnalysisType,
    pub timestamp: Instant,
    pub execution_time: Duration,
    pub result: CaseAnalysisResult,
}

/// 分析性能指标
#[derive(Debug, Clone, Default)]
pub struct AnalysisPerformanceMetrics {
    pub total_analyses: usize,
    pub average_execution_time: Duration,
    pub cache_hit_rate: f64,
    pub analysis_accuracy: f64,
}

/// 改进模式
#[derive(Debug, Clone)]
pub struct ImprovementPattern {
    pub pattern_id: String,
    pub description: String,
    pub applicable_conditions: Vec<String>,
    pub expected_improvement: f64,
    pub implementation_difficulty: f64,
    pub success_rate: f64,
    pub usage_count: usize,
}

impl CaseAnalyzer {
    pub fn new(config: AnalysisConfig) -> Self {
        Self {
            config,
            analysis_history: Arc::new(Mutex::new(VecDeque::new())),
            performance_metrics: Arc::new(Mutex::new(AnalysisPerformanceMetrics::default())),
            improvement_patterns: Arc::new(RwLock::new(HashMap::new())),
            case_similarity_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 执行案例分析
    pub async fn analyze_case(
        &self,
        case_record: &CaseRecord,
        analysis_types: Option<Vec<AnalysisType>>,
    ) -> Result<CaseAnalysisResult, ProofError> {
        let start_time = Instant::now();
        
        let types = analysis_types.unwrap_or_else(|| vec![AnalysisType::Comprehensive]);
        
        let mut analysis_result = CaseAnalysisResult {
            case_id: case_record.case_id.clone(),
            complexity_score: 0.0,
            difficulty_assessment: DifficultyLevel::Beginner,
            learning_value: 0.0,
            common_mistakes: Vec::new(),
            improvement_suggestions: Vec::new(),
            related_cases: Vec::new(),
            strategy_recommendations: Vec::new(),
        };
        
        // 执行各种分析
        for analysis_type in &types {
            match analysis_type {
                AnalysisType::Complexity => {
                    if self.config.enable_complexity_analysis {
                        let complexity_metrics = self.analyze_complexity(&case_record.proof).await?;
                        analysis_result.complexity_score = complexity_metrics.overall_complexity;
                    }
                }
                AnalysisType::Quality => {
                    if self.config.enable_quality_analysis {
                        let quality_metrics = self.analyze_quality(case_record).await?;
                        // 基于质量评估调整其他指标
                    }
                }
                AnalysisType::LearningValue => {
                    if self.config.enable_learning_analysis {
                        let learning_metrics = self.analyze_learning_value(case_record).await?;
                        analysis_result.learning_value = learning_metrics.overall_learning_value;
                    }
                }
                AnalysisType::DifficultyAssessment => {
                    analysis_result.difficulty_assessment = self.assess_difficulty(case_record).await?;
                }
                AnalysisType::ImprovementSuggestions => {
                    if self.config.enable_improvement_suggestions {
                        analysis_result.improvement_suggestions = self.generate_improvement_suggestions(case_record).await?;
                    }
                }
                AnalysisType::RelatedCases => {
                    if self.config.enable_related_cases {
                        analysis_result.related_cases = self.find_related_cases(case_record).await?;
                    }
                }
                AnalysisType::Comprehensive => {
                    // 执行所有分析
                    let complexity_metrics = self.analyze_complexity(&case_record.proof).await?;
                    let quality_metrics = self.analyze_quality(case_record).await?;
                    let learning_metrics = self.analyze_learning_value(case_record).await?;
                    
                    analysis_result.complexity_score = complexity_metrics.overall_complexity;
                    analysis_result.learning_value = learning_metrics.overall_learning_value;
                    analysis_result.difficulty_assessment = self.assess_difficulty(case_record).await?;
                    analysis_result.improvement_suggestions = self.generate_improvement_suggestions(case_record).await?;
                    analysis_result.related_cases = self.find_related_cases(case_record).await?;
                    analysis_result.strategy_recommendations = self.generate_strategy_recommendations(case_record).await?;
                }
            }
        }
        
        // 记录分析历史
        let execution_time = start_time.elapsed();
        self.record_analysis_history(&analysis_result, &types, execution_time).await?;
        
        // 更新性能指标
        self.update_performance_metrics(execution_time).await?;
        
        Ok(analysis_result)
    }

    /// 分析案例复杂度
    async fn analyze_complexity(&self, proof: &Proof) -> Result<ComplexityMetrics, ProofError> {
        let structural_complexity = self.calculate_structural_complexity(proof).await?;
        let logical_complexity = self.calculate_logical_complexity(proof).await?;
        let computational_complexity = self.calculate_computational_complexity(proof).await?;
        
        // 计算总体复杂度
        let overall_complexity = (structural_complexity * 0.4 + 
                                 logical_complexity * 0.4 + 
                                 computational_complexity * 0.2);
        
        // 确定复杂度级别
        let complexity_level = self.determine_complexity_level(overall_complexity);
        
        Ok(ComplexityMetrics {
            structural_complexity,
            logical_complexity,
            computational_complexity,
            overall_complexity,
            complexity_level,
        })
    }

    /// 计算结构复杂度
    async fn calculate_structural_complexity(&self, proof: &Proof) -> Result<f64, ProofError> {
        let step_count = proof.steps.len() as f64;
        let proposition_count = proof.propositions.len() as f64;
        let conclusion_complexity = self.calculate_proposition_complexity(&proof.conclusion).await?;
        
        // 基于步数、命题数量和结论复杂度的加权计算
        let complexity = (step_count * 0.4 + 
                         proposition_count * 0.3 + 
                         conclusion_complexity * 0.3) / 10.0;
        
        Ok(complexity.min(1.0))
    }

    /// 计算逻辑复杂度
    async fn calculate_logical_complexity(&self, proof: &Proof) -> Result<f64, ProofError> {
        let mut rule_variety = 0.0;
        let mut step_dependencies = 0.0;
        
        // 计算推理规则多样性
        let unique_rules: std::collections::HashSet<_> = proof.steps.iter()
            .map(|step| &step.rule_name)
            .collect();
        rule_variety = unique_rules.len() as f64 / proof.steps.len().max(1) as f64;
        
        // 计算步骤依赖关系（简化实现）
        step_dependencies = proof.steps.len() as f64 / 20.0; // 假设20步为基准
        
        let complexity = (rule_variety * 0.6 + step_dependencies * 0.4);
        Ok(complexity.min(1.0))
    }

    /// 计算计算复杂度
    async fn calculate_computational_complexity(&self, proof: &Proof) -> Result<f64, ProofError> {
        // 基于证明步骤的计算复杂度评估
        let mut total_complexity = 0.0;
        
        for step in &proof.steps {
            let step_complexity = self.calculate_step_complexity(step).await?;
            total_complexity += step_complexity;
        }
        
        let average_complexity = if proof.steps.is_empty() { 0.0 } else { total_complexity / proof.steps.len() as f64 };
        Ok(average_complexity.min(1.0))
    }

    /// 计算命题复杂度
    async fn calculate_proposition_complexity(&self, proposition: &crate::core::Proposition) -> Result<f64, ProofError> {
        // 基于命题符号数量和结构的复杂度计算
        let symbol_count = proposition.symbol.chars().count() as f64;
        let has_quantifiers = proposition.symbol.contains("∀") || proposition.symbol.contains("∃");
        let has_connectives = proposition.symbol.contains("∧") || proposition.symbol.contains("∨") || 
                             proposition.symbol.contains("→") || proposition.symbol.contains("↔");
        
        let mut complexity = symbol_count / 20.0; // 假设20个符号为基准
        
        if has_quantifiers {
            complexity += 0.2;
        }
        if has_connectives {
            complexity += 0.1;
        }
        
        Ok(complexity.min(1.0))
    }

    /// 计算步骤复杂度
    async fn calculate_step_complexity(&self, step: &ProofStep) -> Result<f64, ProofError> {
        // 基于步骤类型和内容的复杂度计算
        let rule_complexity = match step.rule_name.as_str() {
            "modus_ponens" => 0.1,
            "modus_tollens" => 0.2,
            "hypothetical_syllogism" => 0.3,
            "disjunctive_syllogism" => 0.2,
            "addition" => 0.1,
            "simplification" => 0.1,
            "conjunction" => 0.1,
            _ => 0.3, // 默认复杂度
        };
        
        let premise_count = step.premises.len() as f64;
        let premise_complexity = premise_count / 5.0; // 假设5个前提为基准
        
        let complexity = (rule_complexity * 0.7 + premise_complexity * 0.3);
        Ok(complexity.min(1.0))
    }

    /// 确定复杂度级别
    fn determine_complexity_level(&self, complexity_score: f64) -> ComplexityLevel {
        if complexity_score <= 0.25 {
            ComplexityLevel::Simple
        } else if complexity_score <= 0.5 {
            ComplexityLevel::Medium
        } else if complexity_score <= 0.75 {
            ComplexityLevel::Complex
        } else {
            ComplexityLevel::VeryComplex
        }
    }

    /// 分析案例质量
    async fn analyze_quality(&self, case_record: &CaseRecord) -> Result<QualityMetrics, ProofError> {
        let correctness = self.assess_correctness(&case_record.proof).await?;
        let completeness = self.assess_completeness(&case_record.proof).await?;
        let clarity = self.assess_clarity(case_record).await?;
        let consistency = self.assess_consistency(&case_record.proof).await?;
        let maintainability = self.assess_maintainability(case_record).await?;
        
        let overall_quality = (correctness * 0.3 + 
                              completeness * 0.25 + 
                              clarity * 0.2 + 
                              consistency * 0.15 + 
                              maintainability * 0.1);
        
        let quality_level = self.determine_quality_level(overall_quality);
        
        Ok(QualityMetrics {
            correctness,
            completeness,
            clarity,
            consistency,
            maintainability,
            overall_quality,
            quality_level,
        })
    }

    /// 评估正确性
    async fn assess_correctness(&self, proof: &Proof) -> Result<f64, ProofError> {
        // 基于证明状态和步骤正确性的评估
        let status_score = match proof.status {
            crate::core::ProofStatus::Completed => 1.0,
            crate::core::ProofStatus::Pending => 0.5,
            crate::core::ProofStatus::Failed => 0.0,
            _ => 0.3,
        };
        
        // 步骤正确性评估（简化实现）
        let step_correctness = if proof.steps.is_empty() { 0.0 } else { 0.8 };
        
        let correctness = (status_score * 0.7 + step_correctness * 0.3);
        Ok(correctness)
    }

    /// 评估完整性
    async fn assess_completeness(&self, proof: &Proof) -> Result<f64, ProofError> {
        // 基于证明步骤数量和逻辑完整性的评估
        let step_completeness = if proof.steps.len() >= 3 { 1.0 } else { proof.steps.len() as f64 / 3.0 };
        let proposition_coverage = if proof.propositions.is_empty() { 0.0 } else { 0.8 };
        
        let completeness = (step_completeness * 0.6 + proposition_coverage * 0.4);
        Ok(completeness)
    }

    /// 评估清晰性
    async fn assess_clarity(&self, case_record: &CaseRecord) -> Result<f64, ProofError> {
        // 基于描述和标签的清晰性评估
        let description_clarity = if case_record.metadata.description.len() > 50 { 1.0 } else { 0.6 };
        let tag_clarity = if case_record.metadata.tags.len() >= 2 { 1.0 } else { case_record.metadata.tags.len() as f64 / 2.0 };
        
        let clarity = (description_clarity * 0.7 + tag_clarity * 0.3);
        Ok(clarity)
    }

    /// 评估一致性
    async fn assess_consistency(&self, proof: &Proof) -> Result<f64, ProofError> {
        // 基于证明步骤逻辑一致性的评估
        let mut consistency_score = 1.0;
        
        // 检查步骤之间的逻辑一致性（简化实现）
        if proof.steps.len() > 1 {
            // 假设有多个步骤时一致性较高
            consistency_score = 0.9;
        }
        
        Ok(consistency_score)
    }

    /// 评估可维护性
    async fn assess_maintainability(&self, case_record: &CaseRecord) -> Result<f64, ProofError> {
        // 基于版本历史和元数据的可维护性评估
        let version_history_score = if case_record.version_history.len() > 1 { 1.0 } else { 0.5 };
        let metadata_completeness = if case_record.metadata.learning_objectives.len() > 0 { 1.0 } else { 0.6 };
        
        let maintainability = (version_history_score * 0.6 + metadata_completeness * 0.4);
        Ok(maintainability)
    }

    /// 确定质量级别
    fn determine_quality_level(&self, quality_score: f64) -> QualityLevel {
        if quality_score <= 0.25 {
            QualityLevel::Poor
        } else if quality_score <= 0.5 {
            QualityLevel::Fair
        } else if quality_score <= 0.75 {
            QualityLevel::Good
        } else {
            QualityLevel::Excellent
        }
    }

    /// 分析学习价值
    async fn analyze_learning_value(&self, case_record: &CaseRecord) -> Result<LearningValueMetrics, ProofError> {
        let educational_value = self.assess_educational_value(case_record).await?;
        let skill_development = self.assess_skill_development(case_record).await?;
        let knowledge_transfer = self.assess_knowledge_transfer(case_record).await?;
        let practical_applicability = self.assess_practical_applicability(case_record).await?;
        
        let overall_learning_value = (educational_value * 0.3 + 
                                     skill_development * 0.25 + 
                                     knowledge_transfer * 0.25 + 
                                     practical_applicability * 0.2);
        
        let learning_level = self.determine_learning_level(overall_learning_value);
        
        Ok(LearningValueMetrics {
            educational_value,
            skill_development,
            knowledge_transfer,
            practical_applicability,
            overall_learning_value,
            learning_level,
        })
    }

    /// 评估教育价值
    async fn assess_educational_value(&self, case_record: &CaseRecord) -> Result<f64, ProofError> {
        let learning_objectives_count = case_record.metadata.learning_objectives.len() as f64;
        let difficulty_factor = match case_record.metadata.difficulty {
            DifficultyLevel::Beginner => 0.8,
            DifficultyLevel::Intermediate => 1.0,
            DifficultyLevel::Advanced => 0.9,
            DifficultyLevel::Expert => 0.7,
        };
        
        let educational_value = (learning_objectives_count / 3.0).min(1.0) * difficulty_factor;
        Ok(educational_value)
    }

    /// 评估技能发展
    async fn assess_skill_development(&self, case_record: &CaseRecord) -> Result<f64, ProofError> {
        let step_count = case_record.proof.steps.len() as f64;
        let rule_variety = case_record.proof.steps.iter()
            .map(|step| &step.rule_name)
            .collect::<std::collections::HashSet<_>>()
            .len() as f64;
        
        let skill_development = (step_count / 10.0).min(1.0) * 0.6 + (rule_variety / 5.0).min(1.0) * 0.4;
        Ok(skill_development)
    }

    /// 评估知识转移
    async fn assess_knowledge_transfer(&self, case_record: &CaseRecord) -> Result<f64, ProofError> {
        let prerequisites_count = case_record.metadata.prerequisites.len() as f64;
        let references_count = case_record.metadata.references.len() as f64;
        
        let knowledge_transfer = (prerequisites_count / 2.0).min(1.0) * 0.5 + (references_count / 3.0).min(1.0) * 0.5;
        Ok(knowledge_transfer)
    }

    /// 评估实际应用性
    async fn assess_practical_applicability(&self, case_record: &CaseRecord) -> Result<f64, ProofError> {
        let category_factor = match case_record.metadata.tags.first().map(|tag| &tag.category) {
            Some(CaseCategory::ProgramVerification) => 1.0,
            Some(CaseCategory::SystemVerification) => 0.9,
            Some(CaseCategory::ProtocolVerification) => 0.8,
            Some(CaseCategory::MathematicalProof) => 0.7,
            Some(CaseCategory::LogicalReasoning) => 0.6,
            _ => 0.5,
        };
        
        let practical_applicability = category_factor * 0.8 + 0.2; // 基础分数0.2
        Ok(practical_applicability)
    }

    /// 确定学习级别
    fn determine_learning_level(&self, learning_score: f64) -> LearningLevel {
        if learning_score <= 0.25 {
            LearningLevel::Basic
        } else if learning_score <= 0.5 {
            LearningLevel::Intermediate
        } else if learning_score <= 0.75 {
            LearningLevel::Advanced
        } else {
            LearningLevel::Expert
        }
    }

    /// 评估难度
    async fn assess_difficulty(&self, case_record: &CaseRecord) -> Result<DifficultyLevel, ProofError> {
        // 基于复杂度分数、学习目标和案例类型的综合难度评估
        let complexity_factor = case_record.proof.steps.len() as f64 / 20.0; // 假设20步为高难度基准
        let learning_objectives_factor = case_record.metadata.learning_objectives.len() as f64 / 5.0;
        
        let difficulty_score = (complexity_factor * 0.6 + learning_objectives_factor * 0.4).min(1.0);
        
        let difficulty = if difficulty_score <= 0.25 {
            DifficultyLevel::Beginner
        } else if difficulty_score <= 0.5 {
            DifficultyLevel::Intermediate
        } else if difficulty_score <= 0.75 {
            DifficultyLevel::Advanced
        } else {
            DifficultyLevel::Expert
        };
        
        Ok(difficulty)
    }

    /// 生成改进建议
    async fn generate_improvement_suggestions(&self, case_record: &CaseRecord) -> Result<Vec<String>, ProofError> {
        let mut suggestions = Vec::new();
        
        // 基于复杂度生成建议
        if case_record.proof.steps.len() > 15 {
            suggestions.push("考虑将复杂证明分解为多个子证明，提高可读性".to_string());
        }
        
        // 基于学习目标生成建议
        if case_record.metadata.learning_objectives.is_empty() {
            suggestions.push("添加明确的学习目标，帮助学习者理解案例价值".to_string());
        }
        
        // 基于标签生成建议
        if case_record.metadata.tags.len() < 2 {
            suggestions.push("增加更多相关标签，提高案例的可发现性".to_string());
        }
        
        // 基于描述生成建议
        if case_record.metadata.description.len() < 100 {
            suggestions.push("丰富案例描述，提供更多背景信息和上下文".to_string());
        }
        
        Ok(suggestions)
    }

    /// 查找相关案例
    async fn find_related_cases(&self, case_record: &CaseRecord) -> Result<Vec<String>, ProofError> {
        // 基于相似性查找相关案例（简化实现）
        let mut related_cases = Vec::new();
        
        // 这里应该实现实际的相似性计算和案例查找
        // 目前返回空列表
        
        Ok(related_cases)
    }

    /// 生成策略建议
    async fn generate_strategy_recommendations(&self, case_record: &CaseRecord) -> Result<Vec<String>, ProofError> {
        let mut recommendations = Vec::new();
        
        // 基于难度生成策略建议
        match case_record.metadata.difficulty {
            DifficultyLevel::Beginner => {
                recommendations.push("建议使用自动证明策略，逐步引导学习者".to_string());
            }
            DifficultyLevel::Intermediate => {
                recommendations.push("建议使用混合证明策略，平衡自动化和交互性".to_string());
            }
            DifficultyLevel::Advanced => {
                recommendations.push("建议使用交互式证明策略，允许深度探索".to_string());
            }
            DifficultyLevel::Expert => {
                recommendations.push("建议使用学习模式，从专家案例中学习新策略".to_string());
            }
        }
        
        // 基于复杂度生成策略建议
        if case_record.proof.steps.len() > 20 {
            recommendations.push("考虑使用分阶段证明策略，逐步构建复杂证明".to_string());
        }
        
        Ok(recommendations)
    }

    /// 记录分析历史
    async fn record_analysis_history(
        &self,
        result: &CaseAnalysisResult,
        analysis_types: &[AnalysisType],
        execution_time: Duration,
    ) -> Result<(), ProofError> {
        let mut history = self.analysis_history.lock().unwrap();
        
        let record = AnalysisRecord {
            analysis_id: uuid::Uuid::new_v4().to_string(),
            case_id: result.case_id.clone(),
            analysis_type: analysis_types.first().unwrap_or(&AnalysisType::Comprehensive).clone(),
            timestamp: Instant::now(),
            execution_time,
            result: result.clone(),
        };
        
        history.push_back(record);
        
        // 限制历史记录数量
        if history.len() > 1000 {
            history.pop_front();
        }
        
        Ok(())
    }

    /// 更新性能指标
    async fn update_performance_metrics(&self, execution_time: Duration) -> Result<(), ProofError> {
        let mut metrics = self.performance_metrics.lock().unwrap();
        
        metrics.total_analyses += 1;
        metrics.average_execution_time = 
            (metrics.average_execution_time + execution_time) / 2;
        
        Ok(())
    }

    /// 获取性能指标
    pub fn get_performance_metrics(&self) -> AnalysisPerformanceMetrics {
        let metrics = self.performance_metrics.lock().unwrap();
        metrics.clone()
    }

    /// 获取分析历史
    pub fn get_analysis_history(&self) -> Vec<AnalysisRecord> {
        let history = self.analysis_history.lock().unwrap();
        history.iter().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cases::{CaseRecord, CaseMetadata, CaseTag, CaseCategory, DifficultyLevel};

    fn create_test_case_record() -> CaseRecord {
        CaseRecord {
            case_id: "test_case".to_string(),
            proof: crate::core::Proof {
                id: "test_proof".to_string(),
                name: "测试证明".to_string(),
                description: "这是一个测试证明".to_string(),
                status: crate::core::ProofStatus::Pending,
                propositions: Vec::new(),
                steps: Vec::new(),
                conclusion: crate::core::Proposition::new("结论", "测试结论"),
                metadata: HashMap::new(),
            },
            metadata: CaseMetadata {
                title: "测试案例".to_string(),
                description: "这是一个测试案例".to_string(),
                author: "测试作者".to_string(),
                created_at: Instant::now(),
                updated_at: Instant::now(),
                version: "1.0.0".to_string(),
                tags: vec![
                    CaseTag {
                        name: "基础".to_string(),
                        description: "基础案例".to_string(),
                        category: CaseCategory::LogicalReasoning,
                        color: "#FF0000".to_string(),
                    }
                ],
                difficulty: DifficultyLevel::Beginner,
                estimated_time: Duration::from_secs(300),
                prerequisites: Vec::new(),
                learning_objectives: Vec::new(),
                references: Vec::new(),
            },
            usage_count: 0,
            last_used: None,
            rating: 0.0,
            comments: Vec::new(),
            version_history: Vec::new(),
            is_active: true,
            is_archived: false,
        }
    }

    #[tokio::test]
    async fn test_case_analyzer_creation() {
        let config = AnalysisConfig::default();
        let analyzer = CaseAnalyzer::new(config);
        
        assert_eq!(analyzer.config.enable_complexity_analysis, true);
    }

    #[tokio::test]
    async fn test_complexity_analysis() {
        let config = AnalysisConfig::default();
        let analyzer = CaseAnalyzer::new(config);
        
        let case_record = create_test_case_record();
        let result = analyzer.analyze_case(&case_record, Some(vec![AnalysisType::Complexity])).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.complexity_score >= 0.0 && result.complexity_score <= 1.0);
    }

    #[tokio::test]
    async fn test_quality_analysis() {
        let config = AnalysisConfig::default();
        let analyzer = CaseAnalyzer::new(config);
        
        let case_record = create_test_case_record();
        let result = analyzer.analyze_case(&case_record, Some(vec![AnalysisType::Quality])).await;
        
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_comprehensive_analysis() {
        let config = AnalysisConfig::default();
        let analyzer = CaseAnalyzer::new(config);
        
        let case_record = create_test_case_record();
        let result = analyzer.analyze_case(&case_record, Some(vec![AnalysisType::Comprehensive])).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.improvement_suggestions.is_empty());
    }
}
