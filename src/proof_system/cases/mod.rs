pub mod manager;
pub mod searcher;
pub mod analyzer;
pub mod generator;

pub use manager::*;
pub use searcher::*;
pub use analyzer::*;
pub use generator::*;

use crate::core::{Proof, ProofStep, InferenceRule, ProofError, ProofStatus, Proposition};
use crate::strategies::{StrategyExecutionResult, StrategyPerformanceMetrics, StrategyConfig};
use std::collections::HashMap;
use std::time::{Instant, Duration};
use serde::{Serialize, Deserialize};

/// 案例类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CaseType {
    /// 基础案例
    Basic,
    /// 进阶案例
    Advanced,
    /// 复杂案例
    Complex,
    /// 教学案例
    Educational,
    /// 研究案例
    Research,
    /// 自定义案例
    Custom(String),
}

/// 案例难度级别
#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum DifficultyLevel {
    /// 初级
    Beginner,
    /// 中级
    Intermediate,
    /// 高级
    Advanced,
    /// 专家级
    Expert,
}

/// 案例分类
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CaseCategory {
    /// 逻辑推理
    LogicalReasoning,
    /// 数学证明
    MathematicalProof,
    /// 程序验证
    ProgramVerification,
    /// 系统验证
    SystemVerification,
    /// 协议验证
    ProtocolVerification,
    /// 其他
    Other,
}

/// 案例标签
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseTag {
    pub name: String,
    pub description: String,
    pub category: CaseCategory,
    pub color: String,
}

/// 案例元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseMetadata {
    pub title: String,
    pub description: String,
    pub author: String,
    pub created_at: Instant,
    pub updated_at: Instant,
    pub version: String,
    pub tags: Vec<CaseTag>,
    pub difficulty: DifficultyLevel,
    pub estimated_time: Duration,
    pub prerequisites: Vec<String>,
    pub learning_objectives: Vec<String>,
    pub references: Vec<String>,
}

/// 案例统计信息
#[derive(Debug, Clone, Default)]
pub struct CaseStatistics {
    pub total_cases: usize,
    pub cases_by_type: HashMap<CaseType, usize>,
    pub cases_by_difficulty: HashMap<DifficultyLevel, usize>,
    pub cases_by_category: HashMap<CaseCategory, usize>,
    pub average_completion_time: Duration,
    pub success_rate: f64,
    pub popularity_score: f64,
}

/// 案例搜索条件
#[derive(Debug, Clone)]
pub struct CaseSearchCriteria {
    pub case_type: Option<CaseType>,
    pub difficulty_level: Option<DifficultyLevel>,
    pub category: Option<CaseCategory>,
    pub tags: Vec<String>,
    pub author: Option<String>,
    pub time_range: Option<(Instant, Instant)>,
    pub min_success_rate: Option<f64>,
    pub max_completion_time: Option<Duration>,
    pub keywords: Vec<String>,
}

/// 案例搜索结果
#[derive(Debug, Clone)]
pub struct CaseSearchResult {
    pub case_id: String,
    pub title: String,
    pub description: String,
    pub relevance_score: f64,
    pub difficulty: DifficultyLevel,
    pub estimated_time: Duration,
    pub success_rate: f64,
    pub tags: Vec<CaseTag>,
    pub metadata: CaseMetadata,
}

/// 案例分析结果
#[derive(Debug, Clone)]
pub struct CaseAnalysisResult {
    pub case_id: String,
    pub complexity_score: f64,
    pub difficulty_assessment: DifficultyLevel,
    pub learning_value: f64,
    pub common_mistakes: Vec<String>,
    pub improvement_suggestions: Vec<String>,
    pub related_cases: Vec<String>,
    pub strategy_recommendations: Vec<String>,
}

/// 案例生成配置
#[derive(Debug, Clone)]
pub struct CaseGenerationConfig {
    pub target_difficulty: DifficultyLevel,
    pub target_type: CaseType,
    pub target_category: CaseCategory,
    pub complexity_range: (f64, f64),
    pub step_count_range: (usize, usize),
    pub include_prerequisites: bool,
    pub learning_objectives: Vec<String>,
    pub custom_constraints: HashMap<String, String>,
}

/// 案例生成结果
#[derive(Debug, Clone)]
pub struct CaseGenerationResult {
    pub case_id: String,
    pub proof: Proof,
    pub metadata: CaseMetadata,
    pub generation_quality: f64,
    pub validation_result: CaseValidationResult,
    pub difficulty_calibration: DifficultyLevel,
}

/// 案例验证结果
#[derive(Debug, Clone)]
pub struct CaseValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub suggestions: Vec<String>,
    pub quality_score: f64,
}

/// 案例库配置
#[derive(Debug, Clone)]
pub struct CaseLibraryConfig {
    pub max_cases: usize,
    pub enable_auto_cleanup: bool,
    pub backup_interval: Duration,
    pub search_index_update_interval: Duration,
    pub case_validation_enabled: bool,
    pub learning_mode_enabled: bool,
    pub collaborative_editing_enabled: bool,
}

impl Default for CaseLibraryConfig {
    fn default() -> Self {
        Self {
            max_cases: 10000,
            enable_auto_cleanup: true,
            backup_interval: Duration::from_secs(86400), // 24小时
            search_index_update_interval: Duration::from_secs(3600), // 1小时
            case_validation_enabled: true,
            learning_mode_enabled: true,
            collaborative_editing_enabled: false,
        }
    }
}

/// 案例库状态
#[derive(Debug, Clone)]
pub struct CaseLibraryStatus {
    pub total_cases: usize,
    pub active_cases: usize,
    pub archived_cases: usize,
    pub last_backup: Option<Instant>,
    pub last_index_update: Option<Instant>,
    pub storage_usage: f64,
    pub performance_metrics: CaseLibraryPerformanceMetrics,
}

/// 案例库性能指标
#[derive(Debug, Clone, Default)]
pub struct CaseLibraryPerformanceMetrics {
    pub search_response_time: Duration,
    pub case_retrieval_time: Duration,
    pub analysis_processing_time: Duration,
    pub generation_time: Duration,
    pub cache_hit_rate: f64,
    pub index_efficiency: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_case_type_creation() {
        let case_type = CaseType::Basic;
        assert_eq!(case_type, CaseType::Basic);
    }

    #[test]
    fn test_difficulty_level_ordering() {
        assert!(DifficultyLevel::Beginner < DifficultyLevel::Expert);
        assert!(DifficultyLevel::Intermediate < DifficultyLevel::Advanced);
    }

    #[test]
    fn test_case_tag_creation() {
        let tag = CaseTag {
            name: "逻辑推理".to_string(),
            description: "基础逻辑推理案例".to_string(),
            category: CaseCategory::LogicalReasoning,
            color: "#FF0000".to_string(),
        };
        
        assert_eq!(tag.name, "逻辑推理");
        assert_eq!(tag.category, CaseCategory::LogicalReasoning);
    }

    #[test]
    fn test_case_metadata_creation() {
        let metadata = CaseMetadata {
            title: "测试案例".to_string(),
            description: "这是一个测试案例".to_string(),
            author: "测试作者".to_string(),
            created_at: Instant::now(),
            updated_at: Instant::now(),
            version: "1.0.0".to_string(),
            tags: Vec::new(),
            difficulty: DifficultyLevel::Beginner,
            estimated_time: Duration::from_secs(300),
            prerequisites: Vec::new(),
            learning_objectives: Vec::new(),
            references: Vec::new(),
        };
        
        assert_eq!(metadata.title, "测试案例");
        assert_eq!(metadata.difficulty, DifficultyLevel::Beginner);
    }

    #[test]
    fn test_case_search_criteria() {
        let criteria = CaseSearchCriteria {
            case_type: Some(CaseType::Basic),
            difficulty_level: Some(DifficultyLevel::Beginner),
            category: Some(CaseCategory::LogicalReasoning),
            tags: vec!["基础".to_string()],
            author: Some("测试作者".to_string()),
            time_range: None,
            min_success_rate: Some(0.8),
            max_completion_time: Some(Duration::from_secs(600)),
            keywords: vec!["逻辑".to_string(), "推理".to_string()],
        };
        
        assert_eq!(criteria.difficulty_level, Some(DifficultyLevel::Beginner));
        assert_eq!(criteria.tags.len(), 1);
    }

    #[test]
    fn test_case_generation_config() {
        let config = CaseGenerationConfig {
            target_difficulty: DifficultyLevel::Intermediate,
            target_type: CaseType::Educational,
            target_category: CaseCategory::MathematicalProof,
            complexity_range: (0.3, 0.7),
            step_count_range: (5, 15),
            include_prerequisites: true,
            learning_objectives: vec!["理解数学归纳法".to_string()],
            custom_constraints: HashMap::new(),
        };
        
        assert_eq!(config.target_difficulty, DifficultyLevel::Intermediate);
        assert_eq!(config.step_count_range, (5, 15));
    }
}
