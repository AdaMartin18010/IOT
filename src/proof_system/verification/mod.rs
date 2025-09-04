pub mod verifier;
pub mod checker;
pub mod analyzer;
pub mod reporter;

pub use verifier::*;
pub use checker::*;
pub use analyzer::*;
pub use reporter::*;

use crate::core::{Proof, ProofStep, InferenceRule, ProofError, ProofStatus, Proposition};
use std::collections::HashMap;

/// 验证级别
#[derive(Debug, Clone, PartialEq)]
pub enum VerificationLevel {
    /// 基础验证：检查语法和基本结构
    Basic,
    /// 标准验证：检查逻辑一致性和规则应用
    Standard,
    /// 严格验证：检查完整性和正确性
    Strict,
    /// 专家验证：深度分析和形式化检查
    Expert,
}

/// 验证结果
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// 是否通过验证
    pub passed: bool,
    /// 验证级别
    pub level: VerificationLevel,
    /// 验证时间（毫秒）
    pub verification_time_ms: u64,
    /// 发现的错误
    pub errors: Vec<VerificationError>,
    /// 发现的警告
    pub warnings: Vec<VerificationWarning>,
    /// 验证建议
    pub suggestions: Vec<String>,
    /// 详细报告
    pub detailed_report: Option<String>,
}

/// 验证错误
#[derive(Debug, Clone)]
pub struct VerificationError {
    /// 错误类型
    pub error_type: VerificationErrorType,
    /// 错误位置
    pub location: ErrorLocation,
    /// 错误描述
    pub description: String,
    /// 严重程度
    pub severity: ErrorSeverity,
    /// 修复建议
    pub fix_suggestion: Option<String>,
}

/// 验证警告
#[derive(Debug, Clone)]
pub struct VerificationWarning {
    /// 警告类型
    pub warning_type: VerificationWarningType,
    /// 警告位置
    pub location: ErrorLocation,
    /// 警告描述
    pub description: String,
    /// 建议操作
    pub suggestion: String,
}

/// 错误类型
#[derive(Debug, Clone, PartialEq)]
pub enum VerificationErrorType {
    /// 语法错误
    SyntaxError,
    /// 逻辑错误
    LogicalError,
    /// 结构错误
    StructuralError,
    /// 规则应用错误
    RuleApplicationError,
    /// 完整性错误
    CompletenessError,
    /// 一致性错误
    ConsistencyError,
}

/// 警告类型
#[derive(Debug, Clone, PartialEq)]
pub enum VerificationWarningType {
    /// 性能警告
    PerformanceWarning,
    /// 风格警告
    StyleWarning,
    /// 最佳实践警告
    BestPracticeWarning,
    /// 可读性警告
    ReadabilityWarning,
}

/// 错误位置
#[derive(Debug, Clone)]
pub struct ErrorLocation {
    /// 步骤ID
    pub step_id: Option<u64>,
    /// 规则ID
    pub rule_id: Option<u64>,
    /// 命题ID
    pub proposition_id: Option<u64>,
    /// 行号
    pub line_number: Option<usize>,
    /// 列号
    pub column_number: Option<usize>,
}

/// 错误严重程度
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum ErrorSeverity {
    /// 信息
    Info,
    /// 警告
    Warning,
    /// 错误
    Error,
    /// 严重错误
    Critical,
}

/// 验证配置
#[derive(Debug, Clone)]
pub struct VerificationConfig {
    /// 验证级别
    pub level: VerificationLevel,
    /// 是否启用详细报告
    pub enable_detailed_report: bool,
    /// 是否启用性能分析
    pub enable_performance_analysis: bool,
    /// 是否启用自动修复建议
    pub enable_auto_fix_suggestions: bool,
    /// 最大验证时间（毫秒）
    pub max_verification_time_ms: u64,
    /// 是否并行验证
    pub enable_parallel_verification: bool,
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            level: VerificationLevel::Standard,
            enable_detailed_report: true,
            enable_performance_analysis: true,
            enable_auto_fix_suggestions: true,
            max_verification_time_ms: 60000, // 60秒
            enable_parallel_verification: true,
        }
    }
}

/// 验证统计
#[derive(Debug, Clone)]
pub struct VerificationStats {
    /// 总验证次数
    pub total_verifications: u64,
    /// 成功验证次数
    pub successful_verifications: u64,
    /// 失败验证次数
    pub failed_verifications: u64,
    /// 平均验证时间
    pub avg_verification_time_ms: u64,
    /// 总错误数
    pub total_errors: u64,
    /// 总警告数
    pub total_warnings: u64,
    /// 最常见错误类型
    pub most_common_error_type: Option<VerificationErrorType>,
}

impl Default for VerificationStats {
    fn default() -> Self {
        Self {
            total_verifications: 0,
            successful_verifications: 0,
            failed_verifications: 0,
            avg_verification_time_ms: 0,
            total_errors: 0,
            total_warnings: 0,
            most_common_error_type: None,
        }
    }
}
