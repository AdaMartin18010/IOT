//! 形式化证明系统核心模块
//! 
//! 本模块定义了形式化证明系统的核心接口和数据结构，
//! 包括证明、规则、步骤等基础概念。

pub mod proof;
pub mod rule;
pub mod step;
pub mod strategy;
pub mod verifier;

pub use proof::*;
pub use rule::*;
pub use step::*;
pub use strategy::*;
pub use verifier::*;

/// 形式化证明系统核心特征
pub trait ProofFramework {
    /// 创建新的证明
    fn create_proof(&mut self, goal: Proposition) -> Result<ProofId, ProofError>;
    
    /// 添加证明步骤
    fn add_step(&mut self, proof_id: ProofId, step: ProofStep) -> Result<StepId, ProofError>;
    
    /// 验证证明步骤
    fn verify_step(&mut self, proof_id: ProofId, step_id: StepId) -> Result<bool, ProofError>;
    
    /// 完成证明
    fn complete_proof(&mut self, proof_id: ProofId) -> Result<bool, ProofError>;
    
    /// 获取证明状态
    fn get_proof_status(&self, proof_id: ProofId) -> Result<ProofStatus, ProofError>;
}

/// 证明策略特征
pub trait ProofStrategy {
    /// 策略名称
    fn name(&self) -> &str;
    
    /// 策略描述
    fn description(&self) -> &str;
    
    /// 应用策略
    fn apply(&self, proof: &mut Proof) -> Result<Vec<ProofStep>, ProofError>;
    
    /// 策略适用性检查
    fn is_applicable(&self, proof: &Proof) -> bool;
    
    /// 策略优先级
    fn priority(&self) -> u32;
}

/// 证明验证器特征
pub trait ProofVerifier {
    /// 验证证明结构
    fn verify_structure(&self, proof: &Proof) -> Result<VerificationResult, ProofError>;
    
    /// 验证证明逻辑
    fn verify_logic(&self, proof: &Proof) -> Result<VerificationResult, ProofError>;
    
    /// 验证证明完整性
    fn verify_completeness(&self, proof: &Proof) -> Result<VerificationResult, ProofError>;
    
    /// 生成验证报告
    fn generate_report(&self, proof: &Proof) -> Result<VerificationReport, ProofError>;
}

/// 证明错误类型
#[derive(Debug, thiserror::Error)]
pub enum ProofError {
    #[error("证明不存在: {0}")]
    ProofNotFound(ProofId),
    
    #[error("步骤不存在: {0}")]
    StepNotFound(StepId),
    
    #[error("规则不存在: {0}")]
    RuleNotFound(RuleId),
    
    #[error("验证失败: {0}")]
    VerificationFailed(String),
    
    #[error("逻辑错误: {0}")]
    LogicError(String),
    
    #[error("语法错误: {0}")]
    SyntaxError(String),
    
    #[error("内部错误: {0}")]
    InternalError(String),
}

/// 证明ID类型
pub type ProofId = u64;
/// 步骤ID类型
pub type StepId = u64;
/// 规则ID类型
pub type RuleId = u64;

/// 证明状态
#[derive(Debug, Clone, PartialEq)]
pub enum ProofStatus {
    /// 创建中
    Creating,
    /// 进行中
    InProgress,
    /// 验证中
    Verifying,
    /// 已完成
    Completed,
    /// 失败
    Failed,
    /// 已撤销
    Revoked,
}

/// 验证结果
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// 是否成功
    pub success: bool,
    /// 错误信息
    pub errors: Vec<String>,
    /// 警告信息
    pub warnings: Vec<String>,
    /// 验证时间
    pub verification_time: std::time::Duration,
}

/// 验证报告
#[derive(Debug, Clone)]
pub struct VerificationReport {
    /// 报告ID
    pub report_id: String,
    /// 验证结果
    pub result: VerificationResult,
    /// 详细说明
    pub details: String,
    /// 生成时间
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

/// 命题类型
#[derive(Debug, Clone, PartialEq)]
pub struct Proposition {
    /// 命题ID
    pub id: String,
    /// 命题内容
    pub content: String,
    /// 命题类型
    pub proposition_type: PropositionType,
    /// 元数据
    pub metadata: std::collections::HashMap<String, String>,
}

/// 命题类型
#[derive(Debug, Clone, PartialEq)]
pub enum PropositionType {
    /// 公理
    Axiom,
    /// 定理
    Theorem,
    /// 引理
    Lemma,
    /// 推论
    Corollary,
    /// 假设
    Hypothesis,
    /// 结论
    Conclusion,
}
