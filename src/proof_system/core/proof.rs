//! 证明模块
//! 
//! 本模块定义了证明的核心数据结构，包括证明、证明步骤等。

use super::{ProofId, StepId, ProofStatus, Proposition, ProofError};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// 证明结构
#[derive(Debug, Clone)]
pub struct Proof {
    /// 证明ID
    pub id: ProofId,
    /// 证明名称
    pub name: String,
    /// 证明描述
    pub description: String,
    /// 证明目标
    pub goal: Proposition,
    /// 证明前提
    pub premises: Vec<Proposition>,
    /// 证明步骤
    pub steps: HashMap<StepId, ProofStep>,
    /// 证明状态
    pub status: ProofStatus,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 更新时间
    pub updated_at: DateTime<Utc>,
    /// 完成时间
    pub completed_at: Option<DateTime<Utc>>,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

impl Proof {
    /// 创建新的证明
    pub fn new(id: ProofId, name: String, goal: Proposition) -> Self {
        let now = Utc::now();
        Self {
            id,
            name,
            description: String::new(),
            goal,
            premises: Vec::new(),
            steps: HashMap::new(),
            status: ProofStatus::Creating,
            created_at: now,
            updated_at: now,
            completed_at: None,
            metadata: HashMap::new(),
        }
    }
    
    /// 添加前提
    pub fn add_premise(&mut self, premise: Proposition) {
        self.premises.push(premise);
        self.updated_at = Utc::now();
    }
    
    /// 添加证明步骤
    pub fn add_step(&mut self, step: ProofStep) -> Result<(), ProofError> {
        let step_id = step.id;
        if self.steps.contains_key(&step_id) {
            return Err(ProofError::InternalError(format!("步骤ID {} 已存在", step_id)));
        }
        
        self.steps.insert(step_id, step);
        self.status = ProofStatus::InProgress;
        self.updated_at = Utc::now();
        Ok(())
    }
    
    /// 获取证明步骤
    pub fn get_step(&self, step_id: StepId) -> Option<&ProofStep> {
        self.steps.get(&step_id)
    }
    
    /// 获取所有步骤
    pub fn get_all_steps(&self) -> Vec<&ProofStep> {
        self.steps.values().collect()
    }
    
    /// 获取步骤数量
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }
    
    /// 检查证明是否完成
    pub fn is_completed(&self) -> bool {
        matches!(self.status, ProofStatus::Completed)
    }
    
    /// 完成证明
    pub fn complete(&mut self) {
        self.status = ProofStatus::Completed;
        self.completed_at = Some(Utc::now());
        self.updated_at = Utc::now();
    }
    
    /// 标记证明失败
    pub fn mark_failed(&mut self) {
        self.status = ProofStatus::Failed;
        self.updated_at = Utc::now();
    }
    
    /// 获取证明摘要
    pub fn get_summary(&self) -> ProofSummary {
        ProofSummary {
            id: self.id,
            name: self.name.clone(),
            status: self.status.clone(),
            step_count: self.step_count(),
            created_at: self.created_at,
            completed_at: self.completed_at,
        }
    }
}

/// 证明步骤
#[derive(Debug, Clone)]
pub struct ProofStep {
    /// 步骤ID
    pub id: StepId,
    /// 步骤序号
    pub sequence: u32,
    /// 步骤描述
    pub description: String,
    /// 步骤类型
    pub step_type: ProofStepType,
    /// 应用的规则
    pub applied_rule: Option<String>,
    /// 输入命题
    pub input_propositions: Vec<Proposition>,
    /// 输出命题
    pub output_propositions: Vec<Proposition>,
    /// 证明理由
    pub justification: String,
    /// 依赖步骤
    pub dependencies: Vec<StepId>,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

impl ProofStep {
    /// 创建新的证明步骤
    pub fn new(
        id: StepId,
        sequence: u32,
        description: String,
        step_type: ProofStepType,
    ) -> Self {
        Self {
            id,
            sequence,
            description,
            step_type,
            applied_rule: None,
            input_propositions: Vec::new(),
            output_propositions: Vec::new(),
            justification: String::new(),
            dependencies: Vec::new(),
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }
    
    /// 设置应用的规则
    pub fn with_rule(mut self, rule: String) -> Self {
        self.applied_rule = Some(rule);
        self
    }
    
    /// 添加输入命题
    pub fn with_input(mut self, proposition: Proposition) -> Self {
        self.input_propositions.push(proposition);
        self
    }
    
    /// 添加输出命题
    pub fn with_output(mut self, proposition: Proposition) -> Self {
        self.output_propositions.push(proposition);
        self
    }
    
    /// 设置证明理由
    pub fn with_justification(mut self, justification: String) -> Self {
        self.justification = justification;
        self
    }
    
    /// 添加依赖步骤
    pub fn with_dependency(mut self, step_id: StepId) -> Self {
        self.dependencies.push(step_id);
        self
    }
    
    /// 检查步骤是否有效
    pub fn is_valid(&self) -> bool {
        !self.description.is_empty() && !self.justification.is_empty()
    }
}

/// 证明步骤类型
#[derive(Debug, Clone, PartialEq)]
pub enum ProofStepType {
    /// 假设引入
    Assumption,
    /// 规则应用
    RuleApplication,
    /// 逻辑推理
    LogicalInference,
    /// 数学运算
    MathematicalOperation,
    /// 定义展开
    DefinitionExpansion,
    /// 引理应用
    LemmaApplication,
    /// 结论推导
    Conclusion,
    /// 其他
    Other(String),
}

/// 证明摘要
#[derive(Debug, Clone)]
pub struct ProofSummary {
    /// 证明ID
    pub id: ProofId,
    /// 证明名称
    pub name: String,
    /// 证明状态
    pub status: ProofStatus,
    /// 步骤数量
    pub step_count: usize,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 完成时间
    pub completed_at: Option<DateTime<Utc>>,
}

/// 证明构建器
pub struct ProofBuilder {
    proof: Proof,
    next_step_id: StepId,
    next_sequence: u32,
}

impl ProofBuilder {
    /// 创建新的证明构建器
    pub fn new(id: ProofId, name: String, goal: Proposition) -> Self {
        Self {
            proof: Proof::new(id, name, goal),
            next_step_id: 1,
            next_sequence: 1,
        }
    }
    
    /// 设置证明描述
    pub fn with_description(mut self, description: String) -> Self {
        self.proof.description = description;
        self
    }
    
    /// 添加前提
    pub fn with_premise(mut self, premise: Proposition) -> Self {
        self.proof.add_premise(premise);
        self
    }
    
    /// 添加证明步骤
    pub fn with_step(mut self, description: String, step_type: ProofStepType) -> Result<Self, ProofError> {
        let step = ProofStep::new(self.next_step_id, self.next_sequence, description, step_type);
        self.proof.add_step(step)?;
        self.next_step_id += 1;
        self.next_sequence += 1;
        Ok(self)
    }
    
    /// 构建证明
    pub fn build(self) -> Proof {
        self.proof
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_proof_creation() {
        let goal = Proposition {
            id: "goal1".to_string(),
            content: "A ∧ B → B ∧ A".to_string(),
            proposition_type: super::super::PropositionType::Theorem,
            metadata: HashMap::new(),
        };
        
        let proof = Proof::new(1, "交换律证明".to_string(), goal);
        
        assert_eq!(proof.id, 1);
        assert_eq!(proof.name, "交换律证明");
        assert_eq!(proof.status, ProofStatus::Creating);
        assert_eq!(proof.step_count(), 0);
    }
    
    #[test]
    fn test_proof_step_creation() {
        let step = ProofStep::new(
            1,
            1,
            "引入假设A".to_string(),
            ProofStepType::Assumption,
        );
        
        assert_eq!(step.id, 1);
        assert_eq!(step.sequence, 1);
        assert_eq!(step.description, "引入假设A");
        assert_eq!(step.step_type, ProofStepType::Assumption);
    }
    
    #[test]
    fn test_proof_builder() {
        let goal = Proposition {
            id: "goal1".to_string(),
            content: "A ∧ B → B ∧ A".to_string(),
            proposition_type: super::super::PropositionType::Theorem,
            metadata: HashMap::new(),
        };
        
        let proof = ProofBuilder::new(1, "交换律证明".to_string(), goal)
            .with_description("证明逻辑与的交换律".to_string())
            .with_step("引入假设A ∧ B".to_string(), ProofStepType::Assumption)
            .unwrap()
            .with_step("应用交换律".to_string(), ProofStepType::RuleApplication)
            .unwrap()
            .build();
        
        assert_eq!(proof.step_count(), 2);
        assert_eq!(proof.status, ProofStatus::InProgress);
    }
}
