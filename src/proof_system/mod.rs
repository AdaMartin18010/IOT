//! 形式化证明系统
//! 
//! 本模块提供了完整的IoT系统形式化证明体系，包括：
//! - 证明框架核心
//! - 证明策略系统
//! - 证明验证系统
//! - 证明自动化系统
//! - 证明案例库

pub mod core;
#[cfg(feature = "automation")]
pub mod automation;
pub mod strategies;
pub mod verification;

pub use core::*;
pub use strategies::ProofStrategy;

/// 形式化证明系统主结构
pub struct FormalProofSystem {
    /// 证明框架
    framework: Box<dyn ProofFramework>,
    /// 证明策略
    strategies: Vec<Box<dyn ProofStrategy>>,
    /// 证明验证器
    verifier: Box<dyn ProofVerifier>,
    /// 规则库
    rule_library: super::core::rule::RuleLibrary,
}

impl FormalProofSystem {
    /// 创建新的形式化证明系统
    pub fn new() -> Self {
        Self {
            framework: Box::new(DefaultProofFramework::new()),
            strategies: Vec::new(),
            verifier: Box::new(ProofVerifierImpl::new()),
            rule_library: super::core::rule::RuleLibrary::new(),
        }
    }
    
    /// 添加证明策略
    pub fn add_strategy(&mut self, strategy: Box<dyn ProofStrategy>) {
        self.strategies.push(strategy);
    }
    
    /// 设置证明框架
    pub fn set_framework(&mut self, framework: Box<dyn ProofFramework>) {
        self.framework = framework;
    }
    
    /// 设置证明验证器
    pub fn set_verifier(&mut self, verifier: Box<dyn ProofVerifier>) {
        self.verifier = verifier;
    }
    
    /// 创建新证明
    pub fn create_proof(&mut self, goal: Proposition) -> Result<ProofId, ProofError> {
        self.framework.create_proof(goal)
    }
    
    /// 添加证明步骤
    pub fn add_step(&mut self, proof_id: ProofId, step: ProofStep) -> Result<StepId, ProofError> {
        self.framework.add_step(proof_id, step)
    }
    
    /// 验证证明步骤
    pub fn verify_step(&mut self, proof_id: ProofId, step_id: StepId) -> Result<bool, ProofError> {
        self.framework.verify_step(proof_id, step_id)
    }
    
    /// 完成证明
    pub fn complete_proof(&mut self, proof_id: ProofId) -> Result<bool, ProofError> {
        self.framework.complete_proof(proof_id)
    }
    
    /// 获取证明状态
    pub fn get_proof_status(&self, proof_id: ProofId) -> Result<ProofStatus, ProofError> {
        self.framework.get_proof_status(proof_id)
    }
    
    /// 应用证明策略
    pub fn apply_strategy(&mut self, proof_id: ProofId, strategy_name: &str) -> Result<Vec<ProofStep>, ProofError> {
        // 查找策略
        let strategy = self.strategies
            .iter()
            .find(|s| s.name() == strategy_name)
            .ok_or_else(|| ProofError::InternalError(format!("策略 {} 不存在", strategy_name)))?;
        
        // 获取证明
        let mut proof = self.get_proof(proof_id)?;
        
        // 应用策略
        let new_steps = strategy.apply(&mut proof)?;
        
        // 更新证明
        for step in &new_steps {
            self.framework.add_step(proof_id, step.clone())?;
        }
        
        Ok(new_steps)
    }
    
    /// 验证证明
    pub fn verify_proof(&self, proof_id: ProofId) -> Result<VerificationReport, ProofError> {
        let proof = self.get_proof(proof_id)?;
        self.verifier.generate_report(&proof)
    }
    
    /// 获取证明（转发到框架）
    fn get_proof(&self, proof_id: ProofId) -> Result<Proof, ProofError> {
        self.framework.get_proof(proof_id)
    }
}

/// 默认证明框架实现
pub struct DefaultProofFramework {
    /// 证明存储
    proofs: std::collections::HashMap<ProofId, Proof>,
    /// 下一个证明ID
    next_proof_id: ProofId,
}

impl DefaultProofFramework {
    /// 创建新的默认证明框架
    pub fn new() -> Self {
        Self {
            proofs: std::collections::HashMap::new(),
            next_proof_id: 1,
        }
    }
}

impl ProofFramework for DefaultProofFramework {
    fn create_proof(&mut self, goal: Proposition) -> Result<ProofId, ProofError> {
        let proof_id = self.next_proof_id;
        let proof = Proof::new(proof_id, format!("证明_{}", proof_id), goal);
        
        self.proofs.insert(proof_id, proof);
        self.next_proof_id += 1;
        
        Ok(proof_id)
    }
    
    fn add_step(&mut self, proof_id: ProofId, step: ProofStep) -> Result<StepId, ProofError> {
        let proof = self.proofs.get_mut(&proof_id)
            .ok_or_else(|| ProofError::ProofNotFound(proof_id))?;
        
        let step_id = step.id;
        proof.add_step(step)?;
        Ok(step_id)
    }
    
    fn verify_step(&mut self, proof_id: ProofId, step_id: StepId) -> Result<bool, ProofError> {
        let proof = self.proofs.get(&proof_id)
            .ok_or_else(|| ProofError::ProofNotFound(proof_id))?;
        
        let step = proof.get_step(step_id)
            .ok_or_else(|| ProofError::StepNotFound(step_id))?;
        
        // 简单的验证：检查步骤是否有效
        Ok(step.is_valid())
    }
    
    fn complete_proof(&mut self, proof_id: ProofId) -> Result<bool, ProofError> {
        let proof = self.proofs.get_mut(&proof_id)
            .ok_or_else(|| ProofError::ProofNotFound(proof_id))?;
        
        proof.complete();
        Ok(true)
    }
    
    fn get_proof_status(&self, proof_id: ProofId) -> Result<ProofStatus, ProofError> {
        let proof = self.proofs.get(&proof_id)
            .ok_or_else(|| ProofError::ProofNotFound(proof_id))?;
        
        Ok(proof.status.clone())
    }

    fn get_proof(&self, proof_id: ProofId) -> Result<Proof, ProofError> {
        self.proofs.get(&proof_id)
            .cloned()
            .ok_or_else(|| ProofError::ProofNotFound(proof_id))
    }
}

/// 证明系统构建器
pub struct ProofSystemBuilder {
    system: FormalProofSystem,
}

impl ProofSystemBuilder {
    /// 创建新的证明系统构建器
    pub fn new() -> Self {
        Self {
            system: FormalProofSystem::new(),
        }
    }
    
    /// 添加自动证明策略
    pub fn with_automated_strategy(mut self) -> Self {
        let strategy = Box::new(super::strategies::AutomatedProofStrategy::new(
            super::strategies::StrategyConfig::default(),
        ));
        self.system.add_strategy(strategy);
        self
    }
    
    /// 添加交互式证明策略
    pub fn with_interactive_strategy(mut self) -> Self {
        let strategy = Box::new(super::strategies::InteractiveProofStrategy::new(
            super::strategies::StrategyConfig::default(),
        ));
        self.system.add_strategy(strategy);
        self
    }
    
    /// 添加混合证明策略
    pub fn with_hybrid_strategy(mut self) -> Self {
        let strategy = Box::new(super::strategies::HybridProofStrategy::new(
            super::strategies::StrategyConfig::default(),
        ));
        self.system.add_strategy(strategy);
        self
    }
    
    /// 设置自定义验证器
    pub fn with_custom_verifier(mut self, verifier: Box<dyn ProofVerifier>) -> Self {
        self.system.set_verifier(verifier);
        self
    }
    
    /// 构建证明系统
    pub fn build(self) -> FormalProofSystem {
        self.system
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::core::{Proposition, PropositionType};
    use std::collections::HashMap;
    
    fn create_test_proposition(id: &str, content: &str) -> Proposition {
        Proposition {
            id: id.to_string(),
            content: content.to_string(),
            proposition_type: PropositionType::Theorem,
            metadata: HashMap::new(),
        }
    }
    
    #[test]
    fn test_formal_proof_system_creation() {
        let system = FormalProofSystem::new();
        assert_eq!(system.strategies.len(), 0);
    }
    
    #[test]
    fn test_proof_system_builder() {
        let system = ProofSystemBuilder::new()
            .with_automated_strategy()
            .with_interactive_strategy()
            .with_hybrid_strategy()
            .build();
        
        assert_eq!(system.strategies.len(), 3);
    }
    
    #[test]
    fn test_default_proof_framework() {
        let mut framework = DefaultProofFramework::new();
        let goal = create_test_proposition("goal", "A ∧ B → B ∧ A");
        
        let proof_id = framework.create_proof(goal).unwrap();
        assert_eq!(proof_id, 1);
        
        let status = framework.get_proof_status(proof_id).unwrap();
        assert_eq!(status, ProofStatus::Creating);
    }
}
