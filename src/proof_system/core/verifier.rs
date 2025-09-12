//! 证明验证器模块
//! 
//! 本模块定义了证明验证的核心功能，包括结构验证、逻辑验证等。

use super::{ProofVerifier, Proof, ProofError, VerificationResult, VerificationReport};
use chrono::Utc;

/// 证明验证器实现
pub struct ProofVerifierImpl {
    /// 结构验证器
    structure_verifier: StructureVerifier,
    /// 逻辑验证器
    logic_verifier: LogicVerifier,
    /// 完整性验证器
    completeness_verifier: CompletenessVerifier,
}

impl ProofVerifierImpl {
    /// 创建新的证明验证器
    pub fn new() -> Self {
        Self {
            structure_verifier: StructureVerifier::new(),
            logic_verifier: LogicVerifier::new(),
            completeness_verifier: CompletenessVerifier::new(),
        }
    }
}

impl ProofVerifier for ProofVerifierImpl {
    fn verify_structure(&self, proof: &Proof) -> Result<VerificationResult, ProofError> {
        self.structure_verifier.verify(proof)
    }
    
    fn verify_logic(&self, proof: &Proof) -> Result<VerificationResult, ProofError> {
        self.logic_verifier.verify(proof)
    }
    
    fn verify_completeness(&self, proof: &Proof) -> Result<VerificationResult, ProofError> {
        self.completeness_verifier.verify(proof)
    }
    
    fn generate_report(&self, proof: &Proof) -> Result<VerificationReport, ProofError> {
        let start_time = std::time::Instant::now();
        
        // 执行所有验证
        let structure_result = self.verify_structure(proof)?;
        let logic_result = self.verify_logic(proof)?;
        let completeness_result = self.verify_completeness(proof)?;
        
        let verification_time = start_time.elapsed();
        
        // 合并验证结果
        let overall_success = structure_result.success && logic_result.success && completeness_result.success;
        let mut all_errors = Vec::new();
        let mut all_warnings = Vec::new();
        
        all_errors.extend(structure_result.errors.clone());
        all_errors.extend(logic_result.errors.clone());
        all_errors.extend(completeness_result.errors.clone());
        
        all_warnings.extend(structure_result.warnings.clone());
        all_warnings.extend(logic_result.warnings.clone());
        all_warnings.extend(completeness_result.warnings.clone());
        
        let overall_result = VerificationResult {
            success: overall_success,
            errors: all_errors,
            warnings: all_warnings,
            verification_time,
        };
        
        // 生成报告
        let report = VerificationReport {
            report_id: format!("VR_{}", proof.id),
            result: overall_result,
            details: self.generate_detailed_report(proof, &structure_result, &logic_result, &completeness_result),
            generated_at: Utc::now(),
        };
        
        Ok(report)
    }
}

impl ProofVerifierImpl {
    /// 生成详细报告
    fn generate_detailed_report(
        &self,
        proof: &Proof,
        structure_result: &VerificationResult,
        logic_result: &VerificationResult,
        completeness_result: &VerificationResult,
    ) -> String {
        let mut report = String::new();
        
        report.push_str(&format!("证明验证详细报告\n"));
        report.push_str(&format!("证明ID: {}\n", proof.id));
        report.push_str(&format!("证明名称: {}\n", proof.name));
        report.push_str(&format!("验证时间: {}\n\n", Utc::now()));
        
        // 结构验证结果
        report.push_str("1. 结构验证结果:\n");
        report.push_str(&format!("   状态: {}\n", if structure_result.success { "通过" } else { "失败" }));
        if !structure_result.errors.is_empty() {
            report.push_str("   错误:\n");
            for error in &structure_result.errors {
                report.push_str(&format!("     - {}\n", error));
            }
        }
        if !structure_result.warnings.is_empty() {
            report.push_str("   警告:\n");
            for warning in &structure_result.warnings {
                report.push_str(&format!("     - {}\n", warning));
            }
        }
        report.push('\n');
        
        // 逻辑验证结果
        report.push_str("2. 逻辑验证结果:\n");
        report.push_str(&format!("   状态: {}\n", if logic_result.success { "通过" } else { "失败" }));
        if !logic_result.errors.is_empty() {
            report.push_str("   错误:\n");
            for error in &logic_result.errors {
                report.push_str(&format!("     - {}\n", error));
            }
        }
        if !logic_result.warnings.is_empty() {
            report.push_str("   警告:\n");
            for warning in &logic_result.warnings {
                report.push_str(&format!("     - {}\n", warning));
            }
        }
        report.push('\n');
        
        // 完整性验证结果
        report.push_str("3. 完整性验证结果:\n");
        report.push_str(&format!("   状态: {}\n", if completeness_result.success { "通过" } else { "失败" }));
        if !completeness_result.errors.is_empty() {
            report.push_str("   错误:\n");
            for error in &completeness_result.errors {
                report.push_str(&format!("     - {}\n", error));
            }
        }
        if !completeness_result.warnings.is_empty() {
            report.push_str("   警告:\n");
            for warning in &completeness_result.warnings {
                report.push_str(&format!("     - {}\n", warning));
            }
        }
        
        report
    }
}

/// 结构验证器
pub struct StructureVerifier {
    /// 验证规则
    validation_rules: Vec<Box<dyn StructureValidationRule>>,
}

impl StructureVerifier {
    /// 创建新的结构验证器
    pub fn new() -> Self {
        let mut verifier = Self {
            validation_rules: Vec::new(),
        };
        
        // 添加默认验证规则
        verifier.add_rule(Box::new(BasicStructureRule));
        verifier.add_rule(Box::new(StepDependencyRule));
        verifier.add_rule(Box::new(PropositionConsistencyRule));
        
        verifier
    }
    
    /// 添加验证规则
    pub fn add_rule(&mut self, rule: Box<dyn StructureValidationRule>) {
        self.validation_rules.push(rule);
    }
    
    /// 验证证明结构
    pub fn verify(&self, proof: &Proof) -> Result<VerificationResult, ProofError> {
        let mut result = VerificationResult {
            success: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            verification_time: std::time::Duration::from_millis(0),
        };
        
        let start_time = std::time::Instant::now();
        
        // 应用所有验证规则
        for rule in &self.validation_rules {
            let rule_result = rule.validate(proof);
            result.merge(rule_result);
        }
        
        result.verification_time = start_time.elapsed();
        Ok(result)
    }
}

/// 结构验证规则特征
pub trait StructureValidationRule {
    /// 验证规则名称
    fn name(&self) -> &str;
    
    /// 应用验证规则
    fn validate(&self, proof: &Proof) -> VerificationResult;
}

/// 基本结构规则
pub struct BasicStructureRule;

impl StructureValidationRule for BasicStructureRule {
    fn name(&self) -> &str {
        "基本结构规则"
    }
    
    fn validate(&self, proof: &Proof) -> VerificationResult {
        let mut result = VerificationResult {
            success: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            verification_time: std::time::Duration::from_millis(0),
        };
        
        // 检查证明ID
        if proof.id == 0 {
            result.errors.push("证明ID不能为0".to_string());
            result.success = false;
        }
        
        // 检查证明名称
        if proof.name.is_empty() {
            result.errors.push("证明名称不能为空".to_string());
            result.success = false;
        }
        
        // 检查证明目标
        if proof.goal.content.is_empty() {
            result.errors.push("证明目标不能为空".to_string());
            result.success = false;
        }
        
        // 检查步骤数量
        if proof.steps.is_empty() && proof.status != super::ProofStatus::Creating {
            result.warnings.push("证明没有步骤".to_string());
        }
        
        result
    }
}

/// 步骤依赖规则
pub struct StepDependencyRule;

impl StructureValidationRule for StepDependencyRule {
    fn name(&self) -> &str {
        "步骤依赖规则"
    }
    
    fn validate(&self, proof: &Proof) -> VerificationResult {
        let mut result = VerificationResult {
            success: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            verification_time: std::time::Duration::from_millis(0),
        };
        
        let step_ids: std::collections::HashSet<u64> = proof.steps.keys().copied().collect();
        
        // 检查每个步骤的依赖
        for step in proof.steps.values() {
            for &dependency_id in &step.dependencies {
                if !step_ids.contains(&dependency_id) {
                    result.errors.push(format!(
                        "步骤 {} 依赖的步骤 {} 不存在",
                        step.id, dependency_id
                    ));
                    result.success = false;
                }
            }
        }
        
        // 检查循环依赖
        if self.has_circular_dependencies(proof) {
            result.errors.push("检测到循环依赖".to_string());
            result.success = false;
        }
        
        result
    }
}

impl StepDependencyRule {
    /// 检查是否有循环依赖
    fn has_circular_dependencies(&self, proof: &Proof) -> bool {
        let mut visited = std::collections::HashSet::new();
        let mut rec_stack = std::collections::HashSet::new();
        
        for step_id in proof.steps.keys() {
            if !visited.contains(step_id) {
                if self.is_cyclic_util(*step_id, proof, &mut visited, &mut rec_stack) {
                    return true;
                }
            }
        }
        
        false
    }
    
    /// 递归检查循环依赖
    fn is_cyclic_util(
        &self,
        step_id: u64,
        proof: &Proof,
        visited: &mut std::collections::HashSet<u64>,
        rec_stack: &mut std::collections::HashSet<u64>,
    ) -> bool {
        visited.insert(step_id);
        rec_stack.insert(step_id);
        
        if let Some(step) = proof.steps.get(&step_id) {
            for &dependency_id in &step.dependencies {
                if !visited.contains(&dependency_id) {
                    if self.is_cyclic_util(dependency_id, proof, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(&dependency_id) {
                    return true;
                }
            }
        }
        
        rec_stack.remove(&step_id);
        false
    }
}

/// 命题一致性规则
pub struct PropositionConsistencyRule;

impl StructureValidationRule for PropositionConsistencyRule {
    fn name(&self) -> &str {
        "命题一致性规则"
    }
    
    fn validate(&self, proof: &Proof) -> VerificationResult {
        let mut result = VerificationResult {
            success: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            verification_time: std::time::Duration::from_millis(0),
        };
        
        // 检查步骤中的命题引用
        for step in proof.steps.values() {
            // 检查输入命题
            for prop in &step.input_propositions {
                if prop.content.is_empty() {
                    result.errors.push(format!(
                        "步骤 {} 的输入命题内容为空",
                        step.id
                    ));
                    result.success = false;
                }
            }
            
            // 检查输出命题
            for prop in &step.output_propositions {
                if prop.content.is_empty() {
                    result.errors.push(format!(
                        "步骤 {} 的输出命题内容为空",
                        step.id
                    ));
                    result.success = false;
                }
            }
        }
        
        result
    }
}

/// 逻辑验证器
pub struct LogicVerifier {
    /// 验证规则
    validation_rules: Vec<Box<dyn LogicValidationRule>>,
}

impl LogicVerifier {
    /// 创建新的逻辑验证器
    pub fn new() -> Self {
        let mut verifier = Self {
            validation_rules: Vec::new(),
        };
        
        // 添加默认验证规则
        verifier.add_rule(Box::new(LogicalConsistencyRule));
        verifier.add_rule(Box::new(RuleApplicationRule));
        
        verifier
    }
    
    /// 添加验证规则
    pub fn add_rule(&mut self, rule: Box<dyn LogicValidationRule>) {
        self.validation_rules.push(rule);
    }
    
    /// 验证证明逻辑
    pub fn verify(&self, proof: &Proof) -> Result<VerificationResult, ProofError> {
        let mut result = VerificationResult {
            success: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            verification_time: std::time::Duration::from_millis(0),
        };
        
        let start_time = std::time::Instant::now();
        
        // 应用所有验证规则
        for rule in &self.validation_rules {
            let rule_result = rule.validate(proof);
            result.merge(rule_result);
        }
        
        result.verification_time = start_time.elapsed();
        Ok(result)
    }
}

/// 逻辑验证规则特征
pub trait LogicValidationRule {
    /// 验证规则名称
    fn name(&self) -> &str;
    
    /// 应用验证规则
    fn validate(&self, proof: &Proof) -> VerificationResult;
}

/// 逻辑一致性规则
pub struct LogicalConsistencyRule;

impl LogicValidationRule for LogicalConsistencyRule {
    fn name(&self) -> &str {
        "逻辑一致性规则"
    }
    
    fn validate(&self, proof: &Proof) -> VerificationResult {
        let mut result = VerificationResult {
            success: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            verification_time: std::time::Duration::from_millis(0),
        };
        
        // 检查证明步骤的逻辑一致性
        for step in proof.steps.values() {
            // 检查步骤描述和类型的一致性
            if step.description.is_empty() {
                result.errors.push(format!(
                    "步骤 {} 的描述为空",
                    step.id
                ));
                result.success = false;
            }
            
            // 检查证明理由
            if step.justification.is_empty() {
                result.warnings.push(format!(
                    "步骤 {} 缺少证明理由",
                    step.id
                ));
            }
        }
        
        result
    }
}

/// 规则应用规则
pub struct RuleApplicationRule;

impl LogicValidationRule for RuleApplicationRule {
    fn name(&self) -> &str {
        "规则应用规则"
    }
    
    fn validate(&self, proof: &Proof) -> VerificationResult {
        let mut result = VerificationResult {
            success: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            verification_time: std::time::Duration::from_millis(0),
        };
        
        // 检查规则应用的合理性
        for step in proof.steps.values() {
            if let Some(rule_name) = &step.applied_rule {
                if rule_name.is_empty() {
                    result.warnings.push(format!(
                        "步骤 {} 应用的规则名称为空",
                        step.id
                    ));
                }
            }
        }
        
        result
    }
}

/// 完整性验证器
pub struct CompletenessVerifier {
    /// 验证规则
    validation_rules: Vec<Box<dyn CompletenessValidationRule>>,
}

impl CompletenessVerifier {
    /// 创建新的完整性验证器
    pub fn new() -> Self {
        let mut verifier = Self {
            validation_rules: Vec::new(),
        };
        
        // 添加默认验证规则
        verifier.add_rule(Box::new(ProofCompletionRule));
        verifier.add_rule(Box::new(StepSequenceRule));
        
        verifier
    }
    
    /// 添加验证规则
    pub fn add_rule(&mut self, rule: Box<dyn CompletenessValidationRule>) {
        self.validation_rules.push(rule);
    }
    
    /// 验证证明完整性
    pub fn verify(&self, proof: &Proof) -> Result<VerificationResult, ProofError> {
        let mut result = VerificationResult {
            success: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            verification_time: std::time::Duration::from_millis(0),
        };
        
        let start_time = std::time::Instant::now();
        
        // 应用所有验证规则
        for rule in &self.validation_rules {
            let rule_result = rule.validate(proof);
            result.merge(rule_result);
        }
        
        result.verification_time = start_time.elapsed();
        Ok(result)
    }
}

/// 完整性验证规则特征
pub trait CompletenessValidationRule {
    /// 验证规则名称
    fn name(&self) -> &str;
    
    /// 应用验证规则
    fn validate(&self, proof: &Proof) -> VerificationResult;
}

/// 证明完成规则
pub struct ProofCompletionRule;

impl CompletenessValidationRule for ProofCompletionRule {
    fn name(&self) -> &str {
        "证明完成规则"
    }
    
    fn validate(&self, proof: &Proof) -> VerificationResult {
        let mut result = VerificationResult {
            success: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            verification_time: std::time::Duration::from_millis(0),
        };
        
        // 检查证明是否完成
        if proof.status == super::ProofStatus::Completed {
            // 检查是否有足够的步骤
            if proof.steps.len() < 2 {
                result.warnings.push("完成的证明步骤数量较少".to_string());
            }
            
            // 检查最后一步是否到达目标
            if let Some(last_step) = proof.steps.values().max_by_key(|s| s.sequence) {
                if last_step.step_type != super::ProofStepType::Conclusion {
                    result.warnings.push("最后一步不是结论步骤".to_string());
                }
            }
        }
        
        result
    }
}

/// 步骤序列规则
pub struct StepSequenceRule;

impl CompletenessValidationRule for StepSequenceRule {
    fn name(&self) -> &str {
        "步骤序列规则"
    }
    
    fn validate(&self, proof: &Proof) -> VerificationResult {
        let mut result = VerificationResult {
            success: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            verification_time: std::time::Duration::from_millis(0),
        };
        
        // 检查步骤序列的完整性
        let mut step_sequences: Vec<u32> = proof.steps.values().map(|s| s.sequence).collect();
        step_sequences.sort();
        
        // 检查序列是否连续
        for (i, &seq) in step_sequences.iter().enumerate() {
            if seq != (i + 1) as u32 {
                result.warnings.push(format!(
                    "步骤序列不连续，期望 {}，实际 {}",
                    i + 1,
                    seq
                ));
            }
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::{Proof, Proposition, PropositionType};
    use std::collections::HashMap;
    
    fn create_test_proposition(id: &str, content: &str) -> Proposition {
        Proposition {
            id: id.to_string(),
            content: content.to_string(),
            proposition_type: PropositionType::Theorem,
            metadata: HashMap::new(),
        }
    }
    
    fn create_test_proof() -> Proof {
        let goal = create_test_proposition("goal", "A ∧ B → B ∧ A");
        Proof::new(1, "测试证明".to_string(), goal)
    }
    
    #[test]
    fn test_structure_verifier() {
        let verifier = StructureVerifier::new();
        let proof = create_test_proof();
        let result = verifier.verify(&proof).unwrap();
        
        assert!(result.success);
        assert_eq!(result.error_count(), 0);
    }
    
    #[test]
    fn test_logic_verifier() {
        let verifier = LogicVerifier::new();
        let proof = create_test_proof();
        let result = verifier.verify(&proof).unwrap();
        
        assert!(result.success);
    }
    
    #[test]
    fn test_completeness_verifier() {
        let verifier = CompletenessVerifier::new();
        let proof = create_test_proof();
        let result = verifier.verify(&proof).unwrap();
        
        assert!(result.success);
    }
    
    #[test]
    fn test_proof_verifier_impl() {
        let verifier = ProofVerifierImpl::new();
        let proof = create_test_proof();
        
        let structure_result = verifier.verify_structure(&proof).unwrap();
        let logic_result = verifier.verify_logic(&proof).unwrap();
        let completeness_result = verifier.verify_completeness(&proof).unwrap();
        
        assert!(structure_result.success);
        assert!(logic_result.success);
        assert!(completeness_result.success);
    }
}
