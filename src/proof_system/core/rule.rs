//! 证明规则模块
//! 
//! 本模块定义了证明规则的核心数据结构，包括推理规则、公理等。

use super::{RuleId, ProofError, Proposition};
use std::collections::HashMap;

/// 推理规则结构
#[derive(Debug, Clone)]
pub struct InferenceRule {
    /// 规则ID
    pub id: RuleId,
    /// 规则名称
    pub name: String,
    /// 规则描述
    pub description: String,
    /// 规则类型
    pub rule_type: RuleType,
    /// 前提条件
    pub premises: Vec<Proposition>,
    /// 结论
    pub conclusion: Proposition,
    /// 规则模式
    pub pattern: RulePattern,
    /// 规则约束
    pub constraints: Vec<RuleConstraint>,
    /// 规则优先级
    pub priority: u32,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

impl InferenceRule {
    /// 创建新的推理规则
    pub fn new(
        id: RuleId,
        name: String,
        description: String,
        rule_type: RuleType,
        premises: Vec<Proposition>,
        conclusion: Proposition,
    ) -> Self {
        Self {
            id,
            name,
            description,
            rule_type,
            premises,
            conclusion,
            pattern: RulePattern::Exact,
            constraints: Vec::new(),
            priority: 100,
            metadata: HashMap::new(),
        }
    }
    
    /// 设置规则模式
    pub fn with_pattern(mut self, pattern: RulePattern) -> Self {
        self.pattern = pattern;
        self
    }
    
    /// 添加规则约束
    pub fn with_constraint(mut self, constraint: RuleConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }
    
    /// 设置规则优先级
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }
    
    /// 检查规则是否适用于给定的前提
    pub fn is_applicable(&self, given_premises: &[Proposition]) -> bool {
        // 检查前提数量是否匹配
        if given_premises.len() != self.premises.len() {
            return false;
        }
        
        // 检查前提类型是否匹配
        for (given, required) in given_premises.iter().zip(self.premises.iter()) {
            if !self.matches_proposition(given, required) {
                return false;
            }
        }
        
        // 检查约束条件
        for constraint in &self.constraints {
            if !constraint.is_satisfied(given_premises) {
                return false;
            }
        }
        
        true
    }
    
    /// 检查命题是否匹配
    fn matches_proposition(&self, given: &Proposition, required: &Proposition) -> bool {
        match &self.pattern {
            RulePattern::Exact => given.content == required.content,
            RulePattern::TypeMatch => given.proposition_type == required.proposition_type,
            RulePattern::PatternMatch(pattern) => {
                // 实现模式匹配逻辑
                given.content.contains(pattern)
            }
            RulePattern::Flexible => true,
        }
    }
    
    /// 应用规则
    pub fn apply(&self, premises: &[Proposition]) -> Result<Proposition, ProofError> {
        if !self.is_applicable(premises) {
            return Err(ProofError::LogicError(format!(
                "规则 {} 不适用于给定的前提",
                self.name
            )));
        }
        
        // 创建结论的副本，可能需要根据输入进行调整
        let conclusion = self.conclusion.clone();
        
        // 根据模式调整结论
        match &self.pattern {
            RulePattern::Exact => Ok(conclusion),
            RulePattern::TypeMatch => Ok(conclusion),
            RulePattern::PatternMatch(_) => {
                // 实现模式替换逻辑
                Ok(conclusion)
            }
            RulePattern::Flexible => Ok(conclusion),
        }
    }
}

/// 规则类型
#[derive(Debug, Clone, PartialEq)]
pub enum RuleType {
    /// 公理
    Axiom,
    /// 定理
    Theorem,
    /// 引理
    Lemma,
    /// 推论
    Corollary,
    /// 定义
    Definition,
    /// 推理规则
    InferenceRule,
    /// 重写规则
    RewriteRule,
    /// 其他
    Other(String),
}

/// 规则模式
#[derive(Debug, Clone, PartialEq)]
pub enum RulePattern {
    /// 精确匹配
    Exact,
    /// 类型匹配
    TypeMatch,
    /// 模式匹配
    PatternMatch(String),
    /// 灵活匹配
    Flexible,
}

/// 规则约束
#[derive(Debug, Clone)]
pub struct RuleConstraint {
    /// 约束名称
    pub name: String,
    /// 约束描述
    pub description: String,
    /// 约束类型
    pub constraint_type: ConstraintType,
    /// 约束条件
    pub condition: String,
}

impl RuleConstraint {
    /// 创建新的规则约束
    pub fn new(name: String, description: String, constraint_type: ConstraintType) -> Self {
        Self {
            name,
            description,
            constraint_type,
            condition: String::new(),
        }
    }
    
    /// 设置约束条件
    pub fn with_condition(mut self, condition: String) -> Self {
        self.condition = condition;
        self
    }
    
    /// 检查约束是否满足
    pub fn is_satisfied(&self, premises: &[Proposition]) -> bool {
        match &self.constraint_type {
            ConstraintType::Precondition => {
                // 检查前提条件
                self.check_precondition(premises)
            }
            ConstraintType::TypeConstraint => {
                // 检查类型约束
                self.check_type_constraint(premises)
            }
            ConstraintType::LogicalConstraint => {
                // 检查逻辑约束
                self.check_logical_constraint(premises)
            }
            ConstraintType::Custom => {
                // 自定义约束检查
                self.check_custom_constraint(premises)
            }
        }
    }
    
    fn check_precondition(&self, _premises: &[Proposition]) -> bool {
        // 实现前提条件检查逻辑
        true
    }
    
    fn check_type_constraint(&self, _premises: &[Proposition]) -> bool {
        // 实现类型约束检查逻辑
        true
    }
    
    fn check_logical_constraint(&self, _premises: &[Proposition]) -> bool {
        // 实现逻辑约束检查逻辑
        true
    }
    
    fn check_custom_constraint(&self, _premises: &[Proposition]) -> bool {
        // 实现自定义约束检查逻辑
        true
    }
}

/// 约束类型
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintType {
    /// 前提条件
    Precondition,
    /// 类型约束
    TypeConstraint,
    /// 逻辑约束
    LogicalConstraint,
    /// 自定义约束
    Custom,
}

/// 规则库
pub struct RuleLibrary {
    /// 规则集合
    rules: HashMap<RuleId, InferenceRule>,
    /// 规则索引
    rule_index: HashMap<String, Vec<RuleId>>,
}

impl RuleLibrary {
    /// 创建新的规则库
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
            rule_index: HashMap::new(),
        }
    }
    
    /// 添加规则
    pub fn add_rule(&mut self, rule: InferenceRule) -> Result<(), ProofError> {
        let rule_id = rule.id;
        let rule_name = rule.name.clone();
        
        // 检查规则ID是否已存在
        if self.rules.contains_key(&rule_id) {
            return Err(ProofError::InternalError(format!(
                "规则ID {} 已存在",
                rule_id
            )));
        }
        
        // 添加规则
        self.rules.insert(rule_id, rule);
        
        // 更新索引
        self.rule_index
            .entry(rule_name)
            .or_insert_with(Vec::new)
            .push(rule_id);
        
        Ok(())
    }
    
    /// 获取规则
    pub fn get_rule(&self, rule_id: RuleId) -> Option<&InferenceRule> {
        self.rules.get(&rule_id)
    }
    
    /// 根据名称查找规则
    pub fn find_rules_by_name(&self, name: &str) -> Vec<&InferenceRule> {
        self.rule_index
            .get(name)
            .map(|ids| ids.iter().filter_map(|id| self.rules.get(id)).collect())
            .unwrap_or_default()
    }
    
    /// 查找适用于给定前提的规则
    pub fn find_applicable_rules(&self, premises: &[Proposition]) -> Vec<&InferenceRule> {
        self.rules
            .values()
            .filter(|rule| rule.is_applicable(premises))
            .collect()
    }
    
    /// 获取所有规则
    pub fn get_all_rules(&self) -> Vec<&InferenceRule> {
        self.rules.values().collect()
    }
    
    /// 获取规则数量
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

/// 规则构建器
pub struct RuleBuilder {
    rule: InferenceRule,
}

impl RuleBuilder {
    /// 创建新的规则构建器
    pub fn new(
        id: RuleId,
        name: String,
        description: String,
        rule_type: RuleType,
        premises: Vec<Proposition>,
        conclusion: Proposition,
    ) -> Self {
        Self {
            rule: InferenceRule::new(id, name, description, rule_type, premises, conclusion),
        }
    }
    
    /// 设置规则模式
    pub fn with_pattern(mut self, pattern: RulePattern) -> Self {
        self.rule.pattern = pattern;
        self
    }
    
    /// 添加规则约束
    pub fn with_constraint(mut self, constraint: RuleConstraint) -> Self {
        self.rule.constraints.push(constraint);
        self
    }
    
    /// 设置规则优先级
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.rule.priority = priority;
        self
    }
    
    /// 构建规则
    pub fn build(self) -> InferenceRule {
        self.rule
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::PropositionType;
    
    fn create_test_proposition(id: &str, content: &str) -> Proposition {
        Proposition {
            id: id.to_string(),
            content: content.to_string(),
            proposition_type: PropositionType::Theorem,
            metadata: HashMap::new(),
        }
    }
    
    #[test]
    fn test_inference_rule_creation() {
        let premises = vec![
            create_test_proposition("p1", "A ∧ B"),
            create_test_proposition("p2", "A"),
        ];
        let conclusion = create_test_proposition("c1", "B");
        
        let rule = InferenceRule::new(
            1,
            "与消除".to_string(),
            "从A ∧ B推导出A".to_string(),
            RuleType::InferenceRule,
            premises.clone(),
            conclusion.clone(),
        );
        
        assert_eq!(rule.id, 1);
        assert_eq!(rule.name, "与消除");
        assert_eq!(rule.premises.len(), 2);
        assert_eq!(rule.conclusion.content, "B");
    }
    
    #[test]
    fn test_rule_library() {
        let mut library = RuleLibrary::new();
        
        let premises = vec![create_test_proposition("p1", "A ∧ B")];
        let conclusion = create_test_proposition("c1", "A");
        
        let rule = InferenceRule::new(
            1,
            "与消除".to_string(),
            "从A ∧ B推导出A".to_string(),
            RuleType::InferenceRule,
            premises,
            conclusion,
        );
        
        assert!(library.add_rule(rule).is_ok());
        assert_eq!(library.rule_count(), 1);
    }
    
    #[test]
    fn test_rule_builder() {
        let premises = vec![create_test_proposition("p1", "A ∧ B")];
        let conclusion = create_test_proposition("c1", "A");
        
        let rule = RuleBuilder::new(
            1,
            "与消除".to_string(),
            "从A ∧ B推导出A".to_string(),
            RuleType::InferenceRule,
            premises,
            conclusion,
        )
        .with_pattern(RulePattern::Exact)
        .with_priority(50)
        .build();
        
        assert_eq!(rule.pattern, RulePattern::Exact);
        assert_eq!(rule.priority, 50);
    }
}
