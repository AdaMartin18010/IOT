//! IoT形式化证明系统
//! 
//! 本库提供了完整的IoT系统形式化证明功能，包括：
//! - 证明系统核心
//! - 证明策略
//! - 证明验证
//! - 自动化系统
//! 
//! # 使用示例
//! 
//! ```rust
//! use iot_proof_system::proof_system::FormalProofSystem;
//! use iot_proof_system::core::{Proposition, PropositionType};
//! use std::collections::HashMap;
//! 
//! let mut proof_system = FormalProofSystem::new();
//! 
//! let goal = Proposition {
//!     id: "goal1".to_string(),
//!     content: "A ∧ B → B ∧ A".to_string(),
//!     proposition_type: PropositionType::Theorem,
//!     metadata: HashMap::new(),
//! };
//! 
//! let proof_id = proof_system.create_proof(goal).unwrap();
//! ```

pub mod proof_system;

// 重新导出主要模块
pub use proof_system::*;
