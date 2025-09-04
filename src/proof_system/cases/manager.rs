use crate::cases::{
    CaseType, DifficultyLevel, CaseCategory, CaseTag, CaseMetadata, CaseStatistics,
    CaseLibraryConfig, CaseLibraryStatus, CaseLibraryPerformanceMetrics
};
use crate::core::{Proof, ProofError};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Instant, Duration};
use tokio::sync::mpsc;
use uuid::Uuid;

/// 案例记录
#[derive(Debug, Clone)]
pub struct CaseRecord {
    pub case_id: String,
    pub proof: Proof,
    pub metadata: CaseMetadata,
    pub usage_count: usize,
    pub last_used: Option<Instant>,
    pub rating: f64,
    pub comments: Vec<CaseComment>,
    pub version_history: Vec<CaseVersion>,
    pub is_active: bool,
    pub is_archived: bool,
}

/// 案例评论
#[derive(Debug, Clone)]
pub struct CaseComment {
    pub id: String,
    pub author: String,
    pub content: String,
    pub rating: f64,
    pub timestamp: Instant,
    pub is_helpful: bool,
}

/// 案例版本
#[derive(Debug, Clone)]
pub struct CaseVersion {
    pub version: String,
    pub proof: Proof,
    pub metadata: CaseMetadata,
    pub changes: Vec<String>,
    pub timestamp: Instant,
    pub author: String,
}

/// 案例管理器
pub struct CaseManager {
    config: CaseLibraryConfig,
    cases: Arc<RwLock<HashMap<String, CaseRecord>>>,
    case_index: Arc<RwLock<CaseIndex>>,
    statistics: Arc<Mutex<CaseStatistics>>,
    performance_metrics: Arc<Mutex<CaseLibraryPerformanceMetrics>>,
    backup_manager: Arc<Mutex<BackupManager>>,
    event_sender: mpsc::Sender<CaseManagerEvent>,
}

/// 案例索引
#[derive(Debug)]
pub struct CaseIndex {
    pub by_type: HashMap<CaseType, Vec<String>>,
    pub by_difficulty: HashMap<DifficultyLevel, Vec<String>>,
    pub by_category: HashMap<CaseCategory, Vec<String>>,
    pub by_tag: HashMap<String, Vec<String>>,
    pub by_author: HashMap<String, Vec<String>>,
    pub by_keyword: HashMap<String, Vec<String>>,
    pub full_text_index: HashMap<String, Vec<String>>,
}

/// 备份管理器
#[derive(Debug)]
pub struct BackupManager {
    pub last_backup: Option<Instant>,
    pub backup_history: VecDeque<BackupRecord>,
    pub backup_config: BackupConfig,
}

/// 备份记录
#[derive(Debug, Clone)]
pub struct BackupRecord {
    pub backup_id: String,
    pub timestamp: Instant,
    pub case_count: usize,
    pub size_bytes: usize,
    pub status: BackupStatus,
    pub location: String,
}

/// 备份状态
#[derive(Debug, Clone, PartialEq)]
pub enum BackupStatus {
    /// 成功
    Success,
    /// 失败
    Failed,
    /// 进行中
    InProgress,
}

/// 备份配置
#[derive(Debug, Clone)]
pub struct BackupConfig {
    pub auto_backup_enabled: bool,
    pub backup_interval: Duration,
    pub max_backup_count: usize,
    pub backup_location: String,
    pub compression_enabled: bool,
}

/// 案例管理器事件
#[derive(Debug, Clone)]
pub enum CaseManagerEvent {
    /// 案例添加
    CaseAdded { case_id: String, title: String },
    /// 案例更新
    CaseUpdated { case_id: String, version: String },
    /// 案例删除
    CaseDeleted { case_id: String },
    /// 案例归档
    CaseArchived { case_id: String },
    /// 备份完成
    BackupCompleted { backup_id: String, case_count: usize },
    /// 索引更新
    IndexUpdated { timestamp: Instant },
}

impl CaseManager {
    pub fn new(config: CaseLibraryConfig) -> (Self, mpsc::Receiver<CaseManagerEvent>) {
        let (event_sender, event_receiver) = mpsc::channel(100);
        
        let manager = Self {
            config,
            cases: Arc::new(RwLock::new(HashMap::new())),
            case_index: Arc::new(RwLock::new(CaseIndex::new())),
            statistics: Arc::new(Mutex::new(CaseStatistics::default())),
            performance_metrics: Arc::new(Mutex::new(CaseLibraryPerformanceMetrics::default())),
            backup_manager: Arc::new(Mutex::new(BackupManager::new())),
            event_sender,
        };

        (manager, event_receiver)
    }

    /// 添加案例
    pub async fn add_case(
        &self,
        proof: Proof,
        metadata: CaseMetadata,
    ) -> Result<String, ProofError> {
        let start_time = Instant::now();
        
        // 生成案例ID
        let case_id = Uuid::new_v4().to_string();
        
        // 创建案例记录
        let case_record = CaseRecord {
            case_id: case_id.clone(),
            proof: proof.clone(),
            metadata: metadata.clone(),
            usage_count: 0,
            last_used: None,
            rating: 0.0,
            comments: Vec::new(),
            version_history: vec![CaseVersion {
                version: metadata.version.clone(),
                proof: proof.clone(),
                metadata: metadata.clone(),
                changes: vec!["初始版本".to_string()],
                timestamp: Instant::now(),
                author: metadata.author.clone(),
            }],
            is_active: true,
            is_archived: false,
        };

        // 添加到案例库
        let mut cases = self.cases.write().await;
        cases.insert(case_id.clone(), case_record);
        
        // 更新索引
        self.update_case_index(&case_id, &metadata).await?;
        
        // 更新统计
        self.update_statistics().await?;
        
        // 记录性能指标
        let retrieval_time = start_time.elapsed();
        self.update_performance_metrics(retrieval_time).await?;
        
        // 发送事件
        let _ = self.event_sender.try_send(CaseManagerEvent::CaseAdded {
            case_id: case_id.clone(),
            title: metadata.title.clone(),
        });

        Ok(case_id)
    }

    /// 获取案例
    pub async fn get_case(&self, case_id: &str) -> Result<Option<CaseRecord>, ProofError> {
        let start_time = Instant::now();
        
        let cases = self.cases.read().await;
        let case = cases.get(case_id).cloned();
        
        // 更新使用统计
        if let Some(ref case_record) = case {
            self.update_case_usage(case_id).await?;
        }
        
        // 记录性能指标
        let retrieval_time = start_time.elapsed();
        self.update_performance_metrics(retrieval_time).await?;
        
        Ok(case)
    }

    /// 更新案例
    pub async fn update_case(
        &self,
        case_id: &str,
        proof: Proof,
        metadata: CaseMetadata,
    ) -> Result<(), ProofError> {
        let mut cases = self.cases.write().await;
        
        if let Some(case_record) = cases.get_mut(case_id) {
            // 保存当前版本到历史
            let current_version = CaseVersion {
                version: case_record.metadata.version.clone(),
                proof: case_record.proof.clone(),
                metadata: case_record.metadata.clone(),
                changes: vec!["更新前版本".to_string()],
                timestamp: Instant::now(),
                author: case_record.metadata.author.clone(),
            };
            
            case_record.version_history.push(current_version);
            
            // 更新案例
            case_record.proof = proof;
            case_record.metadata = metadata;
            case_record.metadata.updated_at = Instant::now();
            
            // 更新索引
            self.update_case_index(case_id, &case_record.metadata).await?;
            
            // 发送事件
            let _ = self.event_sender.try_send(CaseManagerEvent::CaseUpdated {
                case_id: case_id.to_string(),
                version: case_record.metadata.version.clone(),
            });
            
            Ok(())
        } else {
            Err(ProofError::CaseNotFound {
                case_id: case_id.to_string(),
            })
        }
    }

    /// 删除案例
    pub async fn delete_case(&self, case_id: &str) -> Result<(), ProofError> {
        let mut cases = self.cases.write().await;
        
        if cases.remove(case_id).is_some() {
            // 从索引中移除
            self.remove_case_from_index(case_id).await?;
            
            // 更新统计
            self.update_statistics().await?;
            
            // 发送事件
            let _ = self.event_sender.try_send(CaseManagerEvent::CaseDeleted {
                case_id: case_id.to_string(),
            });
            
            Ok(())
        } else {
            Err(ProofError::CaseNotFound {
                case_id: case_id.to_string(),
            })
        }
    }

    /// 归档案例
    pub async fn archive_case(&self, case_id: &str) -> Result<(), ProofError> {
        let mut cases = self.cases.write().await;
        
        if let Some(case_record) = cases.get_mut(case_id) {
            case_record.is_archived = true;
            case_record.is_active = false;
            
            // 发送事件
            let _ = self.event_sender.try_send(CaseManagerEvent::CaseArchived {
                case_id: case_id.to_string(),
            });
            
            Ok(())
        } else {
            Err(ProofError::CaseNotFound {
                case_id: case_id.to_string(),
            })
        }
    }

    /// 获取案例列表
    pub async fn list_cases(
        &self,
        case_type: Option<CaseType>,
        difficulty: Option<DifficultyLevel>,
        category: Option<CaseCategory>,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<Vec<CaseRecord>, ProofError> {
        let cases = self.cases.read().await;
        let mut filtered_cases: Vec<CaseRecord> = cases.values()
            .filter(|case| case.is_active && !case.is_archived)
            .filter(|case| {
                if let Some(ref case_type_filter) = case_type {
                    case.metadata.tags.iter().any(|tag| tag.category == *case_type_filter)
                } else {
                    true
                }
            })
            .filter(|case| {
                if let Some(ref difficulty_filter) = difficulty {
                    case.metadata.difficulty == *difficulty_filter
                } else {
                    true
                }
            })
            .filter(|case| {
                if let Some(ref category_filter) = category {
                    case.metadata.tags.iter().any(|tag| tag.category == *category_filter)
                } else {
                    true
                }
            })
            .cloned()
            .collect();

        // 排序（按评分和使用次数）
        filtered_cases.sort_by(|a, b| {
            let score_a = a.rating * 0.7 + a.usage_count as f64 * 0.3;
            let score_b = b.rating * 0.7 + b.usage_count as f64 * 0.3;
            score_b.partial_cmp(&score_a).unwrap()
        });

        // 分页
        let offset = offset.unwrap_or(0);
        let limit = limit.unwrap_or(100);
        
        Ok(filtered_cases.into_iter()
            .skip(offset)
            .take(limit)
            .collect())
    }

    /// 搜索案例
    pub async fn search_cases(
        &self,
        query: &str,
        limit: Option<usize>,
    ) -> Result<Vec<CaseRecord>, ProofError> {
        let start_time = Instant::now();
        
        let cases = self.cases.read().await;
        let mut results: Vec<(CaseRecord, f64)> = Vec::new();
        
        for case in cases.values() {
            if !case.is_active || case.is_archived {
                continue;
            }
            
            let relevance_score = self.calculate_relevance_score(case, query);
            if relevance_score > 0.0 {
                results.push((case.clone(), relevance_score));
            }
        }
        
        // 按相关性排序
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // 限制结果数量
        let limit = limit.unwrap_or(50);
        let cases: Vec<CaseRecord> = results.into_iter()
            .take(limit)
            .map(|(case, _)| case)
            .collect();
        
        // 记录性能指标
        let search_time = start_time.elapsed();
        let mut metrics = self.performance_metrics.lock().unwrap();
        metrics.search_response_time = search_time;
        
        Ok(cases)
    }

    /// 计算相关性分数
    fn calculate_relevance_score(&self, case: &CaseRecord, query: &str) -> f64 {
        let query_lower = query.to_lowercase();
        let mut score = 0.0;
        
        // 标题匹配
        if case.metadata.title.to_lowercase().contains(&query_lower) {
            score += 10.0;
        }
        
        // 描述匹配
        if case.metadata.description.to_lowercase().contains(&query_lower) {
            score += 5.0;
        }
        
        // 标签匹配
        for tag in &case.metadata.tags {
            if tag.name.to_lowercase().contains(&query_lower) {
                score += 3.0;
            }
            if tag.description.to_lowercase().contains(&query_lower) {
                score += 2.0;
            }
        }
        
        // 关键词匹配
        for keyword in &case.metadata.learning_objectives {
            if keyword.to_lowercase().contains(&query_lower) {
                score += 2.0;
            }
        }
        
        score
    }

    /// 更新案例索引
    async fn update_case_index(&self, case_id: &str, metadata: &CaseMetadata) -> Result<(), ProofError> {
        let mut index = self.case_index.write().await;
        
        // 按类型索引
        for tag in &metadata.tags {
            index.by_type.entry(tag.category.clone())
                .or_insert_with(Vec::new)
                .push(case_id.to_string());
        }
        
        // 按难度索引
        index.by_difficulty.entry(metadata.difficulty.clone())
            .or_insert_with(Vec::new)
            .push(case_id.to_string());
        
        // 按分类索引
        for tag in &metadata.tags {
            index.by_category.entry(tag.category.clone())
                .or_insert_with(Vec::new)
                .push(case_id.to_string());
        }
        
        // 按标签索引
        for tag in &metadata.tags {
            index.by_tag.entry(tag.name.clone())
                .or_insert_with(Vec::new)
                .push(case_id.to_string());
        }
        
        // 按作者索引
        index.by_author.entry(metadata.author.clone())
            .or_insert_with(Vec::new)
            .push(case_id.to_string());
        
        Ok(())
    }

    /// 从索引中移除案例
    async fn remove_case_from_index(&self, case_id: &str) -> Result<(), ProofError> {
        let mut index = self.case_index.write().await;
        
        // 从所有索引中移除
        for case_ids in index.by_type.values_mut() {
            case_ids.retain(|id| id != case_id);
        }
        for case_ids in index.by_difficulty.values_mut() {
            case_ids.retain(|id| id != case_id);
        }
        for case_ids in index.by_category.values_mut() {
            case_ids.retain(|id| id != case_id);
        }
        for case_ids in index.by_tag.values_mut() {
            case_ids.retain(|id| id != case_id);
        }
        for case_ids in index.by_author.values_mut() {
            case_ids.retain(|id| id != case_id);
        }
        
        Ok(())
    }

    /// 更新案例使用统计
    async fn update_case_usage(&self, case_id: &str) -> Result<(), ProofError> {
        let mut cases = self.cases.write().await;
        
        if let Some(case_record) = cases.get_mut(case_id) {
            case_record.usage_count += 1;
            case_record.last_used = Some(Instant::now());
        }
        
        Ok(())
    }

    /// 更新统计信息
    async fn update_statistics(&self) -> Result<(), ProofError> {
        let cases = self.cases.read().await;
        let mut stats = self.statistics.lock().unwrap();
        
        stats.total_cases = cases.len();
        stats.cases_by_type.clear();
        stats.cases_by_difficulty.clear();
        stats.cases_by_category.clear();
        
        for case in cases.values() {
            if case.is_active && !case.is_archived {
                // 按类型统计
                for tag in &case.metadata.tags {
                    *stats.cases_by_type.entry(tag.category.clone()).or_insert(0) += 1;
                }
                
                // 按难度统计
                *stats.cases_by_difficulty.entry(case.metadata.difficulty.clone()).or_insert(0) += 1;
                
                // 按分类统计
                for tag in &case.metadata.tags {
                    *stats.cases_by_category.entry(tag.category.clone()).or_insert(0) += 1;
                }
            }
        }
        
        Ok(())
    }

    /// 更新性能指标
    async fn update_performance_metrics(&self, retrieval_time: Duration) -> Result<(), ProofError> {
        let mut metrics = self.performance_metrics.lock().unwrap();
        metrics.case_retrieval_time = retrieval_time;
        Ok(())
    }

    /// 获取统计信息
    pub fn get_statistics(&self) -> CaseStatistics {
        let stats = self.statistics.lock().unwrap();
        stats.clone()
    }

    /// 获取案例库状态
    pub async fn get_status(&self) -> CaseLibraryStatus {
        let cases = self.cases.read().await;
        let active_cases = cases.values().filter(|c| c.is_active && !c.is_archived).count();
        let archived_cases = cases.values().filter(|c| c.is_archived).count();
        
        let backup_manager = self.backup_manager.lock().unwrap();
        let performance_metrics = self.performance_metrics.lock().unwrap();
        
        CaseLibraryStatus {
            total_cases: cases.len(),
            active_cases,
            archived_cases,
            last_backup: backup_manager.last_backup,
            last_index_update: Some(Instant::now()),
            storage_usage: 0.0, // 简化实现
            performance_metrics: performance_metrics.clone(),
        }
    }

    /// 执行备份
    pub async fn perform_backup(&self) -> Result<String, ProofError> {
        let mut backup_manager = self.backup_manager.lock().unwrap();
        
        let backup_id = Uuid::new_v4().to_string();
        let cases = self.cases.read().await;
        
        let backup_record = BackupRecord {
            backup_id: backup_id.clone(),
            timestamp: Instant::now(),
            case_count: cases.len(),
            size_bytes: 0, // 简化实现
            status: BackupStatus::Success,
            location: backup_manager.backup_config.backup_location.clone(),
        };
        
        backup_manager.backup_history.push_back(backup_record);
        backup_manager.last_backup = Some(Instant::now());
        
        // 限制备份历史数量
        if backup_manager.backup_history.len() > backup_manager.backup_config.max_backup_count {
            backup_manager.backup_history.pop_front();
        }
        
        // 发送事件
        let _ = self.event_sender.try_send(CaseManagerEvent::BackupCompleted {
            backup_id: backup_id.clone(),
            case_count: cases.len(),
        });
        
        Ok(backup_id)
    }
}

impl CaseIndex {
    pub fn new() -> Self {
        Self {
            by_type: HashMap::new(),
            by_difficulty: HashMap::new(),
            by_category: HashMap::new(),
            by_tag: HashMap::new(),
            by_author: HashMap::new(),
            by_keyword: HashMap::new(),
            full_text_index: HashMap::new(),
        }
    }
}

impl BackupManager {
    pub fn new() -> Self {
        Self {
            last_backup: None,
            backup_history: VecDeque::new(),
            backup_config: BackupConfig {
                auto_backup_enabled: true,
                backup_interval: Duration::from_secs(86400), // 24小时
                max_backup_count: 10,
                backup_location: "./backups".to_string(),
                compression_enabled: true,
            },
        }
    }
}

impl Default for BackupManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Proof, ProofStatus, Proposition};

    fn create_test_case() -> (Proof, CaseMetadata) {
        let proof = Proof {
            id: "test_proof".to_string(),
            name: "测试证明".to_string(),
            description: "这是一个测试证明".to_string(),
            status: ProofStatus::Pending,
            propositions: vec![
                Proposition::new("P", "前提1"),
                Proposition::new("Q", "前提2"),
            ],
            steps: Vec::new(),
            conclusion: Proposition::new("P & Q", "结论"),
            metadata: HashMap::new(),
        };
        
        let metadata = CaseMetadata {
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
        };
        
        (proof, metadata)
    }

    #[tokio::test]
    async fn test_case_manager_creation() {
        let config = CaseLibraryConfig::default();
        let (manager, _) = CaseManager::new(config);
        
        let status = manager.get_status().await;
        assert_eq!(status.total_cases, 0);
    }

    #[tokio::test]
    async fn test_add_case() {
        let config = CaseLibraryConfig::default();
        let (manager, _) = CaseManager::new(config);
        
        let (proof, metadata) = create_test_case();
        let case_id = manager.add_case(proof, metadata).await.unwrap();
        
        assert!(!case_id.is_empty());
        
        let status = manager.get_status().await;
        assert_eq!(status.total_cases, 1);
    }

    #[tokio::test]
    async fn test_get_case() {
        let config = CaseLibraryConfig::default();
        let (manager, _) = CaseManager::new(config);
        
        let (proof, metadata) = create_test_case();
        let case_id = manager.add_case(proof, metadata).await.unwrap();
        
        let case = manager.get_case(&case_id).await.unwrap();
        assert!(case.is_some());
        
        let case = case.unwrap();
        assert_eq!(case.case_id, case_id);
        assert_eq!(case.metadata.title, "测试案例");
    }

    #[tokio::test]
    async fn test_search_cases() {
        let config = CaseLibraryConfig::default();
        let (manager, _) = CaseManager::new(config);
        
        let (proof, metadata) = create_test_case();
        manager.add_case(proof, metadata).await.unwrap();
        
        let results = manager.search_cases("测试", Some(10)).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].metadata.title, "测试案例");
    }
}
