use crate::cases::{
    CaseType, DifficultyLevel, CaseCategory, CaseTag, CaseMetadata, CaseSearchCriteria,
    CaseSearchResult, CaseRecord, CaseStatistics
};
use crate::core::ProofError;
use std::collections::{HashMap, VecDeque, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Instant, Duration};

/// 搜索算法类型
#[derive(Debug, Clone, PartialEq)]
pub enum SearchAlgorithm {
    /// 精确匹配
    ExactMatch,
    /// 模糊搜索
    FuzzySearch,
    /// 语义搜索
    SemanticSearch,
    /// 混合搜索
    HybridSearch,
}

/// 搜索排序方式
#[derive(Debug, Clone, PartialEq)]
pub enum SortOrder {
    /// 相关性排序
    Relevance,
    /// 难度排序
    Difficulty,
    /// 评分排序
    Rating,
    /// 使用次数排序
    UsageCount,
    /// 创建时间排序
    CreationTime,
}

/// 搜索过滤器
#[derive(Debug, Clone)]
pub struct SearchFilter {
    pub case_types: Option<Vec<CaseType>>,
    pub difficulty_levels: Option<Vec<DifficultyLevel>>,
    pub categories: Option<Vec<CaseCategory>>,
    pub tags: Option<Vec<String>>,
    pub authors: Option<Vec<String>>,
    pub time_range: Option<(Instant, Instant)>,
    pub rating_range: Option<(f64, f64)>,
}

/// 搜索配置
#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub algorithm: SearchAlgorithm,
    pub sort_order: SortOrder,
    pub max_results: usize,
    pub enable_highlighting: bool,
    pub enable_suggestions: bool,
    pub fuzzy_threshold: f64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            algorithm: SearchAlgorithm::HybridSearch,
            sort_order: SortOrder::Relevance,
            max_results: 100,
            enable_highlighting: true,
            enable_suggestions: true,
            fuzzy_threshold: 0.7,
        }
    }
}

/// 搜索索引
#[derive(Debug)]
pub struct SearchIndex {
    pub inverted_index: HashMap<String, Vec<IndexEntry>>,
    pub metadata_index: HashMap<String, MetadataIndex>,
    pub tag_index: HashMap<String, Vec<String>>,
    pub author_index: HashMap<String, Vec<String>>,
    pub last_updated: Instant,
}

/// 索引条目
#[derive(Debug, Clone)]
pub struct IndexEntry {
    pub case_id: String,
    pub term: String,
    pub frequency: usize,
    pub weight: f64,
}

/// 元数据索引
#[derive(Debug, Clone)]
pub struct MetadataIndex {
    pub case_id: String,
    pub title: String,
    pub description: String,
    pub tags: Vec<String>,
    pub author: String,
    pub difficulty: DifficultyLevel,
    pub category: CaseCategory,
    pub rating: f64,
    pub usage_count: usize,
    pub created_at: Instant,
    pub updated_at: Instant,
}

/// 案例搜索器
pub struct CaseSearcher {
    config: SearchConfig,
    search_index: Arc<RwLock<SearchIndex>>,
    case_records: Arc<RwLock<HashMap<String, CaseRecord>>>,
    search_history: Arc<Mutex<VecDeque<SearchQuery>>>,
    performance_metrics: Arc<Mutex<SearchPerformanceMetrics>>,
}

/// 搜索查询
#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub query_id: String,
    pub query_text: String,
    pub filters: SearchFilter,
    pub config: SearchConfig,
    pub timestamp: Instant,
    pub result_count: usize,
    pub execution_time: Duration,
}

/// 搜索性能指标
#[derive(Debug, Clone, Default)]
pub struct SearchPerformanceMetrics {
    pub total_searches: usize,
    pub average_response_time: Duration,
    pub cache_hit_rate: f64,
    pub index_efficiency: f64,
}

impl CaseSearcher {
    pub fn new(config: SearchConfig) -> Self {
        Self {
            config,
            search_index: Arc::new(RwLock::new(SearchIndex::new())),
            case_records: Arc::new(RwLock::new(HashMap::new())),
            search_history: Arc::new(Mutex::new(VecDeque::new())),
            performance_metrics: Arc::new(Mutex::new(SearchPerformanceMetrics::default())),
        }
    }

    /// 执行搜索
    pub async fn search(
        &self,
        query: &str,
        filters: Option<SearchFilter>,
        config: Option<SearchConfig>,
    ) -> Result<Vec<CaseSearchResult>, ProofError> {
        let start_time = Instant::now();
        
        let search_config = config.unwrap_or_else(|| self.config.clone());
        let search_filters = filters.unwrap_or_else(SearchFilter::default);
        
        // 根据算法选择搜索策略
        let results = match search_config.algorithm {
            SearchAlgorithm::ExactMatch => {
                self.exact_match_search(query, &search_filters).await?
            }
            SearchAlgorithm::FuzzySearch => {
                self.fuzzy_search(query, &search_filters, search_config.fuzzy_threshold).await?
            }
            SearchAlgorithm::SemanticSearch => {
                self.semantic_search(query, &search_filters).await?
            }
            SearchAlgorithm::HybridSearch => {
                self.hybrid_search(query, &search_filters, &search_config).await?
            }
        };
        
        // 应用过滤器
        let filtered_results = self.apply_filters(results, &search_filters).await?;
        
        // 排序结果
        let sorted_results = self.sort_results(filtered_results, &search_config.sort_order).await?;
        
        // 限制结果数量
        let final_results = sorted_results.into_iter()
            .take(search_config.max_results)
            .collect();
        
        // 记录搜索历史
        let execution_time = start_time.elapsed();
        self.record_search_history(query, search_filters, search_config, execution_time, final_results.len()).await?;
        
        // 更新性能指标
        self.update_performance_metrics(execution_time).await?;
        
        Ok(final_results)
    }

    /// 精确匹配搜索
    async fn exact_match_search(
        &self,
        query: &str,
        filters: &SearchFilter,
    ) -> Result<Vec<CaseSearchResult>, ProofError> {
        let index = self.search_index.read().await;
        let cases = self.case_records.read().await;
        
        let mut results = Vec::new();
        
        if let Some(entries) = index.inverted_index.get(query) {
            for entry in entries {
                if let Some(case_record) = cases.get(&entry.case_id) {
                    let search_result = self.create_search_result(case_record, entry.weight).await?;
                    results.push(search_result);
                }
            }
        }
        
        Ok(results)
    }

    /// 模糊搜索
    async fn fuzzy_search(
        &self,
        query: &str,
        filters: &SearchFilter,
        threshold: f64,
    ) -> Result<Vec<CaseSearchResult>, ProofError> {
        let index = self.search_index.read().await;
        let cases = self.case_records.read().await;
        
        let mut results = Vec::new();
        
        for (term, entries) in &index.inverted_index {
            let similarity = self.calculate_string_similarity(query, term);
            if similarity >= threshold {
                for entry in entries {
                    if let Some(case_record) = cases.get(&entry.case_id) {
                        let adjusted_weight = entry.weight * similarity;
                        let search_result = self.create_search_result(case_record, adjusted_weight).await?;
                        results.push(search_result);
                    }
                }
            }
        }
        
        Ok(results)
    }

    /// 语义搜索
    async fn semantic_search(
        &self,
        query: &str,
        filters: &SearchFilter,
    ) -> Result<Vec<CaseSearchResult>, ProofError> {
        let index = self.search_index.read().await;
        let cases = self.case_records.read().await;
        
        let mut results = Vec::new();
        
        for (case_id, metadata) in &index.metadata_index {
            let semantic_score = self.calculate_semantic_similarity(query, metadata).await?;
            if semantic_score > 0.0 {
                if let Some(case_record) = cases.get(case_id) {
                    let search_result = self.create_search_result(case_record, semantic_score).await?;
                    results.push(search_result);
                }
            }
        }
        
        Ok(results)
    }

    /// 混合搜索
    async fn hybrid_search(
        &self,
        query: &str,
        filters: &SearchFilter,
        config: &SearchConfig,
    ) -> Result<Vec<CaseSearchResult>, ProofError> {
        // 组合多种搜索算法的结果
        let exact_results = self.exact_match_search(query, filters).await?;
        let fuzzy_results = self.fuzzy_search(query, filters, config.fuzzy_threshold).await?;
        let semantic_results = self.semantic_search(query, filters).await?;
        
        // 合并结果并去重
        let mut all_results = Vec::new();
        all_results.extend(exact_results);
        all_results.extend(fuzzy_results);
        all_results.extend(semantic_results);
        
        // 去重并合并权重
        let mut unique_results: HashMap<String, CaseSearchResult> = HashMap::new();
        for result in all_results {
            if let Some(existing) = unique_results.get_mut(&result.case_id) {
                existing.relevance_score = (existing.relevance_score + result.relevance_score) / 2.0;
            } else {
                unique_results.insert(result.case_id.clone(), result);
            }
        }
        
        Ok(unique_results.into_values().collect())
    }

    /// 应用搜索过滤器
    async fn apply_filters(
        &self,
        results: Vec<CaseSearchResult>,
        filters: &SearchFilter,
    ) -> Result<Vec<CaseSearchResult>, ProofError> {
        let mut filtered_results = results;
        
        // 按案例类型过滤
        if let Some(ref case_types) = filters.case_types {
            filtered_results.retain(|result| {
                result.metadata.tags.iter().any(|tag| case_types.contains(&tag.category))
            });
        }
        
        // 按难度级别过滤
        if let Some(ref difficulty_levels) = filters.difficulty_levels {
            filtered_results.retain(|result| {
                difficulty_levels.contains(&result.difficulty)
            });
        }
        
        // 按分类过滤
        if let Some(ref categories) = filters.categories {
            filtered_results.retain(|result| {
                result.metadata.tags.iter().any(|tag| categories.contains(&tag.category))
            });
        }
        
        // 按标签过滤
        if let Some(ref tags) = filters.tags {
            filtered_results.retain(|result| {
                result.metadata.tags.iter().any(|tag| tags.contains(&tag.name))
            });
        }
        
        // 按作者过滤
        if let Some(ref authors) = filters.authors {
            filtered_results.retain(|result| {
                authors.contains(&result.metadata.author)
            });
        }
        
        // 按时间范围过滤
        if let Some((start_time, end_time)) = filters.time_range {
            filtered_results.retain(|result| {
                result.metadata.created_at >= start_time && result.metadata.created_at <= end_time
            });
        }
        
        // 按评分范围过滤
        if let Some((min_rating, max_rating)) = filters.rating_range {
            filtered_results.retain(|result| {
                result.success_rate >= min_rating && result.success_rate <= max_rating
            });
        }
        
        Ok(filtered_results)
    }

    /// 排序搜索结果
    async fn sort_results(
        &self,
        mut results: Vec<CaseSearchResult>,
        sort_order: &SortOrder,
    ) -> Result<Vec<CaseSearchResult>, ProofError> {
        match sort_order {
            SortOrder::Relevance => {
                results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
            }
            SortOrder::Difficulty => {
                results.sort_by(|a, b| a.difficulty.partial_cmp(&b.difficulty).unwrap());
            }
            SortOrder::Rating => {
                results.sort_by(|a, b| b.success_rate.partial_cmp(&a.success_rate).unwrap());
            }
            SortOrder::UsageCount => {
                // 简化实现，按相关性排序
                results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
            }
            SortOrder::CreationTime => {
                results.sort_by(|a, b| b.metadata.created_at.cmp(&a.metadata.created_at));
            }
        }
        
        Ok(results)
    }

    /// 创建搜索结果
    async fn create_search_result(
        &self,
        case_record: &CaseRecord,
        relevance_score: f64,
    ) -> Result<CaseSearchResult, ProofError> {
        Ok(CaseSearchResult {
            case_id: case_record.case_id.clone(),
            title: case_record.metadata.title.clone(),
            description: case_record.metadata.description.clone(),
            relevance_score,
            difficulty: case_record.metadata.difficulty.clone(),
            estimated_time: case_record.metadata.estimated_time,
            success_rate: case_record.rating,
            tags: case_record.metadata.tags.clone(),
            metadata: case_record.metadata.clone(),
        })
    }

    /// 计算字符串相似度
    fn calculate_string_similarity(&self, s1: &str, s2: &str) -> f64 {
        if s1 == s2 {
            return 1.0;
        }
        
        let len1 = s1.len();
        let len2 = s2.len();
        let max_len = len1.max(len2) as f64;
        
        if max_len == 0.0 {
            return 1.0;
        }
        
        // 简化的编辑距离计算
        let distance = self.levenshtein_distance(s1, s2);
        1.0 - (distance as f64 / max_len)
    }

    /// 计算编辑距离
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let len1 = s1.len();
        let len2 = s2.len();
        
        if len1 == 0 {
            return len2;
        }
        if len2 == 0 {
            return len1;
        }
        
        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];
        
        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }
        
        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if s1.chars().nth(i - 1) == s2.chars().nth(j - 1) { 0 } else { 1 };
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }
        
        matrix[len1][len2]
    }

    /// 计算语义相似度
    async fn calculate_semantic_similarity(
        &self,
        query: &str,
        metadata: &MetadataIndex,
    ) -> Result<f64, ProofError> {
        let mut score = 0.0;
        let query_lower = query.to_lowercase();
        
        // 标题匹配
        if metadata.title.to_lowercase().contains(&query_lower) {
            score += 0.4;
        }
        
        // 描述匹配
        if metadata.description.to_lowercase().contains(&query_lower) {
            score += 0.3;
        }
        
        // 标签匹配
        for tag in &metadata.tags {
            if tag.to_lowercase().contains(&query_lower) {
                score += 0.2;
            }
        }
        
        // 作者匹配
        if metadata.author.to_lowercase().contains(&query_lower) {
            score += 0.1;
        }
        
        Ok(score.min(1.0))
    }

    /// 记录搜索历史
    async fn record_search_history(
        &self,
        query: &str,
        filters: SearchFilter,
        config: SearchConfig,
        execution_time: Duration,
        result_count: usize,
    ) -> Result<(), ProofError> {
        let mut history = self.search_history.lock().unwrap();
        
        let search_query = SearchQuery {
            query_id: uuid::Uuid::new_v4().to_string(),
            query_text: query.to_string(),
            filters,
            config,
            timestamp: Instant::now(),
            result_count,
            execution_time,
        };
        
        history.push_back(search_query);
        
        // 限制历史记录数量
        if history.len() > 1000 {
            history.pop_front();
        }
        
        Ok(())
    }

    /// 更新性能指标
    async fn update_performance_metrics(&self, execution_time: Duration) -> Result<(), ProofError> {
        let mut metrics = self.performance_metrics.lock().unwrap();
        
        metrics.total_searches += 1;
        metrics.average_response_time = 
            (metrics.average_response_time + execution_time) / 2;
        
        Ok(())
    }

    /// 获取搜索建议
    pub async fn get_search_suggestions(&self, partial_query: &str) -> Result<Vec<String>, ProofError> {
        let mut suggestions = Vec::new();
        let index = self.search_index.read().await;
        
        for term in index.inverted_index.keys() {
            if term.starts_with(partial_query) && suggestions.len() < 10 {
                suggestions.push(term.clone());
            }
        }
        
        Ok(suggestions)
    }

    /// 获取性能指标
    pub fn get_performance_metrics(&self) -> SearchPerformanceMetrics {
        let metrics = self.performance_metrics.lock().unwrap();
        metrics.clone()
    }
}

impl SearchIndex {
    pub fn new() -> Self {
        Self {
            inverted_index: HashMap::new(),
            metadata_index: HashMap::new(),
            tag_index: HashMap::new(),
            author_index: HashMap::new(),
            last_updated: Instant::now(),
        }
    }
}

impl Default for SearchFilter {
    fn default() -> Self {
        Self {
            case_types: None,
            difficulty_levels: None,
            categories: None,
            tags: None,
            authors: None,
            time_range: None,
            rating_range: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_case_searcher_creation() {
        let config = SearchConfig::default();
        let searcher = CaseSearcher::new(config);
        
        assert_eq!(searcher.config.algorithm, SearchAlgorithm::HybridSearch);
    }

    #[test]
    fn test_string_similarity() {
        let config = SearchConfig::default();
        let searcher = CaseSearcher::new(config);
        
        let similarity = searcher.calculate_string_similarity("hello", "hello");
        assert_eq!(similarity, 1.0);
        
        let similarity = searcher.calculate_string_similarity("hello", "world");
        assert!(similarity < 1.0);
    }

    #[test]
    fn test_levenshtein_distance() {
        let config = SearchConfig::default();
        let searcher = CaseSearcher::new(config);
        
        let distance = searcher.levenshtein_distance("hello", "world");
        assert!(distance > 0);
        
        let distance = searcher.levenshtein_distance("hello", "hello");
        assert_eq!(distance, 0);
    }

    #[tokio::test]
    async fn test_search_suggestions() {
        let config = SearchConfig::default();
        let searcher = CaseSearcher::new(config);
        
        let suggestions = searcher.get_search_suggestions("test").await.unwrap();
        assert!(suggestions.is_empty()); // 空索引，所以没有建议
    }
}
