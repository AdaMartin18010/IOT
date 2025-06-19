# IoT分布式算法分析

## 1. 概述

本文档分析IoT系统中的核心分布式算法，包括一致性算法、路由算法、负载均衡算法等，并提供形式化分析和Rust实现。

## 2. 分布式一致性算法

### 2.1 Raft算法

**定义 2.1.1 (Raft状态)** Raft节点状态：

$$State_{raft} \in \{Follower, Candidate, Leader\}$$

**定义 2.1.2 (Raft术语)** Raft术语：

- $Term_{current}$: 当前任期
- $VotedFor$: 投票给哪个候选者
- $Log[]$: 日志条目数组
- $CommitIndex$: 已提交的日志索引
- $LastApplied$: 最后应用的日志索引

**算法 2.1.1 (Raft领导者选举)**

```rust
// Raft节点状态
#[derive(Debug, Clone, PartialEq)]
pub enum RaftState {
    Follower,
    Candidate,
    Leader,
}

// Raft节点
#[derive(Debug, Clone)]
pub struct RaftNode {
    pub id: u64,
    pub state: RaftState,
    pub current_term: u64,
    pub voted_for: Option<u64>,
    pub log: Vec<LogEntry>,
    pub commit_index: u64,
    pub last_applied: u64,
    pub next_index: HashMap<u64, u64>,
    pub match_index: HashMap<u64, u64>,
}

// 日志条目
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub term: u64,
    pub index: u64,
    pub command: String,
}

// Raft算法实现
impl RaftNode {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            state: RaftState::Follower,
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
            commit_index: 0,
            last_applied: 0,
            next_index: HashMap::new(),
            match_index: HashMap::new(),
        }
    }

    /// 开始领导者选举
    pub fn start_election(&mut self) {
        self.current_term += 1;
        self.state = RaftState::Candidate;
        self.voted_for = Some(self.id);
        
        // 发送RequestVote RPC给所有其他节点
        self.send_request_vote();
    }

    /// 处理RequestVote RPC
    pub fn handle_request_vote(&mut self, request: RequestVoteRequest) -> RequestVoteResponse {
        if request.term < self.current_term {
            return RequestVoteResponse {
                term: self.current_term,
                vote_granted: false,
            };
        }

        if request.term > self.current_term {
            self.current_term = request.term;
            self.state = RaftState::Follower;
            self.voted_for = None;
        }

        if self.voted_for.is_none() || self.voted_for == Some(request.candidate_id) {
            if self.is_log_up_to_date(&request) {
                self.voted_for = Some(request.candidate_id);
                return RequestVoteResponse {
                    term: self.current_term,
                    vote_granted: true,
                };
            }
        }

        RequestVoteResponse {
            term: self.current_term,
            vote_granted: false,
        }
    }

    /// 检查日志是否最新
    fn is_log_up_to_date(&self, request: &RequestVoteRequest) -> bool {
        let last_log_index = self.log.len() as u64;
        let last_log_term = if last_log_index > 0 {
            self.log[last_log_index as usize - 1].term
        } else {
            0
        };

        request.last_log_term > last_log_term ||
        (request.last_log_term == last_log_term && request.last_log_index >= last_log_index)
    }

    /// 发送RequestVote RPC
    fn send_request_vote(&self) {
        // 实际实现中会发送网络请求
        println!("Sending RequestVote RPC from node {}", self.id);
    }
}

// RequestVote请求
#[derive(Debug, Clone)]
pub struct RequestVoteRequest {
    pub term: u64,
    pub candidate_id: u64,
    pub last_log_index: u64,
    pub last_log_term: u64,
}

// RequestVote响应
#[derive(Debug, Clone)]
pub struct RequestVoteResponse {
    pub term: u64,
    pub vote_granted: bool,
}
```

**定理 2.1.1 (Raft安全性)** Raft算法保证在任何时刻最多只有一个领导者。

**证明：**

1. 每个任期最多只能有一个领导者
2. 如果两个候选者在同一任期获得多数票，则它们的日志必须相同
3. 根据日志匹配属性，不可能有两个不同的领导者
因此Raft算法保证安全性。$\square$

### 2.2 拜占庭容错算法

**定义 2.2.1 (拜占庭节点)** 拜占庭节点可能发送任意消息，包括错误消息。

**定义 2.2.2 (拜占庭容错)** 系统能够容忍最多 $f$ 个拜占庭节点，当总节点数为 $n$ 时，需要 $n > 3f$。

**算法 2.2.1 (PBFT算法)**

```rust
use std::collections::HashMap;

// PBFT节点状态
#[derive(Debug, Clone, PartialEq)]
pub enum PBFTState {
    PrePrepare,
    Prepare,
    Commit,
    Reply,
}

// PBFT节点
#[derive(Debug, Clone)]
pub struct PBFTNode {
    pub id: u64,
    pub state: PBFTState,
    pub view_number: u64,
    pub sequence_number: u64,
    pub primary_id: u64,
    pub prepared_messages: HashMap<String, PreparedMessage>,
    pub committed_messages: HashMap<String, CommittedMessage>,
}

// 预准备消息
#[derive(Debug, Clone)]
pub struct PrePrepareMessage {
    pub view_number: u64,
    pub sequence_number: u64,
    pub message_digest: String,
    pub message: String,
}

// 准备消息
#[derive(Debug, Clone)]
pub struct PrepareMessage {
    pub view_number: u64,
    pub sequence_number: u64,
    pub message_digest: String,
    pub node_id: u64,
}

// 已准备消息
#[derive(Debug, Clone)]
pub struct PreparedMessage {
    pub view_number: u64,
    pub sequence_number: u64,
    pub message_digest: String,
    pub prepare_messages: Vec<PrepareMessage>,
}

// 提交消息
#[derive(Debug, Clone)]
pub struct CommitMessage {
    pub view_number: u64,
    pub sequence_number: u64,
    pub message_digest: String,
    pub node_id: u64,
}

// 已提交消息
#[derive(Debug, Clone)]
pub struct CommittedMessage {
    pub view_number: u64,
    pub sequence_number: u64,
    pub message_digest: String,
    pub commit_messages: Vec<CommitMessage>,
}

impl PBFTNode {
    pub fn new(id: u64, total_nodes: u64) -> Self {
        Self {
            id,
            state: PBFTState::PrePrepare,
            view_number: 0,
            sequence_number: 0,
            primary_id: 0,
            prepared_messages: HashMap::new(),
            committed_messages: HashMap::new(),
        }
    }

    /// 处理客户端请求
    pub fn handle_client_request(&mut self, request: String) -> Result<(), String> {
        if self.id != self.primary_id {
            return Err("Not primary node".to_string());
        }

        let digest = self.compute_digest(&request);
        let pre_prepare = PrePrepareMessage {
            view_number: self.view_number,
            sequence_number: self.sequence_number,
            message_digest: digest.clone(),
            message: request,
        };

        self.broadcast_pre_prepare(pre_prepare);
        Ok(())
    }

    /// 处理预准备消息
    pub fn handle_pre_prepare(&mut self, message: PrePrepareMessage) -> Result<(), String> {
        // 验证消息
        if !self.verify_pre_prepare(&message) {
            return Err("Invalid pre-prepare message".to_string());
        }

        // 发送准备消息
        let prepare = PrepareMessage {
            view_number: message.view_number,
            sequence_number: message.sequence_number,
            message_digest: message.message_digest.clone(),
            node_id: self.id,
        };

        self.broadcast_prepare(prepare);
        Ok(())
    }

    /// 处理准备消息
    pub fn handle_prepare(&mut self, message: PrepareMessage) -> Result<(), String> {
        // 收集准备消息
        let key = format!("{}-{}", message.view_number, message.sequence_number);
        
        if let Some(prepared) = self.prepared_messages.get_mut(&key) {
            prepared.prepare_messages.push(message.clone());
            
            // 检查是否达到准备条件
            if self.is_prepared(&key) {
                self.broadcast_commit(CommitMessage {
                    view_number: message.view_number,
                    sequence_number: message.sequence_number,
                    message_digest: message.message_digest.clone(),
                    node_id: self.id,
                });
            }
        }

        Ok(())
    }

    /// 处理提交消息
    pub fn handle_commit(&mut self, message: CommitMessage) -> Result<(), String> {
        let key = format!("{}-{}", message.view_number, message.sequence_number);
        
        if let Some(committed) = self.committed_messages.get_mut(&key) {
            committed.commit_messages.push(message.clone());
            
            // 检查是否达到提交条件
            if self.is_committed(&key) {
                self.execute_request(&key);
            }
        }

        Ok(())
    }

    /// 验证预准备消息
    fn verify_pre_prepare(&self, message: &PrePrepareMessage) -> bool {
        message.view_number == self.view_number &&
        message.sequence_number > self.sequence_number
    }

    /// 检查是否已准备
    fn is_prepared(&self, key: &str) -> bool {
        if let Some(prepared) = self.prepared_messages.get(key) {
            prepared.prepare_messages.len() >= 2 * self.get_faulty_nodes() + 1
        } else {
            false
        }
    }

    /// 检查是否已提交
    fn is_committed(&self, key: &str) -> bool {
        if let Some(committed) = self.committed_messages.get(key) {
            committed.commit_messages.len() >= 2 * self.get_faulty_nodes() + 1
        } else {
            false
        }
    }

    /// 计算消息摘要
    fn compute_digest(&self, message: &str) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(message.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// 获取故障节点数
    fn get_faulty_nodes(&self) -> usize {
        // 假设总节点数为4，故障节点数为1
        1
    }

    /// 广播预准备消息
    fn broadcast_pre_prepare(&self, message: PrePrepareMessage) {
        println!("Broadcasting pre-prepare message: {:?}", message);
    }

    /// 广播准备消息
    fn broadcast_prepare(&self, message: PrepareMessage) {
        println!("Broadcasting prepare message: {:?}", message);
    }

    /// 广播提交消息
    fn broadcast_commit(&self, message: CommitMessage) {
        println!("Broadcasting commit message: {:?}", message);
    }

    /// 执行请求
    fn execute_request(&mut self, key: &str) {
        println!("Executing request: {}", key);
    }
}
```

## 3. 分布式路由算法

### 3.1 最短路径算法

**定义 3.1.1 (网络图)** IoT网络图：

$$G = (V, E, W)$$

其中：

- $V$ 是节点集合（设备）
- $E$ 是边集合（连接）
- $W: E \rightarrow \mathbb{R}^+$ 是权重函数

**算法 3.1.1 (Dijkstra算法)**

```rust
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

// 图节点
#[derive(Debug, Clone)]
pub struct Node {
    pub id: u64,
    pub neighbors: Vec<u64>,
}

// 图边
#[derive(Debug, Clone)]
pub struct Edge {
    pub from: u64,
    pub to: u64,
    pub weight: f64,
}

// 网络图
#[derive(Debug, Clone)]
pub struct NetworkGraph {
    pub nodes: HashMap<u64, Node>,
    pub edges: HashMap<(u64, u64), Edge>,
}

// 距离节点
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DistanceNode {
    pub id: u64,
    pub distance: u64,
}

impl Ord for DistanceNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.distance.cmp(&self.distance)
    }
}

impl PartialOrd for DistanceNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl NetworkGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
        }
    }

    /// 添加节点
    pub fn add_node(&mut self, id: u64) {
        self.nodes.insert(id, Node {
            id,
            neighbors: Vec::new(),
        });
    }

    /// 添加边
    pub fn add_edge(&mut self, from: u64, to: u64, weight: f64) {
        self.edges.insert((from, to), Edge {
            from,
            to,
            weight,
        });

        if let Some(node) = self.nodes.get_mut(&from) {
            node.neighbors.push(to);
        }
    }

    /// Dijkstra最短路径算法
    pub fn dijkstra(&self, start: u64, end: u64) -> Option<(Vec<u64>, f64)> {
        let mut distances: HashMap<u64, f64> = HashMap::new();
        let mut previous: HashMap<u64, u64> = HashMap::new();
        let mut heap = BinaryHeap::new();

        // 初始化距离
        for &node_id in self.nodes.keys() {
            distances.insert(node_id, f64::INFINITY);
        }
        distances.insert(start, 0.0);

        heap.push(DistanceNode {
            id: start,
            distance: 0,
        });

        while let Some(current) = heap.pop() {
            if current.id == end {
                break;
            }

            if current.distance as f64 > distances[&current.id] {
                continue;
            }

            if let Some(node) = self.nodes.get(&current.id) {
                for &neighbor_id in &node.neighbors {
                    if let Some(edge) = self.edges.get(&(current.id, neighbor_id)) {
                        let new_distance = distances[&current.id] + edge.weight;

                        if new_distance < distances[&neighbor_id] {
                            distances.insert(neighbor_id, new_distance);
                            previous.insert(neighbor_id, current.id);

                            heap.push(DistanceNode {
                                id: neighbor_id,
                                distance: new_distance as u64,
                            });
                        }
                    }
                }
            }
        }

        // 重建路径
        if distances[&end] == f64::INFINITY {
            None
        } else {
            let mut path = Vec::new();
            let mut current = end;

            while current != start {
                path.push(current);
                current = previous[&current];
            }
            path.push(start);
            path.reverse();

            Some((path, distances[&end]))
        }
    }
}
```

**定理 3.1.1 (Dijkstra正确性)** Dijkstra算法找到从起点到终点的最短路径。

**证明：**

1. 算法维护距离数组，初始时起点距离为0，其他为无穷大
2. 每次选择距离最小的未访问节点
3. 更新该节点的邻居距离
4. 重复直到访问终点
5. 根据贪心选择性质，每次选择的节点距离都是最短的
因此算法正确。$\square$

### 3.2 负载均衡算法

**定义 3.2.1 (负载)** 节点负载：

$$Load(node) = \frac{ActiveConnections(node)}{MaxConnections(node)}$$

**算法 3.2.1 (轮询负载均衡)**

```rust
use std::sync::{Arc, Mutex};

// 负载均衡器
#[derive(Debug, Clone)]
pub struct LoadBalancer {
    pub nodes: Vec<LoadBalancerNode>,
    pub current_index: usize,
    pub algorithm: LoadBalancingAlgorithm,
}

// 负载均衡节点
#[derive(Debug, Clone)]
pub struct LoadBalancerNode {
    pub id: u64,
    pub address: String,
    pub weight: u32,
    pub current_connections: u32,
    pub max_connections: u32,
    pub health_status: HealthStatus,
}

// 健康状态
#[derive(Debug, Clone)]
pub enum HealthStatus {
    Healthy,
    Unhealthy,
    Unknown,
}

// 负载均衡算法
#[derive(Debug, Clone)]
pub enum LoadBalancingAlgorithm {
    RoundRobin,
    WeightedRoundRobin,
    LeastConnections,
    WeightedLeastConnections,
    IPHash,
}

impl LoadBalancer {
    pub fn new(algorithm: LoadBalancingAlgorithm) -> Self {
        Self {
            nodes: Vec::new(),
            current_index: 0,
            algorithm,
        }
    }

    /// 添加节点
    pub fn add_node(&mut self, node: LoadBalancerNode) {
        self.nodes.push(node);
    }

    /// 选择下一个节点
    pub fn select_node(&mut self) -> Option<&LoadBalancerNode> {
        match self.algorithm {
            LoadBalancingAlgorithm::RoundRobin => self.round_robin(),
            LoadBalancingAlgorithm::WeightedRoundRobin => self.weighted_round_robin(),
            LoadBalancingAlgorithm::LeastConnections => self.least_connections(),
            LoadBalancingAlgorithm::WeightedLeastConnections => self.weighted_least_connections(),
            LoadBalancingAlgorithm::IPHash => self.ip_hash(),
        }
    }

    /// 轮询算法
    fn round_robin(&mut self) -> Option<&LoadBalancerNode> {
        if self.nodes.is_empty() {
            return None;
        }

        let healthy_nodes: Vec<_> = self.nodes.iter()
            .filter(|node| node.health_status == HealthStatus::Healthy)
            .collect();

        if healthy_nodes.is_empty() {
            return None;
        }

        let selected = healthy_nodes[self.current_index % healthy_nodes.len()];
        self.current_index = (self.current_index + 1) % healthy_nodes.len();

        Some(selected)
    }

    /// 加权轮询算法
    fn weighted_round_robin(&mut self) -> Option<&LoadBalancerNode> {
        if self.nodes.is_empty() {
            return None;
        }

        let healthy_nodes: Vec<_> = self.nodes.iter()
            .filter(|node| node.health_status == HealthStatus::Healthy)
            .collect();

        if healthy_nodes.is_empty() {
            return None;
        }

        // 计算总权重
        let total_weight: u32 = healthy_nodes.iter()
            .map(|node| node.weight)
            .sum();

        if total_weight == 0 {
            return self.round_robin();
        }

        // 选择节点
        let mut current_weight = 0;
        for node in &healthy_nodes {
            current_weight += node.weight;
            if self.current_index < current_weight as usize {
                self.current_index = (self.current_index + 1) % total_weight as usize;
                return Some(node);
            }
        }

        None
    }

    /// 最少连接算法
    fn least_connections(&self) -> Option<&LoadBalancerNode> {
        self.nodes.iter()
            .filter(|node| node.health_status == HealthStatus::Healthy)
            .min_by_key(|node| node.current_connections)
    }

    /// 加权最少连接算法
    fn weighted_least_connections(&self) -> Option<&LoadBalancerNode> {
        self.nodes.iter()
            .filter(|node| node.health_status == HealthStatus::Healthy)
            .min_by(|a, b| {
                let a_ratio = a.current_connections as f64 / a.weight as f64;
                let b_ratio = b.current_connections as f64 / b.weight as f64;
                a_ratio.partial_cmp(&b_ratio).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// IP哈希算法
    fn ip_hash(&self) -> Option<&LoadBalancerNode> {
        // 这里简化实现，实际应该基于客户端IP计算哈希
        let client_ip = "192.168.1.1"; // 示例IP
        let hash = self.compute_hash(client_ip);
        
        let healthy_nodes: Vec<_> = self.nodes.iter()
            .filter(|node| node.health_status == HealthStatus::Healthy)
            .collect();

        if healthy_nodes.is_empty() {
            return None;
        }

        let index = hash as usize % healthy_nodes.len();
        Some(&healthy_nodes[index])
    }

    /// 计算哈希值
    fn compute_hash(&self, input: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        input.hash(&mut hasher);
        hasher.finish()
    }

    /// 更新节点连接数
    pub fn update_connection_count(&mut self, node_id: u64, increment: bool) {
        if let Some(node) = self.nodes.iter_mut().find(|n| n.id == node_id) {
            if increment {
                node.current_connections += 1;
            } else if node.current_connections > 0 {
                node.current_connections -= 1;
            }
        }
    }

    /// 健康检查
    pub fn health_check(&mut self) {
        for node in &mut self.nodes {
            // 实际实现中会发送健康检查请求
            node.health_status = HealthStatus::Healthy;
        }
    }
}
```

## 4. 分布式数据算法

### 4.1 一致性哈希算法

**定义 4.1.1 (一致性哈希)** 一致性哈希将数据映射到环形哈希空间：

$$Hash(key) \rightarrow [0, 2^{32}-1]$$

**算法 4.1.1 (一致性哈希实现)**

```rust
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

// 一致性哈希环
#[derive(Debug, Clone)]
pub struct ConsistentHashRing {
    pub ring: BTreeMap<u32, String>,
    pub virtual_nodes: u32,
    pub nodes: Vec<String>,
}

impl ConsistentHashRing {
    pub fn new(virtual_nodes: u32) -> Self {
        Self {
            ring: BTreeMap::new(),
            virtual_nodes,
            nodes: Vec::new(),
        }
    }

    /// 添加节点
    pub fn add_node(&mut self, node: String) {
        self.nodes.push(node.clone());
        
        for i in 0..self.virtual_nodes {
            let virtual_node = format!("{}#{}", node, i);
            let hash = self.compute_hash(&virtual_node);
            self.ring.insert(hash, node.clone());
        }
    }

    /// 移除节点
    pub fn remove_node(&mut self, node: &str) {
        self.nodes.retain(|n| n != node);
        
        for i in 0..self.virtual_nodes {
            let virtual_node = format!("{}#{}", node, i);
            let hash = self.compute_hash(&virtual_node);
            self.ring.remove(&hash);
        }
    }

    /// 获取负责节点
    pub fn get_node(&self, key: &str) -> Option<&String> {
        if self.ring.is_empty() {
            return None;
        }

        let hash = self.compute_hash(key);
        
        // 查找大于等于hash的第一个节点
        if let Some((_, node)) = self.ring.range(hash..).next() {
            return Some(node);
        }
        
        // 如果没找到，返回第一个节点（环形）
        self.ring.values().next()
    }

    /// 计算哈希值
    fn compute_hash(&self, key: &str) -> u32 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;
        
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish() as u32
    }

    /// 获取节点分布
    pub fn get_distribution(&self) -> HashMap<String, u32> {
        let mut distribution = HashMap::new();
        
        for (_, node) in &self.ring {
            *distribution.entry(node.clone()).or_insert(0) += 1;
        }
        
        distribution
    }
}
```

**定理 4.1.1 (一致性哈希平衡性)** 一致性哈希算法在节点数量足够大时，数据分布趋于均匀。

**证明：**

1. 虚拟节点增加了哈希环的密度
2. 当虚拟节点数量足够大时，每个真实节点在环上的分布趋于均匀
3. 因此数据分布也趋于均匀。$\square$

### 4.2 分布式缓存算法

**定义 4.2.1 (缓存一致性)** 缓存一致性保证所有副本的数据最终一致。

**算法 4.2.1 (最终一致性缓存)**

```rust
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

// 缓存条目
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub key: String,
    pub value: String,
    pub timestamp: SystemTime,
    pub version: u64,
    pub ttl: Option<Duration>,
}

// 分布式缓存节点
#[derive(Debug, Clone)]
pub struct DistributedCacheNode {
    pub id: u64,
    pub cache: HashMap<String, CacheEntry>,
    pub replicas: Vec<u64>,
    pub version_counter: u64,
}

impl DistributedCacheNode {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            cache: HashMap::new(),
            replicas: Vec::new(),
            version_counter: 0,
        }
    }

    /// 设置缓存
    pub fn set(&mut self, key: String, value: String, ttl: Option<Duration>) {
        self.version_counter += 1;
        
        let entry = CacheEntry {
            key: key.clone(),
            value,
            timestamp: SystemTime::now(),
            version: self.version_counter,
            ttl,
        };

        self.cache.insert(key, entry);
        
        // 异步复制到其他节点
        self.replicate_to_others(&entry);
    }

    /// 获取缓存
    pub fn get(&self, key: &str) -> Option<&CacheEntry> {
        if let Some(entry) = self.cache.get(key) {
            // 检查TTL
            if let Some(ttl) = entry.ttl {
                if let Ok(elapsed) = entry.timestamp.elapsed() {
                    if elapsed > ttl {
                        return None;
                    }
                }
            }
            Some(entry)
        } else {
            None
        }
    }

    /// 删除缓存
    pub fn delete(&mut self, key: &str) {
        self.cache.remove(key);
        
        // 异步删除其他节点的副本
        self.delete_from_others(key);
    }

    /// 复制到其他节点
    fn replicate_to_others(&self, entry: &CacheEntry) {
        // 实际实现中会发送网络请求
        println!("Replicating entry to other nodes: {:?}", entry);
    }

    /// 从其他节点删除
    fn delete_from_others(&self, key: &str) {
        // 实际实现中会发送网络请求
        println!("Deleting key from other nodes: {}", key);
    }

    /// 合并来自其他节点的更新
    pub fn merge_update(&mut self, entry: CacheEntry) {
        if let Some(existing) = self.cache.get(&entry.key) {
            if entry.version > existing.version {
                self.cache.insert(entry.key.clone(), entry);
            }
        } else {
            self.cache.insert(entry.key.clone(), entry);
        }
    }

    /// 清理过期条目
    pub fn cleanup_expired(&mut self) {
        let now = SystemTime::now();
        self.cache.retain(|_, entry| {
            if let Some(ttl) = entry.ttl {
                if let Ok(elapsed) = entry.timestamp.elapsed() {
                    return elapsed <= ttl;
                }
            }
            true
        });
    }
}
```

## 5. 复杂度分析

### 5.1 时间复杂度

**定理 5.1.1 (Raft时间复杂度)** Raft算法的领导者选举时间复杂度为 $O(n)$，其中 $n$ 是节点数量。

**证明：**

1. 每个节点最多发送一次RequestVote RPC
2. 每个节点最多接收 $n-1$ 个RequestVote RPC
3. 总时间复杂度为 $O(n)$。$\square$

**定理 5.1.2 (Dijkstra时间复杂度)** Dijkstra算法的时间复杂度为 $O((V + E) \log V)$，其中 $V$ 是节点数，$E$ 是边数。

**证明：**

1. 使用优先队列，每次操作时间复杂度为 $O(\log V)$
2. 最多进行 $V$ 次出队操作
3. 最多进行 $E$ 次入队操作
4. 总时间复杂度为 $O((V + E) \log V)$。$\square$

### 5.2 空间复杂度

**定理 5.2.1 (一致性哈希空间复杂度)** 一致性哈希算法的空间复杂度为 $O(n \times v)$，其中 $n$ 是节点数，$v$ 是虚拟节点数。

**证明：**

1. 每个真实节点创建 $v$ 个虚拟节点
2. 总共需要存储 $n \times v$ 个哈希映射
3. 空间复杂度为 $O(n \times v)$。$\square$

## 6. 总结

本文档分析了IoT系统中的核心分布式算法：

1. **一致性算法**：Raft、PBFT等
2. **路由算法**：Dijkstra最短路径
3. **负载均衡算法**：轮询、最少连接等
4. **数据算法**：一致性哈希、分布式缓存

每个算法都提供了：

- 形式化定义和数学分析
- 详细的Rust实现
- 复杂度分析和正确性证明

这些算法为构建可靠的IoT分布式系统提供了理论基础和实践指导。

---

**参考文献：**

1. Diego Ongaro and John Ousterhout. "In Search of an Understandable Consensus Algorithm"
2. Miguel Castro and Barbara Liskov. "Practical Byzantine Fault Tolerance"
3. Edsger W. Dijkstra. "A Note on Two Problems in Connexion with Graphs"
4. David Karger, Eric Lehman, Tom Leighton, Rina Panigrahy, Matthew Levine, and Daniel Lewin. "Consistent Hashing and Random Trees: Distributed Caching Protocols for Relieving Hot Spots on the World Wide Web"
