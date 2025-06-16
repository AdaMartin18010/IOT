# IoT算法基础

## 目录

1. [引言](#引言)
2. [分布式算法](#分布式算法)
3. [实时算法](#实时算法)
4. [优化算法](#优化算法)
5. [安全算法](#安全算法)
6. [Rust实现](#rust实现)
7. [结论](#结论)

## 引言

IoT系统需要高效的算法来处理分布式计算、实时响应、资源优化和安全保障。本文从形式化角度分析IoT系统中的核心算法。

### 定义 1.1 (IoT算法)
一个IoT算法是一个四元组 $\mathcal{A} = (I, O, F, C)$，其中：
- $I$ 是输入空间
- $O$ 是输出空间
- $F: I \rightarrow O$ 是算法函数
- $C$ 是复杂度约束

## 分布式算法

### 定义 1.2 (分布式共识算法)
分布式共识算法解决在异步网络中，多个节点就某个值达成一致的问题。

**问题形式化**：
给定 $n$ 个节点，每个节点有一个初始值 $v_i$，算法需要满足：
1. **一致性**：所有非故障节点决定相同的值
2. **有效性**：如果所有节点初始值相同，则决定值等于初始值
3. **终止性**：所有非故障节点最终做出决定

### 算法 1.1 (Raft共识算法)

```rust
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RaftState {
    Follower,
    Candidate,
    Leader,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub term: u64,
    pub index: u64,
    pub command: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

    /// 开始选举
    pub async fn start_election(&mut self) -> Result<(), String> {
        self.current_term += 1;
        self.state = RaftState::Candidate;
        self.voted_for = Some(self.id);
        
        // 发送投票请求
        let vote_request = VoteRequest {
            term: self.current_term,
            candidate_id: self.id,
            last_log_index: self.log.len() as u64,
            last_log_term: self.log.last().map(|e| e.term).unwrap_or(0),
        };
        
        // 这里应该发送给其他节点
        println!("Node {} started election for term {}", self.id, self.current_term);
        
        Ok(())
    }

    /// 处理投票请求
    pub fn handle_vote_request(&mut self, request: &VoteRequest) -> VoteResponse {
        if request.term < self.current_term {
            return VoteResponse {
                term: self.current_term,
                vote_granted: false,
            };
        }
        
        if request.term > self.current_term {
            self.current_term = request.term;
            self.state = RaftState::Follower;
            self.voted_for = None;
        }
        
        let can_vote = self.voted_for.is_none() || self.voted_for == Some(request.candidate_id);
        let log_ok = request.last_log_term > self.log.last().map(|e| e.term).unwrap_or(0) ||
                    (request.last_log_term == self.log.last().map(|e| e.term).unwrap_or(0) &&
                     request.last_log_index >= self.log.len() as u64);
        
        if can_vote && log_ok {
            self.voted_for = Some(request.candidate_id);
            VoteResponse {
                term: self.current_term,
                vote_granted: true,
            }
        } else {
            VoteResponse {
                term: self.current_term,
                vote_granted: false,
            }
        }
    }

    /// 成为领导者
    pub fn become_leader(&mut self) {
        self.state = RaftState::Leader;
        
        // 初始化领导者状态
        for node_id in 0..5 { // 假设有5个节点
            if node_id != self.id {
                self.next_index.insert(node_id, self.log.len() as u64 + 1);
                self.match_index.insert(node_id, 0);
            }
        }
        
        println!("Node {} became leader for term {}", self.id, self.current_term);
    }

    /// 添加日志条目
    pub fn append_log(&mut self, command: String) -> Result<u64, String> {
        if self.state != RaftState::Leader {
            return Err("Not leader".to_string());
        }
        
        let entry = LogEntry {
            term: self.current_term,
            index: self.log.len() as u64 + 1,
            command,
        };
        
        self.log.push(entry);
        Ok(self.log.len() as u64)
    }

    /// 提交日志
    pub fn commit_logs(&mut self) {
        for i in (self.commit_index + 1)..=self.log.len() as u64 {
            let mut replicated_count = 1; // 领导者自己
            
            for (node_id, match_index) in &self.match_index {
                if *node_id != self.id && *match_index >= i {
                    replicated_count += 1;
                }
            }
            
            if replicated_count > 5 / 2 { // 多数派
                self.commit_index = i;
            }
        }
        
        // 应用已提交的日志
        while self.last_applied < self.commit_index {
            self.last_applied += 1;
            if let Some(entry) = self.log.get(self.last_applied as usize - 1) {
                self.apply_command(&entry.command);
            }
        }
    }

    /// 应用命令
    fn apply_command(&self, command: &str) {
        println!("Applying command: {}", command);
        // 这里应该执行实际的命令
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteRequest {
    pub term: u64,
    pub candidate_id: u64,
    pub last_log_index: u64,
    pub last_log_term: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteResponse {
    pub term: u64,
    pub vote_granted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesRequest {
    pub term: u64,
    pub leader_id: u64,
    pub prev_log_index: u64,
    pub prev_log_term: u64,
    pub entries: Vec<LogEntry>,
    pub leader_commit: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesResponse {
    pub term: u64,
    pub success: bool,
}
```

### 定理 1.1 (Raft安全性)
Raft算法保证在任何时刻最多只有一个领导者。

**证明**：
假设存在两个领导者 $L_1$ 和 $L_2$，任期分别为 $T_1$ 和 $T_2$。

1. 领导者选举需要获得多数派投票
2. 每个节点在每个任期内最多只能投票给一个候选人
3. 如果 $T_1 \neq T_2$，则高任期的领导者会强制低任期的领导者退位
4. 如果 $T_1 = T_2$，则不可能有两个领导者获得多数派投票

因此，最多只能有一个领导者。

## 实时算法

### 定义 1.3 (实时调度算法)
实时调度算法在满足时间约束的前提下，为任务分配处理器时间。

### 算法 1.2 (最早截止时间优先调度)

```rust
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct RealTimeTask {
    pub id: u64,
    pub execution_time: Duration,
    pub deadline: Duration,
    pub period: Duration,
    pub priority: u8,
    pub arrival_time: Instant,
}

impl PartialEq for RealTimeTask {
    fn eq(&self, other: &Self) -> bool {
        self.deadline == other.deadline
    }
}

impl Eq for RealTimeTask {}

impl PartialOrd for RealTimeTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RealTimeTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // 最早截止时间优先
        other.deadline.cmp(&self.deadline)
    }
}

pub struct EDFScheduler {
    task_queue: BinaryHeap<RealTimeTask>,
    current_time: Instant,
    completed_tasks: Vec<RealTimeTask>,
    missed_deadlines: Vec<RealTimeTask>,
}

impl EDFScheduler {
    pub fn new() -> Self {
        Self {
            task_queue: BinaryHeap::new(),
            current_time: Instant::now(),
            completed_tasks: Vec::new(),
            missed_deadlines: Vec::new(),
        }
    }

    /// 添加任务
    pub fn add_task(&mut self, task: RealTimeTask) {
        self.task_queue.push(task);
    }

    /// 调度任务
    pub fn schedule(&mut self) -> Option<RealTimeTask> {
        if let Some(mut task) = self.task_queue.pop() {
            // 检查截止时间
            let time_until_deadline = task.deadline;
            let current_elapsed = self.current_time.elapsed();
            
            if current_elapsed > time_until_deadline {
                // 错过截止时间
                self.missed_deadlines.push(task);
                return None;
            }
            
            // 执行任务
            if current_elapsed + task.execution_time <= time_until_deadline {
                // 任务可以在截止时间内完成
                self.completed_tasks.push(task.clone());
                return Some(task);
            } else {
                // 任务无法在截止时间内完成
                self.missed_deadlines.push(task);
                return None;
            }
        }
        None
    }

    /// 检查可调度性
    pub fn is_schedulable(&self) -> bool {
        let total_utilization: f64 = self.task_queue.iter()
            .map(|task| {
                task.execution_time.as_secs_f64() / task.period.as_secs_f64()
            })
            .sum();
        
        total_utilization <= 1.0
    }

    /// 获取统计信息
    pub fn get_statistics(&self) -> SchedulerStatistics {
        SchedulerStatistics {
            total_tasks: self.completed_tasks.len() + self.missed_deadlines.len(),
            completed_tasks: self.completed_tasks.len(),
            missed_deadlines: self.missed_deadlines.len(),
            success_rate: if self.completed_tasks.len() + self.missed_deadlines.len() > 0 {
                self.completed_tasks.len() as f64 / 
                (self.completed_tasks.len() + self.missed_deadlines.len()) as f64
            } else {
                0.0
            },
        }
    }
}

#[derive(Debug)]
pub struct SchedulerStatistics {
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub missed_deadlines: usize,
    pub success_rate: f64,
}

/// 速率单调调度器
pub struct RMScheduler {
    task_queue: BinaryHeap<RealTimeTask>,
    current_time: Instant,
    completed_tasks: Vec<RealTimeTask>,
    missed_deadlines: Vec<RealTimeTask>,
}

impl RMScheduler {
    pub fn new() -> Self {
        Self {
            task_queue: BinaryHeap::new(),
            current_time: Instant::now(),
            completed_tasks: Vec::new(),
            missed_deadlines: Vec::new(),
        }
    }

    /// 添加任务（按周期排序）
    pub fn add_task(&mut self, mut task: RealTimeTask) {
        // 速率单调：周期越短，优先级越高
        task.priority = (1000.0 / task.period.as_secs_f64()) as u8;
        self.task_queue.push(task);
    }

    /// 检查Liu-Layland可调度性
    pub fn is_liu_layland_schedulable(&self) -> bool {
        let n = self.task_queue.len();
        if n == 0 {
            return true;
        }
        
        let utilization_bound = n as f64 * ((2.0_f64).powf(1.0 / n as f64) - 1.0);
        
        let total_utilization: f64 = self.task_queue.iter()
            .map(|task| {
                task.execution_time.as_secs_f64() / task.period.as_secs_f64()
            })
            .sum();
        
        total_utilization <= utilization_bound
    }
}
```

### 定理 1.2 (EDF最优性)
EDF算法在单处理器系统中是最优的，当且仅当处理器利用率不超过100%时，任务集合是可调度的。

**证明**：
1. **充分性**：如果处理器利用率 $\leq 100\%$，则EDF可以调度所有任务
2. **必要性**：如果处理器利用率 $> 100\%$，则任何算法都无法调度

## 优化算法

### 定义 1.4 (资源优化算法)
资源优化算法在满足性能约束的前提下，最小化资源消耗。

### 算法 1.3 (动态电压频率调节)

```rust
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct DVFSState {
    pub frequency: f64,  // GHz
    pub voltage: f64,    // V
    pub power: f64,      // W
    pub performance: f64, // 相对性能
}

pub struct DVFSController {
    states: Vec<DVFSState>,
    current_state: usize,
    workload_history: Vec<f64>,
    power_budget: f64,
    performance_threshold: f64,
}

impl DVFSController {
    pub fn new() -> Self {
        let states = vec![
            DVFSState { frequency: 0.8, voltage: 0.8, power: 0.5, performance: 0.6 },
            DVFSState { frequency: 1.2, voltage: 1.0, power: 1.0, performance: 0.8 },
            DVFSState { frequency: 1.6, voltage: 1.2, power: 1.8, performance: 1.0 },
            DVFSState { frequency: 2.0, voltage: 1.4, power: 3.0, performance: 1.2 },
        ];
        
        Self {
            states,
            current_state: 1,
            workload_history: Vec::new(),
            power_budget: 2.0,
            performance_threshold: 0.8,
        }
    }

    /// 根据工作负载调整频率
    pub fn adjust_frequency(&mut self, current_workload: f64) -> DVFSState {
        self.workload_history.push(current_workload);
        
        // 保持历史记录大小
        if self.workload_history.len() > 10 {
            self.workload_history.remove(0);
        }
        
        // 计算平均工作负载
        let avg_workload: f64 = self.workload_history.iter().sum::<f64>() / 
                               self.workload_history.len() as f64;
        
        // 选择合适的状态
        let target_state = self.select_optimal_state(avg_workload);
        
        if target_state != self.current_state {
            self.current_state = target_state;
            println!("DVFS: Switching to state {} (freq: {}GHz, power: {}W)", 
                    target_state, 
                    self.states[target_state].frequency,
                    self.states[target_state].power);
        }
        
        self.states[self.current_state].clone()
    }

    /// 选择最优状态
    fn select_optimal_state(&self, workload: f64) -> usize {
        let mut best_state = 0;
        let mut best_efficiency = 0.0;
        
        for (i, state) in self.states.iter().enumerate() {
            // 检查性能约束
            if state.performance < self.performance_threshold {
                continue;
            }
            
            // 检查功耗约束
            if state.power > self.power_budget {
                continue;
            }
            
            // 计算能效比
            let efficiency = workload / state.power;
            
            if efficiency > best_efficiency {
                best_efficiency = efficiency;
                best_state = i;
            }
        }
        
        best_state
    }

    /// 预测功耗
    pub fn predict_power_consumption(&self, duration: std::time::Duration) -> f64 {
        let current_power = self.states[self.current_state].power;
        let hours = duration.as_secs_f64() / 3600.0;
        current_power * hours
    }

    /// 获取能效统计
    pub fn get_efficiency_stats(&self) -> EfficiencyStats {
        let total_workload: f64 = self.workload_history.iter().sum();
        let total_power = self.states[self.current_state].power * 
                         self.workload_history.len() as f64;
        
        EfficiencyStats {
            total_workload,
            total_power,
            efficiency: if total_power > 0.0 { total_workload / total_power } else { 0.0 },
            current_frequency: self.states[self.current_state].frequency,
            current_voltage: self.states[self.current_state].voltage,
        }
    }
}

#[derive(Debug)]
pub struct EfficiencyStats {
    pub total_workload: f64,
    pub total_power: f64,
    pub efficiency: f64,
    pub current_frequency: f64,
    pub current_voltage: f64,
}
```

## 安全算法

### 定义 1.5 (加密算法)
加密算法将明文转换为密文，确保数据机密性。

### 算法 1.4 (AES加密)

```rust
use aes::Aes128;
use aes::cipher::{
    BlockEncrypt, BlockDecrypt,
    KeyInit,
    generic_array::GenericArray,
};
use rand::Rng;

pub struct IoTSecurity {
    key: [u8; 16],
    iv: [u8; 16],
}

impl IoTSecurity {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let key: [u8; 16] = rng.gen();
        let iv: [u8; 16] = rng.gen();
        
        Self { key, iv }
    }

    /// AES加密
    pub fn encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>, String> {
        let cipher = Aes128::new_from_slice(&self.key)
            .map_err(|e| format!("Key error: {}", e))?;
        
        let mut encrypted = Vec::new();
        
        // 分块加密
        for chunk in plaintext.chunks(16) {
            let mut block = GenericArray::clone_from_slice(chunk);
            
            // 填充最后一个块
            if chunk.len() < 16 {
                let padding = 16 - chunk.len();
                for i in chunk.len()..16 {
                    block[i] = padding as u8;
                }
            }
            
            cipher.encrypt_block(&mut block);
            encrypted.extend_from_slice(&block);
        }
        
        Ok(encrypted)
    }

    /// AES解密
    pub fn decrypt(&self, ciphertext: &[u8]) -> Result<Vec<u8>, String> {
        let cipher = Aes128::new_from_slice(&self.key)
            .map_err(|e| format!("Key error: {}", e))?;
        
        let mut decrypted = Vec::new();
        
        // 分块解密
        for chunk in ciphertext.chunks(16) {
            let mut block = GenericArray::clone_from_slice(chunk);
            cipher.decrypt_block(&mut block);
            
            // 移除填充
            if chunk == ciphertext.chunks(16).last().unwrap() {
                let padding = block[15] as usize;
                if padding <= 16 {
                    decrypted.extend_from_slice(&block[..16-padding]);
                } else {
                    decrypted.extend_from_slice(&block);
                }
            } else {
                decrypted.extend_from_slice(&block);
            }
        }
        
        Ok(decrypted)
    }

    /// 生成哈希
    pub fn hash(&self, data: &[u8]) -> [u8; 32] {
        use sha2::{Sha256, Digest};
        
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().into()
    }

    /// 数字签名
    pub fn sign(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        // 简化的数字签名实现
        let hash = self.hash(data);
        let signature = self.encrypt(&hash)?;
        Ok(signature)
    }

    /// 验证签名
    pub fn verify(&self, data: &[u8], signature: &[u8]) -> Result<bool, String> {
        let hash = self.hash(data);
        let decrypted_signature = self.decrypt(signature)?;
        
        Ok(hash.as_slice() == decrypted_signature.as_slice())
    }
}

/// 安全通信协议
pub struct SecureProtocol {
    security: IoTSecurity,
    session_key: Option<[u8; 16]>,
}

impl SecureProtocol {
    pub fn new() -> Self {
        Self {
            security: IoTSecurity::new(),
            session_key: None,
        }
    }

    /// 建立安全会话
    pub fn establish_session(&mut self, challenge: &[u8]) -> Result<Vec<u8>, String> {
        // 生成会话密钥
        let mut rng = rand::thread_rng();
        let session_key: [u8; 16] = rng.gen();
        self.session_key = Some(session_key);
        
        // 加密挑战响应
        let response = self.security.encrypt(challenge)?;
        Ok(response)
    }

    /// 发送安全消息
    pub fn send_secure_message(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        if let Some(session_key) = self.session_key {
            // 使用会话密钥加密
            let mut temp_security = IoTSecurity::new();
            temp_security.key = session_key;
            temp_security.encrypt(message)
        } else {
            Err("No active session".to_string())
        }
    }

    /// 接收安全消息
    pub fn receive_secure_message(&self, encrypted_message: &[u8]) -> Result<Vec<u8>, String> {
        if let Some(session_key) = self.session_key {
            // 使用会话密钥解密
            let mut temp_security = IoTSecurity::new();
            temp_security.key = session_key;
            temp_security.decrypt(encrypted_message)
        } else {
            Err("No active session".to_string())
        }
    }
}
```

## 结论

本文分析了IoT系统中的核心算法：

1. **分布式算法**：Raft共识算法确保分布式系统的一致性
2. **实时算法**：EDF和RMS调度算法满足实时约束
3. **优化算法**：DVFS算法优化能耗和性能
4. **安全算法**：AES加密和数字签名保障数据安全

这些算法为IoT系统提供了理论基础和实现方案，确保系统的正确性、实时性和安全性。

---

**参考文献**：
1. Ongaro, D., & Ousterhout, J. (2014). In search of an understandable consensus algorithm. USENIX Annual Technical Conference.
2. Liu, C. L., & Layland, J. W. (1973). Scheduling algorithms for multiprogramming in a hard-real-time environment. Journal of the ACM, 20(1), 46-61.
3. Dertouzos, M. L. (1974). Control robotics: The procedural control of physical processes. IFIP Congress.
4. Daemen, J., & Rijmen, V. (2002). The design of Rijndael: AES-the advanced encryption standard. Springer Science & Business Media.
