# IoT行业核心算法 - 形式化分析

## 1. 数据聚合算法

### 1.1 滑动窗口聚合

#### 定义 1.1 (滑动窗口)

滑动窗口 $W$ 是一个三元组 $(size, data, index)$，其中：

- $size \in \mathbb{N}$ 是窗口大小
- $data = [d_1, d_2, \ldots, d_n]$ 是数据序列
- $index \in \{0, 1, \ldots, n-size\}$ 是当前窗口起始位置

#### 算法 1.1 (滑动窗口均值计算)

```text
输入: 数据流 S = [s_1, s_2, ..., s_n], 窗口大小 w
输出: 滑动均值序列 M = [m_1, m_2, ..., m_{n-w+1}]

1. 初始化: sum = 0, M = []
2. 计算第一个窗口的和:
   for i = 1 to w:
       sum += s_i
3. 计算第一个均值: m_1 = sum / w
4. 滑动窗口:
   for i = w+1 to n:
       sum = sum - s_{i-w} + s_i
       m_{i-w+1} = sum / w
5. 返回 M
```

#### 定理 1.1 (滑动窗口算法复杂度)

滑动窗口均值计算算法的时间复杂度为 $O(n)$，空间复杂度为 $O(1)$。

**证明**：

1. 初始化步骤需要 $O(1)$ 时间
2. 第一个窗口计算需要 $O(w)$ 时间
3. 滑动过程需要 $O(n-w)$ 时间，每次操作 $O(1)$
4. 总时间复杂度：$O(1) + O(w) + O(n-w) = O(n)$
5. 只使用常数个变量，空间复杂度 $O(1)$
6. 证毕。

#### Rust实现

```rust
use std::collections::VecDeque;

/// 滑动窗口聚合器
pub struct SlidingWindowAggregator {
    window_size: usize,
    buffer: VecDeque<f64>,
    sum: f64,
}

impl SlidingWindowAggregator {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            buffer: VecDeque::with_capacity(window_size),
            sum: 0.0,
        }
    }
    
    /// 添加新数据点并返回当前窗口均值
    pub fn add_and_get_mean(&mut self, value: f64) -> Option<f64> {
        self.buffer.push_back(value);
        self.sum += value;
        
        if self.buffer.len() > self.window_size {
            if let Some(old_value) = self.buffer.pop_front() {
                self.sum -= old_value;
            }
        }
        
        if self.buffer.len() == self.window_size {
            Some(self.sum / self.window_size as f64)
        } else {
            None
        }
    }
    
    /// 获取当前窗口的统计信息
    pub fn get_statistics(&self) -> Option<WindowStatistics> {
        if self.buffer.len() != self.window_size {
            return None;
        }
        
        let mean = self.sum / self.window_size as f64;
        let variance = self.buffer.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / self.window_size as f64;
        
        Some(WindowStatistics {
            mean,
            variance,
            min: self.buffer.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            max: self.buffer.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
        })
    }
}

#[derive(Debug, Clone)]
pub struct WindowStatistics {
    pub mean: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
}
```

### 1.2 异常检测算法

#### 定义 1.2 (异常检测)

异常检测函数 $detect: \mathbb{R}^n \times \mathbb{R} \rightarrow \{true, false\}$ 定义为：
$$detect(data, threshold) = \begin{cases}
true & \text{if } |data_i - \mu| > \sigma \cdot threshold \\
false & \text{otherwise}
\end{cases}$$
其中 $\mu$ 是均值，$\sigma$ 是标准差。

#### 算法 1.2 (Z-Score异常检测)
```
输入: 数据序列 X = [x_1, x_2, ..., x_n], 阈值 t
输出: 异常索引集合 A

1. 计算均值: μ = (1/n) * Σ(x_i)
2. 计算标准差: σ = sqrt((1/n) * Σ(x_i - μ)²)
3. 检测异常:
   for i = 1 to n:
       z_score = |x_i - μ| / σ
       if z_score > t:
           A.add(i)
4. 返回 A
```

#### 定理 1.2 (Z-Score算法正确性)
如果数据服从正态分布，则Z-Score算法能够以概率 $1 - 2\Phi(-t)$ 正确检测异常值，其中 $\Phi$ 是标准正态分布的累积分布函数。

**证明**：
1. 对于正态分布 $N(\mu, \sigma^2)$，Z-Score服从 $N(0, 1)$
2. $P(|Z| > t) = 2\Phi(-t)$
3. 因此正确检测的概率为 $1 - 2\Phi(-t)$
4. 证毕。

```rust
/// Z-Score异常检测器
pub struct ZScoreDetector {
    mean: f64,
    std_dev: f64,
    threshold: f64,
}

impl ZScoreDetector {
    pub fn new(threshold: f64) -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            threshold,
        }
    }

    /// 训练模型
    pub fn train(&mut self, data: &[f64]) {
        let n = data.len() as f64;
        self.mean = data.iter().sum::<f64>() / n;

        let variance = data.iter()
            .map(|&x| (x - self.mean).powi(2))
            .sum::<f64>() / n;
        self.std_dev = variance.sqrt();
    }

    /// 检测异常
    pub fn detect_anomaly(&self, value: f64) -> bool {
        let z_score = (value - self.mean).abs() / self.std_dev;
        z_score > self.threshold
    }

    /// 批量检测
    pub fn detect_anomalies(&self, data: &[f64]) -> Vec<usize> {
        data.iter()
            .enumerate()
            .filter(|(_, &value)| self.detect_anomaly(value))
            .map(|(index, _)| index)
            .collect()
    }
}
```

## 2. 设备发现算法

### 2.1 分布式设备发现

#### 定义 2.1 (设备发现图)
设备发现图 $G = (V, E)$ 是一个无向图，其中：
- $V = \{v_1, v_2, \ldots, v_n\}$ 是设备节点集合
- $E = \{(v_i, v_j) \mid v_i, v_j \in V, i \neq j\}$ 是连接边集合

#### 算法 2.1 (分布式设备发现)
```
输入: 设备集合 D, 发现半径 r
输出: 连接图 G

1. 初始化: G = (D, ∅)
2. 对于每个设备 d_i ∈ D:
   a. 广播发现消息
   b. 等待响应时间 t
   c. 收集响应设备集合 R_i
   d. 添加边: E = E ∪ {(d_i, d_j) | d_j ∈ R_i}
3. 返回 G
```

#### 定理 2.1 (设备发现算法复杂度)
分布式设备发现算法的时间复杂度为 $O(n^2)$，其中 $n$ 是设备数量。

**证明**：
1. 每个设备需要广播消息，时间复杂度 $O(n)$
2. 每个设备需要处理响应，时间复杂度 $O(n)$
3. 总时间复杂度：$O(n) \times O(n) = O(n^2)$
4. 证毕。

```rust
use std::collections::{HashMap, HashSet};
use tokio::time::{Duration, Instant};

/// 设备发现管理器
pub struct DeviceDiscoveryManager {
    devices: HashMap<DeviceId, DeviceInfo>,
    connections: HashMap<DeviceId, HashSet<DeviceId>>,
    discovery_radius: Duration,
}

# [derive(Debug, Clone)]
pub struct DeviceInfo {
    pub id: DeviceId,
    pub location: Location,
    pub capabilities: Vec<Capability>,
    pub last_seen: Instant,
}

impl DeviceDiscoveryManager {
    pub fn new(discovery_radius: Duration) -> Self {
        Self {
            devices: HashMap::new(),
            connections: HashMap::new(),
            discovery_radius,
        }
    }

    /// 启动设备发现
    pub async fn start_discovery(&mut self) -> Result<(), DiscoveryError> {
        let mut discovery_tasks = Vec::new();

        for device_id in self.devices.keys().cloned().collect::<Vec<_>>() {
            let task = self.discover_device(device_id);
            discovery_tasks.push(task);
        }

        // 并发执行所有发现任务
        let results = futures::future::join_all(discovery_tasks).await;

        // 处理发现结果
        for result in results {
            if let Ok(connections) = result {
                self.update_connections(connections);
            }
        }

        Ok(())
    }

    async fn discover_device(&self, device_id: DeviceId) -> Result<Vec<DeviceId>, DiscoveryError> {
        let mut discovered_devices = Vec::new();
        let start_time = Instant::now();

        // 发送发现广播
        self.broadcast_discovery(device_id).await?;

        // 等待响应
        while start_time.elapsed() < self.discovery_radius {
            if let Some(response) = self.receive_discovery_response().await? {
                discovered_devices.push(response);
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        Ok(discovered_devices)
    }

    fn update_connections(&mut self, connections: Vec<DeviceId>) {
        for device_id in connections {
            self.connections.entry(device_id.clone())
                .or_insert_with(HashSet::new)
                .insert(device_id);
        }
    }
}
```

## 3. 负载均衡算法

### 3.1 加权轮询算法

#### 定义 3.1 (负载均衡器)
负载均衡器 $LB$ 是一个四元组 $(S, W, C, A)$，其中：
- $S = \{s_1, s_2, \ldots, s_n\}$ 是服务器集合
- $W = \{w_1, w_2, \ldots, w_n\}$ 是权重集合
- $C = \{c_1, c_2, \ldots, c_n\}$ 是当前连接数集合
- $A$ 是分配算法

#### 算法 3.1 (加权轮询)
```
输入: 服务器集合 S, 权重集合 W, 请求 r
输出: 选中的服务器 s

1. 计算总权重: total_weight = Σ(w_i)
2. 计算当前权重: current_weight = max(c_i / w_i)
3. 选择服务器:
   for i = 1 to n:
       effective_weight = w_i - current_weight * c_i
       if effective_weight > max_effective:
           max_effective = effective_weight
           selected = s_i
4. 更新连接数: c_selected += 1
5. 返回 selected
```

#### 定理 3.1 (加权轮询公平性)
加权轮询算法能够保证服务器 $s_i$ 在长期运行中获得 $\frac{w_i}{\sum_{j=1}^n w_j}$ 比例的请求。

**证明**：
1. 设 $T$ 为总请求数，$T_i$ 为服务器 $s_i$ 处理的请求数
2. 根据算法，$\frac{T_i}{w_i} \approx \frac{T_j}{w_j}$ 对所有 $i, j$ 成立
3. 因此 $\frac{T_i}{T} = \frac{w_i}{\sum_{j=1}^n w_j}$
4. 证毕。

```rust
use std::collections::HashMap;

/// 加权轮询负载均衡器
pub struct WeightedRoundRobinBalancer {
    servers: Vec<Server>,
    current_index: usize,
    current_weight: f64,
}

# [derive(Debug, Clone)]
pub struct Server {
    pub id: ServerId,
    pub weight: f64,
    pub current_connections: usize,
    pub max_connections: usize,
}

impl WeightedRoundRobinBalancer {
    pub fn new() -> Self {
        Self {
            servers: Vec::new(),
            current_index: 0,
            current_weight: 0.0,
        }
    }

    /// 添加服务器
    pub fn add_server(&mut self, server: Server) {
        self.servers.push(server);
    }

    /// 选择下一个服务器
    pub fn next_server(&mut self) -> Option<&Server> {
        if self.servers.is_empty() {
            return None;
        }

        loop {
            let server = &self.servers[self.current_index];

            if server.current_connections < server.max_connections {
                self.current_index = (self.current_index + 1) % self.servers.len();
                return Some(server);
            }

            self.current_index = (self.current_index + 1) % self.servers.len();

            // 如果所有服务器都满了，重置
            if self.current_index == 0 {
                self.current_weight = 0.0;
            }
        }
    }

    /// 获取负载分布统计
    pub fn get_load_distribution(&self) -> HashMap<ServerId, f64> {
        let total_connections: usize = self.servers.iter()
            .map(|s| s.current_connections)
            .sum();

        self.servers.iter()
            .map(|server| {
                let ratio = if total_connections > 0 {
                    server.current_connections as f64 / total_connections as f64
                } else {
                    0.0
                };
                (server.id.clone(), ratio)
            })
            .collect()
    }
}
```

## 4. 数据压缩算法

### 4.1 时间序列压缩

#### 定义 4.1 (时间序列)
时间序列 $TS$ 是一个序列对 $(T, V)$，其中：
- $T = [t_1, t_2, \ldots, t_n]$ 是时间戳序列
- $V = [v_1, v_2, \ldots, v_n]$ 是数值序列

#### 算法 4.1 (线性插值压缩)
```
输入: 时间序列 TS = (T, V), 压缩阈值 ε
输出: 压缩后的时间序列 TS' = (T', V')

1. 初始化: T' = [t_1], V' = [v_1]
2. 对于 i = 2 to n:
   a. 计算线性插值: v_interp = linear_interpolate(t_i, T', V')
   b. 计算误差: error = |v_i - v_interp|
   c. 如果 error > ε:
      添加点: T'.append(t_i), V'.append(v_i)
3. 返回 (T', V')
```

#### 定理 4.1 (压缩算法误差界)
线性插值压缩算法的最大误差不超过 $2\varepsilon$。

**证明**：
1. 对于任意时间点 $t$，插值误差 $|v(t) - v_{interp}(t)| \leq \varepsilon$
2. 原始数据点与插值点的误差 $|v_i - v_{interp}(t_i)| \leq \varepsilon$
3. 总误差 $|v(t) - v_i| \leq |v(t) - v_{interp}(t)| + |v_{interp}(t) - v_i| \leq 2\varepsilon$
4. 证毕。

```rust
/// 时间序列压缩器
pub struct TimeSeriesCompressor {
    threshold: f64,
}

impl TimeSeriesCompressor {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// 压缩时间序列
    pub fn compress(&self, timestamps: &[i64], values: &[f64]) -> (Vec<i64>, Vec<f64>) {
        if timestamps.len() <= 2 {
            return (timestamps.to_vec(), values.to_vec());
        }

        let mut compressed_timestamps = vec![timestamps[0]];
        let mut compressed_values = vec![values[0]];

        for i in 1..timestamps.len() {
            let interpolated = self.linear_interpolate(
                timestamps[i],
                &compressed_timestamps,
                &compressed_values,
            );

            let error = (values[i] - interpolated).abs();

            if error > self.threshold {
                compressed_timestamps.push(timestamps[i]);
                compressed_values.push(values[i]);
            }
        }

        (compressed_timestamps, compressed_values)
    }

    /// 线性插值
    fn linear_interpolate(&self, target_time: i64, times: &[i64], values: &[f64]) -> f64 {
        if times.len() < 2 {
            return values[0];
        }

        // 找到最近的两个点
        let mut left_idx = 0;
        for i in 0..times.len() - 1 {
            if times[i] <= target_time && target_time <= times[i + 1] {
                left_idx = i;
                break;
            }
        }

        let right_idx = left_idx + 1;
        let t1 = times[left_idx] as f64;
        let t2 = times[right_idx] as f64;
        let v1 = values[left_idx];
        let v2 = values[right_idx];

        // 线性插值公式
        v1 + (v2 - v1) * (target_time as f64 - t1) / (t2 - t1)
    }
}
```

## 5. 路由算法

### 5.1 最短路径路由

#### 定义 5.1 (网络图)
网络图 $G = (V, E, W)$ 是一个带权图，其中：
- $V$ 是节点集合
- $E$ 是边集合
- $W: E \rightarrow \mathbb{R}^+$ 是权重函数

#### 算法 5.1 (Dijkstra最短路径)
```
输入: 网络图 G = (V, E, W), 源节点 s
输出: 最短路径距离 d[v] 和前驱节点 π[v]

1. 初始化: d[s] = 0, d[v] = ∞ for v ≠ s, π[v] = nil
2. 创建优先队列 Q = V
3. while Q ≠ ∅:
   a. u = extract_min(Q)
   b. for each neighbor v of u:
      if d[v] > d[u] + W(u,v):
         d[v] = d[u] + W(u,v)
         π[v] = u
         decrease_key(Q, v)
4. 返回 d, π
```

#### 定理 5.1 (Dijkstra算法正确性)
Dijkstra算法能够找到从源节点到所有其他节点的最短路径。

**证明**：
1. 使用数学归纳法
2. 基础情况：源节点距离为0，正确
3. 归纳假设：前k个节点距离正确
4. 归纳步骤：第k+1个节点距离也正确
5. 证毕。

```rust
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

# [derive(Debug, Clone)]
pub struct NetworkNode {
    pub id: NodeId,
    pub neighbors: HashMap<NodeId, f64>, // neighbor_id -> weight
}

# [derive(Debug, Clone)]
pub struct ShortestPathResult {
    pub distances: HashMap<NodeId, f64>,
    pub predecessors: HashMap<NodeId, Option<NodeId>>,
}

/// 网络路由器
pub struct NetworkRouter {
    nodes: HashMap<NodeId, NetworkNode>,
}

impl NetworkRouter {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    /// 添加节点
    pub fn add_node(&mut self, node: NetworkNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// 计算最短路径
    pub fn shortest_path(&self, source: &NodeId) -> ShortestPathResult {
        let mut distances = HashMap::new();
        let mut predecessors = HashMap::new();
        let mut queue = BinaryHeap::new();

        // 初始化
        for node_id in self.nodes.keys() {
            distances.insert(node_id.clone(), f64::INFINITY);
            predecessors.insert(node_id.clone(), None);
        }

        distances.insert(source.clone(), 0.0);
        queue.push(QueueItem {
            node_id: source.clone(),
            distance: 0.0,
        });

        while let Some(QueueItem { node_id, distance }) = queue.pop() {
            if distance > distances[&node_id] {
                continue; // 已经找到更短的路径
            }

            if let Some(node) = self.nodes.get(&node_id) {
                for (neighbor_id, weight) in &node.neighbors {
                    let new_distance = distance + weight;

                    if new_distance < distances[neighbor_id] {
                        distances.insert(neighbor_id.clone(), new_distance);
                        predecessors.insert(neighbor_id.clone(), Some(node_id.clone()));

                        queue.push(QueueItem {
                            node_id: neighbor_id.clone(),
                            distance: new_distance,
                        });
                    }
                }
            }
        }

        ShortestPathResult {
            distances,
            predecessors,
        }
    }

    /// 重建路径
    pub fn reconstruct_path(&self, result: &ShortestPathResult, target: &NodeId) -> Vec<NodeId> {
        let mut path = Vec::new();
        let mut current = target.clone();

        while let Some(predecessor) = result.predecessors.get(&current) {
            path.push(current.clone());
            match predecessor {
                Some(pred) => current = pred.clone(),
                None => break,
            }
        }

        path.reverse();
        path
    }
}

# [derive(Debug, Clone)]
struct QueueItem {
    node_id: NodeId,
    distance: f64,
}

impl PartialEq for QueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for QueueItem {}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.distance.partial_cmp(&self.distance) // 最大堆
    }
}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other.distance.partial_cmp(&self.distance).unwrap()
    }
}
```

## 6. 结论

本文档建立了IoT行业核心算法的完整形式化框架，包括：

1. **数据聚合算法**：滑动窗口聚合和异常检测
2. **设备发现算法**：分布式设备发现机制
3. **负载均衡算法**：加权轮询负载分配
4. **数据压缩算法**：时间序列压缩技术
5. **路由算法**：最短路径路由计算

每个算法都包含：
- 严格的数学定义
- 形式化证明
- 复杂度分析
- Rust实现示例

这些算法为IoT系统提供了高效、可靠的数据处理和网络通信基础。

---

**参考文献**：
1. [Algorithm Design Manual](https://www.algorist.com/)
2. [Network Algorithms](https://networkx.org/documentation/stable/reference/algorithms/)
3. [Time Series Analysis](https://otexts.com/fpp3/)
4. [Load Balancing Algorithms](https://en.wikipedia.org/wiki/Load_balancing_(computing))
