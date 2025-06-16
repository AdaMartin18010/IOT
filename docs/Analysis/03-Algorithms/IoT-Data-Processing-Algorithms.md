# IoT数据处理算法

## 目录

1. [引言](#引言)
2. [数据压缩算法](#数据压缩算法)
3. [数据流处理算法](#数据流处理算法)
4. [机器学习算法](#机器学习算法)
5. [时间序列分析](#时间序列分析)
6. [Rust实现](#rust实现)
7. [结论](#结论)

## 引言

IoT系统产生海量数据，需要高效的算法进行实时处理、压缩和分析。本文分析IoT数据处理的核心算法及其形式化理论。

### 定义 1.1 (IoT数据处理)
IoT数据处理是一个四元组 $\mathcal{P} = (D, A, T, Q)$，其中：
- $D$ 是数据流集合
- $A$ 是算法集合
- $T$ 是时间约束
- $Q$ 是质量指标

## 数据压缩算法

### 定义 1.2 (数据压缩)
数据压缩算法将原始数据 $X$ 转换为压缩表示 $C(X)$，满足：
$$|C(X)| < |X|$$

### 定理 1.1 (压缩率下界)
对于任意无损压缩算法，存在数据使得压缩率不能无限小。

**证明**：
设压缩算法 $C$ 将长度为 $n$ 的数据压缩为长度 $m$。

对于 $n$ 位数据，总共有 $2^n$ 种可能。
对于 $m$ 位压缩数据，总共有 $2^m$ 种可能。

如果 $m < n$，则 $2^m < 2^n$，无法表示所有原始数据。
因此，存在数据无法被压缩到任意小的长度。

### 算法 1.1 (LZ77压缩算法)

```rust
use std::collections::HashMap;

/// LZ77压缩器
pub struct LZ77Compressor {
    window_size: usize,
    look_ahead_size: usize,
}

#[derive(Debug, Clone)]
pub struct LZ77Token {
    pub offset: usize,
    pub length: usize,
    pub next_char: Option<u8>,
}

impl LZ77Compressor {
    pub fn new(window_size: usize, look_ahead_size: usize) -> Self {
        Self {
            window_size,
            look_ahead_size,
        }
    }

    /// 压缩数据
    pub fn compress(&self, data: &[u8]) -> Vec<LZ77Token> {
        let mut tokens = Vec::new();
        let mut pos = 0;
        
        while pos < data.len() {
            let (offset, length) = self.find_longest_match(data, pos);
            
            let next_char = if pos + length < data.len() {
                Some(data[pos + length])
            } else {
                None
            };
            
            tokens.push(LZ77Token {
                offset,
                length,
                next_char,
            });
            
            pos += length + 1;
        }
        
        tokens
    }

    /// 查找最长匹配
    fn find_longest_match(&self, data: &[u8], current_pos: usize) -> (usize, usize) {
        let window_start = if current_pos > self.window_size {
            current_pos - self.window_size
        } else {
            0
        };
        
        let look_ahead_end = std::cmp::min(
            current_pos + self.look_ahead_size,
            data.len()
        );
        
        let mut best_offset = 0;
        let mut best_length = 0;
        
        // 在滑动窗口中查找匹配
        for start in window_start..current_pos {
            let mut length = 0;
            while current_pos + length < look_ahead_end &&
                   start + length < current_pos &&
                   data[start + length] == data[current_pos + length] {
                length += 1;
            }
            
            if length > best_length {
                best_length = length;
                best_offset = current_pos - start;
            }
        }
        
        (best_offset, best_length)
    }

    /// 解压缩数据
    pub fn decompress(&self, tokens: &[LZ77Token]) -> Vec<u8> {
        let mut output = Vec::new();
        
        for token in tokens {
            if token.length > 0 {
                // 复制之前的数据
                let start = output.len() - token.offset;
                for i in 0..token.length {
                    output.push(output[start + i]);
                }
            }
            
            // 添加下一个字符
            if let Some(c) = token.next_char {
                output.push(c);
            }
        }
        
        output
    }

    /// 计算压缩率
    pub fn compression_ratio(&self, original: &[u8], compressed: &[LZ77Token]) -> f64 {
        let original_size = original.len();
        let compressed_size = compressed.len() * std::mem::size_of::<LZ77Token>();
        
        1.0 - (compressed_size as f64 / original_size as f64)
    }
}

/// 霍夫曼编码器
pub struct HuffmanEncoder {
    frequency_table: HashMap<u8, u32>,
    code_table: HashMap<u8, Vec<bool>>,
}

impl HuffmanEncoder {
    pub fn new() -> Self {
        Self {
            frequency_table: HashMap::new(),
            code_table: HashMap::new(),
        }
    }

    /// 构建频率表
    pub fn build_frequency_table(&mut self, data: &[u8]) {
        self.frequency_table.clear();
        
        for &byte in data {
            *self.frequency_table.entry(byte).or_insert(0) += 1;
        }
    }

    /// 构建霍夫曼树
    pub fn build_huffman_tree(&mut self) -> HuffmanNode {
        let mut nodes: Vec<HuffmanNode> = self.frequency_table
            .iter()
            .map(|(&byte, &freq)| {
                HuffmanNode::Leaf { byte, frequency: freq }
            })
            .collect();
        
        while nodes.len() > 1 {
            nodes.sort_by(|a, b| a.frequency().cmp(&b.frequency()));
            
            let left = nodes.remove(0);
            let right = nodes.remove(0);
            
            let internal = HuffmanNode::Internal {
                left: Box::new(left),
                right: Box::new(right),
                frequency: left.frequency() + right.frequency(),
            };
            
            nodes.push(internal);
        }
        
        nodes.remove(0)
    }

    /// 生成编码表
    pub fn generate_code_table(&mut self, root: &HuffmanNode) {
        self.code_table.clear();
        self.generate_codes_recursive(root, Vec::new());
    }

    fn generate_codes_recursive(&mut self, node: &HuffmanNode, code: Vec<bool>) {
        match node {
            HuffmanNode::Leaf { byte, .. } => {
                self.code_table.insert(*byte, code);
            }
            HuffmanNode::Internal { left, right, .. } => {
                let mut left_code = code.clone();
                left_code.push(false);
                self.generate_codes_recursive(left, left_code);
                
                let mut right_code = code;
                right_code.push(true);
                self.generate_codes_recursive(right, right_code);
            }
        }
    }

    /// 编码数据
    pub fn encode(&self, data: &[u8]) -> Vec<bool> {
        let mut encoded = Vec::new();
        
        for &byte in data {
            if let Some(code) = self.code_table.get(&byte) {
                encoded.extend(code);
            }
        }
        
        encoded
    }

    /// 解码数据
    pub fn decode(&self, encoded: &[bool], root: &HuffmanNode) -> Vec<u8> {
        let mut decoded = Vec::new();
        let mut current = root;
        
        for &bit in encoded {
            match current {
                HuffmanNode::Leaf { byte, .. } => {
                    decoded.push(*byte);
                    current = root;
                }
                HuffmanNode::Internal { left, right, .. } => {
                    current = if bit { right } else { left };
                }
            }
        }
        
        // 处理最后一个字符
        if let HuffmanNode::Leaf { byte, .. } = current {
            decoded.push(*byte);
        }
        
        decoded
    }
}

#[derive(Debug, Clone)]
pub enum HuffmanNode {
    Leaf { byte: u8, frequency: u32 },
    Internal { left: Box<HuffmanNode>, right: Box<HuffmanNode>, frequency: u32 },
}

impl HuffmanNode {
    pub fn frequency(&self) -> u32 {
        match self {
            HuffmanNode::Leaf { frequency, .. } => *frequency,
            HuffmanNode::Internal { frequency, .. } => *frequency,
        }
    }
}
```

## 数据流处理算法

### 定义 1.3 (数据流)
数据流是一个无限序列 $S = (s_1, s_2, s_3, \ldots)$，其中每个元素 $s_i$ 在时间 $t_i$ 到达。

### 定义 1.4 (滑动窗口)
滑动窗口是数据流的一个有限子序列 $W(t) = (s_{t-w+1}, s_{t-w+2}, \ldots, s_t)$，其中 $w$ 是窗口大小。

### 算法 1.2 (滑动窗口聚合)

```rust
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// 滑动窗口聚合器
pub struct SlidingWindowAggregator<T> {
    window_size: Duration,
    data_points: VecDeque<(T, Instant)>,
}

impl<T> SlidingWindowAggregator<T> {
    pub fn new(window_size: Duration) -> Self {
        Self {
            window_size,
            data_points: VecDeque::new(),
        }
    }

    /// 添加数据点
    pub fn add_data_point(&mut self, value: T, timestamp: Instant) {
        // 移除过期数据
        self.remove_expired_data(timestamp);
        
        // 添加新数据
        self.data_points.push_back((value, timestamp));
    }

    /// 移除过期数据
    fn remove_expired_data(&mut self, current_time: Instant) {
        while let Some((_, timestamp)) = self.data_points.front() {
            if current_time.duration_since(*timestamp) > self.window_size {
                self.data_points.pop_front();
            } else {
                break;
            }
        }
    }

    /// 获取窗口大小
    pub fn window_size(&self) -> usize {
        self.data_points.len()
    }

    /// 清空窗口
    pub fn clear(&mut self) {
        self.data_points.clear();
    }
}

/// 数值滑动窗口聚合器
pub struct NumericSlidingWindowAggregator {
    aggregator: SlidingWindowAggregator<f64>,
}

impl NumericSlidingWindowAggregator {
    pub fn new(window_size: Duration) -> Self {
        Self {
            aggregator: SlidingWindowAggregator::new(window_size),
        }
    }

    /// 添加数值
    pub fn add_value(&mut self, value: f64, timestamp: Instant) {
        self.aggregator.add_data_point(value, timestamp);
    }

    /// 计算平均值
    pub fn mean(&self) -> Option<f64> {
        if self.aggregator.data_points.is_empty() {
            None
        } else {
            let sum: f64 = self.aggregator.data_points
                .iter()
                .map(|(value, _)| value)
                .sum();
            Some(sum / self.aggregator.data_points.len() as f64)
        }
    }

    /// 计算中位数
    pub fn median(&self) -> Option<f64> {
        if self.aggregator.data_points.is_empty() {
            None
        } else {
            let mut values: Vec<f64> = self.aggregator.data_points
                .iter()
                .map(|(value, _)| *value)
                .collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let len = values.len();
            if len % 2 == 0 {
                Some((values[len / 2 - 1] + values[len / 2]) / 2.0)
            } else {
                Some(values[len / 2])
            }
        }
    }

    /// 计算标准差
    pub fn standard_deviation(&self) -> Option<f64> {
        if let Some(mean) = self.mean() {
            let variance: f64 = self.aggregator.data_points
                .iter()
                .map(|(value, _)| (value - mean).powi(2))
                .sum::<f64>() / self.aggregator.data_points.len() as f64;
            Some(variance.sqrt())
        } else {
            None
        }
    }

    /// 计算最大值
    pub fn max(&self) -> Option<f64> {
        self.aggregator.data_points
            .iter()
            .map(|(value, _)| *value)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
    }

    /// 计算最小值
    pub fn min(&self) -> Option<f64> {
        self.aggregator.data_points
            .iter()
            .map(|(value, _)| *value)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
    }
}

/// 流式K-means聚类
pub struct StreamingKMeans {
    k: usize,
    centroids: Vec<Vec<f64>>,
    counts: Vec<u32>,
    learning_rate: f64,
}

impl StreamingKMeans {
    pub fn new(k: usize, learning_rate: f64) -> Self {
        Self {
            k,
            centroids: Vec::new(),
            counts: vec![0; k],
            learning_rate,
        }
    }

    /// 初始化质心
    pub fn initialize_centroids(&mut self, data_points: &[Vec<f64>]) {
        if data_points.len() < self.k {
            return;
        }
        
        // 随机选择初始质心
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..data_points.len()).collect();
        indices.shuffle(&mut rng);
        
        self.centroids.clear();
        for i in 0..self.k {
            self.centroids.push(data_points[indices[i]].clone());
        }
    }

    /// 更新聚类
    pub fn update(&mut self, data_point: &[f64]) -> usize {
        if self.centroids.is_empty() {
            self.centroids.push(data_point.to_vec());
            self.counts[0] = 1;
            return 0;
        }
        
        // 找到最近的质心
        let mut min_distance = f64::INFINITY;
        let mut closest_centroid = 0;
        
        for (i, centroid) in self.centroids.iter().enumerate() {
            let distance = self.euclidean_distance(data_point, centroid);
            if distance < min_distance {
                min_distance = distance;
                closest_centroid = i;
            }
        }
        
        // 更新质心
        self.counts[closest_centroid] += 1;
        let count = self.counts[closest_centroid] as f64;
        
        for (j, &value) in data_point.iter().enumerate() {
            let centroid_value = self.centroids[closest_centroid][j];
            self.centroids[closest_centroid][j] = 
                centroid_value + (self.learning_rate / count) * (value - centroid_value);
        }
        
        closest_centroid
    }

    /// 计算欧几里得距离
    fn euclidean_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// 获取质心
    pub fn get_centroids(&self) -> &[Vec<f64>] {
        &self.centroids
    }

    /// 获取聚类大小
    pub fn get_cluster_sizes(&self) -> &[u32] {
        &self.counts
    }
}
```

## 机器学习算法

### 定义 1.5 (在线学习)
在线学习算法在数据流上逐步更新模型，最小化累积损失：
$$\min_{w} \sum_{t=1}^{T} \ell_t(w_t)$$

### 算法 1.3 (在线线性回归)

```rust
use nalgebra::{DVector, DMatrix};

/// 在线线性回归
pub struct OnlineLinearRegression {
    weights: DVector<f64>,
    learning_rate: f64,
    regularization: f64,
}

impl OnlineLinearRegression {
    pub fn new(feature_count: usize, learning_rate: f64, regularization: f64) -> Self {
        Self {
            weights: DVector::zeros(feature_count),
            learning_rate,
            regularization,
        }
    }

    /// 预测
    pub fn predict(&self, features: &[f64]) -> f64 {
        let feature_vector = DVector::from_column_slice(features);
        self.weights.dot(&feature_vector)
    }

    /// 在线更新
    pub fn update(&mut self, features: &[f64], target: f64) -> f64 {
        let prediction = self.predict(features);
        let error = target - prediction;
        
        // 计算梯度
        let feature_vector = DVector::from_column_slice(features);
        let gradient = -2.0 * error * feature_vector + 2.0 * self.regularization * &self.weights;
        
        // 更新权重
        self.weights -= self.learning_rate * gradient;
        
        error * error // 返回平方误差
    }

    /// 获取权重
    pub fn get_weights(&self) -> &DVector<f64> {
        &self.weights
    }

    /// 设置权重
    pub fn set_weights(&mut self, weights: DVector<f64>) {
        self.weights = weights;
    }
}

/// 在线逻辑回归
pub struct OnlineLogisticRegression {
    weights: DVector<f64>,
    learning_rate: f64,
    regularization: f64,
}

impl OnlineLogisticRegression {
    pub fn new(feature_count: usize, learning_rate: f64, regularization: f64) -> Self {
        Self {
            weights: DVector::zeros(feature_count),
            learning_rate,
            regularization,
        }
    }

    /// 预测概率
    pub fn predict_probability(&self, features: &[f64]) -> f64 {
        let feature_vector = DVector::from_column_slice(features);
        let linear_output = self.weights.dot(&feature_vector);
        1.0 / (1.0 + (-linear_output).exp())
    }

    /// 预测类别
    pub fn predict(&self, features: &[f64]) -> bool {
        self.predict_probability(features) > 0.5
    }

    /// 在线更新
    pub fn update(&mut self, features: &[f64], target: bool) -> f64 {
        let probability = self.predict_probability(features);
        let target_value = if target { 1.0 } else { 0.0 };
        let error = target_value - probability;
        
        // 计算梯度
        let feature_vector = DVector::from_column_slice(features);
        let gradient = -error * feature_vector + self.regularization * &self.weights;
        
        // 更新权重
        self.weights -= self.learning_rate * gradient;
        
        // 返回对数损失
        if target {
            -probability.ln()
        } else {
            -(1.0 - probability).ln()
        }
    }
}

/// 决策树
#[derive(Debug, Clone)]
pub struct DecisionTreeNode {
    pub feature_index: Option<usize>,
    pub threshold: Option<f64>,
    pub left_child: Option<Box<DecisionTreeNode>>,
    pub right_child: Option<Box<DecisionTreeNode>>,
    pub prediction: Option<f64>,
}

impl DecisionTreeNode {
    pub fn new_leaf(prediction: f64) -> Self {
        Self {
            feature_index: None,
            threshold: None,
            left_child: None,
            right_child: None,
            prediction: Some(prediction),
        }
    }

    pub fn new_split(feature_index: usize, threshold: f64, left: DecisionTreeNode, right: DecisionTreeNode) -> Self {
        Self {
            feature_index: Some(feature_index),
            threshold: Some(threshold),
            left_child: Some(Box::new(left)),
            right_child: Some(Box::new(right)),
            prediction: None,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.prediction.is_some()
    }

    pub fn predict(&self, features: &[f64]) -> f64 {
        if let Some(prediction) = self.prediction {
            prediction
        } else if let (Some(feature_index), Some(threshold)) = (self.feature_index, self.threshold) {
            if features[feature_index] <= threshold {
                self.left_child.as_ref().unwrap().predict(features)
            } else {
                self.right_child.as_ref().unwrap().predict(features)
            }
        } else {
            panic!("Invalid decision tree node");
        }
    }
}

/// 在线决策树
pub struct OnlineDecisionTree {
    root: Option<DecisionTreeNode>,
    max_depth: usize,
    min_samples_split: usize,
    samples_seen: usize,
}

impl OnlineDecisionTree {
    pub fn new(max_depth: usize, min_samples_split: usize) -> Self {
        Self {
            root: None,
            max_depth,
            min_samples_split,
            samples_seen: 0,
        }
    }

    /// 在线更新
    pub fn update(&mut self, features: &[f64], target: f64) {
        self.samples_seen += 1;
        
        if self.root.is_none() {
            self.root = Some(DecisionTreeNode::new_leaf(target));
            return;
        }
        
        // 简化的在线更新：当样本数达到阈值时重建树
        if self.samples_seen % self.min_samples_split == 0 {
            // 这里应该实现完整的树重建逻辑
            // 为了简化，我们只更新叶子节点的预测
            self.update_leaf_predictions(features, target);
        }
    }

    fn update_leaf_predictions(&mut self, features: &[f64], target: f64) {
        // 简化的叶子节点更新
        if let Some(ref mut root) = self.root {
            if root.is_leaf() {
                // 更新叶子节点的预测（使用移动平均）
                let current_pred = root.prediction.unwrap_or(0.0);
                let new_pred = current_pred + 0.1 * (target - current_pred);
                root.prediction = Some(new_pred);
            }
        }
    }

    /// 预测
    pub fn predict(&self, features: &[f64]) -> Option<f64> {
        self.root.as_ref().map(|root| root.predict(features))
    }
}
```

## 时间序列分析

### 定义 1.6 (时间序列)
时间序列是一个有序的数据序列 $X = (x_1, x_2, \ldots, x_n)$，其中每个 $x_i$ 对应时间点 $t_i$。

### 算法 1.4 (时间序列预测)

```rust
use std::collections::VecDeque;

/// 移动平均
pub struct MovingAverage {
    window_size: usize,
    values: VecDeque<f64>,
    sum: f64,
}

impl MovingAverage {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            values: VecDeque::new(),
            sum: 0.0,
        }
    }

    /// 添加值
    pub fn add_value(&mut self, value: f64) {
        self.values.push_back(value);
        self.sum += value;
        
        if self.values.len() > self.window_size {
            if let Some(old_value) = self.values.pop_front() {
                self.sum -= old_value;
            }
        }
    }

    /// 获取平均值
    pub fn get_average(&self) -> Option<f64> {
        if self.values.is_empty() {
            None
        } else {
            Some(self.sum / self.values.len() as f64)
        }
    }

    /// 预测下一个值
    pub fn predict_next(&self) -> Option<f64> {
        self.get_average()
    }
}

/// 指数平滑
pub struct ExponentialSmoothing {
    alpha: f64,
    last_value: Option<f64>,
    smoothed_value: Option<f64>,
}

impl ExponentialSmoothing {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            last_value: None,
            smoothed_value: None,
        }
    }

    /// 添加值
    pub fn add_value(&mut self, value: f64) {
        self.last_value = Some(value);
        
        if let Some(smoothed) = self.smoothed_value {
            self.smoothed_value = Some(self.alpha * value + (1.0 - self.alpha) * smoothed);
        } else {
            self.smoothed_value = Some(value);
        }
    }

    /// 获取平滑值
    pub fn get_smoothed_value(&self) -> Option<f64> {
        self.smoothed_value
    }

    /// 预测下一个值
    pub fn predict_next(&self) -> Option<f64> {
        self.smoothed_value
    }
}

/// ARIMA模型（简化版）
pub struct ARIMAModel {
    p: usize, // AR阶数
    d: usize, // 差分阶数
    q: usize, // MA阶数
    ar_coeffs: Vec<f64>,
    ma_coeffs: Vec<f64>,
    residuals: VecDeque<f64>,
    values: VecDeque<f64>,
}

impl ARIMAModel {
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self {
            p,
            d,
            q,
            ar_coeffs: vec![0.1; p],
            ma_coeffs: vec![0.1; q],
            residuals: VecDeque::new(),
            values: VecDeque::new(),
        }
    }

    /// 添加值
    pub fn add_value(&mut self, value: f64) {
        self.values.push_back(value);
        
        // 保持历史记录大小
        let max_history = std::cmp::max(self.p, self.q) + 10;
        if self.values.len() > max_history {
            self.values.pop_front();
        }
        if self.residuals.len() > max_history {
            self.residuals.pop_front();
        }
    }

    /// 预测下一个值
    pub fn predict_next(&self) -> Option<f64> {
        if self.values.len() < self.p + self.d {
            return None;
        }
        
        let mut prediction = 0.0;
        
        // AR部分
        for i in 0..self.p {
            if let Some(value) = self.values.get(self.values.len() - 1 - i) {
                prediction += self.ar_coeffs[i] * value;
            }
        }
        
        // MA部分
        for i in 0..self.q {
            if let Some(residual) = self.residuals.get(self.residuals.len() - 1 - i) {
                prediction += self.ma_coeffs[i] * residual;
            }
        }
        
        Some(prediction)
    }

    /// 更新模型参数（简化版）
    pub fn update_parameters(&mut self, actual: f64, predicted: f64) {
        let residual = actual - predicted;
        self.residuals.push_back(residual);
        
        // 简化的参数更新（实际应用中应使用更复杂的优化算法）
        let learning_rate = 0.01;
        
        // 更新AR系数
        for i in 0..self.p {
            if let Some(value) = self.values.get(self.values.len() - 1 - i) {
                self.ar_coeffs[i] += learning_rate * residual * value;
            }
        }
        
        // 更新MA系数
        for i in 0..self.q {
            if let Some(residual_hist) = self.residuals.get(self.residuals.len() - 1 - i) {
                self.ma_coeffs[i] += learning_rate * residual * residual_hist;
            }
        }
    }
}

/// 异常检测
pub struct AnomalyDetector {
    window_size: usize,
    threshold: f64,
    values: VecDeque<f64>,
    mean: f64,
    variance: f64,
}

impl AnomalyDetector {
    pub fn new(window_size: usize, threshold: f64) -> Self {
        Self {
            window_size,
            threshold,
            values: VecDeque::new(),
            mean: 0.0,
            variance: 0.0,
        }
    }

    /// 添加值并检测异常
    pub fn add_value(&mut self, value: f64) -> bool {
        self.values.push_back(value);
        
        if self.values.len() > self.window_size {
            self.values.pop_front();
        }
        
        // 更新统计量
        self.update_statistics();
        
        // 检测异常
        if self.variance > 0.0 {
            let z_score = (value - self.mean).abs() / self.variance.sqrt();
            z_score > self.threshold
        } else {
            false
        }
    }

    /// 更新统计量
    fn update_statistics(&mut self) {
        if self.values.is_empty() {
            return;
        }
        
        let n = self.values.len() as f64;
        let sum: f64 = self.values.iter().sum();
        let sum_sq: f64 = self.values.iter().map(|x| x * x).sum();
        
        self.mean = sum / n;
        self.variance = (sum_sq / n) - (self.mean * self.mean);
    }

    /// 获取统计信息
    pub fn get_statistics(&self) -> (f64, f64) {
        (self.mean, self.variance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lz77_compression() {
        let compressor = LZ77Compressor::new(4096, 64);
        let data = b"hello world hello world";
        let tokens = compressor.compress(data);
        let decompressed = compressor.decompress(&tokens);
        
        assert_eq!(data, decompressed.as_slice());
    }

    #[test]
    fn test_sliding_window_aggregator() {
        let mut aggregator = NumericSlidingWindowAggregator::new(Duration::from_secs(10));
        let now = Instant::now();
        
        aggregator.add_value(1.0, now);
        aggregator.add_value(2.0, now);
        aggregator.add_value(3.0, now);
        
        assert_eq!(aggregator.mean(), Some(2.0));
        assert_eq!(aggregator.median(), Some(2.0));
    }

    #[test]
    fn test_online_linear_regression() {
        let mut model = OnlineLinearRegression::new(2, 0.01, 0.1);
        
        // 训练数据：y = 2*x1 + 3*x2
        let features = vec![1.0, 2.0];
        let target = 8.0; // 2*1 + 3*2 = 8
        
        let error = model.update(&features, target);
        assert!(error >= 0.0);
    }

    #[test]
    fn test_moving_average() {
        let mut ma = MovingAverage::new(3);
        
        ma.add_value(1.0);
        ma.add_value(2.0);
        ma.add_value(3.0);
        
        assert_eq!(ma.get_average(), Some(2.0));
        assert_eq!(ma.predict_next(), Some(2.0));
    }

    #[test]
    fn test_anomaly_detector() {
        let mut detector = AnomalyDetector::new(5, 2.0);
        
        // 正常值
        for i in 1..=5 {
            assert!(!detector.add_value(i as f64));
        }
        
        // 异常值
        assert!(detector.add_value(100.0));
    }
}
```

## 结论

本文分析了IoT数据处理的核心算法：

1. **数据压缩算法**：LZ77和霍夫曼编码减少存储和传输开销
2. **数据流处理算法**：滑动窗口聚合和流式聚类处理实时数据
3. **机器学习算法**：在线学习算法适应数据流变化
4. **时间序列分析**：预测和异常检测算法

这些算法为IoT系统提供了高效的数据处理能力，支持实时分析和决策。

---

**参考文献**：
1. Ziv, J., & Lempel, A. (1977). A universal algorithm for sequential data compression. IEEE Transactions on information theory, 23(3), 337-343.
2. Cormode, G., & Muthukrishnan, S. (2005). An improved data stream summary: the count-min sketch and its applications. Journal of Algorithms, 55(1), 58-75.
3. Bottou, L. (2010). Large-scale machine learning with stochastic gradient descent. Proceedings of COMPSTAT'2010, 177-186.
4. Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time series analysis: forecasting and control. John Wiley & Sons. 