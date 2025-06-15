# IoT数据压缩算法理论基础

## 目录

1. [概述](#1-概述)
2. [数据压缩形式化模型](#2-数据压缩形式化模型)
3. [差分压缩算法](#3-差分压缩算法)
4. [时间序列压缩算法](#4-时间序列压缩算法)
5. [传感器数据压缩算法](#5-传感器数据压缩算法)
6. [Rust算法实现](#6-rust算法实现)
7. [性能分析与优化](#7-性能分析与优化)
8. [定理与证明](#8-定理与证明)
9. [参考文献](#9-参考文献)

## 1. 概述

### 1.1 研究背景

IoT系统产生大量传感器数据，在资源受限的环境中，高效的数据压缩算法对于减少存储空间、降低传输成本和节省能源至关重要。

### 1.2 核心问题

**定义 1.1** (IoT数据压缩问题)
给定数据序列 $D = \langle d_1, d_2, \ldots, d_n \rangle$，压缩算法 $C$，解压算法 $D$，要求：

$$\text{CompressionRatio}(C) = \frac{|C(D)|}{|D|} \leq \alpha$$
$$\text{ReconstructionError}(D, D(C(D))) \leq \epsilon$$
$$\text{CompressionTime}(C, D) \leq \tau$$

其中 $\alpha$ 为压缩比阈值，$\epsilon$ 为误差阈值，$\tau$ 为时间阈值。

## 2. 数据压缩形式化模型

### 2.1 信息论基础

**定义 2.1** (信息熵)
对于概率分布 $P = (p_1, p_2, \ldots, p_n)$，信息熵定义为：
$$H(P) = -\sum_{i=1}^{n} p_i \log_2 p_i$$

**定义 2.2** (压缩率)
压缩率 $R$ 定义为：
$$R = \frac{\text{Compressed Size}}{\text{Original Size}}$$

**定理 2.1** (香农无损压缩定理)
对于任意数据源，无损压缩的平均码长下界为：
$$L \geq H(P)$$

**证明**：
根据信息论，任意无损压缩算法的平均码长不能小于信息熵，否则将违反唯一可解码性。

### 2.2 有损压缩模型

**定义 2.3** (失真度量)
失真度量 $d(x, y)$ 定义为原始数据 $x$ 和重构数据 $y$ 之间的距离：
$$d: \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}^+$$

**定义 2.4** (率失真函数)
率失真函数 $R(D)$ 定义为在给定失真 $D$ 下的最小码率：
$$R(D) = \min_{p(y|x): E[d(X,Y)] \leq D} I(X; Y)$$

**定理 2.2** (率失真定理)
对于任意数据源和失真度量，存在编码方案使得：
$$R \geq R(D)$$

## 3. 差分压缩算法

### 3.1 差分编码理论

**定义 3.1** (差分序列)
对于数据序列 $X = \langle x_1, x_2, \ldots, x_n \rangle$，差分序列 $D$ 定义为：
$$d_i = x_i - x_{i-1}, \quad i = 2, 3, \ldots, n$$
$$d_1 = x_1$$

**定理 3.1** (差分压缩效率)
对于具有时间相关性的数据序列，差分压缩的压缩比为：
$$R_{diff} = \frac{H(D)}{H(X)}$$

其中 $H(D)$ 为差分序列的熵，$H(X)$ 为原始序列的熵。

**证明**：
由于时间相关性，差分序列的方差通常小于原始序列，因此熵更小，压缩比更高。

### 3.2 自适应差分编码

**定义 3.2** (自适应差分编码)
自适应差分编码 $ADC$ 定义为：
$$ADC(x_i) = x_i - \hat{x}_i$$

其中 $\hat{x}_i$ 为基于历史数据的预测值：
$$\hat{x}_i = \sum_{j=1}^{k} w_j x_{i-j}$$

**定理 3.2** (预测误差界)
对于线性预测器，预测误差的方差为：
$$\sigma_e^2 = \sigma_x^2 (1 - \rho^2)$$

其中 $\rho$ 为自相关系数。

**证明**：
根据线性预测理论，预测误差与预测器系数正交，因此误差方差为原始方差减去预测增益。

## 4. 时间序列压缩算法

### 4.1 分段线性近似

**定义 4.1** (分段线性近似)
分段线性近似 $PLA$ 将时间序列 $T$ 分解为 $k$ 个线性段：
$$T = \bigcup_{i=1}^{k} L_i$$

其中每个线性段 $L_i$ 由起点 $(t_i, v_i)$ 和终点 $(t_{i+1}, v_{i+1})$ 定义。

**定义 4.2** (近似误差)
近似误差 $E$ 定义为：
$$E = \sum_{i=1}^{k} \sum_{t \in L_i} |v(t) - \hat{v}(t)|^2$$

其中 $\hat{v}(t)$ 为线性插值值。

**定理 4.1** (最优分段)
对于给定误差阈值 $\epsilon$，最优分段数 $k^*$ 满足：
$$k^* = \arg\min_k \{k: E(k) \leq \epsilon\}$$

**证明**：
这是一个约束优化问题，可以通过动态规划求解。

### 4.2 小波压缩

**定义 4.3** (离散小波变换)
离散小波变换 $DWT$ 定义为：
$$W_{j,k} = \sum_{n} x[n] \psi_{j,k}[n]$$

其中 $\psi_{j,k}[n]$ 为小波基函数。

**定理 4.3** (小波压缩率)
对于具有 $N$ 个非零系数的信号，小波压缩的压缩比为：
$$R_{wavelet} = \frac{N}{M}$$

其中 $M$ 为原始信号长度。

## 5. 传感器数据压缩算法

### 5.1 传感器数据特征

**定义 5.1** (传感器数据模型)
传感器数据 $S$ 可以建模为：
$$S(t) = \mu(t) + \sigma(t) \cdot \epsilon(t)$$

其中：

- $\mu(t)$ 为趋势分量
- $\sigma(t)$ 为波动分量
- $\epsilon(t)$ 为噪声分量

**定义 5.2** (数据质量度量)
数据质量 $Q$ 定义为：
$$Q = \frac{\text{Signal Power}}{\text{Noise Power}} = \frac{\sigma_s^2}{\sigma_n^2}$$

### 5.2 自适应压缩算法

**定义 5.3** (自适应压缩)
自适应压缩算法 $ACA$ 定义为：
$$ACA(S) = \begin{cases}
C_{lossless}(S) & \text{if } Q > \theta \\
C_{lossy}(S) & \text{otherwise}
\end{cases}$$

其中 $\theta$ 为质量阈值。

**定理 5.1** (自适应压缩最优性)
自适应压缩算法在给定质量约束下的压缩比最优。

**证明**：
根据率失真理论，自适应算法可以根据数据特征选择最优压缩策略。

## 6. Rust算法实现

### 6.1 差分压缩实现

```rust
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// 差分压缩算法
pub struct DifferentialCompressor {
    window_size: usize,
    threshold: f64,
}

impl DifferentialCompressor {
    pub fn new(window_size: usize, threshold: f64) -> Self {
        Self {
            window_size,
            threshold,
        }
    }

    /// 压缩时间序列数据
    pub fn compress(&self, data: &[f64]) -> CompressedData {
        let mut compressed = Vec::new();
        let mut metadata = CompressionMetadata::new();

        // 存储第一个值
        if let Some(&first_value) = data.first() {
            compressed.push(first_value);
            metadata.original_size = data.len();
            metadata.compressed_size = 1;
        }

        // 计算差分
        for window in data.windows(2) {
            let diff = window[1] - window[0];

            // 如果差分小于阈值，存储为0
            if diff.abs() < self.threshold {
                compressed.push(0.0);
            } else {
                compressed.push(diff);
            }
            metadata.compressed_size += 1;
        }

        CompressedData {
            data: compressed,
            metadata,
        }
    }

    /// 解压数据
    pub fn decompress(&self, compressed: &CompressedData) -> Vec<f64> {
        let mut decompressed = Vec::new();

        if compressed.data.is_empty() {
            return decompressed;
        }

        // 恢复第一个值
        decompressed.push(compressed.data[0]);

        // 恢复后续值
        for i in 1..compressed.data.len() {
            let diff = compressed.data[i];
            let prev_value = decompressed[i - 1];
            let current_value = prev_value + diff;
            decompressed.push(current_value);
        }

        decompressed
    }

    /// 计算压缩比
    pub fn compression_ratio(&self, original: &[f64], compressed: &CompressedData) -> f64 {
        let original_size = original.len() * std::mem::size_of::<f64>();
        let compressed_size = compressed.data.len() * std::mem::size_of::<f64>();

        compressed_size as f64 / original_size as f64
    }

    /// 计算重构误差
    pub fn reconstruction_error(&self, original: &[f64], decompressed: &[f64]) -> f64 {
        if original.len() != decompressed.len() {
            return f64::INFINITY;
        }

        let mut total_error = 0.0;
        for (orig, decomp) in original.iter().zip(decompressed.iter()) {
            total_error += (orig - decomp).abs();
        }

        total_error / original.len() as f64
    }
}

/// 压缩数据
# [derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedData {
    pub data: Vec<f64>,
    pub metadata: CompressionMetadata,
}

/// 压缩元数据
# [derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionMetadata {
    pub original_size: usize,
    pub compressed_size: usize,
    pub algorithm: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl CompressionMetadata {
    pub fn new() -> Self {
        Self {
            original_size: 0,
            compressed_size: 0,
            algorithm: "differential".to_string(),
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn compression_ratio(&self) -> f64 {
        self.compressed_size as f64 / self.original_size as f64
    }
}

/// 分段线性近似压缩器
pub struct PiecewiseLinearCompressor {
    max_error: f64,
    min_segment_length: usize,
}

impl PiecewiseLinearCompressor {
    pub fn new(max_error: f64, min_segment_length: usize) -> Self {
        Self {
            max_error,
            min_segment_length,
        }
    }

    /// 压缩时间序列
    pub fn compress(&self, data: &[f64]) -> CompressedData {
        let segments = self.find_optimal_segments(data);
        let mut compressed_data = Vec::new();
        let mut metadata = CompressionMetadata::new();
        metadata.algorithm = "piecewise_linear".to_string();
        metadata.original_size = data.len();

        for segment in &segments {
            compressed_data.push(segment.start_time);
            compressed_data.push(segment.start_value);
            compressed_data.push(segment.end_time);
            compressed_data.push(segment.end_value);
            metadata.compressed_size += 4;
        }

        CompressedData {
            data: compressed_data,
            metadata,
        }
    }

    /// 解压数据
    pub fn decompress(&self, compressed: &CompressedData) -> Vec<f64> {
        let mut decompressed = Vec::new();
        let data = &compressed.data;

        for i in (0..data.len()).step_by(4) {
            if i + 3 >= data.len() {
                break;
            }

            let start_time = data[i] as usize;
            let start_value = data[i + 1];
            let end_time = data[i + 2] as usize;
            let end_value = data[i + 3];

            // 线性插值
            for t in start_time..=end_time {
                if t >= start_time && t <= end_time {
                    let alpha = (t - start_time) as f64 / (end_time - start_time) as f64;
                    let interpolated_value = start_value + alpha * (end_value - start_value);
                    decompressed.push(interpolated_value);
                }
            }
        }

        decompressed
    }

    /// 寻找最优分段
    fn find_optimal_segments(&self, data: &[f64]) -> Vec<LinearSegment> {
        let mut segments = Vec::new();
        let mut start = 0;

        while start < data.len() {
            let end = self.find_segment_end(data, start);
            let segment = LinearSegment {
                start_time: start,
                start_value: data[start],
                end_time: end,
                end_value: data[end],
            };
            segments.push(segment);
            start = end + 1;
        }

        segments
    }

    /// 寻找分段结束点
    fn find_segment_end(&self, data: &[f64], start: usize) -> usize {
        let mut end = start + self.min_segment_length;

        while end < data.len() {
            // 计算线性拟合误差
            let error = self.calculate_linear_error(data, start, end);
            if error > self.max_error {
                break;
            }
            end += 1;
        }

        (end - 1).min(data.len() - 1)
    }

    /// 计算线性拟合误差
    fn calculate_linear_error(&self, data: &[f64], start: usize, end: usize) -> f64 {
        if end - start < 2 {
            return 0.0;
        }

        // 最小二乘法拟合直线
        let n = end - start + 1;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;

        for i in start..=end {
            let x = (i - start) as f64;
            let y = data[i];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        let slope = (n as f64 * sum_xy - sum_x * sum_y) / (n as f64 * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n as f64;

        // 计算误差
        let mut total_error = 0.0;
        for i in start..=end {
            let x = (i - start) as f64;
            let predicted = slope * x + intercept;
            let actual = data[i];
            total_error += (predicted - actual).abs();
        }

        total_error / n as f64
    }
}

/// 线性段
# [derive(Debug, Clone)]
pub struct LinearSegment {
    pub start_time: usize,
    pub start_value: f64,
    pub end_time: usize,
    pub end_value: f64,
}

/// 小波压缩器
pub struct WaveletCompressor {
    wavelet_type: WaveletType,
    decomposition_levels: usize,
    threshold: f64,
}

impl WaveletCompressor {
    pub fn new(wavelet_type: WaveletType, decomposition_levels: usize, threshold: f64) -> Self {
        Self {
            wavelet_type,
            decomposition_levels,
            threshold,
        }
    }

    /// 压缩数据
    pub fn compress(&self, data: &[f64]) -> CompressedData {
        let mut coefficients = data.to_vec();

        // 小波分解
        for level in 0..self.decomposition_levels {
            coefficients = self.wavelet_decomposition(&coefficients, level);
        }

        // 阈值处理
        let mut compressed_coeffs = Vec::new();
        let mut indices = Vec::new();

        for (i, &coeff) in coefficients.iter().enumerate() {
            if coeff.abs() > self.threshold {
                compressed_coeffs.push(coeff);
                indices.push(i);
            }
        }

        let mut compressed_data = Vec::new();
        compressed_data.extend_from_slice(&compressed_coeffs);
        compressed_data.extend_from_slice(&indices.iter().map(|&x| x as f64).collect::<Vec<f64>>());

        let mut metadata = CompressionMetadata::new();
        metadata.algorithm = "wavelet".to_string();
        metadata.original_size = data.len();
        metadata.compressed_size = compressed_data.len();

        CompressedData {
            data: compressed_data,
            metadata,
        }
    }

    /// 解压数据
    pub fn decompress(&self, compressed: &CompressedData) -> Vec<f64> {
        let data = &compressed.data;
        let mid_point = data.len() / 2;

        let coefficients = &data[..mid_point];
        let indices: Vec<usize> = data[mid_point..].iter().map(|&x| x as usize).collect();

        // 重建系数向量
        let mut full_coefficients = vec![0.0; compressed.metadata.original_size];
        for (i, &coeff) in indices.iter().zip(coefficients.iter()) {
            if i < full_coefficients.len() {
                full_coefficients[i] = coeff;
            }
        }

        // 小波重构
        for level in (0..self.decomposition_levels).rev() {
            full_coefficients = self.wavelet_reconstruction(&full_coefficients, level);
        }

        full_coefficients
    }

    /// 小波分解
    fn wavelet_decomposition(&self, data: &[f64], level: usize) -> Vec<f64> {
        // 简化的Haar小波分解
        let mut result = Vec::new();
        let step = 1 << level;

        for i in (0..data.len()).step_by(2 * step) {
            if i + step < data.len() {
                let avg = (data[i] + data[i + step]) / 2.0;
                let diff = (data[i] - data[i + step]) / 2.0;
                result.push(avg);
                result.push(diff);
            } else {
                result.push(data[i]);
            }
        }

        result
    }

    /// 小波重构
    fn wavelet_reconstruction(&self, data: &[f64], level: usize) -> Vec<f64> {
        // 简化的Haar小波重构
        let mut result = Vec::new();
        let step = 1 << level;

        for i in (0..data.len()).step_by(2) {
            if i + 1 < data.len() {
                let avg = data[i];
                let diff = data[i + 1];
                result.push(avg + diff);
                result.push(avg - diff);
            } else {
                result.push(data[i]);
            }
        }

        result
    }
}

/// 小波类型
# [derive(Debug, Clone)]
pub enum WaveletType {
    Haar,
    Daubechies,
    Symlets,
}

/// 自适应压缩器
pub struct AdaptiveCompressor {
    quality_threshold: f64,
    differential_compressor: DifferentialCompressor,
    linear_compressor: PiecewiseLinearCompressor,
    wavelet_compressor: WaveletCompressor,
}

impl AdaptiveCompressor {
    pub fn new(quality_threshold: f64) -> Self {
        Self {
            quality_threshold,
            differential_compressor: DifferentialCompressor::new(10, 0.01),
            linear_compressor: PiecewiseLinearCompressor::new(0.1, 5),
            wavelet_compressor: WaveletCompressor::new(WaveletType::Haar, 3, 0.01),
        }
    }

    /// 自适应压缩
    pub fn compress(&self, data: &[f64]) -> CompressedData {
        // 计算数据质量
        let quality = self.calculate_data_quality(data);

        // 根据质量选择压缩算法
        if quality > self.quality_threshold {
            // 高质量数据使用无损压缩
            self.differential_compressor.compress(data)
        } else {
            // 低质量数据使用有损压缩
            let linear_compressed = self.linear_compressor.compress(data);
            let wavelet_compressed = self.wavelet_compressor.compress(data);

            // 选择压缩比更高的结果
            if linear_compressed.metadata.compression_ratio() < wavelet_compressed.metadata.compression_ratio() {
                linear_compressed
            } else {
                wavelet_compressed
            }
        }
    }

    /// 计算数据质量
    fn calculate_data_quality(&self, data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 1.0;
        }

        // 计算信号功率
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let signal_power: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum();

        // 计算噪声功率（使用差分估计）
        let mut noise_power = 0.0;
        for window in data.windows(2) {
            let diff = window[1] - window[0];
            noise_power += diff.powi(2);
        }
        noise_power /= (data.len() - 1) as f64;

        if noise_power == 0.0 {
            1.0
        } else {
            signal_power / noise_power
        }
    }
}

/// 压缩性能分析器
pub struct CompressionAnalyzer;

impl CompressionAnalyzer {
    /// 分析压缩性能
    pub fn analyze_performance(
        &self,
        original: &[f64],
        compressed: &CompressedData,
        decompressed: &[f64],
    ) -> CompressionPerformance {
        let compression_ratio = compressed.metadata.compression_ratio();
        let reconstruction_error = self.calculate_reconstruction_error(original, decompressed);
        let compression_time = compressed.metadata.timestamp.timestamp_millis();

        CompressionPerformance {
            compression_ratio,
            reconstruction_error,
            compression_time,
            algorithm: compressed.metadata.algorithm.clone(),
        }
    }

    /// 计算重构误差
    fn calculate_reconstruction_error(&self, original: &[f64], decompressed: &[f64]) -> f64 {
        if original.len() != decompressed.len() {
            return f64::INFINITY;
        }

        let mut total_error = 0.0;
        for (orig, decomp) in original.iter().zip(decompressed.iter()) {
            total_error += (orig - decomp).abs();
        }

        total_error / original.len() as f64
    }
}

/// 压缩性能指标
# [derive(Debug, Clone)]
pub struct CompressionPerformance {
    pub compression_ratio: f64,
    pub reconstruction_error: f64,
    pub compression_time: i64,
    pub algorithm: String,
}

/// 传感器数据压缩器
pub struct SensorDataCompressor {
    adaptive_compressor: AdaptiveCompressor,
    analyzer: CompressionAnalyzer,
}

impl SensorDataCompressor {
    pub fn new() -> Self {
        Self {
            adaptive_compressor: AdaptiveCompressor::new(10.0),
            analyzer: CompressionAnalyzer,
        }
    }

    /// 压缩传感器数据
    pub fn compress_sensor_data(&self, sensor_data: &SensorData) -> CompressedSensorData {
        let compressed = self.adaptive_compressor.compress(&sensor_data.values);
        let decompressed = self.adaptive_compressor.decompress(&compressed);
        let performance = self.analyzer.analyze_performance(
            &sensor_data.values,
            &compressed,
            &decompressed,
        );

        CompressedSensorData {
            sensor_id: sensor_data.sensor_id.clone(),
            compressed_data: compressed,
            performance,
            timestamp: chrono::Utc::now(),
        }
    }
}

/// 传感器数据
# [derive(Debug, Clone)]
pub struct SensorData {
    pub sensor_id: String,
    pub values: Vec<f64>,
    pub timestamps: Vec<chrono::DateTime<chrono::Utc>>,
}

/// 压缩后的传感器数据
# [derive(Debug, Clone)]
pub struct CompressedSensorData {
    pub sensor_id: String,
    pub compressed_data: CompressedData,
    pub performance: CompressionPerformance,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}
```

### 6.2 算法复杂度分析

**定理 6.1** (差分压缩复杂度)
差分压缩算法的时间复杂度为 $O(n)$，空间复杂度为 $O(n)$。

**证明**：
1. 差分计算需要遍历整个序列一次：$O(n)$
2. 存储压缩结果需要与原序列相同大小的空间：$O(n)$

**定理 6.2** (分段线性压缩复杂度)
分段线性压缩算法的时间复杂度为 $O(n^2)$，空间复杂度为 $O(n)$。

**证明**：
1. 寻找最优分段需要动态规划：$O(n^2)$
2. 存储分段信息需要线性空间：$O(n)$

## 7. 性能分析与优化

### 7.1 压缩率分析

**定理 7.1** (压缩率下界)
对于任意压缩算法，压缩率的下界为：
$$R \geq \frac{H(X)}{8}$$

其中 $H(X)$ 为数据熵，8为字节位数。

**证明**：
根据香农定理，平均码长不能小于熵，因此压缩率不能小于熵与字节位数的比值。

### 7.2 误差分析

**定理 7.2** (重构误差界)
对于分段线性近似，重构误差的上界为：
$$E \leq \frac{L^2}{4} \max_{i} |f''(t_i)|$$

其中 $L$ 为最大段长，$f''(t_i)$ 为二阶导数。

**证明**：
根据泰勒展开和线性插值误差公式，重构误差与二阶导数成正比。

## 8. 定理与证明

### 8.1 压缩算法正确性

**定理 8.1** (差分压缩正确性)
差分压缩算法在无噪声条件下可以完全重构原始数据。

**证明**：
设原始序列为 $X = \langle x_1, x_2, \ldots, x_n \rangle$，差分序列为 $D = \langle d_1, d_2, \ldots, d_n \rangle$。

根据差分定义：
$$d_1 = x_1$$
$$d_i = x_i - x_{i-1}, \quad i = 2, 3, \ldots, n$$

重构时：
$$x_1 = d_1$$
$$x_i = d_i + x_{i-1}, \quad i = 2, 3, \ldots, n$$

因此可以完全重构原始序列。

### 8.2 自适应压缩最优性

**定理 8.2** (自适应压缩最优性)
自适应压缩算法在给定质量约束下的压缩比最优。

**证明**：
根据率失真理论，自适应算法可以根据数据特征选择最优压缩策略，因此压缩比最优。

## 9. 参考文献

1. Shannon, C. E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal, 27(3), 379-423.
2. Berger, T. (1971). Rate Distortion Theory: A Mathematical Basis for Data Compression. Prentice-Hall.
3. Mallat, S. G. (1989). A Theory for Multiresolution Signal Decomposition: The Wavelet Representation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 11(7), 674-693.
4. Keogh, E., Chakrabarti, K., Pazzani, M., & Mehrotra, S. (2001). Locally Adaptive Dimensionality Reduction for Indexing Large Time Series Databases. ACM SIGMOD Record, 30(2), 151-162.
5. Rust Programming Language. (2023). The Rust Programming Language. https://doc.rust-lang.org/book/
6. Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory. Wiley-Interscience.
7. Sayood, K. (2017). Introduction to Data Compression. Morgan Kaufmann.
8. Vitter, J. S. (1987). Design and Analysis of Dynamic Huffman Codes. Journal of the ACM, 34(4), 825-845.

---

**文档版本**: v1.0  
**最后更新**: 2024-12-19  
**作者**: 算法分析团队  
**状态**: 已完成IoT数据压缩算法理论基础
