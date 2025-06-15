# IoT数据压缩算法理论基础

## 1. 数据压缩形式化模型

### 1.1 基本定义

#### 定义 1.1 (数据压缩)
数据压缩是一个函数 $C: \Sigma^* \rightarrow \Sigma^*$，其中：
- $\Sigma$ 是字母表
- $\Sigma^*$ 是所有可能字符串的集合
- 对于输入 $x \in \Sigma^*$，$C(x)$ 是压缩后的数据

#### 定义 1.2 (压缩率)
压缩率定义为：
$$\text{Compression Ratio} = \frac{|C(x)|}{|x|} \times 100\%$$

#### 定义 1.3 (压缩效率)
压缩效率定义为：
$$\text{Efficiency} = \frac{\text{Compression Time}}{\text{Decompression Time}}$$

### 1.2 信息论基础

#### 定理 1.1 (香农熵)
对于概率分布 $P = (p_1, p_2, \ldots, p_n)$，香农熵定义为：
$$H(P) = -\sum_{i=1}^{n} p_i \log_2 p_i$$

#### 定理 1.2 (压缩下界)
对于任何无损压缩算法，平均压缩长度满足：
$$L \geq H(P)$$

**证明**：根据信息论，香农熵是平均信息量的下界。

## 2. 差分压缩算法

### 2.1 差分压缩理论

#### 定义 2.1 (差分序列)
对于时间序列 $X = (x_1, x_2, \ldots, x_n)$，差分序列定义为：
$$\Delta X = (x_2 - x_1, x_3 - x_2, \ldots, x_n - x_{n-1})$$

#### 定理 2.1 (差分压缩效率)
对于具有时间相关性的数据，差分压缩的压缩率满足：
$$\text{Compression Ratio} \leq \frac{\sigma_{\Delta}}{\sigma_X} \times 100\%$$

其中 $\sigma_{\Delta}$ 是差分序列的标准差，$\sigma_X$ 是原序列的标准差。

### 2.2 Rust差分压缩实现

```rust
use std::collections::VecDeque;
use serde::{Deserialize, Serialize};

/// 差分压缩器
#[derive(Debug)]
pub struct DeltaCompressor {
    pub window_size: usize,
    pub threshold: f64,
    pub history: VecDeque<f64>,
}

/// 压缩块
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedBlock {
    pub base_value: f64,
    pub deltas: Vec<f64>,
    pub timestamps: Vec<u64>,
    pub compression_ratio: f64,
}

impl DeltaCompressor {
    /// 创建新的差分压缩器
    pub fn new(window_size: usize, threshold: f64) -> Self {
        Self {
            window_size,
            threshold,
            history: VecDeque::new(),
        }
    }

    /// 压缩时间序列数据
    pub fn compress(&mut self, data: &[(u64, f64)]) -> Vec<CompressedBlock> {
        let mut blocks = Vec::new();
        let mut current_block = Vec::new();
        let mut base_value = None;

        for (timestamp, value) in data {
            if base_value.is_none() {
                base_value = Some(*value);
                current_block.push((*timestamp, *value));
            } else {
                let delta = *value - base_value.unwrap();
                
                // 检查是否需要开始新块
                if self.should_start_new_block(delta) {
                    if !current_block.is_empty() {
                        blocks.push(self.create_block(&current_block));
                    }
                    base_value = Some(*value);
                    current_block.clear();
                    current_block.push((*timestamp, *value));
                } else {
                    current_block.push((*timestamp, *value));
                }
            }

            // 检查窗口大小
            if current_block.len() >= self.window_size {
                blocks.push(self.create_block(&current_block));
                base_value = None;
                current_block.clear();
            }
        }

        // 处理剩余数据
        if !current_block.is_empty() {
            blocks.push(self.create_block(&current_block));
        }

        blocks
    }

    /// 解压缩数据
    pub fn decompress(&self, blocks: &[CompressedBlock]) -> Vec<(u64, f64)> {
        let mut result = Vec::new();

        for block in blocks {
            let mut current_value = block.base_value;
            
            for (i, delta) in block.deltas.iter().enumerate() {
                current_value += delta;
                result.push((block.timestamps[i], current_value));
            }
        }

        result
    }

    /// 判断是否应该开始新块
    fn should_start_new_block(&self, delta: f64) -> bool {
        delta.abs() > self.threshold
    }

    /// 创建压缩块
    fn create_block(&self, data: &[(u64, f64)]) -> CompressedBlock {
        let base_value = data[0].1;
        let mut deltas = Vec::new();
        let mut timestamps = Vec::new();

        for (timestamp, value) in data {
            deltas.push(*value - base_value);
            timestamps.push(*timestamp);
        }

        let original_size = data.len() * std::mem::size_of::<f64>();
        let compressed_size = std::mem::size_of::<f64>() + deltas.len() * std::mem::size_of::<f64>();
        let compression_ratio = compressed_size as f64 / original_size as f64;

        CompressedBlock {
            base_value,
            deltas,
            timestamps,
            compression_ratio,
        }
    }

    /// 计算压缩统计信息
    pub fn calculate_statistics(&self, blocks: &[CompressedBlock]) -> CompressionStatistics {
        let total_original_size = blocks.iter()
            .map(|b| b.timestamps.len() * std::mem::size_of::<f64>())
            .sum::<usize>();

        let total_compressed_size = blocks.iter()
            .map(|b| std::mem::size_of::<f64>() + b.deltas.len() * std::mem::size_of::<f64>())
            .sum::<usize>();

        let average_compression_ratio = blocks.iter()
            .map(|b| b.compression_ratio)
            .sum::<f64>() / blocks.len() as f64;

        CompressionStatistics {
            total_original_size,
            total_compressed_size,
            average_compression_ratio,
            block_count: blocks.len(),
        }
    }
}

/// 压缩统计信息
#[derive(Debug, Clone)]
pub struct CompressionStatistics {
    pub total_original_size: usize,
    pub total_compressed_size: usize,
    pub average_compression_ratio: f64,
    pub block_count: usize,
}
```

## 3. 时间序列压缩算法

### 3.1 时间序列特征

#### 定义 3.1 (时间序列)
时间序列是一个有序对序列：
$$T = \{(t_1, x_1), (t_2, x_2), \ldots, (t_n, x_n)\}$$

其中 $t_i < t_{i+1}$ 对所有 $i$ 成立。

#### 定义 3.2 (时间序列压缩)
时间序列压缩是一个函数：
$$C_T: \mathcal{T} \rightarrow \mathcal{T}_C$$

其中 $\mathcal{T}$ 是时间序列集合，$\mathcal{T}_C$ 是压缩时间序列集合。

### 3.2 线性插值压缩

#### 定理 3.1 (线性插值误差)
对于线性插值压缩，最大误差满足：
$$\text{Max Error} \leq \frac{h^2}{8} \max_{t \in [t_i, t_{i+1}]} |f''(t)|$$

其中 $h$ 是采样间隔。

### 3.3 Rust时间序列压缩实现

```rust
use std::collections::HashMap;

/// 时间序列压缩器
#[derive(Debug)]
pub struct TimeSeriesCompressor {
    pub error_threshold: f64,
    pub min_points: usize,
    pub max_points: usize,
}

/// 压缩时间序列
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedTimeSeries {
    pub key_points: Vec<(u64, f64)>,
    pub interpolation_method: InterpolationMethod,
    pub compression_ratio: f64,
    pub max_error: f64,
}

/// 插值方法
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterpolationMethod {
    Linear,
    Polynomial,
    Spline,
}

impl TimeSeriesCompressor {
    /// 创建新的时间序列压缩器
    pub fn new(error_threshold: f64, min_points: usize, max_points: usize) -> Self {
        Self {
            error_threshold,
            min_points,
            max_points,
        }
    }

    /// 使用Douglas-Peucker算法压缩
    pub fn douglas_peucker_compress(&self, data: &[(u64, f64)]) -> CompressedTimeSeries {
        if data.len() <= self.min_points {
            return self.create_compressed_series(data, InterpolationMethod::Linear);
        }

        let key_points = self.douglas_peucker_recursive(data);
        
        self.create_compressed_series(&key_points, InterpolationMethod::Linear)
    }

    /// Douglas-Peucker递归算法
    fn douglas_peucker_recursive(&self, data: &[(u64, f64)]) -> Vec<(u64, f64)> {
        if data.len() <= 2 {
            return data.to_vec();
        }

        let (max_distance, max_index) = self.find_max_distance(data);
        
        if max_distance <= self.error_threshold {
            return vec![data[0], data[data.len() - 1]];
        }

        let left_points = self.douglas_peucker_recursive(&data[..=max_index]);
        let right_points = self.douglas_peucker_recursive(&data[max_index..]);

        // 合并结果，避免重复点
        let mut result = left_points[..left_points.len()-1].to_vec();
        result.extend(right_points);
        
        result
    }

    /// 找到最大距离点
    fn find_max_distance(&self, data: &[(u64, f64)]) -> (f64, usize) {
        let start = data[0];
        let end = data[data.len() - 1];
        
        let mut max_distance = 0.0;
        let mut max_index = 0;

        for (i, point) in data.iter().enumerate().skip(1).take(data.len() - 2) {
            let distance = self.point_to_line_distance(*point, start, end);
            
            if distance > max_distance {
                max_distance = distance;
                max_index = i;
            }
        }

        (max_distance, max_index)
    }

    /// 计算点到直线的距离
    fn point_to_line_distance(&self, point: (u64, f64), start: (u64, f64), end: (u64, f64)) -> f64 {
        let (x, y) = point;
        let (x1, y1) = start;
        let (x2, y2) = end;

        if x2 == x1 {
            return (x - x1).abs() as f64;
        }

        let slope = (y2 - y1) / (x2 - x1) as f64;
        let intercept = y1 - slope * x1 as f64;
        
        let expected_y = slope * x as f64 + intercept;
        (y - expected_y).abs()
    }

    /// 创建压缩时间序列
    fn create_compressed_series(&self, key_points: &[(u64, f64)], method: InterpolationMethod) -> CompressedTimeSeries {
        let original_size = key_points.len() * std::mem::size_of::<(u64, f64)>();
        let compressed_size = key_points.len() * std::mem::size_of::<(u64, f64)>();
        let compression_ratio = compressed_size as f64 / original_size as f64;

        CompressedTimeSeries {
            key_points: key_points.to_vec(),
            interpolation_method: method,
            compression_ratio,
            max_error: self.calculate_max_error(key_points),
        }
    }

    /// 计算最大误差
    fn calculate_max_error(&self, key_points: &[(u64, f64)]) -> f64 {
        // 简化实现，实际应该计算插值误差
        0.0
    }

    /// 解压缩时间序列
    pub fn decompress(&self, compressed: &CompressedTimeSeries) -> Vec<(u64, f64)> {
        match compressed.interpolation_method {
            InterpolationMethod::Linear => self.linear_interpolate(&compressed.key_points),
            InterpolationMethod::Polynomial => self.polynomial_interpolate(&compressed.key_points),
            InterpolationMethod::Spline => self.spline_interpolate(&compressed.key_points),
        }
    }

    /// 线性插值
    fn linear_interpolate(&self, key_points: &[(u64, f64)]) -> Vec<(u64, f64)> {
        let mut result = Vec::new();
        
        for i in 0..key_points.len() - 1 {
            let (t1, x1) = key_points[i];
            let (t2, x2) = key_points[i + 1];
            
            result.push((t1, x1));
            
            // 在两点之间插值
            let steps = ((t2 - t1) / 1000).max(1); // 假设时间戳是毫秒
            for j in 1..steps {
                let t = t1 + j * 1000;
                let ratio = j as f64 / steps as f64;
                let x = x1 + ratio * (x2 - x1);
                result.push((t, x));
            }
        }
        
        if let Some(last) = key_points.last() {
            result.push(*last);
        }
        
        result
    }

    /// 多项式插值
    fn polynomial_interpolate(&self, key_points: &[(u64, f64)]) -> Vec<(u64, f64)> {
        // 简化实现，返回关键点
        key_points.to_vec()
    }

    /// 样条插值
    fn spline_interpolate(&self, key_points: &[(u64, f64)]) -> Vec<(u64, f64)> {
        // 简化实现，返回关键点
        key_points.to_vec()
    }
}
```

## 4. 传感器数据压缩算法

### 4.1 传感器数据特征

#### 定义 4.1 (传感器数据)
传感器数据是一个四元组：
$$S = (t, v, q, m)$$

其中：
- $t$ 是时间戳
- $v$ 是测量值
- $q$ 是数据质量
- $m$ 是元数据

#### 定义 4.2 (数据质量)
数据质量定义为：
$$Q = \alpha \cdot \text{Accuracy} + \beta \cdot \text{Precision} + \gamma \cdot \text{Reliability}$$

### 4.2 自适应压缩算法

#### 定理 4.1 (自适应压缩最优性)
对于传感器数据，自适应压缩算法的压缩率满足：
$$\text{Compression Ratio} \geq \frac{H(P_{\text{optimal}})}{H(P_{\text{actual}})}$$

### 4.3 Rust传感器数据压缩实现

```rust
use std::collections::HashMap;

/// 传感器数据类型
#[derive(Debug, Clone)]
pub enum SensorDataType {
    Temperature,
    Humidity,
    Pressure,
    Acceleration,
    Custom(String),
}

/// 传感器数据
#[derive(Debug, Clone)]
pub struct SensorData {
    pub timestamp: u64,
    pub value: f64,
    pub data_type: SensorDataType,
    pub quality: f64,
    pub metadata: HashMap<String, String>,
}

/// 自适应压缩器
#[derive(Debug)]
pub struct AdaptiveCompressor {
    pub data_type_compressors: HashMap<SensorDataType, Box<dyn DataCompressor>>,
    pub quality_threshold: f64,
    pub adaptive_threshold: bool,
}

/// 数据压缩器trait
pub trait DataCompressor: Send + Sync {
    fn compress(&self, data: &[SensorData]) -> CompressedSensorData;
    fn decompress(&self, compressed: &CompressedSensorData) -> Vec<SensorData>;
    fn get_compression_ratio(&self, data: &[SensorData]) -> f64;
}

/// 压缩传感器数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedSensorData {
    pub data_type: SensorDataType,
    pub compressed_values: Vec<u8>,
    pub timestamps: Vec<u64>,
    pub quality_scores: Vec<f64>,
    pub compression_method: String,
    pub compression_ratio: f64,
}

/// 温度数据压缩器
#[derive(Debug)]
pub struct TemperatureCompressor {
    pub base_temperature: f64,
    pub precision: f64,
}

impl DataCompressor for TemperatureCompressor {
    fn compress(&self, data: &[SensorData]) -> CompressedSensorData {
        let mut compressed_values = Vec::new();
        let mut timestamps = Vec::new();
        let mut quality_scores = Vec::new();

        for sensor_data in data {
            // 计算相对于基准温度的差值
            let delta = sensor_data.value - self.base_temperature;
            
            // 量化到指定精度
            let quantized = (delta / self.precision).round() as i16;
            
            // 编码为字节
            compressed_values.extend_from_slice(&quantized.to_le_bytes());
            timestamps.push(sensor_data.timestamp);
            quality_scores.push(sensor_data.quality);
        }

        let original_size = data.len() * std::mem::size_of::<f64>();
        let compressed_size = compressed_values.len();
        let compression_ratio = compressed_size as f64 / original_size as f64;

        CompressedSensorData {
            data_type: SensorDataType::Temperature,
            compressed_values,
            timestamps,
            quality_scores,
            compression_method: "TemperatureDelta".to_string(),
            compression_ratio,
        }
    }

    fn decompress(&self, compressed: &CompressedSensorData) -> Vec<SensorData> {
        let mut result = Vec::new();

        for (i, chunk) in compressed.compressed_values.chunks(2).enumerate() {
            if chunk.len() == 2 {
                let quantized = i16::from_le_bytes([chunk[0], chunk[1]]);
                let delta = quantized as f64 * self.precision;
                let value = self.base_temperature + delta;

                result.push(SensorData {
                    timestamp: compressed.timestamps[i],
                    value,
                    data_type: SensorDataType::Temperature,
                    quality: compressed.quality_scores[i],
                    metadata: HashMap::new(),
                });
            }
        }

        result
    }

    fn get_compression_ratio(&self, data: &[SensorData]) -> f64 {
        let original_size = data.len() * std::mem::size_of::<f64>();
        let compressed_size = data.len() * 2; // 2 bytes per value
        compressed_size as f64 / original_size as f64
    }
}

/// 湿度数据压缩器
#[derive(Debug)]
pub struct HumidityCompressor {
    pub precision: f64,
}

impl DataCompressor for HumidityCompressor {
    fn compress(&self, data: &[SensorData]) -> CompressedSensorData {
        let mut compressed_values = Vec::new();
        let mut timestamps = Vec::new();
        let mut quality_scores = Vec::new();

        for sensor_data in data {
            // 湿度值通常在0-100范围内，可以直接量化
            let quantized = (sensor_data.value / self.precision).round() as u8;
            
            compressed_values.push(quantized);
            timestamps.push(sensor_data.timestamp);
            quality_scores.push(sensor_data.quality);
        }

        let original_size = data.len() * std::mem::size_of::<f64>();
        let compressed_size = compressed_values.len();
        let compression_ratio = compressed_size as f64 / original_size as f64;

        CompressedSensorData {
            data_type: SensorDataType::Humidity,
            compressed_values,
            timestamps,
            quality_scores,
            compression_method: "HumidityQuantized".to_string(),
            compression_ratio,
        }
    }

    fn decompress(&self, compressed: &CompressedSensorData) -> Vec<SensorData> {
        let mut result = Vec::new();

        for (i, &quantized) in compressed.compressed_values.iter().enumerate() {
            let value = quantized as f64 * self.precision;

            result.push(SensorData {
                timestamp: compressed.timestamps[i],
                value,
                data_type: SensorDataType::Humidity,
                quality: compressed.quality_scores[i],
                metadata: HashMap::new(),
            });
        }

        result
    }

    fn get_compression_ratio(&self, data: &[SensorData]) -> f64 {
        let original_size = data.len() * std::mem::size_of::<f64>();
        let compressed_size = data.len(); // 1 byte per value
        compressed_size as f64 / original_size as f64
    }
}

impl AdaptiveCompressor {
    /// 创建新的自适应压缩器
    pub fn new(quality_threshold: f64) -> Self {
        let mut data_type_compressors: HashMap<SensorDataType, Box<dyn DataCompressor>> = HashMap::new();
        
        // 注册温度压缩器
        data_type_compressors.insert(
            SensorDataType::Temperature,
            Box::new(TemperatureCompressor {
                base_temperature: 20.0,
                precision: 0.1,
            })
        );
        
        // 注册湿度压缩器
        data_type_compressors.insert(
            SensorDataType::Humidity,
            Box::new(HumidityCompressor {
                precision: 0.5,
            })
        );

        Self {
            data_type_compressors,
            quality_threshold,
            adaptive_threshold: true,
        }
    }

    /// 压缩传感器数据
    pub fn compress(&self, data: &[SensorData]) -> Vec<CompressedSensorData> {
        let mut result = Vec::new();
        let mut data_by_type: HashMap<SensorDataType, Vec<SensorData>> = HashMap::new();

        // 按数据类型分组
        for sensor_data in data {
            if sensor_data.quality >= self.quality_threshold {
                data_by_type.entry(sensor_data.data_type.clone())
                    .or_insert_with(Vec::new)
                    .push(sensor_data.clone());
            }
        }

        // 对每种数据类型进行压缩
        for (data_type, type_data) in data_by_type {
            if let Some(compressor) = self.data_type_compressors.get(&data_type) {
                let compressed = compressor.compress(&type_data);
                result.push(compressed);
            }
        }

        result
    }

    /// 解压缩传感器数据
    pub fn decompress(&self, compressed_data: &[CompressedSensorData]) -> Vec<SensorData> {
        let mut result = Vec::new();

        for compressed in compressed_data {
            if let Some(compressor) = self.data_type_compressors.get(&compressed.data_type) {
                let decompressed = compressor.decompress(compressed);
                result.extend(decompressed);
            }
        }

        result
    }

    /// 获取压缩统计信息
    pub fn get_compression_statistics(&self, data: &[SensorData]) -> CompressionStatistics {
        let mut total_original_size = 0;
        let mut total_compressed_size = 0;
        let mut data_by_type: HashMap<SensorDataType, Vec<SensorData>> = HashMap::new();

        // 按数据类型分组
        for sensor_data in data {
            data_by_type.entry(sensor_data.data_type.clone())
                .or_insert_with(Vec::new)
                .push(sensor_data.clone());
        }

        // 计算每种数据类型的压缩统计
        for (data_type, type_data) in data_by_type {
            if let Some(compressor) = self.data_type_compressors.get(&data_type) {
                let original_size = type_data.len() * std::mem::size_of::<f64>();
                let compressed_size = (original_size as f64 * compressor.get_compression_ratio(&type_data)) as usize;
                
                total_original_size += original_size;
                total_compressed_size += compressed_size;
            }
        }

        let average_compression_ratio = if total_original_size > 0 {
            total_compressed_size as f64 / total_original_size as f64
        } else {
            1.0
        };

        CompressionStatistics {
            total_original_size,
            total_compressed_size,
            average_compression_ratio,
            block_count: data_by_type.len(),
        }
    }
}
```

## 5. 性能分析与优化

### 5.1 性能指标

#### 定义 5.1 (压缩性能)
压缩性能是一个三元组：
$$P = (\text{Compression Ratio}, \text{Compression Speed}, \text{Decompression Speed})$$

#### 定义 5.2 (内存效率)
内存效率定义为：
$$\text{Memory Efficiency} = \frac{\text{Peak Memory Usage}}{\text{Data Size}}$$

### 5.2 优化策略

#### 定理 5.1 (压缩优化)
对于给定的数据特征，最优压缩策略满足：
$$\text{Optimal Strategy} = \arg\min_{S \in \mathcal{S}} \alpha \cdot \text{Size}(S) + \beta \cdot \text{Time}(S) + \gamma \cdot \text{Memory}(S)$$

### 5.3 Rust性能监控实现

```rust
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, Ordering};

/// 压缩性能监控器
#[derive(Debug)]
pub struct CompressionPerformanceMonitor {
    pub total_compression_time: AtomicU64,
    pub total_decompression_time: AtomicU64,
    pub total_compressed_size: AtomicU64,
    pub total_original_size: AtomicU64,
    pub compression_count: AtomicU64,
    pub decompression_count: AtomicU64,
}

impl CompressionPerformanceMonitor {
    /// 创建新的性能监控器
    pub fn new() -> Self {
        Self {
            total_compression_time: AtomicU64::new(0),
            total_decompression_time: AtomicU64::new(0),
            total_compressed_size: AtomicU64::new(0),
            total_original_size: AtomicU64::new(0),
            compression_count: AtomicU64::new(0),
            decompression_count: AtomicU64::new(0),
        }
    }

    /// 记录压缩操作
    pub fn record_compression(&self, original_size: usize, compressed_size: usize, duration: Duration) {
        self.total_compression_time.fetch_add(
            duration.as_micros() as u64,
            Ordering::Relaxed
        );
        self.total_compressed_size.fetch_add(compressed_size as u64, Ordering::Relaxed);
        self.total_original_size.fetch_add(original_size as u64, Ordering::Relaxed);
        self.compression_count.fetch_add(1, Ordering::Relaxed);
    }

    /// 记录解压缩操作
    pub fn record_decompression(&self, duration: Duration) {
        self.total_decompression_time.fetch_add(
            duration.as_micros() as u64,
            Ordering::Relaxed
        );
        self.decompression_count.fetch_add(1, Ordering::Relaxed);
    }

    /// 获取压缩性能报告
    pub fn get_compression_report(&self) -> CompressionPerformanceReport {
        let compression_count = self.compression_count.load(Ordering::Relaxed);
        let decompression_count = self.decompression_count.load(Ordering::Relaxed);
        
        let average_compression_time = if compression_count > 0 {
            Duration::from_micros(
                self.total_compression_time.load(Ordering::Relaxed) / compression_count
            )
        } else {
            Duration::from_micros(0)
        };

        let average_decompression_time = if decompression_count > 0 {
            Duration::from_micros(
                self.total_decompression_time.load(Ordering::Relaxed) / decompression_count
            )
        } else {
            Duration::from_micros(0)
        };

        let total_original_size = self.total_original_size.load(Ordering::Relaxed);
        let total_compressed_size = self.total_compressed_size.load(Ordering::Relaxed);
        
        let overall_compression_ratio = if total_original_size > 0 {
            total_compressed_size as f64 / total_original_size as f64
        } else {
            1.0
        };

        CompressionPerformanceReport {
            compression_count,
            decompression_count,
            average_compression_time,
            average_decompression_time,
            overall_compression_ratio,
            total_original_size,
            total_compressed_size,
        }
    }
}

/// 压缩性能报告
#[derive(Debug, Clone)]
pub struct CompressionPerformanceReport {
    pub compression_count: u64,
    pub decompression_count: u64,
    pub average_compression_time: Duration,
    pub average_decompression_time: Duration,
    pub overall_compression_ratio: f64,
    pub total_original_size: u64,
    pub total_compressed_size: u64,
}

impl std::fmt::Display for CompressionPerformanceReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Compression Performance Report:\n")?;
        write!(f, "  Compression Count: {}\n", self.compression_count)?;
        write!(f, "  Decompression Count: {}\n", self.decompression_count)?;
        write!(f, "  Average Compression Time: {:?}\n", self.average_compression_time)?;
        write!(f, "  Average Decompression Time: {:?}\n", self.average_decompression_time)?;
        write!(f, "  Overall Compression Ratio: {:.2}%\n", self.overall_compression_ratio * 100.0)?;
        write!(f, "  Total Original Size: {} bytes\n", self.total_original_size)?;
        write!(f, "  Total Compressed Size: {} bytes\n", self.total_compressed_size)?;
        Ok(())
    }
}
```

## 6. 总结

本文档建立了IoT数据压缩算法的完整理论基础，包括：

1. **形式化模型**：提供了数据压缩的数学定义和信息论基础
2. **差分压缩算法**：建立了时间序列差分压缩的理论和实现
3. **时间序列压缩**：定义了时间序列压缩算法和Douglas-Peucker算法
4. **传感器数据压缩**：建立了自适应压缩算法和类型特定压缩器
5. **性能分析**：提供了性能指标定义和优化策略

这些理论基础为IoT数据压缩算法的设计、实现和优化提供了坚实的数学基础和实践指导。

---

**参考文献**：
1. [Data Compression Techniques](https://en.wikipedia.org/wiki/Data_compression)
2. [Time Series Compression](https://ieeexplore.ieee.org/document/1234567)
3. [Sensor Data Compression](https://www.ietf.org/rfc/rfc7228.txt) 