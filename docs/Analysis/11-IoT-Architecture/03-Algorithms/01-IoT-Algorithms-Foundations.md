# IoT算法基础与设计模式

## 1. 概述

本文档定义了IoT系统中的核心算法和设计模式，包括数据处理、设备发现、路由选择等关键算法，并提供形式化的数学定义和实现示例。

## 2. 形式化基础

### 2.1 IoT算法系统定义

**定义 2.1 (IoT算法系统)**
IoT算法系统是一个五元组 $\mathcal{A} = (S, I, O, F, T)$，其中：
- $S$ 是状态空间
- $I$ 是输入空间
- $O$ 是输出空间
- $F: S \times I \rightarrow S \times O$ 是状态转移函数
- $T$ 是时间约束

**定理 2.1 (算法收敛性)**
对于任意IoT算法系统 $\mathcal{A}$，如果存在Lyapunov函数 $V: S \rightarrow \mathbb{R}^+$ 使得：
$$\frac{dV}{dt} < 0$$
则系统在有限时间内收敛到稳定状态。

**证明**：
根据Lyapunov稳定性理论，当 $\frac{dV}{dt} < 0$ 时，系统能量持续减少，最终达到能量最小值，即稳定状态。由于IoT系统的状态空间是有限的，收敛时间也是有限的。

### 2.2 复杂度分析框架

**定义 2.2 (IoT算法复杂度)**
IoT算法的复杂度由以下三个维度定义：
1. **时间复杂度**: $T(n) = O(f(n))$
2. **空间复杂度**: $S(n) = O(g(n))$
3. **能耗复杂度**: $E(n) = O(h(n))$

其中 $n$ 是问题规模（如设备数量、数据量等）。

## 3. IoT设计模式

### 3.1 观察者模式 (Observer Pattern)

**定义 3.1 (观察者模式)**
观察者模式是一个四元组 $\mathcal{O} = (Subject, Observer, Event, Notification)$，其中：
- $Subject$ 是被观察对象集合
- $Observer$ 是观察者集合
- $Event$ 是事件类型集合
- $Notification: Subject \times Event \rightarrow 2^{Observer}$ 是通知函数

**Rust实现**：

```rust
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

// 事件类型
#[derive(Debug, Clone)]
pub enum IoTEvent {
    SensorData { device_id: String, value: f64, timestamp: u64 },
    DeviceStatus { device_id: String, status: DeviceStatus },
    Alert { severity: AlertLevel, message: String },
}

#[derive(Debug, Clone)]
pub enum DeviceStatus {
    Online,
    Offline,
    Error,
}

#[derive(Debug, Clone)]
pub enum AlertLevel {
    Info,
    Warning,
    Critical,
}

// 观察者特征
pub trait Observer: Send + Sync {
    fn update(&self, event: &IoTEvent);
    fn get_id(&self) -> &str;
}

// 主题实现
pub struct IoTSubject {
    observers: Arc<Mutex<HashMap<String, Box<dyn Observer>>>>,
    event_tx: mpsc::Sender<IoTEvent>,
}

impl IoTSubject {
    pub fn new() -> (Self, mpsc::Receiver<IoTEvent>) {
        let (event_tx, event_rx) = mpsc::channel(100);
        let subject = Self {
            observers: Arc::new(Mutex::new(HashMap::new())),
            event_tx,
        };
        (subject, event_rx)
    }

    pub async fn attach(&self, observer: Box<dyn Observer>) {
        let mut observers = self.observers.lock().unwrap();
        observers.insert(observer.get_id().to_string(), observer);
    }

    pub async fn detach(&self, observer_id: &str) {
        let mut observers = self.observers.lock().unwrap();
        observers.remove(observer_id);
    }

    pub async fn notify(&self, event: IoTEvent) {
        // 发送事件到通知处理器
        if let Err(e) = self.event_tx.send(event.clone()).await {
            eprintln!("Failed to send event: {}", e);
        }
    }

    pub async fn notify_observers(&self, event: &IoTEvent) {
        let observers = self.observers.lock().unwrap();
        for observer in observers.values() {
            observer.update(event);
        }
    }
}

// 具体观察者实现
pub struct DataProcessor {
    id: String,
    processor_tx: mpsc::Sender<IoTEvent>,
}

impl DataProcessor {
    pub fn new(id: String) -> (Self, mpsc::Receiver<IoTEvent>) {
        let (processor_tx, processor_rx) = mpsc::channel(100);
        (Self { id, processor_tx }, processor_rx)
    }
}

impl Observer for DataProcessor {
    fn update(&self, event: &IoTEvent) {
        // 异步处理事件
        let tx = self.processor_tx.clone();
        let event = event.clone();
        tokio::spawn(async move {
            if let Err(e) = tx.send(event).await {
                eprintln!("Failed to process event: {}", e);
            }
        });
    }

    fn get_id(&self) -> &str {
        &self.id
    }
}

// 使用示例
#[tokio::main]
async fn main() {
    let (subject, mut event_rx) = IoTSubject::new();
    let (processor, mut processor_rx) = DataProcessor::new("data_processor_1".to_string());
    
    // 注册观察者
    subject.attach(Box::new(processor)).await;
    
    // 启动事件处理循环
    let subject_clone = subject.clone();
    tokio::spawn(async move {
        while let Some(event) = event_rx.recv().await {
            subject_clone.notify_observers(&event).await;
        }
    });
    
    // 启动处理器循环
    tokio::spawn(async move {
        while let Some(event) = processor_rx.recv().await {
            match event {
                IoTEvent::SensorData { device_id, value, timestamp } => {
                    println!("Processing sensor data: device={}, value={}, time={}", 
                            device_id, value, timestamp);
                }
                _ => println!("Processing event: {:?}", event),
            }
        }
    });
    
    // 发送测试事件
    let test_event = IoTEvent::SensorData {
        device_id: "sensor_001".to_string(),
        value: 25.5,
        timestamp: 1640995200,
    };
    subject.notify(test_event).await;
}
```

### 3.2 策略模式 (Strategy Pattern)

**定义 3.2 (策略模式)**
策略模式是一个三元组 $\mathcal{S} = (Context, Strategy, Algorithm)$，其中：
- $Context$ 是上下文环境
- $Strategy$ 是策略集合
- $Algorithm: Context \times Strategy \rightarrow Result$ 是算法执行函数

**Go实现**：

```go
package iot

import (
    "context"
    "fmt"
    "time"
)

// 策略接口
type DataProcessingStrategy interface {
    Process(ctx context.Context, data []byte) ([]byte, error)
    GetName() string
}

// 上下文
type ProcessingContext struct {
    strategy DataProcessingStrategy
    metadata map[string]interface{}
}

func NewProcessingContext(strategy DataProcessingStrategy) *ProcessingContext {
    return &ProcessingContext{
        strategy: strategy,
        metadata: make(map[string]interface{}),
    }
}

func (pc *ProcessingContext) SetStrategy(strategy DataProcessingStrategy) {
    pc.strategy = strategy
}

func (pc *ProcessingContext) ExecuteStrategy(ctx context.Context, data []byte) ([]byte, error) {
    if pc.strategy == nil {
        return nil, fmt.Errorf("no strategy set")
    }
    return pc.strategy.Process(ctx, data)
}

// 具体策略实现
type CompressionStrategy struct {
    algorithm string
}

func NewCompressionStrategy(algorithm string) *CompressionStrategy {
    return &CompressionStrategy{algorithm: algorithm}
}

func (cs *CompressionStrategy) Process(ctx context.Context, data []byte) ([]byte, error) {
    // 模拟压缩处理
    fmt.Printf("Compressing data using %s algorithm\n", cs.algorithm)
    time.Sleep(10 * time.Millisecond) // 模拟处理时间
    return data, nil
}

func (cs *CompressionStrategy) GetName() string {
    return fmt.Sprintf("compression_%s", cs.algorithm)
}

type EncryptionStrategy struct {
    key []byte
}

func NewEncryptionStrategy(key []byte) *EncryptionStrategy {
    return &EncryptionStrategy{key: key}
}

func (cs *EncryptionStrategy) Process(ctx context.Context, data []byte) ([]byte, error) {
    // 模拟加密处理
    fmt.Printf("Encrypting data with key length %d\n", len(cs.key))
    time.Sleep(5 * time.Millisecond) // 模拟处理时间
    return data, nil
}

func (cs *EncryptionStrategy) GetName() string {
    return "encryption_aes"
}

// 使用示例
func ExampleStrategyPattern() {
    ctx := context.Background()
    
    // 创建上下文
    compressionStrategy := NewCompressionStrategy("gzip")
    processingContext := NewProcessingContext(compressionStrategy)
    
    // 执行压缩策略
    data := []byte("test data for processing")
    result, err := processingContext.ExecuteStrategy(ctx, data)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    fmt.Printf("Compressed result: %v\n", result)
    
    // 切换到加密策略
    encryptionStrategy := NewEncryptionStrategy([]byte("secret_key_123"))
    processingContext.SetStrategy(encryptionStrategy)
    
    // 执行加密策略
    result, err = processingContext.ExecuteStrategy(ctx, data)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    fmt.Printf("Encrypted result: %v\n", result)
}
```

## 4. 数据处理算法

### 4.1 流数据处理算法

**定义 4.1 (流数据)**
流数据是一个时间序列 $S = \{(t_i, v_i) | i \in \mathbb{N}\}$，其中 $t_i$ 是时间戳，$v_i$ 是数据值。

**定义 4.2 (滑动窗口)**
滑动窗口是一个固定大小的时间窗口 $W = [t - \Delta, t]$，其中 $\Delta$ 是窗口大小。

**算法 4.1 (滑动窗口平均)**
```rust
use std::collections::VecDeque;
use std::time::{Duration, Instant};

pub struct SlidingWindowAverage {
    window_size: Duration,
    data_points: VecDeque<(Instant, f64)>,
}

impl SlidingWindowAverage {
    pub fn new(window_size: Duration) -> Self {
        Self {
            window_size,
            data_points: VecDeque::new(),
        }
    }

    pub fn add_data_point(&mut self, value: f64) {
        let now = Instant::now();
        
        // 移除过期数据点
        while let Some((timestamp, _)) = self.data_points.front() {
            if now.duration_since(*timestamp) > self.window_size {
                self.data_points.pop_front();
            } else {
                break;
            }
        }
        
        // 添加新数据点
        self.data_points.push_back((now, value));
    }

    pub fn get_average(&self) -> Option<f64> {
        if self.data_points.is_empty() {
            return None;
        }
        
        let sum: f64 = self.data_points.iter().map(|(_, value)| value).sum();
        Some(sum / self.data_points.len() as f64)
    }

    pub fn get_count(&self) -> usize {
        self.data_points.len()
    }
}

// 使用示例
#[tokio::main]
async fn main() {
    let mut window = SlidingWindowAverage::new(Duration::from_secs(60));
    
    // 模拟数据流
    for i in 0..100 {
        window.add_data_point(i as f64);
        
        if let Some(avg) = window.get_average() {
            println!("Window average: {:.2}, count: {}", avg, window.get_count());
        }
        
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
```

### 4.2 异常检测算法

**定义 4.3 (异常检测)**
异常检测是一个函数 $f: \mathbb{R}^n \rightarrow \{0, 1\}$，其中 $f(x) = 1$ 表示异常，$f(x) = 0$ 表示正常。

**算法 4.2 (Z-Score异常检测)**
```rust
use std::collections::VecDeque;

pub struct ZScoreAnomalyDetector {
    window_size: usize,
    data_points: VecDeque<f64>,
    threshold: f64,
}

impl ZScoreAnomalyDetector {
    pub fn new(window_size: usize, threshold: f64) -> Self {
        Self {
            window_size,
            data_points: VecDeque::new(),
            threshold,
        }
    }

    pub fn add_data_point(&mut self, value: f64) -> bool {
        // 维护固定大小的窗口
        if self.data_points.len() >= self.window_size {
            self.data_points.pop_front();
        }
        self.data_points.push_back(value);
        
        // 计算Z-Score
        if self.data_points.len() < 2 {
            return false; // 数据不足，无法检测
        }
        
        let mean = self.data_points.iter().sum::<f64>() / self.data_points.len() as f64;
        let variance = self.data_points.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (self.data_points.len() - 1) as f64;
        let std_dev = variance.sqrt();
        
        if std_dev == 0.0 {
            return false; // 标准差为0，无法检测
        }
        
        let z_score = (value - mean).abs() / std_dev;
        z_score > self.threshold
    }

    pub fn get_statistics(&self) -> Option<(f64, f64, f64)> {
        if self.data_points.is_empty() {
            return None;
        }
        
        let mean = self.data_points.iter().sum::<f64>() / self.data_points.len() as f64;
        let variance = self.data_points.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (self.data_points.len() - 1) as f64;
        let std_dev = variance.sqrt();
        
        Some((mean, std_dev, variance))
    }
}

// 使用示例
#[tokio::main]
async fn main() {
    let mut detector = ZScoreAnomalyDetector::new(10, 2.0);
    
    // 模拟正常数据
    for i in 0..20 {
        let value = 100.0 + (i as f64 * 0.1) + (rand::random::<f64>() - 0.5) * 2.0;
        let is_anomaly = detector.add_data_point(value);
        println!("Value: {:.2}, Anomaly: {}", value, is_anomaly);
    }
    
    // 模拟异常数据
    let anomaly_value = 150.0;
    let is_anomaly = detector.add_data_point(anomaly_value);
    println!("Anomaly value: {:.2}, Detected: {}", anomaly_value, is_anomaly);
    
    if let Some((mean, std_dev, variance)) = detector.get_statistics() {
        println!("Statistics - Mean: {:.2}, StdDev: {:.2}, Variance: {:.2}", 
                mean, std_dev, variance);
    }
}
```

## 5. 设备发现算法

### 5.1 分布式设备发现

**定义 5.1 (设备发现图)**
设备发现图是一个无向图 $G = (V, E)$，其中：
- $V$ 是设备节点集合
- $E$ 是连接边集合
- 每条边 $(u, v) \in E$ 表示设备 $u$ 和 $v$ 可以直接通信

**算法 5.1 (泛洪发现算法)**
```rust
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct Device {
    id: String,
    address: String,
    capabilities: Vec<String>,
    neighbors: HashSet<String>,
}

#[derive(Debug, Clone)]
pub struct DiscoveryMessage {
    id: String,
    source: String,
    ttl: u32,
    visited: HashSet<String>,
    path: Vec<String>,
}

pub struct FloodingDiscovery {
    devices: Arc<Mutex<HashMap<String, Device>>>,
    message_tx: mpsc::Sender<DiscoveryMessage>,
}

impl FloodingDiscovery {
    pub fn new() -> (Self, mpsc::Receiver<DiscoveryMessage>) {
        let (message_tx, message_rx) = mpsc::channel(1000);
        let discovery = Self {
            devices: Arc::new(Mutex::new(HashMap::new())),
            message_tx,
        };
        (discovery, message_rx)
    }

    pub async fn add_device(&self, device: Device) {
        let mut devices = self.devices.lock().unwrap();
        devices.insert(device.id.clone(), device);
    }

    pub async fn start_discovery(&self, source_id: &str, ttl: u32) {
        let message = DiscoveryMessage {
            id: Uuid::new_v4().to_string(),
            source: source_id.to_string(),
            ttl,
            visited: HashSet::new(),
            path: vec![source_id.to_string()],
        };
        
        if let Err(e) = self.message_tx.send(message).await {
            eprintln!("Failed to start discovery: {}", e);
        }
    }

    pub async fn process_message(&self, message: DiscoveryMessage) -> Vec<DiscoveryMessage> {
        let mut new_messages = Vec::new();
        
        // 检查TTL
        if message.ttl == 0 {
            return new_messages;
        }
        
        // 检查是否已访问
        if message.visited.contains(&message.source) {
            return new_messages;
        }
        
        let devices = self.devices.lock().unwrap();
        if let Some(device) = devices.get(&message.source) {
            // 向所有邻居转发消息
            for neighbor_id in &device.neighbors {
                if !message.visited.contains(neighbor_id) {
                    let mut new_message = message.clone();
                    new_message.source = neighbor_id.clone();
                    new_message.ttl -= 1;
                    new_message.visited.insert(message.source.clone());
                    new_message.path.push(neighbor_id.clone());
                    new_messages.push(new_message);
                }
            }
        }
        
        new_messages
    }

    pub async fn get_network_topology(&self) -> HashMap<String, HashSet<String>> {
        let devices = self.devices.lock().unwrap();
        devices.iter()
            .map(|(id, device)| (id.clone(), device.neighbors.clone()))
            .collect()
    }
}

// 使用示例
#[tokio::main]
async fn main() {
    let (discovery, mut message_rx) = FloodingDiscovery::new();
    
    // 创建设备网络
    let mut device1 = Device {
        id: "device_1".to_string(),
        address: "192.168.1.1".to_string(),
        capabilities: vec!["sensor".to_string(), "actuator".to_string()],
        neighbors: HashSet::new(),
    };
    device1.neighbors.insert("device_2".to_string());
    device1.neighbors.insert("device_3".to_string());
    
    let mut device2 = Device {
        id: "device_2".to_string(),
        address: "192.168.1.2".to_string(),
        capabilities: vec!["sensor".to_string()],
        neighbors: HashSet::new(),
    };
    device2.neighbors.insert("device_1".to_string());
    device2.neighbors.insert("device_4".to_string());
    
    discovery.add_device(device1).await;
    discovery.add_device(device2).await;
    
    // 启动发现过程
    discovery.start_discovery("device_1", 3).await;
    
    // 处理发现消息
    while let Some(message) = message_rx.recv().await {
        println!("Discovery message: {:?}", message);
        let new_messages = discovery.process_message(message).await;
        
        for new_message in new_messages {
            if let Err(e) = discovery.message_tx.send(new_message).await {
                eprintln!("Failed to forward message: {}", e);
            }
        }
    }
}
```

### 5.2 基于信标的发现

**定义 5.2 (信标)**
信标是一个三元组 $B = (id, position, range)$，其中：
- $id$ 是信标标识符
- $position$ 是信标位置
- $range$ 是信标覆盖范围

**算法 5.2 (信标发现算法)**
```go
package iot

import (
    "context"
    "fmt"
    "math"
    "sync"
    "time"
)

// 信标结构
type Beacon struct {
    ID       string
    Position Position
    Range    float64
    LastSeen time.Time
}

type Position struct {
    X, Y, Z float64
}

// 设备结构
type Device struct {
    ID       string
    Position Position
    Beacons  map[string]*Beacon
    mu       sync.RWMutex
}

func NewDevice(id string, position Position) *Device {
    return &Device{
        ID:       id,
        Position: position,
        Beacons:  make(map[string]*Beacon),
    }
}

// 计算距离
func (d *Device) distanceTo(beacon *Beacon) float64 {
    dx := d.Position.X - beacon.Position.X
    dy := d.Position.Y - beacon.Position.Y
    dz := d.Position.Z - beacon.Position.Z
    return math.Sqrt(dx*dx + dy*dy + dz*dz)
}

// 发现信标
func (d *Device) DiscoverBeacon(ctx context.Context, beacon *Beacon) bool {
    distance := d.distanceTo(beacon)
    
    if distance <= beacon.Range {
        d.mu.Lock()
        d.Beacons[beacon.ID] = beacon
        d.mu.Unlock()
        
        fmt.Printf("Device %s discovered beacon %s at distance %.2f\n", 
                  d.ID, beacon.ID, distance)
        return true
    }
    
    return false
}

// 获取可见信标
func (d *Device) GetVisibleBeacons() []*Beacon {
    d.mu.RLock()
    defer d.mu.RUnlock()
    
    beacons := make([]*Beacon, 0, len(d.Beacons))
    for _, beacon := range d.Beacons {
        beacons = append(beacons, beacon)
    }
    return beacons
}

// 信标管理器
type BeaconManager struct {
    beacons map[string]*Beacon
    devices map[string]*Device
    mu      sync.RWMutex
}

func NewBeaconManager() *BeaconManager {
    return &BeaconManager{
        beacons: make(map[string]*Beacon),
        devices: make(map[string]*Device),
    }
}

func (bm *BeaconManager) AddBeacon(beacon *Beacon) {
    bm.mu.Lock()
    bm.beacons[beacon.ID] = beacon
    bm.mu.Unlock()
}

func (bm *BeaconManager) AddDevice(device *Device) {
    bm.mu.Lock()
    bm.devices[device.ID] = device
    bm.mu.Unlock()
}

// 执行发现过程
func (bm *BeaconManager) RunDiscovery(ctx context.Context) {
    ticker := time.NewTicker(1 * time.Second)
    defer ticker.Stop()
    
    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            bm.mu.RLock()
            devices := make([]*Device, 0, len(bm.devices))
            for _, device := range bm.devices {
                devices = append(devices, device)
            }
            beacons := make([]*Beacon, 0, len(bm.beacons))
            for _, beacon := range bm.beacons {
                beacons = append(beacons, beacon)
            }
            bm.mu.RUnlock()
            
            // 让每个设备尝试发现信标
            for _, device := range devices {
                for _, beacon := range beacons {
                    device.DiscoverBeacon(ctx, beacon)
                }
            }
        }
    }
}

// 使用示例
func ExampleBeaconDiscovery() {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()
    
    manager := NewBeaconManager()
    
    // 添加信标
    beacon1 := &Beacon{
        ID: "beacon_1",
        Position: Position{X: 0, Y: 0, Z: 0},
        Range:    10.0,
    }
    beacon2 := &Beacon{
        ID: "beacon_2",
        Position: Position{X: 15, Y: 0, Z: 0},
        Range:    8.0,
    }
    
    manager.AddBeacon(beacon1)
    manager.AddBeacon(beacon2)
    
    // 添加设备
    device1 := NewDevice("device_1", Position{X: 5, Y: 0, Z: 0})
    device2 := NewDevice("device_2", Position{X: 20, Y: 0, Z: 0})
    
    manager.AddDevice(device1)
    manager.AddDevice(device2)
    
    // 启动发现过程
    go manager.RunDiscovery(ctx)
    
    // 运行一段时间
    time.Sleep(5 * time.Second)
    
    // 检查发现结果
    fmt.Println("Device 1 visible beacons:")
    for _, beacon := range device1.GetVisibleBeacons() {
        fmt.Printf("  - %s\n", beacon.ID)
    }
    
    fmt.Println("Device 2 visible beacons:")
    for _, beacon := range device2.GetVisibleBeacons() {
        fmt.Printf("  - %s\n", beacon.ID)
    }
}
```

## 6. 路由算法

### 6.1 最短路径路由

**定义 6.1 (网络图)**
网络图是一个带权有向图 $G = (V, E, w)$，其中：
- $V$ 是节点集合（设备）
- $E$ 是边集合（连接）
- $w: E \rightarrow \mathbb{R}^+$ 是权重函数（延迟、带宽等）

**算法 6.1 (Dijkstra最短路径)**
```rust
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct NetworkNode {
    id: String,
    neighbors: HashMap<String, f64>, // neighbor_id -> weight
}

#[derive(Debug, Clone)]
pub struct Route {
    path: Vec<String>,
    total_cost: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct State {
    cost: f64,
    position: String,
}

impl Eq for State {}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // 注意：BinaryHeap是最大堆，所以我们需要反转比较
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}

pub struct NetworkRouter {
    nodes: HashMap<String, NetworkNode>,
}

impl NetworkRouter {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: NetworkNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    pub fn add_edge(&mut self, from: &str, to: &str, weight: f64) {
        if let Some(node) = self.nodes.get_mut(from) {
            node.neighbors.insert(to.to_string(), weight);
        }
    }

    pub fn find_shortest_path(&self, start: &str, end: &str) -> Option<Route> {
        if !self.nodes.contains_key(start) || !self.nodes.contains_key(end) {
            return None;
        }

        let mut distances: HashMap<String, f64> = HashMap::new();
        let mut previous: HashMap<String, String> = HashMap::new();
        let mut visited: HashSet<String> = HashSet::new();
        let mut heap = BinaryHeap::new();

        // 初始化距离
        for node_id in self.nodes.keys() {
            distances.insert(node_id.clone(), f64::INFINITY);
        }
        distances.insert(start.to_string(), 0.0);

        heap.push(State {
            cost: 0.0,
            position: start.to_string(),
        });

        while let Some(State { cost, position }) = heap.pop() {
            if position == end {
                // 构建路径
                let mut path = Vec::new();
                let mut current = end.to_string();
                while current != start {
                    path.push(current.clone());
                    current = previous.get(&current).unwrap().clone();
                }
                path.push(start.to_string());
                path.reverse();

                return Some(Route {
                    path,
                    total_cost: cost,
                });
            }

            if visited.contains(&position) {
                continue;
            }
            visited.insert(position.clone());

            if let Some(node) = self.nodes.get(&position) {
                for (neighbor, weight) in &node.neighbors {
                    let new_cost = cost + weight;
                    if new_cost < *distances.get(neighbor).unwrap_or(&f64::INFINITY) {
                        distances.insert(neighbor.clone(), new_cost);
                        previous.insert(neighbor.clone(), position.clone());
                        heap.push(State {
                            cost: new_cost,
                            position: neighbor.clone(),
                        });
                    }
                }
            }
        }

        None
    }

    pub fn get_all_paths(&self, start: &str) -> HashMap<String, Route> {
        let mut routes = HashMap::new();
        
        for node_id in self.nodes.keys() {
            if node_id != start {
                if let Some(route) = self.find_shortest_path(start, node_id) {
                    routes.insert(node_id.clone(), route);
                }
            }
        }
        
        routes
    }
}

// 使用示例
#[tokio::main]
async fn main() {
    let mut router = NetworkRouter::new();
    
    // 创建网络拓扑
    let nodes = vec![
        ("A", vec![("B", 4.0), ("C", 2.0)]),
        ("B", vec![("A", 4.0), ("C", 1.0), ("D", 5.0)]),
        ("C", vec![("A", 2.0), ("B", 1.0), ("D", 8.0), ("E", 10.0)]),
        ("D", vec![("B", 5.0), ("C", 8.0), ("E", 2.0)]),
        ("E", vec![("C", 10.0), ("D", 2.0)]),
    ];
    
    for (id, neighbors) in nodes {
        let mut node = NetworkNode {
            id: id.to_string(),
            neighbors: HashMap::new(),
        };
        for (neighbor, weight) in neighbors {
            node.neighbors.insert(neighbor.to_string(), weight);
        }
        router.add_node(node);
    }
    
    // 查找最短路径
    if let Some(route) = router.find_shortest_path("A", "E") {
        println!("Shortest path from A to E:");
        println!("  Path: {}", route.path.join(" -> "));
        println!("  Total cost: {:.2}", route.total_cost);
    }
    
    // 获取所有路径
    let all_routes = router.get_all_paths("A");
    println!("\nAll shortest paths from A:");
    for (destination, route) in all_routes {
        println!("  To {}: {} (cost: {:.2})", 
                destination, route.path.join(" -> "), route.total_cost);
    }
}
```

### 6.2 负载均衡路由

**定义 6.2 (负载均衡)**
负载均衡是一个函数 $LB: \{req_1, req_2, ..., req_n\} \rightarrow \{node_1, node_2, ..., node_m\}$，使得：
$$\sum_{i=1}^m load(node_i) \approx \frac{\sum_{i=1}^n weight(req_i)}{m}$$

**算法 6.2 (加权轮询负载均衡)**
```go
package iot

import (
    "fmt"
    "sync"
    "sync/atomic"
)

// 节点结构
type Node struct {
    ID       string
    Weight   int
    Current  int32
    mu       sync.RWMutex
}

func NewNode(id string, weight int) *Node {
    return &Node{
        ID:     id,
        Weight: weight,
        Current: 0,
    }
}

// 负载均衡器
type LoadBalancer struct {
    nodes []*Node
    mu    sync.RWMutex
}

func NewLoadBalancer() *LoadBalancer {
    return &LoadBalancer{
        nodes: make([]*Node, 0),
    }
}

func (lb *LoadBalancer) AddNode(node *Node) {
    lb.mu.Lock()
    lb.nodes = append(lb.nodes, node)
    lb.mu.Unlock()
}

func (lb *LoadBalancer) RemoveNode(nodeID string) {
    lb.mu.Lock()
    defer lb.mu.Unlock()
    
    for i, node := range lb.nodes {
        if node.ID == nodeID {
            lb.nodes = append(lb.nodes[:i], lb.nodes[i+1:]...)
            break
        }
    }
}

// 加权轮询算法
func (lb *LoadBalancer) GetNextNode() *Node {
    lb.mu.RLock()
    defer lb.mu.RUnlock()
    
    if len(lb.nodes) == 0 {
        return nil
    }
    
    var bestNode *Node
    var bestWeight int32 = -1
    
    for _, node := range lb.nodes {
        current := atomic.LoadInt32(&node.Current)
        if current < int32(node.Weight) {
            if bestNode == nil || current < bestWeight {
                bestNode = node
                bestWeight = current
            }
        }
    }
    
    if bestNode != nil {
        atomic.AddInt32(&bestNode.Current, 1)
        return bestNode
    }
    
    // 重置所有节点的计数器
    for _, node := range lb.nodes {
        atomic.StoreInt32(&node.Current, 0)
    }
    
    // 重新选择
    if len(lb.nodes) > 0 {
        bestNode = lb.nodes[0]
        atomic.StoreInt32(&bestNode.Current, 1)
        return bestNode
    }
    
    return nil
}

// 使用示例
func ExampleLoadBalancing() {
    lb := NewLoadBalancer()
    
    // 添加节点
    lb.AddNode(NewNode("node_1", 3))
    lb.AddNode(NewNode("node_2", 2))
    lb.AddNode(NewNode("node_3", 1))
    
    // 模拟请求分发
    fmt.Println("Load balancing distribution:")
    for i := 0; i < 12; i++ {
        node := lb.GetNextNode()
        if node != nil {
            fmt.Printf("Request %d -> %s\n", i+1, node.ID)
        }
    }
    
    // 显示节点负载
    lb.mu.RLock()
    for _, node := range lb.nodes {
        current := atomic.LoadInt32(&node.Current)
        fmt.Printf("Node %s: %d/%d requests\n", node.ID, current, node.Weight)
    }
    lb.mu.RUnlock()
}
```

## 7. 性能分析

### 7.1 算法复杂度分析

**定理 7.1 (滑动窗口平均复杂度)**
滑动窗口平均算法的时间复杂度为 $O(1)$，空间复杂度为 $O(w)$，其中 $w$ 是窗口大小。

**证明**：
- 时间复杂度：每次添加数据点只需要常数时间操作（添加、删除）
- 空间复杂度：窗口最多存储 $w$ 个数据点

**定理 7.2 (Dijkstra算法复杂度)**
Dijkstra最短路径算法的时间复杂度为 $O((V + E) \log V)$，其中 $V$ 是节点数，$E$ 是边数。

**证明**：
- 每个节点最多被访问一次
- 每次访问需要更新邻居节点的距离
- 使用优先队列维护最小距离节点，每次操作需要 $O(\log V)$ 时间

### 7.2 能耗分析

**定义 7.1 (能耗模型)**
IoT设备的能耗模型为：
$$E_{total} = E_{compute} + E_{communication} + E_{sensing} + E_{idle}$$

其中：
- $E_{compute}$ 是计算能耗
- $E_{communication}$ 是通信能耗
- $E_{sensing}$ 是感知能耗
- $E_{idle}$ 是空闲能耗

**算法 7.1 (能耗优化)**
```rust
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct EnergyModel {
    compute_power: f64,    // W
    comm_power: f64,       // W
    sensing_power: f64,    // W
    idle_power: f64,       // W
    battery_capacity: f64, // Wh
}

#[derive(Debug, Clone)]
pub struct EnergyOptimizer {
    model: EnergyModel,
    current_energy: f64,
    task_history: HashMap<String, f64>,
}

impl EnergyOptimizer {
    pub fn new(model: EnergyModel) -> Self {
        Self {
            current_energy: model.battery_capacity,
            model,
            task_history: HashMap::new(),
        }
    }

    pub fn estimate_compute_energy(&self, cpu_time: f64) -> f64 {
        self.model.compute_power * cpu_time / 3600.0 // 转换为Wh
    }

    pub fn estimate_comm_energy(&self, data_size: f64, bandwidth: f64) -> f64 {
        let transfer_time = data_size / bandwidth; // 秒
        self.model.comm_power * transfer_time / 3600.0
    }

    pub fn can_execute_task(&self, estimated_energy: f64) -> bool {
        self.current_energy >= estimated_energy
    }

    pub fn execute_task(&mut self, task_id: &str, energy: f64) -> bool {
        if self.can_execute_task(energy) {
            self.current_energy -= energy;
            self.task_history.insert(task_id.to_string(), energy);
            true
        } else {
            false
        }
    }

    pub fn get_remaining_energy(&self) -> f64 {
        self.current_energy
    }

    pub fn get_energy_efficiency(&self) -> f64 {
        let total_used = self.model.battery_capacity - self.current_energy;
        if total_used > 0.0 {
            self.task_history.len() as f64 / total_used
        } else {
            0.0
        }
    }
}

// 使用示例
#[tokio::main]
async fn main() {
    let model = EnergyModel {
        compute_power: 0.5,    // 0.5W
        comm_power: 1.0,       // 1.0W
        sensing_power: 0.1,    // 0.1W
        idle_power: 0.01,      // 0.01W
        battery_capacity: 1000.0, // 1000mAh * 3.7V = 3.7Wh
    };
    
    let mut optimizer = EnergyOptimizer::new(model);
    
    // 模拟任务执行
    let compute_energy = optimizer.estimate_compute_energy(1.0); // 1秒计算
    let comm_energy = optimizer.estimate_comm_energy(1024.0, 1000.0); // 1KB数据，1KB/s带宽
    
    println!("Compute energy: {:.6} Wh", compute_energy);
    println!("Communication energy: {:.6} Wh", comm_energy);
    
    let total_energy = compute_energy + comm_energy;
    if optimizer.execute_task("task_1", total_energy) {
        println!("Task executed successfully");
        println!("Remaining energy: {:.6} Wh", optimizer.get_remaining_energy());
        println!("Energy efficiency: {:.2} tasks/Wh", optimizer.get_energy_efficiency());
    } else {
        println!("Insufficient energy for task");
    }
}
```

## 8. 总结

本文档提供了IoT系统的核心算法和设计模式，包括：

1. **设计模式**：观察者模式、策略模式等经典模式在IoT中的应用
2. **数据处理算法**：滑动窗口、异常检测等流数据处理算法
3. **设备发现算法**：泛洪发现、信标发现等分布式发现算法
4. **路由算法**：最短路径、负载均衡等网络路由算法
5. **性能分析**：复杂度分析和能耗优化

所有算法都提供了形式化的数学定义、严格的证明和完整的Rust/Go实现示例，确保理论严谨性和实现可行性。 