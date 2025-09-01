# Python语言特性深度分析

## 目录

- [Python语言特性深度分析](#python语言特性深度分析)
  - [目录](#目录)
  - [概述](#概述)
  - [1. 动态类型系统](#1-动态类型系统)
    - [1.1 类型注解和类型检查](#11-类型注解和类型检查)
    - [1.2 动态特性](#12-动态特性)
  - [2. 异步编程](#2-异步编程)
    - [2.1 asyncio基础](#21-asyncio基础)
    - [2.2 异步上下文管理](#22-异步上下文管理)
  - [3. 科学计算和数据处理](#3-科学计算和数据处理)
    - [3.1 NumPy和Pandas集成](#31-numpy和pandas集成)
    - [3.2 机器学习集成](#32-机器学习集成)
  - [4. 元编程和动态特性](#4-元编程和动态特性)
    - [4.1 装饰器和元类](#41-装饰器和元类)
    - [4.2 反射和动态属性](#42-反射和动态属性)
  - [5. 总结](#5-总结)

## 概述

本文档深入分析Python语言在IoT系统开发中的核心特性，包括动态类型系统、异步编程、元编程、科学计算库、机器学习集成等，为IoT开发者提供全面的Python技术指导。

## 1. 动态类型系统

### 1.1 类型注解和类型检查

```python
from typing import Dict, List, Optional, Union, Protocol, TypeVar, Generic
from dataclasses import dataclass
from datetime import datetime
import asyncio

# 类型注解示例
@dataclass
class IoTDevice:
    device_id: str
    name: str
    device_type: str
    status: str
    last_seen: datetime
    sensors: List['Sensor']

@dataclass
class Sensor:
    sensor_id: str
    sensor_type: str
    value: float
    unit: str
    timestamp: datetime

# 协议定义（类似接口）
class DataProcessor(Protocol):
    def process(self, data: List[Sensor]) -> Dict[str, float]:
        ...

class DeviceManager(Protocol):
    def register_device(self, device: IoTDevice) -> bool:
        ...
    
    def get_device(self, device_id: str) -> Optional[IoTDevice]:
        ...

# 泛型类型
T = TypeVar('T')

class IoTRepository(Generic[T]):
    def __init__(self):
        self._data: Dict[str, T] = {}
    
    def store(self, key: str, value: T) -> None:
        self._data[key] = value
    
    def retrieve(self, key: str) -> Optional[T]:
        return self._data.get(key)
    
    def get_all(self) -> List[T]:
        return list(self._data.values())

# 联合类型
def process_sensor_data(data: Union[Sensor, List[Sensor]]) -> Dict[str, float]:
    if isinstance(data, Sensor):
        return {data.sensor_type: data.value}
    else:
        return {sensor.sensor_type: sensor.value for sensor in data}

# 可选类型
def find_device_by_name(devices: List[IoTDevice], name: str) -> Optional[IoTDevice]:
    for device in devices:
        if device.name == name:
            return device
    return None
```

### 1.2 动态特性

```python
import inspect
from functools import wraps
import json

# 动态属性访问
class DynamicDevice:
    def __init__(self, device_id: str, **kwargs):
        self.device_id = device_id
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __getattr__(self, name):
        # 动态属性获取
        if name.startswith('sensor_'):
            sensor_type = name[7:]  # 移除 'sensor_' 前缀
            return self._get_sensor_value(sensor_type)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def _get_sensor_value(self, sensor_type: str) -> float:
        # 模拟传感器数据获取
        import random
        return random.uniform(0, 100)

# 使用示例
device = DynamicDevice("device_001", location="room_1", zone="building_a")
print(device.sensor_temperature)  # 动态获取温度传感器值
print(device.sensor_humidity)     # 动态获取湿度传感器值

# 装饰器模式
def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        import time
                        time.sleep(delay * (2 ** attempt))  # 指数退避
            raise last_exception
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def connect_to_device(device_id: str) -> bool:
    # 模拟设备连接
    import random
    if random.random() < 0.3:  # 30% 失败率
        raise ConnectionError(f"Failed to connect to device {device_id}")
    return True

# 元类编程
class DeviceMeta(type):
    def __new__(cls, name, bases, attrs):
        # 自动添加设备ID验证
        if 'device_id' in attrs:
            original_init = attrs.get('__init__')
            
            def new_init(self, *args, **kwargs):
                if original_init:
                    original_init(self, *args, **kwargs)
                if not hasattr(self, 'device_id') or not self.device_id:
                    raise ValueError("Device ID is required")
            
            attrs['__init__'] = new_init
        
        return super().__new__(cls, name, bases, attrs)

class SmartDevice(metaclass=DeviceMeta):
    def __init__(self, device_id: str, name: str):
        self.device_id = device_id
        self.name = name
```

## 2. 异步编程

### 2.1 asyncio基础

```python
import asyncio
import aiohttp
import json
from typing import AsyncGenerator, List
import time

# 异步设备管理器
class AsyncIoTDeviceManager:
    def __init__(self):
        self.devices: Dict[str, IoTDevice] = {}
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def register_device(self, device: IoTDevice) -> bool:
        """异步注册设备"""
        try:
            # 模拟异步设备注册
            await asyncio.sleep(0.1)
            self.devices[device.device_id] = device
            return True
        except Exception as e:
            print(f"Failed to register device {device.device_id}: {e}")
            return False
    
    async def collect_data_from_device(self, device_id: str) -> Optional[List[Sensor]]:
        """从单个设备收集数据"""
        if device_id not in self.devices:
            return None
        
        device = self.devices[device_id]
        
        # 模拟异步数据收集
        await asyncio.sleep(0.2)
        
        sensors = [
            Sensor(
                sensor_id=f"{device_id}_temp",
                sensor_type="temperature",
                value=25.5 + (time.time() % 10),
                unit="°C",
                timestamp=datetime.now()
            ),
            Sensor(
                sensor_id=f"{device_id}_humidity",
                sensor_type="humidity",
                value=60.0 + (time.time() % 20),
                unit="%",
                timestamp=datetime.now()
            )
        ]
        
        return sensors
    
    async def collect_data_from_all_devices(self) -> Dict[str, List[Sensor]]:
        """并发从所有设备收集数据"""
        tasks = []
        for device_id in self.devices.keys():
            task = asyncio.create_task(self.collect_data_from_device(device_id))
            tasks.append((device_id, task))
        
        results = {}
        for device_id, task in tasks:
            try:
                data = await task
                if data:
                    results[device_id] = data
            except Exception as e:
                print(f"Error collecting data from {device_id}: {e}")
        
        return results
    
    async def process_data_stream(self, device_id: str) -> AsyncGenerator[Sensor, None]:
        """异步数据流处理"""
        while True:
            try:
                sensors = await self.collect_data_from_device(device_id)
                if sensors:
                    for sensor in sensors:
                        yield sensor
                await asyncio.sleep(1.0)  # 每秒收集一次数据
            except Exception as e:
                print(f"Error in data stream for {device_id}: {e}")
                break

# 异步数据处理
class AsyncDataProcessor:
    def __init__(self):
        self.processors: List[DataProcessor] = []
    
    def add_processor(self, processor: DataProcessor):
        self.processors.append(processor)
    
    async def process_sensor_data(self, sensors: List[Sensor]) -> Dict[str, float]:
        """异步处理传感器数据"""
        results = {}
        
        # 并发处理每个传感器
        tasks = []
        for sensor in sensors:
            task = asyncio.create_task(self._process_single_sensor(sensor))
            tasks.append(task)
        
        processed_sensors = await asyncio.gather(*tasks, return_exceptions=True)
        
        for sensor, result in zip(sensors, processed_sensors):
            if isinstance(result, Exception):
                print(f"Error processing sensor {sensor.sensor_id}: {result}")
            else:
                results[sensor.sensor_type] = result
        
        return results
    
    async def _process_single_sensor(self, sensor: Sensor) -> float:
        """处理单个传感器数据"""
        # 模拟异步处理
        await asyncio.sleep(0.05)
        
        # 应用所有处理器
        value = sensor.value
        for processor in self.processors:
            value = processor.process([sensor]).get(sensor.sensor_type, value)
        
        return value

# 异步Web API
from aiohttp import web
import aiohttp_cors

class IoTWebAPI:
    def __init__(self, device_manager: AsyncIoTDeviceManager):
        self.device_manager = device_manager
        self.app = web.Application()
        self._setup_routes()
    
    def _setup_routes(self):
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        self.app.router.add_get('/devices', self.get_devices)
        self.app.router.add_get('/devices/{device_id}/data', self.get_device_data)
        self.app.router.add_post('/devices', self.register_device)
        
        # 添加CORS支持
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    async def get_devices(self, request):
        """获取所有设备信息"""
        devices = list(self.device_manager.devices.values())
        return web.json_response([{
            'device_id': device.device_id,
            'name': device.name,
            'device_type': device.device_type,
            'status': device.status,
            'last_seen': device.last_seen.isoformat()
        } for device in devices])
    
    async def get_device_data(self, request):
        """获取设备数据"""
        device_id = request.match_info['device_id']
        sensors = await self.device_manager.collect_data_from_device(device_id)
        
        if sensors is None:
            return web.json_response({'error': 'Device not found'}, status=404)
        
        return web.json_response([{
            'sensor_id': sensor.sensor_id,
            'sensor_type': sensor.sensor_type,
            'value': sensor.value,
            'unit': sensor.unit,
            'timestamp': sensor.timestamp.isoformat()
        } for sensor in sensors])
    
    async def register_device(self, request):
        """注册新设备"""
        try:
            data = await request.json()
            device = IoTDevice(
                device_id=data['device_id'],
                name=data['name'],
                device_type=data['device_type'],
                status='offline',
                last_seen=datetime.now(),
                sensors=[]
            )
            
            success = await self.device_manager.register_device(device)
            if success:
                return web.json_response({'status': 'success'})
            else:
                return web.json_response({'error': 'Registration failed'}, status=400)
        except Exception as e:
            return web.json_response({'error': str(e)}, status=400)
```

### 2.2 异步上下文管理

```python
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncContextManager

# 异步上下文管理器
class AsyncDeviceConnection:
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.connected = False
    
    async def __aenter__(self):
        print(f"Connecting to device {self.device_id}")
        await asyncio.sleep(0.1)  # 模拟连接时间
        self.connected = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print(f"Disconnecting from device {self.device_id}")
        self.connected = False
        if exc_type:
            print(f"Connection error: {exc_val}")
    
    async def send_data(self, data: str):
        if not self.connected:
            raise ConnectionError("Device not connected")
        print(f"Sending data to {self.device_id}: {data}")
        await asyncio.sleep(0.05)

# 使用异步上下文管理器
async def use_device_connection():
    async with AsyncDeviceConnection("device_001") as conn:
        await conn.send_data("Hello from IoT system")
        await conn.send_data("Temperature reading: 25.5°C")

# 异步上下文管理器装饰器
@asynccontextmanager
async def device_data_collection(device_id: str):
    """异步数据收集上下文管理器"""
    print(f"Starting data collection for {device_id}")
    start_time = time.time()
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        print(f"Data collection completed for {device_id} in {duration:.2f}s")

# 使用示例
async def collect_device_data():
    async with device_data_collection("device_001"):
        # 执行数据收集操作
        await asyncio.sleep(2.0)
        print("Data collection in progress...")

# 异步资源池
class AsyncResourcePool:
    def __init__(self, max_size: int, factory_func):
        self.max_size = max_size
        self.factory_func = factory_func
        self.pool: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.created_count = 0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """获取资源"""
        try:
            return self.pool.get_nowait()
        except asyncio.QueueEmpty:
            async with self.lock:
                if self.created_count < self.max_size:
                    resource = await self.factory_func()
                    self.created_count += 1
                    return resource
                else:
                    return await self.pool.get()
    
    async def release(self, resource):
        """释放资源"""
        await self.pool.put(resource)
    
    @asynccontextmanager
    async def get_resource(self):
        """获取资源的上下文管理器"""
        resource = await self.acquire()
        try:
            yield resource
        finally:
            await self.release(resource)

# 使用资源池
async def create_connection():
    return AsyncDeviceConnection("pooled_device")

async def use_resource_pool():
    pool = AsyncResourcePool(max_size=5, factory_func=create_connection)
    
    async with pool.get_resource() as conn:
        await conn.send_data("Using pooled connection")
```

## 3. 科学计算和数据处理

### 3.1 NumPy和Pandas集成

```python
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy import signal, stats

# IoT数据分析类
class IoTDataAnalyzer:
    def __init__(self):
        self.data_cache: Dict[str, pd.DataFrame] = {}
    
    def add_sensor_data(self, device_id: str, sensors: List[Sensor]):
        """添加传感器数据到分析器"""
        data = {
            'timestamp': [s.timestamp for s in sensors],
            'sensor_type': [s.sensor_type for s in sensors],
            'value': [s.value for s in sensors],
            'unit': [s.unit for s in sensors]
        }
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if device_id in self.data_cache:
            self.data_cache[device_id] = pd.concat([self.data_cache[device_id], df])
        else:
            self.data_cache[device_id] = df
    
    def analyze_temperature_trends(self, device_id: str) -> Dict[str, float]:
        """分析温度趋势"""
        if device_id not in self.data_cache:
            return {}
        
        df = self.data_cache[device_id]
        temp_data = df[df['sensor_type'] == 'temperature']
        
        if len(temp_data) < 2:
            return {}
        
        values = temp_data['value'].values
        timestamps = temp_data['timestamp'].values
        
        # 计算统计指标
        mean_temp = np.mean(values)
        std_temp = np.std(values)
        min_temp = np.min(values)
        max_temp = np.max(values)
        
        # 计算趋势
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # 检测异常值
        z_scores = np.abs(stats.zscore(values))
        outliers = np.sum(z_scores > 2)
        
        return {
            'mean_temperature': mean_temp,
            'std_temperature': std_temp,
            'min_temperature': min_temp,
            'max_temperature': max_temp,
            'trend_slope': slope,
            'trend_correlation': r_value,
            'outlier_count': outliers,
            'data_points': len(values)
        }
    
    def detect_anomalies(self, device_id: str, sensor_type: str, threshold: float = 2.0) -> List[Dict]:
        """检测异常值"""
        if device_id not in self.data_cache:
            return []
        
        df = self.data_cache[device_id]
        sensor_data = df[df['sensor_type'] == sensor_type]
        
        if len(sensor_data) < 3:
            return []
        
        values = sensor_data['value'].values
        z_scores = np.abs(stats.zscore(values))
        
        anomalies = []
        for i, (z_score, row) in enumerate(zip(z_scores, sensor_data.itertuples())):
            if z_score > threshold:
                anomalies.append({
                    'timestamp': row.timestamp,
                    'value': row.value,
                    'z_score': z_score,
                    'sensor_type': sensor_type
                })
        
        return anomalies
    
    def apply_smoothing_filter(self, device_id: str, sensor_type: str, window_size: int = 5) -> np.ndarray:
        """应用平滑滤波器"""
        if device_id not in self.data_cache:
            return np.array([])
        
        df = self.data_cache[device_id]
        sensor_data = df[df['sensor_type'] == sensor_type]
        
        if len(sensor_data) < window_size:
            return sensor_data['value'].values
        
        values = sensor_data['value'].values
        
        # 使用移动平均滤波器
        smoothed = signal.savgol_filter(values, window_size, 2)
        
        return smoothed
    
    def generate_report(self, device_id: str) -> Dict:
        """生成设备数据分析报告"""
        if device_id not in self.data_cache:
            return {'error': 'No data available'}
        
        df = self.data_cache[device_id]
        report = {
            'device_id': device_id,
            'total_data_points': len(df),
            'sensor_types': df['sensor_type'].unique().tolist(),
            'time_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            },
            'sensor_analysis': {}
        }
        
        # 分析每种传感器类型
        for sensor_type in df['sensor_type'].unique():
            sensor_data = df[df['sensor_type'] == sensor_type]
            values = sensor_data['value'].values
            
            report['sensor_analysis'][sensor_type] = {
                'count': len(values),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'anomalies': len(self.detect_anomalies(device_id, sensor_type))
            }
        
        return report

# 实时数据处理
class RealTimeDataProcessor:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data_buffers: Dict[str, List[float]] = {}
        self.processors: Dict[str, callable] = {}
    
    def add_processor(self, sensor_type: str, processor_func: callable):
        """添加数据处理器"""
        self.processors[sensor_type] = processor_func
    
    def process_data_point(self, device_id: str, sensor_type: str, value: float) -> Dict:
        """处理单个数据点"""
        key = f"{device_id}_{sensor_type}"
        
        if key not in self.data_buffers:
            self.data_buffers[key] = []
        
        buffer = self.data_buffers[key]
        buffer.append(value)
        
        # 保持窗口大小
        if len(buffer) > self.window_size:
            buffer.pop(0)
        
        # 应用处理器
        result = {'raw_value': value, 'processed_value': value}
        
        if sensor_type in self.processors and len(buffer) >= 5:
            try:
                processed = self.processors[sensor_type](np.array(buffer))
                result['processed_value'] = float(processed[-1]) if hasattr(processed, '__len__') else float(processed)
            except Exception as e:
                print(f"Processing error for {sensor_type}: {e}")
        
        return result
    
    def get_statistics(self, device_id: str, sensor_type: str) -> Dict:
        """获取统计信息"""
        key = f"{device_id}_{sensor_type}"
        if key not in self.data_buffers:
            return {}
        
        buffer = self.data_buffers[key]
        if not buffer:
            return {}
        
        values = np.array(buffer)
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'count': len(values)
        }
```

### 3.2 机器学习集成

```python
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# IoT机器学习分析器
class IoTMLAnalyzer:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.anomaly_models: Dict[str, IsolationForest] = {}
        self.prediction_models: Dict[str, RandomForestRegressor] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        os.makedirs(model_dir, exist_ok=True)
    
    def prepare_features(self, df: pd.DataFrame, sensor_types: List[str]) -> np.ndarray:
        """准备机器学习特征"""
        features = []
        
        for sensor_type in sensor_types:
            sensor_data = df[df['sensor_type'] == sensor_type]
            if len(sensor_data) > 0:
                values = sensor_data['value'].values
                
                # 基本统计特征
                features.extend([
                    np.mean(values),
                    np.std(values),
                    np.min(values),
                    np.max(values),
                    np.median(values)
                ])
                
                # 时间特征
                if len(values) > 1:
                    features.extend([
                        np.mean(np.diff(values)),  # 平均变化率
                        np.std(np.diff(values))    # 变化率标准差
                    ])
                else:
                    features.extend([0, 0])
            else:
                features.extend([0, 0, 0, 0, 0, 0, 0])
        
        return np.array(features).reshape(1, -1)
    
    def train_anomaly_detection(self, device_id: str, sensor_types: List[str], 
                              window_size: int = 50) -> bool:
        """训练异常检测模型"""
        if device_id not in self.data_cache:
            return False
        
        df = self.data_cache[device_id]
        
        # 准备训练数据
        features_list = []
        for i in range(len(df) - window_size + 1):
            window_df = df.iloc[i:i + window_size]
            features = self.prepare_features(window_df, sensor_types)
            features_list.append(features.flatten())
        
        if len(features_list) < 10:
            return False
        
        X = np.array(features_list)
        
        # 训练异常检测模型
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X)
        
        # 保存模型
        model_path = os.path.join(self.model_dir, f"{device_id}_anomaly_model.joblib")
        joblib.dump(model, model_path)
        
        self.anomaly_models[device_id] = model
        return True
    
    def detect_anomalies_ml(self, device_id: str, sensor_types: List[str], 
                          window_size: int = 50) -> List[Dict]:
        """使用机器学习检测异常"""
        if device_id not in self.anomaly_models:
            if not self.train_anomaly_detection(device_id, sensor_types, window_size):
                return []
        
        model = self.anomaly_models[device_id]
        df = self.data_cache[device_id]
        
        anomalies = []
        for i in range(len(df) - window_size + 1):
            window_df = df.iloc[i:i + window_size]
            features = self.prepare_features(window_df, sensor_types)
            
            prediction = model.predict(features)
            if prediction[0] == -1:  # 异常
                anomaly_info = {
                    'window_start': window_df.iloc[0]['timestamp'],
                    'window_end': window_df.iloc[-1]['timestamp'],
                    'anomaly_score': model.score_samples(features)[0],
                    'sensor_types': sensor_types
                }
                anomalies.append(anomaly_info)
        
        return anomalies
    
    def train_prediction_model(self, device_id: str, target_sensor: str, 
                             feature_sensors: List[str], window_size: int = 20) -> bool:
        """训练预测模型"""
        if device_id not in self.data_cache:
            return False
        
        df = self.data_cache[device_id]
        
        # 准备训练数据
        X_list = []
        y_list = []
        
        for i in range(len(df) - window_size):
            # 特征窗口
            feature_window = df.iloc[i:i + window_size]
            features = self.prepare_features(feature_window, feature_sensors)
            X_list.append(features.flatten())
            
            # 目标值（下一个时间点的目标传感器值）
            target_data = df[df['sensor_type'] == target_sensor]
            if i + window_size < len(target_data):
                y_list.append(target_data.iloc[i + window_size]['value'])
            else:
                break
        
        if len(X_list) < 10:
            return False
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 训练预测模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # 保存模型和标准化器
        model_path = os.path.join(self.model_dir, f"{device_id}_{target_sensor}_prediction_model.joblib")
        scaler_path = os.path.join(self.model_dir, f"{device_id}_{target_sensor}_scaler.joblib")
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        self.prediction_models[f"{device_id}_{target_sensor}"] = model
        self.scalers[f"{device_id}_{target_sensor}"] = scaler
        
        return True
    
    def predict_sensor_value(self, device_id: str, target_sensor: str, 
                           feature_sensors: List[str], window_size: int = 20) -> Optional[float]:
        """预测传感器值"""
        model_key = f"{device_id}_{target_sensor}"
        
        if model_key not in self.prediction_models:
            if not self.train_prediction_model(device_id, target_sensor, feature_sensors, window_size):
                return None
        
        model = self.prediction_models[model_key]
        scaler = self.scalers[model_key]
        
        if device_id not in self.data_cache:
            return None
        
        df = self.data_cache[device_id]
        
        # 准备预测特征
        if len(df) < window_size:
            return None
        
        feature_window = df.iloc[-window_size:]
        features = self.prepare_features(feature_window, feature_sensors)
        features_scaled = scaler.transform(features)
        
        # 进行预测
        prediction = model.predict(features_scaled)[0]
        return float(prediction)
```

## 4. 元编程和动态特性

### 4.1 装饰器和元类

```python
import functools
import time
from typing import Callable, Any

# 性能监控装饰器
def performance_monitor(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# 缓存装饰器
def cache_result(max_size: int = 128):
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 创建缓存键
            key = str(args) + str(sorted(kwargs.items()))
            
            if key in cache:
                return cache[key]
            
            result = func(*args, **kwargs)
            
            # 限制缓存大小
            if len(cache) >= max_size:
                # 删除最旧的条目
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            
            cache[key] = result
            return result
        
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: f"Cache size: {len(cache)}"
        return wrapper
    return decorator

# 重试装饰器
def retry_on_exception(max_retries: int = 3, delay: float = 1.0, 
                      exceptions: tuple = (Exception,)):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(delay * (2 ** attempt))  # 指数退避
                    else:
                        raise last_exception
            
            return None
        return wrapper
    return decorator

# 使用装饰器
@performance_monitor
@cache_result(max_size=64)
@retry_on_exception(max_retries=3, delay=0.5)
def process_sensor_data(device_id: str, sensor_type: str) -> Dict[str, Any]:
    """处理传感器数据"""
    # 模拟数据处理
    time.sleep(0.1)
    
    return {
        'device_id': device_id,
        'sensor_type': sensor_type,
        'processed_at': time.time(),
        'status': 'success'
    }

# 元类示例
class IoTDeviceMeta(type):
    def __new__(cls, name, bases, attrs):
        # 自动添加设备验证
        if 'device_id' in attrs:
            original_init = attrs.get('__init__')
            
            def new_init(self, *args, **kwargs):
                if original_init:
                    original_init(self, *args, **kwargs)
                
                # 验证设备ID
                if not hasattr(self, 'device_id') or not self.device_id:
                    raise ValueError("Device ID is required")
                
                # 自动设置创建时间
                if not hasattr(self, 'created_at'):
                    self.created_at = time.time()
            
            attrs['__init__'] = new_init
        
        # 自动添加日志方法
        if 'log' not in attrs:
            def log(self, message: str, level: str = 'INFO'):
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{timestamp}] [{level}] {self.__class__.__name__}: {message}")
            
            attrs['log'] = log
        
        return super().__new__(cls, name, bases, attrs)

# 使用元类
class SmartDevice(metaclass=IoTDeviceMeta):
    def __init__(self, device_id: str, name: str):
        self.device_id = device_id
        self.name = name
    
    def get_status(self) -> str:
        return "online"

# 动态类创建
def create_device_class(device_type: str, sensors: List[str]) -> type:
    """动态创建设备类"""
    
    class DynamicDevice(metaclass=IoTDeviceMeta):
        def __init__(self, device_id: str, name: str):
            self.device_id = device_id
            self.name = name
            self.device_type = device_type
            self.sensors = sensors
            self.sensor_data = {sensor: [] for sensor in sensors}
        
        def add_sensor_data(self, sensor_type: str, value: float):
            if sensor_type in self.sensors:
                self.sensor_data[sensor_type].append({
                    'value': value,
                    'timestamp': time.time()
                })
        
        def get_sensor_data(self, sensor_type: str) -> List[Dict]:
            return self.sensor_data.get(sensor_type, [])
    
    # 设置类名
    DynamicDevice.__name__ = f"{device_type}Device"
    DynamicDevice.__qualname__ = f"{device_type}Device"
    
    return DynamicDevice

# 使用动态类创建
TemperatureDevice = create_device_class("Temperature", ["temperature", "humidity"])
device = TemperatureDevice("temp_001", "Room Temperature Sensor")
device.add_sensor_data("temperature", 25.5)
device.add_sensor_data("humidity", 60.0)
```

### 4.2 反射和动态属性

```python
import inspect
import importlib
from typing import Any, Dict, List

# 动态属性访问
class DynamicIoTDevice:
    def __init__(self, device_id: str, **kwargs):
        self.device_id = device_id
        self._attributes = kwargs
        self._sensors = {}
    
    def __getattr__(self, name: str) -> Any:
        """动态属性获取"""
        if name in self._attributes:
            return self._attributes[name]
        
        # 传感器数据访问
        if name.startswith('sensor_'):
            sensor_type = name[7:]  # 移除 'sensor_' 前缀
            return self._get_sensor_value(sensor_type)
        
        # 方法调用
        if name.startswith('get_'):
            attr_name = name[4:]  # 移除 'get_' 前缀
            return lambda: self._attributes.get(attr_name)
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """动态属性设置"""
        if name.startswith('_') or name in ['device_id']:
            super().__setattr__(name, value)
        else:
            if not hasattr(self, '_attributes'):
                super().__setattr__('_attributes', {})
            self._attributes[name] = value
    
    def _get_sensor_value(self, sensor_type: str) -> float:
        """获取传感器值"""
        if sensor_type in self._sensors:
            return self._sensors[sensor_type]
        
        # 模拟传感器数据
        import random
        value = random.uniform(0, 100)
        self._sensors[sensor_type] = value
        return value
    
    def get_all_attributes(self) -> Dict[str, Any]:
        """获取所有属性"""
        return self._attributes.copy()
    
    def set_sensor_value(self, sensor_type: str, value: float):
        """设置传感器值"""
        self._sensors[sensor_type] = value

# 反射工具类
class IoTReflection:
    @staticmethod
    def get_class_methods(cls: type) -> List[str]:
        """获取类的所有方法"""
        return [name for name, method in inspect.getmembers(cls, inspect.ismethod)]
    
    @staticmethod
    def get_class_attributes(cls: type) -> List[str]:
        """获取类的所有属性"""
        return [name for name, attr in inspect.getmembers(cls, lambda x: not inspect.ismethod(x))]
    
    @staticmethod
    def call_method_dynamically(obj: Any, method_name: str, *args, **kwargs) -> Any:
        """动态调用方法"""
        if hasattr(obj, method_name):
            method = getattr(obj, method_name)
            if callable(method):
                return method(*args, **kwargs)
            else:
                raise AttributeError(f"'{method_name}' is not callable")
        else:
            raise AttributeError(f"'{obj.__class__.__name__}' object has no method '{method_name}'")
    
    @staticmethod
    def create_instance_from_config(config: Dict[str, Any]) -> Any:
        """从配置创建实例"""
        class_name = config.get('class_name')
        module_name = config.get('module_name')
        init_params = config.get('init_params', {})
        
        if not class_name or not module_name:
            raise ValueError("class_name and module_name are required")
        
        # 动态导入模块
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        
        # 创建实例
        return cls(**init_params)
    
    @staticmethod
    def inspect_function(func: Callable) -> Dict[str, Any]:
        """检查函数信息"""
        sig = inspect.signature(func)
        
        return {
            'name': func.__name__,
            'parameters': list(sig.parameters.keys()),
            'return_annotation': sig.return_annotation,
            'docstring': func.__doc__,
            'is_async': inspect.iscoroutinefunction(func),
            'is_generator': inspect.isgeneratorfunction(func)
        }

# 使用反射
def reflection_demo():
    # 创建动态设备
    device = DynamicIoTDevice("device_001", location="room_1", zone="building_a")
    
    # 动态属性访问
    print(f"Location: {device.location}")
    print(f"Zone: {device.zone}")
    print(f"Temperature: {device.sensor_temperature}")
    print(f"Humidity: {device.sensor_humidity}")
    
    # 动态方法调用
    print(f"Get location: {device.get_location()}")
    
    # 反射检查
    reflection = IoTReflection()
    methods = reflection.get_class_methods(DynamicIoTDevice)
    attributes = reflection.get_class_attributes(DynamicIoTDevice)
    
    print(f"Methods: {methods}")
    print(f"Attributes: {attributes}")
    
    # 动态方法调用
    result = reflection.call_method_dynamically(device, 'get_all_attributes')
    print(f"All attributes: {result}")

# 插件系统
class IoTPluginManager:
    def __init__(self):
        self.plugins: Dict[str, Any] = {}
        self.plugin_configs: Dict[str, Dict] = {}
    
    def load_plugin(self, plugin_name: str, config: Dict[str, Any]):
        """加载插件"""
        try:
            # 动态导入插件模块
            module_name = config.get('module_name')
            class_name = config.get('class_name')
            
            if not module_name or not class_name:
                raise ValueError("module_name and class_name are required")
            
            module = importlib.import_module(module_name)
            plugin_class = getattr(module, class_name)
            
            # 创建插件实例
            init_params = config.get('init_params', {})
            plugin_instance = plugin_class(**init_params)
            
            self.plugins[plugin_name] = plugin_instance
            self.plugin_configs[plugin_name] = config
            
            print(f"Plugin '{plugin_name}' loaded successfully")
            
        except Exception as e:
            print(f"Failed to load plugin '{plugin_name}': {e}")
    
    def unload_plugin(self, plugin_name: str):
        """卸载插件"""
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]
            del self.plugin_configs[plugin_name]
            print(f"Plugin '{plugin_name}' unloaded")
    
    def call_plugin_method(self, plugin_name: str, method_name: str, *args, **kwargs):
        """调用插件方法"""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin '{plugin_name}' not found")
        
        plugin = self.plugins[plugin_name]
        return IoTReflection.call_method_dynamically(plugin, method_name, *args, **kwargs)
    
    def list_plugins(self) -> List[str]:
        """列出所有插件"""
        return list(self.plugins.keys())
```

## 5. 总结

Python语言在IoT系统开发中具有显著优势：

1. **动态类型系统**：灵活的类型注解和运行时类型检查
2. **异步编程**：强大的asyncio库支持高并发处理
3. **科学计算**：丰富的NumPy、Pandas、SciPy生态系统
4. **机器学习**：完整的scikit-learn、TensorFlow、PyTorch集成
5. **元编程**：强大的装饰器、元类和反射机制
6. **开发效率**：简洁的语法和丰富的第三方库

通过合理利用Python的这些特性，能够快速构建功能丰富、易于维护的IoT系统。
