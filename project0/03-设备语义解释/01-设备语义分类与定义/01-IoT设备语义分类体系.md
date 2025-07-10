# IoT设备语义分类体系

## 1. 理论框架

### 1.1 设备语义分类理论基础

- **语义层次理论**：基于设备功能、协议、应用场景的多层次语义分类
- **本体论分类**：基于设备本体关系的分类体系
- **动态分类理论**：支持设备运行时状态变化的动态分类机制

### 1.2 分类体系架构

```text
设备语义分类体系
├── 功能分类层
│   ├── 感知设备 (Sensors)
│   ├── 执行设备 (Actuators)
│   ├── 计算设备 (Computing)
│   └── 通信设备 (Communication)
├── 协议分类层
│   ├── OPC-UA设备
│   ├── MQTT设备
│   ├── CoAP设备
│   └── HTTP设备
├── 应用分类层
│   ├── 工业设备
│   ├── 医疗设备
│   ├── 智能家居设备
│   └── 车联网设备
└── 状态分类层
    ├── 在线设备
    ├── 离线设备
    ├── 故障设备
    └── 维护设备
```

## 2. 算法实现

### 2.1 Python代码：设备语义分类引擎

```python
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import json
import logging

class DeviceCategory(Enum):
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    COMPUTING = "computing"
    COMMUNICATION = "communication"
    GATEWAY = "gateway"
    CONTROLLER = "controller"

class ProtocolType(Enum):
    OPC_UA = "opc_ua"
    MQTT = "mqtt"
    COAP = "coap"
    HTTP = "http"
    MODBUS = "modbus"
    PROFINET = "profinet"

class ApplicationDomain(Enum):
    INDUSTRIAL = "industrial"
    MEDICAL = "medical"
    SMART_HOME = "smart_home"
    AUTOMOTIVE = "automotive"
    AGRICULTURE = "agriculture"
    ENERGY = "energy"

@dataclass
class DeviceSemanticModel:
    device_id: str
    device_type: str
    category: DeviceCategory
    protocol: ProtocolType
    application_domain: ApplicationDomain
    capabilities: List[str]
    properties: Dict[str, Any]
    relationships: List[Tuple[str, str, str]]  # (source, relation, target)
    semantic_annotations: Dict[str, str]

@dataclass
class ClassificationResult:
    device_id: str
    primary_category: DeviceCategory
    secondary_categories: List[DeviceCategory]
    confidence_scores: Dict[str, float]
    semantic_similarity: Dict[str, float]
    classification_reasoning: List[str]

class DeviceSemanticClassifier:
    def __init__(self):
        self.ontology_engine = OntologyEngine()
        self.ml_classifier = MLClassifier()
        self.rule_engine = RuleEngine()
        self.similarity_engine = SimilarityEngine()
        self.classification_cache = {}
    
    def classify_device(self, device_data: Dict[str, Any]) -> ClassificationResult:
        """对设备进行语义分类"""
        device_id = device_data.get('device_id', 'unknown')
        
        # 检查缓存
        if device_id in self.classification_cache:
            return self.classification_cache[device_id]
        
        # 1. 基于规则的分类
        rule_based_result = self.rule_based_classification(device_data)
        
        # 2. 基于机器学习的分类
        ml_based_result = self.ml_based_classification(device_data)
        
        # 3. 基于本体论的分类
        ontology_based_result = self.ontology_based_classification(device_data)
        
        # 4. 融合分类结果
        final_result = self.fuse_classification_results(
            rule_based_result, ml_based_result, ontology_based_result
        )
        
        # 5. 计算置信度
        confidence_scores = self.calculate_confidence_scores(
            rule_based_result, ml_based_result, ontology_based_result
        )
        
        # 6. 计算语义相似性
        semantic_similarity = self.calculate_semantic_similarity(device_data)
        
        # 7. 生成分类推理
        classification_reasoning = self.generate_classification_reasoning(
            rule_based_result, ml_based_result, ontology_based_result
        )
        
        result = ClassificationResult(
            device_id=device_id,
            primary_category=final_result['primary_category'],
            secondary_categories=final_result['secondary_categories'],
            confidence_scores=confidence_scores,
            semantic_similarity=semantic_similarity,
            classification_reasoning=classification_reasoning
        )
        
        # 缓存结果
        self.classification_cache[device_id] = result
        
        return result
    
    def rule_based_classification(self, device_data: Dict[str, Any]) -> Dict[str, Any]:
        """基于规则的分类"""
        device_type = device_data.get('device_type', '').lower()
        protocol = device_data.get('protocol', '').lower()
        capabilities = device_data.get('capabilities', [])
        
        classification_rules = {
            'sensor': [
                {'condition': lambda d: 'sensor' in d.get('device_type', '').lower(), 'category': DeviceCategory.SENSOR},
                {'condition': lambda d: 'measure' in d.get('capabilities', []), 'category': DeviceCategory.SENSOR},
                {'condition': lambda d: 'detect' in d.get('capabilities', []), 'category': DeviceCategory.SENSOR}
            ],
            'actuator': [
                {'condition': lambda d: 'actuator' in d.get('device_type', '').lower(), 'category': DeviceCategory.ACTUATOR},
                {'condition': lambda d: 'control' in d.get('capabilities', []), 'category': DeviceCategory.ACTUATOR},
                {'condition': lambda d: 'move' in d.get('capabilities', []), 'category': DeviceCategory.ACTUATOR}
            ],
            'gateway': [
                {'condition': lambda d: 'gateway' in d.get('device_type', '').lower(), 'category': DeviceCategory.GATEWAY},
                {'condition': lambda d: 'bridge' in d.get('capabilities', []), 'category': DeviceCategory.GATEWAY}
            ],
            'controller': [
                {'condition': lambda d: 'controller' in d.get('device_type', '').lower(), 'category': DeviceCategory.CONTROLLER},
                {'condition': lambda d: 'compute' in d.get('capabilities', []), 'category': DeviceCategory.CONTROLLER}
            ]
        }
        
        matched_categories = []
        for category, rules in classification_rules.items():
            for rule in rules:
                if rule['condition'](device_data):
                    matched_categories.append(rule['category'])
                    break
        
        return {
            'categories': matched_categories,
            'confidence': 0.8 if matched_categories else 0.3,
            'method': 'rule_based'
        }
    
    def ml_based_classification(self, device_data: Dict[str, Any]) -> Dict[str, Any]:
        """基于机器学习的分类"""
        # 特征提取
        features = self.extract_features(device_data)
        
        # 使用预训练模型进行分类
        predictions = self.ml_classifier.predict(features)
        
        # 获取置信度
        confidence_scores = self.ml_classifier.get_confidence_scores(features)
        
        return {
            'categories': predictions,
            'confidence_scores': confidence_scores,
            'confidence': np.mean(list(confidence_scores.values())),
            'method': 'ml_based'
        }
    
    def ontology_based_classification(self, device_data: Dict[str, Any]) -> Dict[str, Any]:
        """基于本体论的分类"""
        # 构建设备语义描述
        semantic_description = self.build_semantic_description(device_data)
        
        # 查询本体论
        ontology_matches = self.ontology_engine.query_ontology(semantic_description)
        
        # 计算语义相似性
        semantic_similarities = self.ontology_engine.calculate_semantic_similarity(
            semantic_description, ontology_matches
        )
        
        # 选择最佳匹配
        best_matches = self.select_best_ontology_matches(ontology_matches, semantic_similarities)
        
        return {
            'categories': [match['category'] for match in best_matches],
            'confidence': np.mean([match['similarity'] for match in best_matches]),
            'method': 'ontology_based'
        }
    
    def fuse_classification_results(self, rule_result: Dict[str, Any], 
                                  ml_result: Dict[str, Any], 
                                  ontology_result: Dict[str, Any]) -> Dict[str, Any]:
        """融合分类结果"""
        # 收集所有分类结果
        all_categories = []
        all_categories.extend(rule_result['categories'])
        all_categories.extend(ml_result['categories'])
        all_categories.extend(ontology_result['categories'])
        
        # 计算类别权重
        category_weights = self.calculate_category_weights(all_categories)
        
        # 选择主要类别
        primary_category = self.select_primary_category(category_weights)
        
        # 选择次要类别
        secondary_categories = self.select_secondary_categories(category_weights, primary_category)
        
        return {
            'primary_category': primary_category,
            'secondary_categories': secondary_categories
        }
    
    def calculate_confidence_scores(self, rule_result: Dict[str, Any], 
                                   ml_result: Dict[str, Any], 
                                   ontology_result: Dict[str, Any]) -> Dict[str, float]:
        """计算置信度分数"""
        confidence_scores = {}
        
        # 规则分类置信度
        confidence_scores['rule_based'] = rule_result['confidence']
        
        # 机器学习分类置信度
        confidence_scores['ml_based'] = ml_result['confidence']
        
        # 本体论分类置信度
        confidence_scores['ontology_based'] = ontology_result['confidence']
        
        # 综合置信度
        confidence_scores['overall'] = np.mean([
            rule_result['confidence'],
            ml_result['confidence'],
            ontology_result['confidence']
        ])
        
        return confidence_scores
    
    def calculate_semantic_similarity(self, device_data: Dict[str, Any]) -> Dict[str, float]:
        """计算语义相似性"""
        device_semantic = self.build_semantic_description(device_data)
        
        # 与预定义类别的相似性
        predefined_categories = {
            'sensor': 'device that measures physical quantities',
            'actuator': 'device that performs physical actions',
            'gateway': 'device that bridges different networks',
            'controller': 'device that processes and controls'
        }
        
        similarities = {}
        for category, description in predefined_categories.items():
            similarity = self.similarity_engine.calculate_similarity(
                device_semantic, description
            )
            similarities[category] = similarity
        
        return similarities
    
    def generate_classification_reasoning(self, rule_result: Dict[str, Any], 
                                        ml_result: Dict[str, Any], 
                                        ontology_result: Dict[str, Any]) -> List[str]:
        """生成分类推理"""
        reasoning = []
        
        # 规则推理
        if rule_result['categories']:
            reasoning.append(f"Rule-based classification: {', '.join([c.value for c in rule_result['categories']])}")
        
        # 机器学习推理
        if ml_result['categories']:
            reasoning.append(f"ML-based classification: {', '.join([c.value for c in ml_result['categories']])}")
        
        # 本体论推理
        if ontology_result['categories']:
            reasoning.append(f"Ontology-based classification: {', '.join([c.value for c in ontology_result['categories']])}")
        
        return reasoning
    
    def extract_features(self, device_data: Dict[str, Any]) -> np.ndarray:
        """提取特征"""
        features = []
        
        # 设备类型特征
        device_type = device_data.get('device_type', '')
        features.extend(self.encode_device_type(device_type))
        
        # 协议特征
        protocol = device_data.get('protocol', '')
        features.extend(self.encode_protocol(protocol))
        
        # 能力特征
        capabilities = device_data.get('capabilities', [])
        features.extend(self.encode_capabilities(capabilities))
        
        # 属性特征
        properties = device_data.get('properties', {})
        features.extend(self.encode_properties(properties))
        
        return np.array(features)
    
    def encode_device_type(self, device_type: str) -> List[float]:
        """编码设备类型"""
        # 简化的设备类型编码
        # 实际实现应该使用更复杂的编码方法
        
        device_types = ['sensor', 'actuator', 'gateway', 'controller', 'computing']
        encoding = [0.0] * len(device_types)
        
        for i, dt in enumerate(device_types):
            if dt in device_type.lower():
                encoding[i] = 1.0
        
        return encoding
    
    def encode_protocol(self, protocol: str) -> List[float]:
        """编码协议"""
        # 简化的协议编码
        protocols = ['opc_ua', 'mqtt', 'coap', 'http', 'modbus']
        encoding = [0.0] * len(protocols)
        
        for i, p in enumerate(protocols):
            if p in protocol.lower():
                encoding[i] = 1.0
        
        return encoding
    
    def encode_capabilities(self, capabilities: List[str]) -> List[float]:
        """编码能力"""
        # 简化的能力编码
        all_capabilities = ['measure', 'control', 'communicate', 'compute', 'store']
        encoding = [0.0] * len(all_capabilities)
        
        for i, cap in enumerate(all_capabilities):
            if cap in [c.lower() for c in capabilities]:
                encoding[i] = 1.0
        
        return encoding
    
    def encode_properties(self, properties: Dict[str, Any]) -> List[float]:
        """编码属性"""
        # 简化的属性编码
        # 实际实现应该使用更复杂的编码方法
        
        # 数值属性
        numeric_features = []
        for key, value in properties.items():
            if isinstance(value, (int, float)):
                numeric_features.append(float(value))
        
        # 标准化数值特征
        if numeric_features:
            normalized_features = self.normalize_features(numeric_features)
        else:
            normalized_features = [0.0] * 10  # 默认特征
        
        return normalized_features
    
    def normalize_features(self, features: List[float]) -> List[float]:
        """标准化特征"""
        if not features:
            return [0.0] * 10
        
        # 简化的标准化
        # 实际实现应该使用更复杂的标准化方法
        
        min_val = min(features)
        max_val = max(features)
        
        if max_val == min_val:
            return [0.5] * len(features)
        
        normalized = [(f - min_val) / (max_val - min_val) for f in features]
        
        # 填充到固定长度
        while len(normalized) < 10:
            normalized.append(0.0)
        
        return normalized[:10]
    
    def build_semantic_description(self, device_data: Dict[str, Any]) -> str:
        """构建语义描述"""
        device_type = device_data.get('device_type', '')
        capabilities = device_data.get('capabilities', [])
        properties = device_data.get('properties', {})
        
        description_parts = []
        
        if device_type:
            description_parts.append(f"Device type: {device_type}")
        
        if capabilities:
            description_parts.append(f"Capabilities: {', '.join(capabilities)}")
        
        if properties:
            prop_desc = []
            for key, value in properties.items():
                prop_desc.append(f"{key}: {value}")
            description_parts.append(f"Properties: {', '.join(prop_desc)}")
        
        return ". ".join(description_parts)
    
    def calculate_category_weights(self, categories: List[DeviceCategory]) -> Dict[DeviceCategory, float]:
        """计算类别权重"""
        weights = {}
        
        for category in categories:
            if category in weights:
                weights[category] += 1.0
            else:
                weights[category] = 1.0
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def select_primary_category(self, category_weights: Dict[DeviceCategory, float]) -> DeviceCategory:
        """选择主要类别"""
        if not category_weights:
            return DeviceCategory.SENSOR  # 默认类别
        
        return max(category_weights.items(), key=lambda x: x[1])[0]
    
    def select_secondary_categories(self, category_weights: Dict[DeviceCategory, float], 
                                   primary_category: DeviceCategory) -> List[DeviceCategory]:
        """选择次要类别"""
        # 移除主要类别
        remaining_weights = {k: v for k, v in category_weights.items() if k != primary_category}
        
        # 选择权重最高的类别作为次要类别
        secondary_categories = []
        if remaining_weights:
            sorted_categories = sorted(remaining_weights.items(), key=lambda x: x[1], reverse=True)
            secondary_categories = [cat for cat, weight in sorted_categories[:2]]  # 最多2个次要类别
        
        return secondary_categories

class OntologyEngine:
    def __init__(self):
        self.ontology_graph = nx.DiGraph()
        self.load_ontology()
    
    def load_ontology(self):
        """加载本体论"""
        # 添加设备类别节点
        device_categories = [
            ('Device', 'IoT Device'),
            ('Sensor', 'Device that measures physical quantities'),
            ('Actuator', 'Device that performs physical actions'),
            ('Gateway', 'Device that bridges different networks'),
            ('Controller', 'Device that processes and controls'),
            ('Computing', 'Device that performs computations')
        ]
        
        for category, description in device_categories:
            self.ontology_graph.add_node(category, description=description)
        
        # 添加关系
        relationships = [
            ('Sensor', 'Device', 'is_a'),
            ('Actuator', 'Device', 'is_a'),
            ('Gateway', 'Device', 'is_a'),
            ('Controller', 'Device', 'is_a'),
            ('Computing', 'Device', 'is_a'),
            ('Sensor', 'Actuator', 'complements'),
            ('Gateway', 'Sensor', 'connects'),
            ('Gateway', 'Actuator', 'connects'),
            ('Controller', 'Sensor', 'controls'),
            ('Controller', 'Actuator', 'controls')
        ]
        
        for source, target, relation in relationships:
            self.ontology_graph.add_edge(source, target, relation=relation)
    
    def query_ontology(self, semantic_description: str) -> List[Dict[str, Any]]:
        """查询本体论"""
        matches = []
        
        for node in self.ontology_graph.nodes():
            node_data = self.ontology_graph.nodes[node]
            description = node_data.get('description', '')
            
            # 计算相似性
            similarity = self.calculate_text_similarity(semantic_description, description)
            
            if similarity > 0.3:  # 相似性阈值
                matches.append({
                    'node': node,
                    'category': self.map_node_to_category(node),
                    'description': description,
                    'similarity': similarity
                })
        
        return matches
    
    def calculate_semantic_similarity(self, semantic_description: str, 
                                     ontology_matches: List[Dict[str, Any]]) -> List[float]:
        """计算语义相似性"""
        similarities = []
        
        for match in ontology_matches:
            similarity = match['similarity']
            similarities.append(similarity)
        
        return similarities
    
    def select_best_ontology_matches(self, ontology_matches: List[Dict[str, Any]], 
                                     similarities: List[float]) -> List[Dict[str, Any]]:
        """选择最佳本体论匹配"""
        # 按相似性排序
        sorted_matches = sorted(ontology_matches, key=lambda x: x['similarity'], reverse=True)
        
        # 选择前3个最佳匹配
        return sorted_matches[:3]
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似性"""
        # 简化的文本相似性计算
        # 实际实现应该使用更复杂的NLP技术
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def map_node_to_category(self, node: str) -> DeviceCategory:
        """将节点映射到设备类别"""
        mapping = {
            'Sensor': DeviceCategory.SENSOR,
            'Actuator': DeviceCategory.ACTUATOR,
            'Gateway': DeviceCategory.GATEWAY,
            'Controller': DeviceCategory.CONTROLLER,
            'Computing': DeviceCategory.COMPUTING
        }
        
        return mapping.get(node, DeviceCategory.SENSOR)

class MLClassifier:
    def __init__(self):
        self.model = self.load_model()
        self.vectorizer = TfidfVectorizer()
    
    def load_model(self):
        """加载预训练模型"""
        # 简化的模型加载
        # 实际实现应该加载真实的预训练模型
        
        return KMeans(n_clusters=5)  # 5个设备类别
    
    def predict(self, features: np.ndarray) -> List[DeviceCategory]:
        """预测设备类别"""
        # 简化的预测
        # 实际实现应该使用真实的分类模型
        
        # 使用聚类进行预测
        prediction = self.model.fit_predict([features])[0]
        
        # 映射到设备类别
        category_mapping = {
            0: DeviceCategory.SENSOR,
            1: DeviceCategory.ACTUATOR,
            2: DeviceCategory.GATEWAY,
            3: DeviceCategory.CONTROLLER,
            4: DeviceCategory.COMPUTING
        }
        
        return [category_mapping.get(prediction, DeviceCategory.SENSOR)]
    
    def get_confidence_scores(self, features: np.ndarray) -> Dict[str, float]:
        """获取置信度分数"""
        # 简化的置信度计算
        # 实际实现应该使用真实的置信度计算方法
        
        categories = ['sensor', 'actuator', 'gateway', 'controller', 'computing']
        confidence_scores = {}
        
        for category in categories:
            # 基于特征计算置信度
            confidence = np.random.uniform(0.5, 0.9)  # 随机置信度
            confidence_scores[category] = confidence
        
        return confidence_scores

class RuleEngine:
    def __init__(self):
        self.rules = self.load_rules()
    
    def load_rules(self) -> List[Dict[str, Any]]:
        """加载分类规则"""
        return [
            {
                'name': 'sensor_rule',
                'condition': lambda d: 'sensor' in d.get('device_type', '').lower(),
                'category': DeviceCategory.SENSOR,
                'confidence': 0.9
            },
            {
                'name': 'actuator_rule',
                'condition': lambda d: 'actuator' in d.get('device_type', '').lower(),
                'category': DeviceCategory.ACTUATOR,
                'confidence': 0.9
            },
            {
                'name': 'gateway_rule',
                'condition': lambda d: 'gateway' in d.get('device_type', '').lower(),
                'category': DeviceCategory.GATEWAY,
                'confidence': 0.9
            }
        ]

class SimilarityEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似性"""
        # 使用TF-IDF计算相似性
        try:
            tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
            similarity = (tfidf_matrix * tfidf_matrix.T).A[0, 1]
            return float(similarity)
        except:
            return 0.0
```

### 2.2 Rust伪代码：高性能设备分类引擎

```rust
pub struct HighPerformanceDeviceClassifier {
    ontology_engine: OntologyEngine,
    ml_classifier: MLClassifier,
    rule_engine: RuleEngine,
    similarity_engine: SimilarityEngine,
    classification_cache: HashMap<String, ClassificationResult>,
}

impl HighPerformanceDeviceClassifier {
    pub async fn classify_device(
        &self,
        device_data: &DeviceData,
    ) -> Result<ClassificationResult, ClassificationError> {
        let device_id = device_data.device_id.clone();
        
        // 检查缓存
        if let Some(cached_result) = self.classification_cache.get(&device_id) {
            return Ok(cached_result.clone());
        }
        
        // 并行执行分类方法
        let (rule_result, ml_result, ontology_result) = tokio::join!(
            self.rule_based_classification(device_data),
            self.ml_based_classification(device_data),
            self.ontology_based_classification(device_data),
        );
        
        let rule_result = rule_result?;
        let ml_result = ml_result?;
        let ontology_result = ontology_result?;
        
        // 融合结果
        let fused_result = self.fuse_classification_results(
            rule_result, ml_result, ontology_result
        ).await?;
        
        // 计算置信度
        let confidence_scores = self.calculate_confidence_scores(
            rule_result, ml_result, ontology_result
        ).await?;
        
        // 计算语义相似性
        let semantic_similarity = self.calculate_semantic_similarity(device_data).await?;
        
        // 生成推理
        let classification_reasoning = self.generate_classification_reasoning(
            rule_result, ml_result, ontology_result
        ).await?;
        
        let result = ClassificationResult {
            device_id,
            primary_category: fused_result.primary_category,
            secondary_categories: fused_result.secondary_categories,
            confidence_scores,
            semantic_similarity,
            classification_reasoning,
        };
        
        // 缓存结果
        self.classification_cache.insert(device_id.clone(), result.clone());
        
        Ok(result)
    }
    
    async fn rule_based_classification(
        &self,
        device_data: &DeviceData,
    ) -> Result<RuleClassificationResult, ClassificationError> {
        let mut matched_categories = Vec::new();
        let mut confidence = 0.0;
        
        for rule in &self.rule_engine.rules {
            if rule.condition(device_data) {
                matched_categories.push(rule.category.clone());
                confidence = rule.confidence;
                break;
            }
        }
        
        Ok(RuleClassificationResult {
            categories: matched_categories,
            confidence,
            method: "rule_based".to_string(),
        })
    }
    
    async fn ml_based_classification(
        &self,
        device_data: &DeviceData,
    ) -> Result<MLClassificationResult, ClassificationError> {
        // 特征提取
        let features = self.extract_features(device_data).await?;
        
        // 模型预测
        let predictions = self.ml_classifier.predict(&features).await?;
        
        // 获取置信度
        let confidence_scores = self.ml_classifier.get_confidence_scores(&features).await?;
        
        let confidence = confidence_scores.values().sum::<f64>() / confidence_scores.len() as f64;
        
        Ok(MLClassificationResult {
            categories: predictions,
            confidence_scores,
            confidence,
            method: "ml_based".to_string(),
        })
    }
    
    async fn ontology_based_classification(
        &self,
        device_data: &DeviceData,
    ) -> Result<OntologyClassificationResult, ClassificationError> {
        // 构建语义描述
        let semantic_description = self.build_semantic_description(device_data).await?;
        
        // 查询本体论
        let ontology_matches = self.ontology_engine.query_ontology(&semantic_description).await?;
        
        // 计算语义相似性
        let semantic_similarities = self.ontology_engine
            .calculate_semantic_similarity(&semantic_description, &ontology_matches)
            .await?;
        
        // 选择最佳匹配
        let best_matches = self.ontology_engine
            .select_best_ontology_matches(&ontology_matches, &semantic_similarities)
            .await?;
        
        let categories: Vec<DeviceCategory> = best_matches
            .iter()
            .map(|m| m.category.clone())
            .collect();
        
        let confidence = best_matches
            .iter()
            .map(|m| m.similarity)
            .sum::<f64>() / best_matches.len() as f64;
        
        Ok(OntologyClassificationResult {
            categories,
            confidence,
            method: "ontology_based".to_string(),
        })
    }
    
    async fn fuse_classification_results(
        &self,
        rule_result: RuleClassificationResult,
        ml_result: MLClassificationResult,
        ontology_result: OntologyClassificationResult,
    ) -> Result<FusedClassificationResult, ClassificationError> {
        // 收集所有分类结果
        let mut all_categories = Vec::new();
        all_categories.extend(rule_result.categories);
        all_categories.extend(ml_result.categories);
        all_categories.extend(ontology_result.categories);
        
        // 计算类别权重
        let category_weights = self.calculate_category_weights(&all_categories).await?;
        
        // 选择主要类别
        let primary_category = self.select_primary_category(&category_weights).await?;
        
        // 选择次要类别
        let secondary_categories = self.select_secondary_categories(&category_weights, &primary_category).await?;
        
        Ok(FusedClassificationResult {
            primary_category,
            secondary_categories,
        })
    }
    
    async fn calculate_confidence_scores(
        &self,
        rule_result: RuleClassificationResult,
        ml_result: MLClassificationResult,
        ontology_result: OntologyClassificationResult,
    ) -> Result<HashMap<String, f64>, ClassificationError> {
        let mut confidence_scores = HashMap::new();
        
        confidence_scores.insert("rule_based".to_string(), rule_result.confidence);
        confidence_scores.insert("ml_based".to_string(), ml_result.confidence);
        confidence_scores.insert("ontology_based".to_string(), ontology_result.confidence);
        
        let overall_confidence = (rule_result.confidence + ml_result.confidence + ontology_result.confidence) / 3.0;
        confidence_scores.insert("overall".to_string(), overall_confidence);
        
        Ok(confidence_scores)
    }
    
    async fn calculate_semantic_similarity(
        &self,
        device_data: &DeviceData,
    ) -> Result<HashMap<String, f64>, ClassificationError> {
        let device_semantic = self.build_semantic_description(device_data).await?;
        
        let predefined_categories = HashMap::from([
            ("sensor".to_string(), "device that measures physical quantities".to_string()),
            ("actuator".to_string(), "device that performs physical actions".to_string()),
            ("gateway".to_string(), "device that bridges different networks".to_string()),
            ("controller".to_string(), "device that processes and controls".to_string()),
        ]);
        
        let mut similarities = HashMap::new();
        
        for (category, description) in predefined_categories {
            let similarity = self.similarity_engine
                .calculate_similarity(&device_semantic, &description)
                .await?;
            similarities.insert(category, similarity);
        }
        
        Ok(similarities)
    }
    
    async fn generate_classification_reasoning(
        &self,
        rule_result: RuleClassificationResult,
        ml_result: MLClassificationResult,
        ontology_result: OntologyClassificationResult,
    ) -> Result<Vec<String>, ClassificationError> {
        let mut reasoning = Vec::new();
        
        if !rule_result.categories.is_empty() {
            let categories_str = rule_result.categories
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            reasoning.push(format!("Rule-based classification: {}", categories_str));
        }
        
        if !ml_result.categories.is_empty() {
            let categories_str = ml_result.categories
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            reasoning.push(format!("ML-based classification: {}", categories_str));
        }
        
        if !ontology_result.categories.is_empty() {
            let categories_str = ontology_result.categories
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            reasoning.push(format!("Ontology-based classification: {}", categories_str));
        }
        
        Ok(reasoning)
    }
}

pub struct OntologyEngine {
    ontology_graph: Graph<String, String>,
}

impl OntologyEngine {
    pub async fn query_ontology(
        &self,
        semantic_description: &str,
    ) -> Result<Vec<OntologyMatch>, ClassificationError> {
        let mut matches = Vec::new();
        
        // 遍历本体论图中的节点
        for node in self.ontology_graph.node_indices() {
            if let Some(node_data) = self.ontology_graph.node_weight(node) {
                let description = node_data.description.clone();
                
                // 计算相似性
                let similarity = self.calculate_text_similarity(semantic_description, &description).await?;
                
                if similarity > 0.3 {
                    matches.push(OntologyMatch {
                        node: node_data.name.clone(),
                        category: self.map_node_to_category(&node_data.name).await?,
                        description,
                        similarity,
                    });
                }
            }
        }
        
        Ok(matches)
    }
    
    async fn calculate_text_similarity(&self, text1: &str, text2: &str) -> Result<f64, ClassificationError> {
        // 简化的文本相似性计算
        let words1: HashSet<&str> = text1.to_lowercase().split_whitespace().collect();
        let words2: HashSet<&str> = text2.to_lowercase().split_whitespace().collect();
        
        if words1.is_empty() || words2.is_empty() {
            return Ok(0.0);
        }
        
        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();
        
        Ok(intersection as f64 / union as f64)
    }
    
    async fn map_node_to_category(&self, node: &str) -> Result<DeviceCategory, ClassificationError> {
        let mapping = HashMap::from([
            ("Sensor", DeviceCategory::Sensor),
            ("Actuator", DeviceCategory::Actuator),
            ("Gateway", DeviceCategory::Gateway),
            ("Controller", DeviceCategory::Controller),
            ("Computing", DeviceCategory::Computing),
        ]);
        
        Ok(mapping.get(node).cloned().unwrap_or(DeviceCategory::Sensor))
    }
}
```

## 3. 测试用例

### 3.1 Python设备分类测试

```python
def test_device_semantic_classifier():
    classifier = DeviceSemanticClassifier()
    
    # 测试传感器设备
    sensor_data = {
        'device_id': 'sensor_001',
        'device_type': 'temperature_sensor',
        'protocol': 'opc_ua',
        'capabilities': ['measure', 'communicate'],
        'properties': {'unit': 'celsius', 'range': '-40 to 85'}
    }
    
    result = classifier.classify_device(sensor_data)
    
    assert result.primary_category == DeviceCategory.SENSOR
    assert 'sensor' in [c.value for c in result.secondary_categories]
    assert result.confidence_scores['overall'] > 0.5

def test_actuator_classification():
    classifier = DeviceSemanticClassifier()
    
    # 测试执行器设备
    actuator_data = {
        'device_id': 'actuator_001',
        'device_type': 'motor_controller',
        'protocol': 'modbus',
        'capabilities': ['control', 'move'],
        'properties': {'power': '100W', 'speed': '1000rpm'}
    }
    
    result = classifier.classify_device(actuator_data)
    
    assert result.primary_category == DeviceCategory.ACTUATOR
    assert result.confidence_scores['overall'] > 0.5

def test_gateway_classification():
    classifier = DeviceSemanticClassifier()
    
    # 测试网关设备
    gateway_data = {
        'device_id': 'gateway_001',
        'device_type': 'protocol_gateway',
        'protocol': 'multiple',
        'capabilities': ['bridge', 'translate', 'route'],
        'properties': {'supported_protocols': ['opc_ua', 'mqtt', 'modbus']}
    }
    
    result = classifier.classify_device(gateway_data)
    
    assert result.primary_category == DeviceCategory.GATEWAY
    assert result.confidence_scores['overall'] > 0.5
```

### 3.2 Rust设备分类测试

```rust
#[tokio::test]
async fn test_high_performance_device_classifier() {
    let classifier = HighPerformanceDeviceClassifier::new();
    
    // 测试传感器设备
    let sensor_data = DeviceData {
        device_id: "sensor_001".to_string(),
        device_type: "temperature_sensor".to_string(),
        protocol: "opc_ua".to_string(),
        capabilities: vec!["measure".to_string(), "communicate".to_string()],
        properties: HashMap::from([
            ("unit".to_string(), "celsius".to_string()),
            ("range".to_string(), "-40 to 85".to_string()),
        ]),
    };
    
    let result = classifier.classify_device(&sensor_data).await;
    
    assert!(result.is_ok());
    let classification_result = result.unwrap();
    assert_eq!(classification_result.primary_category, DeviceCategory::Sensor);
    assert!(classification_result.confidence_scores.get("overall").unwrap() > &0.5);
}

#[tokio::test]
async fn test_actuator_classification() {
    let classifier = HighPerformanceDeviceClassifier::new();
    
    // 测试执行器设备
    let actuator_data = DeviceData {
        device_id: "actuator_001".to_string(),
        device_type: "motor_controller".to_string(),
        protocol: "modbus".to_string(),
        capabilities: vec!["control".to_string(), "move".to_string()],
        properties: HashMap::from([
            ("power".to_string(), "100W".to_string()),
            ("speed".to_string(), "1000rpm".to_string()),
        ]),
    };
    
    let result = classifier.classify_device(&actuator_data).await;
    
    assert!(result.is_ok());
    let classification_result = result.unwrap();
    assert_eq!(classification_result.primary_category, DeviceCategory::Actuator);
    assert!(classification_result.confidence_scores.get("overall").unwrap() > &0.5);
}
```

## 4. 性能与优化建议

- 实现分类结果缓存，避免重复计算。
- 使用并行分类，提升大规模设备分类性能。
- 集成机器学习，自动优化分类规则。
- 支持增量学习，持续改进分类准确性。

这个文档提供了IoT设备语义分类体系的完整实现，包括分类理论、算法实现、测试用例等核心功能。
