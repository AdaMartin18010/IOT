# IoT威胁建模框架

## 文档概述

本文档建立IoT威胁建模的理论框架，分析威胁识别、评估和缓解策略。

## 一、威胁建模基础

### 1.1 威胁建模原则

```rust
#[derive(Debug, Clone)]
pub struct ThreatModelingPrinciples {
    pub systematic_approach: bool,
    pub asset_centric: bool,
    pub attacker_perspective: bool,
    pub continuous_process: bool,
    pub risk_based: bool,
    pub defense_in_depth: bool,
}

#[derive(Debug, Clone)]
pub struct ThreatModel {
    pub model_id: String,
    pub system_boundary: SystemBoundary,
    pub assets: Vec<Asset>,
    pub threats: Vec<Threat>,
    pub vulnerabilities: Vec<Vulnerability>,
    pub controls: Vec<Control>,
    pub risk_assessment: RiskAssessment,
}

#[derive(Debug, Clone)]
pub struct SystemBoundary {
    pub boundary_id: String,
    pub description: String,
    pub components: Vec<Component>,
    pub interfaces: Vec<Interface>,
    pub trust_boundaries: Vec<TrustBoundary>,
}

#[derive(Debug, Clone)]
pub struct Component {
    pub component_id: String,
    pub name: String,
    pub component_type: ComponentType,
    pub responsibilities: Vec<String>,
    pub dependencies: Vec<String>,
    pub security_requirements: Vec<SecurityRequirement>,
}

#[derive(Debug, Clone)]
pub enum ComponentType {
    Device,
    Gateway,
    Cloud,
    Network,
    Application,
    Database,
    Service,
}
```

### 1.2 IoT威胁特性

```rust
#[derive(Debug, Clone)]
pub struct IoTThreatCharacteristics {
    pub device_heterogeneity: DeviceHeterogeneity,
    pub resource_constraints: ResourceConstraints,
    pub physical_access: PhysicalAccess,
    pub network_exposure: NetworkExposure,
    pub data_sensitivity: DataSensitivity,
    pub operational_impact: OperationalImpact,
}

#[derive(Debug, Clone)]
pub struct DeviceHeterogeneity {
    pub device_types: Vec<DeviceType>,
    pub protocols: Vec<Protocol>,
    pub manufacturers: Vec<String>,
    pub firmware_versions: Vec<String>,
    pub security_capabilities: Vec<SecurityCapability>,
}

#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    pub processing_limitations: ProcessingLimitation,
    pub memory_limitations: MemoryLimitation,
    pub power_limitations: PowerLimitation,
    pub network_limitations: NetworkLimitation,
}

#[derive(Debug, Clone)]
pub struct PhysicalAccess {
    pub access_level: AccessLevel,
    pub tamper_resistance: TamperResistance,
    pub environmental_factors: Vec<EnvironmentalFactor>,
    pub physical_security: PhysicalSecurity,
}

#[derive(Debug, Clone)]
pub enum AccessLevel {
    Public,
    Restricted,
    Controlled,
    Secure,
}

#[derive(Debug, Clone)]
pub enum TamperResistance {
    None,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub struct NetworkExposure {
    pub network_type: NetworkType,
    pub connectivity: Connectivity,
    pub protocols: Vec<Protocol>,
    pub encryption: EncryptionLevel,
    pub authentication: AuthenticationLevel,
}

#[derive(Debug, Clone)]
pub enum NetworkType {
    Local,
    Private,
    Public,
    Internet,
}

#[derive(Debug, Clone)]
pub enum Connectivity {
    Wired,
    Wireless,
    Cellular,
    Satellite,
    Mesh,
}
```

## 二、威胁识别框架

### 2.1 威胁分类体系

```rust
pub struct ThreatClassificationSystem {
    pub threat_categories: Vec<ThreatCategory>,
    pub threat_taxonomy: ThreatTaxonomy,
    pub classification_rules: Vec<ClassificationRule>,
}

impl ThreatClassificationSystem {
    pub fn classify_threat(&self, threat: &Threat) -> ThreatClassification {
        let mut classification = ThreatClassification {
            threat: threat.clone(),
            categories: Vec::new(),
            severity: ThreatSeverity::Unknown,
            likelihood: ThreatLikelihood::Unknown,
            impact: ThreatImpact::Unknown,
        };
        
        for category in &self.threat_categories {
            if self.matches_category(threat, category) {
                classification.categories.push(category.clone());
            }
        }
        
        classification.severity = self.assess_severity(threat);
        classification.likelihood = self.assess_likelihood(threat);
        classification.impact = self.assess_impact(threat);
        
        classification
    }
    
    fn matches_category(&self, threat: &Threat, category: &ThreatCategory) -> bool {
        category.classification_rules.iter().any(|rule| {
            rule.matches(threat)
        })
    }
    
    fn assess_severity(&self, threat: &Threat) -> ThreatSeverity {
        let impact_score = self.calculate_impact_score(threat);
        let likelihood_score = self.calculate_likelihood_score(threat);
        let severity_score = impact_score * likelihood_score;
        
        if severity_score >= 0.8 {
            ThreatSeverity::Critical
        } else if severity_score >= 0.6 {
            ThreatSeverity::High
        } else if severity_score >= 0.4 {
            ThreatSeverity::Medium
        } else if severity_score >= 0.2 {
            ThreatSeverity::Low
        } else {
            ThreatSeverity::Minimal
        }
    }
    
    fn assess_likelihood(&self, threat: &Threat) -> ThreatLikelihood {
        let attacker_capability = self.assess_attacker_capability(threat);
        let attack_surface = self.assess_attack_surface(threat);
        let existing_controls = self.assess_existing_controls(threat);
        
        let likelihood_score = (attacker_capability + attack_surface - existing_controls) / 3.0;
        
        if likelihood_score >= 0.8 {
            ThreatLikelihood::VeryHigh
        } else if likelihood_score >= 0.6 {
            ThreatLikelihood::High
        } else if likelihood_score >= 0.4 {
            ThreatLikelihood::Medium
        } else if likelihood_score >= 0.2 {
            ThreatLikelihood::Low
        } else {
            ThreatLikelihood::VeryLow
        }
    }
    
    fn assess_impact(&self, threat: &Threat) -> ThreatImpact {
        let confidentiality_impact = self.assess_confidentiality_impact(threat);
        let integrity_impact = self.assess_integrity_impact(threat);
        let availability_impact = self.assess_availability_impact(threat);
        let safety_impact = self.assess_safety_impact(threat);
        
        let impact_score = (confidentiality_impact + integrity_impact + availability_impact + safety_impact) / 4.0;
        
        if impact_score >= 0.8 {
            ThreatImpact::Critical
        } else if impact_score >= 0.6 {
            ThreatImpact::High
        } else if impact_score >= 0.4 {
            ThreatImpact::Medium
        } else if impact_score >= 0.2 {
            ThreatImpact::Low
        } else {
            ThreatImpact::Minimal
        }
    }
}

#[derive(Debug, Clone)]
pub struct ThreatCategory {
    pub category_id: String,
    pub name: String,
    pub description: String,
    pub classification_rules: Vec<ClassificationRule>,
    pub examples: Vec<ThreatExample>,
}

#[derive(Debug, Clone)]
pub struct ClassificationRule {
    pub rule_id: String,
    pub rule_type: RuleType,
    pub conditions: Vec<RuleCondition>,
    pub weight: f64,
}

#[derive(Debug, Clone)]
pub enum RuleType {
    AttackVector,
    AttackTarget,
    AttackMethod,
    AttackMotivation,
    AttackImpact,
}

#[derive(Debug, Clone)]
pub struct RuleCondition {
    pub condition_type: ConditionType,
    pub operator: ConditionOperator,
    pub value: String,
}

#[derive(Debug, Clone)]
pub enum ConditionType {
    ThreatType,
    TargetType,
    AttackVector,
    Motivation,
    Impact,
}

#[derive(Debug, Clone)]
pub enum ConditionOperator {
    Equals,
    NotEquals,
    Contains,
    NotContains,
    GreaterThan,
    LessThan,
}

#[derive(Debug, Clone)]
pub struct ThreatClassification {
    pub threat: Threat,
    pub categories: Vec<ThreatCategory>,
    pub severity: ThreatSeverity,
    pub likelihood: ThreatLikelihood,
    pub impact: ThreatImpact,
}

#[derive(Debug, Clone)]
pub enum ThreatSeverity {
    Minimal,
    Low,
    Medium,
    High,
    Critical,
    Unknown,
}

#[derive(Debug, Clone)]
pub enum ThreatLikelihood {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
    Unknown,
}

#[derive(Debug, Clone)]
pub enum ThreatImpact {
    Minimal,
    Low,
    Medium,
    High,
    Critical,
    Unknown,
}
```

### 2.2 威胁识别方法

```rust
pub struct ThreatIdentificationEngine {
    pub identification_methods: Vec<IdentificationMethod>,
    pub threat_patterns: Vec<ThreatPattern>,
    pub attack_trees: Vec<AttackTree>,
}

impl ThreatIdentificationEngine {
    pub fn identify_threats(&self, system: &SystemBoundary) -> Vec<Threat> {
        let mut threats = Vec::new();
        
        // 使用STRIDE方法识别威胁
        let stride_threats = self.stride_analysis(system);
        threats.extend(stride_threats);
        
        // 使用攻击树分析识别威胁
        let attack_tree_threats = self.attack_tree_analysis(system);
        threats.extend(attack_tree_threats);
        
        // 使用威胁模式识别威胁
        let pattern_threats = self.pattern_based_analysis(system);
        threats.extend(pattern_threats);
        
        // 去重和合并
        self.deduplicate_threats(threats)
    }
    
    fn stride_analysis(&self, system: &SystemBoundary) -> Vec<Threat> {
        let mut threats = Vec::new();
        
        for component in &system.components {
            // Spoofing威胁
            if let Some(spoofing_threats) = self.identify_spoofing_threats(component) {
                threats.extend(spoofing_threats);
            }
            
            // Tampering威胁
            if let Some(tampering_threats) = self.identify_tampering_threats(component) {
                threats.extend(tampering_threats);
            }
            
            // Repudiation威胁
            if let Some(repudiation_threats) = self.identify_repudiation_threats(component) {
                threats.extend(repudiation_threats);
            }
            
            // Information Disclosure威胁
            if let Some(disclosure_threats) = self.identify_disclosure_threats(component) {
                threats.extend(disclosure_threats);
            }
            
            // Denial of Service威胁
            if let Some(dos_threats) = self.identify_dos_threats(component) {
                threats.extend(dos_threats);
            }
            
            // Elevation of Privilege威胁
            if let Some(elevation_threats) = self.identify_elevation_threats(component) {
                threats.extend(elevation_threats);
            }
        }
        
        threats
    }
    
    fn identify_spoofing_threats(&self, component: &Component) -> Option<Vec<Threat>> {
        let mut threats = Vec::new();
        
        match component.component_type {
            ComponentType::Device => {
                threats.push(Threat {
                    threat_id: format!("spoofing_device_{}", component.component_id),
                    name: "Device Spoofing".to_string(),
                    description: "Attacker impersonates a legitimate device".to_string(),
                    threat_type: ThreatType::Spoofing,
                    target: component.component_id.clone(),
                    attack_vector: AttackVector::Network,
                    severity: ThreatSeverity::High,
                });
            }
            ComponentType::Gateway => {
                threats.push(Threat {
                    threat_id: format!("spoofing_gateway_{}", component.component_id),
                    name: "Gateway Spoofing".to_string(),
                    description: "Attacker impersonates a legitimate gateway".to_string(),
                    threat_type: ThreatType::Spoofing,
                    target: component.component_id.clone(),
                    attack_vector: AttackVector::Network,
                    severity: ThreatSeverity::Critical,
                });
            }
            ComponentType::Cloud => {
                threats.push(Threat {
                    threat_id: format!("spoofing_cloud_{}", component.component_id),
                    name: "Cloud Service Spoofing".to_string(),
                    description: "Attacker impersonates a legitimate cloud service".to_string(),
                    threat_type: ThreatType::Spoofing,
                    target: component.component_id.clone(),
                    attack_vector: AttackVector::Network,
                    severity: ThreatSeverity::Critical,
                });
            }
            _ => {}
        }
        
        if threats.is_empty() {
            None
        } else {
            Some(threats)
        }
    }
    
    fn identify_tampering_threats(&self, component: &Component) -> Option<Vec<Threat>> {
        let mut threats = Vec::new();
        
        match component.component_type {
            ComponentType::Device => {
                threats.push(Threat {
                    threat_id: format!("tampering_device_{}", component.component_id),
                    name: "Device Tampering".to_string(),
                    description: "Attacker modifies device data or configuration".to_string(),
                    threat_type: ThreatType::Tampering,
                    target: component.component_id.clone(),
                    attack_vector: AttackVector::Physical,
                    severity: ThreatSeverity::High,
                });
            }
            ComponentType::Database => {
                threats.push(Threat {
                    threat_id: format!("tampering_database_{}", component.component_id),
                    name: "Database Tampering".to_string(),
                    description: "Attacker modifies database data".to_string(),
                    threat_type: ThreatType::Tampering,
                    target: component.component_id.clone(),
                    attack_vector: AttackVector::Network,
                    severity: ThreatSeverity::Critical,
                });
            }
            _ => {}
        }
        
        if threats.is_empty() {
            None
        } else {
            Some(threats)
        }
    }
    
    fn identify_repudiation_threats(&self, component: &Component) -> Option<Vec<Threat>> {
        let mut threats = Vec::new();
        
        if component.component_type == ComponentType::Device || 
           component.component_type == ComponentType::Application {
            threats.push(Threat {
                threat_id: format!("repudiation_{}", component.component_id),
                name: "Action Repudiation".to_string(),
                description: "User denies performing an action".to_string(),
                threat_type: ThreatType::Repudiation,
                target: component.component_id.clone(),
                attack_vector: AttackVector::Application,
                severity: ThreatSeverity::Medium,
            });
        }
        
        if threats.is_empty() {
            None
        } else {
            Some(threats)
        }
    }
    
    fn identify_disclosure_threats(&self, component: &Component) -> Option<Vec<Threat>> {
        let mut threats = Vec::new();
        
        match component.component_type {
            ComponentType::Device => {
                threats.push(Threat {
                    threat_id: format!("disclosure_device_{}", component.component_id),
                    name: "Device Data Disclosure".to_string(),
                    description: "Sensitive device data is exposed".to_string(),
                    threat_type: ThreatType::InformationDisclosure,
                    target: component.component_id.clone(),
                    attack_vector: AttackVector::Network,
                    severity: ThreatSeverity::High,
                });
            }
            ComponentType::Database => {
                threats.push(Threat {
                    threat_id: format!("disclosure_database_{}", component.component_id),
                    name: "Database Data Disclosure".to_string(),
                    description: "Sensitive database data is exposed".to_string(),
                    threat_type: ThreatType::InformationDisclosure,
                    target: component.component_id.clone(),
                    attack_vector: AttackVector::Network,
                    severity: ThreatSeverity::Critical,
                });
            }
            _ => {}
        }
        
        if threats.is_empty() {
            None
        } else {
            Some(threats)
        }
    }
    
    fn identify_dos_threats(&self, component: &Component) -> Option<Vec<Threat>> {
        let mut threats = Vec::new();
        
        match component.component_type {
            ComponentType::Device => {
                threats.push(Threat {
                    threat_id: format!("dos_device_{}", component.component_id),
                    name: "Device Denial of Service".to_string(),
                    description: "Device becomes unavailable due to attack".to_string(),
                    threat_type: ThreatType::DenialOfService,
                    target: component.component_id.clone(),
                    attack_vector: AttackVector::Network,
                    severity: ThreatSeverity::High,
                });
            }
            ComponentType::Network => {
                threats.push(Threat {
                    threat_id: format!("dos_network_{}", component.component_id),
                    name: "Network Denial of Service".to_string(),
                    description: "Network becomes unavailable due to attack".to_string(),
                    threat_type: ThreatType::DenialOfService,
                    target: component.component_id.clone(),
                    attack_vector: AttackVector::Network,
                    severity: ThreatSeverity::Critical,
                });
            }
            _ => {}
        }
        
        if threats.is_empty() {
            None
        } else {
            Some(threats)
        }
    }
    
    fn identify_elevation_threats(&self, component: &Component) -> Option<Vec<Threat>> {
        let mut threats = Vec::new();
        
        if component.component_type == ComponentType::Application ||
           component.component_type == ComponentType::Service {
            threats.push(Threat {
                threat_id: format!("elevation_{}", component.component_id),
                name: "Privilege Elevation".to_string(),
                description: "Attacker gains elevated privileges".to_string(),
                threat_type: ThreatType::ElevationOfPrivilege,
                target: component.component_id.clone(),
                attack_vector: AttackVector::Application,
                severity: ThreatSeverity::Critical,
            });
        }
        
        if threats.is_empty() {
            None
        } else {
            Some(threats)
        }
    }
    
    fn attack_tree_analysis(&self, system: &SystemBoundary) -> Vec<Threat> {
        let mut threats = Vec::new();
        
        for attack_tree in &self.attack_trees {
            let tree_threats = self.analyze_attack_tree(attack_tree, system);
            threats.extend(tree_threats);
        }
        
        threats
    }
    
    fn pattern_based_analysis(&self, system: &SystemBoundary) -> Vec<Threat> {
        let mut threats = Vec::new();
        
        for pattern in &self.threat_patterns {
            let pattern_threats = self.apply_threat_pattern(pattern, system);
            threats.extend(pattern_threats);
        }
        
        threats
    }
    
    fn deduplicate_threats(&self, threats: Vec<Threat>) -> Vec<Threat> {
        let mut unique_threats = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();
        
        for threat in threats {
            if !seen_ids.contains(&threat.threat_id) {
                seen_ids.insert(threat.threat_id.clone());
                unique_threats.push(threat);
            }
        }
        
        unique_threats
    }
}

#[derive(Debug, Clone)]
pub struct IdentificationMethod {
    pub method_id: String,
    pub name: String,
    pub description: String,
    pub method_type: MethodType,
    pub parameters: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub enum MethodType {
    STRIDE,
    AttackTree,
    ThreatPattern,
    AttackSurface,
    RiskAssessment,
}

#[derive(Debug, Clone)]
pub struct ThreatPattern {
    pub pattern_id: String,
    pub name: String,
    pub description: String,
    pub pattern_type: PatternType,
    pub indicators: Vec<PatternIndicator>,
    pub mitigations: Vec<PatternMitigation>,
}

#[derive(Debug, Clone)]
pub enum PatternType {
    Authentication,
    Authorization,
    DataFlow,
    Communication,
    Storage,
    Processing,
}

#[derive(Debug, Clone)]
pub struct AttackTree {
    pub tree_id: String,
    pub name: String,
    pub root_goal: String,
    pub nodes: Vec<AttackNode>,
    pub edges: Vec<AttackEdge>,
}

#[derive(Debug, Clone)]
pub struct AttackNode {
    pub node_id: String,
    pub name: String,
    pub node_type: NodeType,
    pub description: String,
    pub difficulty: AttackDifficulty,
    pub cost: AttackCost,
}

#[derive(Debug, Clone)]
pub enum NodeType {
    Goal,
    SubGoal,
    Attack,
    Condition,
}

#[derive(Debug, Clone)]
pub enum AttackDifficulty {
    Trivial,
    Easy,
    Medium,
    Hard,
    VeryHard,
}

#[derive(Debug, Clone)]
pub enum AttackCost {
    Low,
    Medium,
    High,
    VeryHigh,
}
```

## 三、威胁评估框架

### 3.1 风险评估模型

```rust
pub struct RiskAssessmentEngine {
    pub risk_models: Vec<RiskModel>,
    pub assessment_methods: Vec<AssessmentMethod>,
    pub risk_calculator: RiskCalculator,
}

impl RiskAssessmentEngine {
    pub fn assess_risk(&self, threat: &Threat, system: &SystemBoundary) -> RiskAssessment {
        let threat_analysis = self.analyze_threat(threat);
        let vulnerability_analysis = self.analyze_vulnerabilities(threat, system);
        let impact_analysis = self.analyze_impact(threat, system);
        let control_analysis = self.analyze_controls(threat, system);
        
        let risk_score = self.calculate_risk_score(
            &threat_analysis,
            &vulnerability_analysis,
            &impact_analysis,
            &control_analysis,
        );
        
        let risk_level = self.determine_risk_level(risk_score);
        
        RiskAssessment {
            threat: threat.clone(),
            risk_score,
            risk_level,
            threat_analysis,
            vulnerability_analysis,
            impact_analysis,
            control_analysis,
            recommendations: self.generate_recommendations(threat, risk_level),
        }
    }
    
    fn analyze_threat(&self, threat: &Threat) -> ThreatAnalysis {
        let attacker_profile = self.assess_attacker_profile(threat);
        let attack_surface = self.assess_attack_surface(threat);
        let attack_complexity = self.assess_attack_complexity(threat);
        let motivation = self.assess_attacker_motivation(threat);
        
        ThreatAnalysis {
            attacker_profile,
            attack_surface,
            attack_complexity,
            motivation,
            overall_threat_level: self.calculate_threat_level(attacker_profile, attack_surface, attack_complexity, motivation),
        }
    }
    
    fn analyze_vulnerabilities(&self, threat: &Threat, system: &SystemBoundary) -> VulnerabilityAnalysis {
        let mut vulnerabilities = Vec::new();
        
        for component in &system.components {
            if self.is_component_vulnerable(component, threat) {
                let vulnerability = self.identify_vulnerability(component, threat);
                vulnerabilities.push(vulnerability);
            }
        }
        
        let exploitability = self.assess_exploitability(&vulnerabilities);
        let prevalence = self.assess_vulnerability_prevalence(&vulnerabilities);
        
        VulnerabilityAnalysis {
            vulnerabilities,
            exploitability,
            prevalence,
            overall_vulnerability_level: self.calculate_vulnerability_level(exploitability, prevalence),
        }
    }
    
    fn analyze_impact(&self, threat: &Threat, system: &SystemBoundary) -> ImpactAnalysis {
        let confidentiality_impact = self.assess_confidentiality_impact(threat, system);
        let integrity_impact = self.assess_integrity_impact(threat, system);
        let availability_impact = self.assess_availability_impact(threat, system);
        let safety_impact = self.assess_safety_impact(threat, system);
        let financial_impact = self.assess_financial_impact(threat, system);
        let reputational_impact = self.assess_reputational_impact(threat, system);
        
        ImpactAnalysis {
            confidentiality_impact,
            integrity_impact,
            availability_impact,
            safety_impact,
            financial_impact,
            reputational_impact,
            overall_impact_level: self.calculate_impact_level(
                confidentiality_impact,
                integrity_impact,
                availability_impact,
                safety_impact,
                financial_impact,
                reputational_impact,
            ),
        }
    }
    
    fn analyze_controls(&self, threat: &Threat, system: &SystemBoundary) -> ControlAnalysis {
        let preventive_controls = self.assess_preventive_controls(threat, system);
        let detective_controls = self.assess_detective_controls(threat, system);
        let corrective_controls = self.assess_corrective_controls(threat, system);
        let deterrent_controls = self.assess_deterrent_controls(threat, system);
        
        ControlAnalysis {
            preventive_controls,
            detective_controls,
            corrective_controls,
            deterrent_controls,
            overall_control_effectiveness: self.calculate_control_effectiveness(
                preventive_controls,
                detective_controls,
                corrective_controls,
                deterrent_controls,
            ),
        }
    }
    
    fn calculate_risk_score(&self, threat: &ThreatAnalysis, vulnerability: &VulnerabilityAnalysis, impact: &ImpactAnalysis, control: &ControlAnalysis) -> f64 {
        let threat_score = threat.overall_threat_level as f64 / 5.0;
        let vulnerability_score = vulnerability.overall_vulnerability_level as f64 / 5.0;
        let impact_score = impact.overall_impact_level as f64 / 5.0;
        let control_score = 1.0 - (control.overall_control_effectiveness as f64 / 5.0);
        
        // 风险计算公式：风险 = 威胁 × 脆弱性 × 影响 × (1 - 控制有效性)
        threat_score * vulnerability_score * impact_score * control_score
    }
    
    fn determine_risk_level(&self, risk_score: f64) -> RiskLevel {
        if risk_score >= 0.8 {
            RiskLevel::Critical
        } else if risk_score >= 0.6 {
            RiskLevel::High
        } else if risk_score >= 0.4 {
            RiskLevel::Medium
        } else if risk_score >= 0.2 {
            RiskLevel::Low
        } else {
            RiskLevel::Minimal
        }
    }
    
    fn generate_recommendations(&self, threat: &Threat, risk_level: RiskLevel) -> Vec<RiskRecommendation> {
        let mut recommendations = Vec::new();
        
        match risk_level {
            RiskLevel::Critical => {
                recommendations.push(RiskRecommendation {
                    recommendation_id: format!("immediate_{}", threat.threat_id),
                    priority: RecommendationPriority::Immediate,
                    action: "Implement immediate controls".to_string(),
                    description: "This threat requires immediate attention and control implementation".to_string(),
                    timeline: "Within 24 hours".to_string(),
                });
            }
            RiskLevel::High => {
                recommendations.push(RiskRecommendation {
                    recommendation_id: format!("urgent_{}", threat.threat_id),
                    priority: RecommendationPriority::Urgent,
                    action: "Implement controls within short timeframe".to_string(),
                    description: "This threat should be addressed within a short timeframe".to_string(),
                    timeline: "Within 1 week".to_string(),
                });
            }
            RiskLevel::Medium => {
                recommendations.push(RiskRecommendation {
                    recommendation_id: format!("planned_{}", threat.threat_id),
                    priority: RecommendationPriority::Planned,
                    action: "Plan and implement controls".to_string(),
                    description: "This threat should be addressed in planned manner".to_string(),
                    timeline: "Within 1 month".to_string(),
                });
            }
            RiskLevel::Low => {
                recommendations.push(RiskRecommendation {
                    recommendation_id: format!("monitor_{}", threat.threat_id),
                    priority: RecommendationPriority::Monitor,
                    action: "Monitor and review periodically".to_string(),
                    description: "This threat should be monitored and reviewed periodically".to_string(),
                    timeline: "Within 3 months".to_string(),
                });
            }
            RiskLevel::Minimal => {
                recommendations.push(RiskRecommendation {
                    recommendation_id: format!("accept_{}", threat.threat_id),
                    priority: RecommendationPriority::Accept,
                    action: "Accept risk".to_string(),
                    description: "This threat poses minimal risk and can be accepted".to_string(),
                    timeline: "No action required".to_string(),
                });
            }
        }
        
        recommendations
    }
}

#[derive(Debug, Clone)]
pub struct RiskAssessment {
    pub threat: Threat,
    pub risk_score: f64,
    pub risk_level: RiskLevel,
    pub threat_analysis: ThreatAnalysis,
    pub vulnerability_analysis: VulnerabilityAnalysis,
    pub impact_analysis: ImpactAnalysis,
    pub control_analysis: ControlAnalysis,
    pub recommendations: Vec<RiskRecommendation>,
}

#[derive(Debug, Clone)]
pub enum RiskLevel {
    Minimal,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub struct ThreatAnalysis {
    pub attacker_profile: AttackerProfile,
    pub attack_surface: AttackSurface,
    pub attack_complexity: AttackComplexity,
    pub motivation: AttackerMotivation,
    pub overall_threat_level: ThreatLevel,
}

#[derive(Debug, Clone)]
pub struct VulnerabilityAnalysis {
    pub vulnerabilities: Vec<Vulnerability>,
    pub exploitability: Exploitability,
    pub prevalence: VulnerabilityPrevalence,
    pub overall_vulnerability_level: VulnerabilityLevel,
}

#[derive(Debug, Clone)]
pub struct ImpactAnalysis {
    pub confidentiality_impact: ImpactLevel,
    pub integrity_impact: ImpactLevel,
    pub availability_impact: ImpactLevel,
    pub safety_impact: ImpactLevel,
    pub financial_impact: ImpactLevel,
    pub reputational_impact: ImpactLevel,
    pub overall_impact_level: ImpactLevel,
}

#[derive(Debug, Clone)]
pub struct ControlAnalysis {
    pub preventive_controls: ControlEffectiveness,
    pub detective_controls: ControlEffectiveness,
    pub corrective_controls: ControlEffectiveness,
    pub deterrent_controls: ControlEffectiveness,
    pub overall_control_effectiveness: ControlEffectiveness,
}

#[derive(Debug, Clone)]
pub struct RiskRecommendation {
    pub recommendation_id: String,
    pub priority: RecommendationPriority,
    pub action: String,
    pub description: String,
    pub timeline: String,
}

#[derive(Debug, Clone)]
pub enum RecommendationPriority {
    Immediate,
    Urgent,
    Planned,
    Monitor,
    Accept,
}
```

## 四、总结

本文档建立了IoT威胁建模的理论框架，包括：

1. **威胁建模基础**：威胁建模原则、IoT威胁特性
2. **威胁识别框架**：威胁分类体系、威胁识别方法
3. **威胁评估框架**：风险评估模型

通过威胁建模框架，IoT项目能够系统性地识别、评估和缓解安全威胁。

---

**文档版本**：v1.0
**创建时间**：2024年12月
**对标标准**：Stanford CS155, MIT 6.858
**负责人**：AI助手
**审核人**：用户
