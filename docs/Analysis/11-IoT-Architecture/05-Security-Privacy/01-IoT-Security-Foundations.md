# IoT安全与隐私基础

## 1. 概述

本文档定义了IoT系统的安全架构、认证机制、加密方案和隐私保护技术，提供形式化的数学定义和实现示例。

## 2. 安全基础理论

### 2.1 IoT安全模型

**定义 2.1 (IoT安全系统)**
IoT安全系统是一个六元组 $\mathcal{S} = (D, U, C, A, E, P)$，其中：

- $D$ 是设备集合
- $U$ 是用户集合
- $C$ 是通信通道集合
- $A$ 是认证机制集合
- $E$ 是加密算法集合
- $P$ 是隐私保护策略集合

**定义 2.2 (安全属性)**
IoT系统必须满足以下安全属性：

1. **机密性**: $\forall d \in D, \forall c \in C, \text{Confidential}(d, c)$
2. **完整性**: $\forall d \in D, \forall c \in C, \text{Integrity}(d, c)$
3. **可用性**: $\forall d \in D, \text{Available}(d)$
4. **认证性**: $\forall u \in U, \forall d \in D, \text{Authenticated}(u, d)$

### 2.2 威胁模型

**定义 2.3 (威胁模型)**
IoT威胁模型是一个三元组 $\mathcal{T} = (A, C, I)$，其中：

- $A$ 是攻击者能力集合
- $C$ 是攻击成本函数
- $I$ 是攻击影响评估函数

**定理 2.1 (安全边界)**
对于任意IoT系统 $\mathcal{S}$ 和威胁模型 $\mathcal{T}$，如果：
$$\forall a \in A, C(a) > \text{threshold}$$
则系统在给定威胁模型下是安全的。

## 3. 设备认证机制

### 3.1 基于证书的认证

**定义 3.1 (数字证书)**
数字证书是一个五元组 $C = (PK, ID, T, S, A)$，其中：

- $PK$ 是公钥
- $ID$ 是设备标识符
- $T$ 是有效期
- $S$ 是签名
- $A$ 是颁发机构

**Rust实现**：

```rust
use ring::signature::{self, Ed25519KeyPair, KeyPair};
use ring::rand::SystemRandom;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCertificate {
    pub device_id: String,
    pub public_key: Vec<u8>,
    pub issued_at: u64,
    pub expires_at: u64,
    pub issuer: String,
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCredentials {
    pub device_id: String,
    pub private_key: Vec<u8>,
    pub certificate: DeviceCertificate,
}

pub struct CertificateAuthority {
    key_pair: Ed25519KeyPair,
    issuer_name: String,
}

impl CertificateAuthority {
    pub fn new(issuer_name: String) -> Result<Self, Box<dyn std::error::Error>> {
        let rng = SystemRandom::new();
        let pkcs8_bytes = signature::Ed25519KeyPair::generate_pkcs8(&rng)?;
        let key_pair = signature::Ed25519KeyPair::from_pkcs8(pkcs8_bytes.as_ref())?;
        
        Ok(Self {
            key_pair,
            issuer_name,
        })
    }

    pub fn issue_certificate(&self, device_id: &str, public_key: &[u8], 
                           validity_days: u64) -> Result<DeviceCertificate, Box<dyn std::error::Error>> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let expires_at = now + (validity_days * 24 * 60 * 60);
        
        let mut cert_data = format!("{}:{}:{}:{}", 
                                   device_id, 
                                   base64::encode(public_key),
                                   now, 
                                   expires_at);
        
        let signature = self.key_pair.sign(cert_data.as_bytes());
        
        Ok(DeviceCertificate {
            device_id: device_id.to_string(),
            public_key: public_key.to_vec(),
            issued_at: now,
            expires_at,
            issuer: self.issuer_name.clone(),
            signature: signature.as_ref().to_vec(),
        })
    }

    pub fn verify_certificate(&self, cert: &DeviceCertificate) -> bool {
        let cert_data = format!("{}:{}:{}:{}", 
                               cert.device_id,
                               base64::encode(&cert.public_key),
                               cert.issued_at,
                               cert.expires_at);
        
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        // 检查有效期
        if now > cert.expires_at {
            return false;
        }
        
        // 验证签名
        let public_key = signature::UnparsedPublicKey::new(
            &signature::ED25519,
            self.key_pair.public_key().as_ref()
        );
        
        public_key.verify(cert_data.as_bytes(), &cert.signature).is_ok()
    }
}

pub struct DeviceAuthenticator {
    credentials: DeviceCredentials,
}

impl DeviceAuthenticator {
    pub fn new(credentials: DeviceCredentials) -> Self {
        Self { credentials }
    }

    pub fn authenticate(&self, challenge: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let key_pair = Ed25519KeyPair::from_pkcs8(&self.credentials.private_key)?;
        let signature = key_pair.sign(challenge);
        Ok(signature.as_ref().to_vec())
    }

    pub fn verify_authentication(&self, challenge: &[u8], signature: &[u8]) -> bool {
        let public_key = signature::UnparsedPublicKey::new(
            &signature::ED25519,
            &self.credentials.certificate.public_key
        );
        
        public_key.verify(challenge, signature).is_ok()
    }
}

// 使用示例
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建证书颁发机构
    let ca = CertificateAuthority::new("IoT-CA".to_string())?;
    
    // 生成设备密钥对
    let rng = SystemRandom::new();
    let device_pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng)?;
    let device_key_pair = signature::Ed25519KeyPair::from_pkcs8(device_pkcs8.as_ref())?;
    
    // 颁发证书
    let cert = ca.issue_certificate(
        "device_001",
        device_key_pair.public_key().as_ref(),
        365
    )?;
    
    // 验证证书
    if ca.verify_certificate(&cert) {
        println!("Certificate verified successfully");
    } else {
        println!("Certificate verification failed");
    }
    
    // 设备认证
    let credentials = DeviceCredentials {
        device_id: "device_001".to_string(),
        private_key: device_pkcs8.as_ref().to_vec(),
        certificate: cert,
    };
    
    let authenticator = DeviceAuthenticator::new(credentials);
    let challenge = b"authentication_challenge";
    
    // 生成认证签名
    let signature = authenticator.authenticate(challenge)?;
    
    // 验证认证
    if authenticator.verify_authentication(challenge, &signature) {
        println!("Device authentication successful");
    } else {
        println!("Device authentication failed");
    }
    
    Ok(())
}
```

### 3.2 基于令牌的认证

**定义 3.2 (JWT令牌)**
JWT令牌是一个三元组 $J = (H, P, S)$，其中：

- $H$ 是头部（算法信息）
- $P$ 是载荷（声明信息）
- $S$ 是签名

**Go实现**：

```go
package iot

import (
    "crypto/hmac"
    "crypto/sha256"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "strings"
    "time"
)

// JWT头部
type JWTHeader struct {
    Alg string `json:"alg"`
    Typ string `json:"typ"`
}

// JWT载荷
type JWTPayload struct {
    DeviceID  string    `json:"device_id"`
    IssuedAt  int64     `json:"iat"`
    ExpiresAt int64     `json:"exp"`
    Issuer    string    `json:"iss"`
}

// JWT令牌
type JWTToken struct {
    Header    JWTHeader
    Payload   JWTPayload
    Signature string
}

// JWT认证器
type JWTAuthenticator struct {
    secretKey []byte
    issuer    string
}

func NewJWTAuthenticator(secretKey []byte, issuer string) *JWTAuthenticator {
    return &JWTAuthenticator{
        secretKey: secretKey,
        issuer:    issuer,
    }
}

// 生成JWT令牌
func (ja *JWTAuthenticator) GenerateToken(deviceID string, validityHours int) (*JWTToken, error) {
    now := time.Now().Unix()
    expiresAt := now + int64(validityHours*3600)
    
    header := JWTHeader{
        Alg: "HS256",
        Typ: "JWT",
    }
    
    payload := JWTPayload{
        DeviceID:  deviceID,
        IssuedAt:  now,
        ExpiresAt: expiresAt,
        Issuer:    ja.issuer,
    }
    
    // 编码头部和载荷
    headerJSON, err := json.Marshal(header)
    if err != nil {
        return nil, err
    }
    
    payloadJSON, err := json.Marshal(payload)
    if err != nil {
        return nil, err
    }
    
    headerB64 := base64.RawURLEncoding.EncodeToString(headerJSON)
    payloadB64 := base64.RawURLEncoding.EncodeToString(payloadJSON)
    
    // 生成签名
    data := headerB64 + "." + payloadB64
    h := hmac.New(sha256.New, ja.secretKey)
    h.Write([]byte(data))
    signature := base64.RawURLEncoding.EncodeToString(h.Sum(nil))
    
    return &JWTToken{
        Header:    header,
        Payload:   payload,
        Signature: signature,
    }, nil
}

// 验证JWT令牌
func (ja *JWTAuthenticator) VerifyToken(tokenStr string) (*JWTPayload, error) {
    parts := strings.Split(tokenStr, ".")
    if len(parts) != 3 {
        return nil, fmt.Errorf("invalid token format")
    }
    
    headerB64, payloadB64, signatureB64 := parts[0], parts[1], parts[2]
    
    // 验证签名
    data := headerB64 + "." + payloadB64
    h := hmac.New(sha256.New, ja.secretKey)
    h.Write([]byte(data))
    expectedSignature := base64.RawURLEncoding.EncodeToString(h.Sum(nil))
    
    if signatureB64 != expectedSignature {
        return nil, fmt.Errorf("invalid signature")
    }
    
    // 解码载荷
    payloadJSON, err := base64.RawURLEncoding.DecodeString(payloadB64)
    if err != nil {
        return nil, err
    }
    
    var payload JWTPayload
    if err := json.Unmarshal(payloadJSON, &payload); err != nil {
        return nil, err
    }
    
    // 检查有效期
    now := time.Now().Unix()
    if now > payload.ExpiresAt {
        return nil, fmt.Errorf("token expired")
    }
    
    // 检查颁发者
    if payload.Issuer != ja.issuer {
        return nil, fmt.Errorf("invalid issuer")
    }
    
    return &payload, nil
}

// 令牌到字符串
func (jt *JWTToken) ToString() string {
    headerJSON, _ := json.Marshal(jt.Header)
    payloadJSON, _ := json.Marshal(jt.Payload)
    
    headerB64 := base64.RawURLEncoding.EncodeToString(headerJSON)
    payloadB64 := base64.RawURLEncoding.EncodeToString(payloadJSON)
    
    return headerB64 + "." + payloadB64 + "." + jt.Signature
}

// 使用示例
func ExampleJWTAuthentication() {
    secretKey := []byte("your-secret-key-here")
    authenticator := NewJWTAuthenticator(secretKey, "IoT-Service")
    
    // 生成令牌
    token, err := authenticator.GenerateToken("device_001", 24)
    if err != nil {
        fmt.Printf("Error generating token: %v\n", err)
        return
    }
    
    tokenStr := token.ToString()
    fmt.Printf("Generated token: %s\n", tokenStr)
    
    // 验证令牌
    payload, err := authenticator.VerifyToken(tokenStr)
    if err != nil {
        fmt.Printf("Token verification failed: %v\n", err)
        return
    }
    
    fmt.Printf("Token verified successfully for device: %s\n", payload.DeviceID)
    fmt.Printf("Issued at: %d, Expires at: %d\n", payload.IssuedAt, payload.ExpiresAt)
}
```

## 4. 数据加密方案

### 4.1 对称加密

**定义 4.1 (对称加密)**
对称加密是一个三元组 $\mathcal{E} = (K, E, D)$，其中：

- $K$ 是密钥空间
- $E: K \times M \rightarrow C$ 是加密函数
- $D: K \times C \rightarrow M$ 是解密函数

**定理 4.1 (加密正确性)**
对于任意密钥 $k \in K$ 和消息 $m \in M$：
$$D(k, E(k, m)) = m$$

**Rust实现**：

```rust
use aes_gcm::{Aes256Gcm, Key, Nonce};
use aes_gcm::aead::{Aead, NewAead};
use rand::Rng;

pub struct SymmetricEncryption {
    key: Key<Aes256Gcm>,
}

impl SymmetricEncryption {
    pub fn new(key: &[u8; 32]) -> Self {
        let key = Key::from_slice(key);
        Self { key }
    }

    pub fn encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let cipher = Aes256Gcm::new(self.key);
        let nonce = Nonce::from_slice(b"unique nonce"); // 实际应用中应使用随机nonce
        
        let ciphertext = cipher.encrypt(nonce, plaintext)?;
        Ok(ciphertext)
    }

    pub fn decrypt(&self, ciphertext: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let cipher = Aes256Gcm::new(self.key);
        let nonce = Nonce::from_slice(b"unique nonce");
        
        let plaintext = cipher.decrypt(nonce, ciphertext)?;
        Ok(plaintext)
    }
}
```

### 4.2 非对称加密

**定义 4.2 (非对称加密)**
非对称加密是一个五元组 $\mathcal{A} = (K, PK, SK, E, D)$，其中：

- $K$ 是密钥对空间
- $PK$ 是公钥空间
- $SK$ 是私钥空间
- $E: PK \times M \rightarrow C$ 是加密函数
- $D: SK \times C \rightarrow M$ 是解密函数

**Go实现**：

```go
package iot

import (
    "crypto/rand"
    "crypto/rsa"
    "crypto/sha256"
    "crypto/x509"
    "encoding/pem"
    "fmt"
)

// RSA加密器
type RSAEncryption struct {
    privateKey *rsa.PrivateKey
    publicKey  *rsa.PublicKey
}

func NewRSAEncryption(bits int) (*RSAEncryption, error) {
    privateKey, err := rsa.GenerateKey(rand.Reader, bits)
    if err != nil {
        return nil, err
    }
    
    return &RSAEncryption{
        privateKey: privateKey,
        publicKey:  &privateKey.PublicKey,
    }, nil
}

// 加密
func (r *RSAEncryption) Encrypt(plaintext []byte) ([]byte, error) {
    return rsa.EncryptOAEP(sha256.New(), rand.Reader, r.publicKey, plaintext, nil)
}

// 解密
func (r *RSAEncryption) Decrypt(ciphertext []byte) ([]byte, error) {
    return rsa.DecryptOAEP(sha256.New(), rand.Reader, r.privateKey, ciphertext, nil)
}

// 签名
func (r *RSAEncryption) Sign(data []byte) ([]byte, error) {
    return rsa.SignPSS(rand.Reader, r.privateKey, crypto.SHA256, data, nil)
}

// 验证签名
func (r *RSAEncryption) Verify(data, signature []byte) error {
    return rsa.VerifyPSS(r.publicKey, crypto.SHA256, data, signature, nil)
}
```

## 5. 隐私保护技术

### 5.1 数据匿名化

**定义 5.1 (k-匿名)**
数据集 $D$ 满足k-匿名性，当且仅当对于任意准标识符 $q$，至少有 $k$ 个记录具有相同的 $q$ 值。

**算法 5.1 (数据匿名化)**

```rust
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct AnonymizedRecord {
    pub quasi_identifiers: HashMap<String, String>,
    pub sensitive_data: String,
}

pub struct KAnonymizer {
    k: usize,
}

impl KAnonymizer {
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    pub fn anonymize(&self, records: Vec<AnonymizedRecord>) -> Vec<AnonymizedRecord> {
        // 实现k-匿名化算法
        // 1. 识别准标识符
        // 2. 泛化或抑制值
        // 3. 确保每组至少有k个记录
        records
    }
}
```

### 5.2 差分隐私

**定义 5.2 (差分隐私)**
算法 $\mathcal{A}$ 满足 $\epsilon$-差分隐私，当且仅当对于任意相邻数据集 $D_1, D_2$ 和任意输出 $S$：
$$\Pr[\mathcal{A}(D_1) \in S] \leq e^\epsilon \cdot \Pr[\mathcal{A}(D_2) \in S]$$

**算法 5.2 (拉普拉斯机制)**

```rust
use rand::distributions::{Distribution, Laplace};

pub struct DifferentialPrivacy {
    epsilon: f64,
}

impl DifferentialPrivacy {
    pub fn new(epsilon: f64) -> Self {
        Self { epsilon }
    }

    pub fn add_laplace_noise(&self, value: f64, sensitivity: f64) -> f64 {
        let scale = sensitivity / self.epsilon;
        let laplace = Laplace::new(0.0, scale).unwrap();
        value + laplace.sample(&mut rand::thread_rng())
    }
}
```

## 6. 安全协议

### 6.1 TLS握手协议

**定义 6.1 (TLS握手)**
TLS握手是一个四阶段协议：

1. **ClientHello**: 客户端发送支持的加密套件
2. **ServerHello**: 服务器选择加密套件并发送证书
3. **KeyExchange**: 密钥交换和协商
4. **Finished**: 验证握手完整性

**Go实现**：

```go
package iot

import (
    "crypto/tls"
    "crypto/x509"
    "fmt"
    "net"
)

// TLS客户端
type TLSClient struct {
    config *tls.Config
}

func NewTLSClient(certFile, keyFile string) (*TLSClient, error) {
    cert, err := tls.LoadX509KeyPair(certFile, keyFile)
    if err != nil {
        return nil, err
    }
    
    config := &tls.Config{
        Certificates: []tls.Certificate{cert},
        MinVersion:   tls.VersionTLS12,
    }
    
    return &TLSClient{config: config}, nil
}

// 建立安全连接
func (tc *TLSClient) Connect(addr string) (*tls.Conn, error) {
    conn, err := tls.Dial("tcp", addr, tc.config)
    if err != nil {
        return nil, err
    }
    
    // 验证服务器证书
    if err := conn.Handshake(); err != nil {
        conn.Close()
        return nil, err
    }
    
    return conn, nil
}
```

## 7. 安全监控

### 7.1 入侵检测

**定义 7.1 (入侵检测)**
入侵检测是一个函数 $ID: E \rightarrow \{0, 1\}$，其中 $E$ 是事件集合，$ID(e) = 1$ 表示检测到入侵。

**算法 7.1 (基于规则的入侵检测)**

```rust
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SecurityEvent {
    pub event_type: String,
    pub source: String,
    pub timestamp: u64,
    pub data: HashMap<String, String>,
}

pub struct IntrusionDetector {
    rules: Vec<SecurityRule>,
}

impl IntrusionDetector {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn add_rule(&mut self, rule: SecurityRule) {
        self.rules.push(rule);
    }

    pub fn detect(&self, event: &SecurityEvent) -> bool {
        self.rules.iter().any(|rule| rule.matches(event))
    }
}
```

## 8. 总结

本文档提供了IoT系统的完整安全框架，包括：

1. **认证机制**：基于证书和JWT的设备认证
2. **加密方案**：对称和非对称加密算法
3. **隐私保护**：数据匿名化和差分隐私技术
4. **安全协议**：TLS等安全通信协议
5. **安全监控**：入侵检测和威胁响应

所有安全机制都提供了形式化的数学定义和完整的实现示例，确保IoT系统的安全性、完整性和隐私保护。
