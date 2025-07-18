# 设备全生命周期API与SDK文档（递归细化版）

## 1. API接口定义

### 1.1 设备注册与自描述

- `POST /api/devices/register`：设备注册，上传语义描述、寿命、维护、监管等信息。
- `GET /api/devices/{id}/description`：获取设备语义描述及全生命周期信息。

### 1.2 健康监测与寿命预测

- `GET /api/devices/{id}/health`：获取设备健康评分、健康状态、异常检测结果。
- `GET /api/devices/{id}/lifetime`：获取设计寿命、实际寿命、剩余寿命预测。

### 1.3 维护管理

- `GET /api/devices/{id}/maintenance/plan`：获取维护计划。
- `POST /api/devices/{id}/maintenance/record`：上报维护历史。
- `GET /api/devices/{id}/maintenance/history`：查询维护历史。

### 1.4 合规性管理

- `GET /api/devices/{id}/compliance/status`：获取合规状态、合规评分、审计记录。
- `POST /api/devices/{id}/compliance/report`：上报合规性报告。

### 1.5 异常与预警

- `POST /api/devices/{id}/anomaly`：上报异常事件。
- `GET /api/devices/{id}/alerts`：获取设备预警信息。

### 1.6 闭环治理与智能联动

- `POST /api/devices/{id}/governance/feedback`：上报治理反馈，实现数据-决策-执行-反馈闭环。
- `GET /api/devices/{id}/governance/loop`：查询设备闭环治理状态。

## 2. API安全机制

- 支持OAuth2、API Key、JWT等多种认证方式。
- 细粒度权限模型，支持设备级、用户级、角色级访问控制。
- 所有敏感操作需HTTPS加密。

## 3. 错误码与返回结构

- 统一返回结构：

```json
{
  "code": 0,
  "message": "success",
  "data": { ... }
}
```

- 常见错误码：
  - 0：成功
  - 1001：认证失败
  - 1002：权限不足
  - 2001：参数错误
  - 2002：设备不存在
  - 3001：数据写入失败
  - 4001：合规性校验失败
  - 5000：系统异常

## 4. 分页、批量、异步通知

- 支持分页查询（如`?page=1&size=20`），返回`total`、`page`、`size`等字段。
- 支持批量操作（如批量上报维护历史、批量获取健康状态）。
- 支持WebHook、WebSocket等异步通知机制，推送健康异常、合规预警、维护提醒等事件。

## 5. 多语言SDK调用示例

### 5.1 Python

```python
import requests
headers = {"Authorization": "Bearer <token>"}
resp = requests.get('https://iot-platform/api/devices/dev-001/health', headers=headers)
print(resp.json())
```

### 5.2 Go

```go
import (
    "net/http"
    "io/ioutil"
)
req, _ := http.NewRequest("GET", "https://iot-platform/api/devices/dev-001/health", nil)
req.Header.Set("Authorization", "Bearer <token>")
resp, _ := http.DefaultClient.Do(req)
data, _ := ioutil.ReadAll(resp.Body)
fmt.Println(string(data))
```

### 5.3 Rust

```rust
let client = reqwest::blocking::Client::new();
let resp = client.get("https://iot-platform/api/devices/dev-001/health")
    .bearer_auth("<token>")
    .send().unwrap();
let body = resp.text().unwrap();
println!("{}", body);
```

### 5.4 Java

```java
HttpRequest request = HttpRequest.newBuilder()
    .uri(URI.create("https://iot-platform/api/devices/dev-001/health"))
    .header("Authorization", "Bearer <token>")
    .build();
HttpResponse<String> response = HttpClient.newHttpClient().send(request, BodyHandlers.ofString());
System.out.println(response.body());
```

### 5.5 JavaScript

```javascript
fetch('https://iot-platform/api/devices/dev-001/health', {
  headers: { 'Authorization': 'Bearer <token>' }
})
  .then(res => res.json())
  .then(data => console.log(data));
```

## 6. 递归扩展说明

- 所有API与SDK均递归集成寿命、维护、监管等全生命周期信息，支持多平台、多行业、多监管场景的智能治理与合规性闭环。
- 持续递归完善API安全、治理机制、行业适配等细节，支撑极限智能治理。
