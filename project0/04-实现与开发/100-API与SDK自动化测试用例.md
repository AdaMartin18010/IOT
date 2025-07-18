# API与SDK自动化测试用例

## 1. 测试用例设计原则

- 覆盖设备寿命、维护、监管等全生命周期API/SDK的所有功能、异常、边界、批量、权限、合规场景。
- 支持多语言（Python、Go、JavaScript等）自动化测试脚本。
- 可集成CI/CD流水线，实现持续集成与质量保障。

## 2. 典型测试用例

### 2.1 正常流程

- 注册设备，上传寿命、维护、监管信息，断言返回成功。
- 查询设备健康状态、寿命预测、维护计划、合规状态，断言数据完整。
- 上报维护历史、合规报告，断言数据写入成功。

### 2.2 异常与边界

- 注册设备时缺失必填字段，断言返回参数错误。
- 查询不存在设备，断言返回设备不存在错误码。
- 上报异常事件，断言系统自动生成预警。
- 权限不足时访问敏感API，断言返回权限不足。

### 2.3 批量与分页

- 批量注册设备、批量上报维护历史，断言全部成功。
- 分页查询设备健康状态，断言分页字段正确。

### 2.4 合规性校验

- 上报不合规数据，断言系统自动识别并推送整改建议。
- 合规性报告自动归档，断言可追溯。

## 3. Python自动化测试脚本示例

```python
import requests
headers = {"Authorization": "Bearer <token>"}

def test_register_device():
    data = {
        "device_id": "dev-001",
        "design_lifetime": 10000,
        "maintenance_plan": "predictive",
        "compliance_status": "compliant"
    }
    resp = requests.post('https://iot-platform/api/devices/register', json=data, headers=headers)
    assert resp.status_code == 200
    assert resp.json()['code'] == 0

def test_get_health():
    resp = requests.get('https://iot-platform/api/devices/dev-001/health', headers=headers)
    assert resp.status_code == 200
    assert 'health_score' in resp.json()['data']

# 更多用例省略
```

## 4. Go自动化测试脚本示例

```go
import (
    "net/http"
    "testing"
)
func TestGetComplianceStatus(t *testing.T) {
    req, _ := http.NewRequest("GET", "https://iot-platform/api/devices/dev-001/compliance/status", nil)
    req.Header.Set("Authorization", "Bearer <token>")
    resp, _ := http.DefaultClient.Do(req)
    if resp.StatusCode != 200 {
        t.Errorf("Expected 200, got %d", resp.StatusCode)
    }
}
```

## 5. JavaScript自动化测试脚本示例

```javascript
test('Get device lifetime', async () => {
  const res = await fetch('https://iot-platform/api/devices/dev-001/lifetime', {
    headers: { 'Authorization': 'Bearer <token>' }
  });
  const data = await res.json();
  expect(data.code).toBe(0);
  expect(data.data).toHaveProperty('remaining_lifetime');
});
```

## 6. CI/CD集成说明

- 推荐将测试脚本集成到GitHub Actions、GitLab CI、Jenkins等CI/CD流水线。
- 每次API/SDK变更自动触发测试，保障全链路质量。
- 支持测试报告自动生成与合规性追溯。
