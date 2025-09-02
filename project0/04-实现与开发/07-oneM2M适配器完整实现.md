# oneM2M适配器完整实现

## 目录

- [oneM2M适配器完整实现](#onem2m适配器完整实现)
  - [目录](#目录)
  - [1. oneM2M协议核心实现](#1-onem2m协议核心实现)
    - [1.1 资源管理](#11-资源管理)
    - [1.2 订阅管理](#12-订阅管理)
  - [2. 数据转换与映射](#2-数据转换与映射)
    - [2.1 资源转换器](#21-资源转换器)
  - [3. 安全机制](#3-安全机制)
    - [3.1 访问控制](#31-访问控制)

## 1. oneM2M协议核心实现

### 1.1 资源管理

```rust
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

// oneM2M资源管理器
#[derive(Debug, Clone)]
pub struct OneM2MResourceManager {
    pub resource_store: ResourceStore,
    pub resource_validator: ResourceValidator,
    pub resource_indexer: ResourceIndexer,
}

impl OneM2MResourceManager {
    pub fn new() -> Self {
        Self {
            resource_store: ResourceStore::new(),
            resource_validator: ResourceValidator::new(),
            resource_indexer: ResourceIndexer::new(),
        }
    }

    // 创建资源
    pub async fn create_resource(
        &self,
        resource: Resource,
        parent_id: Option<String>,
    ) -> Result<Resource, ResourceError> {
        // 验证资源
        self.resource_validator.validate(&resource).await?;
        
        // 存储资源
        let stored_resource = self.resource_store.store(resource).await?;
        
        // 建立父子关系
        if let Some(parent_id) = parent_id {
            self.resource_store.link_parent_child(&parent_id, &stored_resource.resource_id).await?;
        }
        
        // 索引资源
        self.resource_indexer.index(&stored_resource).await?;
        
        Ok(stored_resource)
    }

    // 检索资源
    pub async fn retrieve_resource(
        &self,
        resource_id: &str,
    ) -> Result<Resource, ResourceError> {
        self.resource_store.get(resource_id).await
    }

    // 更新资源
    pub async fn update_resource(
        &self,
        resource_id: &str,
        updates: ResourceUpdates,
    ) -> Result<Resource, ResourceError> {
        // 获取原资源
        let mut resource = self.resource_store.get(resource_id).await?;
        
        // 应用更新
        resource.apply_updates(updates)?;
        
        // 验证更新后的资源
        self.resource_validator.validate(&resource).await?;
        
        // 存储更新
        let updated_resource = self.resource_store.update(resource).await?;
        
        // 更新索引
        self.resource_indexer.update_index(&updated_resource).await?;
        
        Ok(updated_resource)
    }

    // 删除资源
    pub async fn delete_resource(
        &self,
        resource_id: &str,
    ) -> Result<(), ResourceError> {
        // 检查子资源
        let children = self.resource_store.get_children(resource_id).await?;
        if !children.is_empty() {
            return Err(ResourceError::HasChildren);
        }
        
        // 删除资源
        self.resource_store.delete(resource_id).await?;
        
        // 从索引中移除
        self.resource_indexer.remove_from_index(resource_id).await?;
        
        Ok(())
    }
}
```

### 1.2 订阅管理

```rust
// 订阅管理器
#[derive(Debug, Clone)]
pub struct SubscriptionManager {
    pub subscription_store: SubscriptionStore,
    pub notification_manager: NotificationManager,
    pub event_filter: EventFilter,
}

impl SubscriptionManager {
    pub fn new() -> Self {
        Self {
            subscription_store: SubscriptionStore::new(),
            notification_manager: NotificationManager::new(),
            event_filter: EventFilter::new(),
        }
    }

    // 创建订阅
    pub async fn create_subscription(
        &self,
        subscription: Subscription,
    ) -> Result<Subscription, SubscriptionError> {
        // 验证订阅
        self.validate_subscription(&subscription).await?;
        
        // 存储订阅
        let stored_subscription = self.subscription_store.store(subscription).await?;
        
        // 注册事件过滤器
        self.event_filter.register_filter(&stored_subscription).await?;
        
        Ok(stored_subscription)
    }

    // 处理事件
    pub async fn handle_event(
        &self,
        event: ResourceEvent,
    ) -> Result<(), SubscriptionError> {
        // 过滤事件
        let matching_subscriptions = self.event_filter.filter_event(&event).await?;
        
        // 发送通知
        for subscription in matching_subscriptions {
            self.notification_manager.send_notification(&subscription, &event).await?;
        }
        
        Ok(())
    }
}
```

## 2. 数据转换与映射

### 2.1 资源转换器

```rust
// 资源转换器
#[derive(Debug, Clone)]
pub struct ResourceConverter {
    pub type_converter: TypeConverter,
    pub format_converter: FormatConverter,
}

impl ResourceConverter {
    pub fn new() -> Self {
        Self {
            type_converter: TypeConverter::new(),
            format_converter: FormatConverter::new(),
        }
    }

    // 转换到通用格式
    pub async fn convert_to_common(
        &self,
        resource: &Resource,
    ) -> Result<CommonResource, ConversionError> {
        let common_resource = CommonResource {
            id: resource.resource_id.clone(),
            type_uri: self.map_resource_type(&resource.resource_type),
            properties: self.convert_properties(&resource.attributes).await?,
            relationships: self.convert_relationships(&resource.children).await?,
        };
        
        Ok(common_resource)
    }

    // 转换从通用格式
    pub async fn convert_from_common(
        &self,
        common_resource: &CommonResource,
    ) -> Result<Resource, ConversionError> {
        let resource = Resource {
            resource_id: common_resource.id.clone(),
            resource_type: self.map_common_type(&common_resource.type_uri),
            attributes: self.convert_to_properties(&common_resource.properties).await?,
            children: self.convert_to_relationships(&common_resource.relationships).await?,
        };
        
        Ok(resource)
    }
}
```

## 3. 安全机制

### 3.1 访问控制

```rust
// 访问控制管理器
#[derive(Debug, Clone)]
pub struct AccessControlManager {
    pub policy_store: PolicyStore,
    pub permission_checker: PermissionChecker,
}

impl AccessControlManager {
    pub fn new() -> Self {
        Self {
            policy_store: PolicyStore::new(),
            permission_checker: PermissionChecker::new(),
        }
    }

    // 检查访问权限
    pub async fn check_access(
        &self,
        user: &User,
        resource: &Resource,
        operation: &Operation,
    ) -> Result<bool, AccessControlError> {
        // 获取访问控制策略
        let policies = self.policy_store.get_policies(resource).await?;
        
        // 检查权限
        for policy in policies {
            if self.permission_checker.check_policy(policy, user, operation).await? {
                return Ok(true);
            }
        }
        
        Ok(false)
    }
}
```

---

**oneM2M适配器完整实现完成** - 包含资源管理、订阅管理、数据转换、安全机制等核心功能。
