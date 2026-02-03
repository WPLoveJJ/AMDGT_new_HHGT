# 智能表接口使用说明

## 接口概述
所有智能表相关接口都通过统一的 `handleSmartTableOperation` 函数处理，需要在请求体中指定 `action` 和 `company` 参数。

## 接口列表

### 1. 通用查询表单列名
```javascript
/**
 * 通用查询表单列名
 * {
 *   "action": "通用查询表单列名",
 *   "company": "海达",
 *   "WordList": {
 *     "docid": "文档ID",
 *     "sheet_id": "表格ID"
 *   }
 * }
 */
```

### 2. 通用查询表单
```javascript
/**
 * 通用查询表单
 * {
 *   "action": "通用查询表单",
 *   "company": "海达",
 *   "WordList": {
 *     "docid": "文档ID",
 *     "sheet_id": "表格ID",
 *     "view_id": "视图ID（可选）"
 *   }
 * }
 */
```

### 3. 通用筛选查询表单
```javascript
/**
 * 通用筛选查询表单
 * {
 *   "action": "通用筛选查询表单",
 *   "company": "海达",
 *   "WordList": {
 *     "docid": "文档ID",
 *     "sheet_id": "表格ID",
 *     "view_id": "视图ID（可选）",
 *     "record_ids": "记录ID数组（可选）"
 *   }
 * }
 */
```

### 4. 条件筛选查询表单
```javascript
/**
 * 条件筛选查询表单
 * {
 *   "action": "条件筛选查询表单",
 *   "company": "海达",
 *   "WordList": {
 *     "docid": "文档ID",
 *     "sheet_id": "表格ID",
 *     "view_id": "视图ID（可选）",
 *     "field_titles": "字段标题数组（可选）",
 *     "filter_spec": "筛选条件（可选）"
 *   }
 * }
 */
```

### 5. 循环通用查询表单
```javascript
/**
 * 循环通用查询表单
 * {
 *   "action": "循环通用查询表单",
 *   "company": "海达",
 *   "WordList": {
 *     "docid": "文档ID",
 *     "sheet_id": "表格ID",
 *     "view_id": "视图ID（可选）",
 *     "offset": "偏移量（可选，默认0）"
 *   }
 * }
 */
```

### 6. 通用查询指定列表单
```javascript
/**
 * 通用查询指定列表单
 * {
 *   "action": "通用查询指定列表单",
 *   "company": "海达",
 *   "WordList": {
 *     "docid": "文档ID",
 *     "sheet_id": "表格ID",
 *     "view_id": "视图ID（可选）",
 *     "record_ids": "记录ID数组（可选）",
 *     "field_titles": "字段标题数组（可选）"
 *   }
 * }
 */
```

### 7. 通用更新表单
```javascript
/**
 * 通用更新表单
 * {
 *   "action": "通用更新表单",
 *   "company": "海达",
 *   "WordList": {
 *     "docid": "文档ID",
 *     "sheet_id": "表格ID",
 *     "record_id": "记录ID",
 *     "values": "更新值对象",
 *     "view_id": "视图ID（可选）"
 *   }
 * }
 */
```

### 8. 通用批量更新表单
```javascript
/**
 * 通用批量更新表单
 * {
 *   "action": "通用批量更新表单",
 *   "company": "海达",
 *   "WordList": {
 *     "docid": "文档ID",
 *     "sheet_id": "表格ID",
 *     "records": "记录数组",
 *     "view_id": "视图ID（可选）"
 *   }
 * }
 */
```

### 9. 通用写入表单
```javascript
/**
 * 通用写入表单
 * {
 *   "action": "通用写入表单",
 *   "company": "海达",
 *   "WordList": {
 *     "docid": "文档ID",
 *     "sheet_id": "表格ID",
 *     "values": "写入值对象",
 *     "view_id": "视图ID（可选）"
 *   }
 * }
 */
```

### 10. 通用批量写入表单
```javascript
/**
 * 通用批量写入表单
 * {
 *   "action": "通用批量写入表单",
 *   "company": "海达",
 *   "WordList": {
 *     "docid": "文档ID",
 *     "sheet_id": "表格ID",
 *     "records": "记录数组",
 *     "view_id": "视图ID（可选）"
 *   }
 * }
 */
```

### 11. 通用批量删除表单
```javascript
/**
 * 通用批量删除表单
 * {
 *   "action": "通用批量删除表单",
 *   "company": "海达",
 *   "WordList": {
 *     "docid": "文档ID",
 *     "sheet_id": "表格ID",
 *     "record_ids": "记录ID数组"
 *   }
 * }
 */
```

### 12. 创建智能表
```javascript
/**
 * 创建智能表
 * {
 *   "action": "创建智能表",
 *   "createForm": {
 *     "company": "海达",
 *     "formname": "表单名称",
 *     "phone": "管理员手机号数组"
 *   }
 * }
 */
```

### 13. 管理员权限
```javascript
/**
 * 管理员权限
 * {
 *   "action": "管理员权限",
 *   "company": "海达",
 *   "docid": "文档ID",
 *   "operation": "操作类型（view/add/delete）",
 *   "userid": "用户ID（add/delete时需要）",
 *   "auth": "权限级别（add时可选，默认7）"
 * }
 */
```

## 使用示例

### 请求格式
所有接口都使用 POST 方法，请求体为 JSON 格式：

```javascript
// 示例：查询表单列名
const requestBody = {
  "action": "通用查询表单列名",
  "company": "海达",
  "WordList": {
    "docid": "doc123456",
    "sheet_id": "sheet789"
  }
};

// 发送请求
fetch('/api/smarttable', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(requestBody)
});
```

### 响应格式
所有接口都返回统一的响应格式：

```javascript
// 成功响应
{
  "success": true,
  "message": "操作成功",
  "data": { /* 具体数据 */ }
}

// 错误响应
{
  "success": false,
  "message": "错误信息",
  "error": "详细错误信息"
}
```

## 注意事项

1. 所有接口都需要 `action` 和 `company` 参数
2. `company` 参数用于指定企业微信配置
3. 大部分接口都需要 `WordList` 对象，包含 `docid` 和 `sheet_id`
4. 可选参数如果不提供会使用默认值
5. 创建智能表接口会自动同步到钉钉数据表汇总
6. 创建智能表接口支持通过手机号数组自动转换为用户ID
7. 条件筛选查询表单接口包含错误日志记录
8. 管理员权限操作支持三种类型：
   - `view`：查看权限
   - `add`：添加权限
   - `delete`：删除权限
9. 所有接口都有统一的错误处理机制 