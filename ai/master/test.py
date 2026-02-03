import json
import requests

# 硬编码目标参数
DOCID = "dcPbCgiFT361NMXCjtOXHJRssdGcQcFBNmx-ej23sFFCjZJO1PmrZOGHDn_4dRUnUw1Nt-SD5-3fxIhNB42H1Gbw"
SHEET_ID = "t3ZSGj"
VIEW_ID = "vnBl18"
TARGET_RECORD_ID = "rFp8Qe"  # 目标记录ID
API_URL = "https://smallwecom.yesboss.work/smarttable"
HEADERS = {"Content-Type": "application/json"}


def get_target_record():
    """精准查询目标记录（确认字段格式）"""
    query_payload = {
        "action": "通用查询表单",
        "company": "拉伸大师",
        "WordList": {
            "docid": DOCID,
            "sheet_id": SHEET_ID,
            "view_id": VIEW_ID,
            "record_id": TARGET_RECORD_ID  # 直接定位目标记录
        }
    }
    print("\n📤 发送精准查询请求:")
    print(json.dumps(query_payload, ensure_ascii=False, indent=2))

    try:
        response = requests.post(API_URL, headers=HEADERS, json=query_payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        print("\n📥 查询响应（重点看字段格式）:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        if result.get("success") and result.get("data"):
            return result["data"][0]
        return None
    except Exception as e:
        print(f"❌ 查询异常: {str(e)}")
        return None


def format_target_record(record):
    """格式化目标记录（提取文本内容）"""
    if not record or record.get("record_id") != TARGET_RECORD_ID:
        print(f"❌ 非目标记录（ID不匹配）")
        return None

    values = record.get("values", {})

    # 提取字段值（按API返回的列表格式解析）
    def get_value(field_name):
        val_list = values.get(field_name, [])
        if isinstance(val_list, list) and val_list:
            # 取列表中第一个元素的text值（和API格式一致）
            first_item = val_list[0]
            return first_item.get("text", str(first_item)) if isinstance(first_item, dict) else str(first_item)
        return ""

    # 构造最终要写入的JSON结构（先转字符串）
    formatted_json_str = json.dumps({
        "info": {
            "客户": get_value("客户"),
            "会员电话": get_value("会员电话")
        },
        "tags": {
            "其他特定人群标签": get_value("有赞客户标签")
        }
    }, ensure_ascii=False)

    return {
        "record_id": TARGET_RECORD_ID,
        "json_field_value": [{"text": formatted_json_str, "type": "text"}]  # 按API格式包装
    }


def update_target_record(formatted_data):
    """按API字段格式更新目标记录"""
    if not formatted_data or formatted_data.get("record_id") != TARGET_RECORD_ID:
        print("❌ 数据无效")
        return False

    # 构造更新请求（values.json格式与查询返回一致）
    update_payload = {
        "action": "通用更新表单",
        "company": "拉伸大师",
        "WordList": {
            "docid": DOCID,
            "sheet_id": SHEET_ID,
            "record_id": TARGET_RECORD_ID,
            "values": {
                "json": formatted_data["json_field_value"]  # 直接使用包装后的列表结构
            },
            "view_id": VIEW_ID
        }
    }
    print("\n📤 发送精准更新请求（重点看json字段格式）:")
    print(json.dumps(update_payload, ensure_ascii=False, indent=2))

    try:
        response = requests.post(API_URL, headers=HEADERS, json=update_payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        print("\n📥 更新响应:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        if result.get("success"):
            updated_record = result.get("data", {}).get("records", [{}])[0]
            updated_json_val = updated_record.get("values", {}).get("json", [])

            # 验证更新结果（是否包含正确的text内容）
            if isinstance(updated_json_val, list) and updated_json_val:
                updated_text = updated_json_val[0].get("text", "")
                expected_text = formatted_data["json_field_value"][0]["text"]
                if updated_text == expected_text:
                    print("✅ 验证通过！JSON字段已正确写入")
                    print(f"写入内容: {updated_text}")
                    return True
                print(f"⚠️ 内容不匹配：预期{expected_text[:50]}，实际{updated_text[:50]}")
            else:
                print(f"⚠️ JSON字段格式异常：{updated_json_val}")
        print(f"❌ 更新失败：{result.get('message', '未知错误')}")
        return False
    except Exception as e:
        print(f"❌ 更新异常：{str(e)}")
        return False


if __name__ == "__main__":
    print("===== 开始处理目标记录（record_id: rFp8Qe） =====")
    # 1. 查询目标记录（确认字段格式）
    target_record = get_target_record()
    if not target_record:
        print("❌ 未查询到目标记录，退出")
        exit(1)
    # 2. 按API格式格式化数据
    formatted_data = format_target_record(target_record)
    if not formatted_data:
        exit(1)
    # 3. 按API格式更新记录
    is_success = update_target_record(formatted_data)
    print("\n===== 处理结束 =====")
    print("✅ 成功" if is_success else "❌ 失败")