import re
import json
import requests
import datetime
import time
import random
from typing import List, Dict

# 钉钉配置
DINGTALK_CONFIG = {
    "app_key": "dingoicseqn2bmdcazpl",
    "app_secret": "hiiqLe8teDkAADlJh9eklgsbtGIvrG8hPJyOC8as04wzG69OGmgaY_vQ_gyKTXEg",
    "base_id": "YndMj49yWjDEYy3ECQwPlLkgJ3pmz5aA",
    "sheet_name": "配置表",
    "operator_id": "jYEXEC84RV3QE3sm0UaeDwiEiE"
}

API_URL = "https://smallwecom.yesboss.work/smarttable"
HEADERS = {
    "Content-Type": "application/json"
}


def get_dingtalk_access_token() -> str:
    """获取钉钉API访问令牌"""
    url = "https://api.dingtalk.com/v1.0/oauth2/accessToken"
    headers = {"Content-Type": "application/json"}
    payload = {
        "appKey": DINGTALK_CONFIG["app_key"],
        "appSecret": DINGTALK_CONFIG["app_secret"]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("accessToken", "")
    except requests.exceptions.RequestException as e:
        print(f"获取钉钉access_token失败: {str(e)}")
        return ""


def get_dingtalk_config(task_name: str) -> Dict:
    """获取指定任务的钉钉配置（包含文档ID字段）"""
    access_token = get_dingtalk_access_token()
    if not access_token:
        return {}

    url = f"https://api.dingtalk.com/v1.0/notable/bases/{DINGTALK_CONFIG['base_id']}/sheets/{DINGTALK_CONFIG['sheet_name']}/records"
    headers = {"x-acs-dingtalk-access-token": access_token}
    params = {"maxResults": 100, "operatorId": DINGTALK_CONFIG["operator_id"]}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        records = data.get("records", [])
    except Exception as e:
        print(f"获取钉钉配置失败: {str(e)}")
        return {}

    for record in records:
        fields = record.get("fields", {})
        if fields.get("任务名称") == task_name:
            return fields

    return {}


def parse_general_config(config_data: Dict) -> Dict:
    """解析通用配置表字段的值，获取docid和config参数"""
    general_config = config_data.get("通用配置表", "")

    # 尝试解析为JSON（如果是字符串）
    try:
        if isinstance(general_config, str):
            # 去除可能存在的特殊字符（零宽空格）
            general_config = general_config.replace("\u200b", "").strip()
            config_dict = json.loads(general_config)
        else:
            config_dict = general_config

        # 从WordList中提取docid和config参数
        word_list = config_dict.get("WordList", {})
        return {
            "docid": word_list.get("docid", ""),
            "config": word_list.get("config", {})
        }
    except (TypeError, json.JSONDecodeError) as e:
        print(f"解析通用配置表失败: {str(e)}")
        print(f"原始值: {general_config}")
        return {}


def get_intermediate_table_data(docid: str, config: Dict) -> any:
    """使用docid和config参数读取企微表，获取文档ID字段的值"""
    if not docid or not config:
        return {}

    sheet_id = config.get("sheet_id", "")
    view_id = config.get("view_id", "")

    if not sheet_id:
        print("配置参数中缺少sheet_id")
        return {}

    params = {
        "action": "通用查询表单",
        "company": "拉伸大师",
        "WordList": {
            "docid": docid,
            "sheet_id": sheet_id,
            "view_id": view_id
        }
    }

    print(f"发送请求到企微中间表: docid={docid}, sheet_id={sheet_id}, view_id={view_id}")

    # 重试机制
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = 2 ** attempt + random.uniform(0, 1)
                print(f"第 {attempt + 1} 次重试，等待 {delay:.1f} 秒...")
                time.sleep(delay)

            response = requests.post(
                API_URL,
                headers=HEADERS,
                json=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            if data.get("success") and "data" in data:
                records = data["data"]
                if not records:
                    print("企微中间表没有记录")
                    return {}

                record = records[0]
                values = record.get("values", {})

                # 处理文档ID字段可能是文本片段列表的情况
                docid_field = values.get("文档ID", {})

                # 如果字段是列表，合并所有文本片段
                if isinstance(docid_field, list):
                    full_text = ""
                    for item in docid_field:
                        if isinstance(item, dict) and "text" in item:
                            full_text += item["text"]
                        elif isinstance(item, str):
                            full_text += item
                    return full_text

                return docid_field
            else:
                print(f"获取企微中间表数据失败: {data.get('message', '未知错误')}")
        except Exception as e:
            print(f"获取企微中间表数据异常 (第 {attempt + 1} 次尝试): {str(e)}")

        if attempt == max_retries - 1:
            print("所有重试均失败")
            return {}

    return {}


def extract_with_regex(text: str) -> Dict:
    """使用正则表达式从文本中提取任务配置信息"""
    result = {
        "docid": "",
        "pour": {"sheet_id": "", "view_id": ""},
        "task_rules": {"tab": "", "viewId": ""}
    }

    # 提取docid
    docid_match = re.search(r'"docid"\s*:\s*"([^"]+)"', text)
    if not docid_match:
        docid_match = re.search(r"'docid'\s*:\s*'([^']+)'", text)
    if docid_match:
        result["docid"] = docid_match.group(1)

    # 使用与tasks.py中masses相同的模式来提取pour参数
    pour_match = re.search(
        r'"pour"\s*:\s*{\s*"tab"\s*:\s*"([^"]+)"\s*,\s*"viewId"\s*:\s*"([^"]+)"',
        text
    )
    if pour_match:
        result["pour"] = {
            "sheet_id": pour_match.group(1),
            "view_id": pour_match.group(2)
        }
    else:
        # 如果上面的模式不匹配，尝试更宽松的模式
        pour_match_loose = re.search(r'"pour"\s*:\s*{([^}]+)}', text)
        if pour_match_loose:
            pour_text = pour_match_loose.group(1)
            
            # 提取pour.tab
            tab_match = re.search(r'"tab"\s*:\s*"([^"]+)"', pour_text)
            if tab_match:
                result["pour"]["sheet_id"] = tab_match.group(1)
            
            # 提取pour.viewId
            view_id_match = re.search(r'"viewId"\s*:\s*"([^"]+)"', pour_text)
            if view_id_match:
                result["pour"]["view_id"] = view_id_match.group(1)

    # 提取Taskrules配置
    task_rules_match = re.search(
        r'"Taskrules"\s*:\s*{\s*"tab"\s*:\s*"([^"]+)"\s*,\s*"viewId"\s*:\s*"([^"]+)"',
        text
    )
    if task_rules_match:
        result["task_rules"] = {
            "tab": task_rules_match.group(1),
            "viewId": task_rules_match.group(2)
        }

    return result


def parse_docid_field(docid_field: any) -> Dict:
    """解析文档ID字段的值，获取各种配置参数"""
    try:
        # 如果字段是字符串，尝试解析为JSON
        if isinstance(docid_field, str):
            try:
                docid_data = json.loads(docid_field)
            except json.JSONDecodeError:
                # 使用正则表达式提取关键信息
                return extract_with_regex(docid_field)
        else:
            docid_data = docid_field

        # 如果解析后是字典，提取所需信息
        if isinstance(docid_data, dict):
            # 直接提取pour部分
            pour_config = docid_data.get("pour", {})
            pour_params = {
                "sheet_id": pour_config.get("tab", ""),
                "view_id": pour_config.get("viewId", "")
            }

            return {
                "pour": pour_params,
                "docid": docid_data.get("docid", "")
            }

        # 如果解析后不是字典，使用正则表达式提取
        return extract_with_regex(str(docid_data))
    except Exception as e:
        print(f"解析文档ID字段失败: {str(e)}")
        print(f"原始值: {docid_field}")
        return {
            "pour": {"sheet_id": "", "view_id": ""},
            "docid": ""
        }


def get_customer_archive_data(docid: str, pour_config: Dict) -> List[Dict]:
    """使用pour参数获取客户档案表数据"""
    if not docid or not pour_config:
        return []

    sheet_id = pour_config.get("sheet_id", "")
    view_id = pour_config.get("view_id", "")

    if not sheet_id:
        print("pour参数中缺少sheet_id")
        return []

    params = {
        "action": "通用查询表单",
        "company": "拉伸大师",
        "WordList": {
            "docid": docid,
            "sheet_id": sheet_id,
            "view_id": view_id
        }
    }

    print(f"发送请求到客户档案表: docid={docid}, sheet_id={sheet_id}, view_id={view_id}")

    # 重试机制
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = 2 ** attempt + random.uniform(0, 1)
                print(f"第 {attempt + 1} 次重试，等待 {delay:.1f} 秒...")
                time.sleep(delay)

            response = requests.post(
                API_URL,
                headers=HEADERS,
                json=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            if data.get("success") and "data" in data:
                records = data["data"]
                print(f"成功获取 {len(records)} 条客户档案数据")
                return records
            else:
                print(f"获取客户档案数据失败: {data.get('message', '未知错误')}")
        except Exception as e:
            print(f"获取客户档案数据异常 (第 {attempt + 1} 次尝试): {str(e)}")

        if attempt == max_retries - 1:
            print("所有重试均失败")
            return []

    return []


def format_customer_json(customer_data: List[Dict]) -> List[Dict]:
    """将客户数据格式化为简洁的JSON格式"""
    formatted_customers = []

    for record in customer_data:
        values = record.get("values", {})

        # 提取各个字段
        customer_json = {
            "客户": extract_field_value(values, "客户"),
            "微信昵称": extract_field_value(values, "微信昵称"),
            "微信备注": extract_field_value(values, "微信备注"),
            "会员电话": extract_field_value(values, "会员电话"),
            "有赞客户标签": extract_field_value(values, "有赞客户标签"),
            "特定人群（标签）": extract_field_value(values, "特定人群（标签）"),
            "externalUserid": extract_field_value(values, "externalUserid"),
            "谁加的好友": extract_field_value(values, "谁加的好友"),
            "最后编辑时间": extract_field_value(values, "最后编辑时间")
        }

        formatted_customers.append(customer_json)

    return formatted_customers


def extract_field_value(values: Dict, field_name: str, default="") -> str:
    """提取字段值，处理多种格式，特别处理时间戳转换"""
    field_data = values.get(field_name)

    if not field_data:
        return default

    # 特殊处理"最后编辑时间"字段
    if field_name == "最后编辑时间":
        return convert_timestamp_field(field_data)

    # 处理列表格式（通常是多值字段）
    if isinstance(field_data, list):
        # 只取第一个值
        if len(field_data) > 0:
            item = field_data[0]
            if isinstance(item, dict):
                return item.get("text", item.get("value", default))
            else:
                return str(item)
        return default

    # 处理字典格式
    if isinstance(field_data, dict):
        return field_data.get("text", field_data.get("value", default))

    # 直接返回字符串
    return str(field_data)


def convert_timestamp_field(timestamp_data) -> str:
    """专门处理时间戳字段的转换"""
    if not timestamp_data:
        return ""

    # 如果已经是字符串，直接返回
    if isinstance(timestamp_data, str):
        return timestamp_data

    # 处理列表格式
    if isinstance(timestamp_data, list) and len(timestamp_data) > 0:
        item = timestamp_data[0]
        if isinstance(item, dict):
            # 提取时间戳值
            timestamp_value = item.get("text", item.get("value", ""))
            return convert_timestamp(timestamp_value)
        else:
            return convert_timestamp(item)

    # 处理字典格式
    if isinstance(timestamp_data, dict):
        timestamp_value = timestamp_data.get("text", timestamp_data.get("value", ""))
        return convert_timestamp(timestamp_value)

    # 直接处理数字时间戳
    return convert_timestamp(timestamp_data)


def convert_timestamp(timestamp_value):
    """将时间戳转换为中文日期字符串格式：xxxx年xx月xx日"""
    if not timestamp_value:
        return ""

    try:
        # 尝试解析为整数时间戳（可能是秒或毫秒）
        if isinstance(timestamp_value, str):
            # 如果是字符串，尝试转换为数字
            try:
                ts = float(timestamp_value)
            except ValueError:
                # 如果不是数字，直接返回原始字符串
                return timestamp_value
        else:
            ts = float(timestamp_value)

        # 判断是秒级(10位)还是毫秒级(13位)
        if ts > 1e12:  # 毫秒级时间戳
            dt = datetime.datetime.fromtimestamp(ts / 1000)
        else:  # 秒级时间戳
            dt = datetime.datetime.fromtimestamp(ts)

        # 返回中文日期格式：xxxx年xx月xx日
        return dt.strftime("%Y年%m月%d日")
    except (ValueError, TypeError, OSError):
        # 如果转换失败，直接返回原始值
        return str(timestamp_value) if timestamp_value is not None else ""


def main():
    print("===== 开始处理拉伸大师群发任务 =====")

    # 1. 读取钉钉配置（任务名称：拉伸大师群发任务生成）
    print("> 获取钉钉配置...")
    config_data = get_dingtalk_config("拉伸大师群发任务生成")
    if not config_data:
        print("未找到有效的钉钉配置")
        return
    print("钉钉配置获取成功")

    # 2. 解析通用配置表字段，获取docid和config参数
    print("> 解析通用配置表...")
    general_config = parse_general_config(config_data)
    if not general_config.get("docid") or not general_config.get("config"):
        print("无法解析通用配置表")
        print(f"配置数据: {json.dumps(config_data, ensure_ascii=False)}")
        return

    docid1 = general_config["docid"]
    config_params = general_config["config"]
    print(f"解析成功: docid={docid1}, config={json.dumps(config_params)}")

    # 3. 使用docid和config参数读取企微表，获取文档ID字段的值
    print("> 读取企微中间表...")
    docid_field = get_intermediate_table_data(docid1, config_params)
    if not docid_field:
        print("获取文档ID字段失败")
        return
    print("文档ID字段获取成功")
    print(f"文档ID字段原始值: {json.dumps(docid_field, ensure_ascii=False)[:200]}...")

    # 4. 解析文档ID字段的值，获取各种配置参数
    print("> 解析文档ID字段...")
    parsed_docid = parse_docid_field(docid_field)
    if not parsed_docid.get("pour") or not parsed_docid.get("docid"):
        print("无法解析文档ID字段")
        print(f"原始值: {docid_field}")
        return

    pour_params = parsed_docid["pour"]
    task_rules_params = parsed_docid["task_rules"]
    docid2 = parsed_docid["docid"]

    print(f"解析成功: pour={json.dumps(pour_params)}, task_rules={json.dumps(task_rules_params)}, docid={docid2}")

    # 5. 使用pour参数和docid读取客户档案表
    print("> 读取客户档案表...")
    archive_data = get_customer_archive_data(docid2, pour_params)
    if not archive_data:
        print("获取客户档案数据失败")
        return
    print(f"成功获取 {len(archive_data)} 条客户档案记录")

    # 6. 格式化客户数据为简洁的JSON
    print("> 格式化客户数据...")
    formatted_customers = format_customer_json(archive_data)

    # 7. 输出前三条记录
    print("> 输出前三条客户JSON数据:")
    for i, customer in enumerate(formatted_customers[:3], 1):
        print(f"\n客户 {i}:")
        print(json.dumps(customer, ensure_ascii=False, indent=2))

    # 8. 保存所有数据到文件
    output_file = "customer_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_customers, f, ensure_ascii=False, indent=2)

    print(f"\n处理完成，共格式化 {len(formatted_customers)} 条客户数据")
    print(f"结果已保存到: {output_file}")
    print("===== 处理完成 =====")


if __name__ == "__main__":
    main()