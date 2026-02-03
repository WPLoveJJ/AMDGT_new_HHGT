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
    """使用pour参数获取客户档案表数据，支持分页读取所有记录"""
    if not docid or not pour_config:
        return []

    sheet_id = pour_config.get("sheet_id", "")
    view_id = pour_config.get("view_id", "")

    if not sheet_id:
        print("pour参数中缺少sheet_id")
        return []

    all_records = []
    offset = 0
    page_size = 1000  # 每次读取1000条记录
    total_count = 0

    print("分页读取客户档案表...")

    while True:
        params = {
            "action": "循环通用查询表单",
            "company": "拉伸大师",
            "WordList": {
                "docid": docid,
                "sheet_id": sheet_id,
                "view_id": view_id,
                "offset": offset,
                "page_size": page_size,
                # 添加过滤条件：只查询json字段为空的记录
                #"filter": {
                #    "condition": "and",
                #    "conditions": [
                #        {
                #            "field": "json",
                #            "operator": "is_empty",
                #            "value": True
                #        }
                #    ]
                #}
            }
        }

        print(f"> 读取偏移量 {offset}...")

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
                    timeout=60  # 增加超时时间
                )

                # 打印响应状态码
                print(f"响应状态码: {response.status_code}")
                if response.status_code != 200:
                    print(f"响应内容: {response.text[:500]}")

                response.raise_for_status()
                data = response.json()

                if data.get("success") and "data" in data:
                    records = data["data"]
                    count = len(records)
                    total_count += count
                    print(f"成功获取 {count} 条客户档案数据，累计 {total_count} 条")

                    if count == 0:
                        print("没有更多数据")
                        return all_records

                    all_records.extend(records)

                    # 如果返回的记录数小于请求的page_size，说明已读取完毕
                    if count < page_size:
                        print(f"数据读取完毕，共 {len(all_records)} 条记录")
                        return all_records

                    # 更新偏移量
                    offset += count
                    break  # 跳出重试循环，继续下一页
                else:
                    error_msg = data.get('message', '未知错误')
                    print(f"获取客户档案数据失败: {error_msg}")
                    if attempt == max_retries - 1:
                        return all_records
            except Exception as e:
                print(f"获取客户档案数据异常 (第 {attempt + 1} 次尝试): {str(e)}")
                if attempt == max_retries - 1:
                    return all_records

        # 添加延迟避免请求过于频繁
        time.sleep(1)

    return all_records


def extract_user_id(field_data) -> str:
    """从用户字段中提取user_id"""
    if not field_data:
        return ""

    # 处理列表格式（通常是多值字段）
    if isinstance(field_data, list):
        # 只取第一个值
        if len(field_data) > 0:
            item = field_data[0]
            if isinstance(item, dict):
                return item.get("user_id", "")
        return ""

    # 处理字典格式
    if isinstance(field_data, dict):
        return field_data.get("user_id", "")

    # 直接返回字符串
    return str(field_data)


def format_customer_json(customer_data: List[Dict]) -> List[Dict]:
    """将客户数据格式化为简洁的JSON格式，包含所有必要字段"""
    formatted_customers = []

    for record in customer_data:
        values = record.get("values", {})

        # 构建info部分
        info = {
            "客户": extract_field_value(values, "客户"),
            "微信昵称": extract_field_value(values, "微信昵称"),
            "微信备注": extract_field_value(values, "微信备注"),
            "会员电话": extract_field_value(values, "会员电话"),
            "externalUserid": extract_field_value(values, "externalUserid"),
            "谁加的好友": extract_user_id(values.get("谁加的好友", {})),  # 使用专门函数提取user_id

        }

        # 构建tags部分 - 收集所有标签
        tags_list = []

        # 如果有赞客户标签不为空，添加到标签列表
        yz_tag = extract_field_value(values, "有赞客户标签")
        if yz_tag:
            tags_list.append(yz_tag)

        # 如果特定人群标签不为空，添加到标签列表
        specific_tag = extract_field_value(values, "特定人群（标签）")
        if specific_tag:
            tags_list.append(specific_tag)

        # 将标签列表转换为逗号分隔的字符串
        tags_str = ", ".join(tags_list)

        # 创建tags对象
        tags = {
            "其他特定人群标签": tags_str
        }

        # 合并为完整的客户JSON
        customer_json = {
            "info": info,
            "tags": tags
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


def update_customer_json_data(
        docid: str,
        sheet_id: str,
        view_id: str,
        formatted_customers: List[Dict],
        batch_size: int = 3
) -> int:
    """将格式化后的客户JSON数据批量更新到原表"""
    if not docid or not sheet_id:
        print("缺少必要的参数来更新数据")
        return 0

    updated_count = 0
    total_records = len(formatted_customers)

    print(f"准备更新 {total_records} 条记录的JSON字段...")
    print(f"批量处理大小: {batch_size} 条记录/批")

    # 分批处理记录
    for i in range(0, total_records, batch_size):
        batch = formatted_customers[i:i + batch_size]
        batch_records = []

        # 为批量更新创建记录结构
        for customer in batch:
            record_id = customer.get("record_id")
            if not record_id:
                print(f"记录缺少record_id，跳过")
                continue

            # 创建更新记录，保留record_id但移除它从values中
            customer_copy = customer.copy()
            customer_copy.pop("record_id", None)

            # 关键修改：按照API要求的格式包装JSON字段
            # 1. 将JSON对象转换为字符串
            json_str = json.dumps(customer_copy, ensure_ascii=False)
            # 2. 按照API要求的格式包装为列表[{"text": "...", "type": "text"}]
            json_field_value = [{"text": json_str, "type": "text"}]

            batch_records.append({
                "record_id": record_id,
                "values": {
                    "json": json_field_value  # 使用包装后的格式
                }
            })

        # 如果没有有效的record_id，跳过此批
        if not batch_records:
            print(f"批次 {i // batch_size + 1} 没有record_id，跳过")
            continue

        # 创建批量更新请求数据
        update_data = {
            "action": "通用批量更新表单",
            "company": "拉伸大师",
            "WordList": {
                "docid": docid,
                "sheet_id": sheet_id,
                "view_id": view_id,
                "records": batch_records
            }
        }

        # 发送更新请求
        print(f"更新批次 {i // batch_size + 1} (记录 {i + 1} 到 {min(i + batch_size, total_records)})")
        print(f"请求数据: {json.dumps(update_data, ensure_ascii=False, indent=2)[:500]}...")

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
                    json=update_data,
                    timeout=120
                )

                # 打印响应状态和部分内容用于调试
                print(f"响应状态码: {response.status_code}")
                if response.status_code != 200:
                    print(f"响应内容: {response.text[:500]}")

                response.raise_for_status()
                result = response.json()

                # 打印完整响应JSON用于调试
                print(f"完整响应JSON:\n{json.dumps(result, ensure_ascii=False, indent=2)[:1000]}...")

                if result.get("success"):
                    updated_in_batch = len(result.get("data", []))
                    updated_count += updated_in_batch
                    print(f"成功更新 {updated_in_batch} 条记录")
                    break  # 成功则跳出重试循环
                else:
                    error_msg = result.get('message', '未知错误')
                    print(f"批量更新失败: {error_msg}")

                    # 如果是最后一次尝试，继续处理下一批
                    if attempt == max_retries - 1:
                        print("所有重试均失败")
                        break
            except Exception as e:
                print(f"更新批次 {i // batch_size + 1} 时出错 (第 {attempt + 1} 次尝试): {str(e)}")
                if attempt == max_retries - 1:
                    print("所有重试均失败")
                    break

        # 添加延迟避免请求过于频繁
        time.sleep(1)

    return updated_count


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
    docid2 = parsed_docid["docid"]  # 客户档案表的docid

    print(f"解析成功: pour={json.dumps(pour_params)}, docid={docid2}")

    # 5. 使用pour参数和docid读取客户档案表
    print("> 读取客户档案表...")
    archive_data = get_customer_archive_data(docid2, pour_params)
    if not archive_data:
        print("获取客户档案数据失败")
        return
    print(f"成功获取 {len(archive_data)} 条客户档案记录")

    # 6. 格式化客户数据为简洁的JSON
    print("> 格式化客户数据并添加record_id...")
    formatted_customers = format_customer_json(archive_data)

    # 7. 将record_id添加到格式化后的客户数据中
    for i, record in enumerate(archive_data):
        if i < len(formatted_customers):
            formatted_customers[i]["record_id"] = record.get("record_id", "")

    # 8. 将格式化后的JSON更新回原表
    print("> 更新JSON字段到客户档案表...")
    sheet_id = pour_params.get("sheet_id", "")
    view_id = pour_params.get("view_id", "")

    # 如果没有视图ID，使用原始视图ID
    if not view_id:
        view_id = config_params.get("view_id", "")

    updated_count = update_customer_json_data(
        docid=docid2,
        sheet_id=sheet_id,
        view_id=view_id,
        formatted_customers=formatted_customers,
        batch_size=3
    )

    print(f"\n处理完成: 成功更新 {updated_count}/{len(formatted_customers)} 条记录的JSON字段")
    print("===== 处理完成 =====")


if __name__ == "__main__":
    main()