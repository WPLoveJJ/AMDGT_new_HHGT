import requests
import json
import re
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from fastgpt import call_fastgpt_for_personalized_script


# API基础配置
API_URL = "https://smallwecom.yesboss.work/smarttable"
HEADERS = {"Content-Type": "application/json"}

# 主配置表查询参数
MASTER_CONFIG = {
    "action": "通用查询表单",
    "company": "花都家庭医生",
    "WordList": {
        "docid": "dcijMKYnKxjShY-j07ZOHvts5m58JesjOINDiFoLNuZLGLvCpilhV5P9UjJ8ABc1b9d_tFBRu_Xvl8FvmlZk1jXA",
        "sheet_id": "q979lj",
        "view_id": "vukaF8"
    }
}


# 定义解析日期字符串的函数，参数为日期字符串
def parse_date(date_str):
    if not date_str or date_str == "无数据":
        return None

    date_formats = ["%Y年%m月%d日", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt).date()
            # 按列表格式来解析date_str
        except ValueError:
            continue
    return None




# 从主配置表提取各医院的配置信息，返回值为包含有效医院配置的列表（每个元素是一个医院的配置字典）
def extract_target_config():
    #  构造API请求，查询主配置表
    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            data=json.dumps(MASTER_CONFIG)
            #   json.dumps(MASTER_CONFIG)：将主配置查询参数（MASTER_CONFIG，字典类型）转换为JSON字符串
            #   MASTER_CONFIG中包含查询动作、公司名称、文档ID等信息，用于指定查询哪个表单
        )
        response.raise_for_status()  # 检查请求是否成功：如果响应状态码不是200（如404、500等），会抛出HTTPError异常
        result = response.json()  # 将响应内容解析为JSON格式（转换为Python字典），方便后续提取数据

        # 验证响应数据是否符合预期：主配置表的数据应该在result的"data"字段，且是列表类型
        # 如果不是列表，说明没有获取到有效数据，打印提示并返回空列表
        if not isinstance(result.get("data"), list):
            print("未获取到有效数据列表")
            return []

        config_list = []   # 用于存储所有提取成功的医院配置信息

        # 遍历主配置表中的每条记录（每条记录对应一个医院的配置）
        # enumerate(result["data"], 1)：遍历data列表，idx从1开始计数（方便打印序号），item是每条记录
        for idx, item in enumerate(result["data"], 1):
            # 提取医院名称：从当前记录的"values"字段中获取"医院"信息
            hospital_info = item["values"].get("医院", [])  # 若"医院"字段不存在，默认是空列表
            # 处理医院名称：如果hospital_info非空且第一个元素是字典，取其"text"作为名称；否则用"未命名医院_序号"
            hospital_name = hospital_info[0]["text"] if (
                    hospital_info and isinstance(hospital_info[0], dict)
            ) else f"未命名医院_{idx}"

            # 提取文档ID相关的文本内容（后续需要从这些文本中解析具体配置）
            docid_array = item["values"].get("文档ID", [])  # 从记录中获取"文档ID"字段，默认空列表
            if not docid_array:
                print(f"\n【第{idx}条】{hospital_name}：无文档ID配置")  # 如果没有文档ID相关配置，打印提示并跳过当前医院
                continue

            full_doc_text = ""  # 用于拼接文档ID相关的所有文本内容
            for segment in docid_array:
                if isinstance(segment, dict):
                    full_doc_text += segment.get("text", "").strip()  # 如果片段是字典，取其"text"字段的内容（去除首尾空格）

            target_info = {"医院": hospital_name}  # 存储当前医院的配置信息，先存入医院名称

            # 用正则表达式提取docid（文档唯一标识）
            # 正则模式r'"docid"\s*:\s*"([^"]+)"'：匹配类似"docid": "xxx"的字符串，捕获引号中的xxx
            # 提取docid
            docid_match = re.search(r'"docid"\s*:\s*"([^"]+)"', full_doc_text)
            target_info["docid"] = docid_match.group(1) if docid_match else None
            # 如果匹配到，将捕获的内容作为docid；否则为None

            # 用正则表达式提取masses配置（群众表的信息，包含tab和viewId）
            # 正则模式匹配类似"masses": {"tab": "xxx", "viewId": "yyy"}的结构，捕获xxx和yyy
            # 提取masses配置
            masses_match = re.search(
                r'"masses"\s*:\s*{\s*"tab"\s*:\s*"([^"]+)"\s*,\s*"viewId"\s*:\s*"([^"]+)"',
                full_doc_text
            )
            if masses_match:   # 如果匹配到，将tab和viewId存入masses字段
                target_info["masses"] = {
                    "tab": masses_match.group(1),  # 第一个捕获组是tab的值
                    "viewId": masses_match.group(2)  # 第二个捕获组是viewId的值
                }
            else:
                target_info["masses"] = None  # 未匹配到则为None

            # 用正则表达式提取SendTask配置（任务表的信息，包含tab和viewId）
            # 正则模式匹配类似"SendTask": {"tab": "xxx", "viewId": "yyy"}的结构，捕获xxx和yyy
            # 提取SendTask配置
            send_task_match = re.search(
                r'"SendTask"\s*:\s*{\s*"tab"\s*:\s*"([^"]+)"\s*,\s*"viewId"\s*:\s*"([^"]+)"',
                full_doc_text
            )
            if send_task_match:  # 如果匹配到，将tab和viewId存入send_task字段
                target_info["send_task"] = {
                    "tab": send_task_match.group(1),
                    "viewId": send_task_match.group(2)
                }
            else:
                target_info["send_task"] = None
                
            # 提取Taskrules配置（任务规则表的信息，包含tab和viewId）
            # 正则模式匹配类似"Taskrules": {"tab": "xxx", "viewId": "yyy"}的结构，捕获xxx和yyy
            task_rules_match = re.search(
                r'"Taskrules"\s*:\s*{\s*"tab"\s*:\s*"([^"]+)"\s*,\s*"viewId"\s*:\s*"([^"]+)"',
                full_doc_text
            )
            if task_rules_match:  # 如果匹配到，将tab和viewId存入task_rules字段
                target_info["task_rules"] = {
                    "tab": task_rules_match.group(1),
                    "viewId": task_rules_match.group(2)
                }
            else:
                target_info["task_rules"] = None

            # 验证配置完整性
            if target_info["docid"] and target_info["masses"] and target_info["send_task"]:
                config_list.append(target_info)  # 配置完整，加入有效配置列表
                print(f"【第{idx}条】{hospital_name}：提取配置成功")
                # 打印任务规则表配置状态
                if target_info["task_rules"]:
                    print(f"【第{idx}条】{hospital_name}：成功提取任务规则表配置")
                else:
                    print(f"【第{idx}条】{hospital_name}：未配置任务规则表，将使用默认任务映射")
            else:  # 配置不完整，记录缺少的部分并提示
                missing = []
                if not target_info["docid"]: missing.append("docid")
                if not target_info["masses"]: missing.append("masses")
                if not target_info["send_task"]: missing.append("SendTask")
                print(f"【第{idx}条】{hospital_name}：缺少{','.join(missing)}配置，跳过")

        return config_list  # 返回所有有效配置的列表

    # 捕获请求相关的异常（如网络错误、连接超时、服务器错误等）
    except requests.exceptions.RequestException as e:
        print(f"API请求失败: {e}")
        return []
    except Exception as e:
        print(f"处理数据时发生错误: {e}")
        return []


def extract_specific_fields_for_task(record, task_rule):
    """为特定任务提取字段"""
    values = record.get("values", {})
    
    # 提取externalUserid
    external_userid = ""
    external_field = values.get("externalUserid", [])
    if isinstance(external_field, list) and len(external_field) > 0:
        external_userid = external_field[0].get("text", "") if isinstance(external_field[0], dict) else external_field[0]
    external_userid = external_userid or "无数据"

    # 提取谁加的好友_user_id
    added_by_user_id = ""
    added_by_field = values.get("谁加的好友", [])
    if isinstance(added_by_field, list) and len(added_by_field) > 0:
        added_by_user_id = added_by_field[0].get("user_id", "") if isinstance(added_by_field[0], dict) else ""
    added_by_user_id = added_by_user_id or "无数据"
    
    # 只提取当前任务需要的日期字段
    date_field_to_extract = task_rule.get("看群众哪个日期", "")
    if not date_field_to_extract:
        print(f"  任务'{task_rule.get('任务名', '')}'没有配置日期字段，跳过")
        return []
    
    print(f"  当前任务需要提取的日期字段: {date_field_to_extract}")
    
    # 解析JSON字段
    json_text = ""
    json_field = values.get("json", [])
    if isinstance(json_field, list) and len(json_field) > 0:
        json_text = json_field[0].get("text", "") if isinstance(json_field[0], dict) else json_field[0]
    elif isinstance(json_field, str):
        json_text = json_field
    
    valid_records = []
    if json_text:
        try:
            json_data = json.loads(json_text)
            info_objects = []
            if isinstance(json_data, list):
                info_objects = [obj for obj in json_data if isinstance(obj, dict)]
            elif isinstance(json_data, dict):
                info_objects = [json_data]
            
            for info_idx, info_obj in enumerate(info_objects, 1):
                info_dict = info_obj.get("info", {})
                tags_dict = info_obj.get("tags", {})
                
                # 只提取当前任务需要的日期字段
                date_value = info_dict.get(date_field_to_extract, "").strip() or tags_dict.get(date_field_to_extract, "").strip() or "无数据"
                
                if date_value == "无数据":
                    print(f"  跳过第{info_idx}个info对象（日期字段'{date_field_to_extract}'为空）")
                    continue
                
                # 提取标签字段
                specific_tags = info_dict.get("其他特定人群标签", "").strip() or tags_dict.get("其他特定人群标签", "").strip() or ""
                
                current_info = {
                    "externalUserid": external_userid,
                    "谁加的好友_user_id": added_by_user_id,
                    "info对象序号": info_idx,
                    date_field_to_extract: date_value,
                    "其他特定人群标签": specific_tags
                }
                
                valid_records.append(current_info)
                print(f"  ✅ 第{info_idx}个info对象有效：{date_field_to_extract}='{date_value}'")
        
        except json.JSONDecodeError:
            print(f"  JSON解析失败: {json_text[:100]}...")
        except Exception as e:
            print(f"  数据处理异常: {str(e)}")
    
    return valid_records


def match_tasks_for_record(record, task_rules):
    matched_tasks = []

    if not task_rules:
        return matched_tasks

    # 兼容传入的任务规则为 dict 或 list/tuple
    if isinstance(task_rules, dict):
        rules_iter = task_rules.values()
    elif isinstance(task_rules, (list, tuple)):
        rules_iter = task_rules
    else:
        # 非预期类型，直接返回
        return matched_tasks

    # 动态执行任务规则中的判断式
    for task_info in rules_iter:
        date_field = task_info.get("看群众哪个日期", "")
        judgment_code = task_info.get("判断式", "")
        task_name = task_info.get("任务名", "")
        specific_tags_required = task_info.get("特定人群（标签", "").strip()

        if not date_field or not judgment_code or not task_name:
            continue

        # 获取对应的日期值
        date_value = record.get(date_field)
        if not date_value or date_value == "无数据":
            continue

        # 解析日期
        parsed_date = parse_date(date_value)
        if not parsed_date:
            continue

        # 如果任务规则中有特定人群标签要求，进行标签匹配检查
        if specific_tags_required:
            record_tags = record.get("其他特定人群标签", "")
            if not record_tags:
                print(f"任务'{task_name}'要求特定标签，但记录中无标签信息，跳过")
                continue
            
            # 解析任务要求的标签（用逗号分隔）
            required_tags = [tag.strip() for tag in specific_tags_required.split(",") if tag.strip()]
            
            # 检查记录中的标签是否包含所有要求的标签
            tags_matched = all(required_tag in record_tags for required_tag in required_tags)
            
            if not tags_matched:
                print(f"任务'{task_name}'标签不匹配：要求{required_tags}，记录中有'{record_tags}'，跳过")
                continue
            else:
                print(f"任务'{task_name}'标签匹配成功：要求{required_tags}，记录中有'{record_tags}'")

        try:
            # 直接执行判断表达式
            local_namespace = {
                'check': parsed_date,
                'datetime': datetime,
                'timedelta': timedelta
            }
            result = eval(judgment_code, {"__builtins__": {}}, local_namespace)

            if result:
                task_obj = {
                    "任务名": task_name,
                    "externalUserid": record["externalUserid"],
                    "谁加的好友_user_id": record["谁加的好友_user_id"],
                    "话术": task_info.get("沟通话术", "")
                }
                task_obj[date_field] = date_value
                matched_tasks.append(task_obj)

        except Exception as e:
            print(f"判断式执行失败: {task_name}, 错误: {e}")
            print(f"原始判断式: {repr(judgment_code)}")
            print(f"check值: {parsed_date}")
            continue

    return matched_tasks


def query_task_rules(config):
    """查询任务规则表，严格验证四个必需字段：任务名、看群众哪个日期、通用话术、判断式，回访账号可为空"""
    if not config.get("task_rules"):
        print("  未配置任务规则表，返回空列表")
        return []
    
    query_params = {
        "action": "通用查询表单",
        "company": "花都家庭医生",
        "WordList": {
            "docid": config["docid"],
            "sheet_id": config["task_rules"]["tab"],
            "view_id": config["task_rules"]["viewId"]
        }
    }
    
    try:
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(query_params))
        response.raise_for_status()
        result = response.json()
        
        if not isinstance(result.get("data"), list):
            print("  任务规则表查询失败")
            return []
        
        task_rules_list = []  # 改为列表存储，保持顺序
        valid_count = 0
        
        for idx, item in enumerate(result["data"], 1):
            values = item.get("values", {})
            
            # 提取任务名（必需）
            task_name_field = values.get("任务名", [])
            task_name = ""
            if task_name_field and isinstance(task_name_field[0], dict):
                task_name = task_name_field[0].get("text", "").strip()
            
            # 提取看群众哪个日期（必需）
            date_field_field = values.get("看群众哪个日期", [])
            date_field = ""
            if date_field_field and isinstance(date_field_field[0], dict):
                date_field = date_field_field[0].get("text", "").strip()
            
            # 提取通用话术（区分个性化任务）
            talk_field = values.get("通用话术", [])
            talk_script = ""
            if talk_field and isinstance(talk_field[0], dict):
                talk_script = talk_field[0].get("text", "").strip()
            
            # 判断是否为个性化任务
            is_personalized = not talk_script  # 通用话术为空则为个性化任务
            
            # 提取判断式（必需）
            judgment_field = values.get("判断式", [])
            judgment_code = ""
            if judgment_field and isinstance(judgment_field[0], dict):
                # 尝试获取完整的判断式内容
                judgment_code = judgment_field[0].get("text", "").strip()
                
                # 如果text字段为空或不完整，尝试其他字段
                if not judgment_code or 'def ' in judgment_code:
                    for key in ['raw_text', 'full_text', 'content', 'value']:
                        if key in judgment_field[0]:
                            alt_content = judgment_field[0].get(key, "").strip()
                            if alt_content and len(alt_content) > len(judgment_code):
                                judgment_code = alt_content
                                break
            
            # 提取回访账号（可为空）
            visit_account_field = values.get("回访账号", [])
            visit_account = ""
            if visit_account_field and isinstance(visit_account_field[0], dict):
                visit_account = visit_account_field[0].get("user_id", "").strip()
            
            # 提取是否需要查重字段（新增）
            dedup_field = values.get("是否需要查重", [])
            dedup_value = ""
            if dedup_field and isinstance(dedup_field[0], dict):
                dedup_value = dedup_field[0].get("text", "").strip()
            
            # 提取特定人群（标签字段（新增）
            specific_tags_field = values.get("特定人群（标签", [])
            specific_tags = ""
            if specific_tags_field and isinstance(specific_tags_field[0], dict):
                specific_tags = specific_tags_field[0].get("text", "").strip()
            
            # 严格验证四个必需字段
            if not task_name:
                print(f"  第{idx}条规则缺少任务名，跳过")
                continue
            if not date_field:
                print(f"  第{idx}条规则'{task_name}'缺少看群众哪个日期，跳过")
                continue
            if not talk_script:
                print(f"  第{idx}条规则'{task_name}'缺少通用话术，跳过")  # 提示信息也要修正
                continue
            if not judgment_code:
                print(f"  第{idx}条规则'{task_name}'缺少判断式，跳过")
                continue# 严格验证必需字段（个性化任务的话术可为空）
            if not task_name:
                print(f"  第{idx}条规则缺少任务名，跳过")
                continue
            if not date_field:
                print(f"  第{idx}条规则'{task_name}'缺少看群众哪个日期，跳过")
                continue
            if not is_personalized and not talk_script:  # 非个性化任务必须有话术
                print(f"  第{idx}条规则'{task_name}'缺少通用话术，跳过")
                continue
            if not judgment_code:
                print(f"  第{idx}条规则'{task_name}'缺少判断式，跳过")
                continue
            
            # 根据是否需要查重字段设置check标志
            if dedup_value.lower() in ['是', 'true', '1', 'yes']:
                check_flag = True
            elif dedup_value.lower() in ['否', 'false', '0', 'no']:
                check_flag = False
            else:
                # 如果字段为空或无效值，使用原来的逻辑（检查判断式注释是否包含"仅一天"）
                check_flag = "仅一天" not in judgment_code
            
            # 构建完整的任务规则对象
            task_rule = {
                "任务名": task_name,
                "看群众哪个日期": date_field,
                "沟通话术": talk_script if not is_personalized else None,  # 个性化任务话术设为None
                "判断式": judgment_code,
                "回访账号": visit_account,
                "特定人群（标签": specific_tags,
                "check": check_flag,
                "is_personalized": is_personalized  # 新增：标记是否为个性化任务
            }
            
            task_rules_list.append(task_rule)
            valid_count += 1
            print(f"  第{idx}条规则'{task_name}'提取成功，check={check_flag}")
        
        print(f"  成功读取到 {valid_count} 个有效任务规则")
        return task_rules_list
        
    except Exception as e:
        print(f"  查询任务规则表失败: {str(e)}")
        return []


def check_task_already_sent(config, task_name, external_userid, friend_user_id):
    """
    检查指定任务名 + externalUserid 是否已经发送给指定的 user_id
    返回 True 表示已发送，False 表示未发送
    """
    def _get_text(field_val):
        if isinstance(field_val, list) and field_val:
            first = field_val[0]
            if isinstance(first, dict):
                return str(first.get("text") or first.get("label") or first.get("value") or "").strip()
            return str(first).strip()
        if isinstance(field_val, (str, int, float)):
            return str(field_val).strip()
        return ""

    def _get_user_ids_from_sent_field(field_val):
        """从已发送字段中提取所有user_id"""
        user_ids = []
        if isinstance(field_val, list):
            for item in field_val:
                if isinstance(item, dict):
                    user_id = str(item.get("user_id", "")).strip()
                    if user_id:
                        user_ids.append(user_id)
        return user_ids
    
    try:
        if not config.get("send_task"):
            return False

        query_params = {
            "action": "通用查询表单",
            "company": "花都家庭医生",
            "WordList": {
                "docid": config["docid"],
                "sheet_id": config["send_task"]["tab"],
                "view_id": config["send_task"]["viewId"]
            }
        }
        
        resp = requests.post(API_URL, headers=HEADERS, data=json.dumps(query_params))
        resp.raise_for_status()
        result = resp.json()

        if "data" not in result or not isinstance(result["data"], list):
            return False

        # 遍历沟通任务表中的每条记录
        for item in result["data"]:
            values = item.get("values", {})
            tn = _get_text(values.get("任务名", []))
            eu = _get_text(values.get("externalUserid", []))

            # 同时匹配 任务名 + externalUserid
            if tn == task_name and eu == external_userid:
                # 提取已发送字段中的所有 user_id
                sent_user_ids = _get_user_ids_from_sent_field(values.get("已发送", []))
                # 检查 friend_user_id 是否在已发送列表中
                if friend_user_id in sent_user_ids:
                    return True  # 已发送过
        
        return False  # 未发送过
        
    except Exception as e:
        print(f"检查任务发送状态失败: {str(e)}")
        return False  # 出错时默认为未发送


def write_task_to_form_by_category(config, task_name, task_list, check_flag):
    """
    根据check标志决定写入策略：
    - 如果check为True，需要逐条查询沟通任务表进行去重判断
    """
    if not config.get("send_task"):
        print(f"错误：缺少SendTask配置，无法写入任务「{task_name}」")
        return False

    if not task_list:
        print(f"任务「{task_name}」列表为空，跳过写入")
        return True

    print(f"\n=== 写入任务「{task_name}」({len(task_list)}个) ===")
    print(f"check标志: {check_flag}")
    
    # 第一步：分离个性化任务和普通任务
    personalized_tasks = []
    regular_tasks = []
    
    for task_info in task_list:
        script_content = task_info.get("话术", "")
        if not script_content or script_content.strip() == "":
            personalized_tasks.append(task_info)
        else:
            regular_tasks.append(task_info)
    
    # 第二步：批量处理个性化任务
    if personalized_tasks:
        print(f"发现 {len(personalized_tasks)} 个个性化任务，准备调用FastGPT生成话术...")
        
        # 收集所有个性化任务的JSON数据
        json_data_list = []
        for task_info in personalized_tasks:
            try:
                external_userid = task_info["externalUserid"]
                info_index = task_info.get("info对象序号", 0)
                
                # 从群众表查询完整JSON数据
                json_data = query_user_json_data(config, external_userid)
                if json_data and "info" in json_data:
                    info_list = json_data["info"]
                    tags_list = json_data.get("tags", [])
                    
                    if info_index < len(info_list):
                        info_obj = info_list[info_index]
                        tags_obj = tags_list[info_index] if info_index < len(tags_list) else {}
                        
                        json_data_list.append({
                            "info": info_obj,
                            "tags": tags_obj,
                            "externalUserid": external_userid
                        })
                    else:
                        print(f"  用户 {external_userid} 的info对象序号 {info_index} 超出范围，跳过")
                        json_data_list.append(None)  # 占位符
                else:
                    print(f"  无法获取用户 {external_userid} 的JSON数据，跳过")
                    json_data_list.append(None)  # 占位符
                    
            except Exception as e:
                print(f"  处理个性化任务数据异常: {e}")
                json_data_list.append(None)  # 占位符
        
        # 批量调用FastGPT生成话术
        try:
            # 过滤掉None值，只传入有效数据
            valid_json_data = [data for data in json_data_list if data is not None]
            
            if valid_json_data:
                personalized_scripts = call_fastgpt_for_personalized_script(task_name, valid_json_data)
                
                # 将生成的话术分配回原任务（处理None占位符）
                script_index = 0
                for i, task_info in enumerate(personalized_tasks):
                    if json_data_list[i] is not None:  # 有效数据
                        if script_index < len(personalized_scripts):
                            task_info["话术"] = personalized_scripts[script_index]
                            print(f"  为用户 {task_info['externalUserid']} 分配个性化话术成功")
                        else:
                            print(f"  用户 {task_info['externalUserid']} 话术生成不足，跳过")
                            task_info["话术"] = ""  # 设为空，后续会跳过
                        script_index += 1
                    else:  # 无效数据
                        task_info["话术"] = ""  # 设为空，后续会跳过
            else:
                print("  没有有效的JSON数据，跳过FastGPT调用")
                
        except Exception as e:
            print(f"  FastGPT批量生成话术失败: {e}")
            # 将所有个性化任务的话术设为空，后续会跳过
            for task_info in personalized_tasks:
                task_info["话术"] = ""
    
    # 第三步：合并所有任务并写入
    all_tasks = regular_tasks + personalized_tasks
    
    today_timestamp = str(int(datetime.now().timestamp() * 1000))
    success_count = 0
    total_count = len(all_tasks)
    
    if check_flag:
        print("check=True，将逐条检查沟通任务表进行去重...")
    else:
        print("check=False，跳过去重检查，直接写入")

    for i, task_info in enumerate(all_tasks, 1):
        # 检查必要字段
        required_fields = ["externalUserid", "谁加的好友_user_id"]
        if not all(key in task_info for key in required_fields):
            print(f"第{i}个任务信息不完整，缺少{[k for k in required_fields if k not in task_info]}，跳过")
            continue
        
        # 检查话术是否有效
        script_content = task_info.get("话术", "")
        if not script_content or script_content.strip() == "":
            print(f"第{i}个任务话术为空，跳过写入")
            continue

        external_userid = task_info["externalUserid"]
        friend_user_id = task_info["谁加的好友_user_id"]
        
        # 如果check为True，逐条检查是否已发送（按 任务名 + externalUserid 过滤）
        if check_flag:
            if check_task_already_sent(config, task_name, external_userid, friend_user_id):
                print(f"第{i}个任务已存在于沟通任务表（任务名：{task_name}，externalUserid：{external_userid}，已发送给：{friend_user_id}），跳过写入")
                continue

        write_data = {
            "action": "通用写入表单",
            "company": "花都家庭医生",
            "WordList": {
                "docid": config["docid"],
                "sheet_id": config["send_task"]["tab"],
                "view_id": config["send_task"]["viewId"],
                "values": {
                    "任务发送日期": today_timestamp,
                    "截止日期": today_timestamp,
                    "回访账号": [{"user_id": friend_user_id}],
                    "externalUserid": [{"type": "text", "text": external_userid}],
                    "任务名": [{"type": "text", "text": task_name}],
                    "话术": [{"type": "text", "text": script_content}]
                }
            }
        }

        print(f"\n===== 写入第{i}个任务 =====")
        print(API_URL)
        print("POST")
        print(json.dumps(write_data, indent=4, ensure_ascii=False))
        print("======================")

        try:
            response = requests.post(API_URL, headers=HEADERS, data=json.dumps(write_data))
            response.raise_for_status()
            result = response.json()
            print("API响应结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            if result and result.get("success", False):
                success_count += 1
                print(f"✅ 第{i}个任务写入成功")
            else:
                print(f"❌ 第{i}个任务写入失败: {result}")

        except requests.exceptions.RequestException as e:
            print(f"❌ 第{i}个任务网络请求失败: {e}")
        except Exception as e:
            print(f"❌ 第{i}个任务处理异常: {e}")

    print(f"\n===== 任务「{task_name}」写入完成 =====")
    print(f"成功: {success_count}/{total_count}")
    return success_count > 0

def query_user_json_data(config, external_userid, info_index=0):
    """根据externalUserid查询群众表的完整JSON数据，并根据序号返回对应的info+tags对象"""
    try:
        query_data = {
            "action": "通用查询表单",
            "company": "花都家庭医生",
            "WordList": {
                "docid": config["docid"],
                "sheet_id": config["masses"]["tab"],  # 修正：使用masses而不是people
                "view_id": config["masses"]["viewId"],  # 修正：使用masses而不是people
                "filter": {
                    "conditions": [
                        {
                            "field_name": "externalUserid",
                            "operator": "is",
                            "value": [external_userid]
                        }
                    ]
                }
            }
        }
        
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(query_data))
        response.raise_for_status()
        result = response.json()
        
        if result.get("code") == 0 and result.get("data", {}).get("records"):
            record = result["data"]["records"][0]
            json_text = record.get("JSON字段", "")
            if json_text:
                json_data = json.loads(json_text)
                
                # 根据序号切割JSON数据，返回对应的info+tags对象
                if "info" in json_data:
                    info_list = json_data["info"]
                    tags_list = json_data.get("tags", [])
                    
                    if info_index < len(info_list):
                        info_obj = info_list[info_index]
                        tags_obj = tags_list[info_index] if info_index < len(tags_list) else {}
                        
                        return {
                            "info": info_obj,
                            "tags": tags_obj
                        }
        
        return None
        
    except Exception as e:
        print(f"查询用户 {external_userid} 的JSON数据失败: {e}")
        return None

def query_sent_tasks_for_dedup(config, task_name):
    """
    查询沟通任务表，为指定任务名构建去重索引
    返回格式：{(externalUserid, 任务名, 回访账号_user_id)}
    """
    def _get_text(field_val):
        if isinstance(field_val, list) and field_val:
            first = field_val[0]
            if isinstance(first, dict):
                return str(first.get("text") or first.get("label") or first.get("value") or "").strip()
            return str(first).strip()
        if isinstance(field_val, (str, int, float)):
            return str(field_val).strip()
        return ""

    def _get_user_id(field_val):
        if isinstance(field_val, list) and field_val:
            first = field_val[0]
            if isinstance(first, dict):
                return str(first.get("user_id", "")).strip()
        return ""

    sent_index = set()
    
    try:
        if not config.get("send_task"):
            return sent_index

        query_params = {
            "action": "通用查询表单",
            "company": "花都家庭医生",
            "WordList": {
                "docid": config["docid"],
                "sheet_id": config["send_task"]["tab"],
                "view_id": config["send_task"]["viewId"]
            }
        }
        
        resp = requests.post(API_URL, headers=HEADERS, data=json.dumps(query_params))
        resp.raise_for_status()
        result = resp.json()

        if "data" not in result or not isinstance(result["data"], list):
            return sent_index

        for item in result["data"]:
            values = item.get("values", {})
            eu = _get_text(values.get("externalUserid", []))
            tn = _get_text(values.get("任务名", []))
            visit_account_user_id = _get_user_id(values.get("回访账号", []))
            if tn == task_name and eu and visit_account_user_id:
                sent_index.add((eu, tn, visit_account_user_id))

        return sent_index
        
    except Exception as e:
        print(f"查询沟通任务表失败: {str(e)}")
        return sent_index


def build_yesterday_sent_index(config):
    """
    查询 SendTask 表，构建昨日已发送记录的索引集合：
    key = (externalUserid, 任务名)
    仅当 状态 == '已发送' 且 任务发送日期 == 昨日 时纳入索引
    """
    def _get_text(field_val):
        if isinstance(field_val, list) and field_val:
            first = field_val[0]
            if isinstance(first, dict):
                return str(first.get("text") or first.get("label") or first.get("value") or "").strip()
            return str(first).strip()
        if isinstance(field_val, (str, int, float)):
            return str(field_val).strip()
        return ""

    def _parse_send_date(field_val):
        # 尝试将字段解析为 date 类型（兼容 毫秒时间戳/秒级时间戳/可读日期字符串）
        raw = field_val
        candidate = None
        if isinstance(raw, list) and raw:
            raw = raw[0]
        if isinstance(raw, dict):
            s = str(raw.get("text") or raw.get("value") or "").strip()
            if s.isdigit():
                ts = int(s)
                candidate = datetime.fromtimestamp(ts / 1000 if ts > 10**12 else ts)
            else:
                d = parse_date(s)
                if d:
                    return d
        elif isinstance(raw, (int, float, str)):
            s = str(raw).strip()
            if s.isdigit():
                ts = int(s)
                candidate = datetime.fromtimestamp(ts / 1000 if ts > 10**12 else ts)
            else:
                d = parse_date(s)
                if d:
                    return d
        return candidate.date() if candidate else None

    index = set()
    try:
        if not config.get("send_task"):
            return index

        query_params = {
            "action": "通用查询表单",
            "company": "花都家庭医生",
            "WordList": {
                "docid": config["docid"],
                "sheet_id": config["send_task"]["tab"],
                "view_id": config["send_task"]["viewId"]
            }
        }
        resp = requests.post(API_URL, headers=HEADERS, data=json.dumps(query_params))
        resp.raise_for_status()
        result = resp.json()

        if "data" not in result or not isinstance(result["data"], list):
            return index

        yesterday = (datetime.now() - timedelta(days=1)).date()

        for item in result["data"]:
            values = item.get("values", {})
            eu = _get_text(values.get("externalUserid", []))
            tn = _get_text(values.get("任务名", []))
            status_text = _get_text(values.get("状态", []))
            send_date = _parse_send_date(values.get("任务发送日期", []))

            if not eu or not tn or status_text != "已发送" or not send_date:
                continue

            if send_date == yesterday:
                index.add((eu, tn))

    except Exception as e:
        print(f"构建昨日已发送索引失败，将不进行昨日去重：{str(e)}")

    return index


def build_interval_sent_index(config, task_rules_mapping):
    """
    查询 SendTask 表，根据新的筛选逻辑构建已发送记录索引：
    1. 如果距离特定日期x天（起始）和距离特定日期x天（结束）是同一天，跳过后续检查
    2. 如果不是同一天，则检查任务发送日期与看群众哪个日期的差值是否在范围内，
       并检查已发送字段中的user_id是否包含当前准备写入信息的谁加的好友user_id
    """
    def _get_text(field_val):
        if isinstance(field_val, list) and field_val:
            first = field_val[0]
            if isinstance(first, dict):
                return str(first.get("text") or first.get("label") or first.get("value") or "").strip()
            return str(first).strip()
        if isinstance(field_val, (str, int, float)):
            return str(field_val).strip()
        return ""

    def _get_user_id(field_val):
        if isinstance(field_val, list) and field_val:
            first = field_val[0]
            if isinstance(first, dict):
                return str(first.get("user_id", "")).strip()
        return ""

    def _parse_send_date(field_val):
        raw = field_val
        candidate = None
        if isinstance(raw, list) and raw:
            raw = raw[0]
        if isinstance(raw, dict):
            s = str(raw.get("text") or raw.get("value") or "").strip()
            if s.isdigit():
                ts = int(s)
                candidate = datetime.fromtimestamp(ts / 1000 if ts > 10**12 else ts)
            else:
                d = parse_date(s)
                if d:
                    return d
        elif isinstance(raw, (int, float, str)):
            s = str(raw).strip()
            if s.isdigit():
                ts = int(s)
                candidate = datetime.fromtimestamp(ts / 1000 if ts > 10**12 else ts)
            else:
                d = parse_date(s)
                if d:
                    return d
        return candidate.date() if candidate else None

    # 构建筛选配置：任务名 -> {start_days, end_days, 看群众哪个日期}
    task_filter_config = {}
    for task_key, task_info in task_rules_mapping.items():
        task_name = task_info.get("任务名", "")
        start_days = task_info.get("距离特定日期x天（起始）", 0)
        end_days = task_info.get("距离特定日期x天（结束）", 0)
        date_field = task_info.get("看群众哪个日期", "")
        
        if task_name and date_field:
            task_filter_config[task_name] = {
                "start_days": start_days,
                "end_days": end_days,
                "date_field": date_field
            }

    index = set()
    try:
        if not config.get("send_task") or not task_rules_mapping:
            return index

        query_params = {
            "action": "通用查询表单",
            "company": "花都家庭医生",
            "WordList": {
                "docid": config["docid"],
                "sheet_id": config["send_task"]["tab"],
                "view_id": config["send_task"]["viewId"]
            }
        }
        resp = requests.post(API_URL, headers=HEADERS, data=json.dumps(query_params))
        resp.raise_for_status()
        result = resp.json()

        if "data" not in result or not isinstance(result["data"], list):
            return index

        # 处理已发送记录
        for item in result["data"]:
            values = item.get("values", {})
            eu = _get_text(values.get("externalUserid", []))
            tn = _get_text(values.get("任务名", []))
            status_text = _get_text(values.get("状态", []))
            send_date = _parse_send_date(values.get("任务发送日期", []))
            visit_account_user_id = _get_user_id(values.get("回访账号", []))
            
            # 提取"已发送"字段中的user_id列表
            sent_field = values.get("已发送", [])
            sent_user_ids = []
            if isinstance(sent_field, list):
                for sent_item in sent_field:
                    if isinstance(sent_item, dict):
                        user_id = sent_item.get("user_id", "")
                        if user_id:
                            sent_user_ids.append(user_id)

            if not eu or not tn or status_text != "已发送" or not send_date:
                continue

            # 检查该任务是否有筛选配置
            if tn not in task_filter_config:
                continue
                
            filter_config = task_filter_config[tn]
            start_days = filter_config["start_days"]
            end_days = filter_config["end_days"]
            
            # 如果起始天数和结束天数相同，跳过此任务的筛选（后续写入时直接写入）
            if start_days == end_days:
                continue
            
            # 为每个已发送的user_id创建索引键
            # 这里我们需要存储任务发送日期，以便在写入时进行日期差计算
            for sent_user_id in sent_user_ids:
                # 键格式：(externalUserid, 任务名, 回访账号_user_id, 已发送_user_id, 任务发送日期)
                index.add((eu, tn, visit_account_user_id, sent_user_id, send_date))

    except Exception as e:
        print(f"构建区间已发送索引失败，将不进行区间去重：{str(e)}")

    return index


def query_new_tables(config_list):
    """
    处理群众表，改为按任务逐一判断筛选策略：
    - 如果任务规则的回访账号为空，读取全部群众表数据
    - 如果任务规则的回访账号不为空，只筛选该回访账号的数据
    """
    if not config_list:
        print("没有可用于查询的配置信息")
        return

    for idx, config in enumerate(config_list, 1):
        hospital_name = config.get("医院", "未知医院")
        print(f"\n===== 处理第{idx}个群众表 =====")
        print(f"医院: {hospital_name}")
        
        # 首先查询任务规则表
        print("\n--- 查询任务规则表 ---")
        task_rules_list = query_task_rules(config)  # 改为接收列表
        
        if not task_rules_list:
            print("没有有效的任务规则，跳过该医院")
            continue

        print(f"  {hospital_name}：读取到 {len(task_rules_list)} 个任务规则")
        
        # 按任务逐一处理
        for task_rule in task_rules_list:
            task_name = task_rule.get("任务名", "")
            visit_account = task_rule.get("回访账号", "")
            
            print(f"\n--- 处理任务：{task_name} ---")
            
            # 构建查询参数
            query_params = {
                "action": "通用查询表单",
                "company": "花都家庭医生",
                "WordList": {
                    "docid": config["docid"],
                    "sheet_id": config["masses"]["tab"],
                    "view_id": config["masses"]["viewId"]
                }
            }
            
            # 如果任务规则中指定了回访账号，添加筛选条件
            if visit_account:
                query_params["WordList"]["filter"] = {
                    "谁加的好友": {"user_id": visit_account}
                }
                print(f"  按回访账号筛选：{visit_account}")
            else:
                print(f"  读取全部群众表记录")
            
            try:
                # 查询群众表
                response = requests.post(API_URL, headers=HEADERS, data=json.dumps(query_params))
                response.raise_for_status()
                result = response.json()
                
                if not isinstance(result.get("data"), list):
                    print(f"  {task_name}：群众表查询失败")
                    continue
                
                records = result["data"]
                print(f"  {task_name}：读取到 {len(records)} 条群众表记录")
                
                if not records:
                    print(f"  {task_name}：无群众表记录，跳过")
                    continue
                
                # 为当前任务提取字段并匹配
                task_matched_records = []
                
                for record_idx, record in enumerate(records, 1):
                    # 为当前任务提取特定字段
                    extracted_records = extract_specific_fields_for_task(record, task_rule)
                    
                    if not extracted_records:
                        continue
                    
                    # 对提取的记录进行任务匹配
                    for extracted_record in extracted_records:
                        matched_tasks = match_tasks_for_record(extracted_record, [task_rule])
                        task_matched_records.extend(matched_tasks)
                
                print(f"  {task_name}：匹配到 {len(task_matched_records)} 个有效记录")
                
                if task_matched_records:
                    # 写入任务到表单
                    check_flag = task_rule.get("check", True)
                    write_task_to_form_by_category(config, task_name, task_matched_records, check_flag)
                
            except requests.exceptions.RequestException as e:
                print(f"  {task_name}：API请求失败: {e}")
                continue
            except Exception as e:
                print(f"  {task_name}：处理异常: {e}")
                continue

if __name__ == "__main__":
    configs = extract_target_config()  #  调用提取配置函数，获取有效医院配置
    query_new_tables(configs)  #  调用查询群众表函数，传入配置列表