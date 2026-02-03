import requests
import json
import re
import time
from datetime import datetime, date, timedelta
# 首先在文件顶部添加必要的导入
import asyncio
import aiohttp
from typing import Optional, Dict, Any, List
from aiohttp import ClientTimeout
import uuid
import urllib.parse

# API基础配置
API_URL = "https://smallwecom.yesboss.work/smarttable"
HEADERS = {
    "Content-Type": "application/json; charset=utf-8",  # 关键：指定 UTF-8 编码
    "Accept": "application/json"
}
# 钉钉应用配置
DINGTALK_CONFIG = {
    "app_key": "dingoicseqn2bmdcazpl",
    "app_secret": "hiiqLe8teDkAADlJh9eklgsbtGIvrG8hPJyOC8as04wzG69OGmgaY_vQ_gyKTXEg",
    "base_id": "YndMj49yWjDEYy3ECQwPlLkgJ3pmz5aA",
    "sheet_name": "配置表",
    "operator_id": "jYEXEC84RV3QE3sm0UaeDwiEiE"
}


class WeComTaskHandler:
    """企业微信任务处理类，负责创建群发任务和取消任务"""

    def __init__(self, corpid: str, corpsecret: str):
        self.corpid = corpid
        self.corpsecret = corpsecret
        self.access_token = None
        self.token_expires_at = 0  # 令牌过期时间（时间戳）
        self.mass_url = "https://qyapi.weixin.qq.com/cgi-bin/externalcontact/add_msg_template"
        self.list_url = "https://qyapi.weixin.qq.com/cgi-bin/externalcontact/get_groupmsg_list"
        self.cancel_url = "https://qyapi.weixin.qq.com/cgi-bin/externalcontact/cancel_groupmsg_send"
        self.timeout = ClientTimeout(total=30)
        self._session = aiohttp.ClientSession()  # 补充：初始化会话（原代码遗漏，需添加）

    async def _get_access_token(self, session: aiohttp.ClientSession) -> Optional[str]:
        """获取企业微信访问令牌（内部方法）"""
        # 检查令牌是否有效，有效则直接返回
        if self.access_token and self.token_expires_at > asyncio.get_event_loop().time():
            return self.access_token
        token_url = (
            f"https://qyapi.weixin.qq.com/cgi-bin/gettoken"
            f"?corpid={self.corpid}"
            f"&corpsecret={self.corpsecret}"
        )
        try:
            async with session.get(token_url, timeout=self.timeout) as resp:
                result = await resp.json()
                if result.get("errcode") == 0:
                    self.access_token = result["access_token"]
                    # 设置过期时间（提前100秒过期，避免网络延迟问题）
                    self.token_expires_at = asyncio.get_event_loop().time() + 7100
                    print(f"获取AccessToken成功，有效期7200秒")
                    return self.access_token
                else:
                    print(f"获取AccessToken失败：{result['errmsg']}（错误码：{result['errcode']}）")
                    return None
        except Exception as e:
            print(f"获取AccessToken异常：{str(e)}")
            return None

    async def create_mass_task(self, external_userid: List[str], sender: str, content: str, task_name: str) -> Dict[
        str, Any]:
        """创建企业微信群发任务（支持多个external_userid）"""
        async with aiohttp.ClientSession() as session:
            # 获取访问令牌
            access_token = await self._get_access_token(session)
            if not access_token:
                return {"success": False, "error": "无法获取AccessToken"}

            # 构建请求参数（关键修改：external_userid直接传入列表）
            payload = {
                "chat_type": "single",
                "external_userid": external_userid,  # 直接传入列表
                "sender": sender,
                "allow_select": True,
                "text": {
                    "content": content
                },
                "attachments": []
            }

            # 发送请求
            try:
                url = f"{self.mass_url}?access_token={access_token}"
                async with session.post(
                        url,
                        json=payload,
                        timeout=self.timeout
                ) as resp:
                    result = await resp.json()
                    if result.get("errcode") == 0:
                        return {
                            "success": True,
                            "msgid": result.get("msgid"),
                            "response": result
                        }
                    else:
                        return {
                            "success": False,
                            "error": result.get("errmsg"),
                            "errcode": result.get("errcode"),
                            "response": result
                        }
            except Exception as e:
                error_msg = f"请求异常: {str(e)}"
                print(error_msg)
                return {"success": False, "error": error_msg}

    async def get_yesterday_tasks(self) -> List[str]:
        """获取昨天的群发任务ID列表"""
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        return await self.get_tasks_by_date(yesterday)

    async def get_tasks_by_date(self, target_date: str) -> List[str]:
        """查询指定日期的群发任务ID列表"""
        try:
            # 解析日期并计算时间范围（当天0点至23:59:59）
            date_obj = datetime.strptime(target_date, "%Y-%m-%d")
            start_time = int(datetime(
                date_obj.year, date_obj.month, date_obj.day, 0, 0, 0
            ).timestamp())
            end_time = int(datetime(
                date_obj.year, date_obj.month, date_obj.day, 23, 59, 59
            ).timestamp())
        except ValueError:
            print("日期格式错误，请使用 'YYYY-MM-DD' 格式")
            return []
        async with aiohttp.ClientSession() as session:
            access_token = await self._get_access_token(session)
            if not access_token:
                print("无法获取访问令牌，查询任务失败")
                return []
            all_msgids = []
            cursor = ""  # 分页游标
            # 分页查询所有任务
            while True:
                payload = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "limit": 100,  # 最大每页100条
                    "cursor": cursor,
                    "chat_type": "single"
                }
                try:
                    url = f"{self.list_url}?access_token={access_token}"
                    async with session.post(url, json=payload, timeout=self.timeout) as resp:
                        result = await resp.json()
                        if result.get("errcode") != 0:
                            print(f"查询任务失败：{result['errmsg']}（错误码：{result['errcode']}）")
                            break
                        # 提取当前页的任务ID
                        current_tasks = result.get("group_msg_list", [])
                        all_msgids.extend([task["msgid"] for task in current_tasks])
                        # 检查是否有下一页
                        cursor = result.get("next_cursor", "")
                        if not cursor:
                            break  # 无更多数据，退出循环
                except Exception as e:
                    print(f"查询任务时发生异常：{str(e)}")
                    break
            print(f"查询到 {target_date} 的群发任务共 {len(all_msgids)} 个")
            return all_msgids

    async def cancel_tasks(self, msgids: List[str]) -> Dict[str, Any]:
        """批量停止群发任务"""
        if not msgids:
            return {"success": True, "message": "没有需要停止的任务", "details": {}}
        async with aiohttp.ClientSession() as session:
            access_token = await self._get_access_token(session)
            if not access_token:
                return {"success": False, "message": "无法获取访问令牌", "details": {}}
            result_details = {}
            success_count = 0
            for msgid in msgids:
                try:
                    url = f"{self.cancel_url}?access_token={access_token}"
                    payload = {"msgid": msgid}
                    async with session.post(url, json=payload, timeout=self.timeout) as resp:
                        result = await resp.json()
                        if result.get("errcode") == 0:
                            success_count += 1
                            result_details[msgid] = {"success": True, "message": "停止成功"}
                        else:
                            result_details[msgid] = {
                                "success": False,
                                "message": result.get("errmsg"),
                                "errcode": result.get("errcode")
                            }
                except Exception as e:
                    result_details[msgid] = {"success": False, "message": f"请求异常：{str(e)}"}
            return {
                "success": success_count > 0,
                "total": len(msgids),
                "success_count": success_count,
                "details": result_details
            }

    async def cancel_yesterday_tasks(self) -> Dict[str, Any]:
        """取消昨天的所有群发任务"""
        msgids = await self.get_yesterday_tasks()
        if not msgids:
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            return {"success": True, "message": f"{yesterday} 没有查询到群发任务"}
        return await self.cancel_tasks(msgids)

    # 补充：会话关闭方法（原代码遗漏，需添加到类内部）
    async def _close_session(self):
        """关闭可能残留的aiohttp会话，避免资源泄漏"""
        if hasattr(self, '_session') and not self._session.closed:
            await self._session.close()
            print("   ✅ 已关闭残留的aiohttp会话")
        return True


# 定义解析日期字符串的函数
def parse_date(date_str):
    """解析日期字符串为日期对象"""
    if not date_str or date_str == "无数据":
        return None
    date_formats = [
        "%Y年%m月%d日",  # 双数字月份/日期 (08月23日)
        "%Y年%-m月%d日",  # 单数字月份 (8月23日)
        "%Y年%m月%-d日",  # 单数字日期 (08月23日)
        "%Y年%-m月%-d日",  # 单数字月份和日期 (8月23日)
        "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"
    ]
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None

def get_dingtalk_access_token():
    """获取钉钉访问令牌"""
    url = "https://api.dingtalk.com/v1.0/oauth2/accessToken"
    headers = {"Content-Type": "application/json"}
    payload = {
        "appKey": DINGTALK_CONFIG["app_key"],
        "appSecret": DINGTALK_CONFIG["app_secret"]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("accessToken")
    except Exception as e:
        print(f"获取访问令牌失败: {e}")
        return None


def parse_multi_json(config_value):
    """解析可能包含多个JSON对象的字符串"""
    if not config_value:
        return []

    # 尝试解析为单个JSON
    try:
        return [json.loads(config_value)]
    except json.JSONDecodeError:
        pass

    # 尝试解析多个连续JSON对象
    objects = []
    start = 0
    brace_count = 0

    for i, char in enumerate(config_value):
        if char == '{':
            if brace_count == 0:
                start = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                try:
                    objects.append(json.loads(config_value[start:i + 1]))
                except json.JSONDecodeError:
                    pass

    return objects if objects else [config_value]

def get_family_doctor_configs():
    access_token = get_dingtalk_access_token()
    if not access_token:
        return None
    # 构建API请求
    base_url = "https://api.dingtalk.com/v1.0/notable/bases/"
    full_url = f"{base_url}{DINGTALK_CONFIG['base_id']}/sheets/{urllib.parse.quote(DINGTALK_CONFIG['sheet_name'])}/records"
    headers = {
        "x-acs-dingtalk-access-token": access_token,
        "Content-Type": "application/json"
    }
    params = {"maxResults": 100, "operatorId": DINGTALK_CONFIG["operator_id"]}
    try:
        response = requests.get(full_url, headers=headers, params=params)
        response.raise_for_status()
        records = response.json().get("records", [])
        result = []
        for record in records:
            fields = record.get("fields", {})
            # 筛选家医任务
            if fields.get("任务名称") != "家医":
                continue
            config_value = fields.get("通用配置表")
            if not config_value:
                continue
            # 解析配置（可能多个）
            for config in parse_multi_json(config_value):
                if isinstance(config, dict):
                    # 关键修改：直接返回钉钉原始结构
                    result.append({
                        "record_id": record.get("id"),
                        "region": fields.get("地区", ""),
                        "config": config  # 直接返回钉钉原始结构
                    })
        return result if result else None
    except Exception as e:
        print(f"获取配置失败: {e}")
        return None

def get_master_config_from_dingtalk():
    """从钉钉配置获取主配置表参数和通知表配置"""
    configs = get_family_doctor_configs()
    if not configs:
        print("❌❌ 未从钉钉获取到家医任务配置")
        return None, None

    # 取第一个有效配置
    config = configs[0]

    # 直接访问WordList结构
    if "config" not in config or "WordList" not in config["config"]:
        print("❌❌ 钉钉返回的配置格式不符合预期")
        print(f"完整配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
        return None, None

    wordlist_data = config["config"]["WordList"]

    # 提取主配置表参数
    master_config = {
        "action": "通用查询表单",
        "company": "花都家庭医生",
        "WordList": {
            "docid": wordlist_data["docid"],
            "sheet_id": wordlist_data["config"]["sheet_id"],
            "view_id": wordlist_data["config"]["view_id"]
        }
    }

    print(f"✅ 获取到主配置表参数: docid={master_config['WordList']['docid']}")

    # 提取通知表配置（不存在时返回None）
    notice_config = wordlist_data.get("notice")  # 关键修改：去掉默认空字典
    if notice_config:
        print(f"✅ 获取到通知表配置: sheet_id={notice_config.get('sheet_id')}, view_id={notice_config.get('view_id')}")
    else:
        print("⚠️ 未找到通知表配置，将跳过全区通知处理")

    return master_config, notice_config


# -------------------------- 核心改动1：提取医院配置时增加personalize的tab和viewid --------------------------
def extract_target_config(master_config):
    """从企微主配置表提取各医院配置信息"""
    # 从钉钉获取主配置表参数
    #master_config = get_master_config_from_dingtalk()
    #if not master_config:
    #    print("❌ 无法获取主配置表参数")
    #    return []
    print(f"🔧 使用主配置表参数: docid={master_config['WordList']['docid']}")
    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json=master_config
        )
        response.raise_for_status()
        result = response.json()
        # 验证响应数据是否符合预期
        if not isinstance(result.get("data"), list):
            print("未获取到有效数据列表")
            return []
        config_list = []  # 用于存储所有提取成功的医院配置信息
        print(f"📋 获取到 {len(result['data'])} 条医院配置记录")
        # 遍历主配置表中的每条记录
        for idx, item in enumerate(result["data"], 1):
            # 提取医院名称
            values = item.get("values", {})
            hospital_info = values.get("医院", [])
            hospital_name = hospital_info[0]["text"] if (
                    hospital_info and isinstance(hospital_info[0], dict)
            ) else f"未命名医院_{idx}"
            # 提取文档ID相关的文本内容
            docid_array = values.get("文档ID", [])
            if not docid_array:
                print(f"【第{idx}条】{hospital_name}：无文档ID配置")
                continue
            full_doc_text = ""
            for segment in docid_array:
                if isinstance(segment, dict):
                    full_doc_text += segment.get("text", "").strip()
            target_info = {"医院": hospital_name}
            # 用正则表达式提取docid
            docid_match = re.search(r'"docid"\s*:\s*"([^"]+)"', full_doc_text)
            target_info["docid"] = docid_match.group(1) if docid_match else None
            # 用正则表达式提取masses配置
            masses_match = re.search(
                r'"masses"\s*:\s*{\s*"tab"\s*:\s*"([^"]+)"\s*,\s*"viewId"\s*:\s*"([^"]+)"',
                full_doc_text
            )
            if masses_match:
                target_info["masses"] = {
                    "tab": masses_match.group(1),
                    "viewId": masses_match.group(2)
                }
            else:
                target_info["masses"] = None
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
            # -------------------------- 新增：提取personalize配置（tab和viewId） --------------------------
            personalize_match = re.search(
                r'"Personalize"\s*:\s*{\s*"tab"\s*:\s*"([^"]+)"\s*,\s*"viewId"\s*:\s*"([^"]+)"',
                full_doc_text
            )
            if personalize_match:  # 匹配个性化任务表的tab和viewId
                target_info["personalize"] = {
                    "tab": personalize_match.group(1),
                    "viewId": personalize_match.group(2)
                }
            else:
                target_info["personalize"] = None  # 无配置时设为None

            # 验证配置完整性
            if target_info["docid"] and target_info["masses"] and target_info["send_task"]:
                config_list.append(target_info)  # 配置完整，加入有效配置列表
                print(f"【第{idx}条】{hospital_name}：提取配置成功")
                # 打印任务规则表配置状态
                if target_info["task_rules"]:
                    print(f"【第{idx}条】{hospital_name}：成功提取任务规则表配置")
                else:
                    print(f"【第{idx}条】{hospital_name}：未配置任务规则表，将使用默认任务映射")
                # 新增：打印个性化配置状态
                if target_info["personalize"]:
                    print(f"【第{idx}条】{hospital_name}：成功提取个性化任务表配置（tab：{target_info['personalize']['tab']}）")
                else:
                    print(f"【第{idx}条】{hospital_name}：未配置个性化任务表")
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


# -------------------------- 核心改动2：提取群众字段时处理输入参数（含json多对象剥离） --------------------------
def extract_specific_fields_for_task(record, task_rule):
    """为特定任务提取字段"""
    values = record.get("values", {})
    # 提取externalUserid
    external_userid = ""
    external_field = values.get("externalUserid", [])
    if isinstance(external_field, list) and len(external_field) > 0:
        external_userid = external_field[0].get("text", "") if isinstance(external_field[0], dict) else external_field[
            0]
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

    # -------------------------- 新增：获取当前任务的输入参数（个性化任务专属） --------------------------
    input_param = task_rule.get("输入参数", "")  # 从任务规则中获取需提取的输入参数字段名
    print(f"  当前任务需要提取的输入参数字段: {input_param}")

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
                date_value = info_dict.get(date_field_to_extract, "").strip() or tags_dict.get(date_field_to_extract,
                                                                                               "").strip() or "无数据"
                if date_value == "无数据":
                    print(f"  跳过第{info_idx}个info对象（日期字段'{date_field_to_extract}'为空）")
                    continue

                # -------------------------- 新增：提取输入参数对应的数据（含json对象剥离） --------------------------
                personalized_input = {}
                if input_param:  # 仅当输入参数不为空时提取（个性化任务）
                    if input_param == "json":
                        # 输入参数为json：剥离当前符合条件的单个info对象（而非整个json数组）
                        personalized_input[input_param] = info_obj
                    else:
                        # 输入参数为普通字段：从values中提取对应字段值
                        param_field = values.get(input_param, [])
                        if isinstance(param_field, list) and len(param_field) > 0:
                            param_value = param_field[0].get("text", "") if isinstance(param_field[0], dict) else \
                            param_field[0]
                        else:
                            param_value = str(param_field) if param_field else ""
                        personalized_input[input_param] = param_value.strip()

                # 提取标签字段
                specific_tags = info_dict.get("其他特定人群标签", "").strip() or tags_dict.get("其他特定人群标签", "").strip() or ""
                current_info = {
                    "externalUserid": external_userid,
                    "谁加的好友_user_id": added_by_user_id,
                    "info对象序号": info_idx,
                    date_field_to_extract: date_value,
                    "其他特定人群标签": specific_tags,
                    # -------------------------- 新增：携带个性化输入参数数据 --------------------------
                    "个性化输入参数": personalized_input,
                    "是否个性化任务": task_rule.get("是否个性化任务", False),  # 标记是否为个性化任务
                    "提示词": task_rule.get("提示词", ""),  # 传递提示词
                    "任务类型": task_rule.get("任务类型", "")  # 传递任务类型
                }
                valid_records.append(current_info)
                print(f"  ✅ 第{info_idx}个info对象有效：{date_field_to_extract}='{date_value}'")
        except json.JSONDecodeError:
            print(f"  JSON解析失败: {json_text[:100]}...")
        except Exception as e:
            print(f"  数据处理异常: {str(e)}")
    return valid_records


def match_tasks_for_record(record, task_rules,hospital_name):
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
                'timedelta': timedelta,
                'parse_date': parse_date
            }
            result = eval(judgment_code, {"__builtins__": {}}, local_namespace)
            if result:
                raw_script = task_info.get("沟通话术", "")
                processed_script = raw_script.replace("_", hospital_name) if not task_info.get(
                    "是否个性化任务") else raw_script

                task_obj = {
                    "任务名": task_name,
                    "externalUserid": record["externalUserid"],
                    "谁加的好友_user_id": record["谁加的好友_user_id"],
                    "话术": processed_script,
                    "check": task_info.get("check", True),
                    # -------------------------- 新增：传递个性化任务相关字段 --------------------------
                    "是否个性化任务": task_info.get("是否个性化任务", False),
                    "提示词": task_info.get("提示词", ""),
                    "输入参数": task_info.get("输入参数", ""),
                    "任务类型": task_info.get("任务类型", ""),
                    "个性化输入参数": record.get("个性化输入参数", {})
                }
                matched_tasks.append(task_obj)
        except Exception as e:
            print(f"判断式执行失败: {task_name}, 错误: {e}")
            print(f"原始判断式: {repr(judgment_code)}")
            print(f"check值: {parsed_date}")
            continue
    return matched_tasks


# -------------------------- 核心改动3：查询任务规则时筛选提示词不为空，提取输入参数、任务类型 --------------------------
def query_task_rules(config):
    """查询任务规则表，调整验证逻辑：允许缺少通用话术，通过提示词判断个性化任务"""
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
        normal_count = 0  # 普通任务计数
        personalized_count = 0  # 个性化任务计数

        for idx, item in enumerate(result["data"], 1):
            values = item.get("values", {})
            # 1. 提取原有必需字段
            task_name = ""
            task_name_field = values.get("任务名", [])
            if task_name_field and isinstance(task_name_field[0], dict):
                task_name = task_name_field[0].get("text", "").strip()

            date_field = ""
            date_field_field = values.get("看群众哪个日期", [])
            if date_field_field and isinstance(date_field_field[0], dict):
                date_field = date_field_field[0].get("text", "").strip()

            talk_script = ""
            talk_field = values.get("通用话术", [])
            if talk_field and isinstance(talk_field[0], dict):
                talk_script = talk_field[0].get("text", "").strip()

            judgment_code = ""
            judgment_field = values.get("判断式", [])
            if judgment_field and isinstance(judgment_field[0], dict):
                judgment_code = judgment_field[0].get("text", "").strip()
                if not judgment_code or 'def ' in judgment_code:
                    for key in ['raw_text', 'full_text', 'content', 'value']:
                        if key in judgment_field[0]:
                            alt_content = judgment_field[0].get(key, "").strip()
                            if alt_content and len(alt_content) > len(judgment_code):
                                judgment_code = alt_content
                                break

            # 2. 提取原有非必需字段
            visit_account = ""
            visit_account_field = values.get("回访账号", [])
            if visit_account_field and isinstance(visit_account_field[0], dict):
                visit_account = visit_account_field[0].get("user_id", "").strip()

            dedup_value = ""
            dedup_field = values.get("是否需要查重", [])
            if dedup_field and isinstance(dedup_field[0], dict):
                dedup_value = dedup_field[0].get("text", "").strip()

            specific_tags = ""
            specific_tags_field = values.get("特定人群（标签", [])
            if specific_tags_field and isinstance(specific_tags_field[0], dict):
                specific_tags = specific_tags_field[0].get("text", "").strip()

            # 3. 提取个性化任务专属字段
            prompt = ""
            prompt_field = values.get("提示词", [])
            if prompt_field and isinstance(prompt_field[0], dict):
                prompt = prompt_field[0].get("text", "").strip()

            input_param = ""
            input_param_field = values.get("输入参数", [])
            if input_param_field and isinstance(input_param_field[0], dict):
                input_param = input_param_field[0].get("text", "").strip()

            task_type = ""
            task_type_field = values.get("任务类型", [])
            if task_type_field and isinstance(task_type_field[0], dict):
                task_type = task_type_field[0].get("text", "").strip()

            # 4. 调整验证逻辑：保留3个核心必需字段，通用话术可为空
            if not task_name:
                print(f"  第{idx}条规则缺少任务名，跳过")
                continue
            if not date_field:
                print(f"  第{idx}条规则'{task_name}'缺少看群众哪个日期，跳过")
                continue
            if not judgment_code:
                print(f"  第{idx}条规则'{task_name}'缺少判断式，跳过")
                continue

            # 5. 处理check标志
            if dedup_value.lower() in ['是', 'true', '1', 'yes']:
                check_flag = True
            elif dedup_value.lower() in ['否', 'false', '0', 'no']:
                check_flag = False
            else:
                check_flag = "仅一天" not in judgment_code

            # 6. 调整分类逻辑：优先判断个性化任务（允许通用话术为空）
            is_personalized = False
            # 个性化任务条件：提示词、输入参数、任务类型均不为空（通用话术可为空）
            if prompt and input_param and task_type:
                is_personalized = True
                personalized_count += 1
                print(f"  第{idx}条规则'{task_name}'：个性化任务（输入参数：{input_param}，任务类型：{task_type}）")
                # 即使通用话术为空也保留，个性化任务以提示词为准
                task_rules_list.append({
                    "任务名": task_name,
                    "看群众哪个日期": date_field,
                    "沟通话术": talk_script,  # 可为空
                    "判断式": judgment_code,
                    "回访账号": visit_account,
                    "特定人群（标签": specific_tags,
                    "check": check_flag,
                    "是否个性化任务": is_personalized,
                    "提示词": prompt,
                    "输入参数": input_param,
                    "任务类型": task_type
                })
                valid_count += 1
            else:
                # 普通任务：必须有通用话术
                if not talk_script:
                    print(f"  第{idx}条规则'{task_name}'缺少通用话术且不满足个性化任务条件，跳过")
                    continue
                # 普通任务计数
                normal_count += 1
                print(f"  第{idx}条规则'{task_name}'：普通任务")
                task_rules_list.append({
                    "任务名": task_name,
                    "看群众哪个日期": date_field,
                    "沟通话术": talk_script,
                    "判断式": judgment_code,
                    "回访账号": visit_account,
                    "特定人群（标签": specific_tags,
                    "check": check_flag,
                    "是否个性化任务": is_personalized,
                    "提示词": prompt,
                    "输入参数": input_param,
                    "任务类型": task_type
                })
                valid_count += 1

            print(f"  第{idx}条规则'{task_name}'提取成功，check={check_flag}")

        print(f"  成功读取到 {valid_count} 个有效任务规则（普通任务：{normal_count}个，个性化任务：{personalized_count}个）")
        return task_rules_list
    except Exception as e:
        print(f"  查询任务规则表失败: {str(e)}")
        return []


def process_personalized_tasks(config, personalized_task_list):
    """
    完整遵循文档1逻辑的个性化任务处理：
    1. 写入个性化任务表
    2. 智能等待AI生成话术（最长5分钟，每10秒轮询）
    3. 提取话术并添加医院后缀
    4. 返回带话术的完整任务对象（仅成功生成话术的任务）
    """
    # 1. 基础校验
    personalize_config = config.get("personalize")
    if not personalize_config:
        print(f"❌ {config.get('医院', '未知医院')} 缺少personalize配置")
        return []  # 直接返回空列表，不处理任何任务

    hospital_name = config.get("医院", "未知医院")
    hospital_suffix = f"【{hospital_name}家庭医生】"
    print(f"\n=== {hospital_name} 处理个性化任务（{len(personalized_task_list)}条） ===")

    # 2. 构建写入数据
    write_tasks = []
    task_mapping = []  # 存储原始任务信息

    for task in personalized_task_list:
        # 提取参数
        input_param = task.get("输入参数", "")
        task_type = task.get("任务类型", "")
        external_userid = task.get("externalUserid", "")
        personalized_input = task.get("个性化输入参数", {}).get(input_param, "")

        # 处理输入值
        if isinstance(personalized_input, dict):
            input_value = json.dumps(personalized_input, ensure_ascii=False)
        else:
            input_value = str(personalized_input)

        # 构建写入数据
        write_data = {
            "action": "通用写入表单",
            "company": "花都家庭医生",
            "WordList": {
                "docid": config["docid"],
                "sheet_id": personalize_config["tab"],
                "view_id": personalize_config["viewId"],
                "values": {
                    "输入参数": [{"type": "text", "text": input_value}],
                    "任务类型": [{"type": "text", "text": task_type}],
                    "externalUserid": [{"type": "text", "text": external_userid}],
                    "医院名称": [{"type": "text", "text": hospital_name}]
                }
            }
        }
        write_tasks.append(write_data)

        # 存储原始任务引用（用于后续匹配）
        task_mapping.append({
            "input_value": input_value,  # 用于匹配查询结果
            "task_type": task_type,
            "external_userid": external_userid,
            "original_task": task
        })

    # 3. 批量写入个性化任务表
    success_writes = [False] * len(write_tasks)  # 记录每条写入任务是否成功
    for i, write_data in enumerate(write_tasks):
        try:
            json_body = json.dumps(write_data, ensure_ascii=False)
            response = requests.post(
                API_URL,
                headers=HEADERS,
                data=json_body.encode("utf-8")
            )
            response.raise_for_status()
            result = response.json()

            if result.get("success", False):
                print(f"  ✅ 写入成功：{write_data['WordList']['values']['任务类型'][0]['text']}")
                success_writes[i] = True
            else:
                print(f"  ❌ 写入失败：{result.get('errmsg', '未知错误')}")
        except Exception as e:
            print(f"  ❌ 写入异常：{str(e)}")

    # 4. 智能等待AI生成话术（核心逻辑）
    processed_tasks = []  # 存储已处理的任务（已生成话术）
    start_time = time.time()
    max_wait = 300  # 5分钟
    check_interval = 10  # 10秒

    print(f"\n⌛ 开始等待AI生成话术（最长{max_wait // 60}分钟）")

    while time.time() - start_time < max_wait:
        # 构建查询请求
        query_data = {
            "action": "通用查询表单",
            "company": "花都家庭医生",
            "WordList": {
                "docid": config["docid"],
                "sheet_id": personalize_config["tab"],
                "view_id": personalize_config["viewId"],
                "filter": {"医院名称": hospital_name}
            }
        }

        try:
            # 发送查询请求
            json_query = json.dumps(query_data, ensure_ascii=False)
            response = requests.post(
                API_URL,
                headers=HEADERS,
                data=json_query.encode("utf-8")
            )
            response.raise_for_status()
            result = response.json()

            # 处理查询结果
            if isinstance(result.get("data"), list):
                # 遍历查询到的所有记录
                for item in result["data"]:
                    values = item.get("values", {})
                    # 提取关键字段
                    item_input = values.get("输入参数", [{}])[0].get("text", "")
                    item_type = values.get("任务类型", [{}])[0].get("text", "")
                    item_external = values.get("externalUserid", [{}])[0].get("text", "")
                    ai_script = values.get("话术", [{}])[0].get("text", "")

                    # 只处理有话术的任务
                    if ai_script:
                        # 在task_mapping中查找匹配项
                        for idx, mapping in enumerate(task_mapping):
                            # 跳过写入失败的任务
                            if not success_writes[idx]:
                                continue

                            # 检查是否匹配
                            if (mapping["input_value"] == item_input and
                                    mapping["task_type"] == item_type and
                                    mapping["external_userid"] == item_external):

                                # 检查是否已处理过
                                if mapping["original_task"] not in processed_tasks:
                                    # 添加医院后缀
                                    final_script = f"{ai_script}{hospital_suffix}"

                                    # 更新原始任务的话术字段
                                    mapping["original_task"]["话术"] = final_script
                                    processed_tasks.append(mapping["original_task"])

                                    print(f"  ✅ 话术生成：{item_type}（{len(final_script)}字）")

            # 检查是否所有任务都已完成
            if len(processed_tasks) >= sum(success_writes):  # 只与写入成功的任务数比较
                print("✅ 所有任务话术生成完成")
                break

        except Exception as e:
            print(f"  ❌ 查询异常：{str(e)}")

        # 等待下次轮询
        time.sleep(check_interval)

    # 5. 处理超时未完成的任务
    timeout = int(time.time() - start_time)
    if len(processed_tasks) < sum(success_writes):  # 只考虑写入成功的任务
        unfinished = sum(success_writes) - len(processed_tasks)
        print(f"⌛ 等待超时（{timeout}秒），未完成：{unfinished}条")

    # 6. 返回成功生成话术的任务
    return processed_tasks


# 原有函数：check_task_already_sent（未修改）
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


# 原有函数：create_notice_tasks
async def create_notice_tasks(wecom_handler, notices, user_mapping):
    """创建全区通知群发任务（优化版：每个用户创建一个群发任务）"""
    for notice in notices:
        for user_id, external_userids in user_mapping.items():
            if not external_userids:
                print(f"⚠️ 用户 {user_id} 无对应客户，跳过")
                continue

            print(f"📤 准备为 {user_id} 创建群发任务（{len(external_userids)}个客户）")

            # 创建群发任务（传入所有external_userid）
            result = await wecom_handler.create_mass_task(
                external_userid=external_userids,  # 传入列表
                sender=user_id,
                content=notice,
                task_name="全区通知"
            )

            if result["success"]:
                print(f"✅ 群发任务创建成功：{user_id}（msgid: {result['msgid'][:10]}...）")
            else:
                print(f"❌❌ 群发任务创建失败：{result.get('errmsg', '未知错误')}")



async def write_task_to_form_by_category(
        config,
        task_name,
        task_list,
        check_flag,
        wecom_handler  # 新增：接收全局的企业微信处理器
):
    # 移除：重复创建企业微信处理器的代码（避免资源冲突）
    # 保留：配置校验逻辑
    if not config.get("send_task"):
        print(f"错误：缺少SendTask配置，无法写入任务「{task_name}」")
        return False
    if not task_list:
        print(f"任务「{task_name}」列表为空，跳过写入")
        return True

    print(f"\n=== 写入任务「{task_name}」({len(task_list)}个) ===")
    print(f"check标志: {check_flag}")

    # -------------------------- 关键调整：日期格式改为YYYY-MM-DD（今天的日期） --------------------------
    # 生成今天的日期（格式：2024-05-20），而非时间戳
    today_date = datetime.now().strftime("%Y-%m-%d")
    # 若表单需要毫秒时间戳，可保留原逻辑：today_timestamp = str(int(datetime.now().timestamp() * 1000))

    success_count = 0
    total_count = len(task_list)
    successful_tasks = []

    # 移除：原代码中提取全区通知和创建通知任务的代码（避免重复发送）
    # notices = extract_today_notices(config)
    # asyncio.run(create_notice_tasks(wecom_handler, notices, user_mapping))

    # 保留：去重检查提示
    if check_flag:
        print("check=True，将逐条检查沟通任务表进行去重...")
    else:
        print("check=False，跳过去重检查，直接写入")

    for i, task_info in enumerate(task_list, 1):
        required_fields = ["任务名", "externalUserid", "谁加的好友_user_id"]
        if not all(key in task_info for key in required_fields):
            print(f"第{i}个任务信息不完整，缺少{[k for k in required_fields if k not in task_info]}，跳过")
            continue

        external_userid = task_info["externalUserid"]
        friend_user_id = task_info["谁加的好友_user_id"]

        # 保留：去重检查逻辑
        if check_flag:
            if check_task_already_sent(config, task_name, external_userid, friend_user_id):
                print(f"第{i}个任务已存在于沟通任务表，跳过写入")
                continue
        today_timestamp = str(int(datetime.now().timestamp() * 1000))
        # 修复：写入数据结构（日期字段改为YYYY-MM-DD格式，与withoutfastgpt逻辑一致）
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
                    "回访账号": [{"type": "user", "user_id": friend_user_id}],  # 用户类型字段（必对）
                    "externalUserid": [{"type": "text", "text": external_userid}],
                    "任务名": [{"type": "text", "text": task_name}],
                    "话术": [{"type": "text", "text": task_info.get("话术", "")}]
                }
            }
        }

        # 保留：API请求逻辑
        try:
            response = requests.post(API_URL, headers=HEADERS, data=json.dumps(write_data))
            response.raise_for_status()
            result = response.json()
            print("API响应结果:", json.dumps(result, indent=2, ensure_ascii=False))

            if result and result.get("success", False):
                success_count += 1
                print(f"✅ 第{i}个任务写入成功（发送日期：{today_date}）")
                successful_tasks.append({
                    "external_userid": external_userid,
                    "sender": friend_user_id,
                    "content": task_info.get("话术", ""),
                    "task_name": task_name
                })
            else:
                print(f"❌ 第{i}个任务写入失败: {result}")
        except Exception as e:
            print(f"❌ 第{i}个任务处理异常: {e}")

    print(f"\n任务「{task_name}」写入完成，成功: {success_count}/{total_count}")

    # 保留：创建企业微信群发任务（使用传入的全局处理器）
    if successful_tasks:
        print(f"\n开始创建企业微信群发任务 ({len(successful_tasks)}个)")

        async def create_tasks():
            cancel_result = await wecom_handler.cancel_yesterday_tasks()
            print(f"取消昨天任务结果: {cancel_result}")

            for i, task in enumerate(successful_tasks, 1):
                result = await wecom_handler.create_mass_task(
                    external_userid=task["external_userid"],
                    sender=task["sender"],
                    content=task["content"],
                    task_name=task["task_name"]
                )
                if result["success"]:
                    print(f"✅ 第{i}个群发任务创建成功，msgid: {result['msgid']}")
                else:
                    print(f"❌ 第{i}个群发任务创建失败: {result['error']}")

        await create_tasks()  # 直接await，无需重复创建事件循环
    else:
        async def cancel_yesterday():
            cancel_result = await wecom_handler.cancel_yesterday_tasks()
            print(f"取消昨天任务结果: {cancel_result}")

        await cancel_yesterday()

    return success_count > 0

# 原有函数：query_sent_tasks_for_dedup（未修改）
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


# 原有函数：build_yesterday_sent_index（未修改）
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
                candidate = datetime.fromtimestamp(ts / 1000 if ts > 10 ** 12 else ts)
            else:
                d = parse_date(s)
                if d:
                    return d
        elif isinstance(raw, (int, float, str)):
            s = str(raw).strip()
            if s.isdigit():
                ts = int(s)
                candidate = datetime.fromtimestamp(ts / 1000 if ts > 10 ** 12 else ts)
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


# 原有函数：build_interval_sent_index（未修改）
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
                candidate = datetime.fromtimestamp(ts / 1000 if ts > 10 ** 12 else ts)
            else:
                d = parse_date(s)
                if d:
                    return d
        elif isinstance(raw, (int, float, str)):
            s = str(raw).strip()
            if s.isdigit():
                ts = int(s)
                candidate = datetime.fromtimestamp(ts / 1000 if ts > 10 ** 12 else ts)
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


# 核心修改2：query_new_tables改为异步函数，适配write_task_to_form_by_category的异步调用
async def query_new_tables(config_list, wecom_handler):
    """
    处理群众表，改为按任务逐一判断筛选策略：
    - 如果任务规则的回访账号为空，读取全部群众表数据
    - 如果任务规则的回访账号不为空，只筛选该回访账号的数据
    - 新增：区分普通/个性化任务，个性化任务单独处理
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
            is_personalized = task_rule.get("是否个性化任务", False)  # 新增：判断是否为个性化任务
            print(f"\n--- 处理任务：{task_name}（{'个性化任务' if is_personalized else '普通任务'}） ---")

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
                    # 为当前任务提取特定字段（新增：传递任务规则，含个性化配置）
                    extracted_records = extract_specific_fields_for_task(record, task_rule)
                    if not extracted_records:
                        continue
                    # 对提取的记录进行任务匹配
                    for extracted_record in extracted_records:
                        matched_tasks = match_tasks_for_record(extracted_record, [task_rule],hospital_name)
                        task_matched_records.extend(matched_tasks)
                print(f"  {task_name}：匹配到 {len(task_matched_records)} 个有效记录")

                # -------------------------- 新增：区分普通/个性化任务处理 --------------------------
                if task_matched_records:
                    if is_personalized:
                        # 个性化任务：先处理（填充表→等话术→提话术）
                        print(f"  {task_name}：开始处理个性化任务流程")
                        # 校验个性化任务表配置
                        if not config.get("personalize"):
                            print(f"  ❌ {hospital_name} 缺少personalize配置，个性化任务无法处理，跳过")
                            continue
                        # 处理个性化任务（填充表→等3分钟→提话术）
                        processed_records = process_personalized_tasks(config, task_matched_records)
                        # 处理完成后写入沟通任务表（新增wecom_handler参数）
                        if processed_records:
                            check_flag = task_rule.get("check", True)
                            await write_task_to_form_by_category(
                                config,
                                task_name,
                                processed_records,
                                check_flag,
                                wecom_handler  # 传递全局处理器
                            )
                    else:
                        # 普通任务：直接写入沟通任务表（原有逻辑，新增wecom_handler参数）
                        check_flag = task_rule.get("check", True)
                        await write_task_to_form_by_category(
                            config,
                            task_name,
                            task_matched_records,
                            check_flag,
                            wecom_handler  # 传递全局处理器
                        )

            except requests.exceptions.RequestException as e:
                print(f"  {task_name}：API请求失败: {e}")
                continue
            except Exception as e:
                print(f"  {task_name}：处理异常: {e}")
                continue


# 原有函数：get_user_external_user_mapping（未修改）
def get_user_external_user_mapping(config):
    query_params = {
        "action": "通用查询表单",
        "company": "花都家庭医生",
        "WordList": {
            "docid": config["config"]["WordList"]["docid"],
            "sheet_id": config["config"]["WordList"]["sheet_id"],
            "view_id": config["config"]["WordList"]["view_id"]
        }
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(query_params))
        response.raise_for_status()
        result = response.json()
        if not isinstance(result.get("data"), list):
            return {}
        mapping = {}
        for item in result["data"]:
            values = item.get("values", {})
            user_id = values.get("谁加的好友", [{}])[0].get("user_id", "")
            external_userid = values.get("externalUserid", [{}])[0].get("text", "")
            if user_id and external_userid:
                if user_id not in mapping:
                    mapping[user_id] = []
                mapping[user_id].append(external_userid)
        return mapping
    except requests.exceptions.RequestException as e:
        print(f"查询群众表失败: {e}")
        return {}
    except Exception as e:
        print(f"处理群众表数据时发生错误: {e}")
        return {}


# 原有函数：extract_today_notices（补充完整）
def extract_today_notices(master_docid, notice_config):
    """提取当天的全区通知（使用主配置表的docid）"""
    # 关键修改：如果通知配置不存在，直接返回空列表
    if not notice_config:
        print("⚠️ 通知配置不存在，不提取全区通知")
        return []
    today = datetime.now().strftime("%Y-%m-%d")
    if not notice_config:
        print("❌ 缺少通知表配置")
        return []

    # 确保通知表配置是字典类型
    if isinstance(notice_config, dict):
        sheet_id = notice_config.get("sheet_id")
        view_id = notice_config.get("view_id")
    else:
        # 如果配置不是字典，尝试解析
        sheet_id = None
        view_id = None
        if isinstance(notice_config, str):
            try:
                config_dict = json.loads(notice_config)
                sheet_id = config_dict.get("sheet_id")
                view_id = config_dict.get("view_id")
            except:
                pass

    print(f"✅ 使用通知表配置: sheet_id={sheet_id}, view_id={view_id}")

    if not sheet_id or not view_id:
        print("❌ 通知表配置无效，缺少sheet_id或view_id")
        return []

    query_params = {
        "action": "通用查询表单",
        "company": "花都家庭医生",
        "WordList": {
            "docid": master_docid,
            "sheet_id": sheet_id,
            "view_id": view_id
        }
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(query_params))
        response.raise_for_status()
        result = response.json()
        if not isinstance(result.get("data"), list):
            print(f"❌ 全区通知表查询失败，返回数据非列表")
            return []

        notices = []
        for item in result["data"]:
            values = item.get("values", {})
            # 提取应发送日期（兼容字段格式）
            send_date_field = values.get("应发送日期", [{}])
            send_date = send_date_field[0].get("text", "").strip() if (
                    send_date_field and isinstance(send_date_field[0], dict)
            ) else ""
            # 提取通知文本（兼容字段格式）
            notice_text_field = values.get("文本", [{}])
            notice_text = notice_text_field[0].get("text", "").strip() if (
                    notice_text_field and isinstance(notice_text_field[0], dict)
            ) else ""
            # 只保留当天的有效通知
            if send_date == today and notice_text:
                notices.append(notice_text)
                print(f"  ✅ 提取到全区通知：{notice_text[:50]}...")
        return notices
    except requests.exceptions.RequestException as e:
        print(f"❌ 全区通知表API请求失败: {str(e)}")
        return []
    except Exception as e:
        print(f"❌ 处理全区通知数据异常: {str(e)}")
        return []

# -------------------------- 主函数（核心修改：统一流程+异步调用） --------------------------
async def main():
    print("=" * 60)
    print(f"===== 花都家庭医生任务处理程序启动（{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}） =====")
    print("=" * 60)

    # 1. 初始化企业微信处理器（全局唯一，避免重复创建）
    CORPID = "ww6fffc827ac483f35"  # 实际企业ID
    CORPSECRET = "DxTJu-VblBUVmeQHGaEKvtEzXTRHFSgSfbJIfP39okQ"  # 实际密钥
    wecom_handler = None
    try:
        wecom_handler = WeComTaskHandler(CORPID, CORPSECRET)
        print(f"✅ 企业微信处理器初始化完成（CorpID：{CORPID[:10]}...）")
    except Exception as e:
        print(f"❌ 企业微信处理器初始化失败: {str(e)}")
        return

    try:
        # 2. 提取医院配置（从钉钉表读取）
        print("\n" + "-" * 50)
        print("步骤1/4：提取各医院配置信息")
        print("-" * 50)
        master_config, notice_config = get_master_config_from_dingtalk()
        if not master_config:
            print("❌ 致命错误：未获取到主配置表参数，程序终止")
            return

        # 提取主配置表的docid
        master_docid = master_config["WordList"]["docid"]
        print(f"✅ 获取到主配置表docid: {master_docid}")

        # 提取医院配置列表（使用主配置表参数）
        config_list = extract_target_config(master_config)
        if not config_list:
            print("❌ 致命错误：未获取到任何有效医院配置，程序终止")
            return
        print(f"✅ 成功提取 {len(config_list)} 家医院配置")

        # 3. 处理各医院任务（核心业务逻辑）
        print("\n" + "-" * 50)
        print("步骤2/4：处理各医院群众表与任务匹配")
        print("-" * 50)
        await query_new_tables(config_list, wecom_handler)  # 异步调用，传递全局处理器

        # 4. 统一执行全区通知群发（所有任务处理完成后）
        print("\n" + "-" * 50)
        print("步骤3/4：统一执行全区通知群发")
        print("-" * 50)

        # 关键修改：只有当notice_config存在时才处理全区通知
        if not notice_config:
            print("📢📢 未配置通知表信息，跳过全区通知处理")
        else:
            # 使用主配置表的docid和通知表配置查询全区通知
            notices = extract_today_notices(master_docid, notice_config)
            if not notices:
                print("📢📢 未获取到当天的全区通知，跳过群发")
            else:
                # 为每个医院分别创建群发任务
                for config in config_list:
                    hospital_name = config.get("医院", "未知医院")
                    print(f"\n=== 处理 {hospital_name} 的全区通知群发 ===")

                    # 获取该医院的用户映射
                    user_mapping = get_user_external_user_mapping(config)
                    if not user_mapping:
                        print(f"  ❌❌ 未获取到用户与externalUserid映射关系，跳过")
                        continue

                    print(f"  ✅ 获取到 {len(user_mapping)} 个用户映射")
                    await create_notice_tasks(wecom_handler, notices, user_mapping)

        # 5. 清理过期任务（确保昨日任务已失效）
        print("\n" + "-" * 50)
        print("步骤4/4：清理昨日群发任务")
        print("-" * 50)
        cancel_result = await wecom_handler.cancel_yesterday_tasks()
        print(f"📝 昨日任务清理结果：{cancel_result['message']}")
        if "success_count" in cancel_result:
            print(f"   - 总计{cancel_result['total']}个任务，成功失效{cancel_result['success_count']}个")

    except Exception as e:
        print(f"\n" + "=" * 60)
        print(f"❌ 程序运行异常: {str(e)}")
        print("=" * 60)
    finally:
        # 释放资源（关闭企业微信会话）
        if wecom_handler:
            await wecom_handler._close_session()
        print(f"\n" + "=" * 60)
        print(f"===== 程序执行完成（{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}） =====")
        print("=" * 60)


# -------------------------- 程序入口（异步启动） --------------------------
if __name__ == "__main__":
    import time

    # 启动异步主函数（解决"coroutine was never awaited"警告）
    asyncio.run(main())