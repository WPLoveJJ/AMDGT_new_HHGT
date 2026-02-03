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
    """从钉钉获取家庭医生配置（兼容多格式+详细调试）"""
    access_token = get_dingtalk_access_token()
    if not access_token:
        print("❌ 获取钉钉access_token失败")
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

        print(f"钉钉API返回记录数: {len(records)}")
        if records:
            first_record_fields = list(records[0].get("fields", {}).keys())
            print(f"第一条记录的字段列表: {first_record_fields}")

        result = []
        for record_idx, record in enumerate(records, 1):
            fields = record.get("fields", {})
            task_name = fields.get("任务名称", "").strip()

            print(f"\n===== 处理第{record_idx}条记录，任务名称: '{task_name}' =====")
            # 严格匹配任务名称（需与表格中名称一致，如“拉伸大师群发任务”）
            if "拉伸大师群发任务" not in task_name:
                print(f"  任务名称不匹配，跳过该记录")
                continue

            # 调试：打印所有字段原始结构
            print(f"  原始字段数据: {fields}")

            # 提取 corpid（兼容列表、字典、直接字符串）
            corpid = ""
            corpid_field = fields.get("corpid", "")
            if isinstance(corpid_field, list) and len(corpid_field) > 0:
                first_item = corpid_field[0]
                corpid = first_item.get("text", "") if isinstance(first_item, dict) else str(first_item).strip()
            else:
                corpid = str(corpid_field).strip()
            print(f"  提取corpid: '{corpid}'")

            # 提取 corpsecret（同corpid逻辑）
            corpsecret = ""
            corpsecret_field = fields.get("corpsecret", "")
            if isinstance(corpsecret_field, list) and len(corpsecret_field) > 0:
                first_item = corpsecret_field[0]
                corpsecret = first_item.get("text", "") if isinstance(first_item, dict) else str(first_item).strip()
            else:
                corpsecret = str(corpsecret_field).strip()
            print(f"  提取corpsecret: '{corpsecret}'")

            # 提取 company（同corpid逻辑）
            company = ""
            company_field = fields.get("company", "")
            if isinstance(company_field, list) and len(company_field) > 0:
                first_item = company_field[0]
                company = first_item.get("text", "") if isinstance(first_item, dict) else str(first_item).strip()
            else:
                company = str(company_field).strip()
            print(f"  提取company: '{company}'")

            # 提取通用配置表
            config_value = ""
            config_field = fields.get("通用配置表", "")
            if isinstance(config_field, list) and len(config_field) > 0:
                first_item = config_field[0]
                config_value = first_item.get("text", "") if isinstance(first_item, dict) else str(first_item).strip()
            else:
                config_value = str(config_field).strip()
            config_list = parse_multi_json(config_value) if config_value else []
            print(f"  通用配置表原始值: '{config_value}'")

            # 验证配置完整性
            if corpid and corpsecret and company:
                result.append({
                    "record_id": record.get("id"),
                    "region": fields.get("地区", ""),
                    "corpid": corpid,
                    "corpsecret": corpsecret,
                    "company": company,
                    "config": config_list
                })
                print(f"✅ 成功提取有效配置: corpid前8位='{corpid[:8]}...', corpsecret前8位='{corpsecret[:8]}...', company='{company}'")
            else:
                print(f"⚠️ 配置不完整: corpid='{corpid}', corpsecret='{corpsecret}', company='{company}'（跳过）")

        if result:
            print(f"共提取到 {len(result)} 条有效配置")
            return result
        else:
            print("❌❌ 未从钉钉获取到家医任务配置（所有记录均不匹配或配置不完整）")
            return None
    except Exception as e:
        print(f"获取配置时发生异常: {e}")
        return None

def get_master_config_from_dingtalk():
    """从钉钉配置获取主配置表参数和通知表配置"""
    configs = get_family_doctor_configs()
    if not configs:
        print("❌❌ 未从钉钉获取到家医任务配置")
        return None, None, None, None, None

    # 取第一个有效配置
    config = configs[0]

    # 提取参数
    corpid = config.get("corpid", "")
    corpsecret = config.get("corpsecret", "")
    company_value = config.get("company", "")

    # 直接访问WordList结构
    if "config" not in config or not config["config"]:
        print("❌❌ 钉钉返回的配置格式不符合预期")
        print(f"完整配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
        return None, None, corpid, corpsecret, company_value

    # 取第一个配置对象
    config_data = config["config"][0]

    # 检查是否有WordList
    if "WordList" not in config_data:
        print("❌❌ 钉钉返回的配置格式不符合预期，缺少WordList")
        return None, None, corpid, corpsecret, company_value

    wordlist_data = config_data["WordList"]

    # 提取主极速版配置表参数
    master_config = {
        "action": "通用查询表单",
        "company": company_value,
        "WordList": {
            "docid": wordlist_data.get("docid"),
            "sheet_id": wordlist_data.get("config", {}).get("sheet_id"),
            "view_id": wordlist_data.get("config", {}).get("view_id")
        }
    }

    print(f"✅ 获取到主配置表参数: docid={master_config['WordList']['docid']}")

    # 提取通知表配置（不存在时返回None）
    notice_config = wordlist_data.get("notice")
    if notice_config:
        print(f"✅ 获取到通知表配置: sheet_id={notice_config.get('sheet_id')}, view_id={notice_config.get('view_id')}")
    else:
        print("⚠️ 未找到通知表配置，将跳过全区通知处理")

    return master_config, notice_config, corpid, corpsecret, company_value

# -------------------------- 核心改动1：提取医院配置时增加personalize的tab和viewid --------------------------
def extract_target_config(master_config,company_value):
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
            hospital_info = values.get("门店", [])
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
            target_info = {"门店": hospital_name}
            # 用正则表达式提取docid
            docid_match = re.search(r'"docid"\s*:\s*"([^"]+)"', full_doc_text)
            target_info["docid"] = docid_match.group(1) if docid_match else None
            # 用正则表达式提取masses配置
            masses_match = re.search(
                r'"pour"\s*:\s*{\s*"tab"\s*:\s*"([^"]+)"\s*,\s*"viewId"\s*:\s*"([^"]+)"',
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
    """为特定任务提取字段（支持仅标签/仅日期/日期+标签三种类型）"""
    values = record.get("values", {})
    # 提取externalUserid（必选字段，所有任务都需要）
    external_userid = ""
    external_field = values.get("externalUserid", [])
    if isinstance(external_field, list) and len(external_field) > 0:
        external_userid = external_field[0].get("text", "") if isinstance(external_field[0], dict) else str(external_field[0])
    external_userid = external_userid or "无数据"
    if external_userid == "无数据":
        print(f"  跳过：externalUserid为空（无法匹配任何任务）")
        return []

    # 提取谁加的好友_user_id（必选字段，所有任务都需要）
    added_by_user_id = ""
    added_by_field = values.get("谁加的好友", [])
    if isinstance(added_by_field, list) and len(added_by_field) > 0:
        added_by_user_id = added_by_field[0].get("user_id", "") if isinstance(added_by_field[0], dict) else ""
    added_by_user_id = added_by_user_id or "无数据"
    if added_by_user_id == "无数据":
        print(f"  跳过：externalUserid={external_userid}，但“谁加的好友”为空")
        return []

    # -------------------------- 关键修改：日期字段为空时不跳过，仅标记为“无需日期” --------------------------
    # 提取当前任务需要的日期字段（可选：仅标签任务不需要）
    date_field_to_extract = task_rule.get("看群众哪个日期", "").strip()
    if not date_field_to_extract:
        print(f"  当前任务为【仅标签任务】，无需提取日期字段")
        date_value = "无需日期"  # 用特殊值标记，避免后续判断为空
    else:
        print(f"  当前任务需要提取的日期字段: {date_field_to_extract}")
        date_value = "无数据"  # 初始化日期值

    # -------------------------- 新增：获取当前任务的输入参数（个性化任务专属，与日期无关） --------------------------
    input_param = task_rule.get("输入参数", "").strip()
    if input_param:
        print(f"  当前任务需要提取的输入参数字段: {input_param}")

    # 解析JSON字段（核心：无论是否需要日期，都要处理标签和输入参数）
    json_text = ""
    json_field = values.get("json", [])
    if isinstance(json_field, list) and len(json_field) > 0:
        json_text = json_field[0].get("text", "") if isinstance(json_field[0], dict) else str(json_field[0])
    elif isinstance(json_field, str):
        json_text = json_field

    valid_records = []
    if json_text:
        try:
            json_data = json.loads(json_text)
            info_objects = []
            # 兼容JSON数组或单个JSON对象
            if isinstance(json_data, list):
                info_objects = [obj for obj in json_data if isinstance(obj, dict)]
            elif isinstance(json_data, dict):
                info_objects = [json_data]

            for info_idx, info_obj in enumerate(info_objects, 1):
                info_dict = info_obj.get("info", {})
                tags_dict = info_obj.get("tags", {})
                specific_tags = ""  # 标签字段（所有任务都可能用到）

                # -------------------------- 1. 处理日期字段（仅“仅日期”或“日期+标签”任务需要） --------------------------
                if date_field_to_extract:  # 只有配置了日期字段，才提取日期值
                    date_value = info_dict.get(date_field_to_extract, "").strip() or \
                                 tags_dict.get(date_field_to_extract, "").strip() or "无数据"
                    # 仅日期/日期+标签任务：日期为空则跳过当前info对象
                    if date_value == "无数据":
                        print(f"  跳过第{info_idx}个info对象（日期字段'{date_field_to_extract}'为空）")
                        continue

                # -------------------------- 2. 处理标签字段（所有任务都可能用到） --------------------------
                specific_tags = info_dict.get("其他特定人群标签", "").strip() or \
                                tags_dict.get("其他特定人群标签", "").strip() or ""

                # -------------------------- 3. 处理输入参数（个性化任务专属） --------------------------
                personalized_input = {}
                if input_param:
                    if input_param == "json":
                        # 输入参数为json：剥离当前info对象（而非整个数组）
                        personalized_input[input_param] = info_obj
                    else:
                        # 输入参数为普通字段：从values中提取
                        param_field = values.get(input_param, [])
                        param_value = ""
                        if isinstance(param_field, list) and len(param_field) > 0:
                            param_value = param_field[0].get("text", "") if isinstance(param_field[0], dict) else str(param_field[0])
                        personalized_input[input_param] = param_value.strip()

                # -------------------------- 4. 构造有效记录（无论哪种任务类型，都保留核心字段） --------------------------
                current_info = {
                    "externalUserid": external_userid,
                    "谁加的好友_user_id": added_by_user_id,
                    "info对象序号": info_idx,
                    "其他特定人群标签": specific_tags,  # 标签字段（必含）
                    "个性化输入参数": personalized_input,
                    "是否个性化任务": task_rule.get("是否个性化任务", False),
                    "提示词": task_rule.get("提示词", ""),
                    "任务类型": task_rule.get("任务类型", "")
                }
                # 仅“仅日期”或“日期+标签”任务：添加日期字段到记录
                if date_field_to_extract:
                    current_info[date_field_to_extract] = date_value

                valid_records.append(current_info)
                # 日志区分任务类型
                if date_field_to_extract and specific_tags:
                    print(f"  ✅ 第{info_idx}个info对象有效（日期+标签任务）：{date_field_to_extract}='{date_value}'，标签='{specific_tags}'")
                elif date_field_to_extract:
                    print(f"  ✅ 第{info_idx}个info对象有效（仅日期任务）：{date_field_to_extract}='{date_value}'")
                elif specific_tags:
                    print(f"  ✅ 第{info_idx}个info对象有效（仅标签任务）：标签='{specific_tags}'")
                else:
                    print(f"  ✅ 第{info_idx}个info对象有效（无日期无标签任务）")

        except json.JSONDecodeError:
            print(f"  JSON解析失败: {json_text[:100]}...")
        except Exception as e:
            print(f"  数据处理异常: {str(e)}")
    else:
        # 无JSON字段时：直接构造基础记录（适用于无JSON但有标签的场景）
        specific_tags = values.get("其他特定人群标签", [{}])[0].get("text", "").strip() if (
            values.get("其他特定人群标签") and isinstance(values.get("其他特定人群标签")[0], dict)
        ) else ""
        current_info = {
            "externalUserid": external_userid,
            "谁加的好友_user_id": added_by_user_id,
            "info对象序号": 1,
            "其他特定人群标签": specific_tags,
            "个性化输入参数": {},
            "是否个性化任务": task_rule.get("是否个性化任务", False),
            "提示词": task_rule.get("提示词", ""),
            "任务类型": task_rule.get("任务类型", "")
        }
        if date_field_to_extract:
            current_info[date_field_to_extract] = "无JSON字段（仅标签任务无需日期）"
        valid_records.append(current_info)
        print(f"  ✅ 无JSON字段，构造基础记录（标签='{specific_tags}'）")

    return valid_records

def match_tasks_for_record(record, task_rules, hospital_name):
    matched_tasks = []
    if not task_rules:
        return matched_tasks

    # 兼容传入的任务规则为 dict 或 list/tuple
    if isinstance(task_rules, dict):
        rules_iter = task_rules.values()
    elif isinstance(task_rules, (list, tuple)):
        rules_iter = task_rules
    else:
        return matched_tasks

    # 动态执行任务规则中的判断式
    for task_info in rules_iter:
        task_name = task_info.get("任务名", "")
        # 提取所有可能需要的条件
        date_field = task_info.get("看群众哪个日期", "")
        judgment_code = task_info.get("判断式", "")
        specific_tags_required = task_info.get("特定人群（标签", "").strip()

        # 检查是否有日期条件
        has_date_condition = bool(date_field and judgment_code)
        # 检查是否有标签条件
        has_tag_condition = bool(specific_tags_required)

        # 如果既无日期条件也无标签条件，跳过该任务
        if not has_date_condition and not has_tag_condition:
            print(f"任务'{task_name}'既无日期条件也无标签条件，跳过")
            continue

        # 获取对应的日期值（如果有日期条件）
        date_value = ""
        if has_date_condition:
            date_value = record.get(date_field, "")
            if not date_value or date_value == "无数据":
                print(f"任务'{task_name}'日期字段'{date_field}'为空，跳过")
                continue

        # 标签匹配检查（如果有标签条件）
        tags_matched = True
        if has_tag_condition:
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
            else:
                print(f"任务'{task_name}'标签匹配成功：要求{required_tags}，记录中有'{record_tags}'")

        # 日期判断（如果有日期条件）
        date_judgment_passed = True
        if has_date_condition and tags_matched:  # 只有当标签匹配时才判断日期
            try:
                parsed_date = parse_date(date_value)
                if not parsed_date:
                    print(f"任务'{task_name}'日期值解析失败: {date_value}")
                    continue

                # 直接执行判断表达式
                local_namespace = {
                    'check': parsed_date,
                    'datetime': datetime,
                    'timedelta': timedelta,
                    'parse_date': parse_date
                }
                date_judgment_passed = eval(judgment_code, {"__builtins__": {}}, local_namespace)
            except Exception as e:
                print(f"任务'{task_name}'判断式执行失败: {e}")
                continue

        # 只有当所有条件都满足时才添加任务
        if (not has_tag_condition or tags_matched) and (not has_date_condition or date_judgment_passed):
            raw_script = task_info.get("沟通话术", "")
            processed_script = raw_script.replace("_", hospital_name) if not task_info.get(
                "是否个性化任务") else raw_script

            task_obj = {
                "任务名": task_name,
                "externalUserid": record["externalUserid"],
                "谁加的好友_user_id": record["谁加的好友_user_id"],
                "话术": processed_script,
                "check": task_info.get("check", True),
                "是否个性化任务": task_info.get("是否个性化任务", False),
                "提示词": task_info.get("提示词", ""),
                "输入参数": task_info.get("输入参数", ""),
                "任务类型": task_info.get("任务类型", ""),
                "个性化输入参数": record.get("个性化输入参数", {})
            }
            matched_tasks.append(task_obj)
            print(f"✅ 任务'{task_name}'匹配成功")

    return matched_tasks


# -------------------------- 核心改动3：查询任务规则时筛选提示词不为空，提取输入参数、任务类型 --------------------------
def query_task_rules(config,company_value):
    """查询任务规则表，放宽验证规则：只需任务名存在即可"""
    if not config.get("task_rules"):
        print("  未配置任务规则表，返回空列表")
        return []
    query_params = {
        "action": "通用查询表单",
        "company": company_value,
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
            # 1. 提取任务名（唯一必需字段）
            task_name = ""
            task_name_field = values.get("任务名", [])
            if task_name_field and isinstance(task_name_field[0], dict):
                task_name = task_name_field[0].get("text", "").strip()

            # 放宽验证：只需任务名存在
            if not task_name:
                print(f"  第{idx}条规则缺少任务名，跳过")
                continue

            # 2. 提取其他可选字段
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

            # 修改：处理回访账号字段，提取所有用户ID
            visit_accounts = []
            visit_account_field = values.get("回访账号", [])
            if visit_account_field:
                for account in visit_account_field:
                    if isinstance(account, dict):
                        user_id = account.get("user_id", "").strip()
                        if user_id:
                            visit_accounts.append(user_id)

            dedup_value = ""
            dedup_field = values.get("是否需要查重", [])
            if dedup_field and isinstance(dedup_field[0], dict):
                dedup_value = dedup_field[0].get("text", "").strip()

            specific_tags = ""
            specific_tags_field = values.get("特定人群（标签", [])
            if specific_tags_field and isinstance(specific_tags_field[0], dict):
                specific_tags = specific_tags_field[0].get("text", "").strip()

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

            # 4. 处理check标志
            if dedup_value.lower() in ['是', 'true', '1', 'yes']:
                check_flag = True
            elif dedup_value.lower() in ['否', 'false', '0', 'no']:
                check_flag = False
            else:
                check_flag = "仅一天" not in judgment_code if judgment_code else True

            # 5. 调整分类逻辑：优先判断个性化任务（允许通用话术为空）
            is_personalized = False
            # 个性化任务条件：提示词、输入参数、任务类型均不为空（通用话术可为空）
            if prompt and input_param and task_type:
                is_personalized = True
                personalized_count += 1
                print(f"  第{idx}条规则'{task_name}'：个性化任务（输入参数：{input_param}，任务类型：{task_type}）")
                # 即使通用话术为空也保留，个性化任务以提示词为准
                task_rules_list.append({
                    "任务名": task_name,
                    "看群众哪个日期": date_field,  # 可选
                    "沟通话术": talk_script,  # 可选
                    "判断式": judgment_code,  # 可选
                    "回访账号": visit_accounts,  # 修改：改为列表
                    "特定人群（标签": specific_tags,  # 可选
                    "check": check_flag,
                    "是否个性化任务": is_personalized,
                    "提示词": prompt,
                    "输入参数": input_param,
                    "任务类型": task_type
                })
                valid_count += 1
            else:
                # 普通任务：通用话术可为空（放宽）
                print(f"  第{idx}条规则'{task_name}'：普通任务")
                normal_count += 1
                task_rules_list.append({
                    "任务名": task_name,
                    "看群众哪个日期": date_field,  # 可选
                    "沟通话术": talk_script,  # 可选（可为空）
                    "判断式": judgment_code,  # 可选
                    "回访账号": visit_accounts,  # 修改：改为列表
                    "特定人群（标签": specific_tags,  # 可选
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


def process_personalized_tasks(config, personalized_task_list,company_value):
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
        print(f"❌ {config.get('门店', '未知医院')} 缺少personalize配置")
        return []  # 直接返回空列表，不处理任何任务

    hospital_name = config.get("门店", "未知医院")
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
            "company": company_value,
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
            "company": company_value,
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
def check_task_already_sent(config, task_name, external_userid, friend_user_id, company_value):  # 补充company_value
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
            "company": company_value,  # 使用传入的company_value
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
        wecom_handler,  # 全局企业微信处理器
        company_value  # 企业标识
):
    # 1. 基础校验
    if not config.get("send_task"):
        print(f"错误：缺少SendTask配置，无法写入任务「{task_name}」")
        return False
    if not task_list:
        print(f"任务「{task_name}」列表为空，跳过写入")
        return True

    print(f"\n=== 开始处理任务「{task_name}」（共{len(task_list)}条原始记录，逐条创建群发）===")
    print(f"check标志: {check_flag} | 企业标识: {company_value}")
    print(f"⚠️  注意：每条任务独立群发，发送内容仅包含「话术」，不包含任务名\n")

    today_date = datetime.now().strftime("%Y-%m-%d")
    write_success_count = 0  # 表单写入成功计数
    mass_success_count = 0   # 群发任务创建成功计数
    total_processed = 0      # 总处理记录数

    # 2. 遍历每条原始任务，逐条处理（不聚合，独立处理）
    for task_idx, task in enumerate(task_list, 1):
        total_processed += 1
        # 验证单条任务的必需字段（externalUserid、sender、话术）
        required_fields = ["externalUserid", "谁加的好友_user_id", "话术"]
        if not all(key in task for key in required_fields):
            missing = [k for k in required_fields if k not in task]
            print(f"【第{task_idx}/{len(task_list)}条】信息不完整（缺少{missing}），跳过")
            continue

        # 提取单条任务的核心信息（独立ID，不聚合）
        single_external_id = task["externalUserid"]  # 单个外部联系人ID（长度≤64字符）
        sender = task["谁加的好友_user_id"]          # 回访账号（群发sender）
        pure_content = task["话术"]                  # 仅纯话术，无任务名
        print(f"【第{task_idx}/{len(task_list)}条】待处理：sender={sender}，externalUserid={single_external_id}")

        # 3. 去重检查（单条任务独立去重）
        if check_flag:
            is_duplicate = check_task_already_sent(
                config=config,
                task_name=task_name,
                external_userid=single_external_id,
                friend_user_id=sender,
                company_value=company_value
            )
            if is_duplicate:
                print(f"【第{task_idx}/{len(task_list)}条】已存在于沟通任务表，跳过写入和群发\n")
                continue

        # 4. 构建表单写入数据（单条任务独立写入）
        today_timestamp = str(int(datetime.now().timestamp() * 1000))
        write_data = {
            "action": "通用写入表单",
            "company": company_value,
            "WordList": {
                "docid": config["docid"],
                "sheet_id": config["send_task"]["tab"],
                "view_id": config["send_task"]["viewId"],
                "values": {
                    "任务发送日期": today_timestamp,
                    "截止日期": today_timestamp,
                    "回访账号": [{"type": "user", "user_id": sender}],
                    "externalUserid": [{"type": "text", "text": single_external_id}],  # 单个ID写入
                    "任务名": [{"type": "text", "text": task_name}],  # 任务名仅存表，不发送
                    "话术": [{"type": "text", "text": pure_content}]  # 仅写入纯话术
                }
            }
        }

        # 5. 写入沟通任务表（核心修改：用json参数替代data，自动处理UTF-8编码）
        write_success = False
        try:
            # 关键修改：删除data=json.dumps(...)，改用json=write_data
            # requests会自动将write_data序列化为JSON，并按UTF-8编码发送
            response = requests.post(
                API_URL,
                headers=HEADERS,
                json=write_data,  # 改用json参数，自动处理中文编码
                timeout=10  # 超时保护，避免请求挂起
            )
            response.raise_for_status()
            result = response.json()

            if result.get("success", False):
                write_success = True
                write_success_count += 1
                print(f"【第{task_idx}/{len(task_list)}条】表单写入成功")
            else:
                err_msg = result.get("errmsg", "未知错误")
                print(f"【第{task_idx}/{len(task_list)}条】表单写入失败：{err_msg}\n")
        except Exception as e:
            err_detail = str(e)
            if hasattr(e, "response") and e.response:
                err_detail += f" | 状态码：{e.response.status_code} | 响应内容：{e.response.text[:300]}"
            print(f"【第{task_idx}/{len(task_list)}条】表单处理异常：{err_detail}\n")
            continue

        # 6. 写入成功后，逐条创建群发任务（核心：单条任务独立群发）
        if write_success:
            print(f"【第{task_idx}/{len(task_list)}条】开始创建独立群发任务")
            # 企业微信API要求 external_userid 为列表，即使只有一个ID
            external_list = [single_external_id]
            # 调用群发接口（逐条请求，不批量）
            mass_result = await wecom_handler.create_mass_task(
                external_userid=external_list,  # 单个ID的列表（符合API格式）
                sender=sender,
                content=pure_content,         # 仅纯话术，无任务名
                task_name=task_name          # 任务名仅用于企业微信内部管理（不发送给用户）
            )

            if mass_result["success"]:
                mass_success_count += 1
                msgid = mass_result["msgid"]
                print(f"【第{task_idx}/{len(task_list)}条】群发任务创建成功（msgid：{msgid[:10]}...）")
            else:
                err_msg = mass_result.get("error", "未知错误")
                err_code = mass_result.get("errcode", "未知")
                print(f"【第{task_idx}/{len(task_list)}条】群发任务创建失败（错误码：{err_code}）：{err_msg}")
        print(f"【第{task_idx}/{len(task_list)}条】处理完成\n")

    # 7. 清理昨日任务（无论群发结果，确保残留任务失效）
    cancel_result = await wecom_handler.cancel_yesterday_tasks()

    # 8. 最终总结
    print("=" * 60)
    print(f"任务「{task_name}」全流程总结")
    print(f"总处理记录数：{total_processed}/{len(task_list)}")
    print(f"表单写入成功数：{write_success_count}/{total_processed}")
    print(f"群发任务成功数：{mass_success_count}/{write_success_count}")
    print(f"昨日任务清理结果：{cancel_result.get('message', '未知')}")
    print(f"关键说明：所有发送内容仅包含「话术」，未包含任务名")
    print("=" * 60)

    return write_success_count > 0

# 原有函数：query_sent_tasks_for_dedup（未修改）
def query_sent_tasks_for_dedup(config, task_name,company_value):
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
            "company": company_value,
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
def build_yesterday_sent_index(config,company_value):
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
            "company": company_value,
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
def build_interval_sent_index(config, task_rules_mapping,company_value):
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
            "company": company_value,
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
async def query_new_tables(config_list, wecom_handler, company_value):
    """处理群众表，删除externalUserid合并逻辑，保留单个ID列表"""
    if not config_list:
        print("没有可用于查询的配置信息")
        return
    for idx, config in enumerate(config_list, 1):
        hospital_name = config.get("门店", "未知医院")
        print(f"\n===== 处理第{idx}个群众表 =====")
        print(f"医院: {hospital_name}")
        print("\n--- 查询任务规则表 ---")
        task_rules_list = query_task_rules(config, company_value)
        if not task_rules_list:
            print("没有有效的任务规则，跳过该医院")
            continue
        print(f"  {hospital_name}：读取到 {len(task_rules_list)} 个任务规则")

        for task_rule in task_rules_list:
            task_name = task_rule.get("任务名", "")
            visit_accounts = task_rule.get("回访账号", [])
            is_personalized = task_rule.get("是否个性化任务", False)
            print(f"\n--- 处理任务：{task_name}（{'个性化任务' if is_personalized else '普通任务'}） ---")

            # 构建循环分页查询参数（逻辑不变）
            query_params = {
                "action": "循环通用查询表单",
                "company": company_value,
                "WordList": {
                    "docid": config["docid"],
                    "sheet_id": config["masses"]["tab"],
                    "view_id": config["masses"]["viewId"],
                    "offset": 0
                }
            }
            if visit_accounts:
                query_params["WordList"]["filter"] = {
                    "谁加的好友": {"user_id": visit_accounts}
                }
                print(f"  按回访账号筛选：{', '.join(visit_accounts)}")
            else:
                print(f"  批量读取全部群众表记录（循环分页）")

            # 循环分页读取数据（逻辑不变）
            all_records = []
            page = 1
            try:
                while True:
                    response = requests.post(
                        API_URL,
                        headers=HEADERS,
                        data=json.dumps(query_params),
                        timeout=10  # 新增超时，避免挂起
                    )
                    response.raise_for_status()
                    result = response.json()
                    current_page_records = result.get("data", [])
                    if not isinstance(current_page_records, list):
                        print(f"  第{page}页数据格式错误，终止读取")
                        break
                    page_size = len(current_page_records)
                    all_records.extend(current_page_records)
                    print(f"  已读取第{page}页，累计{len(all_records)}条记录（本页{page_size}条）")
                    if page_size == 0:
                        print(f"  所有数据读取完毕，共{len(all_records)}条记录")
                        break
                    query_params["WordList"]["offset"] += page_size
                    page += 1
            except requests.exceptions.RequestException as e:
                print(f"  分页查询异常：{e}（已读取{len(all_records)}条有效记录）")
                if hasattr(e, 'response') and e.response:
                    print(f"    响应状态码：{e.response.status_code}")
                    print(f"    响应内容：{e.response.text[:500]}")
            except Exception as e:
                print(f"  数据处理异常：{e}（已读取{len(all_records)}条有效记录）")

            # 处理所有读取到的记录（逻辑不变）
            if not all_records:
                print(f"  {task_name}：无群众表记录，跳过")
                continue
            print(f"  {task_name}：开始处理{len(all_records)}条群众表记录")

            task_matched_records = []
            for record_idx, record in enumerate(all_records, 1):
                extracted_records = extract_specific_fields_for_task(record, task_rule)
                if not extracted_records:
                    continue
                for extracted_record in extracted_records:
                    matched_tasks = match_tasks_for_record(extracted_record, [task_rule], hospital_name)
                    task_matched_records.extend(matched_tasks)
            print(f"  {task_name}：匹配到 {len(task_matched_records)} 个有效记录")

            # -------------------------- 核心修改：删除“合并externalUserid”逻辑 --------------------------
            if task_matched_records:
                grouped_by_account = {}
                for record in task_matched_records:
                    account_id = record["谁加的好友_user_id"]
                    if account_id not in grouped_by_account:
                        grouped_by_account[account_id] = []
                    grouped_by_account[account_id].append(record)

                for account_id, account_records in grouped_by_account.items():
                    # 删除合并逻辑：不拼接externalUserid，保留原始单个ID的列表
                    print(f"  处理回访账号 {account_id} 的 {len(account_records)} 个任务")

                    # -------------------------- 删除以下合并代码块 --------------------------
                    # if len(account_records) > 1:
                    #     external_userids = [r["externalUserid"] for r in account_records]
                    #     combined_external_userid = ",".join(external_userids)
                    #     combined_record = account_records[0].copy()
                    #     combined_record["externalUserid"] = combined_external_userid
                    #     combined_record["合并记录数"] = len(account_records)
                    #     account_records = [combined_record]
                    #     print(f"    合并 {len(external_userids)} 个externalUserid: {combined_external_userid}")
                    # -------------------------------------------------------------------

                    # 个性化/普通任务处理逻辑不变（直接使用account_records列表）
                    if is_personalized:
                        print(f"  {task_name}：开始处理个性化任务流程")
                        if not config.get("personalize"):
                            print(f"  ❌ {hospital_name} 缺少personalize配置，个性化任务无法处理，跳过")
                            continue
                        processed_records = process_personalized_tasks(config, account_records, company_value)
                        if processed_records:
                            check_flag = task_rule.get("check", True)
                            await write_task_to_form_by_category(
                                config, task_name, processed_records, check_flag, wecom_handler, company_value
                            )
                    else:
                        check_flag = task_rule.get("check", True)
                        await write_task_to_form_by_category(
                            config, task_name, account_records, check_flag, wecom_handler, company_value
                        )

# 原有函数：get_user_external_user_mapping（未修改）
def get_user_external_user_mapping(config,company_value):
    query_params = {
        "action": "通用查询表单",
        "company": company_value,
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
def extract_today_notices(master_docid, notice_config,company_value):
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
        "company": company_value,
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

    # 1. 首先获取主配置
    master_config, notice_config, corpid, corpsecret, company_value = get_master_config_from_dingtalk()

    # 检查是否成功获取到必要配置
    if not all([corpid, corpsecret, company_value]):
        print("❌ 无法获取必要的企业微信配置参数，程序终止")
        return

    # 2. 初始化企业微信处理器
    try:
        wecom_handler = WeComTaskHandler(corpid, corpsecret)
        print(f"✅ 企业微信处理器初始化完成（CorpID：{corpid[:10]}...）")
    except Exception as e:
        print(f"❌ 企业微信处理器初始化失败: {str(e)}")
        return

    # 3. 提取医院配置
    print("\n" + "-" * 50)
    print("步骤1/4：提取各医院配置信息")
    print("-" * 50)

    # 检查主配置是否有效
    if not master_config:
        print("❌ 未获取到主配置表参数")
        return

    # 提取主配置表的docid
    master_docid = master_config["WordList"]["docid"]
    print(f"✅ 获取到主配置表docid: {master_docid}")

    # 提取医院配置列表
    config_list = extract_target_config(master_config, company_value)
    if not config_list:
        print("❌ 未获取到任何有效医院配置")
        return
    print(f"✅ 成功提取 {len(config_list)} 家医院配置")

    # 4. 处理各医院任务
    print("\n" + "-" * 50)
    print("步骤2/4：处理各医院群众表与任务匹配")
    print("-" * 50)
    await query_new_tables(config_list, wecom_handler, company_value)

    # 5. 处理全区通知
    print("\n" + "-" * 50)
    print("步骤3/4：统一执行全区通知群发")
    print("-" * 50)

    if notice_config:
        notices = extract_today_notices(master_docid, notice_config, company_value)
        if notices:
            for config in config_list:
                hospital_name = config.get("医院", "未知医院")
                print(f"\n=== 处理 {hospital_name} 的全区通知群发 ===")
                user_mapping = get_user_external_user_mapping(config, company_value)
                if user_mapping:
                    await create_notice_tasks(wecom_handler, notices, user_mapping)
                else:
                    print(f"  ❌ 未获取到用户映射关系，跳过")
        else:
            print("📢 未获取到当天的全区通知，跳过群发")
    else:
        print("📢 未配置通知表信息，跳过全区通知处理")

    # 6. 清理过期任务
    print("\n" + "-" * 50)
    print("步骤4/4：清理昨日群发任务")
    print("-" * 50)
    cancel_result = await wecom_handler.cancel_yesterday_tasks()
    print(f"📝 昨日任务清理结果：{cancel_result.get('message', '未知结果')}")

    if "success_count" in cancel_result:
        print(f"   - 总计{cancel_result['total']}个任务，成功失效{cancel_result['success_count']}个")

    print(f"\n" + "=" * 60)
    print(f"===== 程序执行完成（{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}） =====")
    print("=" * 60)

# -------------------------- 程序入口（异步启动） --------------------------
if __name__ == "__main__":
    import time

    # 启动异步主函数（解决"coroutine was never awaited"警告）
    asyncio.run(main())