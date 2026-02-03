import requests
import json
import re
import time
from datetime import datetime, date, timedelta
from total import get_family_doctor_configs  # 导入钉钉配置获取模块
import asyncio
import aiohttp
from typing import Optional, Dict, Any, List
from aiohttp import ClientTimeout
import uuid

# API基础配置（未修改）
API_URL = "https://smallwecom.yesboss.work/smarttable"
HEADERS = {
    "Content-Type": "application/json; charset=utf-8",
    "Accept": "application/json"
}


# 企业微信操作类（未修改，完全保留）
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
        self._session = aiohttp.ClientSession()

    async def _get_access_token(self, session: aiohttp.ClientSession) -> Optional[str]:
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
                    self.token_expires_at = asyncio.get_event_loop().time() + 7100
                    print(f"获取AccessToken成功，有效期7200秒")
                    return self.access_token
                else:
                    print(f"获取AccessToken失败：{result['errmsg']}（错误码：{result['errcode']}）")
                    return None
        except Exception as e:
            print(f"获取AccessToken异常：{str(e)}")
            return None

    async def create_mass_task(self, external_userid: str, sender: str, content: str, task_name: str) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            access_token = await self._get_access_token(session)
            if not access_token:
                return {"success": False, "error": "无法获取AccessToken"}
            payload = {
                "chat_type": "single",
                "external_userid": [external_userid],
                "sender": sender,
                "allow_select": True,
                "text": {
                    "content": f"【{task_name}】\n{content}"
                },
                "attachments": []
            }
            try:
                url = f"{self.mass_url}?access_token={access_token}"
                async with session.post(
                        url,
                        json=payload,
                        timeout=self.timeout
                ) as resp:
                    result = await resp.json()
                    print(f"接口响应: {result}")
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
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        return await self.get_tasks_by_date(yesterday)

    async def get_tasks_by_date(self, target_date: str) -> List[str]:
        try:
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
            cursor = ""
            while True:
                payload = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "limit": 100,
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
                        current_tasks = result.get("group_msg_list", [])
                        all_msgids.extend([task["msgid"] for task in current_tasks])
                        cursor = result.get("next_cursor", "")
                        if not cursor:
                            break
                except Exception as e:
                    print(f"查询任务时发生异常：{str(e)}")
                    break
            print(f"查询到 {target_date} 的群发任务共 {len(all_msgids)} 个")
            return all_msgids

    async def cancel_tasks(self, msgids: List[str]) -> Dict[str, Any]:
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
        msgids = await self.get_yesterday_tasks()
        if not msgids:
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            return {"success": True, "message": f"{yesterday} 没有查询到群发任务"}
        return await self.cancel_tasks(msgids)

    async def _close_session(self):
        if hasattr(self, '_session') and not self._session.closed:
            await self._session.close()
            print("   ✅ 已关闭残留的aiohttp会话")
        return True


# 日期解析函数（未修改）
def parse_date(date_str):
    if not date_str or date_str == "无数据":
        return None
    date_formats = [
        "%Y年%m月%d日", "%Y年%-m月%d日", "%Y年%m月%-d日", "%Y年%-m月%-d日",
        "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"
    ]
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


# -------------------------- 核心修复1：get_master_config_from_dingtalk 完整保留嵌套配置 --------------------------
def get_master_config_from_dingtalk():
    """从钉钉配置获取主配置表参数（完全匹配你的示例格式，分开存储docid、config、notice等）"""
    configs = get_family_doctor_configs()  # 保留原始钉钉配置获取
    if not configs:
        print("❌ 未从钉钉获取到家医任务配置")
        return None
    # 取第一个有效配置（与原逻辑一致）
    config = configs[0]
    print(f"✅ 获取到主配置表基础参数: docid={config['config']['WordList']['docid']}")

    # 完全保留钉钉配置的嵌套结构（匹配你的示例：WordList下含docid、config、total_data等）
    word_list = config["config"]["WordList"]
    return {
        # 1. 接口要求的固定字段（action/company）
        "action": "通用查询表单",
        "company": "花都家庭医生",
        # 2. 完整保留嵌套结构（docid、config、total_data、today_data、week_data，后续可加notice）
        "WordList": {
            "docid": word_list["docid"],  # 主docid（全局共享）
            "config": word_list["config"],  # 子表1：config（含sheet_id/view_id）
            "total_data": word_list.get("total_data"),  # 子表2：total_data（可选，保留）
            "today_data": word_list.get("today_data"),  # 子表3：today_data（可选，保留）
            "week_data": word_list.get("week_data"),    # 子表4：week_data（可选，保留）
            "notice": word_list.get("notice")           # 子表5：notice（新增，用于全区通知，从钉钉配置读取）
        },
        # 3. 本地存储：标记接口需要的核心参数来源（避免后续查询时找不到）
        "_api_param_source": {
            "sheet_id": "config.sheet_id",  # 接口sheet_id来自 WordList.config.sheet_id
            "view_id": "config.view_id"    # 接口view_id来自 WordList.config.view_id
        }
    }


# -------------------------- 核心修复2：extract_target_config 查询接口时仅传核心参数 --------------------------
def extract_target_config():
    """提取各医院配置信息（修复：查询接口时剔除多余嵌套，仅传docid/sheet_id/view_id）"""
    # 1. 获取完整嵌套的全局主配置
    master_config = get_master_config_from_dingtalk()
    if not master_config:
        print("❌ 无法获取主配置表参数")
        return []
    # 单独提取全局配置（用于后续传递给医院配置）
    global_word_list = master_config["WordList"]
    print(f"🔧 全局主配置结构（完全保留嵌套）:")
    print(f"   - docid: {global_word_list['docid']}")
    print(f"   - config子表: {global_word_list['config']}")
    print(f"   - notice子表: {global_word_list.get('notice', '未配置')}")
    print(f"   - total_data子表: {global_word_list.get('total_data', '未配置')}")

    # 2. 构建接口需要的查询参数（仅保留3个核心字段，剔除多余嵌套）
    api_query_params = {
        "action": master_config["action"],
        "company": master_config["company"],
        "WordList": {
            # 仅提取接口需要的核心参数（docid + config子表的sheet_id/view_id）
            "docid": global_word_list["docid"],
            "sheet_id": global_word_list["config"]["sheet_id"],
            "view_id": global_word_list["config"]["view_id"]
        }
    }
    print(f"🔧 接口查询参数（仅核心字段）: {json.dumps(api_query_params['WordList'], ensure_ascii=False)}")

    # 3. 原有逻辑：调用接口查询主配置表（使用修复后的api_query_params）
    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json=api_query_params  # 传给接口的是仅含核心参数的结构，避免400错误
        )
        response.raise_for_status()  # 若接口仍报错，会抛出详细信息（便于排查）
        result = response.json()
        if not isinstance(result.get("data"), list):
            print(f"未获取到有效数据列表，接口返回：{json.dumps(result, ensure_ascii=False)}")
            return []
        config_list = []
        print(f"📋 获取到 {len(result['data'])} 条医院配置记录")

        # 4. 遍历医院配置，提取专属参数（完全保留原有逻辑，补充全局notice配置）
        for idx, item in enumerate(result["data"], 1):
            values = item.get("values", {})
            # 4.1 提取医院名称（原逻辑不变）
            hospital_info = values.get("医院", [])
            hospital_name = hospital_info[0]["text"] if (
                    hospital_info and isinstance(hospital_info[0], dict)
            ) else f"未命名医院_{idx}"
            # 4.2 提取医院专属文档ID（原逻辑不变）
            docid_array = values.get("文档ID", [])
            if not docid_array:
                print(f"【第{idx}条】{hospital_name}：无文档ID配置，跳过")
                continue
            full_doc_text = ""
            for segment in docid_array:
                if isinstance(segment, dict):
                    full_doc_text += segment.get("text", "").strip()
            # 4.3 正则提取医院专属docid（原逻辑不变）
            docid_match = re.search(r'"docid"\s*:\s*"([^"]+)"', full_doc_text)
            hospital_docid = docid_match.group(1) if docid_match else None
            if not hospital_docid:
                print(f"【第{idx}条】{hospital_name}：未提取到docid，跳过")
                continue

            # 4.4 提取医院专属的masses/send_task/task_rules/personalize（原逻辑不变）
            target_info = {"医院": hospital_name, "docid": hospital_docid}
            # 提取masses配置
            masses_match = re.search(
                r'"masses"\s*:\s*{\s*"tab"\s*:\s*"([^"]+)"\s*,\s*"viewId"\s*:\s*"([^"]+)"',
                full_doc_text
            )
            target_info["masses"] = {
                "tab": masses_match.group(1),
                "viewId": masses_match.group(2)
            } if masses_match else None
            # 提取send_task配置
            send_task_match = re.search(
                r'"SendTask"\s*:\s*{\s*"tab"\s*:\s*"([^"]+)"\s*,\s*"viewId"\s*:\s*"([^"]+)"',
                full_doc_text
            )
            target_info["send_task"] = {
                "tab": send_task_match.group(1),
                "viewId": send_task_match.group(2)
            } if send_task_match else None
            # 提取task_rules配置
            task_rules_match = re.search(
                r'"Taskrules"\s*:\s*{\s*"tab"\s*:\s*"([^"]+)"\s*,\s*"viewId"\s*:\s*"([^"]+)"',
                full_doc_text
            )
            target_info["task_rules"] = {
                "tab": task_rules_match.group(1),
                "viewId": task_rules_match.group(2)
            } if task_rules_match else None
            # 提取personalize配置
            personalize_match = re.search(
                r'"Personalize"\s*:\s*{\s*"tab"\s*:\s*"([^"]+)"\s*,\s*"viewId"\s*:\s*"([^"]+)"',
                full_doc_text
            )
            target_info["personalize"] = {
                "tab": personalize_match.group(1),
                "viewId": personalize_match.group(2)
            } if personalize_match else None

            # 4.5 补充：传递全局notice配置（从master_config的WordList.notice获取）
            target_info["notice"] = global_word_list.get("notice")
            if target_info["notice"]:
                print(f"【第{idx}条】{hospital_name}：已加载全局notice配置（sheet_id：{target_info['notice']['sheet_id']}）")
            else:
                print(f"【第{idx}条】{hospital_name}：全局notice配置未找到（需在钉钉配置中添加）")

            # 4.6 验证医院配置完整性（原逻辑不变）
            if target_info["masses"] and target_info["send_task"]:
                config_list.append(target_info)
                print(f"【第{idx}条】{hospital_name}：提取配置成功（docid：{hospital_docid[:10]}...）")
            else:
                missing = []
                if not target_info["masses"]: missing.append("masses")
                if not target_info["send_task"]: missing.append("SendTask")
                print(f"【第{idx}条】{hospital_name}：缺少{','.join(missing)}配置，跳过")

        return config_list
    except requests.exceptions.RequestException as e:
        # 打印详细错误信息（含请求参数），便于排查
        print(f"\nAPI请求失败: {str(e)}")
        print(f"请求参数（接口实际接收）: {json.dumps(api_query_params, ensure_ascii=False, indent=2)}")
        return []
    except Exception as e:
        print(f"处理数据时发生错误: {str(e)}")
        return []


# 查询任务规则表（未修改，完全保留）
def query_task_rules(config):
    if not config.get("task_rules"):
        print("  未配置任务规则表，返回空列表")
        return []
    # 构建任务规则表查询参数（仅传核心字段）
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
        task_rules_list = []
        valid_count = 0
        normal_count = 0
        personalized_count = 0
        for idx, item in enumerate(result["data"], 1):
            values = item.get("values", {})
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
            if not task_name:
                print(f"  第{idx}条规则缺少任务名，跳过")
                continue
            if not date_field:
                print(f"  第{idx}条规则'{task_name}'缺少看群众哪个日期，跳过")
                continue
            if not judgment_code:
                print(f"  第{idx}条规则'{task_name}'缺少判断式，跳过")
                continue
            if dedup_value.lower() in ['是', 'true', '1', 'yes']:
                check_flag = True
            elif dedup_value.lower() in ['否', 'false', '0', 'no']:
                check_flag = False
            else:
                check_flag = "仅一天" not in judgment_code
            is_personalized = False
            if prompt and input_param and task_type:
                is_personalized = True
                personalized_count += 1
                print(f"  第{idx}条规则'{task_name}'：个性化任务（输入参数：{input_param}，任务类型：{task_type}）")
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
            else:
                if not talk_script:
                    print(f"  第{idx}条规则'{task_name}'缺少通用话术且不满足个性化任务条件，跳过")
                    continue
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


# 提取任务字段（未修改，完全保留）
def extract_specific_fields_for_task(record, task_rule):
    values = record.get("values", {})
    external_userid = ""
    external_field = values.get("externalUserid", [])
    if isinstance(external_field, list) and len(external_field) > 0:
        external_userid = external_field[0].get("text", "") if isinstance(external_field[0], dict) else external_field[
            0]
    external_userid = external_userid or "无数据"
    added_by_user_id = ""
    added_by_field = values.get("谁加的好友", [])
    if isinstance(added_by_field, list) and len(added_by_field) > 0:
        added_by_user_id = added_by_field[0].get("user_id", "") if isinstance(added_by_field[0], dict) else ""
    added_by_user_id = added_by_user_id or "无数据"
    date_field_to_extract = task_rule.get("看群众哪个日期", "")
    if not date_field_to_extract:
        print(f"  任务'{task_rule.get('任务名', '')}'没有配置日期字段，跳过")
        return []
    print(f"  当前任务需要提取的日期字段: {date_field_to_extract}")
    input_param = task_rule.get("输入参数", "")
    print(f"  当前任务需要提取的输入参数字段: {input_param}")
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
                date_value = info_dict.get(date_field_to_extract, "").strip() or tags_dict.get(date_field_to_extract,
                                                                                               "").strip() or "无数据"
                if date_value == "无数据":
                    print(f"  跳过第{info_idx}个info对象（日期字段'{date_field_to_extract}'为空）")
                    continue
                personalized_input = {}
                if input_param:
                    if input_param == "json":
                        personalized_input[input_param] = info_obj
                    else:
                        param_field = values.get(input_param, [])
                        if isinstance(param_field, list) and len(param_field) > 0:
                            param_value = param_field[0].get("text", "") if isinstance(param_field[0], dict) else \
                            param_field[0]
                        else:
                            param_value = str(param_field) if param_field else ""
                        personalized_input[input_param] = param_value.strip()
                specific_tags = info_dict.get("其他特定人群标签", "").strip() or tags_dict.get("其他特定人群标签", "").strip() or ""
                current_info = {
                    "externalUserid": external_userid,
                    "谁加的好友_user_id": added_by_user_id,
                    "info对象序号": info_idx,
                    date_field_to_extract: date_value,
                    "其他特定人群标签": specific_tags,
                    "个性化输入参数": personalized_input,
                    "是否个性化任务": task_rule.get("是否个性化任务", False),
                    "提示词": task_rule.get("提示词", ""),
                    "任务类型": task_rule.get("任务类型", "")
                }
                valid_records.append(current_info)
                print(f"  ✅ 第{info_idx}个info对象有效：{date_field_to_extract}='{date_value}'")
        except json.JSONDecodeError:
            print(f"  JSON解析失败: {json_text[:100]}...")
        except Exception as e:
            print(f"  数据处理异常: {str(e)}")
    return valid_records


# 匹配任务（未修改，完全保留）
def match_tasks_for_record(record, task_rules, hospital_name):
    matched_tasks = []
    if not task_rules:
        return matched_tasks
    if isinstance(task_rules, dict):
        rules_iter = task_rules.values()
    elif isinstance(task_rules, (list, tuple)):
        rules_iter = task_rules
    else:
        return matched_tasks
    for task_info in rules_iter:
        date_field = task_info.get("看群众哪个日期", "")
        judgment_code = task_info.get("判断式", "")
        task_name = task_info.get("任务名", "")
        specific_tags_required = task_info.get("特定人群（标签", "").strip()
        if not date_field or not judgment_code or not task_name:
            continue
        date_value = record.get(date_field)
        if not date_value or date_value == "无数据":
            continue
        parsed_date = parse_date(date_value)
        if not parsed_date:
            continue
        if specific_tags_required:
            record_tags = record.get("其他特定人群标签", "")
            if not record_tags:
                print(f"任务'{task_name}'要求特定标签，但记录中无标签信息，跳过")
                continue
            required_tags = [tag.strip() for tag in specific_tags_required.split(",") if tag.strip()]
            tags_matched = all(required_tag in record_tags for required_tag in required_tags)
            if not tags_matched:
                print(f"任务'{task_name}'标签不匹配：要求{required_tags}，记录中有'{record_tags}'，跳过")
                continue
            else:
                print(f"任务'{task_name}'标签匹配成功：要求{required_tags}，记录中有'{record_tags}'")
        try:
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


# 处理个性化任务（未修改，完全保留）
def process_personalized_tasks(config, personalized_task_list):
    """兼容现有表结构的个性化任务处理"""
    # 1. 基础校验
    personalize_config = config.get("personalize")
    if not personalize_config:
        print(f"❌❌ {config.get('医院', '未知医院')} 缺少personalize配置")
        return []

    hospital_name = config.get("医院", "未知医院")
    hospital_suffix = f"【{hospital_name}家庭医生】"
    print(f"\n=== {hospital_name} 处理个性化任务（{len(personalized_task_list)}条） ===")

    # 2. 构建写入数据（兼容现有表结构）
    write_tasks = []
    task_mapping = []  # 存储：输入参数+任务类型 → 原始任务

    for task in personalized_task_list:
        # 获取唯一性标识（使用输入参数+任务类型）
        input_param = task.get("输入参数", "")
        task_type = task.get("任务类型", "")
        unique_key = f"{input_param}|{task_type}|{task['externalUserid']}"

        # 提取参数值
        input_value = task["个性化输入参数"].get(input_param, "")
        if isinstance(input_value, dict):
            input_value = json.dumps(input_value, ensure_ascii=False)

        # 构建写入数据（仅使用现有字段）
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
                    "externalUserid": [{"type": "text", "text": task["externalUserid"]}],
                    "医院名称": [{"type": "text", "text": hospital_name}]
                }
            }
        }
        write_tasks.append(write_data)
        task_mapping.append({
            "unique_key": unique_key,
            "original_task": task,
            "write_time": time.time()  # 记录写入时间
        })

    # 3. 批量写入
    success_writes = []
    for write_data in write_tasks:
        try:
            # UTF-8编码处理
            json_body = json.dumps(write_data, ensure_ascii=False)
            response = requests.post(
                API_URL,
                headers={"Content-Type": "application/json; charset=utf-8"},
                data=json_body.encode("utf-8")
            )
            response.raise_for_status()
            result = response.json()

            if result.get("success"):
                # 使用输入参数+任务类型作为标识
                input_val = write_data["WordList"]["values"]["输入参数"][0]["text"]
                task_type = write_data["WordList"]["values"]["任务类型"][0]["text"]
                print(f"  ✅ 写入成功：{task_type}（输入参数: {input_val[:20]}...）")
                success_writes.append((input_val, task_type))
            else:
                print(f"  ❌❌ 写入失败：{result.get('errmsg', '未知错误')}")
        except Exception as e:
            print(f"  ❌❌ 写入异常：{str(e)}")

    # 4. 等待话术生成（智能查询）
    processed_tasks = []
    start_time = time.time()
    max_wait = 300  # 5分钟
    check_interval = 10  # 10秒

    print(f"\n⌛⌛ 等待AI生成话术（最长{max_wait // 60}分钟）")

    while time.time() - start_time < max_wait:
        try:
            # 查询该医院所有新任务
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

            # UTF-8编码处理
            json_query = json.dumps(query_data, ensure_ascii=False)
            response = requests.post(
                API_URL,
                headers={"Content-Type": "application/json; charset=utf-8"},
                data=json_query.encode("utf-8")
            )
            response.raise_for_status()
            result = response.json()

            # 处理查询结果
            for item in result.get("data", []):
                values = item.get("values", {})
                input_val = values.get("输入参数", [{}])[0].get("text", "")
                task_type = values.get("任务类型", [{}])[0].get("text", "")
                ai_script = values.get("话术", [{}])[0].get("text", "").strip()

                if ai_script:
                    unique_key = f"{input_val}|{task_type}|{values.get('externalUserid', [{}])[0].get('text', '')}"

                    # 查找匹配的原始任务
                    match = next(
                        (m for m in task_mapping if m["unique_key"] == unique_key),
                        None
                    )

                    if match and match["unique_key"] not in [t["unique_key"] for t in processed_tasks]:
                        # 添加医院后缀
                        final_script = f"{ai_script}{hospital_suffix}"
                        match["original_task"]["话术"] = final_script
                        processed_tasks.append(match)
                        print(f"  ✅ 话术生成：{task_type}（{len(final_script)}字）")

            # 完成检查
            if len(processed_tasks) >= len(success_writes):
                print("✅ 所有任务话术生成完成")
                break

        except Exception as e:
            print(f"  ❌❌ 查询异常：{str(e)}")

        time.sleep(check_interval)

    # 5. 返回处理后的任务（仅包含原始任务对象）
    return [t["original_task"] for t in processed_tasks]


# 检查任务是否已发送（未修改，完全保留）
def check_task_already_sent(config, task_name, external_userid, friend_user_id):
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
        for item in result["data"]:
            values = item.get("values", {})
            tn = _get_text(values.get("任务名", []))
            eu = _get_text(values.get("externalUserid", []))
            if tn == task_name and eu == external_userid:
                sent_user_ids = _get_user_ids_from_sent_field(values.get("已发送", []))
                if friend_user_id in sent_user_ids:
                    return True
        return False
    except Exception as e:
        print(f"检查任务发送状态失败: {str(e)}")
        return False


# 写入任务表（移除取消任务逻辑，未修改其他部分）
async def write_task_to_form_by_category(
        config,
        task_name,
        task_list,
        check_flag,
        wecom_handler
):
    if not config.get("send_task"):
        print(f"错误：缺少SendTask配置，无法写入任务「{task_name}」")
        return False
    if not task_list:
        print(f"任务「{task_name}」列表为空，跳过写入")
        return True
    print(f"\n=== 写入任务「{task_name}」({len(task_list)}个) ===")
    print(f"check标志: {check_flag}")
    today_date = datetime.now().strftime("%Y-%m-%d")
    success_count = 0
    total_count = len(task_list)
    successful_tasks = []
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
        if check_flag:
            if check_task_already_sent(config, task_name, external_userid, friend_user_id):
                print(f"第{i}个任务已存在于沟通任务表，跳过写入")
                continue
        today_timestamp = str(int(datetime.now().timestamp() * 1000))
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
                    "回访账号": [{"type": "user", "user_id": friend_user_id}],
                    "externalUserid": [{"type": "text", "text": external_userid}],
                    "任务名": [{"type": "text", "text": task_name}],
                    "话术": [{"type": "text", "text": task_info.get("话术", "")}]
                }
            }
        }
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
    if successful_tasks:
        print(f"\n开始创建企业微信群发任务 ({len(successful_tasks)}个)")

        async def create_tasks():
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

        await create_tasks()
    else:
        print(f"❌ 无成功写入的任务，跳过群发")
    return success_count > 0


# 查询已发送任务（未修改，完全保留）
def query_sent_tasks_for_dedup(config, task_name):
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


# 构建昨日索引（未修改，完全保留）
def build_yesterday_sent_index(config):
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


# 构建区间索引（未修改，完全保留）
def build_interval_sent_index(config, task_rules_mapping):
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
        for item in result["data"]:
            values = item.get("values", {})
            eu = _get_text(values.get("externalUserid", []))
            tn = _get_text(values.get("任务名", []))
            status_text = _get_text(values.get("状态", []))
            send_date = _parse_send_date(values.get("任务发送日期", []))
            visit_account_user_id = _get_user_id(values.get("回访账号", []))
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
            if tn not in task_filter_config:
                continue
            filter_config = task_filter_config[tn]
            start_days = filter_config["start_days"]
            end_days = filter_config["end_days"]
            if start_days == end_days:
                continue
            for sent_user_id in sent_user_ids:
                index.add((eu, tn, visit_account_user_id, sent_user_id, send_date))
    except Exception as e:
        print(f"构建区间已发送索引失败，将不进行区间去重：{str(e)}")
    return index


# 处理群众表（未修改，完全保留）
async def query_new_tables(config_list, wecom_handler):
    if not config_list:
        print("没有可用于查询的配置信息")
        return
    for idx, config in enumerate(config_list, 1):
        hospital_name = config.get("医院", "未知医院")
        print(f"\n===== 处理第{idx}个群众表 =====")
        print(f"医院: {hospital_name}")
        print("\n--- 查询任务规则表 ---")
        task_rules_list = query_task_rules(config)
        if not task_rules_list:
            print("没有有效的任务规则，跳过该医院")
            continue
        print(f"  {hospital_name}：读取到 {len(task_rules_list)} 个任务规则")
        for task_rule in task_rules_list:
            task_name = task_rule.get("任务名", "")
            visit_account = task_rule.get("回访账号", "")
            is_personalized = task_rule.get("是否个性化任务", False)
            print(f"\n--- 处理任务：{task_name}（{'个性化任务' if is_personalized else '普通任务'}） ---")
            query_params = {
                "action": "通用查询表单",
                "company": "花都家庭医生",
                "WordList": {
                    "docid": config["docid"],
                    "sheet_id": config["masses"]["tab"],
                    "view_id": config["masses"]["viewId"]
                }
            }
            if visit_account:
                query_params["WordList"]["filter"] = {
                    "谁加的好友": {"user_id": visit_account}
                }
                print(f"  按回访账号筛选：{visit_account}")
            else:
                print(f"  读取全部群众表记录")
            try:
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
                task_matched_records = []
                for record_idx, record in enumerate(records, 1):
                    extracted_records = extract_specific_fields_for_task(record, task_rule)
                    if not extracted_records:
                        continue
                    for extracted_record in extracted_records:
                        matched_tasks = match_tasks_for_record(extracted_record, [task_rule], hospital_name)
                        task_matched_records.extend(matched_tasks)
                print(f"  {task_name}：匹配到 {len(task_matched_records)} 个有效记录")
                if task_matched_records:
                    if is_personalized:
                        print(f"  {task_name}：开始处理个性化任务流程")
                        if not config.get("personalize"):
                            print(f"  ❌ {hospital_name} 缺少personalize配置，个性化任务无法处理，跳过")
                            continue
                        processed_records = process_personalized_tasks(config, task_matched_records)
                        if processed_records:
                            check_flag = task_rule.get("check", True)
                            await write_task_to_form_by_category(
                                config,
                                task_name,
                                processed_records,
                                check_flag,
                                wecom_handler
                            )
                    else:
                        check_flag = task_rule.get("check", True)
                        await write_task_to_form_by_category(
                            config,
                            task_name,
                            task_matched_records,
                            check_flag,
                            wecom_handler
                        )
            except requests.exceptions.RequestException as e:
                print(f"  {task_name}：API请求失败: {e}")
                continue
            except Exception as e:
                print(f"  {task_name}：处理异常: {e}")
                continue


# 按医院群众表分类用户映射（未修改，完全保留）
def get_user_external_user_mapping(config):
    if not config.get("masses"):
        print(f"❌ {config.get('医院', '未知医院')} 缺少masses配置，无法获取用户映射")
        return {}
    query_params = {
        "action": "通用查询表单",
        "company": "花都家庭医生",
        "WordList": {
            "docid": config["docid"],
            "sheet_id": config["masses"]["tab"],
            "view_id": config["masses"]["viewId"]
        }
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(query_params))
        response.raise_for_status()
        result = response.json()
        if not isinstance(result.get("data"), list):
            print(f"❌ {config.get('医院', '未知医院')} 群众表查询失败，返回数据非列表")
            return {}
        mapping = {}
        for item in result["data"]:
            values = item.get("values", {})
            added_by_field = values.get("谁加的好友", [{}])
            user_id = added_by_field[0].get("user_id", "") if (
                    added_by_field and isinstance(added_by_field[0], dict)
            ) else ""
            external_field = values.get("externalUserid", [{}])
            external_userid = external_field[0].get("text", "") if (
                    external_field and isinstance(external_field[0], dict)
            ) else ""
            if user_id and external_userid:
                if user_id not in mapping:
                    mapping[user_id] = []
                if external_userid not in mapping[user_id]:
                    mapping[user_id].append(external_userid)
        hospital_name = config.get("医院", "未知医院")
        print(f"✅ {hospital_name} 成功获取用户映射：共{len(mapping)}个usrid，{sum(len(v) for v in mapping.values())}个externalusrid")
        return mapping
    except requests.exceptions.RequestException as e:
        print(f"❌ {config.get('医院', '未知医院')} 群众表API请求失败: {str(e)}")
        return {}
    except Exception as e:
        print(f"❌ {config.get('医院', '未知医院')} 处理群众表数据异常: {str(e)}")
        return {}


# 提取当天全区通知（使用全局notice配置，匹配嵌套结构）
def extract_today_notices(config):
    today = datetime.now().strftime("%Y-%m-%d")
    # notice配置从医院配置的notice字段获取（全局嵌套结构）
    notice_config = config.get("notice")
    if not notice_config:
        print(f"❌ {config.get('医院', '未知医院')} 缺少notice配置，跳过全区通知查询")
        return []
    # 构建notice表查询参数（仅核心字段）
    query_params = {
        "action": "通用查询表单",
        "company": "花都家庭医生",
        "WordList": {
            "docid": config["docid"],  # 医院专属docid
            "sheet_id": notice_config["sheet_id"],  # 从notice嵌套中提取sheet_id
            "view_id": notice_config["view_id"]     # 从notice嵌套中提取view_id
        }
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(query_params))
        response.raise_for_status()
        result = response.json()
        if not isinstance(result.get("data"), list):
            print(f"❌ {config.get('医院', '未知医院')} 全区通知表查询失败，返回数据非列表")
            return []
        notices = []
        for item in result["data"]:
            values = item.get("values", {})
            # 提取“应发送日期”（匹配图2的字段名）
            send_date_field = values.get("应发送日期", [{}])
            send_date = send_date_field[0].get("text", "").strip() if (
                    send_date_field and isinstance(send_date_field[0], dict)
            ) else ""
            # 提取“文本”字段（匹配图2的通知内容）
            notice_text_field = values.get("文本", [{}])
            notice_text = notice_text_field[0].get("text", "").strip() if (
                    notice_text_field and isinstance(notice_text_field[0], dict)
            ) else ""
            # 只保留当天的有效通知
            if send_date == today and notice_text:
                notices.append(notice_text)
                print(f"  ✅ {config.get('医院', '未知医院')} 提取到全区通知：{notice_text[:50]}...")
        return notices
    except requests.exceptions.RequestException as e:
        print(f"❌ {config.get('医院', '未知医院')} 全区通知表API请求失败: {str(e)}")
        return []
    except Exception as e:
        print(f"❌ {config.get('医院', '未知医院')} 处理全区通知数据异常: {str(e)}")
        return []


# 全区通知群发（未修改，完全保留）
async def create_notice_tasks(wecom_handler, hospital_config, notices):
    hospital_name = hospital_config.get("医院", "未知医院")
    if not notices:
        print(f"📢 {hospital_name} 无当天全区通知，跳过群发")
        return
    user_mapping = get_user_external_user_mapping(hospital_config)
    if not user_mapping:
        print(f"❌ {hospital_name} 未获取到用户与externalUserid映射，全区通知群发失败")
        return
    print(f"\n=== {hospital_name} 开始处理全区通知群发 ===")
    print(f"  通知数量：{len(notices)}条 | 发送人数量：{len(user_mapping)}个 | 总接收人数量：{sum(len(v) for v in user_mapping.values())}个")

    for notice_idx, notice_content in enumerate(notices, 1):
        print(f"\n--- 处理第{notice_idx}条通知（内容预览：{notice_content[:50]}...） ---")
        for sender_usrid, external_userids in user_mapping.items():
            if not external_userids:
                print(f"  ⚠️  发送人{sender_usrid}无对应客户（externalusrid为空），跳过")
                continue
            print(f"  📤 发送人{sender_usrid}：准备发送给{len(external_userids)}个客户")
            for ext_idx, external_userid in enumerate(external_userids, 1):
                result = await wecom_handler.create_mass_task(
                    external_userid=external_userid,
                    sender=sender_usrid,
                    content=notice_content,
                    task_name="全区通知"
                )
                if result["success"]:
                    print(f"    ✅ 第{ext_idx}个客户{external_userid[:10]}...：群发任务创建成功（msgid：{result['msgid'][:10]}...）")
                else:
                    print(
                        f"    ❌ 第{ext_idx}个客户{external_userid[:10]}...：创建失败（{result['error']}，错误码：{result.get('errcode')}）")
        print(f"\n✅ {hospital_name} 全区通知群发处理完成")

# 主函数（补充完整，确保所有步骤执行）
async def main():
    print("=" * 60)
    print(f"===== 花都家庭医生任务处理程序启动（{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}） =====")
    print("=" * 60)
    # 1. 初始化企业微信处理器（全局唯一）
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
        # 2. 提取医院配置（完整保留嵌套结构，含docid、config、notice等）
        print("\n" + "-" * 50)
        print("步骤1/4：提取各医院配置信息（含全区通知表+群众表）")
        print("-" * 50)
        config_list = extract_target_config()
        if not config_list:
            print("❌ 致命错误：未获取到任何有效医院配置，程序终止")
            return
        print(f"✅ 成功提取 {len(config_list)} 家医院配置（医院列表：{[c['医院'] for c in config_list]}）")

        # 3. 处理各医院业务任务（普通任务+个性化任务）
        print("\n" + "-" * 50)
        print("步骤2/4：处理各医院业务任务（普通+个性化）")
        print("-" * 50)
        await query_new_tables(config_list, wecom_handler)  # 处理业务任务，写入沟通表+创建群发

        # 4. 处理各医院全区通知（不写入沟通表，直接按群众表分类群发）
        print("\n" + "-" * 50)
        print("步骤3/4：处理各医院全区通知群发")
        print("-" * 50)
        for hospital_config in config_list:
            hospital_name = hospital_config.get("医院", "未知医院")
            print(f"\n=== 开始处理{hospital_name}的全区通知 ===")
            # 提取该医院今天的全区通知（使用全局notice配置）
            today_notices = extract_today_notices(hospital_config)
            # 按医院群众表分类群发（发送人=usrid，接收人=对应externalusrid）
            await create_notice_tasks(wecom_handler, hospital_config, today_notices)

        # 5. 统一取消昨日群发任务（所有任务完成后执行，仅一次）
        print("\n" + "-" * 50)
        print("步骤4/4：统一取消昨日所有群发任务")
        print("-" * 50)
        cancel_result = await wecom_handler.cancel_yesterday_tasks()
        print(f"\n📝 昨日群发任务取消结果汇总：")
        print(f"   - 核心消息：{cancel_result.get('message', '无结果')}")
        if "total" in cancel_result and "success_count" in cancel_result:
            print(f"   - 任务总数：{cancel_result['total']}个")
            print(f"   - 成功取消：{cancel_result['success_count']}个")
            print(f"   - 取消失败：{cancel_result['total'] - cancel_result['success_count']}个")
        # 打印部分失败详情（避免日志过长）
        if "details" in cancel_result:
            failed_tasks = {k: v for k, v in cancel_result["details"].items() if not v["success"]}
            if failed_tasks:
                print(f"\n   ⚠️  前3个取消失败任务详情：")
                for msgid, detail in list(failed_tasks.items())[:3]:
                    print(f"     - msgid[{msgid[:10]}...]：{detail['message']}")

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


# 程序入口（异步启动）
if __name__ == "__main__":
    asyncio.run(main())