import requests
import time
import json
import logging
import re
import os
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Set
import urllib.parse

# 配置日志（UTF-8编码，DEBUG级别便于排查）
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('wechat_data_collector.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 钉钉应用配置（保留你的原始配置）
DINGTALK_CONFIG = {
    "app_key": "dingoicseqn2bmdcazpl",
    "app_secret": "hiiqLe8teDkAADlJh9eklgsbtGIvrG8hPJyOC8as04wzG69OGmgaY_vQ_gyKTXEg",
    "base_id": "YndMj49yWjDEYy3ECQwPlLkgJ3pmz5aA",
    "sheet_name": "配置表",
    "operator_id": "jYEXEC84RV3QE3sm0UaeDwiEiE"
}
SMART_TABLE_URL = "https://smallwecom.yesboss.work/smarttable"


class WeChatDataCollector:
    def __init__(self, corp_id: str, corp_secret: str, debug: bool = False):
        self.corp_id = corp_id
        self.corp_secret = corp_secret
        self.access_token = None
        self.users: List[Dict] = []
        self.debug = debug
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json; charset=utf-8'})
        self.dingtalk_access_token = None
        self.region_config = None
        self.customer_table_config = None  # 客户数据表配置
        self.hospital_ids = set()
        self.department_cache = {}  # 批量拉取的部门缓存
        # 新增：目标日期列表
        self.target_dates = [
            datetime(2025, 8, 22)
        ]
        # 存储所有收集到的数据
        self.all_collected_data = []
        # 存储全零数据（四个值均为0）
        self.zero_data_list = []
        # 存储历史用户ID集合
        self.existing_userids: Set[str] = set()
        logger.info("初始化企业微信数据收集器（批量抓取版本，生成CSV文件）")

    # ---------------------- 钉钉相关方法 ----------------------
    def get_dingtalk_access_token(self):
        url = "https://api.dingtalk.com/v1.0/oauth2/accessToken"
        headers = {"Content-Type": "application/json"}
        payload = {
            "appKey": DINGTALK_CONFIG["app_key"],
            "appSecret": DINGTALK_CONFIG["app_secret"]
        }

        try:
            logger.debug(f"请求钉钉令牌 | appKey: {DINGTALK_CONFIG['app_key'][:8]}...")
            response = self.session.post(url, headers=headers, json=payload, timeout=10)
            response_text = response.text
            logger.debug(f"令牌响应 | 状态码: {response.status_code}, 内容: {response_text[:300]}...")

            response.raise_for_status()
            token_data = response.json()
            access_token = token_data.get("accessToken")
            if not access_token:
                error_code = token_data.get("code", "未知")
                error_msg = token_data.get("message", "未知错误")
                logger.error(f"获取令牌失败 | 代码{error_code}, 消息: {error_msg}")
                return False

            self.dingtalk_access_token = access_token
            logger.info(f"获取钉钉令牌成功 | 有效期: {token_data.get('expiresIn', 7200)}秒")
            return True
        except Exception as e:
            logger.error(f"获取令牌异常: {str(e)}")
            return False

    def refresh_dingtalk_token_if_needed(self):
        """检查并刷新钉钉令牌（如果需要）"""
        # 如果没有令牌，直接获取新令牌
        if not self.dingtalk_access_token:
            logger.info("钉钉令牌不存在，尝试获取新令牌")
            return self.get_dingtalk_access_token()

        # 尝试使用现有令牌执行一个简单请求来检查有效性
        test_url = "https://api.dingtalk.com/v1.0/oauth2/accessToken"
        headers = {
            "x-acs-dingtalk-access-token": self.dingtalk_access_token,
            "Content-Type": "application/json"
        }

        try:
            response = self.session.get(test_url, headers=headers, timeout=5)
            if response.status_code == 401:  # 未授权，令牌过期
                logger.warning("钉钉令牌已过期，尝试刷新")
                return self.get_dingtalk_access_token()
            return True
        except Exception:
            # 如果测试请求失败，尝试刷新令牌
            logger.warning("钉钉令牌有效性检查失败，尝试刷新")
            return self.get_dingtalk_access_token()

    def get_ranking_config(self):
        # 先确保钉钉令牌有效
        if not self.dingtalk_access_token:
            if not self.get_dingtalk_access_token():
                logger.error("获取排行榜配置失败: 无法获取钉钉令牌")
                return False

        base_url = "https://api.dingtalk.com/v1.0/notable/bases/"
        full_url = f"{base_url}{DINGTALK_CONFIG['base_id']}/sheets/{urllib.parse.quote(DINGTALK_CONFIG['sheet_name'])}/records"
        headers = {
            "x-acs-dingtalk-access-token": self.dingtalk_access_token,
            "Content-Type": "application/json"
        }
        params = {"maxResults": 100, "operatorId": DINGTALK_CONFIG["operator_id"]}

        max_retries = 3
        token_refresh_attempts = 0

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"请求排行榜配置 | 尝试 {attempt}/{max_retries} | URL: {full_url[:50]}...")
                response = self.session.get(full_url, headers=headers, params=params, timeout=15)

                # 检查令牌过期错误（401 Unauthorized）
                if response.status_code == 401 and token_refresh_attempts < 1:
                    logger.warning("钉钉令牌可能过期，尝试刷新并重试")
                    if self.get_dingtalk_access_token():
                        headers["x-acs-dingtalk-access-token"] = self.dingtalk_access_token
                        token_refresh_attempts += 1
                        continue  # 重试当前请求
                    else:
                        logger.error("刷新令牌失败，无法继续")
                        return False

                # 检查其他错误状态
                response.raise_for_status()

                data = response.json()
                records = data.get("records", [])
                logger.info(f"获取排行榜配置成功 | 共{len(records)}条记录")

                for record in records:
                    fields = record.get("fields", {})
                    if fields.get("任务名称") == "排行榜":
                        config_value = fields.get("通用配置表")
                        if config_value:
                            try:
                                config = json.loads(config_value)
                                word_list = config.get("WordList", {})

                                # 获取地区配置表
                                if "config" in word_list:
                                    self.region_config = {
                                        "docid": word_list.get("docid"),
                                        "sheet_id": word_list["config"].get("sheet_id"),
                                        "view_id": word_list["config"].get("view_id")
                                    }
                                    logger.info(f"找到地区配置表参数: {self.region_config}")

                                # 获取客户数据表配置
                                if "total_data" in word_list:
                                    self.customer_table_config = {
                                        "docid": word_list.get("docid"),
                                        "sheet_id": word_list["total_data"].get("sheet_id"),
                                        "view_id": word_list["total_data"].get("view_id")
                                    }
                                    logger.info(f"找到客户数据表参数: {self.customer_table_config}")
                                    return True
                            except json.JSONDecodeError as e:
                                logger.warning(f"配置解析失败: {str(e)}")

                logger.warning("未找到有效的排行榜配置")
                return False

            except requests.exceptions.RequestException as e:
                if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 401:
                    if token_refresh_attempts < 1:
                        logger.warning("HTTP 401错误，尝试刷新令牌并重试")
                        if self.get_dingtalk_access_token():
                            headers["x-acs-dingtalk-access-token"] = self.dingtalk_access_token
                            token_refresh_attempts += 1
                            continue
                    logger.error("HTTP 401错误且无法刷新令牌")
                    return False

                logger.error(f"请求失败: {str(e)}")
                if attempt < max_retries:
                    sleep_time = 2 ** attempt  # 指数退避策略
                    logger.warning(f"将在 {sleep_time} 秒后重试...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"尝试 {max_retries} 次后仍然失败")
                    return False
            except Exception as e:
                logger.error(f"获取排行榜配置异常: {str(e)}")
                if attempt < max_retries:
                    sleep_time = 2 ** attempt
                    logger.warning(f"将在 {sleep_time} 秒后重试...")
                    time.sleep(sleep_time)
                else:
                    return False

        return False

    def get_region_config_data(self):
        if not self.region_config:
            logger.error("无法获取地区配置表: 缺少参数")
            return False

        docid = self.region_config.get("docid")
        sheet_id = self.region_config.get("sheet_id")
        view_id = self.region_config.get("view_id")
        if not all([docid, sheet_id, view_id]):
            logger.error(f"地区配置表参数不完整 | docid={docid}, sheet_id={sheet_id}, view_id={view_id}")
            return False

        params = {
            "action": "通用查询表单",
            "company": "花都家庭医生",
            "WordList": {
                "docid": docid,
                "sheet_id": sheet_id,
                "view_id": view_id
            }
        }

        max_retries = 3

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"请求地区配置表 | 尝试 {attempt}/{max_retries} | docid: {docid[:10]}...")
                response = self.session.post(SMART_TABLE_URL, json=params, timeout=15)
                result = response.json()

                if not result.get("success") or "data" not in result:
                    error_msg = result.get("message", "未知错误")
                    logger.error(f"获取地区配置表失败 | 消息: {error_msg}")

                    # 检查是否是400错误，可能是参数错误
                    if result.get("code") == 400 and attempt < max_retries:
                        logger.warning("参数可能不正确，尝试检查并修复配置")
                        # 尝试修复配置 - 如果缺少view_id，尝试不带view_id请求
                        if view_id and "view_id" in params["WordList"]:
                            logger.info("尝试不带view_id请求")
                            params["WordList"].pop("view_id")
                            continue

                    # 普通重试逻辑
                    if attempt < max_retries:
                        sleep_time = 2 ** attempt
                        logger.warning(f"将在 {sleep_time} 秒后重试...")
                        time.sleep(sleep_time)
                        continue
                    return False

                records = result["data"]
                logger.info(f"获取地区配置表成功 | 共{len(records)}条记录")

                for record in records:
                    values = record.get("values", {})
                    hospital_id = self._extract_field_value(values.get("医院ID"))
                    if hospital_id:
                        self.hospital_ids.add(hospital_id)

                logger.info(f"获取医院ID成功 | 共{len(self.hospital_ids)}个医院ID")
                return len(self.hospital_ids) > 0

            except requests.exceptions.Timeout:
                logger.warning(f"请求超时，尝试 {attempt}/{max_retries}")
                if attempt < max_retries:
                    sleep_time = 3 * attempt  # 对超时使用更长的等待时间
                    logger.info(f"将在 {sleep_time} 秒后重试...")
                    time.sleep(sleep_time)
                    continue
                logger.error("多次请求超时，无法获取地区配置数据")
                return False

            except Exception as e:
                logger.error(f"获取地区配置表异常: {str(e)}")
                if attempt < max_retries:
                    sleep_time = 2 ** attempt
                    logger.warning(f"将在 {sleep_time} 秒后重试...")
                    time.sleep(sleep_time)
                else:
                    return False

        return False

    # ---------------------- 字段提取工具方法 ----------------------
    def _extract_field_value(self, field_data):
        if not field_data:
            return ""

        if isinstance(field_data, list):
            texts = []
            for item in field_data:
                if isinstance(item, dict):
                    texts.append(item.get("text", item.get("value", "")))
                elif isinstance(item, str):
                    texts.append(item)
            return "\n".join(texts) if texts else ""

        if isinstance(field_data, dict):
            return str(field_data.get("text", field_data.get("value", "")))

        return str(field_data)

    def _extract_number_value(self, field_data):
        value = self._extract_field_value(field_data)
        if not value:
            return 0
        try:
            if isinstance(value, str) and ',' in value:
                value = value.replace(',', '')
            return int(float(value))
        except (ValueError, TypeError):
            numbers = re.findall(r'\d+', str(value))
            return int(numbers[0]) if numbers else 0

    # ---------------------- 核心优化：部门信息获取 ----------------------
    def get_access_token(self) -> bool:
        url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={self.corp_id}&corpsecret={self.corp_secret}"

        try:
            logger.debug(f"请求企微access_token | corp_id: {self.corp_id[:8]}...")
            response = self.session.get(url, timeout=10)
            result = response.json()

            if result.get("errcode") != 0:
                logger.error(f"获取access_token失败: {result.get('errmsg')} (错误码: {result.get('errcode')})")
                return False

            self.access_token = result["access_token"]
            logger.info("获取企微access_token成功")
            # 关键：获取access_token后，立即批量拉取所有部门并填充缓存
            self.get_all_departments()
            return True
        except Exception as e:
            logger.error(f"获取access_token异常: {str(e)}")
            return False

    def get_all_departments(self) -> bool:
        """批量拉取所有部门信息，存入缓存（在get_access_token后立即执行）"""
        if not self.access_token:
            logger.error("批量拉取部门失败：未获取企微access_token")
            return False

        try:
            # 企微批量获取所有部门接口（无需传单个ID，一次返回全量）
            url = f"https://qyapi.weixin.qq.com/cgi-bin/department/list?access_token={self.access_token}"
            response = self.session.get(url, timeout=10)
            result = response.json()

            if result.get("errcode") != 0 or "department" not in result:
                logger.error(f"批量拉取部门失败: {result.get('errmsg')}")
                return False

            # 填充缓存：覆盖原有空缓存，后续所有部门查询优先用这里的数据
            all_depts = result["department"]
            for dept in all_depts:
                dept_id = dept.get("id")
                if dept_id:
                    self.department_cache[dept_id] = {
                        "name": dept.get("name", f"部门{dept_id}"),
                        "parentid": dept.get("parentid", 0)
                    }

            logger.info(f"✅ 批量拉取部门成功 | 共{len(all_depts)}个部门，已存入缓存")
            return True
        except Exception as e:
            logger.error(f"批量拉取部门异常: {str(e)}")
            return False

    def get_department_info(self, dept_id: int) -> Dict[str, Union[str, int]]:
        """优先从批量缓存获取部门信息，缓存未命中才走单ID请求（ fallback ）"""
        # 1. 优先使用批量拉取的缓存（核心优化点）
        if dept_id in self.department_cache:
            return self.department_cache[dept_id]

        # 2. 缓存未命中（批量拉取失败时），才走单ID请求（ fallback ）
        if not self.access_token:
            default_info = {"name": f"部门{dept_id}", "parentid": 0}
            self.department_cache[dept_id] = default_info
            return default_info

        try:
            logger.warning(f"⚠️  部门{dept_id}未在批量缓存中，触发单ID请求（ fallback ）")
            url = f"https://qyapi.weixin.qq.com/cgi-bin/department/get?access_token={self.access_token}&id={dept_id}"
            response = self.session.get(url, timeout=5)
            result = response.json()

            if result.get("errcode") == 0 and "department" in result:
                dept_info = {
                    "name": result["department"].get("name", f"部门{dept_id}"),
                    "parentid": result["department"].get("parentid", 0)
                }
                self.department_cache[dept_id] = dept_info
                return dept_info
            else:
                default_info = {"name": f"部门{dept_id}", "parentid": 0}
                self.department_cache[dept_id] = default_info
                return default_info
        except Exception as e:
            logger.error(f"获取部门{dept_id}信息异常: {str(e)}")
            default_info = {"name": f"部门{dept_id}", "parentid": 0}
            self.department_cache[dept_id] = default_info
            return default_info

    # ---------------------- 部门层级与医院ID匹配 ----------------------
    def get_department_hierarchy(self, dept_id: int) -> List[str]:
        """获取部门完整层级路径（现在优先用批量缓存，无额外请求）"""
        hierarchy = []
        current_dept_id = dept_id
        max_depth = 10  # 防止无限循环

        while max_depth > 0:
            # 优先从批量缓存获取，无额外请求
            dept_info = self.get_department_info(current_dept_id)
            dept_name = dept_info["name"]

            # 避免重复添加（处理可能的循环引用）
            if dept_name in hierarchy:
                break

            hierarchy.insert(0, dept_name)  # 在开头插入，保证层级顺序（如：顶层/中层/底层）

            parent_id = dept_info["parentid"]
            if parent_id == 0:  # 已到达顶层部门
                break

            current_dept_id = parent_id
            max_depth -= 1

        return hierarchy

    def find_hospital_id(self, dept_id: int) -> str:
        """匹配医院ID（依赖批量缓存，无额外请求）"""
        current_dept_id = dept_id
        max_depth = 10

        while max_depth > 0:
            if str(current_dept_id) in self.hospital_ids:
                return str(current_dept_id)

            # 优先从批量缓存获取，无额外请求
            dept_info = self.get_department_info(current_dept_id)
            parent_id = dept_info["parentid"]
            if parent_id == 0:
                return str(current_dept_id)

            current_dept_id = parent_id
            max_depth -= 1

        return str(dept_id)

    # ---------------------- 用户数据获取 ----------------------
    def get_all_users(self) -> bool:
        if not self.access_token:
            logger.error("获取用户列表失败：未获取access_token")
            return False

        logger.info("开始获取所有用户信息")
        url = f"https://qyapi.weixin.qq.com/cgi-bin/user/list?access_token={self.access_token}&department_id=1&fetch_child=1"

        try:
            response = self.session.get(url, timeout=10)
            result = response.json()

            if result.get("errcode") != 0 or "userlist" not in result:
                logger.error(f"获取用户列表失败: {result.get('errmsg')}")
                return False

            user_list = result["userlist"]
            if not user_list:
                logger.warning("企微无用户数据")
                return False

            processed_users = []
            for user in user_list:
                dept_ids = user.get("department", [])
                hospital_ids = set()
                all_dept_ids = set()
                all_dept_hierarchies = []

                for dept_id in dept_ids:
                    # 1. 匹配医院ID（依赖批量缓存，无额外请求）
                    hospital_id = self.find_hospital_id(dept_id)
                    hospital_ids.add(hospital_id)
                    all_dept_ids.add(str(dept_id))

                    # 2. 获取部门层级（依赖批量缓存，无额外请求）
                    hierarchy = self.get_department_hierarchy(dept_id)
                    all_dept_hierarchies.append("/".join(hierarchy))

                user_info = {
                    "name": user.get("name", "未知姓名"),
                    "userid": user["userid"],
                    "department_ids": ",".join(all_dept_ids),
                    "hospital_ids": ",".join(hospital_ids),
                    "department_hierarchies": "/".join(all_dept_hierarchies)  # 格式：顶层/中层/底层
                }
                processed_users.append(user_info)

            self.users = processed_users
            logger.info(f"获取用户列表成功 | 共{len(processed_users)}个用户（部门信息来自批量缓存）")
            return True
        except Exception as e:
            logger.error(f"获取用户列表异常: {str(e)}")
            return False

    # ---------------------- 用户数据统计 ----------------------
    def get_contact_stats(self, userid: str, start_time: int, end_time: int) -> Dict:
        if not self.access_token:
            return {}

        url = f"https://qyapi.weixin.qq.com/cgi-bin/externalcontact/get_user_behavior_data?access_token={self.access_token}"
        data = {
            "userid": [userid],
            "start_time": start_time,
            "end_time": end_time
        }

        try:
            response = self.session.post(url, json=data, timeout=10)
            result = response.json()

            if result.get("errcode") != 0 or "behavior_data" not in result or len(result["behavior_data"]) == 0:
                return {}

            contact_data = result["behavior_data"][0]
            return {
                "new_contact_cnt": self._extract_number_value(contact_data.get("new_contact_cnt", 0)),
                "message_cnt": self._extract_number_value(contact_data.get("message_cnt", 0))
            }
        except Exception as e:
            logger.warning(f"获取联系数据异常: {str(e)}")
            return {}

    def get_group_chat_stats(self, userid: str, day_begin_time: int) -> Dict:
        if not self.access_token:
            return {}

        url = f"https://qyapi.weixin.qq.com/cgi-bin/externalcontact/groupchat/statistic_group_by_day?access_token={self.access_token}"
        data = {
            "day_begin_time": day_begin_time,
            "owner_filter": {
                "userid_list": [userid]
            }
        }

        try:
            response = self.session.post(url, json=data, timeout=10)
            result = response.json()

            if result.get("errcode") != 0 or "items" not in result or len(result["items"]) == 0:
                return {}

            group_data = result["items"][0].get("data", {})
            return {
                "member_total": self._extract_number_value(group_data.get("member_total", 0)),
                "msg_total": self._extract_number_value(group_data.get("msg_total", 0))
            }
        except Exception as e:
            logger.warning(f"获取群聊数据异常: {str(e)}")
            return {}

    # ---------------------- 客户数据表处理（新增） ----------------------
    def get_existing_userids_from_customer_table(self) -> bool:
        """从客户数据表读取所有历史记录的userid"""
        if not self.customer_table_config:
            logger.error("无法读取客户数据表：缺少配置参数")
            return False

        docid = self.customer_table_config.get("docid")
        sheet_id = self.customer_table_config.get("sheet_id")
        view_id = self.customer_table_config.get("view_id")
        if not all([docid, sheet_id]):
            logger.error(f"客户数据表配置不完整 | docid={docid}, sheet_id={sheet_id}")
            return False

        params = {
            "action": "通用查询表单",
            "company": "花都家庭医生",
            "WordList": {
                "docid": docid,
                "sheet_id": sheet_id,
                "view_id": view_id
            }
        }

        try:
            logger.info(f"正在读取客户数据表 | docid: {docid[:10]}...")
            start_time = time.time()
            response = self.session.post(SMART_TABLE_URL, json=params, timeout=30)
            result = response.json()
            elapsed = time.time() - start_time

            if not result.get("success") or "data" not in result:
                logger.error(f"读取客户数据表失败: {result.get('message', '未知错误')}")
                return False

            records = result["data"]
            logger.info(f"✅ 读取客户数据表成功 | 耗时: {elapsed:.2f}秒 | 记录数: {len(records)}")

            # 提取所有userid
            self.existing_userids = set()
            for record in records:
                values = record.get("values", {})
                person_field = values.get("人员", [])

                # 人员字段格式: [{"user_id": "userid123"}]
                for person in person_field:
                    if isinstance(person, dict):
                        userid = person.get("user_id")
                        if userid:
                            self.existing_userids.add(userid)

            logger.info(f"提取到 {len(self.existing_userids)} 个历史userid")
            return True
        except Exception as e:
            logger.error(f"读取客户数据表异常: {str(e)}")
            return False

    # ---------------------- 数据处理 ----------------------
    def process_single_user(self, user: Dict, target_date: datetime):
        """处理单个用户数据，全零数据单独存放"""
        # 计算目标日期的秒级时间戳（用于获取数据）
        date_str = target_date.strftime('%Y-%m-%d')
        timestamp_str = str(int(target_date.timestamp() * 1000))
        target_start = int(target_date.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        target_end = int(target_date.replace(hour=23, minute=59, second=59, microsecond=0).timestamp())

        logger.info(f"处理用户 {user['name']} ({user['userid'][:8]}...)，日期 {date_str}")

        # 获取用户数据
        contact_data = self.get_contact_stats(user["userid"], target_start, target_end)
        group_data = self.get_group_chat_stats(user["userid"], target_start)

        # 提取核心字段
        new_clients = contact_data.get("new_contact_cnt", 0)
        messages_sent = contact_data.get("message_cnt", 0)
        group_members = group_data.get("member_total", 0)
        group_messages = group_data.get("msg_total", 0)

        # 创建数据对象
        user_data = {
            "姓名": user["name"],
            "userid": user["userid"],
            "人员": user["userid"],
            "记录日期": timestamp_str,
            "日期字符串": date_str,  # 便于CSV可读
            "新增客户数": new_clients,
            "发送消息数": messages_sent,
            "截至当天客户群总人数": group_members,
            "截至当天客户群消息总数": group_messages,
            "医院ID": user["hospital_ids"],
            "部门ID": user["department_ids"],
            "所属部门": user["department_hierarchies"]  # 格式：顶层/中层/底层
        }

        # 检查是否为全零数据
        is_all_zero = (
                new_clients == 0 and
                messages_sent == 0 and
                group_members == 0 and
                group_messages == 0
        )

        return user_data, is_all_zero

    # ---------------------- 生成CSV文件 ----------------------
    def generate_csv(self) -> bool:
        """在当前目录下生成以今天日期命名的CSV文件"""
        if not self.all_collected_data:
            logger.warning("没有收集到任何数据，无法生成CSV")
            return False

        try:
            # 确定字段顺序
            field_names = [
                "姓名", "userid", "人员", "记录日期", "日期字符串",
                "新增客户数", "发送消息数", "截至当天客户群总人数", "截至当天客户群消息总数",
                "医院ID", "部门ID", "所属部门"
            ]

            # 获取今天的日期作为文件名
            today = datetime.now().strftime("%Y%m%d")
            output_path = f"wechat_stats_{today}.csv"

            logger.info(f"将在当前目录生成CSV文件: {output_path}")

            # 直接在当前目录写入文件
            with open(output_path, mode='w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=field_names)
                writer.writeheader()

                for record in self.all_collected_data:
                    writer.writerow(record)

            # 获取绝对路径用于日志
            abs_path = os.path.abspath(output_path)
            logger.info(f"✅ CSV文件生成成功 | 路径: {abs_path}")
            logger.info(f"✅ 共写入 {len(self.all_collected_data)} 条记录")
            return True
        except PermissionError:
            logger.error("❌ 文件写入失败：没有写入权限")
            return False
        except Exception as e:
            logger.error(f"❌ 生成CSV文件失败: {str(e)}")
            return False

    # ---------------------- 主流程 ----------------------
    def run(self) -> bool:
        try:
            logger.info("===== 开始执行企业微信数据统计（批量抓取版本） =====")

            # 1. 初始化配置
            if not self.get_dingtalk_access_token():
                logger.error("获取钉钉access_token失败，程序终止")
                return False
            if not self.get_ranking_config():
                logger.error("获取排行榜配置失败，程序终止")
                return False
            if not self.get_region_config_data():
                logger.error("获取地区配置表数据失败，程序终止")
                return False
            if not self.get_access_token():
                logger.error("获取企微access_token失败，程序终止")
                return False

            # 2. 获取用户列表
            if not self.get_all_users():
                logger.error("获取用户列表失败，程序终止")
                return False

            # 3. 处理多个日期
            total_dates = len(self.target_dates)
            total_user = len(self.users)
            collected_count = 0
            skip_count = 0

            for date_idx, target_date in enumerate(self.target_dates, 1):
                date_str = target_date.strftime('%Y-%m-%d')
                logger.info(f"\n===== 处理日期 {date_str} ({date_idx}/{total_dates}) =====")

                # 处理单个日期下的所有用户
                for user_idx, user in enumerate(self.users, 1):
                    logger.info(f"处理进度：日期 {date_idx}/{total_dates}，用户 {user_idx}/{total_user}")

                    # 处理单个用户
                    user_data, is_all_zero = self.process_single_user(user, target_date)

                    if is_all_zero:
                        # 全零数据放入临时列表
                        self.zero_data_list.append(user_data)
                        logger.info(f"用户 {user['name']} 四个核心值全为0，暂存到零值列表")
                        skip_count += 1
                    else:
                        # 非零数据直接加入最终收集列表
                        self.all_collected_data.append(user_data)
                        collected_count += 1
                        logger.info(f"收集到有效数据: {user['name']}")

                    time.sleep(0.5)  # 控制请求频率

            # 4. 读取客户数据表获取历史用户ID
            if not self.get_existing_userids_from_customer_table():
                logger.error("读取客户数据表失败，无法处理零值数据")
            else:
                logger.info(f"准备处理零值数据: {len(self.zero_data_list)}条")

                # 5. 处理全零数据：保留有历史记录的用户
                retained_zero_count = 0
                for zero_data in self.zero_data_list:
                    if zero_data["userid"] in self.existing_userids:
                        # 如果用户有历史记录，保留该零值数据
                        self.all_collected_data.append(zero_data)
                        retained_zero_count += 1
                        logger.info(f"保留零值数据: {zero_data['姓名']} (历史存在)")

                logger.info(f"保留 {retained_zero_count} 条有历史记录的零值数据")
                skip_count -= retained_zero_count  # 调整跳过计数

            # 6. 生成CSV文件
            logger.info(f"\n===== 数据收集完成 =====")
            logger.info(f"总处理量：{total_user}用户 × {total_dates}天 = {total_user * total_dates}条")
            logger.info(f"有效收集：{collected_count}条（非零数据）")
            logger.info(f"保留零值：{len(self.all_collected_data) - collected_count}条（有历史记录）")
            logger.info(f"最终跳过：{skip_count}条（无历史记录的零值数据）")

            if not self.generate_csv():
                logger.error("生成CSV文件失败")
                return False

            logger.info("===== 企业微信数据统计执行完毕 =====")
            return True
        except Exception as e:
            logger.error(f"程序执行异常: {str(e)}", exc_info=True)
            return False
        finally:
            if hasattr(self, 'session'):
                self.session.close()
                logger.info("已关闭网络会话")


# 企业微信配置（使用您的原始配置）
CORPID = "ww6fffc827ac483f35"
CORPSECRET = "DxTJu-VblBUVmeQHGaEKvtEzXTRHFSgSfbJIfP39okQ"
DEBUG_MODE = True

# 实例化并运行收集器
collector = WeChatDataCollector(
    corp_id=CORPID,
    corp_secret=CORPSECRET,
    debug=DEBUG_MODE
)
collector.run()