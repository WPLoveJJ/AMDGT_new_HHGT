import requests
import time
import json
import logging
import re
import os
import csv
import sys
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Set, Tuple
import urllib.parse

# ======================== 配置部分 ========================
# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wechat_data_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 企业微信配置
CORPID = "ww6fffc827ac483f35"
CORPSECRET = "DxTJu-VblBUVmeQHGaEKvtEzXTRHFSgSfbJIfP39okQ"

# 钉钉应用配置
DINGTALK_CONFIG = {
    "app_key": "dingoicseqn2bmdcazpl",
    "app_secret": "hiiqLe8teDkAADlJh9eklgsbtGIvrG8hPJyOC8as04wzG69OGmgaY_vQ_gyKTXEg",
    "base_id": "YndMj49yWjDEYy3ECQwPlLkgJ3pmz5aA",
    "sheet_name": "配置表",
    "operator_id": "jYEXEC84RV3QE3sm0UaeDwiEiE"
}
SMART_TABLE_URL = "https://smallwecom.yesboss.work/smarttable"


# ======================== 数据抓取部分 ========================
class WeChatDataCollector:
    def __init__(self, corp_id: str, corp_secret: str):
        self.corp_id = corp_id
        self.corp_secret = corp_secret
        self.access_token = None
        self.users: List[Dict] = []
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json; charset=utf-8'})
        self.dingtalk_access_token = None
        self.region_config = None
        self.customer_table_config = None
        self.hospital_ids = set()
        self.department_cache = {}
        self.target_dates = [datetime.now() - timedelta(days=1)]
        self.all_collected_data = []
        self.zero_data_list = []
        self.existing_userids: Set[str] = set()
        logger.info("初始化企业微信数据收集器")

    def get_dingtalk_access_token(self, max_retries=3, retry_delay=2):
        url = "https://api.dingtalk.com/v1.0/oauth2/accessToken"
        headers = {"Content-Type": "application/json"}
        payload = {
            "appKey": DINGTALK_CONFIG["app_key"],
            "appSecret": DINGTALK_CONFIG["app_secret"]
        }

        for attempt in range(max_retries):
            try:
                response = self.session.post(url, headers=headers, json=payload, timeout=10)
                response.raise_for_status()
                token_data = response.json()
                access_token = token_data.get("accessToken")

                if access_token:
                    self.dingtalk_access_token = access_token
                    return True

            except Exception as e:
                logger.error(f"获取钉钉令牌异常 (尝试 {attempt + 1}/{max_retries}): {str(e)}")

            if attempt < max_retries - 1:
                logger.info(f"{retry_delay}秒后重试...")
                time.sleep(retry_delay)

        return False

    def get_ranking_config(self):
        if not self.dingtalk_access_token:
            logger.error("无法获取配置: 缺少钉钉访问令牌")
            return False

        base_url = "https://api.dingtalk.com/v1.0/notable/bases/"
        full_url = f"{base_url}{DINGTALK_CONFIG['base_id']}/sheets/{urllib.parse.quote(DINGTALK_CONFIG['sheet_name'])}/records"
        headers = {
            "x-acs-dingtalk-access-token": self.dingtalk_access_token,
            "Content-Type": "application/json"
        }
        params = {"maxResults": 100, "operatorId": DINGTALK_CONFIG["operator_id"]}

        try:
            response = self.session.get(full_url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            records = data.get("records", [])

            for record in records:
                fields = record.get("fields", {})
                task_name = fields.get("任务名称", "")
                # 修改：只要任务名称包含"排行榜"就匹配
                if "排行榜" in task_name:
                    config_value = fields.get("通用配置表")
                    if config_value:
                        try:
                            config = json.loads(config_value)
                            word_list = config.get("WordList", {})

                            if "config" in word_list:
                                self.region_config = {
                                    "docid": word_list.get("docid"),
                                    "sheet_id": word_list["config"].get("sheet_id"),
                                    "view_id": word_list["config"].get("view_id")
                                }

                            if "total_data" in word_list:
                                self.customer_table_config = {
                                    "docid": word_list.get("docid"),
                                    "sheet_id": word_list["total_data"].get("sheet_id"),
                                    "view_id": word_list["total_data"].get("view_id")
                                }
                                return True
                        except json.JSONDecodeError:
                            pass

            logger.warning("未找到有效的排行榜配置")
            return False
        except Exception as e:
            logger.error(f"获取排行榜配置失败: {str(e)}")
            return False

    def get_region_config_data(self):
        if not self.region_config:
            return False

        docid = self.region_config.get("docid")
        sheet_id = self.region_config.get("sheet_id")
        view_id = self.region_config.get("view_id")
        if not all([docid, sheet_id, view_id]):
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
            response = self.session.post(SMART_TABLE_URL, json=params, timeout=15)
            result = response.json()
            if not result.get("success") or "data" not in result:
                return False

            records = result["data"]
            for record in records:
                values = record.get("values", {})
                hospital_id = self._extract_field_value(values.get("医院ID"))
                if hospital_id:
                    self.hospital_ids.add(hospital_id)

            return len(self.hospital_ids) > 0
        except Exception:
            return False

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

    def get_access_token(self) -> bool:
        url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={self.corp_id}&corpsecret={self.corp_secret}"
        try:
            response = self.session.get(url, timeout=10)
            result = response.json()
            if result.get("errcode") != 0:
                return False

            self.access_token = result["access_token"]
            self.get_all_departments()
            return True
        except Exception:
            return False

    def get_all_departments(self) -> bool:
        if not self.access_token:
            return False

        try:
            url = f"https://qyapi.weixin.qq.com/cgi-bin/department/list?access_token={self.access_token}"
            response = self.session.get(url, timeout=10)
            result = response.json()
            if result.get("errcode") != 0 or "department" not in result:
                return False

            all_depts = result["department"]
            for dept in all_depts:
                dept_id = dept.get("id")
                if dept_id:
                    self.department_cache[dept_id] = {
                        "name": dept.get("name", f"部门{dept_id}"),
                        "parentid": dept.get("parentid", 0)
                    }
            return True
        except Exception:
            return False

    def get_department_info(self, dept_id: int) -> Dict[str, Union[str, int]]:
        if dept_id in self.department_cache:
            return self.department_cache[dept_id]

        if not self.access_token:
            default_info = {"name": f"部门{dept_id}", "parentid": 0}
            self.department_cache[dept_id] = default_info
            return default_info

        try:
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
        except Exception:
            default_info = {"name": f"部门{dept_id}", "parentid": 0}
            self.department_cache[dept_id] = default_info
            return default_info

    def get_department_hierarchy(self, dept_id: int) -> List[str]:
        hierarchy = []
        current_dept_id = dept_id
        max_depth = 10

        while max_depth > 0:
            dept_info = self.get_department_info(current_dept_id)
            dept_name = dept_info["name"]
            if dept_name in hierarchy:
                break

            hierarchy.insert(0, dept_name)
            parent_id = dept_info["parentid"]
            if parent_id == 0:
                break

            current_dept_id = parent_id
            max_depth -= 1

        return hierarchy

    def find_hospital_id(self, dept_id: int) -> str:
        current_dept_id = dept_id
        max_depth = 10

        while max_depth > 0:
            if str(current_dept_id) in self.hospital_ids:
                return str(current_dept_id)

            dept_info = self.get_department_info(current_dept_id)
            parent_id = dept_info["parentid"]
            if parent_id == 0:
                return str(current_dept_id)

            current_dept_id = parent_id
            max_depth -= 1

        return str(dept_id)

    def get_all_users(self) -> bool:
        if not self.access_token:
            return False

        url = f"https://qyapi.weixin.qq.com/cgi-bin/user/list?access_token={self.access_token}&department_id=1&fetch_child=1"

        try:
            response = self.session.get(url, timeout=10)
            result = response.json()
            if result.get("errcode") != 0 or "userlist" not in result:
                return False

            user_list = result["userlist"]
            if not user_list:
                return False

            processed_users = []
            for user in user_list:
                dept_ids = user.get("department", [])
                hospital_ids = set()
                all_dept_ids = set()
                all_dept_hierarchies = []

                for dept_id in dept_ids:
                    hospital_id = self.find_hospital_id(dept_id)
                    hospital_ids.add(hospital_id)
                    all_dept_ids.add(str(dept_id))
                    hierarchy = self.get_department_hierarchy(dept_id)
                    all_dept_hierarchies.append("/".join(hierarchy))

                user_info = {
                    "name": user.get("name", "未知姓名"),
                    "userid": user["userid"],
                    "department_ids": ",".join(all_dept_ids),
                    "hospital_ids": ",".join(hospital_ids),
                    "department_hierarchies": "/".join(all_dept_hierarchies)
                }
                processed_users.append(user_info)

            self.users = processed_users
            return True
        except Exception:
            return False

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
        except Exception:
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
        except Exception:
            return {}

    def get_existing_userids_from_customer_table(self) -> bool:
        if not self.customer_table_config:
            return False

        docid = self.customer_table_config.get("docid")
        sheet_id = self.customer_table_config.get("sheet_id")
        view_id = self.customer_table_config.get("view_id")
        if not all([docid, sheet_id]):
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
            response = self.session.post(SMART_TABLE_URL, json=params, timeout=30)
            result = response.json()
            if not result.get("success") or "data" not in result:
                return False

            records = result["data"]
            self.existing_userids = set()
            for record in records:
                values = record.get("values", {})
                person_field = values.get("人员", [])
                for person in person_field:
                    if isinstance(person, dict):
                        userid = person.get("user_id")
                        if userid:
                            self.existing_userids.add(userid)
            return True
        except Exception:
            return False

    def process_single_user(self, user: Dict, target_date: datetime):
        date_str = target_date.strftime('%Y-%m-%d')
        timestamp_str = str(int(target_date.timestamp() * 1000))
        target_start = int(target_date.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        target_end = int(target_date.replace(hour=23, minute=59, second=59, microsecond=0).timestamp())

        contact_data = self.get_contact_stats(user["userid"], target_start, target_end)
        group_data = self.get_group_chat_stats(user["userid"], target_start)

        new_clients = contact_data.get("new_contact_cnt", 0)
        messages_sent = contact_data.get("message_cnt", 0)
        group_members = group_data.get("member_total", 0)
        group_messages = group_data.get("msg_total", 0)

        user_data = {
            "姓名": user["name"],
            "userid": user["userid"],
            "人员": user["userid"],
            "记录日期": timestamp_str,
            "日期字符串": date_str,
            "新增客户数": new_clients,
            "发送消息数": messages_sent,
            "截至当天客户群总人数": group_members,
            "截至当天客户群消息总数": group_messages,
            "医院ID": user["hospital_ids"],
            "部门ID": user["department_ids"],
            "所属部门": user["department_hierarchies"]
        }

        is_all_zero = (
                new_clients == 0 and
                messages_sent == 0 and
                group_members == 0 and
                group_messages == 0
        )

        return user_data, is_all_zero

    def generate_csv(self) -> Tuple[bool, str]:
        if not self.all_collected_data:
            return False, ""

        field_names = [
            "姓名", "userid", "人员", "记录日期", "日期字符串",
            "新增客户数", "发送消息数", "截至当天客户群总人数", "截至当天客户群消息总数",
            "医院ID", "部门ID", "所属部门"
        ]

        # 使用目标日期而不是当前日期来生成文件名
        target_date_str = self.target_dates[0].strftime("%Y%m%d")
        output_path = f"wechat_stats_{target_date_str}.csv"

        try:
            with open(output_path, mode='w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=field_names)
                writer.writeheader()
                for record in self.all_collected_data:
                    writer.writerow(record)
            return True, os.path.abspath(output_path)
        except Exception:
            return False, ""

    def run(self) -> Tuple[bool, str]:
        try:
            logger.info("开始获取钉钉访问令牌...")
            if not self.get_dingtalk_access_token():
                logger.error("获取钉钉访问令牌失败")
                return False, ""
            logger.info("钉钉访问令牌获取成功")

            logger.info("开始获取排行榜配置...")
            if not self.get_ranking_config():
                logger.error("获取排行榜配置失败")
                return False, ""
            logger.info("排行榜配置获取成功")

            logger.info("开始获取地区配置数据...")
            if not self.get_region_config_data():
                logger.error("获取地区配置数据失败")
                return False, ""
            logger.info(f"地区配置数据获取成功，共获取 {len(self.hospital_ids)} 个医院ID")

            logger.info("开始获取企业微信访问令牌...")
            if not self.get_access_token():
                logger.error("获取企业微信访问令牌失败")
                return False, ""
            logger.info("企业微信访问令牌获取成功")

            logger.info("开始获取所有用户信息...")
            if not self.get_all_users():
                logger.error("获取所有用户信息失败")
                return False, ""
            logger.info(f"用户信息获取成功，共 {len(self.users)} 名用户")

            total_dates = len(self.target_dates)
            total_user = len(self.users)
            collected_count = 0
            skip_count = 0

            logger.info(f"开始处理 {total_dates} 天的数据，共 {total_user} 名用户...")

            for target_date in self.target_dates:
                date_str = target_date.strftime('%Y-%m-%d')
                logger.info(f"处理日期: {date_str}")

                for idx, user in enumerate(self.users):
                    logger.info(f"处理用户 {idx + 1}/{total_user}: {user['name']} ({user['userid']})")
                    user_data, is_all_zero = self.process_single_user(user, target_date)
                    if is_all_zero:
                        self.zero_data_list.append(user_data)
                        skip_count += 1
                    else:
                        self.all_collected_data.append(user_data)
                        collected_count += 1
                    time.sleep(0.5)

            logger.info(f"数据采集完成: 有效数据 {collected_count} 条，跳过全零数据 {skip_count} 条")

            logger.info("开始获取现有用户ID...")
            if not self.get_existing_userids_from_customer_table():
                logger.warning("获取现有用户ID失败，将保留所有数据")
            else:
                logger.info(f"成功获取 {len(self.existing_userids)} 个现有用户ID")
                retained_zero_count = 0
                for zero_data in self.zero_data_list:
                    if zero_data["userid"] in self.existing_userids:
                        self.all_collected_data.append(zero_data)
                        retained_zero_count += 1
                skip_count -= retained_zero_count
                logger.info(f"保留 {retained_zero_count} 条有历史记录的全零数据")

            logger.info("开始生成CSV文件...")
            success, csv_path = self.generate_csv()
            if not success:
                logger.error("生成CSV文件失败")
                return False, ""
            logger.info(f"CSV文件生成成功: {csv_path}")
            return True, csv_path
        except Exception as e:
            logger.exception(f"运行过程中发生异常: {str(e)}")
            return False, ""
        finally:
            self.session.close()
            logger.info("会话已关闭")


# ======================== 数据处理与更新部分 ========================
class DataProcessorAndUpdater:
    def __init__(self, target_date: datetime,dingtalk_access_token: str):
        self.target_date = target_date
        self.target_date_str = self.target_date.strftime("%Y-%m-%d")
        self.target_timestamp = str(int(self.target_date.timestamp() * 1000))
        self.prev_date = self.target_date - timedelta(days=1)
        self.prev_timestamp = str(int(self.prev_date.timestamp() * 1000))
        self.week_start_date = self.target_date - timedelta(days=7)
        self.week_start_timestamp = str(int(self.week_start_date.timestamp() * 1000))
        self.week_end_date = self.prev_date
        self.week_end_timestamp = str(int(self.week_end_date.timestamp() * 1000))
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        self.access_token = None
        self.dingtalk_access_token = None
        self.batch_size = 100
        self.region_config = {}
        self.customer_table_config = {}
        self.hospital_ids = set()
        self.lastweek_table_config = {}
        self.yesterday_table_config = {}
        self.sevendays_table_config = {}
        self.dingtalk_access_token = dingtalk_access_token
        logger.info(f"初始化数据处理与更新器（目标日期: {self.target_date_str}）")


    def get_ranking_config(self) -> bool:
        if not self.dingtalk_access_token:
            return False

        base_url = "https://api.dingtalk.com/v1.0/notable/bases/"
        full_url = f"{base_url}{DINGTALK_CONFIG['base_id']}/sheets/{urllib.parse.quote(DINGTALK_CONFIG['sheet_name'])}/records"
        headers = {
            "x-acs-dingtalk-access-token": self.dingtalk_access_token,
            "Content-Type": "application/json"
        }
        params = {"maxResults": 100, "operatorId": DINGTALK_CONFIG["operator_id"]}

        try:
            response = self.session.get(full_url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            records = data.get("records", [])

            for record in records:
                fields = record.get("fields", {})
                if fields.get("任务名称") == "排行榜":
                    config_value = fields.get("通用配置表")
                    if config_value:
                        try:
                            config = json.loads(config_value)
                            word_list = config.get("WordList", {})

                            if "config" in word_list:
                                self.region_config = {
                                    "docid": word_list.get("docid"),
                                    "sheet_id": word_list["config"].get("sheet_id"),
                                    "view_id": word_list["config"].get("view_id")
                                }

                            if "total_data" in word_list:
                                self.customer_table_config = {
                                    "docid": word_list.get("docid"),
                                    "sheet_id": word_list["total_data"].get("sheet_id"),
                                    "view_id": word_list["total_data"].get("view_id")
                                }

                            if "lastweek_data" in word_list:
                                self.lastweek_table_config = {
                                    "docid": word_list.get("docid"),
                                    "sheet_id": word_list["lastweek_data"].get("sheet_id"),
                                    "view_id": word_list["lastweek_data"].get("view_id")
                                }

                            if "yesterday_data" in word_list:
                                self.yesterday_table_config = {
                                    "docid": word_list.get("docid"),
                                    "sheet_id": word_list["yesterday_data"].get("sheet_id"),
                                    "view_id": word_list["yesterday_data"].get("view_id")
                                }

                            if "sevendays_data" in word_list:
                                self.sevendays_table_config = {
                                    "docid": word_list.get("docid"),
                                    "sheet_id": word_list["sevendays_data"].get("sheet_id"),
                                    "view_id": word_list["sevendays_data"].get("view_id")
                                }
                                return True
                        except json.JSONDecodeError:
                            pass
            return False
        except Exception:
            return False

    def get_region_config_data(self) -> bool:
        if not self.region_config:
            return False

        docid = self.region_config.get("docid")
        sheet_id = self.region_config.get("sheet_id")
        view_id = self.region_config.get("view_id")
        if not all([docid, sheet_id, view_id]):
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
            response = self.session.post(SMART_TABLE_URL, json=params, timeout=15)
            result = response.json()
            if not result.get("success") or "data" not in result:
                return False

            records = result["data"]
            for record in records:
                values = record.get("values", {})
                hospital_id = self._extract_field_value(values.get("医院ID"))
                if hospital_id:
                    self.hospital_ids.add(hospital_id)
            return len(self.hospital_ids) > 0
        except Exception:
            return False

    def _extract_field_value(self, field_data) -> str:
        if not field_data:
            return ""

        if isinstance(field_data, list):
            full_text = ""
            for item in field_data:
                if isinstance(item, dict):
                    text_value = (item.get("text") or item.get("value") or item.get("content") or "")
                    if isinstance(text_value, list):
                        full_text += self._extract_field_value(text_value)
                    else:
                        full_text += str(text_value)
                else:
                    full_text += str(item)
            return full_text.strip()

        if isinstance(field_data, dict):
            return (field_data.get("text") or field_data.get("value") or field_data.get("content") or "")

        return str(field_data).strip()

    def _extract_number_value(self, field_data) -> int:
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

    def get_access_token(self) -> bool:
        url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={CORPID}&corpsecret={CORPSECRET}"
        try:
            response = self.session.get(url, timeout=10)
            result = response.json()
            if result.get("errcode") == 0:
                self.access_token = result["access_token"]
                return True
            return False
        except Exception:
            return False

    def read_and_filter_csv(self, csv_path: str) -> List[Dict]:
        try:
            df = pd.read_csv(csv_path, dtype=str)
            filtered_df = df[
                (df["记录日期"] == self.target_date_str) |
                (df["记录日期"] == self.target_timestamp)
                ]
            return filtered_df.to_dict(orient="records")
        except Exception:
            return []

    def process_individual_data(self, csv_data: List[Dict]) -> Tuple[List[Dict], Dict]:
        if not csv_data:
            return [], {}

        user_ids = [self._extract_field_value(row.get("userid", "")) for row in csv_data]
        user_ids = list(filter(None, user_ids))

        yesterday_data = self.query_yesterday_data(user_ids)
        sevendays_data = self.query_sevendays_data(user_ids)
        lastweek_cumulative_data = self.query_lastweek_cumulative_data(user_ids)

        processed = []
        hospital_aggregations = {}

        for row in csv_data:
            userid = self._extract_field_value(row.get("userid", ""))
            name = self._extract_field_value(row.get("姓名", "")).strip()
            hospital_id = self._extract_field_value(row.get("医院ID", "")).strip()

            if not userid and not name:
                continue

            if hospital_id not in hospital_aggregations:
                hospital_aggregations[hospital_id] = {
                    "新增客户数": 0,
                    "发送消息数": 0,
                    "累积新增客户数": 0,
                    "累积消息发送数": 0,
                    "累积客户群人数": 0,
                    "累积群消息数": 0,
                    "用户数": 0,
                    "近一周新增客户数": 0
                }

            today_new = self._extract_number_value(row.get("新增客户数", 0))
            today_msg = self._extract_number_value(row.get("发送消息数", 0))
            today_group_members = self._extract_number_value(row.get("截至当天客户群总人数", 0))
            today_group_msg = self._extract_number_value(row.get("截至当天客户群消息总数", 0))

            yesterday_cum = yesterday_data.get(userid, {
                "累积新增客户数": 0,
                "累积消息发送数": 0
            })

            lastweek_cum = lastweek_cumulative_data.get(userid, {
                "累积新增客户数": 0,
                "累积消息发送数": 0,
                "累积客户群人数": 0,
                "累积群消息数": 0
            })

            cumulative_new = yesterday_cum["累积新增客户数"] + today_new
            cumulative_msg = yesterday_cum["累积消息发送数"] + today_msg
            cumulative_group_members = today_group_members
            cumulative_group_msg = today_group_msg

            weekly_new = sevendays_data.get(userid, 0) + today_new

            cum_new_ratio = self._calculate_ratio(cumulative_new, lastweek_cum["累积新增客户数"])
            cum_msg_ratio = self._calculate_ratio(cumulative_msg, lastweek_cum["累积消息发送数"])
            cum_members_ratio = self._calculate_ratio(cumulative_group_members, lastweek_cum["累积客户群人数"])
            cum_msg_group_ratio = self._calculate_ratio(cumulative_group_msg, lastweek_cum["累积群消息数"])
            weekly_new_ratio = self._calculate_ratio(weekly_new, lastweek_cum.get("近一周新增客户数", 0))

            processed_row = {
                "姓名": name,
                "userid": userid,
                "人员": userid,
                "记录日期": self.target_timestamp,
                "新增客户数": today_new,
                "发送消息数": today_msg,
                "累积新增客户数": cumulative_new,
                "累积消息发送数": cumulative_msg,
                "近一周新增客户数": weekly_new,
                "累积客户群人数": cumulative_group_members,
                "累积群消息数": cumulative_group_msg,
                "累积新增客户数环比": cum_new_ratio,
                "累积消息发送数环比": cum_msg_ratio,
                "累积客户群人数环比": cum_members_ratio,
                "累积群消息数环比": cum_msg_group_ratio,
                "近一周新增客户数环比": weekly_new_ratio,
                "医院ID": hospital_id,
                "部门ID": self._extract_field_value(row.get("部门ID", "")),
                "所属部门": self._extract_field_value(row.get("所属部门", ""))
            }
            processed.append(processed_row)

            hospital_aggregations[hospital_id]["新增客户数"] += today_new
            hospital_aggregations[hospital_id]["发送消息数"] += today_msg
            hospital_aggregations[hospital_id]["累积新增客户数"] += cumulative_new
            hospital_aggregations[hospital_id]["累积消息发送数"] += cumulative_msg
            hospital_aggregations[hospital_id]["累积客户群人数"] += today_group_members
            hospital_aggregations[hospital_id]["累积群消息数"] += today_group_msg
            hospital_aggregations[hospital_id]["用户数"] += 1

        return processed, hospital_aggregations

    def _calculate_ratio(self, current_value: float, last_value: float) -> float:
        if last_value == 0:
            return 0.0
        return round(((current_value - last_value) / last_value) * 100, 2)

    def batch_write_individual_data(self, data: List[Dict]) -> bool:
        if not data:
            return True

        total = len(data)
        success_count = 0
        failure_count = 0

        for i in range(0, total, self.batch_size):
            batch = data[i:i + self.batch_size]

            records = []
            for row in batch:
                record_values = {
                    "姓名": [{"type": "text", "text": row["姓名"]}],
                    "人员": [{"type": "user", "user_id": row["userid"]}],
                    "记录日期": row["记录日期"],
                    "新增客户数": row["新增客户数"],
                    "发送消息数": row["发送消息数"],
                    "累积新增客户数": row["累积新增客户数"],
                    "累积消息发送数": row["累积消息发送数"],
                    "近一周新增客户数": row["近一周新增客户数"],
                    "累积客户群人数": row["累积客户群人数"],
                    "累积群消息数": row["累积群消息数"],
                    "累积新增客户数环比": row["累积新增客户数环比"],
                    "累积消息发送数环比": row["累积消息发送数环比"],
                    "累积客户群人数环比": row["累积客户群人数环比"],
                    "累积群消息数环比": row["累积群消息数环比"],
                    "近一周新增客户数环比": row["近一周新增客户数环比"],
                    "医院ID": [{"type": "text", "text": row["医院ID"]}],
                    "部门ID": [{"type": "text", "text": row["部门ID"]}],
                    "所属部门": [{"type": "text", "text": row["所属部门"]}]
                }
                records.append({"values": record_values})

            payload = {
                "action": "通用批量写入表单",
                "company": "花都家庭医生",
                "WordList": {
                    "docid": self.customer_table_config["docid"],
                    "sheet_id": self.customer_table_config["sheet_id"],
                    "view_id": self.customer_table_config["view_id"],
                    "records": records
                }
            }

            try:
                response = self.session.post(SMART_TABLE_URL, json=payload, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        success_count += len(batch)
                    else:
                        failure_count += len(batch)
                else:
                    failure_count += len(batch)
            except Exception:
                failure_count += len(batch)
            time.sleep(1.5)

        return failure_count == 0

    def update_hospital_data(self, hospital_aggregations: Dict) -> bool:
        if not hospital_aggregations or not self.region_config:
            return True

        hospital_ids = list(hospital_aggregations.keys())
        hospital_weekly_data = self.query_hospital_weekly_data(hospital_ids)
        lastweek_cumulative_data = self.get_lastweek_hospital_cumulative_data(hospital_ids)
        hospital_records = self._get_all_hospital_records()
        if not hospital_records:
            return False

        hospital_record_map = {}
        for rec in hospital_records:
            values = rec.get("values", {})
            record_id = rec.get("record_id") or rec.get("id")
            hospital_id_value = (
                    self._extract_field_value(values.get("企微通讯录部门名", "")) or
                    self._extract_field_value(values.get("医院ID", "")) or
                    self._extract_field_value(values.get("部门ID", ""))
            )
            if hospital_id_value:
                hospital_record_map[str(hospital_id_value)] = rec

        success_count = 0
        failure_count = 0

        for hospital_id, agg_data in hospital_aggregations.items():
            if hospital_id not in hospital_record_map:
                continue

            hospital_record = hospital_record_map[hospital_id]
            record_id = hospital_record.get("record_id") or hospital_record.get("id")
            if not record_id:
                continue

            staff_config = self._extract_staff_config(hospital_record)
            total_staff, certified_count = 0, 0
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                total_staff, certified_count = loop.run_until_complete(self.get_staff_table_stats(staff_config))
                loop.close()
            except Exception:
                pass

            lastweek_cum = lastweek_cumulative_data.get(hospital_id, {
                "累积新增客户数": 0,
                "累积消息发送数": 0,
                "累积客户群人数": 0,
                "累积群消息数": 0,
                "近一周新增客户数": 0
            })

            def calculate_ratio(current, last):
                if last == 0:
                    return 0.0
                return round(((current - last) / last) * 100, 2)

            new_customer_ratio = calculate_ratio(agg_data["累积新增客户数"], lastweek_cum["累积新增客户数"])
            message_ratio = calculate_ratio(agg_data["累积消息发送数"], lastweek_cum["累积消息发送数"])
            group_members_ratio = calculate_ratio(agg_data["累积客户群人数"], lastweek_cum["累积客户群人数"])
            group_messages_ratio = calculate_ratio(agg_data["累积群消息数"], lastweek_cum["累积群消息数"])
            current_week_new = hospital_weekly_data.get(hospital_id, 0) + agg_data["新增客户数"]
            last_week_new = lastweek_cum.get("近一周新增客户数", 0)
            weekly_new_ratio = self._calculate_ratio(current_week_new, last_week_new)

            update_data = {
                "action": "通用更新表单",
                "company": "花都家庭医生",
                "WordList": {
                    "docid": self.region_config["docid"],
                    "sheet_id": self.region_config["sheet_id"],
                    "record_id": record_id,
                    "values": {
                        "新增客户数": agg_data["累积新增客户数"],
                        "发送消息数": agg_data["累积消息发送数"],
                        "近一周新增客户数": hospital_weekly_data.get(hospital_id, 0) + agg_data["新增客户数"],
                        "截至当天客户群总人数": agg_data["累积客户群人数"],
                        "截至当天客户群消息总数": agg_data["累积群消息数"],
                        "家医团队总人数": total_staff,
                        "已实名认证": certified_count,
                        "新增客户数环比": new_customer_ratio,
                        "发送消息数环比": message_ratio,
                        "群人数环比": group_members_ratio,
                        "群消息数环比": group_messages_ratio,
                        "近一周新增客户数环比": weekly_new_ratio
                    },
                    "view_id": self.region_config["view_id"]
                }
            }

            try:
                response = self.session.post(SMART_TABLE_URL, json=update_data, timeout=15)
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        success_count += 1
                    else:
                        failure_count += 1
                else:
                    failure_count += 1
            except Exception:
                failure_count += 1
            time.sleep(1)

        return failure_count == 0

    async def get_staff_table_stats(self, staff_config: Dict) -> Tuple[int, int]:
        if not staff_config or "docid" not in staff_config or "tab" not in staff_config:
            return 0, 0

        try:
            staff_records = await self.get_all_table_data(
                staff_config["docid"],
                staff_config["tab"],
                staff_config.get("view_id")
            )

            total_rows = len(staff_records)
            certified_count = 0
            for record in staff_records:
                values = record.get("values", {})
                qr_code = self._extract_field_value(values.get("本人的二维码链接", ""))
                if qr_code.strip():
                    certified_count += 1
            return total_rows, certified_count
        except Exception:
            return 0, 0

    async def get_all_table_data(self, docid: str, sheet_id: str, view_id: str = None) -> List[Dict]:
        all_data = []
        page_token = None

        while True:
            data = await self.fetch_table_data(docid, sheet_id, view_id, page_token)
            if not data:
                break

            records = data.get("data", [])
            all_data.extend(records)

            if "next_page_token" in data and data["next_page_token"]:
                page_token = data["next_page_token"]
            else:
                break

        return all_data

    async def fetch_table_data(self, docid: str, sheet_id: str, view_id: str = None, page_token: str = None) -> Dict:
        params = {
            "action": "通用查询表单",
            "company": "花都家庭医生",
            "WordList": {
                "docid": docid,
                "sheet_id": sheet_id,
                "view_id": view_id,
                "page_size": self.batch_size
            }
        }

        if page_token:
            params["WordList"]["page_token"] = page_token

        try:
            response = self.session.post(SMART_TABLE_URL, json=params)
            if response.status_code != 200:
                return None

            data = response.json()
            if not data.get("success"):
                return None
            return data
        except Exception:
            return None

    def query_yesterday_data(self, user_ids: List[str]) -> Dict[str, Dict]:
        if not user_ids or not self.yesterday_table_config:
            return {}

        try:
            payload = {
                "action": "通用查询表单",
                "company": "花都家庭医生",
                "WordList": {
                    "docid": self.yesterday_table_config["docid"],
                    "sheet_id": self.yesterday_table_config["sheet_id"],
                    "view_id": self.yesterday_table_config["view_id"]
                }
            }

            response = self.session.post(SMART_TABLE_URL, json=payload, timeout=25)
            result = response.json()
            if not result.get("success"):
                return {}

            records = result.get("data", [])
            all_yesterday_data = {}
            for record in records:
                values = record.get("values", {})
                userid = None
                person_field = values.get("人员")
                if isinstance(person_field, list) and len(person_field) > 0:
                    person_obj = person_field[0]
                    if isinstance(person_obj, dict):
                        userid = person_obj.get("user_id")
                elif isinstance(person_field, str):
                    userid = person_field

                if userid and userid in user_ids:
                    all_yesterday_data[userid] = {
                        "累积新增客户数": self._extract_number_value(values.get("累积新增客户数", 0)),
                        "累积消息发送数": self._extract_number_value(values.get("累积消息发送数", 0))
                    }
            return all_yesterday_data
        except Exception:
            return {}

    def query_sevendays_data(self, user_ids: List[str]) -> Dict[str, int]:
        if not user_ids or not self.sevendays_table_config:
            return {}

        filter_spec = {
            "and": [
                {
                    "field": "人员",
                    "condition": "in",
                    "value": user_ids
                },
                {
                    "field": "记录日期",
                    "condition": "greater_than_or_equal",
                    "value": self.week_start_timestamp
                },
                {
                    "field": "记录日期",
                    "condition": "less_than_or_equal",
                    "value": self.prev_timestamp
                }
            ]
        }

        payload = {
            "action": "条件筛选查询表单",
            "company": "花都家庭医生",
            "WordList": {
                "docid": self.sevendays_table_config["docid"],
                "sheet_id": self.sevendays_table_config["sheet_id"],
                "view_id": self.sevendays_table_config["view_id"],
                "filter_spec": filter_spec,
                "field_titles": ["人员", "新增客户数"]
            }
        }

        try:
            response = self.session.post(SMART_TABLE_URL, json=payload, timeout=20)
            result = response.json()
            if not result.get("success"):
                return {}

            weekly_data = {user_id: 0 for user_id in user_ids}
            for record in result.get("data", []):
                values = record.get("values", {})
                userid = values.get("人员", [{}])[0].get("user_id", "")
                if userid in weekly_data:
                    weekly_data[userid] += self._extract_number_value(values.get("新增客户数", 0))
            return weekly_data
        except Exception:
            return {}

    def query_lastweek_cumulative_data(self, user_ids: List[str]) -> Dict[str, Dict]:
        if not user_ids or not self.lastweek_table_config:
            return {}

        filter_spec = {
            "and": [
                {
                    "field": "人员",
                    "condition": "in",
                    "value": user_ids
                }
            ]
        }

        payload = {
            "action": "条件筛选查询表单",
            "company": "花都家庭医生",
            "WordList": {
                "docid": self.lastweek_table_config["docid"],
                "sheet_id": self.lastweek_table_config["sheet_id"],
                "view_id": self.lastweek_table_config["view_id"],
                "filter_spec": filter_spec,
                "field_titles": ["人员", "累积新增客户数", "累积消息发送数", "累积客户群人数", "累积群消息数", "近一周新增客户数"]
            }
        }

        try:
            response = self.session.post(SMART_TABLE_URL, json=payload, timeout=20)
            result = response.json()
            if not result.get("success"):
                return {}

            lastweek_data = {}
            for record in result.get("data", []):
                values = record.get("values", {})
                userid = values.get("人员", [{}])[0].get("user_id", "")
                if not userid:
                    continue

                lastweek_data[userid] = {
                    "累积新增客户数": self._extract_number_value(values.get("累积新增客户数", 0)),
                    "累积消息发送数": self._extract_number_value(values.get("累积消息发送数", 0)),
                    "累积客户群人数": self._extract_number_value(values.get("累积客户群人数", 0)),
                    "累积群消息数": self._extract_number_value(values.get("累积群消息数", 0)),
                    "近一周新增客户数": self._extract_number_value(values.get("近一周新增客户数", 0))
                }
            return lastweek_data
        except Exception:
            return {}

    def query_hospital_weekly_data(self, hospital_ids: List[str]) -> Dict[str, int]:
        if not hospital_ids or not self.sevendays_table_config:
            return {}

        filter_spec = {
            "and": [
                {
                    "field": "医院ID",
                    "condition": "in",
                    "value": hospital_ids
                }
            ]
        }

        payload = {
            "action": "条件筛选查询表单",
            "company": "花都家庭医生",
            "WordList": {
                "docid": self.sevendays_table_config["docid"],
                "sheet_id": self.sevendays_table_config["sheet_id"],
                "view_id": self.sevendays_table_config["view_id"],
                "filter_spec": filter_spec,
                "field_titles": ["医院ID", "新增客户数"]
            }
        }

        try:
            response = self.session.post(SMART_TABLE_URL, json=payload, timeout=20)
            result = response.json()
            if not result.get("success"):
                return {}

            weekly_data = {hospital_id: 0 for hospital_id in hospital_ids}
            for record in result.get("data", []):
                values = record.get("values", {})
                hospital_id = self._extract_field_value(values.get("医院ID", ""))
                if hospital_id in weekly_data:
                    weekly_data[hospital_id] += self._extract_number_value(values.get("新增客户数", 0))
            return weekly_data
        except Exception:
            return {}

    def get_lastweek_hospital_cumulative_data(self, hospital_ids: List[str]) -> Dict[str, Dict]:
        if not hospital_ids or not self.lastweek_table_config:
            return {}

        filter_spec = {
            "and": [
                {
                    "field": "医院ID",
                    "condition": "in",
                    "value": hospital_ids
                }
            ]
        }

        payload = {
            "action": "条件筛选查询表单",
            "company": "花都家庭医生",
            "WordList": {
                "docid": self.lastweek_table_config["docid"],
                "sheet_id": self.lastweek_table_config["sheet_id"],
                "view_id": self.lastweek_table_config["view_id"],
                "filter_spec": filter_spec,
                "field_titles": ["医院ID", "累积新增客户数", "累积消息发送数", "累积客户群人数", "累积群消息数", "近一周新增客户数"]
            }
        }

        try:
            response = self.session.post(SMART_TABLE_URL, json=payload, timeout=20)
            result = response.json()
            if not result.get("success"):
                return {}

            hospital_data = {}
            for record in result.get("data", []):
                values = record.get("values", {})
                hospital_id = self._extract_field_value(values.get("医院ID", ""))
                if not hospital_id:
                    continue

                if hospital_id not in hospital_data:
                    hospital_data[hospital_id] = {
                        "累积新增客户数": 0,
                        "累积消息发送数": 0,
                        "累积客户群人数": 0,
                        "累积群消息数": 0,
                        "近一周新增客户数": 0
                    }

                hospital_data[hospital_id]["累积新增客户数"] += self._extract_number_value(values.get("累积新增客户数", 0))
                hospital_data[hospital_id]["累积消息发送数"] += self._extract_number_value(values.get("累积消息发送数", 0))
                hospital_data[hospital_id]["累积客户群人数"] += self._extract_number_value(values.get("累积客户群人数", 0))
                hospital_data[hospital_id]["累积群消息数"] += self._extract_number_value(values.get("累积群消息数", 0))
                hospital_data[hospital_id]["近一周新增客户数"] += self._extract_number_value(values.get("近一周新增客户数", 0))
            return hospital_data
        except Exception:
            return {}

    def _get_all_hospital_records(self) -> List[Dict]:
        if not self.region_config:
            return []

        params = {
            "action": "通用查询表单",
            "company": "花都家庭医生",
            "WordList": {
                "docid": self.region_config["docid"],
                "sheet_id": self.region_config["sheet_id"],
                "view_id": self.region_config["view_id"]
            }
        }

        try:
            response = self.session.post(SMART_TABLE_URL, json=params, timeout=15)
            result = response.json()
            if result.get("success") and "data" in result:
                return result["data"]
            return []
        except Exception:
            return []

    def _extract_staff_config(self, hospital_record: Dict) -> Dict:
        values = hospital_record.get("values", {})
        doc_config_str = self._extract_field_value(values.get("文档ID"))

        staff_config = {}

        docid_match = re.search(r'"docid"\s*:\s*"([^"]+)"', doc_config_str)
        if docid_match:
            staff_config["docid"] = docid_match.group(1)

        staff_match = re.search(
            r'"staff"\s*:\s*{\s*"tab"\s*:\s*"([^"]+)"\s*,\s*"viewId"\s*:\s*"([^"]+)"',
            doc_config_str,
            re.DOTALL
        )
        if staff_match:
            staff_config["tab"] = staff_match.group(1)
            staff_config["view_id"] = staff_match.group(2)

        if not staff_config.get("tab") or not staff_config.get("view_id") or not staff_config.get("docid"):
            try:
                config_data = json.loads(doc_config_str)
                staff_config["docid"] = config_data.get("docid", staff_config.get("docid", ""))
                staff_data = config_data.get("staff", {})
                if isinstance(staff_data, dict):
                    staff_config["tab"] = staff_data.get("tab") or staff_config.get("tab", "")
                    staff_config["view_id"] = staff_data.get("viewId", staff_data.get("view_id")) or staff_config.get(
                        "view_id", "")
            except Exception:
                pass

        return staff_config

    def run(self, csv_path: str) -> bool:
        try:
            logger.info("开始钉钉数据处理流程...")

            logger.info("获取钉钉访问令牌...")
            if not self.get_dingtalk_access_token():
                logger.error("获取钉钉访问令牌失败")
                return False
            logger.info("钉钉访问令牌获取成功")

            logger.info("获取排行榜配置...")
            if not self.get_ranking_config():
                logger.error("获取排行榜配置失败")
                return False
            logger.info("排行榜配置获取成功")

            logger.info("获取地区配置数据...")
            if not self.get_region_config_data():
                logger.error("获取地区配置数据失败")
                return False
            logger.info(f"地区配置数据获取成功，共 {len(self.hospital_ids)} 个医院ID")

            logger.info("获取企业微信访问令牌...")
            if not self.get_access_token():
                logger.error("获取企业微信访问令牌失败")
                return False
            logger.info("企业微信访问令牌获取成功")

            logger.info(f"读取CSV文件: {csv_path}")
            csv_data = self.read_and_filter_csv(csv_path)
            if not csv_data:
                logger.error("读取CSV文件失败或未找到匹配数据")
                return False
            logger.info(f"成功读取 {len(csv_data)} 条CSV记录")

            logger.info("处理个人数据...")
            processed_data, hospital_aggregations = self.process_individual_data(csv_data)
            if not processed_data:
                logger.error("处理个人数据失败")
                return False
            logger.info(f"成功处理 {len(processed_data)} 条个人数据")
            logger.info(f"聚合了 {len(hospital_aggregations)} 家医院的数据")

            logger.info("批量写入个人数据到企微表格...")
            if not self.batch_write_individual_data(processed_data):
                logger.error("批量写入个人数据失败")
                return False
            logger.info("个人数据写入成功")

            logger.info("更新医院数据...")
            if not self.update_hospital_data(hospital_aggregations):
                logger.error("更新医院数据失败")
                return False
            logger.info("医院数据更新成功")

            return True
        except Exception as e:
            logger.exception(f"运行过程中发生异常: {str(e)}")
            return False
        finally:
            self.session.close()
            logger.info("会话已关闭")


# ======================== 主流程 ========================
def main():
    logger.info("===== 企业微信数据采集与更新流程开始 =====")

    # 第一步：抓取数据并生成CSV
    logger.info("===== 数据采集阶段开始 =====")
    collector = WeChatDataCollector(
        corp_id=CORPID,
        corp_secret=CORPSECRET
    )
    success, csv_path = collector.run()

    if not success or not csv_path:
        logger.error("数据采集失败，流程终止")
        sys.exit(1)

    logger.info(f"数据采集成功，CSV文件路径: {csv_path}")

    # 第二步：处理CSV数据并更新钉钉表格
    logger.info("===== 数据处理与更新阶段开始 =====")
    processor = DataProcessorAndUpdater(
        target_date=collector.target_dates[0],
        dingtalk_access_token=collector.dingtalk_access_token
    )
    success = processor.run(csv_path)

    if success:
        logger.info("===== 数据处理与更新成功完成 =====")
        sys.exit(0)
    else:
        logger.error("数据处理与更新失败")
        sys.exit(1)


if __name__ == "__main__":
    main()