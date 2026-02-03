import json
import re
import time
import urllib.parse
from datetime import datetime, timedelta
from typing import List, Dict, Set

import pandas as pd
import requests
from requests import Session
import sys

# 配置常量（实际使用时需替换为真实值）
CORPID = "ww6fffc827ac483f35"
CORPSECRET = "DxTJu-VblBUVmeQHGaEKvtEzXTRHFSgSfbJIfP39okQ"
DINGTALK_CONFIG = {
    "app_key": "dingoicseqn2bmdcazpl",
    "app_secret": "hiiqLe8teDkAADlJh9eklgsbtGIvrG8hPJyOC8as04wzG69OGmgaY_vQ_gyKTXEg",
    "base_id": "YndMj49yWjDEYy3ECQwPlLkgJ3pmz5aA",
    "sheet_name": "配置表",
    "operator_id": "jYEXEC84RV3QE3sm0UaeDwiEiE"
}
SMART_TABLE_URL = "https://smallwecom.yesboss.work/smarttable"

# 日志配置
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CSVToWeChatCombinedImporter:
    def __init__(self):
        self.session: Session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'application/json'
        })
        self.access_token = None  # 企微token
        self.dingtalk_access_token = None  # 钉钉token
        self.batch_size = 100  # 批量处理阈值

        # 配置参数
        self.region_config = {}  # 地区配置表参数
        self.customer_table_config = {}  # 客户数据表配置
        self.hospital_ids: Set[str] = set()  # 医院ID集合
        self.lastweek_table_config = {}  # 上周数据表配置
        self.yesterday_table_config = {}  # 昨日数据表配置
        self.sevendays_table_config = {}  # 七日数据表配置

        # 日期配置
        self.target_date = datetime(2025, 9, 1)  # 目标日期，可根据实际情况调整
        self.target_date_str = self.target_date.strftime("%Y-%m-%d")
        self.target_timestamp = str(int(self.target_date.timestamp() * 1000))

        # 修复：添加缺失的 prev_date 和 prev_timestamp
        self.prev_date = self.target_date - timedelta(days=1)  # 前一天
        self.prev_timestamp = str(int(self.prev_date.timestamp() * 1000))

        # 相关日期
        self.week_start_date = self.target_date - timedelta(days=7)  # 七天前
        self.week_start_timestamp = str(int(self.week_start_date.timestamp() * 1000))
        self.week_end_date = self.prev_date  # 前一天
        self.week_end_timestamp = str(int(self.week_end_date.timestamp() * 1000))

        logger.info(f"初始化CSV导入器（目标日期: {self.target_date_str}）")

    # ---------------------- 企微Token获取 ----------------------
    def get_access_token(self) -> bool:
        url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={CORPID}&corpsecret={CORPSECRET}"
        try:
            response = self.session.get(url, timeout=10)
            result = response.json()
            if result.get("errcode") == 0:
                self.access_token = result["access_token"]
                logger.info("获取企微access_token成功")
                return True
            logger.error(f"获取企微Token失败: {result.get('errmsg')}")
            return False
        except Exception as e:
            logger.error(f"获取企微Token异常: {str(e)}")
            return False

    # ---------------------- 钉钉相关方法 ----------------------
    def get_dingtalk_access_token(self) -> bool:
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
                logger.error(f"获取钉钉令牌失败 | 代码{error_code}, 消息: {error_msg}")
                return False

            self.dingtalk_access_token = access_token
            logger.info(f"获取钉钉令牌成功 | 有效期: {token_data.get('expiresIn', 7200)}秒")
            return True
        except Exception as e:
            logger.error(f"获取钉钉令牌异常: {str(e)}")
            return False

    def get_ranking_config(self) -> bool:
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
            logger.debug(f"请求排行榜配置 | URL: {full_url[:50]}...")
            response = self.session.get(full_url, headers=headers, params=params, timeout=15)
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

                            # 获取地区配置表（医院配置表）
                            if "config" in word_list:
                                self.region_config = {
                                    "docid": word_list.get("docid"),
                                    "sheet_id": word_list["config"].get("sheet_id"),
                                    "view_id": word_list["config"].get("view_id")
                                }
                                logger.info(f"找到地区配置表参数: {self.region_config}")

                            # 获取客户数据表配置（total_data）
                            if "total_data" in word_list:
                                self.customer_table_config = {
                                    "docid": word_list.get("docid"),
                                    "sheet_id": word_list["total_data"].get("sheet_id"),
                                    "view_id": word_list["total_data"].get("view_id")
                                }
                                logger.info(f"找到客户数据表参数: {self.customer_table_config}")

                                # 获取上周数据表配置（lastweek_data）
                            if "lastweek_data" in word_list:
                                self.lastweek_table_config = {
                                    "docid": word_list.get("docid"),
                                     "sheet_id": word_list["lastweek_data"].get("sheet_id"),
                                    "view_id": word_list["lastweek_data"].get("view_id")
                                }
                                logger.info(f"找到上周数据表参数: {self.lastweek_table_config}")

                                # 获取昨日数据表配置（yesterday_data）
                            if "yesterday_data" in word_list:
                                self.yesterday_table_config = {
                                    "docid": word_list.get("docid"),
                                    "sheet_id": word_list["yesterday_data"].get("sheet_id"),
                                    "view_id": word_list["yesterday_data"].get("view_id")
                                }
                                logger.info(f"找到昨日数据表参数: {self.yesterday_table_config}")

                                # 获取七日数据表配置（sevendays_data）
                            if "sevendays_data" in word_list:
                                self.sevendays_table_config = {
                                    "docid": word_list.get("docid"),
                                    "sheet_id": word_list["sevendays_data"].get("sheet_id"),
                                    "view_id": word_list["sevendays_data"].get("view_id")
                                }
                                logger.info(f"找到七日数据表参数: {self.sevendays_table_config}")

                                return True
                        except json.JSONDecodeError:
                            logger.warning("配置解析失败，跳过")
            logger.warning("未找到有效的排行榜配置")
            return False
        except Exception as e:
            logger.error(f"获取排行榜配置失败: {str(e)}")
            return False

    def get_region_config_data(self) -> bool:
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

        try:
            logger.debug(f"请求地区配置表 | docid: {docid[:10]}...")
            response = self.session.post(SMART_TABLE_URL, json=params, timeout=15)
            result = response.json()

            if not result.get("success") or "data" not in result:
                logger.error(f"获取地区配置表失败 | 消息: {result.get('message', '未知错误')}")
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
        except Exception as e:
            logger.error(f"获取地区配置表异常: {str(e)}")
            return False

    # ---------------------- 数据查询方法 ----------------------
    def batch_query_historical_data(self, user_ids: List[str], timestamp: str) -> Dict[str, Dict]:
        """批量查询历史累积数据"""
        if not user_ids:
            return {}

        # 构造筛选条件：用户ID在列表中且记录日期等于目标日期
        filter_spec = {
            "and": [
                {
                    "field": "人员",
                    "condition": "in",
                    "value": user_ids
                },
                {
                    "field": "记录日期",
                    "condition": "equal",
                    "value": timestamp
                }
            ]
        }

        payload = {
            "action": "条件筛选查询表单",
            "company": "花都家庭医生",
            "WordList": {
                "docid": self.customer_table_config["docid"],
                "sheet_id": self.customer_table_config["sheet_id"],
                "view_id": self.customer_table_config["view_id"],
                "filter_spec": filter_spec,
                "field_titles": ["人员", "姓名", "累积新增客户数", "累计消息发送数", "累积客户群人数", "累积群消息数"]
            }
        }

        try:
            response = self.session.post(SMART_TABLE_URL, json=payload, timeout=20)
            result = response.json()
            if not result.get("success"):
                logger.error(f"批量查询历史数据失败: {result.get('message')}")
                return {}

            # 提取历史累积数据
            historical_data = {}
            for record in result.get("data", []):
                values = record.get("values", {})
                userid = values.get("人员", [{}])[0].get("user_id", "")
                if not userid:
                    continue

                historical_data[userid] = {
                    "累积新增客户数": self._extract_number_value(values.get("累积新增客户数", 0)),
                    "累积消息发送数": self._extract_number_value(values.get("累积消息发送数", 0)),
                    "累积客户群人数": self._extract_number_value(values.get("累积客户群人数", 0)),
                    "累积群消息数": self._extract_number_value(values.get("累积群消息数", 0))
                }

            logger.info(f"批量查询到 {len(historical_data)} 个用户的历史累积数据")
            return historical_data
        except Exception as e:
            logger.error(f"批量查询历史数据异常: {str(e)}")
            return {}

    def query_weekly_data(self, user_ids: List[str]) -> Dict[str, int]:
        """查询用户近一周的新增客户数"""
        if not user_ids:
            return {}

        # 构造筛选条件：用户ID在列表中且记录日期在近一周范围内
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
                    "value": self.week_end_timestamp
                }
            ]
        }

        payload = {
            "action": "条件筛选查询表单",
            "company": "花都家庭医生",
            "WordList": {
                "docid": self.customer_table_config["docid"],
                "sheet_id": self.customer_table_config["sheet_id"],
                "view_id": self.customer_table_config["view_id"],
                "filter_spec": filter_spec,
                "field_titles": ["人员", "新增客户数"]
            }
        }

        try:
            response = self.session.post(SMART_TABLE_URL, json=payload, timeout=20)
            result = response.json()
            if not result.get("success"):
                logger.error(f"查询周数据失败: {result.get('message')}")
                return {}

            # 汇总近一周新增客户数
            weekly_data = {user_id: 0 for user_id in user_ids}
            for record in result.get("data", []):
                values = record.get("values", {})
                userid = values.get("人员", [{}])[0].get("user_id", "")
                if userid in weekly_data:
                    weekly_data[userid] += self._extract_number_value(values.get("新增客户数", 0))

            logger.info(f"查询到 {len(weekly_data)} 个用户的近一周数据")
            return weekly_data
        except Exception as e:
            logger.error(f"查询周数据异常: {str(e)}")
            return {}

    def query_hospital_weekly_data(self, hospital_ids: List[str]) -> Dict[str, int]:
        """查询医院近一周的新增客户数"""
        if not hospital_ids:
            return {}

        # 构造筛选条件：医院ID在列表中且记录日期在近一周范围内
        filter_spec = {
            "and": [
                {
                    "field": "医院ID",
                    "condition": "in",
                    "value": hospital_ids
                },
                {
                    "field": "记录日期",
                    "condition": "greater_than_or_equal",
                    "value": self.week_start_timestamp
                },
                {
                    "field": "记录日期",
                    "condition": "less_than_or_equal",
                    "value": self.week_end_timestamp
                }
            ]
        }

        payload = {
            "action": "条件筛选查询表单",
            "company": "花都家庭医生",
            "WordList": {
                "docid": self.customer_table_config["docid"],
                "sheet_id": self.customer_table_config["sheet_id"],
                "view_id": self.customer_table_config["view_id"],
                "filter_spec": filter_spec,
                "field_titles": ["医院ID", "新增客户数"]
            }
        }

        try:
            response = self.session.post(SMART_TABLE_URL, json=payload, timeout=20)
            result = response.json()
            if not result.get("success"):
                logger.error(f"查询医院周数据失败: {result.get('message')}")
                return {}

            # 汇总医院近一周新增客户数
            weekly_data = {hospital_id: 0 for hospital_id in hospital_ids}
            for record in result.get("data", []):
                values = record.get("values", {})
                hospital_id = self._extract_field_value(values.get("医院ID", ""))
                if hospital_id in weekly_data:
                    weekly_data[hospital_id] += self._extract_number_value(values.get("新增客户数", 0))

            logger.info(f"查询到 {len(weekly_data)} 个医院的近一周数据")
            return weekly_data
        except Exception as e:
            logger.error(f"查询医院周数据异常: {str(e)}")
            return {}

    # ---------------------- 员工表统计方法 ----------------------
    async def get_staff_table_stats(self, staff_config: Dict) -> tuple[int, int]:
        """读取员工表总行数、二维码非空数"""
        if not staff_config or "docid" not in staff_config or "tab" not in staff_config:
            return 0, 0

        # 获取员工表所有数据
        staff_records = await self.get_all_table_data(
            staff_config["docid"],
            staff_config["tab"],
            staff_config.get("view_id")
        )

        total_rows = len(staff_records)  # 总行数（家医团队总人数）
        certified_count = 0  # 已实名认证数（二维码非空）

        for record in staff_records:
            qr_code = self._extract_field_value(record.get("values", {}).get("本人的二维码链接", ""))
            if qr_code.strip():
                certified_count += 1

        return total_rows, certified_count

    async def get_all_table_data(self, docid: str, sheet_id: str, view_id: str = None) -> List[Dict]:
        """获取所有分页数据"""
        all_data = []
        page_token = None

        while True:
            data = await self.fetch_table_data(docid, sheet_id, view_id, page_token)
            if not data:
                logger.warning("获取分页数据终止 | 无更多数据或请求失败")
                break

            records = data.get("data", [])
            all_data.extend(records)

            if "next_page_token" in data and data["next_page_token"]:
                page_token = data["next_page_token"]
            else:
                break

        logger.info(f"获取所有表格数据完成 | 共{len(all_data)}条记录")
        return all_data

    async def fetch_table_data(self, docid: str, sheet_id: str, view_id: str = None, page_token: str = None) -> Dict:
        """分页获取表格数据"""
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
                logger.warning(f"获取表格数据失败 | 状态码: {response.status_code}")
                return None

            data = response.json()
            if not data.get("success"):
                logger.warning(f"获取表格数据失败 | 消息: {data.get('message', '未知错误')}")
                return None

            return data
        except Exception as e:
            logger.error(f"获取表格数据错误 | {str(e)}")
            return None

    # ---------------------- CSV读取与处理 ----------------------
    def read_and_filter_csv(self, csv_path: str) -> List[Dict]:
        """读取CSV并筛选目标日期数据"""
        try:
            df = pd.read_csv(csv_path, dtype=str)  # 所有字段先按字符串读取
            # 筛选条件：匹配 YYYY-MM-DD 或 毫秒时间戳
            filtered_df = df[
                (df["记录日期"] == self.target_date_str) |
                (df["记录日期"] == self.target_timestamp)
                ]

            if filtered_df.empty:
                logger.warning(f"CSV中未找到目标日期（{self.target_date_str} 或 {self.target_timestamp}）的数据")
                return []

            logger.info(f"从CSV读取到 {len(filtered_df)} 条目标日期数据")
            return filtered_df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"CSV处理失败: {str(e)}")
            return []

    # ---------------------- 数据处理 ----------------------
    def process_individual_data(self, csv_data: List[Dict]) -> List[Dict]:
        """处理个人数据：计算累积值、周数据和环比"""
        if not csv_data:
            return []

        # 提取所有用户ID用于批量查询
        user_ids = [self._extract_field_value(row.get("userid", "")) for row in csv_data
                    if self._extract_field_value(row.get("userid", ""))]
        user_ids = list(filter(None, user_ids))  # 过滤空值

        # 查询各种数据
        logger.info("开始批量查询各种数据用于计算")
        yesterday_data = self.query_yesterday_data(user_ids)
        sevendays_data = self.query_sevendays_data(user_ids)
        lastweek_cumulative_data = self.query_lastweek_cumulative_data(user_ids)  # 新增：查询上周累积数据

        processed = []
        hospital_aggregations = {}  # 用于医院级别的数据聚合

        for row in csv_data:
            userid = self._extract_field_value(row.get("userid", ""))
            name = self._extract_field_value(row.get("姓名", "")).strip()
            hospital_id = self._extract_field_value(row.get("医院ID", "")).strip()

            if not userid and not name:
                logger.warning("跳过缺少userid和姓名的记录")
                continue

            # 初始化医院聚合数据
            if hospital_id not in hospital_aggregations:
                hospital_aggregations[hospital_id] = {
                    "新增客户数": 0,
                    "发送消息数": 0,
                    "累积新增客户数": 0,  # 新增：累积新增客户数汇总
                    "累积消息发送数": 0,  # 新增：累积消息发送数汇总
                    "累积客户群人数": 0,
                    "累积群消息数": 0,
                    "用户数": 0,
                    "近一周新增客户数": 0
                }

            # 1. 清洗CSV当天值
            today_new = self._extract_number_value(row.get("新增客户数", 0))
            today_msg = self._extract_number_value(row.get("发送消息数", 0))
            today_group_members = self._extract_number_value(row.get("截至当天客户群总人数", 0))
            today_group_msg = self._extract_number_value(row.get("截至当天客户群消息总数", 0))

            # 获取前一天累积值
            yesterday_cum = yesterday_data.get(userid, {
                "累积新增客户数": 0,
                "累积消息发送数": 0
            })

            # 获取上周累积值
            lastweek_cum = lastweek_cumulative_data.get(userid, {
                "累积新增客户数": 0,
                "累积消息发送数": 0,
                "累积客户群人数": 0,
                "累积群消息数": 0
            })

            # 计算累积字段
            cumulative_new = yesterday_cum["累积新增客户数"] + today_new
            cumulative_msg = yesterday_cum["累积消息发送数"] + today_msg
            cumulative_group_members = today_group_members  # 直接使用当天的值
            cumulative_group_msg = today_group_msg  # 直接使用当天的值

            # 使用sevendays_data获取周数据
            weekly_new = sevendays_data.get(userid, 0) + today_new

            # 新增：计算环比字段
            # 累积新增客户数环比 = (今天的累积新增客户数 - 上周的累积新增客户数) / 上周的累积新增客户数
            cum_new_lastweek = lastweek_cum["累积新增客户数"]
            cum_new_ratio = self._calculate_ratio(cumulative_new, cum_new_lastweek)

            # 累计消息发送数环比 = (今天的累计消息发送数 - 上周的累计消息发送数) / 上周的累计消息发送数
            cum_msg_lastweek = lastweek_cum["累积消息发送数"]
            cum_msg_ratio = self._calculate_ratio(cumulative_msg, cum_msg_lastweek)

            # 累积客户群人数环比 = (今天的累积客户群人数 - 上周的累积客户群人数) / 上周的累积客户群人数
            cum_members_lastweek = lastweek_cum["累积客户群人数"]
            cum_members_ratio = self._calculate_ratio(cumulative_group_members, cum_members_lastweek)

            # 累积群消息数环比 = (今天的累积群消息数 - 上周的累积群消息数) / 上周的累积群消息数
            cum_msg_group_lastweek = lastweek_cum["累积群消息数"]
            cum_msg_group_ratio = self._calculate_ratio(cumulative_group_msg, cum_msg_group_lastweek)

            # 新增：计算近一周新增客户数环比
            last_week_new = lastweek_cum.get("近一周新增客户数", 0)
            weekly_new_ratio = self._calculate_ratio(weekly_new, last_week_new)

            # 组装清洗后的完整数据
            processed_row = {
                "姓名": name,
                "userid": userid,
                "人员": userid,  # 人员字段传userid
                "记录日期": self.target_timestamp,
                "新增客户数": today_new,
                "发送消息数": today_msg,
                "累积新增客户数": cumulative_new,
                "累积消息发送数": cumulative_msg,
                "近一周新增客户数": weekly_new,
                "累积客户群人数": cumulative_group_members,
                "累积群消息数": cumulative_group_msg,
                # 新增环比字段
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

            # 更新医院聚合数据
            hospital_aggregations[hospital_id]["新增客户数"] += today_new
            hospital_aggregations[hospital_id]["发送消息数"] += today_msg
            hospital_aggregations[hospital_id]["累积新增客户数"] += cumulative_new  # 新增：汇总累积新增客户数
            hospital_aggregations[hospital_id]["累积消息发送数"] += cumulative_msg  # 新增：汇总累积消息发送数
            hospital_aggregations[hospital_id]["累积客户群人数"] += today_group_members
            hospital_aggregations[hospital_id]["累积群消息数"] += today_group_msg
            hospital_aggregations[hospital_id]["用户数"] += 1

        logger.info(f"个人数据处理完成，共生成 {len(processed)} 条有效数据")
        return processed, hospital_aggregations

    # ---------------------- 字段清洗方法 ----------------------
    def _calculate_ratio(self, current_value: float, last_value: float) -> float:
        """计算环比增长率"""
        if last_value == 0:
            return 0.0  # 避免除以零
        return round(((current_value - last_value) / last_value) * 100, 2)  # 返回百分比，保留两位小数


    def _extract_field_value(self, field_data) -> str:
        """更健壮的字段值提取方法，支持多段文本拼接"""
        if not field_data:
            return ""

        # 处理列表类型字段（钉钉API经常返回分段的文本）
        if isinstance(field_data, list):
            full_text = ""
            for item in field_data:
                if isinstance(item, dict):
                    # 尝试获取text/value/content字段
                    text_value = (item.get("text") or
                                  item.get("value") or
                                  item.get("content") or "")
                    if isinstance(text_value, list):
                        # 如果还是列表，递归处理
                        full_text += self._extract_field_value(text_value)
                    else:
                        full_text += str(text_value)
                else:
                    full_text += str(item)
            return full_text.strip()

        # 处理字典类型字段
        if isinstance(field_data, dict):
            return (field_data.get("text") or
                    field_data.get("value") or
                    field_data.get("content") or "")

        # 其他情况直接转为字符串
        return str(field_data).strip()

    def _extract_number_value(self, field_data) -> int:
        """数字字段清洗：处理逗号、文本包装、空值，最终返回整数"""
        value = self._extract_field_value(field_data)
        if not value:
            return 0
        try:
            # 处理千位分隔符（如"1,234"→"1234"）
            if isinstance(value, str) and ',' in value:
                value = value.replace(',', '')
            # 先转float再转int（处理小数如"8.0"→8）
            return int(float(value))
        except (ValueError, TypeError):
            # 提取字符串中的数字（如"新增8人"→8）
            numbers = re.findall(r'\d+', str(value))
            return int(numbers[0]) if numbers else 0

    # ---------------------- 批量写入方法 ----------------------
    def batch_write_individual_data(self, data: List[Dict]) -> bool:
        """批量写入个人数据到客户数据表"""
        if not data:
            logger.info("无个人数据可写入")
            return True

        total = len(data)
        success_count = 0
        failure_count = 0

        for i in range(0, total, self.batch_size):
            batch = data[i:i + self.batch_size]
            logger.info(f"准备写入第 {i // self.batch_size + 1} 批个人数据（共 {len(batch)} 条）")

            # 构造符合要求的写入格式
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
                    # 新增环比字段
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

            # 构造请求体
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
                        logger.info(f"✅ 第 {i // self.batch_size + 1} 批个人数据写入成功")
                    else:
                        failure_count += len(batch)
                        error_msg = result.get("message", "未知错误")
                        logger.error(f"❌ 第 {i // self.batch_size + 1} 批个人数据写入失败: {error_msg}")
                else:
                    failure_count += len(batch)
                    logger.error(f"❌ 第 {i // self.batch_size + 1} 批个人数据写入失败: HTTP {response.status_code}")
            except Exception as e:
                failure_count += len(batch)
                logger.error(f"❌ 第 {i // self.batch_size + 1} 批个人数据写入异常: {str(e)}")

            time.sleep(1.5)  # 避免接口限流

        logger.info(f"个人数据写入完成 | 成功: {success_count}条, 失败: {failure_count}条")
        return failure_count == 0

    def update_hospital_data(self, hospital_aggregations: Dict) -> bool:
        """更新医院数据到地区配置表（包含环比计算）"""
        logger.info("===== 开始更新医院数据（含环比） =====")

        if not hospital_aggregations or not self.region_config:
            logger.info("无医院数据可更新")
            return True

        logger.info(f"医院聚合数据: {json.dumps(hospital_aggregations, indent=2)}")

        # 获取医院ID列表
        hospital_ids = list(hospital_aggregations.keys())
        logger.info(f"需要更新的医院ID: {hospital_ids}")

        # 1. 获取医院近一周数据（从sevendays_data表）
        logger.info("查询医院近一周数据...")
        hospital_weekly_data = self.query_hospital_weekly_data(hospital_ids)
        logger.info(f"医院近一周数据: {hospital_weekly_data}")

        # 2. 获取医院上周累积数据（从lastweek_data表）
        logger.info("查询医院上周累积数据...")
        lastweek_cumulative_data = self.get_lastweek_hospital_cumulative_data(hospital_ids)
        logger.info(f"医院上周累积数据: {lastweek_cumulative_data}")

        # 3. 获取所有医院记录用于更新
        logger.info("获取所有医院记录...")
        hospital_records = self._get_all_hospital_records()
        if not hospital_records:
            logger.error("无法获取医院记录，无法更新医院数据")
            return False

        logger.info(f"获取到 {len(hospital_records)} 条医院记录")

        # 按医院ID分组记录
        hospital_record_map = {}
        for rec in hospital_records:
            values = rec.get("values", {})
            record_id = rec.get("record_id") or rec.get("id")

            # 尝试从多个字段获取医院ID
            hospital_id_value = (
                    self._extract_field_value(values.get("企微通讯录部门名", "")) or
                    self._extract_field_value(values.get("医院ID", "")) or
                    self._extract_field_value(values.get("部门ID", ""))
            )

            if hospital_id_value:
                hospital_record_map[str(hospital_id_value)] = rec
                logger.debug(f"映射医院ID: {hospital_id_value} -> 记录ID: {record_id}")
            else:
                logger.warning(f"记录 {record_id} 缺少医院ID")

        logger.info(f"医院ID映射: {list(hospital_record_map.keys())}")
        logger.info(f"需要更新的医院ID: {list(hospital_aggregations.keys())}")

        total = len(hospital_aggregations)
        success_count = 0
        failure_count = 0

        for hospital_id, agg_data in hospital_aggregations.items():
            logger.info(f"处理医院: {hospital_id}")

            if hospital_id not in hospital_record_map:
                logger.warning(f"未找到医院ID为 {hospital_id} 的记录，跳过更新")
                continue

            hospital_record = hospital_record_map[hospital_id]
            record_id = hospital_record.get("record_id") or hospital_record.get("id")

            if not record_id:
                logger.warning(f"医院ID为 {hospital_id} 的记录缺少record_id，跳过更新")
                continue

            logger.info(f"医院记录ID: {record_id}")

            # 获取员工配置
            logger.info("提取员工表配置...")
            staff_config = self._extract_staff_config(hospital_record)

            # 获取家医团队总人数和已实名认证数
            total_staff, certified_count = 0, 0
            try:
                logger.info("获取员工表统计数据...")
                # 创建事件循环并运行异步方法
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                total_staff, certified_count = loop.run_until_complete(self.get_staff_table_stats(staff_config))
                loop.close()
            except Exception as e:
                logger.error(f"获取员工统计数据失败: {str(e)}")

            logger.info(f"员工统计结果: 总人数={total_staff}, 已认证={certified_count}")

            # 获取上周累积数据
            lastweek_cum = lastweek_cumulative_data.get(hospital_id, {
                "累积新增客户数": 0,
                "累积消息发送数": 0,
                "累积客户群人数": 0,
                "累积群消息数": 0,
                "近一周新增客户数": 0
            })

            # 计算环比增长率
            def calculate_ratio(current, last):
                if last == 0:
                    return 0.0
                return round(((current - last) / last) * 100, 2)

            # 修改：新增客户数环比 = (今天的汇总后累积客户数 - 汇总后上周累积客户数) / 汇总后上周累积客户数
            new_customer_ratio = calculate_ratio(agg_data["累积新增客户数"], lastweek_cum["累积新增客户数"])
            # 修改：发送消息数环比 = (汇总后累积消息发送数 - 汇总后上周消息发送数) / 汇总后上周消息发送数
            message_ratio = calculate_ratio(agg_data["累积消息发送数"], lastweek_cum["累积消息发送数"])
            # 群人数环比 = (今天的累积客户群人数 - 上周累积客户群人数) / 上周累积客户群人数
            group_members_ratio = calculate_ratio(agg_data["累积客户群人数"], lastweek_cum["累积客户群人数"])
            # 群消息数环比 = (今天的累积群消息数 - 上周累积群消息数) / 上周累积群消息数
            group_messages_ratio = calculate_ratio(agg_data["累积群消息数"], lastweek_cum["累积群消息数"])
            # 新增：计算近一周新增客户数环比
            # 计算近一周新增客户数（包括今天）
            current_week_new = hospital_weekly_data.get(hospital_id, 0) + agg_data["新增客户数"]

            # 新增：计算近一周新增客户数环比
            last_week_new = lastweek_cum.get("近一周新增客户数", 0)
            weekly_new_ratio = self._calculate_ratio(current_week_new, last_week_new)

            # 构造更新数据
            update_data = {
                "action": "通用更新表单",
                "company": "花都家庭医生",
                "WordList": {
                    "docid": self.region_config["docid"],
                    "sheet_id": self.region_config["sheet_id"],
                    "record_id": record_id,
                    "values": {
                        # 修改：使用累积数据而不是新增数据
                        "新增客户数": agg_data["累积新增客户数"],
                        "发送消息数": agg_data["累积消息发送数"],
                        "近一周新增客户数": hospital_weekly_data.get(hospital_id, 0) + agg_data["新增客户数"],
                        "截至当天客户群总人数": agg_data["累积客户群人数"],
                        "截至当天客户群消息总数": agg_data["累积群消息数"],
                        "家医团队总人数": total_staff,
                        "已实名认证": certified_count,
                        # 新增环比字段
                        "新增客户数环比": new_customer_ratio,
                        "发送消息数环比": message_ratio,
                        "群人数环比": group_members_ratio,
                        "群消息数环比": group_messages_ratio,
                        "近一周新增客户数环比": weekly_new_ratio
                    },
                    "view_id": self.region_config["view_id"]
                }
            }

            logger.debug(f"更新数据: {json.dumps(update_data, indent=2)}")

            try:
                logger.info(f"发送更新请求...")
                response = self.session.post(SMART_TABLE_URL, json=update_data, timeout=15)
                logger.debug(f"响应状态码: {response.status_code}")
                logger.debug(f"响应内容: {response.text[:500]}...")

                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        success_count += 1
                        logger.info(f"✅ 医院 {hospital_id} 数据更新成功")
                    else:
                        failure_count += 1
                        error_msg = result.get("message", "未知错误")
                        logger.error(f"❌ 医院 {hospital_id} 数据更新失败: {error_msg}")
                else:
                    failure_count += 1
                    logger.error(f"❌ 医院 {hospital_id} 数据更新失败: HTTP {response.status_code}")
            except Exception as e:
                failure_count += 1
                logger.error(f"❌ 医院 {hospital_id} 数据更新异常: {str(e)}")

            time.sleep(1)  # 避免接口限流

        logger.info(f"医院数据更新完成 | 成功: {success_count}家, 失败: {failure_count}家")
        return failure_count == 0

    def get_lastweek_hospital_cumulative_data(self, hospital_ids: List[str]) -> Dict[str, Dict]:
        """查询医院上周的累积数据（用于环比计算）"""
        if not hospital_ids or not self.lastweek_table_config:
            logger.warning("查询条件不满足")
            return {}

        # 构造筛选条件：医院ID在列表中
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
                "field_titles": ["医院ID", "累积新增客户数", "累积消息发送数", "累积客户群人数", "累积群消息数","近一周新增客户数"]
            }
        }

        try:
            response = self.session.post(SMART_TABLE_URL, json=payload, timeout=20)
            result = response.json()
            if not result.get("success"):
                logger.error(f"查询上周累积数据失败: {result.get('message')}")
                return {}

            # 按医院ID分组累积数据
            hospital_data = {}
            for record in result.get("data", []):
                values = record.get("values", {})
                hospital_id = self._extract_field_value(values.get("医院ID", ""))
                if not hospital_id:
                    continue

                # 初始化医院数据
                if hospital_id not in hospital_data:
                    hospital_data[hospital_id] = {
                        "累积新增客户数": 0,
                        "累积消息发送数": 0,
                        "累积客户群人数": 0,
                        "累积群消息数": 0,
                        "近一周新增客户数": 0
                    }

                # 累加数值
                hospital_data[hospital_id]["累积新增客户数"] += self._extract_number_value(values.get("累积新增客户数", 0))
                hospital_data[hospital_id]["累积消息发送数"] += self._extract_number_value(values.get("累积消息发送数", 0))
                hospital_data[hospital_id]["累积客户群人数"] += self._extract_number_value(values.get("累积客户群人数", 0))
                hospital_data[hospital_id]["累积群消息数"] += self._extract_number_value(values.get("累积群消息数", 0))
                hospital_data[hospital_id]["近一周新增客户数"] += self._extract_number_value(values.get("近一周新增客户数", 0))

            logger.info(f"查询到 {len(hospital_data)} 个医院的上周累积数据")
            return hospital_data
        except Exception as e:
            logger.error(f"查询上周累积数据异常: {str(e)}")
            return {}

    def query_hospital_weekly_data(self, hospital_ids: List[str]) -> Dict[str, int]:
        """查询医院近一周的新增客户数（从sevendays_data表）"""
        if not hospital_ids or not self.sevendays_table_config:
            return {}

        # 构造筛选条件：医院ID在列表中
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
                logger.error(f"查询医院周数据失败: {result.get('message')}")
                return {}

            # 汇总医院近一周新增客户数
            weekly_data = {hospital_id: 0 for hospital_id in hospital_ids}
            for record in result.get("data", []):
                values = record.get("values", {})
                hospital_id = self._extract_field_value(values.get("医院ID", ""))
                if hospital_id in weekly_data:
                    weekly_data[hospital_id] += self._extract_number_value(values.get("新增客户数", 0))

            logger.info(f"查询到 {len(weekly_data)} 个医院的近一周数据")
            return weekly_data
        except Exception as e:
            logger.error(f"查询医院周数据异常: {str(e)}")
            return {}

    def _get_all_hospital_records(self) -> List[Dict]:
        """获取所有医院记录"""
        logger.info("获取所有医院记录...")

        if not self.region_config:
            logger.error("缺少地区配置表参数")
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
            logger.debug(f"请求参数: {json.dumps(params)}")
            response = self.session.post(SMART_TABLE_URL, json=params, timeout=15)
            logger.debug(f"响应状态码: {response.status_code}")

            result = response.json()
            logger.debug(f"响应内容: {json.dumps(result, indent=2)[:500]}...")

            if result.get("success") and "data" in result:
                records = result["data"]
                logger.info(f"获取医院记录成功 | 共{len(records)}条")
                return records
            else:
                error_msg = result.get("message", "未知错误")
                logger.error(f"获取医院记录失败: {error_msg}")
                return []
        except Exception as e:
            logger.error(f"获取医院记录异常: {str(e)}")
            return []

    def _extract_staff_config(self, hospital_record: Dict) -> Dict:
        """从医院记录中提取员工表配置（完整版）"""
        logger.info(f"开始提取员工表配置: {hospital_record.get('record_id')}")
        values = hospital_record.get("values", {})
        doc_config_str = self._extract_field_value(values.get("文档ID"))

        # logger.debug(f"原始文档配置字符串: {doc_config_str}")

        staff_config = {}

        # 第一步：提取根层级的docid
        docid_match = re.search(r'"docid"\s*:\s*"([^"]+)"', doc_config_str)
        if docid_match:
            staff_config["docid"] = docid_match.group(1)
            # logger.info(f"正则提取docid成功: {staff_config['docid']}")
        else:
            logger.warning("未找到docid配置")

        # 第二步：提取staff配置
        staff_match = re.search(
            r'"staff"\s*:\s*{\s*"tab"\s*:\s*"([^"]+)"\s*,\s*"viewId"\s*:\s*"([^"]+)"',
            doc_config_str,
            re.DOTALL
        )
        if staff_match:
            staff_config["tab"] = staff_match.group(1)
            staff_config["view_id"] = staff_match.group(2)
            # logger.info(f"正则提取staff配置成功: tab={staff_config['tab']}, view_id={staff_config['view_id']}")
        else:
            logger.warning("正则未找到staff配置")

        # 第三步：如果正则提取失败，尝试JSON解析
        if not staff_config.get("tab") or not staff_config.get("view_id") or not staff_config.get("docid"):
            try:
                config_data = json.loads(doc_config_str)
                # 提取docid
                staff_config["docid"] = config_data.get("docid", staff_config.get("docid", ""))
                # 提取staff配置
                staff_data = config_data.get("staff", {})
                if isinstance(staff_data, dict):
                    staff_config["tab"] = staff_data.get("tab") or staff_config.get("tab", "")
                    staff_config["view_id"] = staff_data.get("viewId", staff_data.get("view_id")) or staff_config.get(
                        "view_id", "")

                   # logger.info(f"JSON解析提取成功: {staff_config}")
            except Exception as e:
                logger.error(f"JSON解析失败: {str(e)}")

        # 验证配置完整性
        required_keys = ["docid", "tab", "view_id"]
        if not all(key in staff_config and staff_config[key] for key in required_keys):
            logger.error("❌ 员工表配置不完整，无法获取数据")
            return {}

        return staff_config

    async def get_staff_table_stats(self, staff_config: Dict) -> tuple[int, int]:
        """读取员工表总行数、二维码非空数"""
        logger.info(f"开始获取员工表统计 | 配置: {staff_config}")

        if not staff_config or "docid" not in staff_config or "tab" not in staff_config:
            logger.error("❌ 员工表配置不完整，无法获取数据")
            return 0, 0

        try:
            # 获取员工表所有数据
            staff_records = await self.get_all_table_data(
                staff_config["docid"],
                staff_config["tab"],
                staff_config.get("view_id")
            )

            if not staff_records:
                logger.warning("⚠️ 员工表无数据")
                return 0, 0

            total_rows = len(staff_records)  # 总行数（家医团队总人数）
            certified_count = 0  # 已实名认证数（二维码非空）
            qr_code_empty_count = 0  # 二维码为空的数量

            logger.info(f"员工表记录数: {total_rows}")

            for idx, record in enumerate(staff_records, 1):
                values = record.get("values", {})
                qr_code = self._extract_field_value(values.get("本人的二维码链接", ""))

                if idx <= 5:  # 只打印前5条记录的二维码值
                    logger.debug(f"记录#{idx} 二维码值: {qr_code[:50]}{'...' if len(qr_code) > 50 else ''}")

                if qr_code.strip():
                    certified_count += 1
                else:
                    qr_code_empty_count += 1

            logger.info(f"员工表统计结果: 总人数={total_rows}, 已实名认证={certified_count}, 未认证={qr_code_empty_count}")
            return total_rows, certified_count
        except Exception as e:
            logger.error(f"获取员工表统计异常: {str(e)}")
            return 0, 0

    def query_lastweek_cumulative_data(self, user_ids: List[str]) -> Dict[str, Dict]:
        """查询用户上周的累积数据（用于环比计算）"""
        if not user_ids or not self.lastweek_table_config:
            return {}

        # 构造筛选条件：用户ID在列表中
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
                "field_titles": ["人员", "累积新增客户数", "累积消息发送数", "累积客户群人数", "累积群消息数","近一周新增客户数"]
            }
        }

        try:
            response = self.session.post(SMART_TABLE_URL, json=payload, timeout=20)
            result = response.json()
            if not result.get("success"):
                logger.error(f"查询上周累积数据失败: {result.get('message')}")
                return {}

            # 提取上周累积数据
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

            logger.info(f"查询到 {len(lastweek_data)} 个用户的上周累积数据")
            return lastweek_data
        except Exception as e:
            logger.error(f"查询上周累积数据异常: {str(e)}")
            return {}



    # 新增方法：查询用户前一天的数据
    def query_yesterday_data(self, user_ids: List[str]) -> Dict[str, Dict]:
        """查询用户前一天的累积数据（根据实际数据结构调整）"""
        if not user_ids or not self.yesterday_table_config:
            logger.warning(f"查询条件不满足: user_ids={len(user_ids)} yesterday_table_config={self.yesterday_table_config}")
            return {}

        logger.info(f"开始查询前一天数据 | 用户数: {len(user_ids)}")
        
        all_yesterday_data = {}

        try:
            # 查询视图数据
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
                logger.error(f"查询失败: {result.get('message', '未知错误')}")
                return {}

            records = result.get("data", [])
            logger.info(f"视图中总共有 {len(records)} 条记录")
            
            # 处理查询结果
            for record in records:
                values = record.get("values", {})
                
                # 提取用户ID
                userid = None
                person_field = values.get("人员")
                
                if isinstance(person_field, list) and len(person_field) > 0:
                    person_obj = person_field[0]
                    if isinstance(person_obj, dict):
                        userid = person_obj.get("user_id")
                elif isinstance(person_field, str):
                    userid = person_field
                
                if userid and userid in user_ids:
                    # 提取数值字段
                    new_customers = self._extract_number_value(values.get("累积新增客户数", 0))
                    messages_sent = self._extract_number_value(values.get("累积消息发送数", 0))
                    
                    all_yesterday_data[userid] = {
                        "累积新增客户数": new_customers,
                        "累积消息发送数": messages_sent
                    }

        except Exception as e:
            logger.error(f"查询异常: {str(e)}")
            import traceback
            logger.error(f"异常详情: {traceback.format_exc()}")

        found_users = len(all_yesterday_data)
        missing_users = len(user_ids) - found_users
        if missing_users > 0:
            logger.warning(f"缺失 {missing_users} 个用户的数据")

        logger.info(f"前一天数据查询完成 | 找到 {found_users} 个用户的数据")
        return all_yesterday_data

    def query_sevendays_data(self, user_ids: List[str]) -> Dict[str, int]:
        """查询用户近一周的新增客户数（使用sevendays_data配置）"""
        if not user_ids or not self.sevendays_table_config:
            return {}

        # 构造筛选条件：用户ID在列表中且记录日期在近一周范围内（不包括今天）
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
                logger.error(f"查询近一周数据失败: {result.get('message')}")
                return {}

            # 汇总近一周新增客户数（不包括今天）
            weekly_data = {user_id: 0 for user_id in user_ids}
            for record in result.get("data", []):
                values = record.get("values", {})
                userid = values.get("人员", [{}])[0].get("user_id", "")
                if userid in weekly_data:
                    weekly_data[userid] += self._extract_number_value(values.get("新增客户数", 0))

            logger.info(f"使用sevendays_data查询到 {len(weekly_data)} 个用户的近一周数据（不包括今天）")
            return weekly_data
        except Exception as e:
            logger.error(f"查询近一周数据异常: {str(e)}")
            return {}

    # ---------------------- 主流程 ----------------------
    def run(self, csv_path: str) -> bool:
        """执行完整流程"""
        try:
            # 1. 获取钉钉配置
            if not self.get_dingtalk_access_token():
                return False

            if not self.get_ranking_config():
                return False

            if not self.get_region_config_data():
                return False

            # 2. 获取企微访问令牌
            if not self.get_access_token():
                return False

            # 3. 读取并筛选CSV数据
            csv_data = self.read_and_filter_csv(csv_path)
            if not csv_data:
                return False

            # 4. 处理个人数据和医院聚合数据
            processed_data, hospital_aggregations = self.process_individual_data(csv_data)
            if not processed_data:
                logger.error("个人数据处理后为空")
                return False

            # 5. 批量写入个人数据到客户数据表
            if not self.batch_write_individual_data(processed_data):
                logger.error("个人数据写入失败")
                # 可以根据实际需求决定是否继续执行医院数据更新
                # return False

            # 6. 更新医院数据到地区配置表
            if not self.update_hospital_data(hospital_aggregations):
                logger.error("医院数据更新失败")
                return False

            logger.info("全部流程执行成功")
            return True

        except Exception as e:
            logger.error(f"执行异常: {str(e)}", exc_info=True)
            return False
        finally:
            self.session.close()
            logger.info("关闭网络会话")


if __name__ == "__main__":
    # 直接指定CSV文件路径
    CSV_FILE_PATH = "wechat_stats_20250901.csv"  # 这里填写你的CSV文件名

    importer = CSVToWeChatCombinedImporter()
    success = importer.run(CSV_FILE_PATH)
    sys.exit(0 if success else 1)