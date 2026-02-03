import requests
import json
import urllib.parse
import schedule
import time
import os
import re
from datetime import datetime, date, timedelta
import logging

# -------------------------- 1. 基础配置 --------------------------
DINGTALK_CONFIG = {
    "app_key": "dingczeweiukv9kue2gv",
    "app_secret": "BC11ILonRquetv-aTv6lrfUlqWHjDrikSQN9NWHhxRHVz8xYQGcnLgtL6h1SPiPU",
    "config_base_id": "pYLaezmVNev7pRZ9t4oxG9aQWrMqPxX6",
    "config_sheet_name": "配置表",
    "operator_id": "xYLFMT7vpx2nLD5iiW81omAiEiE",
    "token_cache_file": "dingtalk_stretch_token_cache.json"
}

# -------------------------- 2. 日志配置 --------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stretch_scheduler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# -------------------------- 3. 工具函数 --------------------------
def safe_strip(value):
    """安全处理字符串，兼容None和非字符串类型"""
    if value is None:
        return ""
    return str(value).strip()


def parse_dingtalk_table_url(table_url):
    """解析钉钉多维表链接，提取base_id和sheet_id"""
    try:
        parsed = urllib.parse.urlparse(table_url)
        query_params = urllib.parse.parse_qs(parsed.query)

        # 1. 优先从常规查询参数提取（旧格式）
        base_id = safe_strip(query_params.get("baseId", [None])[0])
        sheet_id = safe_strip(query_params.get("sheetId", [None])[0])

        # 2. 处理alidocs.dingtalk.com新格式
        if not (base_id and sheet_id) and parsed.netloc == "alidocs.dingtalk.com":
            # 从iframeQuery中解析sheetId
            iframe_query = query_params.get("iframeQuery", [None])[0]
            if iframe_query:
                iframe_params = urllib.parse.parse_qs(iframe_query)
                sheet_id = safe_strip(iframe_params.get("sheetId", [None])[0])

            # 从路径提取baseId
            path_parts = parsed.path.split("/")
            if len(path_parts) >= 4 and path_parts[2] == "nodes":
                base_id = path_parts[3]

        # 3. 兼容旧格式（/bases/{baseId}/sheets/{sheetId}路径）
        if not (base_id and sheet_id):
            path_parts = parsed.path.split("/")
            if len(path_parts) >= 5 and path_parts[3] == "bases":
                base_id = path_parts[4]
                sheet_id = path_parts[6] if len(path_parts) >= 7 else None

        if base_id and sheet_id:
            logger.info(f"✅ 解析表链接成功：base_id={base_id}, sheet_id={sheet_id}")
            return {"base_id": base_id, "sheet_id": sheet_id}
        else:
            logger.error(f"❌ 无法解析表链接，缺失参数：base_id={base_id}, sheet_id={sheet_id} | 链接={table_url}")
            return None
    except Exception as e:
        logger.error(f"❌ 解析表链接出错：{e} | 链接={table_url}")
        return None


# -------------------------- 4. 钉钉Token管理 --------------------------
def load_cached_token():
    """加载Token缓存"""
    cache_file = DINGTALK_CONFIG["token_cache_file"]
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)

            expire_time = datetime.fromisoformat(cache["expire_time"])
            if datetime.now() < expire_time - timedelta(minutes=5):
                logger.info("🔄 使用缓存的Token")
                return cache["access_token"]
            logger.info("⏰ Token即将过期，需重新获取")
    except Exception as e:
        logger.warning(f"⚠️ 读取Token缓存失败：{e}")
    return None


def save_token_to_cache(access_token, expires_in=7200):
    """保存Token到缓存"""
    cache_file = DINGTALK_CONFIG["token_cache_file"]
    try:
        expire_time = datetime.now() + timedelta(seconds=expires_in)
        cache_data = {
            "access_token": access_token,
            "expire_time": expire_time.isoformat(),
            "created_time": datetime.now().isoformat()
        }
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Token已缓存，过期时间：{expire_time.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        logger.error(f"❌ 保存Token缓存失败：{e}")


def get_dingtalk_access_token():
    """获取钉钉Access Token（带缓存和重试机制）"""
    cached_token = load_cached_token()
    if cached_token:
        return cached_token

    logger.info("🔄🔄 重新获取Token...")
    url = "https://api.dingtalk.com/v1.0/oauth2/accessToken"
    headers = {"Content-Type": "application/json"}
    payload = {
        "appKey": DINGTALK_CONFIG["app_key"],
        "appSecret": DINGTALK_CONFIG["app_secret"]
    }

    # 创建带重试机制的会话
    session = requests.Session()
    retry_strategy = requests.adapters.Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    try:
        response = session.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        access_token = data.get("accessToken")
        expires_in = data.get("expireIn", 7200)

        if access_token:
            save_token_to_cache(access_token, expires_in)
            logger.info("✅ Token获取成功")
            return access_token
        logger.error("❌❌ 响应中无Token")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌❌ 获取Token失败（重试后）：{e}")
    except Exception as e:
        logger.error(f"❌❌ 获取Token发生意外错误：{e}")
    return None


# -------------------------- 5. 读取钉钉配置表（支持多推送时间） --------------------------
def get_task_configs():
    """从钉钉配置表获取任务（支持推送时间多值，如8:00,10:00）"""
    logger.info("🔄 读取钉钉配置表...")
    access_token = get_dingtalk_access_token()
    if not access_token:
        logger.error("❌ 无Token，无法读取配置表")
        return []

    base_id = DINGTALK_CONFIG["config_base_id"]
    sheet_name = urllib.parse.quote(DINGTALK_CONFIG["config_sheet_name"])
    url = f"https://api.dingtalk.com/v1.0/notable/bases/{base_id}/sheets/{sheet_name}/records"
    headers = {
        "x-acs-dingtalk-access-token": access_token,
        "Content-Type": "application/json"
    }
    params = {"maxResults": 100, "operatorId": DINGTALK_CONFIG["operator_id"]}

    # 三次重试配置
    max_retries = 3
    retry_interval = 2
    retry_count = 0

    while retry_count < max_retries:
        try:
            logger.info(f"🔄 第{retry_count + 1}次请求配置表API")
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=15,
                verify=True
            )
            response.raise_for_status()
            data = response.json()
            records = data.get("records", [])

            if not records:
                logger.warning("⚠️ 配置表中无记录")
                return []

            task_configs = []
            for idx, record in enumerate(records, 1):
                fields = record.get("fields", {})
                record_id = safe_strip(record.get("recordId") or record.get("id"))

                # 处理勾选框类型"是否启用"
                is_enabled_checkbox = fields.get("是否启用", False)
                is_enabled = "已启用" if is_enabled_checkbox else "未启用"

                # 解析webhook（从link字段取URL）
                webhook_field = fields.get("webhook", {})
                webhook_title = safe_strip(webhook_field.get("text", ""))
                webhook_url = safe_strip(webhook_field.get("link", ""))

                # 解析表链接（从link字段取URL）
                table_url_field = fields.get("表链接", {})
                table_url = safe_strip(table_url_field.get("link", ""))

                # 解析定时类型
                cron_type_field = fields.get("定时类型", {})
                cron_type = safe_strip(cron_type_field.get("name", ""))

                # 获取@人字段的名称
                at_field_name = safe_strip(fields.get("@人字段", ""))

                # 处理"最新完成日期"字段
                latest_exec_date_field = safe_strip(fields.get("最新完成日期", ""))
                latest_exec_date = ""
                if latest_exec_date_field:
                    try:
                        clean_date_str = re.sub(r'[\s\xa0]+', ' ', latest_exec_date_field).strip()
                        clean_date_str = clean_date_str.replace('：', ':')

                        if clean_date_str.isdigit():
                            timestamp_seconds = int(clean_date_str) / 1000
                            date_obj = datetime.fromtimestamp(timestamp_seconds)
                            latest_exec_date = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                            logger.info(f"✅ 转换最新完成日期时间戳: {clean_date_str} -> {latest_exec_date}")
                        else:
                            formats = [
                                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
                                "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M",
                                "%Y-%m-%d", "%Y/%m/%d"
                            ]

                            date_obj = None
                            for fmt in formats:
                                try:
                                    date_obj = datetime.strptime(clean_date_str, fmt)
                                    break
                                except ValueError:
                                    continue

                            if not date_obj:
                                parts = clean_date_str.split(' ')
                                date_part = parts[0] if len(parts) >= 1 else None
                                time_part = parts[1] if len(parts) >= 2 else None

                                if date_part:
                                    date_formats = ["%Y-%m-%d", "%Y/%m/%d"]
                                    for df in date_formats:
                                        try:
                                            date_obj = datetime.strptime(date_part, df)
                                            break
                                        except ValueError:
                                            continue

                                if time_part and date_obj:
                                    time_formats = ["%H:%M:%S", "%H:%M"]
                                    for tf in time_formats:
                                        try:
                                            time_obj = datetime.strptime(time_part, tf)
                                            date_obj = date_obj.replace(
                                                hour=time_obj.hour,
                                                minute=time_obj.minute,
                                                second=time_obj.second
                                            )
                                            break
                                        except ValueError:
                                            continue

                            if not date_obj:
                                raise ValueError(f"所有格式均匹配失败: {clean_date_str}")

                            latest_exec_date = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                            logger.info(f"✅ 转换最新完成日期: {clean_date_str} -> {latest_exec_date}")

                    except Exception as e:
                        logger.warning(f"⚠️ 转换最新完成日期失败: {e}，原始值={latest_exec_date_field}，清理后={clean_date_str}")
                        latest_exec_date = ""
                else:
                    latest_exec_date = ""

                # 解析多推送时间（核心修改）
                push_time_raw = safe_strip(fields.get("推送时间", ""))
                push_times = []  # 存储解析后的有效时间列表[(hour, minute), ...]
                if push_time_raw:
                    time_str_list = [t.strip() for t in push_time_raw.split(',') if t.strip()]
                    for time_str in time_str_list:
                        time_str = time_str.replace("：", ":")
                        if re.match(r'^\d{1,2}:\d{2}$', time_str):
                            hour, minute = map(int, time_str.split(":"))
                            if 0 <= hour <= 23 and 0 <= minute <= 59:
                                push_times.append((hour, minute))
                                logger.info(f"✅ 解析有效推送时间：{time_str}（任务{idx}）")
                            else:
                                logger.warning(f"⚠️ 推送时间{time_str}超出范围，跳过（任务{idx}）")
                        else:
                            logger.warning(f"⚠️ 推送时间{time_str}格式无效，跳过（任务{idx}）")

                # 构建任务字典
                task = {
                    "record_id": record_id,
                    "is_enabled": is_enabled,
                    "cron_type": cron_type,
                    "cron_extra": safe_strip(fields.get("每周-周几", "")) or
                                  safe_strip(fields.get("每月-几号", "")),
                    "push_times": push_times,  # 多推送时间列表
                    "push_time_raw": push_time_raw,  # 原始字符串
                    "table_url": table_url,
                    "target_field": safe_strip(fields.get("字段", "")),
                    "webhook_title": webhook_title,
                    "webhook_url": webhook_url,
                    "at_field_name": at_field_name,
                    "latest_exec_date": latest_exec_date
                }

                # 过滤无效任务
                missing_fields = []
                if not task["is_enabled"].startswith("已启用"):
                    missing_fields.append(f"是否启用（当前：{task['is_enabled']}）")
                if not task["cron_type"]:
                    missing_fields.append("定时类型")
                if not task["push_times"]:
                    missing_fields.append(f"推送时间（原始值：{task['push_time_raw']}，无有效时间）")
                if not task["table_url"]:
                    missing_fields.append("表链接")
                if not task["target_field"]:
                    missing_fields.append("字段")
                if not task["webhook_url"]:
                    missing_fields.append("webhook_url")
                if task["cron_type"] == "每周" and not task["cron_extra"]:
                    missing_fields.append("每周-周几")
                if task["cron_type"] == "每月" and not task["cron_extra"]:
                    missing_fields.append("每月-几号")

                if missing_fields:
                    logger.warning(f"⚠️ 任务{idx}配置不完整（跳过），缺失：{','.join(missing_fields)}")
                    continue

                task_configs.append(task)
                push_times_str = ",".join([f"{h:02d}:{m:02d}" for h, m in task["push_times"]])
                logger.info(
                    f"✅ 有效任务{idx}：{task['is_enabled']} | 定时={task['cron_type']}{task['cron_extra'] if task['cron_extra'] else ''} {push_times_str}")

            logger.info(f"📊 共获取{len(task_configs)}个有效启用任务")
            return task_configs

        except requests.exceptions.ConnectionError as e:
            retry_count += 1
            if retry_count < max_retries:
                logger.warning(f"⚠️ 第{retry_count}次请求失败（网络问题）：{str(e)[:100]}，{retry_interval}秒后重试")
                time.sleep(retry_interval)
            else:
                logger.error(f"❌ 三次请求均失败（网络问题）：{str(e)[:100]}")
                return []

        except requests.exceptions.Timeout as e:
            retry_count += 1
            if retry_count < max_retries:
                logger.warning(f"⚠️ 第{retry_count}次请求失败（超时）：{str(e)[:100]}，{retry_interval}秒后重试")
                time.sleep(retry_interval)
            else:
                logger.error(f"❌ 三次请求均失败（超时）：{str(e)[:100]}")
                return []

        except Exception as e:
            logger.error(f"❌ 读取配置表失败（非网络问题）：{e}")
            return []


# -------------------------- 6. 定时规则解析（适配多推送时间） --------------------------
def parse_cron_config(task):
    """解析任务的定时规则（支持多推送时间）"""
    cron_type = task["cron_type"]
    cron_extra = task["cron_extra"]
    push_times = task["push_times"]

    if not push_times:
        logger.warning(f"⚠️ 任务{task['record_id']}无有效推送时间，跳过解析")
        return None

    try:
        if cron_type == "每日":
            return {
                "type": "daily",
                "push_times": push_times,
                "desc": f"每日{','.join([f'{h:02d}:{m:02d}' for h, m in push_times])}"
            }

        elif cron_type == "每周":
            weekday_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "日": 0}
            if cron_extra not in weekday_map:
                logger.warning(f"⚠️ 无效周几：{cron_extra}（任务ID：{task['record_id']}）")
                return None
            return {
                "type": "weekly",
                "weekday": weekday_map[cron_extra],
                "push_times": push_times,
                "desc": f"每周{cron_extra} {','.join([f'{h:02d}:{m:02d}' for h, m in push_times])}"
            }

        elif cron_type == "每月":
            if not cron_extra.isdigit() or not (1 <= int(cron_extra) <= 31):
                logger.warning(f"⚠️ 无效几号：{cron_extra}（任务ID：{task['record_id']}）")
                return None
            return {
                "type": "monthly",
                "day": int(cron_extra),
                "push_times": push_times,
                "desc": f"每月{cron_extra}号 {','.join([f'{h:02d}:{m:02d}' for h, m in push_times])}"
            }

        elif cron_type == "单次":
            return {
                "type": "once",
                "push_times": push_times,
                "desc": f"每日{','.join([f'{h:02d}:{m:02d}' for h, m in push_times])}（单次任务，筛选新增数据）"
            }

        else:
            logger.warning(f"⚠️ 不支持的定时类型：{cron_type}（任务ID：{task['record_id']}）")
            return None
    except Exception as e:
        logger.error(f"❌ 解析定时规则失败：{e}（任务ID：{task['record_id']}）")
        return None


def should_execute_now(cron_config):
    """判断当前时间是否符合定时规则（适配多推送时间）"""
    if not cron_config or "push_times" not in cron_config:
        return False
    now = datetime.now()
    current_hour = now.hour
    current_minute = now.minute
    current_weekday = now.weekday()
    current_day = now.day

    ding_weekday = current_weekday + 1 if current_weekday != 6 else 0

    # 检查是否匹配任意推送时间点
    is_time_match = any(
        (h == current_hour and m == current_minute)
        for h, m in cron_config["push_times"]
    )
    if not is_time_match:
        return False

    # 检查日期是否匹配
    if cron_config["type"] == "daily":
        return True
    elif cron_config["type"] == "weekly":
        return ding_weekday == cron_config["weekday"]
    elif cron_config["type"] == "monthly":
        return current_day == cron_config["day"]
    elif cron_config["type"] == "once":
        return True
    return False


def is_missed_today_task(cron_config, latest_exec_date):
    """判断是否为今日未执行的任务（适配多推送时间）"""
    if not cron_config or not latest_exec_date or "push_times" not in cron_config:
        return False

    today = date.today()
    try:
        datetime_formats = [
            "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M",
            "%Y-%m-%d", "%Y/%m/%d"
        ]
        last_exec_datetime = None
        for fmt in datetime_formats:
            try:
                last_exec_datetime = datetime.strptime(latest_exec_date, fmt)
                break
            except ValueError:
                continue

        if last_exec_datetime is None and latest_exec_date.isdigit():
            timestamp_seconds = int(latest_exec_date) / 1000
            last_exec_datetime = datetime.fromtimestamp(timestamp_seconds)
        if last_exec_datetime is None:
            raise ValueError(f"无法识别的日期格式：{latest_exec_date}")

        last_exec_date = last_exec_datetime.date()
        if last_exec_date >= today:
            return False

    except Exception as e:
        logger.warning(f"⚠️ 解析最新执行日期失败：{e}，日期值：{latest_exec_date}")
        return False

    now = datetime.now()
    current_hour = now.hour
    current_minute = now.minute
    current_weekday = now.weekday()
    ding_weekday = current_weekday + 1 if current_weekday != 6 else 0

    # 判断日期是否匹配
    is_date_match = False
    if cron_config["type"] in ["daily", "once"]:
        is_date_match = True
    elif cron_config["type"] == "weekly":
        is_date_match = (ding_weekday == cron_config["weekday"])
    elif cron_config["type"] == "monthly":
        is_date_match = (today.day == cron_config["day"])

    if not is_date_match:
        return False

    # 检查是否有未执行的时间点
    for h, m in cron_config["push_times"]:
        if (current_hour > h) or (current_hour == h and current_minute > m):
            target_datetime = datetime.combine(today, datetime.min.time()).replace(hour=h, minute=m)
            if target_datetime > last_exec_datetime:
                return True

    return False


# -------------------------- 7. 数据提取 --------------------------
def get_table_records(table_url, target_field, cron_type="", latest_exec_date="", at_field_name=""):
    """从目标表提取记录，单次任务筛选新增数据"""
    logger.info(f"🔍 提取表数据：链接={table_url}，字段={target_field}，类型={cron_type}")

    table_info = parse_dingtalk_table_url(table_url)
    if not table_info:
        logger.error(f"❌ 无法解析表链接：{table_url}")
        return []
    base_id = table_info["base_id"]
    sheet_id = table_info["sheet_id"]

    access_token = get_dingtalk_access_token()
    if not access_token:
        logger.error("❌ 无Token，无法读取表数据")
        return []

    # 分页读取所有记录
    all_records = []
    next_token = None
    page_count = 0

    while True:
        page_count += 1
        logger.info(f"📄 读取第{page_count}页数据...")

        url = f"https://api.dingtalk.com/v1.0/notable/bases/{base_id}/sheets/{sheet_id}/records"
        headers = {
            "x-acs-dingtalk-access-token": access_token,
            "Content-Type": "application/json"
        }

        params = {
            "maxResults": 100,
            "operatorId": DINGTALK_CONFIG["operator_id"]
        }

        if next_token:
            params["nextToken"] = next_token

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            records = data.get("records", [])
            all_records.extend(records)

            logger.info(f"✅ 第{page_count}页获取{len(records)}条记录")

            next_token = data.get("nextToken")
            if not next_token:
                logger.info(f"📊 所有数据读取完成，共{len(all_records)}条记录")
                break

        except Exception as e:
            logger.error(f"❌ 第{page_count}页读取失败：{e}")
            break

    # 单次任务筛选新增数据
    filtered_records = all_records
    if cron_type == "once":
        filtered_records = []
        if not latest_exec_date:
            base_datetime = datetime.now() - timedelta(days=7)
            logger.info(f"⚠️ 单次任务无最新完成日期，默认筛选7天内数据（基准时间：{base_datetime.strftime('%Y-%m-%d %H:%M:%S')}）")
        else:
            try:
                datetime_formats = [
                    "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
                    "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M",
                    "%Y-%m-%d", "%Y/%m/%d"
                ]
                base_datetime = None
                for fmt in datetime_formats:
                    try:
                        base_datetime = datetime.strptime(latest_exec_date, fmt)
                        break
                    except ValueError:
                        continue
                if base_datetime is None and latest_exec_date.isdigit():
                    timestamp_seconds = int(latest_exec_date) / 1000
                    base_datetime = datetime.fromtimestamp(timestamp_seconds)
                if base_datetime is None:
                    raise ValueError(f"无法解析最新完成日期：{latest_exec_date}")
                logger.info(f"🔍 单次任务筛选基准：创建时间 > {base_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception as e:
                logger.warning(f"⚠️ 解析最新完成日期失败，默认筛选7天内数据：{e}")
                base_datetime = datetime.now() - timedelta(days=7)

        # 筛选逻辑：创建时间 > 基准时间
        datetime_formats = [
            "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M",
            "%Y-%m-%d", "%Y/%m/%d"
        ]

        for record in all_records:
            fields = record.get("fields", {})
            record_id = safe_strip(record.get("recordId") or record.get("id"))
            create_time_value = safe_strip(fields.get("创建时间", ""))

            if not create_time_value:
                logger.debug(f"⚠️ 记录{record_id}无「创建时间」字段，跳过")
                continue

            try:
                create_time = None
                for fmt in datetime_formats:
                    try:
                        create_time = datetime.strptime(create_time_value, fmt)
                        break
                    except ValueError:
                        continue

                if create_time is None and create_time_value.isdigit():
                    timestamp_seconds = int(create_time_value) / 1000
                    create_time = datetime.fromtimestamp(timestamp_seconds)
                if create_time is None:
                    raise ValueError(f"无法识别的创建时间格式：{create_time_value}")

                if create_time > base_datetime:
                    filtered_records.append(record)
                    logger.info(f"✅ 记录{record_id}符合条件：创建时间={create_time.strftime('%Y-%m-%d %H:%M:%S')} > 基准时间")
                else:
                    logger.debug(f"❌ 记录{record_id}不符合条件：创建时间≤基准时间")

            except Exception as e:
                logger.warning(f"⚠️ 解析记录{record_id}的「创建时间」失败：{e}，值={create_time_value}")
                continue

        logger.info(f"🔍 单次任务筛选完成：原{len(all_records)}条 → 保留{len(filtered_records)}条新增记录")

    # 提取内容和@人信息
    content_list = []
    for record in filtered_records:
        fields = record.get("fields", {})
        record_id = safe_strip(record.get("recordId") or record.get("id"))

        target_value = fields.get(target_field, "")
        if isinstance(target_value, dict):
            content = safe_strip(target_value.get("text", "") or target_value.get("name", ""))
        else:
            content = safe_strip(target_value)

        user_id = ""
        if at_field_name:
            at_value = safe_strip(fields.get(at_field_name, ""))
            logger.info(f"📌 记录{record_id}：@人字段名称={at_field_name}，原始值={at_value}")
            user_id = at_value

        if content:
            content_list.append({"content": content, "user_id": user_id})
            logger.info(f"📝 记录{record_id}：content={content[:50]}...，user_id={'[空]' if not user_id else user_id}")

    return content_list


# -------------------------- 8. Webhook发送 --------------------------
def send_webhook(task, content, user_id=""):
    """发送Webhook消息，支持@指定用户"""
    if not task["webhook_url"]:
        logger.error("❌ Webhook URL 为空，无法发送")
        return False

    title = task["webhook_title"] or ""
    body = content

    if not title:
        title_match = re.search(r'^#\s*(.+?)(?:\n|$)', content)
        if title_match:
            title = title_match.group(1).strip()
            logger.info(f"✅ 从内容中提取到标题: '{title}'")
        else:
            title = content[:30].strip() + "..." if len(content) > 30 else content
            logger.info(f"ℹ️ 未找到标题格式，使用内容前部分作为标题: '{title}'")

    markdown_text = f"{body}\n\n> 本消息由定时任务自动发送"
    if user_id.strip():
        markdown_text += f"\n\n@{user_id}"

    message = {
        "msgtype": "markdown",
        "markdown": {
            "title": title,
            "text": markdown_text
        },
        "at": {
            "atUserIds": [user_id.strip()] if user_id.strip() else [],
            "isAtAll": False
        }
    }

    message_log = json.dumps(message, ensure_ascii=False, indent=2).replace(task["webhook_url"], "***")
    logger.info(f"📤 最终发送消息体：{message_log}")

    try:
        logger.info(f"📤 发送 Markdown 消息：标题='{title}'，内容长度={len(body)}，{'包含艾特' if user_id.strip() else '不包含艾特'}")
        resp = requests.post(
            task["webhook_url"],
            headers={"Content-Type": "application/json; charset=utf-8"},
            data=json.dumps(message, ensure_ascii=False).encode("utf-8"),
            timeout=10
        )
        result = resp.json()
        if result.get("errcode") == 0:
            logger.info("✅ Webhook 发送成功")
            return True
        else:
            logger.error(f"❌ Webhook 发送失败：{result.get('errmsg')}")
            return False
    except Exception as e:
        logger.error(f"❌ Webhook 发送异常：{e}")
        return False


# -------------------------- 9. 更新执行日期 --------------------------
def update_task_exec_date(record_id):
    """更新任务的最新执行日期"""
    if not record_id:
        logger.warning("⚠️ 记录ID为空，无法更新执行日期")
        return False

    access_token = get_dingtalk_access_token()
    if not access_token:
        logger.error("❌ 无Token，无法更新执行日期")
        return False

    base_id = DINGTALK_CONFIG["config_base_id"]
    sheet_name = urllib.parse.quote(DINGTALK_CONFIG["config_sheet_name"])
    url = f"https://api.dingtalk.com/v1.0/notable/bases/{base_id}/sheets/{sheet_name}/records"
    headers = {
        "x-acs-dingtalk-access-token": access_token,
        "Content-Type": "application/json"
    }

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    payload = {
        "records": [
            {
                "id": record_id,
                "fields": {"最新完成日期": current_time}
            }
        ],
        "operatorId": DINGTALK_CONFIG["operator_id"]
    }

    try:
        response = requests.put(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        logger.info(f"✅ 已更新任务{record_id}的最新执行日期为：{current_time}")
        return True
    except requests.exceptions.HTTPError as e:
        response = e.response
        logger.error(f"❌ 更新任务{record_id}执行日期失败（HTTP {response.status_code}）：")
        logger.error(f"响应内容：{response.text}")
        return False
    except Exception as e:
        logger.error(f"❌ 更新任务{record_id}执行日期失败：{e}")
        return False


# -------------------------- 10. 任务执行核心逻辑 --------------------------
def check_and_execute_tasks():
    """检查并执行所有启用的任务"""
    logger.info("=" * 60)
    logger.info("🔍 开始检查待执行任务...")
    now = datetime.now()
    today = date.today().strftime("%Y-%m-%d")
    current_time_str = now.strftime("%H:%M")
    weekday_names = ["一", "二", "三", "四", "五", "六", "日"]
    current_weekday = weekday_names[now.weekday()]
    logger.info(f"⏰ 当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}（周{current_weekday}，当前时刻：{current_time_str}）")

    tasks = get_task_configs()
    if not tasks:
        logger.warning("⚠️ 无有效任务，结束检查")
        return

    normal_execution_count = 0
    compensation_execution_count = 0

    for idx, task in enumerate(tasks, 1):
        push_times_str = ",".join([f"{h:02d}:{m:02d}" for h, m in task["push_times"]])
        logger.info(
            f"\n📋 处理任务{idx}：{task['is_enabled']} | 定时规则={task['cron_type']}{task['cron_extra'] if task['cron_extra'] else ''} | 推送时间={push_times_str}")

        cron_config = parse_cron_config(task)
        if not cron_config:
            logger.warning(f"⚠️ 任务{idx}定时规则无效，跳过")
            continue

        # 正常执行
        if should_execute_now(cron_config):
            matched_time = \
            [f"{h:02d}:{m:02d}" for h, m in cron_config["push_times"] if h == now.hour and m == now.minute][0]
            logger.info(f"🎯 任务{idx}触发正常执行（当前匹配时间点：{matched_time}）...")

            content_list = get_table_records(
                table_url=task["table_url"],
                target_field=task["target_field"],
                cron_type=cron_config["type"],
                latest_exec_date=task["latest_exec_date"],
                at_field_name=task["at_field_name"]
            )

            if not content_list:
                logger.info(f"ℹ️ 任务{idx}无符合条件的记录，无需发送")
                update_task_exec_date(task["record_id"])
                normal_execution_count += 1
                continue

            send_success = True
            for i, item in enumerate(content_list, 1):
                logger.info(f"📤 任务{idx}（{matched_time}）发送第{i}/{len(content_list)}条记录")
                if not send_webhook(task, item["content"], item["user_id"]):
                    logger.error(f"💥 任务{idx}（{matched_time}）第{i}条记录发送失败")
                    send_success = False
                else:
                    logger.info(f"✅ 任务{idx}（{matched_time}）第{i}条记录发送成功")

            update_task_exec_date(task["record_id"])
            if send_success:
                logger.info(f"🎉 任务{idx}（{matched_time}）所有记录发送完成")
            else:
                logger.warning(f"⚠️ 任务{idx}（{matched_time}）部分记录发送失败")
            normal_execution_count += 1

        # 补偿执行
        elif is_missed_today_task(cron_config, task.get("latest_exec_date", "")):
            missed_times = [
                f"{h:02d}:{m:02d}" for h, m in cron_config["push_times"]
                if (now.hour > h) or (now.hour == h and now.minute > m)
            ]
            logger.info(f"⏳ 任务{idx}触发补偿执行（未执行的时间点：{','.join(missed_times)}）...")

            content_list = get_table_records(
                table_url=task["table_url"],
                target_field=task["target_field"],
                cron_type=cron_config["type"],
                latest_exec_date=task["latest_exec_date"],
                at_field_name=task["at_field_name"]
            )

            if not content_list:
                logger.info(f"ℹ️ 任务{idx}补偿执行无符合条件的记录")
                update_task_exec_date(task["record_id"])
                compensation_execution_count += 1
                continue

            send_success = True
            for i, item in enumerate(content_list, 1):
                logger.info(f"📤 任务{idx}（补偿）发送第{i}/{len(content_list)}条记录")
                if not send_webhook(task, item["content"], item["user_id"]):
                    logger.error(f"💥 任务{idx}（补偿）第{i}条记录发送失败")
                    send_success = False
                else:
                    logger.info(f"✅ 任务{idx}（补偿）第{i}条记录发送成功")

            update_task_exec_date(task["record_id"])
            if send_success:
                logger.info(f"🎉 任务{idx}补偿执行完成（已补全{','.join(missed_times)}的执行）")
            else:
                logger.warning(f"⚠️ 任务{idx}补偿执行部分记录失败")
            compensation_execution_count += 1

        else:
            logger.debug(f"⏳ 任务{idx}未到执行时间（当前时刻{current_time_str}，未匹配任何推送时间点）")

    logger.info(
        f"\n📊 本次任务检查完成：正常执行{normal_execution_count}个，补偿执行{compensation_execution_count}个，总计{normal_execution_count + compensation_execution_count}个")
    logger.info("=" * 60)


# -------------------------- 11. 主函数 --------------------------
def main():
    logger.info("🚀 拉伸大师定时Webhook调度器启动")
    logger.info("📖 支持的定时类型（推送时间支持多值，如8:00,10:00）：")
    logger.info("   - 每日：定时类型=每日，推送时间=HH:MM,HH:MM → 每天多时间点执行，取全部数据")
    logger.info("   - 每周：定时类型=每周，每周-周几=一~日，推送时间=HH:MM,HH:MM → 每周指定日多时间点执行")
    logger.info("   - 每月：定时类型=每月，每月-几号=1~31，推送时间=HH:MM,HH:MM → 每月指定日多时间点执行")
    logger.info("   - 单次：定时类型=单次，推送时间=HH:MM,HH:MM → 每天多时间点执行，筛选新增数据")
    logger.info("💡 提示：@人字段为空时不添加艾特，无效推送时间会自动过滤")
    logger.info("💡 按Ctrl+C停止调度器")
    logger.info("=" * 60)

    # 启动时先检查一次任务
    check_and_execute_tasks()

    # 每分钟检查一次，确保多时间点及时触发
    schedule.every(1).minutes.do(check_and_execute_tasks)
    logger.info("👂 开始监听任务（每分钟检查一次）...")

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 收到停止信号，正在关闭调度器...")
    except Exception as e:
        logger.error(f"💥 调度器运行异常：{e}")
    finally:
        logger.info("👋 调度器已停止")


if __name__ == "__main__":
    main()
