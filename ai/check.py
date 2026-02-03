import requests
import json
import urllib.parse
import os
import logging
from datetime import datetime, timedelta

# -------------------------- 1. 基础配置（和你可运行代码一致，仅改表格名称） --------------------------
DINGTALK_CONFIG = {
    "app_key": "dingczeweiukv9kue2gv",
    "app_secret": "BC11ILonRquetv-aTv6lrfUlqWHjDrikSQN9NWHhxRHVz8xYQGcnLgtL6h1SPiPU",
    "config_base_id": "pYLaezmVNev7pRZ9t4oxG9aQWrMqPxX6",  # 你的表格base_id
    "config_sheet_name": "全部员工",  # 目标表格名称：全部员工
    "operator_id": "xYLFMT7vpx2nLD5iiW81omAiEiE",
    "token_cache_file": "dingtalk_stretch_token_cache.json"
}

# -------------------------- 2. 日志配置（保留基础日志，便于排查） --------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('read_employee_names.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# -------------------------- 3. 工具函数（保留安全处理字符串） --------------------------
def safe_strip(value):
    if value is None:
        return ""
    return str(value).strip()


# -------------------------- 4. 钉钉Token管理（完全复用你可运行代码的逻辑） --------------------------
def load_cached_token():
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


# -------------------------- 5. 核心：读取“全部员工”表格的“姓名”字段 --------------------------
def read_employee_names():
    """仅读取“全部员工”表格中的“姓名”字段，复用你可运行代码的API逻辑"""
    logger.info("🔄 读取“全部员工”表格...")
    access_token = get_dingtalk_access_token()
    if not access_token:
        logger.error("❌ 无Token，无法读取表格")
        return []

    # 完全复用你可运行代码的API端点（notable/bases，而非之前错误的smartwork）
    base_id = DINGTALK_CONFIG["config_base_id"]
    sheet_name = urllib.parse.quote(DINGTALK_CONFIG["config_sheet_name"])
    url = f"https://api.dingtalk.com/v1.0/notable/bases/{base_id}/sheets/{sheet_name}/records"
    headers = {
        "x-acs-dingtalk-access-token": access_token,
        "Content-Type": "application/json"
    }
    params = {"maxResults": 100, "operatorId": DINGTALK_CONFIG["operator_id"]}

    # 三次重试（和你代码一致的容错逻辑）
    max_retries = 3
    retry_interval = 2
    retry_count = 0
    all_names = []

    while retry_count < max_retries:
        try:
            logger.info(f"🔄 第{retry_count + 1}次请求表格API")
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
                logger.warning("⚠️ 表格中无记录")
                return []

            # 提取所有“姓名”字段
            for idx, record in enumerate(records, 1):
                fields = record.get("fields", {})
                # 处理字段值（兼容文本/字典类型，和你代码的字段解析逻辑一致）
                name_value = fields.get("姓名", "")
                if isinstance(name_value, dict):
                    name = safe_strip(name_value.get("text", name_value.get("value", "")))
                else:
                    name = safe_strip(name_value)

                if name:
                    all_names.append(name)
                    logger.info(f"✅ 读取到姓名[{idx}]：{name}")

            logger.info(f"📊 读取完成，共获取{len(all_names)}个姓名")
            return all_names

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
            logger.error(f"❌ 读取表格失败：{e}")
            return []


# -------------------------- 6. 主函数（仅执行读取和打印） --------------------------
def main():
    logger.info("===== 开始读取“全部员工”表格的姓名 =====")
    names = read_employee_names()

    if names:
        print("\n===== 读取到的员工姓名列表 =====")
        for i, name in enumerate(names, 1):
            print(f"{i}. {name}")
        print(f"\n共读取到 {len(names)} 个姓名")
    else:
        print("\n未读取到任何姓名数据")

    logger.info("===== 操作结束 =====")


if __name__ == "__main__":
    main()