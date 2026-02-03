import requests
import json
import urllib.parse
import schedule
import time
import subprocess
import sys
import os
import re
from datetime import datetime, date, timedelta
import logging
import socket

# 钉钉应用配置
DINGTALK_CONFIG = {
    "app_key": "dingoicseqn2bmdcazpl",
    "app_secret": "hiiqLe8teDkAADlJh9eklgsbtGIvrG8hPJyOC8as04wzG69OGmgaY_vQ_gyKTXEg",
    "base_id": "YndMj49yWjDEYy3ECQwPlLkgJ3pmz5aA",
    "sheet_name": "配置表",
    "operator_id": "jYEXEC84RV3QE3sm0UaeDwiEiE"
}

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 添加令牌缓存配置
TOKEN_CACHE_FILE = "dingtalk_token_cache.json"

# 新增一个安全处理字符串的辅助函数（放在代码顶部合适位置）
def safe_strip(value):
    """安全处理strip()方法，兼容各种类型"""
    if value is None:
        return ""
    return str(value).strip()


def load_cached_token():
    """从缓存文件加载访问令牌"""
    try:
        if os.path.exists(TOKEN_CACHE_FILE):
            with open(TOKEN_CACHE_FILE, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # 检查令牌是否过期（提前5分钟刷新）
            expire_time = datetime.fromisoformat(cache_data['expire_time'])
            if datetime.now() < expire_time - timedelta(minutes=5):
                logger.info("🔄 使用缓存的访问令牌")
                return cache_data['access_token']
            else:
                logger.info("⏰ 缓存的访问令牌即将过期，需要刷新")

    except Exception as e:
        logger.warning(f"⚠️ 读取令牌缓存失败: {e}")

    return None


def save_token_to_cache(access_token, expires_in):
    """保存访问令牌到缓存文件"""
    try:
        expire_time = datetime.now() + timedelta(seconds=expires_in)
        cache_data = {
            'access_token': access_token,
            'expire_time': expire_time.isoformat(),
            'created_time': datetime.now().isoformat()
        }

        with open(TOKEN_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 访问令牌已缓存，过期时间: {expire_time.strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        logger.error(f"❌ 保存令牌缓存失败: {e}")


def get_dingtalk_access_token():
    """获取钉钉访问令牌（带缓存机制）"""
    # 首先尝试使用缓存的令牌
    cached_token = load_cached_token()
    if cached_token:
        return cached_token

    # 缓存无效，重新获取
    logger.info("🔄 正在获取新的访问令牌...")

    url = "https://api.dingtalk.com/v1.0/oauth2/accessToken"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "appKey": DINGTALK_CONFIG["app_key"],
        "appSecret": DINGTALK_CONFIG["app_secret"]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()

        data = response.json()
        access_token = data.get("accessToken")
        expires_in = data.get("expireIn", 7200)  # 默认2小时

        if access_token:
            # 保存到缓存
            save_token_to_cache(access_token, expires_in)
            logger.info("✅ 成功获取并缓存访问令牌")
            return access_token
        else:
            logger.error("❌ 响应中没有访问令牌")
            return None

    except Exception as e:
        logger.error(f"获取访问令牌失败: {e}")
        return None


def get_task_configs():
    """获取任务配置数据（新增“是否启用”字段过滤）"""
    logger.info("🔄 正在从钉钉表格获取任务配置...")

    access_token = get_dingtalk_access_token()
    if not access_token:
        logger.error("❌ 无法获取访问令牌")
        return []

    base_url = "https://api.dingtalk.com/v1.0/notable/bases/"
    base_id = DINGTALK_CONFIG["base_id"]
    sheet_name = urllib.parse.quote(DINGTALK_CONFIG["sheet_name"])
    operator_id = DINGTALK_CONFIG["operator_id"]

    full_url = f"{base_url}{base_id}/sheets/{sheet_name}/records"

    params = {
        "maxResults": 100,
        "operatorId": operator_id
    }

    headers = {
        "x-acs-dingtalk-access-token": access_token,
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(full_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        if "records" not in data:
            logger.warning("⚠️ 响应中没有找到records字段")
            return []

        task_configs = []
        total_records = len(data["records"])
        logger.info(f"📊 从钉钉表格获取到 {total_records} 条记录")

        for idx, record in enumerate(data["records"], 1):
            fields = record.get("fields", {})

            # 1. 获取基础字段（新增“是否启用”字段）
            # 处理“是否启用”：兼容勾选框（True/False）或文本（“启用”/“禁用”/“是”/“否”）
            is_enabled_raw = safe_strip(fields.get("是否启用", ""))
            # 转换为布尔值：勾选框True/文本“启用”“是”→True，其他→False
            if isinstance(is_enabled_raw, bool):
                is_enabled = is_enabled_raw
            else:
                is_enabled = is_enabled_raw.lower() in ["启用", "是", "true", "1"]

            # 2. 获取原有字段
            record_id = (record.get("recordId") or
                         record.get("recordID") or
                         record.get("record_id") or
                         record.get("id"))
            execute_time = safe_strip(fields.get("任务运行时间", ""))
            py_file_path = safe_strip(fields.get("执行py文件路径", ""))
            latest_completion_date = safe_strip(fields.get("最新完成日期", ""))

            # 3. 过滤条件：必需字段齐全 + 已启用
            if execute_time and py_file_path and record_id and is_enabled:
                task_config = {
                    "record_id": record_id,
                    "is_enabled": is_enabled,  # 保存启用状态（便于后续日志）
                    "execute_time": execute_time,
                    "py_file_path": py_file_path,
                    "latest_completion_date": latest_completion_date
                }
                task_configs.append(task_config)
                logger.info(f"✅ 任务{idx}: 【已启用】执行时间=[{execute_time}], 文件=[{os.path.basename(py_file_path)}], "
                            f"最新完成日期=[{latest_completion_date}], record_id=[{record_id}]")
            else:
                # 未启用或配置不完整的任务，仅打印警告日志
                status = "已禁用" if not is_enabled else "配置不完整"
                logger.warning(f"⚠️ 任务{idx}: 【{status}】跳过 - 执行时间=[{execute_time}], 文件路径=[{py_file_path}], "
                               f"是否启用=[{is_enabled}], record_id=[{record_id}]")

        logger.info(f"🎯 成功识别到 {len(task_configs)} 个【已启用且配置完整】的任务")
        return task_configs

    except Exception as e:
        logger.error(f"❌ 获取任务配置失败: {e}")
        return []


def parse_time_expression(time_expr):
    """解析时间表达式，支持多种格式"""
    time_expr = time_expr.strip()

    # 统一处理中文冒号和英文冒号
    time_expr = time_expr.replace('：', ':')

    # 直接时间格式 (HH:MM)
    if re.match(r'^\d{1,2}:\d{2}$', time_expr):
        return {
            "type": "time",
            "hour": int(time_expr.split(':')[0]),
            "minute": int(time_expr.split(':')[1]),
            "description": f"每天{time_expr}"
        }

    # 每天X点Y分 或 每天X:Y
    daily_time_match = re.match(r'每天(\d{1,2})[点:](\d{1,2})分?', time_expr)
    if daily_time_match:
        hour = int(daily_time_match.group(1))
        minute = int(daily_time_match.group(2))
        return {
            "type": "daily",
            "hour": hour,
            "minute": minute,
            "description": f"每天{hour:02d}:{minute:02d}"
        }

    # 每天X点
    daily_match = re.match(r'每天(\d{1,2})点?', time_expr)
    if daily_match:
        hour = int(daily_match.group(1))
        return {
            "type": "daily",
            "hour": hour,
            "minute": 0,
            "description": f"每天{hour:02d}:00"
        }

    # 每周X(星期)Y点
    weekly_match = re.match(r'每周([一二三四五六日天])(\d{1,2})点?', time_expr)
    if weekly_match:
        weekday_map = {
            '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '日': 0, '天': 0
        }
        weekday_name = weekly_match.group(1)
        weekday = weekday_map.get(weekday_name)
        hour = int(weekly_match.group(2))
        return {
            "type": "weekly",
            "weekday": weekday,
            "hour": hour,
            "minute": 0,
            "description": f"每周{weekday_name} {hour:02d}:00"
        }

    # 每月X号Y点
    monthly_match = re.match(r'每月(\d{1,2})号(\d{1,2})点?', time_expr)
    if monthly_match:
        day = int(monthly_match.group(1))
        hour = int(monthly_match.group(2))
        return {
            "type": "monthly",
            "day": day,
            "hour": hour,
            "minute": 0,
            "description": f"每月{day}号 {hour:02d}:00"
        }

    logger.warning(f"⚠️ 无法解析时间表达式: {time_expr}")
    return None


def should_execute_now(time_config):
    """判断当前时间是否应该执行任务"""
    if not time_config:
        return False

    now = datetime.now()
    current_hour = now.hour
    current_minute = now.minute
    current_weekday = now.weekday()  # 0=周一, 6=周日
    current_day = now.day

    if time_config["type"] in ["time", "daily"]:
        # 每天执行类型
        return (current_hour == time_config["hour"] and
                current_minute == time_config["minute"])

    elif time_config["type"] == "weekly":
        # 每周执行
        target_weekday = time_config["weekday"]
        if target_weekday == 0:  # 周日
            target_weekday = 6
        else:
            target_weekday -= 1

        return (current_weekday == target_weekday and
                current_hour == time_config["hour"] and
                current_minute == time_config["minute"])

    elif time_config["type"] == "monthly":
        # 每月执行
        return (current_day == time_config["day"] and
                current_hour == time_config["hour"] and
                current_minute == time_config["minute"])

    return False


def is_missed_today_task(time_config, latest_completion_date):
    """判断任务是否是今天应该执行但未执行的任务"""
    if not time_config:
        return False

    today = date.today().strftime("%Y-%m-%d")
    # 检查最新完成日期是否不是今天
    if latest_completion_date == today:
        return False  # 今天已经执行过了

    now = datetime.now()
    current_hour = now.hour
    current_minute = now.minute
    current_weekday = now.weekday()
    current_day = now.day

    # 判断任务是否应该在今天执行
    if time_config["type"] in ["time", "daily"]:
        # 每天执行的任务
        task_hour = time_config["hour"]
        task_minute = time_config["minute"]

        # 任务时间在今天且已过当前时间
        return (task_hour < current_hour) or (task_hour == current_hour and task_minute < current_minute)

    elif time_config["type"] == "weekly":
        # 每周执行的任务
        target_weekday = time_config["weekday"]
        if target_weekday == 0:  # 周日
            target_weekday = 6
        else:
            target_weekday -= 1

        # 先判断是否是本周的目标工作日
        if current_weekday != target_weekday:
            return False

        # 再判断时间是否已过
        task_hour = time_config["hour"]
        task_minute = time_config["minute"]
        return (task_hour < current_hour) or (task_hour == current_hour and task_minute < current_minute)

    elif time_config["type"] == "monthly":
        # 每月执行的任务
        if current_day != time_config["day"]:
            return False

        # 判断时间是否已过
        task_hour = time_config["hour"]
        task_minute = time_config["minute"]
        return (task_hour < current_hour) or (task_hour == current_hour and task_minute < current_minute)

    return False


def update_completion_date(record_id):
    """更新任务的最新完成日期"""
    if not record_id or record_id == "None":
        logger.error("❌ 无效的record_id，无法更新完成日期")
        return False

    logger.info(f"📝 正在更新任务完成日期... (record_id: {record_id})")

    access_token = get_dingtalk_access_token()
    if not access_token:
        logger.error("❌ 无法获取访问令牌，无法更新完成日期")
        return False

    api_url = f"https://api.dingtalk.com/v1.0/notable/bases/{DINGTALK_CONFIG['base_id']}/sheets/{DINGTALK_CONFIG['sheet_name']}/records"

    headers = {
        "x-acs-dingtalk-access-token": access_token,
        "Content-Type": "application/json"
    }

    # 更新最新完成日期为今天
    today = date.today().strftime("%Y-%m-%d")

    payload = {
        "records": [
            {
                "id": record_id,
                "fields": {
                    "最新完成日期": today
                }
            }
        ],
        "operatorId": DINGTALK_CONFIG["operator_id"]
    }

    try:
        # 测试网络连接
        try:
            socket.create_connection(("api.dingtalk.com", 443), timeout=10)
            logger.info("✅ 网络连接正常")
        except socket.error as e:
            logger.warning(f"⚠️ 网络连接测试失败: {e}")

        response = requests.put(
            api_url,
            headers=headers,
            json=payload,
            timeout=30
        )

        logger.info(f"📤 请求URL: {api_url}")
        logger.info(f"📨 响应状态码: {response.status_code}")

        if response.status_code == 200:
            logger.info(f"✅ 成功更新任务完成日期为 {today}")
            return True
        else:
            logger.error(f"❌ API返回错误状态码: {response.status_code}")
            logger.error(f"❌ 响应内容: {response.text}")
            return False

    except Exception as e:
        logger.error(f"❌ 更新完成日期失败: {e}")
        return False


def execute_python_file(py_file_path, record_id):
    """执行Python文件并实时输出"""
    try:
        if not os.path.exists(py_file_path):
            logger.error(f"❌ Python文件不存在: {py_file_path}")
            return False

        logger.info(f"🚀 开始执行Python文件: {os.path.basename(py_file_path)}")
        logger.info(f"📁 文件完整路径: {py_file_path}")

        start_time = datetime.now()

        # 使用Popen而不是run，并实时输出
        process = subprocess.Popen(
            [sys.executable, py_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 将标准错误合并到标准输出
            text=True,
            encoding='utf-8',
            bufsize=1,  # 行缓冲
            universal_newlines=True
        )

        # 实时读取并输出
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            line = line.rstrip()
            if line:
                print(line)  # 实时输出到控制台
                logger.info(f"📤 {os.path.basename(py_file_path)}: {line}")  # 同时记录到日志
                output_lines.append(line)

        # 等待进程结束
        process.wait()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        if process.returncode == 0:
            logger.info(f"✅ Python文件执行成功! 耗时: {duration:.2f}秒")
            # 执行成功后更新完成日期
            update_completion_date(record_id)
            return True
        else:
            logger.error(f"❌ Python文件执行失败! 耗时: {duration:.2f}秒")
            logger.error(f"💥 错误输出: {''.join(output_lines[-10:])}")  # 只记录最后10行错误
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Python文件执行超时: {py_file_path}")
        return False
    except Exception as e:
        logger.error(f"💥 执行Python文件时发生错误: {e}")
        return False


def check_and_execute_tasks():
    """检查并执行任务（仅处理已启用的任务）"""
    logger.info("🔍 开始检查待执行任务...")

    task_configs = get_task_configs()
    if not task_configs:
        logger.warning("⚠️ 没有获取到有效的【已启用】任务配置")
        return

    current_time = datetime.now()
    weekday_names = ['一', '二', '三', '四', '五', '六', '日']
    current_weekday_name = weekday_names[current_time.weekday()]
    today = date.today().strftime("%Y-%m-%d")

    logger.info(f"⏰ 当前时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')} (周{current_weekday_name})")

    executed_count = 0
    compensated_count = 0  # 补偿执行的任务数

    for idx, task in enumerate(task_configs, 1):
        # 确认任务已启用（双重保险，避免配置过滤遗漏）
        if not task.get("is_enabled", False):
            logger.warning(f"⚠️ 任务{idx}({os.path.basename(task['py_file_path'])}): 未启用，跳过执行")
            continue

        # 提取任务信息
        execute_time = task["execute_time"]
        py_file_path = task["py_file_path"]
        record_id = task["record_id"]
        latest_completion_date = task["latest_completion_date"]
        file_name = os.path.basename(py_file_path)

        # 解析时间表达式
        time_config = parse_time_expression(execute_time)
        if not time_config:
            logger.warning(f"⚠️ 任务{idx}({file_name}): 无法解析时间表达式 [{execute_time}]")
            continue

        logger.info(f"📋 任务{idx}({file_name}): 【已启用】计划执行时间 [{time_config['description']}]，最新完成日期 [{latest_completion_date}]")

        # 检查是否到了执行时间（正常执行）
        if should_execute_now(time_config):
            logger.info(f"🎯 任务{idx}({file_name}): 匹配到执行时间! 开始执行...")
            success = execute_python_file(py_file_path, record_id)
            if success:
                executed_count += 1
                logger.info(f"🎉 任务{idx}({file_name}): 执行完成!")
            else:
                logger.error(f"💥 任务{idx}({file_name}): 执行失败!")

        # 检查是否是今天错过的任务且未执行（补偿执行）
        elif is_missed_today_task(time_config, latest_completion_date):
            logger.info(f"⏳ 任务{idx}({file_name}): 检测到今天未执行，开始补偿执行...")
            success = execute_python_file(py_file_path, record_id)
            if success:
                compensated_count += 1
                logger.info(f"🎉 任务{idx}({file_name}): 补偿执行完成!")
            else:
                logger.error(f"💥 任务{idx}({file_name}): 补偿执行失败!")

        else:
            logger.debug(f"⏳ 任务{idx}({file_name}): 未到执行时间且无需补偿")

    logger.info(f"📊 本次检查完成 - 正常执行: {executed_count}个, 补偿执行: {compensated_count}个")


def main():
    """主函数（更新说明文档，增加“是否启用”字段说明）"""
    logger.info("🚀 定时任务调度器启动")
    logger.info("📖 支持的时间格式:")
    logger.info("   - 每天X点 (如: 每天0点, 每天23点)")
    logger.info("   - 每周X(星期)Y点 (如: 每周一8点, 每周日20点)")
    logger.info("   - 每月X号Y点 (如: 每月1号9点, 每月15号20点)")
    logger.info("   - 直接时间 (如: 00:00, 23:30)")
    logger.info("📌 任务启用规则: 钉钉表格中“是否启用”字段需为【勾选/启用/是】才会执行")
    logger.info("=" * 60)

    # 启动时先检查一次任务配置
    logger.info("🔄 启动时检查任务配置...")
    check_and_execute_tasks()
    logger.info("=" * 60)

    # 每小时检查一次任务
    schedule.every().hour.do(check_and_execute_tasks)

    logger.info("👂 开始监听任务调度... (每小时检查一次)")
    logger.info("💡 提示: 按 Ctrl+C 可以停止调度器")

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 收到停止信号，正在关闭调度器...")
    except Exception as e:
        logger.error(f"💥 调度器运行时发生错误: {e}")
    finally:
        logger.info("👋 定时任务调度器已停止")


if __name__ == "__main__":
    main()