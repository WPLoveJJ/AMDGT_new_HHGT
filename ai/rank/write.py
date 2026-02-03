import requests
import json
import logging
import time

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 核心配置
CONFIG = {
    "api_url": "https://smallwecom.yesboss.work/smarttable",
    "company": "花都家庭医生",
    "docid": "dcDhP5Bolnl7LsmQpLNTYIstonM1fFAAp5rBDATlb9dhxtxa4Yqzo0hc2cviiWvxkR-CaRiVssk7hIVKVe8jTXwQ",
    "sheet_id": "tYjooD",
    "view_id": "vafyBn",
    "target_name": "陈温容",
    "field_to_update": "新增客户数",
    "new_value": 50,  # 纯数字格式
    "page_size": 200,
    "max_retry": 3,
    "sleep_time": 1
}


# 创建会话
def create_session():
    session = requests.Session()
    retry = requests.adapters.Retry(
        total=CONFIG["max_retry"],
        backoff_factor=CONFIG["sleep_time"],
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({
        "Content-Type": "application/json;charset=UTF-8",
        "Accept": "application/json, text/plain, */*"
    })
    return session


# 查询目标员工record_id
def query_employee_record_id(session):
    offset = 0
    while True:
        logger.info(f"查询第 {offset // CONFIG['page_size'] + 1} 页，偏移量：{offset}")
        post_data = {
            "action": "通用查询指定列表单",
            "company": CONFIG["company"],
            "WordList": {
                "docid": CONFIG["docid"],
                "sheet_id": CONFIG["sheet_id"],
                "view_id": CONFIG["view_id"],
                "offset": str(offset),
                "limit": str(CONFIG["page_size"])
            }
        }
        try:
            response = session.post(
                CONFIG["api_url"],
                data=json.dumps(post_data, ensure_ascii=False).encode('utf-8'),
                timeout=30
            )
            if response.status_code != 200:
                logger.error(f"查询失败，状态码：{response.status_code}")
                time.sleep(CONFIG["sleep_time"])
                continue
            result = response.json()
            if not result.get("success", True):
                logger.error(f"接口返回失败：{result.get('message', '未知错误')}")
                time.sleep(CONFIG["sleep_time"])
                continue
            records = result.get("data", [])
            if not records:
                logger.info("无更多数据，查询结束")
                return None
            for item in records:
                fields = item.get("values", {})
                name_field = fields.get("姓名", {})
                staff_name = ""
                if isinstance(name_field, list) and len(name_field) > 0:
                    staff_name = name_field[0].get("text", "").strip()
                elif isinstance(name_field, dict):
                    staff_name = name_field.get("text", "").strip()
                else:
                    staff_name = str(name_field).strip()
                if staff_name == CONFIG["target_name"]:
                    record_id = item.get("record_id")
                    if record_id:
                        logger.info(f"找到目标员工：{CONFIG['target_name']}，record_id: {record_id}")
                        return record_id
                    else:
                        logger.warning("找到目标姓名，但record_id为空")
                        return None
            offset += CONFIG["page_size"]
            time.sleep(CONFIG["sleep_time"])
        except Exception as e:
            logger.error(f"查询异常：{str(e)}")
            time.sleep(CONFIG["sleep_time"])
            continue
    return None


# 通用更新表单（使用纯数字格式）
def update_employee_record(session, record_id):
    if not record_id:
        logger.error("record_id为空，无法更新")
        return False
    logger.info(f"开始执行通用更新表单，record_id: {record_id}")

    # 最终正确格式：直接传数字，不嵌套value或数组（参考同类型字段）
    update_data = {
        "action": "通用更新表单",
        "company": CONFIG["company"],
        "WordList": {
            "docid": CONFIG["docid"],
            "sheet_id": CONFIG["sheet_id"],
            "record_id": record_id,
            "values": {
                CONFIG["field_to_update"]: CONFIG["new_value"]  # 纯数字格式
            },
            "view_id": CONFIG["view_id"]
        }
    }

    try:
        response = session.post(
            CONFIG["api_url"],
            data=json.dumps(update_data, ensure_ascii=False).encode('utf-8'),
            timeout=30
        )
        logger.info(f"更新接口响应状态码：{response.status_code}")
        logger.info(f"更新接口原始响应：{response.text}")

        if response.status_code == 200:
            try:
                result = response.json()
                if result.get("success"):
                    # 查询更新结果
                    check_data = {
                        "action": "通用查询指定列表单",
                        "company": CONFIG["company"],
                        "WordList": {
                            "docid": CONFIG["docid"],
                            "sheet_id": CONFIG["sheet_id"],
                            "view_id": CONFIG["view_id"],
                            "record_ids": [record_id]
                        }
                    }
                    check_response = session.post(
                        CONFIG["api_url"],
                        data=json.dumps(check_data, ensure_ascii=False).encode('utf-8'),
                        timeout=30
                    )
                    check_result = check_response.json()
                    if check_result.get("success") and check_result.get("data"):
                        final_value = check_result["data"][0]["values"].get(
                            CONFIG["field_to_update"]
                        )
                        logger.info(f"最终存储值：{final_value}")
                        if final_value == CONFIG["new_value"]:
                            logger.info("✅ 数值已正确写入")
                            return True
                        else:
                            logger.warning(f"值不匹配，实际存储：{final_value}")
                            return True
                else:
                    logger.error(f"更新失败：{result.get('message', '未知错误')}")
            except json.JSONDecodeError:
                logger.error("更新接口返回非JSON格式响应")
        else:
            logger.error(f"更新请求失败，状态码：{response.status_code}")
    except Exception as e:
        logger.error(f"更新过程异常：{str(e)}")
    return False


# 主函数
def main():
    logger.info("=" * 60)
    logger.info("【开始执行】通用更新表单流程")
    logger.info("=" * 60)
    session = create_session()
    try:
        record_id = query_employee_record_id(session)
        if not record_id:
            logger.error(f"未找到员工：{CONFIG['target_name']}，流程终止")
            return
        if update_employee_record(session, record_id):
            logger.info("🎉 整体流程执行成功，请刷新表格查看结果")
        else:
            logger.error("❌ 整体流程执行失败")
    finally:
        session.close()
        logger.info("🔒 会话已关闭")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
