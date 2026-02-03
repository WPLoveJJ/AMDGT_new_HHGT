import requests
import time
import logging
import os
import csv
import re  # 导入正则表达式模块
from datetime import datetime
from typing import List, Dict, Optional, Set
from flask import Flask, request, Response
import xml.etree.ElementTree as ET
from wechatpy.crypto import WeChatCrypto

app = Flask(__name__)

# ---------------------- 基础配置 ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('wechat_customer_callback.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 企业微信配置
TOKEN = "QE94ubBO"
AES_KEY = "xxxNcAcBDaK8qfNncCBc4WMY8vZvcgEoB7Q5yTql1JL"
CORP_ID = "wwb0728887ce23a4ce"

# 目标企业ID
TARGET_CORP_ID = "wpbb8UCgAAnqdQV_KdxS1yYlgqpomwpg"

# 员工表API配置
STAFF_TABLE_API_URL = "https://smallwecom.yesboss.work/smarttable"
STAFF_TABLE_HEADERS = {
    "Content-Type": "application/json; charset=utf-8",
    "Accept": "application/json"
}

# 手机号正则表达式（匹配11位数字，以1开头）
PHONE_REGEX = re.compile(r'1[3-9]\d{9}')


# ---------------------- 核心类（回调模式） ----------------------
class WeChatCallbackCollector:
    def __init__(self, corp_id: str, corp_secret: str,
                 staff_docid: str, staff_tab: str, staff_viewid: str,
                 staff_personnel_field: str = "人员"):
        # 企微凭证
        self.corp_id = corp_id
        self.corp_secret = corp_secret
        self.access_token = None
        self.token_expires = 0

        # 员工表配置
        self.staff_docid = staff_docid
        self.staff_tab = staff_tab
        self.staff_viewid = staff_viewid
        self.staff_personnel_field = staff_personnel_field

        # 关注的员工列表
        self.target_staff_ids = set()

        # 数据存储
        self.final_customer_data: List[Dict] = []

        # 网络会话
        self.session = requests.Session()

        # 初始化
        self.refresh_staff_list()
        self.get_token()
        logger.info("===== 回调模式初始化完成 =====")

    def refresh_staff_list(self):
        """从员工表刷新关注的员工列表"""
        logger.info("刷新关注的员工列表...")
        params = {
            "action": "通用查询表单",
            "company": "拉伸大师",
            "WordList": {
                "docid": self.staff_docid,
                "sheet_id": self.staff_tab,
                "view_id": self.staff_viewid
            }
        }

        try:
            res = self.session.post(STAFF_TABLE_API_URL, headers=STAFF_TABLE_HEADERS,
                                    json=params, timeout=30)
            res.raise_for_status()
            data = res.json()

            if not data.get("success") or not data.get("data"):
                logger.error(f"员工表没数据：{data.get('message')}")
                return

            # 提取员工usrid
            new_staff_ids = set()
            for item in data["data"]:
                personnel = item.get("values", {}).get(self.staff_personnel_field, [])
                for p in personnel:
                    usrid = p.get("user_id")
                    if usrid:
                        new_staff_ids.add(usrid)

            self.target_staff_ids = new_staff_ids
            logger.info(f"关注{len(self.target_staff_ids)}个员工（示例：{list(new_staff_ids)[:2]}...）")
        except Exception as e:
            logger.error(f"刷新员工列表出错：{str(e)}")

    def get_token(self, force_refresh=False):
        """获取或刷新access_token"""
        if self.access_token and time.time() < self.token_expires - 60 and not force_refresh:
            return True

        logger.info("获取企微token...")
        url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={self.corp_id}&corpsecret={self.corp_secret}"

        try:
            res = self.session.get(url, timeout=10)
            data = res.json()

            if data.get("errcode") != 0:
                logger.error(f"获取token失败：{data.get('errmsg')}")
                return False

            self.access_token = data["access_token"]
            self.token_expires = time.time() + data["expires_in"]
            logger.info(f"token获取成功（有效期至{datetime.fromtimestamp(self.token_expires)}）")
            return True
        except Exception as e:
            logger.error(f"获取token异常：{str(e)}")
            return False

    def extract_phone_from_remark(self, remark: str) -> str:
        """从备注中解析手机号"""
        if not remark:
            return ""

        # 使用正则表达式查找手机号
        match = PHONE_REGEX.search(remark)
        if match:
            return match.group(0)
        return ""

    def get_customer_detail(self, customer_id: str, staff_usrid: str) -> Optional[Dict]:
        """获取客户详情 - 多途径获取联系电话"""
        if not self.get_token():
            logger.error("无有效token，无法获取客户详情")
            return None

        url = f"https://qyapi.weixin.qq.com/cgi-bin/externalcontact/get"
        params = {
            "access_token": self.access_token,
            "external_userid": customer_id
        }

        try:
            response = self.session.get(url, params=params, timeout=15)
            data = response.json()

            if data.get("errcode") != 0 or not data.get("external_contact"):
                logger.warning(f"客户详情API错误：errcode={data.get('errcode')}, errmsg={data.get('errmsg')}")
                # 尝试刷新token后重试一次
                if data.get("errcode") in [40014, 42001, 40001]:
                    logger.info("尝试刷新token后重试...")
                    if self.get_token(force_refresh=True):
                        return self.get_customer_detail(customer_id, staff_usrid)
                return None

            # 解析客户信息
            external_info = data["external_contact"]
            follow_info = data.get("follow_user", [{}])[0]
            remark = follow_info.get("remark", "")

            # 多途径获取联系电话
            phone = ""
            source = "未获取"

            # 1. 优先获取成员设置的手机号码 (remark_mobiles)
            remark_mobiles = follow_info.get("remark_mobiles", [])
            if remark_mobiles:
                phone = remark_mobiles[0]
                source = "remark_mobiles"
                logger.debug(f"从remark_mobiles获取手机号: {phone}")

            # 2. 如果上面没有，尝试获取客户自己填写的电话
            elif external_info.get("phone"):
                phone = external_info.get("phone")
                source = "external_contact.phone"
                logger.debug(f"从external_contact.phone获取手机号: {phone}")

            # 3. 如果还没有，尝试从备注中解析手机号
            elif remark:
                phone = self.extract_phone_from_remark(remark)
                if phone:
                    source = "remark解析"
                    logger.debug(f"从备注中解析出手机号: {phone}")
                else:
                    logger.debug("备注中未找到有效手机号")

            return {
                "external_userid": customer_id,
                "nickname": external_info.get("name", ""),
                "remark": remark,
                "phone": phone,
                "phone_source": source,  # 记录手机号来源
                "oper_staff_usrid": staff_usrid,
                "collect_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logger.error(f"获取客户详情异常：{str(e)}")
            return None

    def handle_callback_event(self, event_data: dict):
        """处理回调事件 - 专注处理添加客户事件"""
        try:
            event_type = event_data.get("Event")
            change_type = event_data.get("ChangeType")
            user_id = event_data.get("UserID")
            external_userid = event_data.get("ExternalUserID")
            to_user_name = event_data.get("ToUserName")

            # 只处理添加客户事件
            if event_type != "change_external_contact" or change_type != "add_external_contact":
                return False

            # 只处理特定企业的通知
            if to_user_name != TARGET_CORP_ID:
                logger.info(f"忽略非目标企业事件：{to_user_name}")
                return False

            # 只关注特定员工
            if user_id not in self.target_staff_ids:
                logger.info(f"忽略非关注员工事件：{user_id}")
                return False

            logger.info(f"检测到新客户添加事件：员工={user_id}, 客户={external_userid}")

            # 获取客户详情
            customer_detail = self.get_customer_detail(external_userid, user_id)
            if customer_detail:
                self.final_customer_data.append(customer_detail)
                log_msg = f"✅ 保存客户信息：{customer_detail['nickname']}({external_userid[:8]})"
                if customer_detail['phone']:
                    log_msg += f" | 电话：{customer_detail['phone']} (来源: {customer_detail['phone_source']})"
                else:
                    log_msg += " | 未获取到电话"
                logger.info(log_msg)
                return True

        except Exception as e:
            logger.error(f"处理回调事件异常：{str(e)}")
        return False

    def make_csv(self):
        """生成CSV文件"""
        if not self.final_customer_data:
            logger.warning("没有客户数据，无法生成CSV")
            return False

        # CSV字段（添加phone_source字段）
        fields = [
            "collect_time", "external_userid", "nickname",
            "remark", "phone", "phone_source", "oper_staff_usrid"
        ]

        # 文件名（含当前日期）
        filename = f"customer_info_callback_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = os.path.abspath(filename)

        try:
            with open(filename, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.final_customer_data)

            logger.info(f"\n✅ CSV生成成功！路径：{filepath}（共{len(self.final_customer_data)}条数据）")
            return True
        except Exception as e:
            logger.error(f"生成CSV出错：{str(e)}")
            return False


# ---------------------- 配置和Flask路由 ----------------------
# 初始化收集器
collector = WeChatCallbackCollector(
    corp_id=CORP_ID,
    corp_secret="vZ7mge0BomfhLaza43spNe9Wb8EmBdjmWQzxrhD10j4",
    staff_docid="dcPbCgiFT361NMXCjtOXHJRssdGcQcFBNmx-ej23sFFCjZJO1PmrZOGHDn_4dRUnUw1Nt-SD5-3fxIhNB42H1Gbw",
    staff_tab="tAP0Vy",
    staff_viewid="vd6htv",
    staff_personnel_field="人员"
)


# 回调验证URL（企业微信需要验证）
@app.route('/wechat_callback', methods=['GET'])
def verify_url():
    msg_signature = request.args.get('msg_signature', '')
    timestamp = request.args.get('timestamp', '')
    nonce = request.args.get('nonce', '')
    echostr = request.args.get('echostr', '')

    logger.info(f"收到URL验证请求: msg_signature={msg_signature}, timestamp={timestamp}")

    try:
        # 使用wechatpy进行验证
        crypto = WeChatCrypto(TOKEN, AES_KEY, CORP_ID)
        echostr_plain = crypto.check_signature(msg_signature, timestamp, nonce, echostr)
        return Response(echostr_plain, content_type='text/plain')
    except Exception as e:
        logger.error(f"验证URL失败: {str(e)}")
        return Response("Invalid signature", status=403)


# 接收事件回调
@app.route('/wechat_callback', methods=['POST'])
def handle_event():
    # 获取请求参数
    msg_signature = request.args.get('msg_signature', '')
    timestamp = request.args.get('timestamp', '')
    nonce = request.args.get('nonce', '')
    xml_data = request.data

    logger.info(f"收到回调事件: msg_signature={msg_signature}, timestamp={timestamp}")

    try:
        # 解密消息
        crypto = WeChatCrypto(TOKEN, AES_KEY, CORP_ID)
        decrypted_xml = crypto.decrypt_message(xml_data, msg_signature, timestamp, nonce)
        logger.info(f"解密后的XML: {decrypted_xml.decode('utf-8')}")

        # 解析XML
        root = ET.fromstring(decrypted_xml)
        event_data = {}
        for child in root:
            event_data[child.tag] = child.text

        # 记录事件类型
        event_type = event_data.get('Event')
        change_type = event_data.get('ChangeType')
        to_user_name = event_data.get('ToUserName')
        user_id = event_data.get('UserID')
        external_userid = event_data.get('ExternalUserID')

        logger.info(f"事件类型: {event_type}, 变更类型: {change_type}")

        # 只处理目标企业事件
        if to_user_name != TARGET_CORP_ID:
            logger.info(f"忽略非目标企业事件：{to_user_name}")
            return Response("success", content_type='text/plain')

        # 只处理添加客户事件
        if event_type == "change_external_contact" and change_type == "add_external_contact":
            logger.info(f"检测到新客户添加事件：员工={user_id}, 客户={external_userid}")

            # 处理事件
            collector.handle_callback_event({
                "Event": event_type,
                "ChangeType": change_type,
                "ToUserName": to_user_name,
                "UserID": user_id,
                "ExternalUserID": external_userid
            })

        return Response("success", content_type='text/plain')
    except Exception as e:
        logger.error(f"处理事件出错: {str(e)}")
        return Response("error", status=500)


# 手动触发CSV生成
@app.route('/generate_csv', methods=['GET'])
def generate_csv():
    if collector.make_csv():
        return "CSV生成成功！", 200
    return "生成CSV失败", 500


# 定时刷新员工列表
def refresh_staff_periodically():
    while True:
        time.sleep(1800)  # 每30分钟刷新一次
        collector.refresh_staff_list()


# ---------------------- 启动服务 ----------------------
if __name__ == '__main__':
    import threading

    # 启动定时刷新线程
    refresh_thread = threading.Thread(target=refresh_staff_periodically, daemon=True)
    refresh_thread.start()

    # 启动Flask服务
    app.run(host='0.0.0.0', port=5000, debug=False)