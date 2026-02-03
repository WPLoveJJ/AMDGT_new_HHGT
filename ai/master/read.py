import requests
import time
import logging
import os
import csv
import json
from datetime import datetime
from typing import List, Dict, Optional, Set

# ---------------------- 基础配置（不用改） ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('wechat_customer_final.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 员工表API配置（和你之前用的一致）
STAFF_TABLE_API_URL = "https://smallwecom.yesboss.work/smarttable"
STAFF_TABLE_HEADERS = {
    "Content-Type": "application/json; charset=utf-8",
    "Accept": "application/json"
}


# ---------------------- 核心类（不用改逻辑，只填配置） ----------------------
class WeChatCustomerFinalCollector:
    def __init__(self, corp_id: str, corp_secret: str,
                 staff_docid: str, staff_tab: str, staff_viewid: str,
                 staff_personnel_field: str = "人员"):
        # 1. 企微凭证（填你的配置）
        self.corp_id = corp_id
        self.corp_secret = corp_secret
        self.access_token = None

        # 2. 员工表配置（填你的配置）
        self.staff_docid = staff_docid
        self.staff_tab = staff_tab
        self.staff_viewid = staff_viewid
        self.staff_personnel_field = staff_personnel_field

        # 3. 数据存储（最终要的客户信息）
        self.staff_usrid_list: List[str] = []  # 要查询的员工usrid
        self.final_customer_data: List[Dict] = []  # 最终客户数据（含所有需要的字段）

        # 4. 网络会话
        self.session = requests.Session()
        logger.info("===== 初始化完成：只抓你要的客户核心信息 =====")

    # ---------------------- 1. 提员工usrid（和你之前的逻辑一致） ----------------------
    def get_staff_usrid(self):
        """从员工表提取要查询的员工usrid"""
        logger.info("开始提员工usrid...")
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
            res = self.session.post(STAFF_TABLE_API_URL, headers=STAFF_TABLE_HEADERS, json=params, timeout=30)
            res.raise_for_status()
            data = res.json()

            if not data.get("success") or not data.get("data"):
                logger.error(f"员工表没数据：{data.get('message')}")
                return False

            # 提员工usrid
            for item in data["data"]:
                personnel = item.get("values", {}).get(self.staff_personnel_field, [])
                for p in personnel:
                    usrid = p.get("user_id")
                    if usrid and usrid not in self.staff_usrid_list:
                        self.staff_usrid_list.append(usrid)

            logger.info(f"提了{len(self.staff_usrid_list)}个员工usrid（示例：{self.staff_usrid_list[:2]}...）")
            return len(self.staff_usrid_list) > 0
        except Exception as e:
            logger.error(f"提员工usrid错了：{str(e)}")
            return False

    # ---------------------- 2. 拿access_token（和你之前能跑的代码一致） ----------------------
    def get_token(self):
        """拿企微token（用你之前能跑的配置，大概率能成功）"""
        logger.info("拿企微token...")
        url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={self.corp_id}&corpsecret={self.corp_secret}"

        try:
            res = self.session.get(url, timeout=10)
            data = res.json()

            if data.get("errcode") != 0:
                logger.error(f"拿token错了：{data.get('errmsg')}（检查corp_id和secret是否和之前能跑的一致！）")
                return False

            self.access_token = data["access_token"]
            logger.info(f"token拿成功了（有效期7200秒）")
            return True
        except Exception as e:
            logger.error(f"拿token异常：{str(e)}")
            return False

    # ---------------------- 3. 核心：抓客户信息（换用宽松接口） ----------------------
    def catch_customer_info(self):
        """
        用「获取跟进人客户列表」接口（externalcontact/getfollowuserlist）
        这接口对应用要求低，你之前能跑的配置大概率能用
        """
        if not self.access_token or not self.staff_usrid_list:
            logger.error("没token或没员工usrid，没法抓")
            return False

        logger.info(f"\n开始抓{len(self.staff_usrid_list)}个员工的客户...")

        for idx, staff_usrid in enumerate(self.staff_usrid_list, 1):
            logger.info(f"\n----- 处理第{idx}个员工：usrid={staff_usrid[:8]}... -----")

            # 步骤1：拿这个员工的所有客户ID（external_userid）
            customer_ids = self._get_customer_ids_by_staff(staff_usrid)
            if not customer_ids:
                logger.info(f"这个员工没客户，跳过")
                time.sleep(0.5)
                continue

            # 步骤2：逐个拿客户详情（昵称、备注、手机号）
            for customer_id in customer_ids:
                detail = self._get_customer_detail(customer_id, staff_usrid)
                if detail:
                    self.final_customer_data.append(detail)
                    logger.info(f"✅ 抓到客户：昵称={detail['nickname']}，external_id={customer_id[:8]}...")

                time.sleep(0.3)  # 控制频率，避免报错

        logger.info(f"\n抓完了！共抓到{len(self.final_customer_data)}个客户的信息")
        return True

    def _get_customer_ids_by_staff(self, staff_usrid: str) -> List[str]:
        """终极修复：用GET请求调用externalcontact/list，确保返回客户ID（external_userid）"""
        # 官方正确接口：externalcontact/list（用GET请求，参数放URL里）
        url = f"https://qyapi.weixin.qq.com/cgi-bin/externalcontact/list"
        # 参数：access_token + userid（员工ID），不传递start_time/end_time（抓所有客户）
        params = {
            "access_token": self.access_token,
            "userid": staff_usrid  # 传员工ID，获取该员工的客户
        }

        try:
            logger.debug(f"请求客户ID（GET方式）：URL={url}，参数={params}")
            response = self.session.get(url, params=params, timeout=15)
            logger.debug(f"响应状态码：{response.status_code}，完整响应：{response.text[:500]}")

            if response.status_code != 200:
                logger.error(f"接口访问失败：状态码{response.status_code}，响应：{response.text[:200]}")
                return []

            # 解析JSON
            data = response.json()
            if data.get("errcode") != 0:
                errmsg = data.get("errmsg", "未知错误")
                logger.error(f"API错误：{data['errcode']} - {errmsg}")
                # 最后一次权限提示（如果报48001，去加权限；报40058，忽略，GET请求会自动兼容）
                if data["errcode"] == 48001:
                    logger.error("！！去企微后台→应用→权限管理→加「外部联系人管理→获取外部联系人列表」权限！！")
                return []

            # ✅ 这次拿到的一定是客户ID（external_userid），格式是wm/wx开头
            customer_ids = data.get("external_userid", [])
            # 过滤掉非客户ID（确保是wm/wx开头）
            valid_customer_ids = [cid for cid in customer_ids if cid.startswith(("wm", "wx"))]

            logger.info(f"✅ 员工{staff_usrid[:8]}...有{len(valid_customer_ids)}个有效客户！（客户ID示例：{valid_customer_ids[:2]}...）")
            return valid_customer_ids

        except Exception as e:
            logger.error(f"接口调用异常：{str(e)}", exc_info=True)
            return []

    def _get_customer_detail(self, customer_id: str, staff_usrid: str) -> Optional[Dict]:
        """内部方法：拿单个客户的详情（同样添加响应日志）"""
        url = f"https://qyapi.weixin.qq.com/cgi-bin/externalcontact/get"
        params = {
            "access_token": self.access_token,
            "external_userid": customer_id
        }

        try:
            logger.debug(f"请求客户详情接口：URL={url}，参数={params}")
            response = self.session.get(url, params=params, timeout=15)

            # ✅ 打印响应信息
            logger.debug(f"详情接口状态码：{response.status_code}")
            logger.debug(f"详情接口原始响应（前500字符）：{response.text[:500]}")

            if response.status_code != 200:
                logger.warning(f"详情接口返回非200状态码：{response.status_code}")
                return None

            try:
                data = response.json()
            except json.JSONDecodeError as e:
                logger.warning(f"解析客户详情JSON失败：{response.text[:200]}（错误：{str(e)}）")
                return None

            if data.get("errcode") != 0 or not data.get("external_contact"):
                logger.warning(f"客户详情API错误：errcode={data.get('errcode')}，errmsg={data.get('errmsg')}")
                return None

            # 正常提取字段
            external_info = data["external_contact"]
            follow_info = data.get("follow_user", [{}])[0]
            return {
                "external_userid": customer_id,
                "nickname": external_info.get("name", ""),
                "remark": follow_info.get("remark", ""),
                "phone": external_info.get("phone", ""),
                "oper_staff_usrid": staff_usrid,
                "collect_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logger.error(f"调用客户详情接口异常：{str(e)}", exc_info=True)
            return None

    # ---------------------- 4. 生成CSV（直接能用，含所有你要的字段） ----------------------
    def make_csv(self):
        """生成CSV文件，字段顺序就是你要的"""
        if not self.final_customer_data:
            logger.warning("没客户数据，没法生成CSV")
            return False

        # CSV字段（严格按你的需求排序）
        fields = [
            "collect_time", "external_userid", "nickname",
            "remark", "phone", "oper_staff_usrid"
        ]

        # 文件名（含当前日期）
        filename = f"customer_info_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = os.path.abspath(filename)

        try:
            with open(filename, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.final_customer_data)

            logger.info(f"\n✅ CSV生成成功！路径：{filepath}（共{len(self.final_customer_data)}条数据）")
            return True
        except Exception as e:
            logger.error(f"生成CSV错了：{str(e)}")
            return False

    # ---------------------- 主流程（一步到位） ----------------------
    def run(self):
        """主流程：提员工usrid → 拿token → 抓客户 → 生成CSV"""
        try:
            if not self.get_staff_usrid():
                return False
            if not self.get_token():
                return False
            if not self.catch_customer_info():
                return False
            if not self.make_csv():
                return False

            logger.info("\n===== 所有操作完成！你要的客户信息都在CSV里了 =====")
            return True
        except Exception as e:
            logger.error(f"程序错了：{str(e)}")
            return False
        finally:
            self.session.close()
            logger.info("网络关了")


# ---------------------- 只需要改这里的配置！（用你之前能跑的配置） ----------------------
if __name__ == "__main__":
    # 1. 企微配置：填你之前“能抓到仅昨天”的代码里的 corp_id 和 corp_secret！
    # （关键：必须和之前能跑的一致，不要用新的应用配置）
    WECHAT_CONFIG = {
        "corp_id": "wwb0728887ce23a4ce",  # 替换成你之前能跑的 corp_id
        "corp_secret": "vZ7mge0BomfhLaza43spNe9Wb8EmBdjmWQzxrhD10j4"  # 替换成你之前能跑的 corp_secret
    }

    # 2. 员工表配置：和之前一样，不用改
    STAFF_TABLE_CONFIG = {
        "docid": "dcPbCgiFT361NMXCjtOXHJRssdGcQcFBNmx-ej23sFFCjZJO1PmrZOGHDn_4dRUnUw1Nt-SD5-3fxIhNB42H1Gbw",
        "staff_tab": "tAP0Vy",
        "staff_viewid": "vd6htv",
        "personnel_field": "人员"  # 员工表的“人员字段”名称，和之前一样
    }

    # 3. 运行（不用动）
    collector = WeChatCustomerFinalCollector(
        corp_id=WECHAT_CONFIG["corp_id"],
        corp_secret=WECHAT_CONFIG["corp_secret"],
        staff_docid=STAFF_TABLE_CONFIG["docid"],
        staff_tab=STAFF_TABLE_CONFIG["staff_tab"],
        staff_viewid=STAFF_TABLE_CONFIG["staff_viewid"],
        staff_personnel_field=STAFF_TABLE_CONFIG["personnel_field"]
    )
    collector.run()