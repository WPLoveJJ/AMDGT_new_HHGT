import json
import requests
import datetime
import logging
from typing import List, Dict, Optional

# -------------------------- 基础配置（需用户替换为自己的真实信息） --------------------------
CONFIG = {
    # 企业微信基础凭证（用于获取access_token）
    "corpid": "你的企业微信CorpID",  # 例：ww3c73189347476992
    "corpsecret": "你的企业微信CorpSecret",  # 例：7GPTkahno_JD1wiCmEI89i9mTutgTQjdjBYMpj1EJek
    # 群众表参数（你提供的真实信息）
    "people_table": {
        "docid": "dcDhP5Bolnl7LsmQpLNTYIstonM1fFAAp5rBDATlb9dhxtxa4Yqzo0hc2cviiWvxkR-CaRiVssk7hIVKVe8jTXwQ",
        "sheet_id": "t4wjUf",  # 对应群众表的"tab"参数
        "view_id": "vukaF8",  # 群众表的视图ID
        "json_field_name": "json"  # 群众表中存储JSON数据的字段名（你确认的"json"字段）
    },
    # 企业微信表格代理API（参考你之前JS代码中的地址，非钉钉地址）
    "table_api_url": "https://api.yxkf120.com/QYCurrencyProxySidebarAPI"
}

# -------------------------- 日志配置（便于调试） --------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class WeChatPeopleBirthdayFilter:
    def __init__(self):
        self.access_token = None
        # 今日日期（仅保留月日，格式：MM-DD，如08-23）
        self.today = datetime.datetime.now()
        self.target_month_day = self.today.strftime("%m-%d")
        logger.info(f"初始化完成 | 今日日期（月-日）：{self.target_month_day} | 需匹配此格式生日")

    # -------------------------- 1. 获取企业微信Access Token（必须步骤） --------------------------
    def get_wechat_access_token(self) -> bool:
        """获取企业微信API调用凭证（access_token），企业微信所有接口都需要这个"""
        try:
            logger.info("开始获取企业微信Access Token...")
            url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={CONFIG['corpid']}&corpsecret={CONFIG['corpsecret']}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # 触发HTTP错误（如400/401）
            result = response.json()

            if result.get("errcode") == 0:
                self.access_token = result["access_token"]
                logger.info(f"Access Token获取成功 | 有效期：{result.get('expires_in', 7200)}秒")
                return True
            else:
                logger.error(f"获取Access Token失败 | 错误码：{result['errcode']}，错误信息：{result['errmsg']}")
                return False
        except Exception as e:
            logger.error(f"获取Access Token异常 | {str(e)}", exc_info=True)
            return False

    # -------------------------- 2. 读取企业微信群众表数据（按你的参考格式） --------------------------
    def get_people_table_data(self) -> List[Dict]:
        """调用企业微信表格代理API，读取群众表数据（严格遵循你给的「通用查询表单」格式）"""
        if not self.access_token:
            logger.error("未获取到Access Token，无法读取群众表")
            return []

        try:
            logger.info("开始读取群众表数据（企业微信表格）...")
            # 构造请求体（完全匹配你提供的「通用查询表单」格式）
            payload = {
                "action": "通用查询表单",  # 参考你给的接口格式
                "company": "花都家庭医生",  # 你提供的公司名称
                "WordList": {
                    "docid": CONFIG["people_table"]["docid"],
                    "sheet_id": CONFIG["people_table"]["sheet_id"],
                    "view_id": CONFIG["people_table"]["view_id"],
                    "access_token": self.access_token  # 部分代理API需携带token到请求体
                }
            }

            # 发起POST请求（企业微信表格代理API通常用POST）
            response = requests.post(
                url=CONFIG["table_api_url"],
                json=payload,
                timeout=20
            )
            response.raise_for_status()
            result = response.json()

            # 处理表格API返回（参考你之前JS代码的成功判断逻辑）
            if result.get("success") or result.get("errcode") == 0:
                people_data = result.get("data", [])
                logger.info(f"群众表数据读取成功 | 共{len(people_data)}条记录（预期49条）")

                # 打印前2条记录的字段名，帮你确认是否有json字段
                if people_data:
                    sample_fields = list(people_data[0].keys())
                    logger.debug(f"群众表字段名示例：{sample_fields}")
                return people_data
            else:
                logger.error(f"读取群众表失败 | 错误信息：{result.get('message', result.get('errmsg', '未知错误'))}")
                return []
        except Exception as e:
            logger.error(f"读取群众表异常 | {str(e)}", exc_info=True)
            return []

    # -------------------------- 3. 解析出生日期（适配你给的JSON格式） --------------------------
    def parse_birthday_from_info(self, info: Dict) -> Optional[str]:
        """从info字典中解析出生日期，返回「MM-DD」格式（适配你给的「1988年8月23日」格式）"""
        # 从info中提取出生日期（你给的JSON示例中字段名是"出生日期"）
        birthday_str = info.get("出生日期", "").strip()
        if not birthday_str:
            logger.debug("info对象中未找到「出生日期」字段")
            return None

        try:
            # 专门处理你给的「YYYY年MM月DD日」格式（如1988年8月23日、2004年8月23日）
            birth_date = datetime.datetime.strptime(birthday_str, "%Y年%m月%d日")
            return birth_date.strftime("%m-%d")  # 转为MM-DD格式
        except ValueError:
            # 兼容单数月/日（如2004年8月3日 → 08-03）
            try:
                birth_date = datetime.datetime.strptime(birthday_str, "%Y年%m月%d日")
                return birth_date.strftime("%m-%d")
            except ValueError:
                logger.warning(f"无法解析出生日期格式 | 原始值：{birthday_str}（预期：YYYY年MM月DD日）")
                return None

    # -------------------------- 4. 筛选今日生日用户（核心逻辑） --------------------------
    def filter_today_birthday_people(self, people_data: List[Dict]) -> List[Dict]:
        """筛选今日生日的用户，包含：记录序号、JSON对象序号、externalUsrid、完整info+tags、空话术"""
        birthday_list = []

        for record_idx, record in enumerate(people_data, 1):  # record_idx：群众表记录序号（从1开始）
            logger.debug(f"\n===== 处理群众表第{record_idx}条记录 =====")

            # 1. 提取json字段（你确认的字段名是"json"）
            json_field = CONFIG["people_table"]["json_field_name"]
            if json_field not in record:
                logger.debug(f"该记录无「{json_field}」字段，跳过")
                continue

            json_str = record[json_field].strip()
            if not json_str:
                logger.debug(f"「{json_field}」字段值为空，跳过")
                continue
            logger.debug(f"读取到json字段值（前150字符）：{json_str[:150]}...")

            # 2. 解析json字段（支持数组/单个对象，你给的示例是数组）
            try:
                json_objects = json.loads(json_str)
                # 若不是数组，转为数组（统一处理）
                if not isinstance(json_objects, list):
                    json_objects = [json_objects]
                logger.debug(f"json解析成功 | 共{len(json_objects)}个对象（info+tags为1个对象）")
            except json.JSONDecodeError as e:
                logger.warning(f"json解析失败 | 错误位置：{e.pos}，错误原因：{e.msg}，原始值：{json_str[:100]}...")
                continue

            # 3. 遍历每个json对象，检查生日
            for obj_idx, obj in enumerate(json_objects, 1):  # obj_idx：json对象序号（从1开始）
                logger.debug(f"----- 处理json中第{obj_idx}个对象 -----")

                # 提取info和tags（你给的JSON结构是「info{}+tags{}」）
                info = obj.get("info", {})
                tags = obj.get("tags", {})
                if not isinstance(info, dict) or not isinstance(tags, dict):
                    logger.debug("对象结构异常（缺少info/tags），跳过")
                    continue

                # 4. 解析出生日期并匹配今日
                birth_month_day = self.parse_birthday_from_info(info)
                if not birth_month_day:
                    continue

                if birth_month_day == self.target_month_day:
                    # 5. 收集所需字段（完全按你的需求）
                    birthday_user = {
                        "群众表记录序号": record_idx,
                        "json对象序号": obj_idx,
                        "externalUsrid": record.get("externalUserid", ""),  # 你表格中的外部用户ID字段
                        "adderUsrid": record.get("谁加的好友的usrid", ""),  # 你表格中的添加人ID字段
                        "json_data": {  # 保留完整的info+tags对象
                            "info": info,
                            "tags": tags
                        },
                        "blessing": ""  # 初始化空话术
                    }
                    birthday_list.append(birthday_user)
                    logger.info(
                        f"✅ 匹配到今日生日用户 | 记录{record_idx}-对象{obj_idx} "
                        f"| 姓名：{info.get('姓名', '未知')} | 出生日期：{info.get('出生日期', '未知')}"
                    )
                else:
                    logger.debug(
                        f"日期不匹配 | 解析生日：{birth_month_day} "
                        f"| 今日目标：{self.target_month_day} "
                        f"| 姓名：{info.get('姓名', '未知')}"
                    )

        logger.info(f"\n筛选完成 | 今日共匹配到{len(birthday_list)}个生日用户")
        return birthday_list

    # -------------------------- 5. 完整执行流程 --------------------------
    def run(self) -> List[Dict]:
        logger.info("=== 开始执行企业微信群众表生日筛选 ===")
        # 步骤1：获取Access Token
        if not self.get_wechat_access_token():
            return []
        # 步骤2：读取群众表数据
        people_data = self.get_people_table_data()
        if not people_data:
            logger.warning("无群众表数据可处理")
            return []
        # 步骤3：筛选今日生日用户
        birthday_people = self.filter_today_birthday_people(people_data)
        # 步骤4：返回结果
        logger.info("=== 生日筛选流程结束 ===")
        return birthday_people


# -------------------------- 执行代码 & 输出结果 --------------------------
if __name__ == "__main__":
    filter = WeChatPeopleBirthdayFilter()
    result = filter.run()

    # 打印最终结果
    print("\n" + "=" * 50)
    print("今日生日用户筛选结果")
    print("=" * 50)
    if not result:
        print("❌ 未匹配到今日生日的用户")
    else:
        print(f"✅ 共匹配到{len(result)}个今日生日用户：")
        for idx, user in enumerate(result, 1):
            print(f"\n【结果{idx}】")
            print(f"1. 群众表记录序号：第{user['群众表记录序号']}条")
            print(f"2. JSON对象序号：第{user['json对象序号']}个")
            print(f"3. 外部用户ID（externalUsrid）：{user['externalUsrid']}")
            print(f"4. 添加人ID（adderUsrid）：{user['adderUsrid']}")
            print(f"5. 用户姓名：{user['json_data']['info'].get('姓名', '未知')}")
            print(f"6. 出生日期：{user['json_data']['info'].get('出生日期', '未知')}")
            print(f"7. 个性化话术：{user['blessing'] or '待生成'}")
            print(f"8. JSON片段（info+tags）：{json.dumps(user['json_data'], ensure_ascii=False)[:200]}...")