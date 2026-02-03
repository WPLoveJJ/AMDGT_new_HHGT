import requests
import json
from datetime import datetime
from typing import List, Dict, Optional

# 钉钉应用配置
DINGTALK_CONFIG = {
    "app_key": "dingoicseqn2bmdcazpl",
    "app_secret": "hiiqLe8teDkAADlJh9eklgsbtGIvrG8hPJyOC8as04wzG69OGmgaY_vQ_gyKTXEg",
    "base_id": "YndMj49yWjDEYy3ECQwPlLkgJ3pmz5aA",
    "sheet_name": "配置表",
    "operator_id": "jYEXEC84RV3QE3sm0UaeDwiEiE"
}

class DingTalkUpdater:
    def __init__(self):
        self.access_token = None
        self.api_url = f"https://api.dingtalk.com/v1.0/notable/bases/{DINGTALK_CONFIG['base_id']}/sheets/{DINGTALK_CONFIG['sheet_name']}/records"
        self.today = datetime.now().strftime("%Y-%m-%d")

    def get_access_token(self) -> Optional[str]:
        """获取访问令牌"""
        try:
            response = requests.post(
                "https://api.dingtalk.com/v1.0/oauth2/accessToken",
                json={
                    "appKey": DINGTALK_CONFIG["app_key"],
                    "appSecret": DINGTALK_CONFIG["app_secret"]
                },
                timeout=10
            )
            response.raise_for_status()
            self.access_token = response.json().get("accessToken")
            return self.access_token if self.access_token else None
        except Exception as e:
            print(f"获取access_token失败: {str(e)}")
            return None

    def batch_update_dates(self, record_ids: List[str]) -> Dict:
        """
        批量更新多个记录的最新完成日期
        :param record_ids: 要更新的记录ID列表
        :return: 更新结果
        """
        if not self.access_token and not self.get_access_token():
            return {"success": False, "error": "无法获取有效的访问令牌"}

        if not record_ids:
            return {"success": False, "error": "未提供任何记录ID"}

        # 构建批量更新的数据结构
        records = [
            {"id": record_id, "fields": {"最新完成日期": self.today}}
            for record_id in record_ids
        ]

        headers = {
            "x-acs-dingtalk-access-token": self.access_token,
            "Content-Type": "application/json"
        }

        payload = {
            "records": records,
            "operatorId": DINGTALK_CONFIG["operator_id"]
        }

        try:
            response = requests.put(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=15
            )

            if response.status_code == 200:
                return {
                    "success": True,
                    "updated_date": self.today,
                    "updated_count": len(record_ids),
                    "record_ids": record_ids,
                    "response": response.json()
                }
            else:
                return {
                    "success": False,
                    "error": f"更新失败: {response.status_code} - {response.text}",
                    "record_ids": record_ids
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"请求异常: {str(e)}",
                "record_ids": record_ids
            }

def main():
    print("🚀 钉钉多维表批量更新工具")
    print(f"📅 更新日期: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"📄 目标表格: {DINGTALK_CONFIG['sheet_name']}\n")

    # 示例：要更新的记录ID列表
    record_ids_to_update = [
        "OaXBjUzVCq",  # 你提供的记录ID
        # 可以添加更多记录ID
        # "record_id_2",
        # "record_id_3"
    ]

    if not record_ids_to_update:
        print("❌ 请在代码中添加要更新的record_id")
        return

    print(f"准备更新 {len(record_ids_to_update)} 条记录...")
    updater = DingTalkUpdater()
    result = updater.batch_update_dates(record_ids_to_update)

    print("\n=== 批量更新结果 ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
