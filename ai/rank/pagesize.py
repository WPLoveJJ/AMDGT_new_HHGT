import requests
import time
import json
from typing import List, Dict, Any, Optional

# 配置参数
CONFIG = {
    "api_url": "https://smallwecom.yesboss.work/smarttable",
    "headers": {"Content-Type": "application/json"},
    "company": "花都家庭医生",  # 根据实际情况修改
    "table": {
        "docid": "dcijMKYnKxjShY-j07ZOHvts5m58JesjOINDiFoLNuZLGLvCpilhV5P9UjJ8ABc1b9d_tFBRu_Xvl8FvmlZk1jXA",
        "sheet_id": "tPQ8b9",
        "view_id": "vflTWJ"
    }
}


class PaginatedFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))

    def fetch_with_pagination(self, expected_total: int = 1300) -> List[Dict[str, Any]]:
        """使用循环通用查询表单API获取数据"""
        all_data = []
        offset = 0
        batch_size = 200  # 每批获取200条
        max_attempts = 3
        consecutive_failures = 0

        print(f"\n开始获取数据（预期总数: {expected_total}）...")

        while len(all_data) < expected_total:
            params = {
                "action": "循环通用查询表单",
                "company": CONFIG["company"],
                "WordList": {
                    "docid": CONFIG["table"]["docid"],
                    "sheet_id": CONFIG["table"]["sheet_id"],
                    "view_id": CONFIG["table"]["view_id"],
                    "offset": offset
                }
            }

            for attempt in range(max_attempts):
                try:
                    start_time = time.time()
                    response = self.session.post(
                        CONFIG["api_url"],
                        headers=CONFIG["headers"],
                        json=params,
                        timeout=60
                    )
                    response.raise_for_status()

                    data = response.json().get("data", [])
                    if not data:
                        print("\n[提示] 已到达数据末尾")
                        return all_data

                    all_data.extend(data)
                    current_count = len(data)
                    offset += current_count  # 使用实际返回数量更新offset

                    print(
                        f"批次 {len(all_data) - current_count + 1}-{len(all_data)} "
                        f"获取 {current_count} 条 | "
                        f"耗时 {time.time() - start_time:.1f}秒"
                    )

                    consecutive_failures = 0
                    time.sleep(0.5)  # 礼貌性延迟
                    break

                except Exception as e:
                    consecutive_failures += 1
                    wait_time = 2 ** attempt
                    print(
                        f"[重试] 第{attempt + 1}次失败: {type(e).__name__} | "
                        f"{wait_time}秒后重试..."
                    )
                    time.sleep(wait_time)
            else:
                print(f"[终止] 连续失败{max_attempts}次，停止获取")
                break

        return all_data


def main():
    print("=" * 60)
    print("客户数据获取系统 - 循环分页版")
    print("=" * 60)

    fetcher = PaginatedFetcher()
    start_time = time.time()

    # 获取数据
    all_data = fetcher.fetch_with_pagination()

    # 结果分析
    print("\n" + "=" * 50)
    print(f"获取完成！共获得 {len(all_data)} 条记录")
    print(f"总耗时: {time.time() - start_time:.2f}秒")

    # 数据去重（基于某个唯一字段，例如ID）
    unique_data = []
    seen_ids = set()
    for item in all_data:
        item_id = item.get("人员", [{}])[0].get("user_id")  # 根据实际唯一字段调整
        if item_id not in seen_ids:
            seen_ids.add(item_id)
            unique_data.append(item)

    print(f"去重后数量: {len(unique_data)}")

    # 保存数据
    try:
        with open("customer_data_paginated.json", "w", encoding="utf-8") as f:
            json.dump(unique_data, f, ensure_ascii=False, indent=2)
        print("✅ 数据已保存到 customer_data_paginated.json")
    except Exception as e:
        print(f"保存失败: {str(e)}")


if __name__ == "__main__":
    main()