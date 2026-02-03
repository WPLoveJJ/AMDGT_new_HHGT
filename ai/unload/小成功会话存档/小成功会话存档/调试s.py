import requests
import time
from datetime import datetime

def fetch_all_by_time_range(start_ts, end_ts):
    all_msgs = []
    seq = 0
    while True:
        url = f"http://localhost:5000/api/messages/advanced"
        params = {
            "seq": seq,
            "limit": 10,
            "start_time": start_ts,
            "end_time": end_ts
        }
        resp = requests.get(url, params=params).json()
        msgs = resp.get("data", {}).get("messages", [])
        if not msgs:
            break
        all_msgs.extend(msgs)
        seq = resp.get("data", {}).get("next_seq", seq + 100)
    return all_msgs

# 示例：2025-08-20 全天
start_ts = int(time.mktime(time.strptime("2025-08-23 00:00:00", "%Y-%m-%d %H:%M:%S")))
end_ts = int(time.mktime(time.strptime("2025-08-23 23:59:59", "%Y-%m-%d %H:%M:%S")))

messages = fetch_all_by_time_range(start_ts, end_ts)

# 打印前3条
for msg in messages[:100]:
    print("=" * 40)
    print(f"发送者: {msg['from']}")
    print(f"接收者: {msg['tolist']}")
    print(f"时间: {msg['datetime']}")
    print(f"内容: {msg.get('content', '[非文本消息]')}")