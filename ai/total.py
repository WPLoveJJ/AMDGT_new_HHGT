import requests
import json
import urllib.parse

# 钉钉应用配置
DINGTALK_CONFIG = {
    "app_key": "dingoicseqn2bmdcazpl",
    "app_secret": "hiiqLe8teDkAADlJh9eklgsbtGIvrG8hPJyOC8as04wzG69OGmgaY_vQ_gyKTXEg",
    "base_id": "YndMj49yWjDEYy3ECQwPlLkgJ3pmz5aA",
    "sheet_name": "配置表",
    "operator_id": "jYEXEC84RV3QE3sm0UaeDwiEiE"
}


def get_dingtalk_access_token():
    """获取钉钉访问令牌"""
    url = "https://api.dingtalk.com/v1.0/oauth2/accessToken"
    headers = {"Content-Type": "application/json"}
    payload = {
        "appKey": DINGTALK_CONFIG["app_key"],
        "appSecret": DINGTALK_CONFIG["app_secret"]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("accessToken")
    except Exception as e:
        print(f"获取访问令牌失败: {e}")
        return None


def parse_multi_json(config_value):
    """解析可能包含多个JSON对象的字符串"""
    if not config_value:
        return []

    # 尝试解析为单个JSON
    try:
        return [json.loads(config_value)]
    except json.JSONDecodeError:
        pass

    # 尝试解析多个连续JSON对象
    objects = []
    start = 0
    brace_count = 0

    for i, char in enumerate(config_value):
        if char == '{':
            if brace_count == 0:
                start = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                try:
                    objects.append(json.loads(config_value[start:i + 1]))
                except json.JSONDecodeError:
                    pass

    return objects if objects else [config_value]


def get_family_doctor_configs():
    access_token = get_dingtalk_access_token()
    if not access_token:
        return None
    # 构建API请求
    base_url = "https://api.dingtalk.com/v1.0/notable/bases/"
    full_url = f"{base_url}{DINGTALK_CONFIG['base_id']}/sheets/{urllib.parse.quote(DINGTALK_CONFIG['sheet_name'])}/records"
    headers = {
        "x-acs-dingtalk-access-token": access_token,
        "Content-Type": "application/json"
    }
    params = {"maxResults": 100, "operatorId": DINGTALK_CONFIG["operator_id"]}
    try:
        response = requests.get(full_url, headers=headers, params=params)
        response.raise_for_status()
        records = response.json().get("records", [])
        result = []
        for record in records:
            fields = record.get("fields", {})
            # 筛选家医任务
            if fields.get("任务名称") != "家医":
                continue
            config_value = fields.get("通用配置表")
            if not config_value:
                continue
            # 解析配置（可能多个）
            for config in parse_multi_json(config_value):
                if isinstance(config, dict):
                    # 关键修改：直接返回钉钉原始结构
                    result.append({
                        "record_id": record.get("id"),
                        "region": fields.get("地区", ""),
                        "config": config  # 直接返回钉钉原始结构
                    })
        return result if result else None
    except Exception as e:
        print(f"获取配置失败: {e}")
        return None



def main():
    """测试输出"""
    if configs := get_family_doctor_configs():
        for i, config in enumerate(configs, 1):
            print(f"\n配置 {i}:")
            print(f"记录ID: {config['record_id']}")
            print(f"地区: {config['region']}")
            print("配置内容:")
            print(json.dumps(config['config'], indent=2, ensure_ascii=False))
    else:
        print("未获取到家医任务配置")


if __name__ == "__main__":
    main()