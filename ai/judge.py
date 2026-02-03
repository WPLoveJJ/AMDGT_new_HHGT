import requests
import json
import re
from datetime import datetime, date, timedelta

# API基础配置
API_URL = "https://smallwecom.yesboss.work/smarttable"
HEADERS = {"Content-Type": "application/json"}

# 主配置表查询参数
MASTER_CONFIG = {
    "action": "通用查询表单",
    "company": "拉伸大师",
    "WordList": {
        "docid": "dc5FGUiUAkxPACyBl4VA_zqIIRDda2aLwH_wEGcYA0L3Md19OtsU2LhFNJBupxwDNbj-ZyI0YSUtrAtyiCiEFSIQ",
        "sheet_id": "q979lj",
        "view_id": "vukaF8"
    }
}

# 文件顶端常量后，新增通用取值函数
def _as_list(field):
    """统一将字段包装成列表"""
    if field is None:
        return []
    if isinstance(field, list):
        return field
    return [field]

def get_text(field) -> str:
    """从字段中提取文本，兼容 list/dict/str/int/float"""
    items = _as_list(field)
    if not items:
        return ""
    first = items[0]
    if isinstance(first, dict):
        return str(first.get("text") or first.get("label") or first.get("value") or "").strip()
    return str(first).strip()

def get_int(field):
    """从字段中提取整数，兼容 list/dict/数值/字符串，失败返回 None"""
    items = _as_list(field)
    if not items:
        return None
    first = items[0]
    if isinstance(first, (int, float)):
        return int(first)
    if isinstance(first, dict):
        for k in ("value", "text", "label"):
            v = first.get(k)
            if v is not None and str(v).strip() != "":
                try:
                    return int(float(str(v).strip()))
                except:
                    pass
        return None
    try:
        return int(float(str(first).strip()))
    except:
        return None

def extract_hospital_configs():
    """获取配置表中各医院的任务规则表配置"""
    try:
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(MASTER_CONFIG))
        response.raise_for_status()
        result = response.json()
        
        if "data" not in result or not isinstance(result["data"], list):
            print("获取配置表失败")
            return []
        
        config_list = []
        for idx, item in enumerate(result["data"], 1):
            hospital_info = item["values"].get("医院", [])
            hospital_name = hospital_info[0].get("text", f"未命名医院_{idx}") if (
                hospital_info and isinstance(hospital_info[0], dict)
            ) else f"未命名医院_{idx}"
            
            # 提取文档ID相关的配置
            docid_array = item["values"].get("文档ID", [])
            if not docid_array:
                print(f"\n【第{idx}条】{hospital_name}：无文档ID配置")
                continue
            
            # 拼接文档ID配置文本
            full_doc_text = ""
            for segment in docid_array:
                if isinstance(segment, dict):
                    full_doc_text += str(segment.get("text", ""))
                else:
                    full_doc_text += str(segment)
            
            # 提取docid
            docid_match = re.search(r'"docid"\s*:\s*"([^"]+)"', full_doc_text)
            docid = docid_match.group(1) if docid_match else None
            
            # 提取Taskrules配置
            task_rules_match = re.search(
                r'"Taskrules"\s*:\s*{\s*"tab"\s*:\s*"([^"]+)"\s*,\s*"viewId"\s*:\s*"([^"]+)"',
                full_doc_text
            )
            
            task_rules = None
            if task_rules_match:
                task_rules = {
                    "tab": task_rules_match.group(1),
                    "viewId": task_rules_match.group(2)
                }
                
            if docid and task_rules:
                config_list.append({
                    "医院": hospital_name,
                    "docid": docid,
                    "task_rules": task_rules
                })
                print(f"【第{idx}条】{hospital_name}：成功提取任务规则表配置")
            else:
                missing = []
                if not docid: missing.append("docid")
                if not task_rules: missing.append("Taskrules")
                print(f"【第{idx}条】{hospital_name}：缺少{','.join(missing)}配置，跳过")
        
        return config_list
        
    except requests.exceptions.RequestException as e:
        print(f"API请求失败: {e}")
        return []
    except Exception as e:
        print(f"处理数据时发生错误: {e}")
        return []

def query_task_rules(config):
    """查询任务规则表，提取距离特定日期x天相关字段"""
    query_params = {
        "action": "通用查询表单",
        "company": "拉伸大师",
        "WordList": {
            "docid": config["docid"],
            "sheet_id": config["task_rules"]["tab"],
            "view_id": config["task_rules"]["viewId"]
        }
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(query_params))
        response.raise_for_status()
        result = response.json()
        
        if not isinstance(result.get("data"), list):
            print("  任务规则表查询失败")
            return []
        
        task_rules = []
        for idx, item in enumerate(result["data"], 1):
            values = item.get("values", {})
            record_id = item.get("record_id", "")
            
            # 添加调试信息
            print(f"  调试：第{idx}条记录的完整数据结构：{item}")
            print(f"  调试：提取到的record_id：'{record_id}'")
            
            # 使用安全取值函数，兼容 list/dict/标量
            task_name = get_text(values.get("任务名", []))
            start_days = get_int(values.get("距离特定日期x天（起始）", []))
            end_days = get_int(values.get("距离特定日期x天（结束）", []))
            interval_days = get_int(values.get("间隔x天重复", []))
            


            if task_name and start_days is not None:
                task_rule = {
                    "记录ID": record_id,
                    "任务名": task_name,
                    "距离特定日期x天（起始）": start_days,
                    "距离特定日期x天（结束）": end_days,
                    "间隔x天重复": interval_days
                }
                task_rules.append(task_rule)
                print(f"  第{idx}条规则'{task_name}'提取成功，起始：{start_days}天，结束：{end_days}天，间隔：{interval_days}天")
            else:
                missing = []
                if not task_name: missing.append("任务名")
                if start_days is None: missing.append("距离特定日期x天（起始）")
                print(f"  第{idx}条规则缺少{','.join(missing)}，跳过")
        
        print(f"  成功读取到 {len(task_rules)} 个有效任务规则")
        return task_rules
        
    except Exception as e:
        print(f"  查询任务规则表失败: {str(e)}")
        return []


def build_judgment_expression(task_rule):
    """根据任务规则字段构建判断表达式"""
    task_name = task_rule["任务名"]

    # 特殊处理：个性化生日祝福任务
    if "生日祝福" in task_name:
        # 增强的日期比较表达式，处理日期对象和字符串格式
        return ("check.month == datetime.now().month and check.day == datetime.now().day")

    # 其他任务保持原来的逻辑
    start_days = task_rule["距离特定日期x天（起始）"]
    end_days = task_rule["距离特定日期x天（结束）"]
    interval_days = task_rule["间隔x天重复"]

    # 基础条件：今天的日期大于等于check+起始天数
    base_condition = f"datetime.now().date() >= check + timedelta(days={start_days})"

    # 如果结束日期不为空，添加小于等于结束日期的条件
    if end_days is not None and end_days > 0:
        base_condition += f" and datetime.now().date() <= check + timedelta(days={end_days})"

    # 如果间隔重复不为空，添加取余条件
    if interval_days is not None and interval_days > 0:
        if end_days is not None and end_days > 0:
            # 有结束日期时，用结束日期作为基准
            interval_condition = f"(datetime.now().date() - check - timedelta(days={end_days})).days % {interval_days} == 0"
        else:
            # 没有结束日期时，用起始日期作为基准
            interval_condition = f"(datetime.now().date() - check - timedelta(days={start_days})).days % {interval_days} == 0"

        judgment_expression = f"({base_condition}) and ({interval_condition})"
    else:
        judgment_expression = base_condition

    return judgment_expression

def update_task_rule_judgment(config, task_rule, judgment_expression):
    """更新任务规则表中的判断式字段"""
    if not task_rule.get("记录ID"):
        print(f"  任务'{task_rule['任务名']}'缺少记录ID，无法更新")
        return False
    
    update_data = {
        "action": "通用更新表单",
        "company": "花都家庭医生",
        "WordList": {
            "docid": config["docid"],
            "sheet_id": config["task_rules"]["tab"],
            "view_id": config["task_rules"]["viewId"],
            "record_id": task_rule["记录ID"],
            "values": {
                "判断式": [{"type": "text", "text": judgment_expression}]
            }
        }
    }
    
    try:
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(update_data))
        response.raise_for_status()
        result = response.json()
        
        if result and result.get("success", False):
            print(f"  ✅ 任务'{task_rule['任务名']}'判断式更新成功")
            print(f"     判断式：{judgment_expression}")
            return True
        else:
            print(f"  ❌ 任务'{task_rule['任务名']}'判断式更新失败: {result}")
            return False
            
    except Exception as e:
        print(f"  ❌ 任务'{task_rule['任务名']}'更新异常: {str(e)}")
        return False

def process_all_hospitals():
    """处理所有医院的任务规则"""
    print("开始获取医院配置...")
    config_list = extract_hospital_configs()
    
    if not config_list:
        print("没有可用的医院配置")
        return
    
    total_success = 0
    total_tasks = 0
    
    for idx, config in enumerate(config_list, 1):
        print(f"\n===== 处理第{idx}个医院：{config['医院']} =====")
        
        # 查询任务规则表
        task_rules = query_task_rules(config)
        
        if not task_rules:
            print("  没有有效的任务规则，跳过")
            continue
        
        success_count = 0
        for task_rule in task_rules:
            total_tasks += 1
            
            # 构建判断表达式
            judgment_expression = build_judgment_expression(task_rule)
            
            # 更新判断式
            if update_task_rule_judgment(config, task_rule, judgment_expression):
                success_count += 1
                total_success += 1
        
        print(f"  医院'{config['医院']}'完成，成功更新 {success_count}/{len(task_rules)} 个任务")
    
    print(f"\n===== 全部处理完成 =====")
    print(f"总计：成功更新 {total_success}/{total_tasks} 个任务判断式")

if __name__ == "__main__":
    process_all_hospitals()