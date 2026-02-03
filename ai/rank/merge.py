import pandas as pd
import os
import re

# 中文数字到阿拉伯数字的映射（用于前两级部门中的数字标准化）
chinese_to_num = {
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9, '十': 10
}


def normalize_department_levels(dept_name):
    """
    提取部门前两级并标准化：
    1. 按'/'分割部门名称
    2. 只保留前两级，忽略第三级及以后
    3. 标准化前两级中的数字（处理二/2/第2等差异）
    """
    if not dept_name or pd.isna(dept_name):
        return ""

    # 转换为字符串并分割
    dept_str = str(dept_name).strip()
    dept_parts = dept_str.split('/')

    # 只取前两级
    first_two_parts = dept_parts[:2]  # 不管有多少级，只保留前两个

    # 标准化每一级中的数字
    normalized_parts = []
    for part in first_two_parts:
        part_norm = part.strip()
        # 处理中文数字
        for ch, num in chinese_to_num.items():
            part_norm = re.sub(rf'第{ch}', rf'第{num}', part_norm)  # 第X→第num
            part_norm = re.sub(rf'(?<!第){ch}', str(num), part_norm)  # X→num（不含第）
        # 去除数字前的"第"
        part_norm = re.sub(r'第(\d+)', r'\1', part_norm)
        normalized_parts.append(part_norm)

    # 合并前两级作为匹配依据
    return '/'.join(normalized_parts)


def merge_keep_all_columns(file1_path, file2_path, output_filename="最终结果.xlsx"):
    """
    按以下规则合并：
    1. 保留两表所有列，添加_表1/_表2后缀
    2. 成员名称必须完全匹配
    3. 部门仅需前两级匹配（忽略第三级及以后）
    4. 部门前两级中的数字自动标准化（二/2/第2视为相同）
    """
    try:
        # 验证文件存在
        for path in [file1_path, file2_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"文件不存在: {path}")

        # 读取数据
        df1 = pd.read_excel(file1_path)
        df2 = pd.read_excel(file2_path)

        # 为所有列添加表来源后缀
        df1 = df1.rename(columns={col: f"{col}_表1" for col in df1.columns})
        df2 = df2.rename(columns={col: f"{col}_表2" for col in df2.columns})

        print(f"数据读取成功：")
        print(f"表1：{df1.shape[0]}行，{df1.shape[1]}列")
        print(f"表2：{df2.shape[0]}行，{df2.shape[1]}列")

        # 提取部门前两级并标准化（用于匹配）
        df1['部门前两级_标准化_表1'] = df1['部门_表1'].apply(normalize_department_levels)
        df2['部门前两级_标准化_表2'] = df2['部门_表2'].apply(normalize_department_levels)

        # 创建匹配键（成员+部门前两级）
        df1['匹配键'] = df1['成员_表1'] + '_' + df1['部门前两级_标准化_表1']
        df2['匹配键'] = df2['成员_表2'] + '_' + df2['部门前两级_标准化_表2']

        # 按匹配键合并
        merged_df = pd.merge(
            df1,
            df2,
            on='匹配键',
            how='outer'
        )

        # 删除临时列
        merged_df = merged_df.drop(columns=[
            '匹配键',
            '部门前两级_标准化_表1',
            '部门前两级_标准化_表2'
        ])

        # 处理缺失值
        numeric_cols = merged_df.select_dtypes(include=['int64', 'float64']).columns
        merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)

        string_cols = merged_df.select_dtypes(include=['object']).columns
        merged_df[string_cols] = merged_df[string_cols].fillna('')

        # 保存为Excel
        output_path = os.path.abspath(output_filename)
        merged_df.to_excel(output_path, index=False)

        if os.path.exists(output_path):
            print(f"\n✅ 处理完成！Excel已生成：")
            print(f"文件路径：{output_path}")
            print(f"合并后：{merged_df.shape[0]}行，{merged_df.shape[1]}列")
            match_count = sum((merged_df['成员_表1'] != '') & (merged_df['成员_表2'] != ''))
            print(f"成员+部门前两级匹配成功的行数：{match_count}")
        else:
            raise Exception("Excel文件生成失败")

        return merged_df

    except Exception as e:
        print(f"❌ 操作失败：{str(e)}")
        return None


if __name__ == "__main__":
    # 替换为你的实际文件路径
    file1 = "成员联系居民统计.xlsx"
    file2 = "成员群聊数据统计 (1).xlsx"
    output_file = "部门前两级匹配合并结果.xlsx"

    result = merge_keep_all_columns(file1, file2, output_file)
