import pandas as pd

# 1. 读取两份 TSV 文件（指定分隔符为 \t）
df_name = pd.read_csv("E:/AMDGT/plus - 副本/AMDGT-main/AMDGT-main/data/B-dataset/proteinid_mapping_genename.tsv", sep="\t")
df_id = pd.read_csv("E:/AMDGT/plus - 副本/AMDGT-main/AMDGT-main/data/B-dataset/proteinid_mapping_geneid.tsv", sep="\t")

# 2. 按共同字段合并（如 protein_id）
#    how="inner" 表示只保留两边都匹配的蛋白质 ID
df_merged = pd.merge(
    df_name,
    df_id,
    on="protein_id",  # 替换为实际的关联字段（如 uniprot_id）
    how="inner"
)

# 3. 保存融合后的结果（输出为新的 TSV 文件）
df_merged.to_csv("protein_gene_merged.tsv", sep="\t", index=False)

# 4. 检查结果
print("融合后的数据：")
print(df_merged.head())
print(f"共融合 {len(df_merged)} 条记录")