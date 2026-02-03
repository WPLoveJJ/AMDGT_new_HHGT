# 安装必要的包（首次使用时运行）
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install(c("clusterProfiler", "org.Hs.eg.db", "GO.db", "GOSemSim"), update = FALSE)

# 加载包
library(clusterProfiler)
library(org.Hs.eg.db)  # 人类基因注释库（其他物种需替换，如小鼠用org.Mm.eg.db）
library(GOSemSim)
library(dplyr)

# 1. 读取融合后的药物-基因关联表（关键：修改为你的文件路径）
# Windows路径需用 / 或 \\ 分隔，此处已按你的路径处理
drug_gene <- read.delim(
  "E:/AMDGT/plus - 副本/AMDGT-main/AMDGT-main/data/B-dataset/protein_gene_merged.tsv",
  sep = "\t",
  stringsAsFactors = FALSE
)

# 查看数据是否正确读取
head(drug_gene)
cat("总记录数：", nrow(drug_gene), "\n")

# 2. 提取去重的Entrez ID（用于GO注释）
all_genes <- unique(drug_gene$entrez_id)  # 确保列名是entrez_id，若不同需修改
all_genes <- as.character(all_genes[!is.na(all_genes)])  # 去除NA值
cat("去重后的基因数量：", length(all_genes), "\n")

# 3. 获取GO注释（以生物过程BP为例，可选MF/CC）
if (length(all_genes) > 0) {
  # 转换基因ID并获取GO注释
  go_annot <- bitr(
    gene = all_genes,
    fromType = "ENTREZID",  # 输入类型为Entrez ID
    toType = "GO",          # 输出GO术语
    OrgDb = org.Hs.eg.db,
    drop = TRUE
  )
  
  # 补充GO术语的类别（BP/MF/CC）
  if (nrow(go_annot) > 0) {
    go_annot$ONTOLOGY <- Ontology(go_annot$GO)  # 从GO.db获取类别
    head(go_annot)
    
    # 保存GO注释结果（路径与输入文件同目录）
    output_go <- "E:/AMDGT/plus - 副本/AMDGT-main/AMDGT-main/data/B-dataset/gene_go_annot.tsv"
    write.table(
      go_annot,
      output_go,
      sep = "\t",
      row.names = FALSE,
      quote = FALSE
    )
    cat("GO注释已保存至：", output_go, "\n")
  } else {
    warning("未获取到任何GO注释，请检查基因ID是否正确（是否为Entrez ID）")
  }
  
  # 4. 计算基因对的GO语义相似性（以BP为例）
  go_data <- godata(
    OrgDb = org.Hs.eg.db,
    ont = "BP",  # 与GO注释的类别保持一致
    keyType = "ENTREZID"
  )
  
  gene_sim_matrix <- mgeneSim(
    gene = all_genes,
    semData = go_data,
    measure = "Wang",  # 推荐的算法
    combine = NULL  # 返回所有基因对的相似性矩阵
  )
  
  # 保存基因相似性矩阵
  output_sim <- "E:/AMDGT/plus - 副本/AMDGT-main/AMDGT-main/data/B-dataset/gene_go_sim_matrix.tsv"
  write.table(
    as.matrix(gene_sim_matrix),
    output_sim,
    sep = "\t",
    row.names = TRUE,
    col.names = TRUE,
    quote = FALSE
  )
  cat("基因相似性矩阵已保存至：", output_sim, "\n")
  
} else {
  stop("未提取到有效基因ID，请检查drug_gene数据中的entrez_id列是否正确")
}
