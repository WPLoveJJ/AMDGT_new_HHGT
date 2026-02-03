import pandas as pd
import logging
import chardet
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def detect_file_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # 读取前10KB检测编码
    result = chardet.detect(raw_data)
    return result['encoding'], result['confidence']


def extract_data(csv_path, target_usrid, output_file="pass.csv"):
    """
    增强版提取工具：自动检测编码和格式，处理非标准CSV
    """
    id_column = "回访账号usrid"
    ext_column = "external_userid"

    # 检查文件是否存在
    if not Path(csv_path).exists():
        logger.error(f"文件不存在：{csv_path}")
        return

    # 检查文件扩展名
    file_ext = Path(csv_path).suffix.lower()
    if file_ext not in ['.csv', '.txt']:
        logger.warning(f"文件扩展名不是CSV或TXT，可能不是表格文件：{file_ext}")

    # 检测文件编码
    detected_encoding, confidence = detect_file_encoding(csv_path)
    logger.info(f"检测到文件编码：{detected_encoding}（可信度：{confidence:.2f}）")

    # 准备尝试的编码列表（检测到的编码 + 常见编码）
    encodings = [detected_encoding] if detected_encoding else []
    encodings += ['utf-8', 'gbk', 'gb2312', 'utf-16', 'iso-8859-1', 'windows-1252']
    encodings = list(set(encodings))  # 去重

    # 尝试读取文件（支持多种分隔符）
    df = None
    separators = [',', '\t', ';', '|']  # 常见分隔符

    for encoding in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(
                    csv_path,
                    encoding=encoding,
                    sep=sep,
                    dtype=str,
                    error_bad_lines=False,  # 跳过错误行
                    warn_bad_lines=True
                )
                logger.info(f"成功读取文件：编码={encoding}，分隔符={repr(sep)}")
                break
            except Exception as e:
                continue
        if df is not None:
            break

    # 终极方案：用Python内置方法读取后转换
    if df is None:
        logger.info("尝试最后方案：强制读取并转换")
        try:
            with open(csv_path, 'r', encoding=detected_encoding or 'utf-8', errors='replace') as f:
                content = f.read()
            # 简单处理后用StringIO读取
            from io import StringIO
            df = pd.read_csv(StringIO(content), sep=None, engine='python', dtype=str)
            logger.info("强制读取成功（可能存在格式问题）")
        except Exception as e:
            logger.error(f"所有尝试均失败，无法读取文件：{str(e)}")
            logger.info("请确认文件是CSV/TXT格式，且内容为表格数据")
            return

    # 检查列名
    if id_column not in df.columns:
        logger.error(f"未找到列名 '{id_column}'，文件中的列名：{df.columns.tolist()}")
        return

    if ext_column not in df.columns:
        logger.error(f"未找到列名 '{ext_column}'，文件中的列名：{df.columns.tolist()}")
        return

    # 筛选数据
    mask = df[id_column] == target_usrid
    result = df[mask][ext_column].dropna()
    logger.info(f"找到 {len(result)} 条匹配记录")

    # 保存结果
    result.to_csv(output_file, index=False, header=[ext_column], encoding='utf-8')
    logger.info(f"结果已保存到 {output_file}")


if __name__ == "__main__":
    # 请修改为你的文件路径
    CSV_FILE_PATH = "customer_info_20250829.csv"  # 确保路径正确
    TARGET_USRID = "wobb8UCgAAg_T9esG_VDnpOF1MZ2IejQ"
    extract_data(CSV_FILE_PATH, TARGET_USRID)
