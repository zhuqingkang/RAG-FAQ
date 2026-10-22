# utils/preprocess.py
# 导入分词库
import jieba
# 导入日志
# from pathlib import Path
# import os
# import sys
# cur_dir = os.path.dirname(os.path.abspath(__file__))
# mysql_dir =os.path.dirname(cur_dir)
# project_dir =os.path.dirname(mysql_dir)
# sys.path.insert(0,project_dir)
# pro_dir = Path(__file__).parent.parent.parent

from base import logger

def preprocess_text(text):
    # 预处理文本
    logger.info("开始预处理文本")
    try:
        # 分词并转换为小写
        return jieba.lcut(text.lower())
    except AttributeError as e:
        # 记录预处理失败
        logger.error(f"文本预处理失败: {e}")
        # 返回空列表
        return []


if __name__ == '__main__':
    text = preprocess_text('黑马程序员')
    print(text)