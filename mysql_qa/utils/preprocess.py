# mysql_qa/utils/preprocess.py
# 分词只在真正调用时才导入 jieba：
# 这样 `from mysql_qa.utils import chat_store` 这类轻量导入不会被分词器拖慢。
from base import logger


def preprocess_text(text):
    # 预处理文本
    logger.info("开始预处理文本")
    # 延迟导入 jieba（首次导入会加载词典，约数百毫秒）
    import jieba
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
