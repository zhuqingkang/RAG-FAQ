# mysql_qa/utils/__init__.py
# 本包放「组合与辅助」模块：qa_service（单例）、mysql_qa_system（组合器）、
# chat_store（会话存储）、preprocess（文本预处理）。
#
# 刻意不做聚合导出：一旦在这里 `from .preprocess import preprocess_text`，
# 那么 `from mysql_qa.utils import chat_store`（只依赖 pymysql + base）也会连带加载 jieba。
# 按需直接导入子模块即可：
#   from mysql_qa.utils import chat_store
#   from mysql_qa.utils.preprocess import preprocess_text
