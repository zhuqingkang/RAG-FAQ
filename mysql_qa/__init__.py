# mysql_qa/__init__.py
# FAQ 能力包（MySQL + Redis + BM25）。对外推荐这些导入路径：
#   from mysql_qa.utils.qa_service import get_qa_system, ensure_ready   # 单例 / 预热
#   from mysql_qa.utils import chat_store                              # 会话存储
#   from mysql_qa.retrieval import BM25Search                          # 检索
#   from mysql_qa.db import MySQLClient                                # 数据源
#   from mysql_qa.cache import RedisClient                             # 缓存
#
# 这里刻意不做聚合导出：mysql_qa.retrieval 会拉起 jieba / rank_bm25，
# 若在包入口 re-export，只想拿 chat_store（纯 sqlite3）的调用方也会被迫付这份开销。
