# mysql_qa/retrieval/__init__.py
# 检索子包：统一从包名导入，不必记模块文件名
#   from mysql_qa.retrieval import BM25Search
from .bm25_search import BM25Search

__all__ = ["BM25Search"]
