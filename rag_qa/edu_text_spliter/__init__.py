# rag_qa/edu_text_spliter/__init__.py
# 这里只导出轻量的中文递归切块器：它只依赖 langchain_text_splitters。
# AliTextSplitter 会拉起 modelscope + transformers（重），需要时请显式导入：
#   from rag_qa.edu_text_spliter.edu_model_text_spliter import AliTextSplitter
from .edu_chinese_recursive_text_splitter import ChineseRecursiveTextSplitter

__all__ = ["ChineseRecursiveTextSplitter"]
