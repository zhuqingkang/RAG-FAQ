# rag_qa/core/__init__.py
# 核心组件包。这里刻意不做聚合导出（不写 from .rag_system import RAGSystem）：
#   query_classifier 顶层就 import torch、vector_store 会用到 pymilvus，
#   一旦聚合导出，只是想 import 轻量模块（如 json_cache）也会把重库全拉起来。
# 需要哪个组件就单独导入：
#   from rag_qa.core.rag_system import RAGSystem
#   from rag_qa.core.json_cache import load_catalog
# 包内部模块之间一律用相对导入（from .prompts import ...），不依赖 sys.path 顺序。
