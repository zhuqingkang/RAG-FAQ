# rag_qa/__init__.py
# RAG 能力包。对外推荐这些导入路径：
#   from rag_qa.rag_service import answer, ensure_ready     # 服务层（惰性初始化，import 无重开销）
#   from rag_qa.core.rag_system import RAGSystem           # 核心编排
#   from rag_qa.core.vector_store import VectorStore       # 向量检索
#
# 这里刻意不做聚合导出：rag_service 会在 import 时构造 Config（读 config.ini），
# core 下的 query_classifier / vector_store 会拉起 torch / pymilvus。
# 若在此处 re-export，任何一个轻量子模块（如 rag_qa.core.json_cache）的导入
# 都会连带把重依赖拖起来，破坏「惰性加载」的设计。
