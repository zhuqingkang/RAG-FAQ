# rag_qa/rag_service.py
# RAG 问答单例：Flask 版与 FastAPI 异步版共用，保证只初始化一次。
# 与 mysql_qa/utils/qa_service.py 的设计保持一致：惰性初始化 + 线程安全 + 失败降级。
import sys
import threading
import time
from pathlib import Path

# ===== 路径处理：只保证项目根在 sys.path，其余一律靠包导入（rag_qa.core.xxx） =====
_PROJECT_DIR = Path(__file__).resolve().parent.parent      # .../integrated_qa_system
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from base import Config, logger

conf = Config()

# 初始化失败后的冷却时间（秒）：避免 Milvus / LLM 不可用时每个请求都重试一次
RETRY_COOLDOWN = 60

_rag_system = None          # 单例
_init_lock = threading.Lock()
_failure_reason = None      # 最近一次初始化失败的原因
_failed_at = 0.0            # 失败时间戳（time.monotonic）


def _build_llm_client():
    """
    构造 OpenAI 兼容客户端，并返回一个「prompt -> answer」的调用函数。

    只依赖 OpenAI 协议，因此 DashScope / DeepSeek / 本地 vLLM / Ollama / OneAPI
    都可以直接用：换供应商只改 config.ini 的 [llm] 段，不用动代码。
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=conf.LLM_API_KEY,
        base_url=conf.LLM_BASE_URL,
        timeout=conf.LLM_TIMEOUT,
    )

    def call_llm(prompt):
        """与 rag_main.py 中的 call_dashscope 行为保持一致。"""
        try:
            completion = client.chat.completions.create(
                model=conf.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手."},
                    {"role": "user", "content": prompt},
                ],
            )
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content
            logger.error("LLM 调用返回空消息")
            return None
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return None

    return call_llm


def _build_rag_system():
    """真正耗时的初始化：连接 Milvus、加载 BGE 模型、加载意图分类器。"""
    # 重库在这里才 import，保证只 import 本模块时几乎没有开销
    from rag_qa.core.vector_store import VectorStore
    from rag_qa.core.rag_system import RAGSystem

    vector_store = VectorStore(
        collection_name=conf.MILVUS_COLLECTION_NAME,
        host=conf.MILVUS_HOST,
        port=conf.MILVUS_PORT,
        database=conf.MILVUS_DATABASE_NAME,
    )
    llm = _build_llm_client()
    rag = RAGSystem(vector_store, llm)

    # BGE-M3 与 reranker 都是惰性加载：这里提前触发一次，
    # 让「预热」真正把模型读进显存，避免首个请求额外等十几秒。
    # 加载失败只记录警告，不阻断服务（真正用到时会再尝试）。
    try:
        _ = rag.vector_store.embedding_function
        _ = rag.vector_store.reranker
    except Exception as e:
        logger.warning(f"RAG 模型预热未完成（请求时会重新加载）: {e}")

    return rag


def get_rag_system() -> "object":
    """
    获取（惰性初始化）RAG 系统单例。
    初始化失败时抛出 RuntimeError，并在 RETRY_COOLDOWN 秒内快速失败，不再重试。
    """
    global _rag_system, _failure_reason, _failed_at

    if _rag_system is not None:
        return _rag_system

    with _init_lock:
        if _rag_system is not None:
            return _rag_system

        # 冷却期内直接失败，避免请求被反复拖住
        if _failure_reason and (time.monotonic() - _failed_at) < RETRY_COOLDOWN:
            raise RuntimeError(_failure_reason)

        try:
            logger.info("正在初始化 RAG 问答系统 ...")
            _rag_system = _build_rag_system()
            _failure_reason = None
            logger.info("RAG 问答系统初始化完成")
        except Exception as e:
            _failure_reason = f"RAG 初始化失败: {e}"
            _failed_at = time.monotonic()
            logger.error(_failure_reason)
            logger.exception(e)
            raise RuntimeError(_failure_reason) from e

    return _rag_system


def ensure_ready():
    """预热入口：供服务启动时在后台线程里调用。异常不外抛，只记录日志。"""
    try:
        get_rag_system()
        return True
    except Exception:
        return False


def answer(question, source_filter=None, history=None):
    """
    调用 RAG 生成答案。
    :param question: 用户问题
    :param source_filter: 学科过滤（None 表示不过滤），取值见 Config().VALID_SOURCES
    :param history: 最近几轮问答 [{"question": ..., "answer": ...}]；None 表示不带历史
    :return: (answer, error)；answer 为 None 时 error 说明原因
    """
    try:
        rag = get_rag_system()
    except Exception as e:
        return None, str(e)

    try:
        result = rag.generate_answer(question, source_filter=source_filter, history=history)
        return result, ""
    except Exception as e:
        logger.error(f"RAG 生成答案失败: {e}")
        logger.exception(e)
        return None, f"RAG 生成答案失败: {e}"
