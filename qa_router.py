# qa_router.py
# 统一问答路由：把 mysql_qa（MySQL + BM25 + Redis 的 FAQ 检索）与
# rag_qa（Milvus + BGE 精排 + 大模型生成的 RAG）合成一个对外的问答能力。
#
# 路由策略（mode）：
#   auto —— 先查 FAQ，命中即返回；未命中或 FAQ 异常时转 RAG（默认）
#   faq  —— 只走 FAQ
#   rag  —— 只走 RAG
#
# 本模块只提供「单条链路的执行」与「预热/能力描述」等无状态能力，
# 请求编排（并发闸门、超时、会话、落库）全部在唯一入口 app_async.py 的 _route 里完成。
import sys
import threading
import time
from pathlib import Path

# ===== 路径处理：只把项目根加入 sys.path，之后一律按包名导入 =====
BASE_DIR = Path(__file__).resolve().parent              # .../integrated_qa_system
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from base import Config, logger

# FAQ 侧与 RAG 侧各自的单例/预热入口
from mysql_qa.utils.qa_service import ensure_ready as faq_ensure_ready, get_qa_system
from rag_qa.rag_service import answer as rag_answer
from rag_qa.rag_service import ensure_ready as rag_ensure_ready

conf = Config()

# 对外暴露的可选模式
MODES = ("auto", "faq", "rag")

# 引擎标识：前端据此显示徽标
ENGINE_FAQ = "faq"
ENGINE_RAG = "rag"
ENGINE_NONE = "none"
# 身份 / 寒暄直答：不经过检索与生成，直接返回模板文案
ENGINE_SMALLTALK = "smalltalk"

# FAQ 未命中时给用户的兜底话术
NO_ANSWER_TEXT = "抱歉，知识库中没有找到可靠的答案，请换个问法试试~"


# ==================== 单条链路的执行 ====================

def _normalize(answer):
    """把答案拍平成字符串并剔除空白；兼容 Redis 里早期存成 list / tuple 的脏数据。"""
    if isinstance(answer, (list, tuple)):
        answer = answer[0] if answer else None
    if answer is None:
        return None
    text = str(answer).strip()
    return text or None


def ask_faq(question):
    """
    走 FAQ 链路。
    :return: (answer, error, need_rag)
             answer 为 None 表示未命中或调用失败；
             need_rag 为 True 表示应当转入 RAG 链路。
    """
    try:
        # 首次调用会在内部惰性初始化（MySQL / Redis / BM25）
        answer, need_rag = get_qa_system().query(question)
        return _normalize(answer), "", bool(need_rag)
    except Exception as e:
        logger.error(f"FAQ 链路异常: {e}")
        logger.exception(e)
        # FAQ 自身故障时同样应转 RAG，否则整条链路直接不可用
        return None, f"FAQ 服务异常: {e}", True


def ask_rag(question, source_filter=None, history=None):
    """
    走 RAG 链路。
    :return: (answer, error)
    """
    answer, error = rag_answer(question, source_filter=source_filter, history=history)
    return _normalize(answer), error


# ==================== 结果结构 ====================

def make_result(answer, engine, elapsed, note=""):
    """统一的结果结构，供 app_async._route 与本模块的链路函数复用。"""
    return {
        "answer": answer,
        "engine": engine,
        "found": bool(answer),
        "elapsed": round(elapsed, 4),
        "note": note or "",
    }


# ==================== 预热 ====================

def warm_up_faq():
    """
    预热 FAQ 侧：加载 jieba 词典、构建 BM25 索引。耗时通常在秒级，可同步执行。
    """
    t0 = time.perf_counter()
    try:
        faq_ensure_ready()
        logger.info(f"FAQ 预热完成，耗时 {time.perf_counter() - t0:.2f} s")
        return True
    except Exception as e:
        logger.error(f"FAQ 预热失败（请求时会重试）: {e}")
        return False


def warm_up_rag(in_background=True):
    """
    预热 RAG 侧：连 Milvus、加载 BGE-M3 / reranker / 意图分类器。
    模型加载较慢（数十秒），默认放后台线程，不阻塞服务启动。
    """
    def _work():
        t0 = time.perf_counter()
        if rag_ensure_ready():
            logger.info(f"RAG 预热完成，耗时 {time.perf_counter() - t0:.2f} s")
        else:
            logger.warning("RAG 预热失败，首次 RAG 请求时会重试")

    if not in_background:
        _work()
        return None

    thread = threading.Thread(target=_work, name="qa-warmup-rag", daemon=True)
    thread.start()
    return thread


def describe():
    """返回服务能力描述，供前端渲染筛选器 / 模式选择。"""
    return {
        "modes": list(MODES),
        "sources": list(conf.VALID_SOURCES),
        "engines": {
            ENGINE_SMALLTALK: "身份与寒暄直答（规则命中，不进入检索链路）",
            ENGINE_FAQ: "FAQ 知识库（BM25 检索 · Redis 缓存）",
            ENGINE_RAG: "RAG 知识库（Milvus 混合检索 · BGE 精排 · 大模型生成）",
        },
    }
