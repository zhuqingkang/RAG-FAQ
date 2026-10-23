# qa_router.py
# 统一问答路由：把 mysql_qa（MySQL + BM25 + Redis 的 FAQ 检索）与
# rag_qa（Milvus + BGE 精排 + 大模型生成的 RAG）合成一个对外的问答能力。
#
# 路由策略（mode）：
#   auto —— 先查 FAQ，命中即返回；未命中或 FAQ 异常时转 RAG（默认）
#   faq  —— 只走 FAQ
#   rag  —— 只走 RAG
#
# 之所以把路由逻辑放在这里，而不是写进 app.py / app_async.py：
# 同步（Flask）与异步（FastAPI）两套服务共用同一份判定逻辑，避免两边行为漂移。
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
from mysql_qa.utils import chat_store
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


def _load_history(conversation_id, turns=5):
    """
    从会话存储里取最近 N 轮问答，供 RAG 注入上下文。
    只在真正要走 RAG 时调用，FAQ 命中路径不产生这次查询。
    读取失败不阻断问答，按「无历史」降级。
    """
    if not conversation_id:
        return None
    try:
        return chat_store.recent_history(int(conversation_id), turns=turns)
    except Exception as e:
        logger.warning(f"读取对话历史失败，本次不带历史: {e}")
        return None


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


# ==================== 统一入口 ====================

def answer_query(question, mode="auto", source_filter=None, conversation_id=None):
    """
    统一问答入口（同步、阻塞）。异步服务请丢线程池调用本函数。

    :param question:        用户问题（调用方需保证非空）
    :param mode:            auto / faq / rag
    :param source_filter:   学科过滤，仅对 RAG 生效；None 表示不过滤
    :param conversation_id: 会话 id，用于给 RAG 取最近几轮对话历史；None 表示不带历史
    :return: dict ——
        {
          "answer":   str | None,      # None 表示没有答案
          "engine":   "faq"|"rag"|"none",  # 实际回答的引擎
          "found":    bool,
          "elapsed":  float,           # 秒，四位小数
          "note":     str,             # 过程说明（例如 FAQ 未命中已转 RAG）
        }
    """
    start = time.perf_counter()

    mode = (mode or "auto").strip().lower()
    if mode not in MODES:
        logger.warning(f"未知 mode={mode!r}，回退为 auto")
        mode = "auto"

    answer = None
    note = ""

    # ---------- 1. FAQ 链路 ----------
    if mode in ("auto", "faq"):
        answer, faq_error, need_rag = ask_faq(question)
        if answer:
            return make_result(answer, ENGINE_FAQ, time.perf_counter() - start, note)
        if mode == "faq":
            # 只走 FAQ：没命中就直接返回兜底话术
            note = faq_error or "FAQ 未命中"
            return make_result(answer, ENGINE_NONE, time.perf_counter() - start, note)
        # auto 模式：need_rag 是 FAQ 给出的调度建议。
        # False 只在「查询本身无效」时出现，此时不要为它启动昂贵的 RAG 链路。
        if not need_rag:
            note = faq_error or "FAQ 判定该查询无需检索，未转 RAG"
            logger.info(note)
            return make_result(None, ENGINE_NONE, time.perf_counter() - start, note)
        # 记录原因，继续转 RAG
        if faq_error:
            note = f"FAQ 异常（{faq_error}），已转 RAG"
        else:
            note = "FAQ 无可靠答案，已转 RAG"
        logger.info(note)

    # ---------- 2. RAG 链路 ----------
    if mode in ("auto", "rag"):
        history = _load_history(conversation_id)
        answer, rag_error = ask_rag(question, source_filter=source_filter, history=history)
        if answer:
            return make_result(answer, ENGINE_RAG, time.perf_counter() - start, note)
        if rag_error:
            note = f"{note}；{rag_error}" if note else rag_error
        elif not note:
            note = "RAG 未检索到有效内容"
        logger.info(f"RAG 未返回有效答案: {note}")

    return make_result(None, ENGINE_NONE, time.perf_counter() - start, note)


def make_result(answer, engine, elapsed, note=""):
    """统一的结果结构：同步版与异步版共用。"""
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
            ENGINE_FAQ: "FAQ 知识库（BM25 检索 · Redis 缓存）",
            ENGINE_RAG: "RAG 知识库（Milvus 混合检索 · BGE 精排 · 大模型生成）",
        },
    }
