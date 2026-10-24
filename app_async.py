# app_async.py
# 集成问答系统（FastAPI 异步版）
#   事件循环只负责调度 I/O；FAQ（BM25/jieba/MySQL/Redis）与 RAG（Milvus/BGE/LLM）
#   这些阻塞调用全部丢进有并发上限的线程池，高并发时不会因为线程数暴涨导致延迟抖动。
#
# 启动：python app_async.py
#   或：uvicorn app_async:app --host 127.0.0.1 --port 8000
import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ===== 路径处理：只把项目根加入 sys.path，之后一律按包名导入 =====
CURRENT_DIR = Path(__file__).resolve().parent       # .../integrated_qa_system
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from anyio import CapacityLimiter
from anyio.to_thread import current_default_thread_limiter, run_sync
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from base import Config, logger
from mysql_qa.utils import chat_store
from mysql_qa.utils.smalltalk import match_smalltalk
from qa_router import (
    ENGINE_SMALLTALK,
    MODES,
    NO_ANSWER_TEXT,
    ask_faq,
    ask_rag,
    describe,
    make_result,
    warm_up_faq,
    warm_up_rag,
)

# ==================== 并发配置 ====================
CPU = os.cpu_count() or 4

# FAQ 是 CPU 密集任务（jieba 分词 + BM25）：闸门略大于核数即可，
# 再提高并发只会互相抢 GIL，尾延迟反而上升。
FAQ_CONCURRENCY = int(os.getenv("FAQ_CONCURRENCY", str(max(4, CPU))))
# RAG 要跑 BGE-M3 向量化 + bge-reranker 精排 + 大模型调用，单次占用显存与算力都很重，
# 并发给 2 已经足够；再高会让单次延迟成倍上升。
RAG_CONCURRENCY = int(os.getenv("RAG_CONCURRENCY", "2"))
# 单次 FAQ 检索超时（秒）
FAQ_TIMEOUT = float(os.getenv("FAQ_TIMEOUT", "30"))
# 单次 RAG 生成超时（秒）：大模型生成较慢，给足时间
RAG_TIMEOUT = float(os.getenv("RAG_TIMEOUT", "180"))
# 会话落库（MySQL）是毫秒级小事务，连接按线程独占，可以给大一点的并发
DB_CONCURRENCY = int(os.getenv("DB_CONCURRENCY", "32"))

faq_limiter = CapacityLimiter(FAQ_CONCURRENCY)
rag_limiter = CapacityLimiter(RAG_CONCURRENCY)
db_limiter = CapacityLimiter(DB_CONCURRENCY)

# 引擎标识 → 展示名（写入消息 meta，历史对话也据此还原徽标）
ENGINE_LABEL = {"smalltalk": "直答", "faq": "FAQ", "rag": "RAG", "none": "未命中"}


# ==================== 应用 ====================
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动预热：加载 jieba 词典 / 建 BM25 索引 / 连 Milvus / 加载 BGE 模型。"""
    # 线程池总容量与三个闸门匹配（此调用需要在事件循环内执行）
    try:
        current_default_thread_limiter().total_tokens = (
            FAQ_CONCURRENCY + RAG_CONCURRENCY + DB_CONCURRENCY + 8
        )
    except Exception as e:
        logger.warning(f"调整线程池容量失败: {e}")

    # FAQ 侧预热很快（jieba + BM25 索引），在启动阶段同步完成，避免首个请求抖动
    await run_sync(warm_up_faq, limiter=faq_limiter)

    # 过期会话清理：临时会话 1 天、登录用户会话 30 天（config.ini [chat] 段可调）
    conf = Config()
    if conf.CHAT_AUTO_PURGE:
        try:
            stats = await run_sync(
                chat_store.purge_expired,
                conf.CHAT_ANON_RETENTION_DAYS,
                conf.CHAT_USER_RETENTION_DAYS,
                limiter=db_limiter,
            )
            logger.info(f"过期会话清理完成：{stats}")
        except Exception as e:
            logger.warning(f"过期会话清理失败（不影响服务）: {e}")
    # RAG 侧要加载 BGE-M3 / reranker，放后台线程，让服务立即具备响应能力
    warm_up_rag(in_background=True)

    logger.info(
        "异步服务就绪："
        f"FAQ_CONCURRENCY={FAQ_CONCURRENCY}({FAQ_TIMEOUT}s), "
        f"RAG_CONCURRENCY={RAG_CONCURRENCY}({RAG_TIMEOUT}s)"
    )
    yield


app = FastAPI(title="教育知识问答 · FAQ + RAG（异步版）", lifespan=lifespan)

# 静态资源与模板
app.mount("/static", StaticFiles(directory=str(CURRENT_DIR / "static")), name="static")

_env = Environment(
    loader=FileSystemLoader(str(CURRENT_DIR / "templates")),
    autoescape=select_autoescape(["html", "xml"]),
)
# 模板里用的是 Flask 风格 url_for，这里提供等价实现
_env.globals["url_for"] = lambda endpoint, **kw: (
    f"/static/{kw.get('filename', '')}" if endpoint == "static" else "/"
)


# ==================== 工具函数 ====================
async def read_payload(request: Request) -> dict:
    """兼容 JSON / 表单两种请求体。"""
    try:
        data = await request.json()
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    try:
        return dict(await request.form())
    except Exception:
        return {}


def _save_exchange(cid: str, question: str, answer: str, meta: str, fallback_title: str, user_id=None) -> str:
    """保存一问一答（单事务）并返回最新标题（在线程池里执行）。"""
    return chat_store.add_exchange(cid, question, answer, meta=meta, user_id=user_id) or fallback_title


def _create_conversation(user_id=None):
    """run_sync 不支持关键字参数，这里包一层。"""
    return chat_store.create_conversation(user_id=user_id)


def _rag_sync(question: str, source_filter: Optional[str], history):
    """run_sync 不支持关键字参数，这里包一层。"""
    return ask_rag(question, source_filter=source_filter, history=history)


async def _call_faq(question: str) -> Tuple[Optional[str], str, bool]:
    return await run_sync(ask_faq, question, limiter=faq_limiter, abandon_on_cancel=True)


async def _call_rag(question: str, source_filter: Optional[str], history) -> Tuple[Optional[str], str]:
    return await run_sync(
        _rag_sync, question, source_filter, history, limiter=rag_limiter, abandon_on_cancel=True
    )


async def _load_history(conversation_id):
    """取最近几轮对话送给 RAG；读取失败按「无历史」降级，不影响问答。"""
    if not conversation_id:
        return None
    try:
        return await run_sync(chat_store.recent_history, str(conversation_id), limiter=db_limiter)
    except Exception as e:
        logger.warning(f"读取对话历史失败，本次不带历史: {e}")
        return None


# ===== 同一问题的并发请求合并（single-flight）=====
# 并发请求往往集中在少数几个问题上，合并重复计算可以避免算力被同一道题反复占满。
_inflight: Dict[Tuple[str, str, str, int], "asyncio.Future[Any]"] = {}


async def _route(
    question: str, mode: str, source_filter: Optional[str], conversation_id: Optional[int]
) -> dict:
    """
    异步版路由：与 qa_router.answer_query 的判定顺序保持一致，
    区别只是每条链路各自走独立的并发闸门与超时。
    返回值结构同 answer_query。
    """
    start = time.perf_counter()
    note = ""

    # ---------- 0. 身份 / 寒暄直答 ----------
    # 与同步版共用同一份规则，保证两套服务行为一致
    reply = match_smalltalk(question)
    if reply:
        return make_result(reply, ENGINE_SMALLTALK, time.perf_counter() - start,
                           "身份/寒暄直答，未进入检索链路")

    # ---------- 1. FAQ 链路 ----------
    if mode in ("auto", "faq"):
        try:
            answer, faq_error, need_rag = await asyncio.wait_for(_call_faq(question), timeout=FAQ_TIMEOUT)
        except (asyncio.TimeoutError, TimeoutError):
            answer, faq_error, need_rag = None, f"FAQ 检索超时（>{FAQ_TIMEOUT}s）", True
            logger.error(faq_error)

        if answer:
            return make_result(answer, "faq", time.perf_counter() - start, "")
        if mode == "faq":
            return make_result(None, "none", time.perf_counter() - start, faq_error or "FAQ 未命中")
        # auto 模式：need_rag 为 False 只在无效查询时出现，不必为它启动 RAG
        if not need_rag:
            note = faq_error or "FAQ 判定该查询无需检索，未转 RAG"
            logger.info(note)
            return make_result(None, "none", time.perf_counter() - start, note)
        note = f"FAQ 异常（{faq_error}），已转 RAG" if faq_error else "FAQ 无可靠答案，已转 RAG"

    # ---------- 2. RAG 链路 ----------
    if mode in ("auto", "rag"):
        history = await _load_history(conversation_id)
        try:
            answer, rag_error = await asyncio.wait_for(
                _call_rag(question, source_filter, history), timeout=RAG_TIMEOUT
            )
        except (asyncio.TimeoutError, TimeoutError):
            answer, rag_error = None, f"RAG 生成超时（>{RAG_TIMEOUT}s）"
            logger.error(rag_error)

        if answer:
            return make_result(answer, "rag", time.perf_counter() - start, note)
        if rag_error:
            note = f"{note}；{rag_error}" if note else rag_error
        elif not note:
            note = "RAG 未检索到有效内容"

    return make_result(None, "none", time.perf_counter() - start, note)


async def route_with_dedup(
    question: str, mode: str, source_filter: Optional[str], conversation_id: Optional[int] = None
) -> dict:
    """
    相同 (question, mode, source, 会话) 在同一时刻只真正执行一次，其余并发请求共享结果。
    请求被取消不会影响底层计算，计算结果仍会被其他等待者使用。

    会话必须进 key：RAG 会按会话注入对话历史，
    同一个问题在不同会话里的答案可能不同，跨会话复用会串味。
    """
    key = (question, mode, source_filter or "", conversation_id or "")
    loop = asyncio.get_running_loop()
    fut = _inflight.get(key)

    if fut is None:
        fut = loop.create_future()
        _inflight[key] = fut

        task = loop.create_task(_route(question, mode, source_filter, conversation_id))

        def _on_done(t: "asyncio.Task[Any]") -> None:
            _inflight.pop(key, None)
            if fut.done():
                return
            if t.cancelled():
                fut.cancel()
                return
            exc = t.exception()
            if exc is not None:
                fut.set_exception(exc)
                # 没有任何请求在等待时，避免 "exception was never retrieved"
                fut.add_done_callback(lambda f: f.cancelled() or f.exception())
            else:
                fut.set_result(t.result())

        task.add_done_callback(_on_done)

    # 总超时 = 两条链路各自的超时上限之和，再留一点余量；真正的单链路超时在上层已处理
    return await asyncio.wait_for(asyncio.shield(fut), timeout=FAQ_TIMEOUT + RAG_TIMEOUT + 10)


# ==================== 页面 ====================
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_env.get_template("index.html").render())


# ==================== 问答接口 ====================
@app.post("/query")
async def query(request: Request):
    """
    POST /query
    请求体：{"query": "...", "session_id": "uuid", "mode": "auto", "source": "ai", "user_id": 1}
        - session_id 可选（36 位 UUID，不传则新建会话）
        - mode 可选：auto（默认）/ faq / rag
        - source 可选：学科过滤，仅对 RAG 生效
        - user_id 可选：仅登录后传，用于把会话与历史记录归到账号名下
    """
    start = time.perf_counter()
    data = await read_payload(request)

    question = str(data.get("query") or "").strip()
    if not question:
        return JSONResponse({
            "code": 400,
            "message": "query 不能为空",
            "query": "",
            "answer": None,
            "found": False,
            "engine": "none",
            "elapsed": 0,
        }, status_code=400)

    mode = str(data.get("mode") or "auto").strip().lower()
    if mode not in MODES:
        mode = "auto"
    source = str(data.get("source") or "").strip() or None

    # user_id 只在登录后由前端带上；匿名访问为 None
    user_id = data.get("user_id")
    try:
        user_id = int(user_id) if user_id not in (None, "", "null") else None
    except (TypeError, ValueError):
        user_id = None

    # ---- 会话处理（阻塞的 MySQL 调用也放线程池）----
    session_id = data.get("session_id")
    cid = None
    if session_id not in (None, "", "null"):
        cid = str(session_id).strip() or None

    conversation = (
        await run_sync(chat_store.get_conversation, cid, limiter=db_limiter)
        if cid is not None
        else None
    )
    if conversation is None:
        conversation = await run_sync(_create_conversation, user_id, limiter=db_limiter)
        logger.info(f"新建会话 id={conversation['id']} user_id={user_id}")
    cid = conversation["id"]

    # ---- 路由：FAQ 优先，未命中转 RAG ----
    try:
        result = await route_with_dedup(question, mode, source, cid)
    except Exception as e:
        logger.exception(e)
        return JSONResponse({
            "code": 500,
            "message": f"服务内部错误: {e}",
            "session_id": cid,
            "query": question,
            "answer": None,
            "found": False,
            "engine": "none",
            "elapsed": round(time.perf_counter() - start, 4),
        }, status_code=500)

    found = result["found"]
    final_answer = result["answer"] if found else NO_ANSWER_TEXT
    engine = result["engine"]
    elapsed = result["elapsed"]

    label = ENGINE_LABEL.get(engine, "未命中")
    meta_text = f"{label} · 耗时 {elapsed} s" if found else f"未命中 · 耗时 {elapsed} s"

    # ---- 落库 ----
    try:
        title = await run_sync(
            _save_exchange,
            cid,
            question,
            final_answer,
            meta_text,
            conversation["title"],
            user_id,
            limiter=db_limiter,
        )
    except Exception as e:
        logger.error(f"保存对话记录失败: {e}")
        title = conversation["title"]

    return JSONResponse({
        "code": 0,
        "message": "ok",
        "session_id": cid,
        "title": title,
        "query": question,
        "answer": final_answer,
        "found": found,
        "engine": engine,
        "elapsed": elapsed,
        "note": result["note"],
    })


# ==================== 能力描述 ====================
@app.get("/api/meta")
async def api_meta():
    """前端初始化时拉取：可选模式、学科列表、引擎说明。"""
    return {"code": 0, **describe()}


# ==================== 会话管理 ====================
@app.get("/api/conversations")
async def api_list_conversations(user_id: Optional[int] = None):
    """会话列表；带 user_id 时只返回该账号名下的会话。"""
    conversations = await run_sync(
        chat_store.list_conversations, 100, user_id, limiter=db_limiter
    )
    return {"code": 0, "conversations": conversations}


@app.post("/api/conversations")
async def api_create_conversation(request: Request):
    data = await read_payload(request)
    user_id = data.get("user_id")
    try:
        user_id = int(user_id) if user_id not in (None, "", "null") else None
    except (TypeError, ValueError):
        user_id = None
    conv = await run_sync(
        chat_store.create_conversation,
        data.get("title") or "新对话",
        user_id,
        limiter=db_limiter,
    )
    return JSONResponse({"code": 0, "conversation": conv}, status_code=201)


@app.get("/api/conversations/{cid}")
async def api_get_conversation(cid: str):
    conv = await run_sync(chat_store.get_conversation, cid, limiter=db_limiter)
    if conv is None:
        return JSONResponse({"code": 404, "message": "会话不存在"}, status_code=404)
    messages = await run_sync(chat_store.list_messages, cid, limiter=db_limiter)
    return {"code": 0, "conversation": conv, "messages": messages}


@app.patch("/api/conversations/{cid}")
async def api_rename_conversation(cid: str, request: Request):
    data = await read_payload(request)
    title = str(data.get("title") or "").strip()
    if not title:
        return JSONResponse({"code": 400, "message": "title 不能为空"}, status_code=400)
    ok = await run_sync(chat_store.rename_conversation, cid, title, limiter=db_limiter)
    if not ok:
        return JSONResponse({"code": 404, "message": "会话不存在"}, status_code=404)
    conv = await run_sync(chat_store.get_conversation, cid, limiter=db_limiter)
    return {"code": 0, "conversation": conv}


@app.delete("/api/conversations/{cid}")
async def api_delete_conversation(cid: str):
    ok = await run_sync(chat_store.delete_conversation, cid, limiter=db_limiter)
    if not ok:
        return JSONResponse({"code": 404, "message": "会话不存在"}, status_code=404)
    logger.info(f"删除会话 id={cid}")
    return {"code": 0, "message": "deleted", "id": cid}


@app.get("/health")
async def health():
    """健康检查：可直接观察 FAQ / RAG 两条链路的并发占用情况。"""
    return {
        "code": 0,
        "status": "running",
        "faq_concurrency": FAQ_CONCURRENCY,
        "faq_inflight": faq_limiter.borrowed_tokens,
        "faq_waiting": faq_limiter.statistics().tasks_waiting,
        "rag_concurrency": RAG_CONCURRENCY,
        "rag_inflight": rag_limiter.borrowed_tokens,
        "rag_waiting": rag_limiter.statistics().tasks_waiting,
        "db_inflight": db_limiter.borrowed_tokens,
    }


# ==================== 兜底异常 ====================
@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    logger.exception(exc)
    return JSONResponse(
        {"code": 500, "message": f"服务内部错误: {exc}"},
        status_code=500,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.getenv("PORT", "8000")),
        log_level="info",
        # 单进程多线程即可（BM25 / 向量库 / BGE 模型在进程内共享）；
        # 会话落库已改为 MySQL，如需多进程直接调大 workers 即可，无需再做文件级兼容
        workers=1,
    )
