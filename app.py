# app.py
# 集成问答系统（Flask 同步版）
#   - FAQ 链路：mysql_qa（MySQL + BM25 + Redis）
#   - RAG 链路：rag_qa（Milvus + BGE 精排 + 大模型生成）
#   - 统一入口：POST /query，由 qa_router 决定先 FAQ 还是转 RAG
#
# 启动：python app.py   →  http://127.0.0.1:5000
import sys
import time
from pathlib import Path

# ===== 路径处理：无论从项目根目录还是本目录启动，都能正确导入模块 =====
CURRENT_DIR = Path(__file__).resolve().parent       # .../integrated_qa_system
MYSQL_QA_DIR = CURRENT_DIR / "mysql_qa"
for _p in (str(CURRENT_DIR), str(MYSQL_QA_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flask import Flask, jsonify, render_template, request

from base import logger
from utils import chat_store
from qa_router import NO_ANSWER_TEXT, answer_query, describe, warm_up_faq, warm_up_rag

app = Flask(
    __name__,
    template_folder=str(CURRENT_DIR / "templates"),
    static_folder=str(CURRENT_DIR / "static"),
)

# 让 JSON 响应支持中文（不转义成 \uXXXX）
try:
    setattr(app.json, "ensure_ascii", False)   # Flask >= 2.3
except Exception:                              # 兼容老版本
    app.config["JSON_AS_ASCII"] = False

# 引擎标识 → 展示名（写入消息 meta，历史对话也据此还原徽标）
ENGINE_LABEL = {"faq": "FAQ", "rag": "RAG", "none": "未命中"}


def _resolve_conversation(session_id):
    """按 session_id 取会话，不存在则新建。返回会话 dict。"""
    conversation = None
    if session_id not in (None, "", "null"):
        try:
            conversation = chat_store.get_conversation(int(session_id))
        except (TypeError, ValueError):
            conversation = None
    if conversation is None:
        conversation = chat_store.create_conversation()
        logger.info(f"新建会话 id={conversation['id']}")
    return conversation


# ================= 页面 =================
@app.route("/", methods=["GET"])
def index():
    """唯一的对外页面（FAQ 与 RAG 共用）。"""
    return render_template("index.html")


# ================= 接口 =================
@app.route("/query", methods=["POST"])
def query():
    """
    POST /query
    请求体（JSON）：{"query": "问题", "session_id": 1, "mode": "auto", "source": "ai"}
        - session_id 可选：不传或不存在时自动新建会话
        - mode 可选：auto（默认）/ faq / rag
        - source 可选：学科过滤，仅对 RAG 生效；不传表示不过滤
    返回：{"code": 0, "session_id": 1, "title": "...", "query": "...",
           "answer": "...", "found": true, "engine": "faq"|"rag"|"none",
           "elapsed": 0.0123, "note": "..."}
    """
    start = time.time()

    # 兼容 JSON 和表单两种提交方式
    data = request.get_json(silent=True) or {}
    if not data:
        data = request.form.to_dict()

    question = (data.get("query") or "").strip()
    if not question:
        logger.warning("收到空查询请求")
        return jsonify({
            "code": 400,
            "message": "query 不能为空",
            "query": "",
            "answer": None,
            "found": False,
            "elapsed": 0,
        }), 400

    mode = (data.get("mode") or "auto").strip().lower()
    source = (data.get("source") or "").strip() or None

    conversation = _resolve_conversation(data.get("session_id"))
    cid = conversation["id"]

    try:
        result = answer_query(question, mode=mode, source_filter=source)
    except Exception as e:
        logger.error(f"处理查询失败: {e}")
        logger.exception(e)
        return jsonify({
            "code": 500,
            "message": f"服务内部错误: {e}",
            "session_id": cid,
            "query": question,
            "answer": None,
            "found": False,
            "engine": "none",
            "elapsed": round(time.time() - start, 4),
        }), 500

    found = result["found"]
    final_answer = result["answer"] if found else NO_ANSWER_TEXT
    engine = result["engine"]
    elapsed = result["elapsed"]

    label = ENGINE_LABEL.get(engine, "未命中")
    meta_text = f"{label} · 耗时 {elapsed} s" if found else f"未命中 · 耗时 {elapsed} s"

    # ---- 落库：用户问题 + 助手回答（单事务） ----
    try:
        title = chat_store.add_exchange(cid, question, final_answer, meta=meta_text) or conversation["title"]
    except Exception as e:
        # 存储失败不影响问答结果
        logger.error(f"保存对话记录失败: {e}")
        title = conversation["title"]

    return jsonify({
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


# ================= 能力描述 =================
@app.route("/api/meta", methods=["GET"])
def api_meta():
    """前端初始化时拉取：可选模式、学科列表、引擎说明。"""
    return jsonify({"code": 0, **describe()})


# ================= 会话管理 =================
@app.route("/api/conversations", methods=["GET"])
def api_list_conversations():
    """获取会话列表（按最近更新排序）。"""
    return jsonify({"code": 0, "conversations": chat_store.list_conversations()})


@app.route("/api/conversations", methods=["POST"])
def api_create_conversation():
    """新建会话，请求体可选：{"title": "自定义标题"}"""
    data = request.get_json(silent=True) or {}
    conv = chat_store.create_conversation(data.get("title") or "新对话")
    logger.info(f"新建会话 id={conv['id']}")
    return jsonify({"code": 0, "conversation": conv}), 201


@app.route("/api/conversations/<int:cid>", methods=["GET"])
def api_get_conversation(cid: int):
    """获取单个会话及全部消息。"""
    conv = chat_store.get_conversation(cid)
    if conv is None:
        return jsonify({"code": 404, "message": "会话不存在"}), 404
    return jsonify({
        "code": 0,
        "conversation": conv,
        "messages": chat_store.list_messages(cid),
    })


@app.route("/api/conversations/<int:cid>", methods=["PATCH"])
def api_rename_conversation(cid: int):
    """重命名会话，请求体：{"title": "新标题"}"""
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"code": 400, "message": "title 不能为空"}), 400
    if not chat_store.rename_conversation(cid, title):
        return jsonify({"code": 404, "message": "会话不存在"}), 404
    return jsonify({"code": 0, "conversation": chat_store.get_conversation(cid)})


@app.route("/api/conversations/<int:cid>", methods=["DELETE"])
def api_delete_conversation(cid: int):
    """删除会话及其消息。"""
    if not chat_store.delete_conversation(cid):
        return jsonify({"code": 404, "message": "会话不存在"}), 404
    logger.info(f"删除会话 id={cid}")
    return jsonify({"code": 0, "message": "deleted", "id": cid})


@app.route("/health", methods=["GET"])
def health():
    """健康检查接口。"""
    return jsonify({"code": 0, "message": "ok", "status": "running"})


if __name__ == "__main__":
    logger.info("集成问答服务启动: http://127.0.0.1:5000")
    # 预热：FAQ 同步完成（秒级，避免首个请求抖动）；RAG 模型加载较慢，放后台线程
    warm_up_faq()
    warm_up_rag(in_background=True)
    # 关闭 reloader，避免重复初始化 BM25 / 向量库 / 后台线程
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
