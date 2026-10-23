# utils/chat_store.py
# 会话存储层：使用 SQLite 持久化对话和消息（零额外依赖，无需配置）
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 数据库文件位置：mysql_qa/data/chat_history.db（本文件在 utils/ 下，需要回退一层）
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "chat_history.db"

# SQLite 的写入用全局锁串行化，避免并发写冲突
_LOCK = threading.Lock()

# 线程本地连接：anyio 线程池会复用线程，连接也随之复用，
# 避免每次操作都新建连接（高并发下开销会被放大）
_local = threading.local()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    # journal_mode=WAL 只需在 init_db() 里设置一次（持久化在数据库文件中），
    # 每次连接都执行会带来额外开销
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _conn() -> sqlite3.Connection:
    """取当前线程的连接，没有则创建（线程本地，天然线程安全）。"""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _connect()
        _local.conn = conn
    return conn


@contextmanager
def _db():
    """复用线程本地连接，正常退出提交，异常回滚。"""
    conn = _conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db() -> None:
    """初始化表结构（幂等，可重复调用）。WAL 只需设置一次。"""
    with _LOCK, _db() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                title      TEXT    NOT NULL DEFAULT '新对话',
                created_at TEXT    NOT NULL,
                updated_at TEXT    NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role            TEXT    NOT NULL,
                content         TEXT    NOT NULL,
                meta            TEXT,
                created_at      TEXT    NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id)"
        )


# ==================== 会话 ====================

def create_conversation(title: str = "新对话") -> Dict[str, Any]:
    """新建一个会话，返回会话信息。"""
    title = (title or "新对话").strip() or "新对话"
    now = _now()
    with _LOCK, _db() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
            (title, now, now),
        )
        cid = cur.lastrowid
    return {
        "id": cid,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "message_count": 0,
    }


def list_conversations(limit: int = 100) -> List[Dict[str, Any]]:
    """按最近更新时间倒序列出会话。"""
    with _LOCK, _db() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
            FROM conversations c
            ORDER BY c.updated_at DESC, c.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(cid: int) -> Optional[Dict[str, Any]]:
    """获取单个会话信息，不存在返回 None。"""
    with _LOCK, _db() as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (cid,)
        ).fetchone()
        if row is None:
            return None
        count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (cid,)
        ).fetchone()[0]
    data = dict(row)
    data["message_count"] = count
    return data


def rename_conversation(cid: int, title: str) -> bool:
    """重命名会话，成功返回 True。"""
    title = (title or "").strip()
    if not title:
        return False
    with _LOCK, _db() as conn:
        cur = conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title[:60], _now(), cid),
        )
        return cur.rowcount > 0


def delete_conversation(cid: int) -> bool:
    """删除会话及其所有消息，成功返回 True。"""
    with _LOCK, _db() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (cid,))
        cur = conn.execute("DELETE FROM conversations WHERE id = ?", (cid,))
        return cur.rowcount > 0


def set_title_if_default(cid: int, question: str) -> Optional[str]:
    """
    如果会话仍是默认标题，则用第一个问题自动命名。
    返回最新标题。
    """
    text = " ".join((question or "").split())
    if not text:
        return None
    auto_title = text[:24] + ("…" if len(text) > 24 else "")

    with _LOCK, _db() as conn:
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ? AND (title IS NULL OR title = '' OR title = '新对话')",
            (auto_title, cid),
        )
        row = conn.execute(
            "SELECT title FROM conversations WHERE id = ?", (cid,)
        ).fetchone()
    return row["title"] if row else None


# ==================== 消息 ====================

def add_message(cid: int, role: str, content: str, meta: Optional[str] = None) -> Dict[str, Any]:
    """追加一条消息，并刷新会话更新时间。"""
    now = _now()
    with _LOCK, _db() as conn:
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, meta, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (cid, role, content, meta, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, cid)
        )
        mid = cur.lastrowid
    return {
        "id": mid,
        "conversation_id": cid,
        "role": role,
        "content": content,
        "meta": meta,
        "created_at": now,
    }


def add_exchange(cid: int, question: str, answer: str, meta: Optional[str] = None) -> Optional[str]:
    """
    一个事务写入“一问一答”，同时刷新更新时间和默认标题。
    比调用两次 add_message 少一次提交，高并发下开销更小。返回最新标题。
    """
    now = _now()
    text = " ".join((question or "").split())
    auto_title = text[:24] + ("…" if len(text) > 24 else "")

    with _LOCK, _db() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, meta, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (cid, "user", question, None, now),
        )
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, meta, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (cid, "assistant", answer, meta, now),
        )
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, cid))
        if auto_title:
            conn.execute(
                "UPDATE conversations SET title = ? "
                "WHERE id = ? AND (title IS NULL OR title = '' OR title = '新对话')",
                (auto_title, cid),
            )
        row = conn.execute(
            "SELECT title FROM conversations WHERE id = ?", (cid,)
        ).fetchone()
    return row["title"] if row else None


def list_messages(cid: int) -> List[Dict[str, Any]]:
    """按时间顺序取出某个会话的全部消息。"""
    with _LOCK, _db() as conn:
        rows = conn.execute(
            "SELECT id, conversation_id, role, content, meta, created_at "
            "FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (cid,),
        ).fetchall()
    return [dict(r) for r in rows]


def recent_history(cid: int, turns: int = 5) -> List[Dict[str, str]]:
    """
    取某个会话最近 N 轮的问答对，供 RAG 注入上下文。

    :param cid:   会话 id
    :param turns: 取最近多少轮（一轮 = 一问一答）
    :return: [{"question": "...", "answer": "..."}, ...]，按时间正序
    """
    if turns <= 0:
        return []

    # 只取最后 turns*2 条消息，避免把整个会话读进内存
    with _LOCK, _db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages "
            "WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            (cid, turns * 2),
        ).fetchall()

    pairs: List[Dict[str, str]] = []
    pending_question = None
    # 查询是倒序的，这里反过来还原成时间正序；截断处可能从 assistant 开始，直接跳过
    for row in reversed(rows):
        if row["role"] == "user":
            pending_question = row["content"]
        elif row["role"] == "assistant" and pending_question is not None:
            pairs.append({"question": pending_question, "answer": row["content"]})
            pending_question = None
    return pairs


# 模块导入时即建表
init_db()
