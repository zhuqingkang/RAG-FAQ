# mysql_qa/utils/chat_store.py
# 会话存储层（MySQL 版）：把「一问一答」按行落库，替代原 SQLite 实现。
#
# 表结构（两张）：
#   conversations —— 对话内容表，一行 = 一轮问答
#       id / session_id / question / answer / timestamp / user_id / meta
#   chat_sessions —— 会话元信息表，一行 = 一个会话
#       session_id / title / user_id / created_at / updated_at
#
# 为什么要拆两张表：
#   1) conversations 保持「一问一答一行」的扁平结构，便于直接 SELECT 观察与统计；
#   2) 会话标题、归属用户属于「会话级」属性，重复写在每一行会带来更新异常，
#      单独放 chat_sessions 才能支持重命名、按用户查看历史。
#
# 关于会话标识：
#   session_id 是 36 位 UUID，未登录用户也能用（临时 id），此时 user_id 为 NULL；
#   用户注册登录后调用 bind_user() 把该会话挂到 user_id 上，历史记录即被保留下来。
import sys
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# 保证「被包导入」和「直接运行本文件自测」都能找到 base 包
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pymysql
from pymysql.cursors import DictCursor

# 配置与日志复用项目统一入口（与 mysql_qa/db/MySQLClient.py 保持同一套连接信息）
from base import Config, logger

# 会话标识长度：标准 UUID 带连字符正好 36 位，与 varchar(36) 对应
SESSION_ID_LEN = 36

# 线程本地连接：pymysql 的连接不是线程安全的，每个线程独占一个连接最稳。
# FastAPI 的线程池会复用线程，连接也随之复用，不需要每次重建。
_local = threading.local()

# 建表只做一次（多线程同时进来时用锁兜住，避免重复 DDL）
_init_lock = threading.Lock()
_init_done = False


# ==================== 连接管理 ====================

def _now() -> datetime:
    """统一取当前时间（截断到秒，与 DATETIME 精度一致）。"""
    return datetime.now().replace(microsecond=0)


def _fmt(value) -> str:
    """把 DATETIME 转成前端可直接展示的字符串。"""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value) if value is not None else ""


def _connect() -> pymysql.Connection:
    """按 config.ini 的 [mysql] 段建立连接。"""
    conf = Config()
    return pymysql.connect(
        host=conf.MYSQL_HOST,
        user=conf.MYSQL_USER,
        password=conf.MYSQL_PASSWORD,
        database=conf.MYSQL_DATABASE,
        port=3306,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=False,
        connect_timeout=5,
    )


def _conn() -> pymysql.Connection:
    """取当前线程的连接，没有或已断开则重建。"""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _connect()
        _local.conn = conn
        return conn
    # 空闲连接可能被 MySQL 的 wait_timeout 断开，ping 一次按需重连
    try:
        conn.ping(reconnect=True)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        conn = _connect()
        _local.conn = conn
    return conn


@contextmanager
def _db():
    """正常退出提交，异常回滚。"""
    conn = _conn()
    _ensure_tables(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _ensure_tables(conn) -> None:
    """建表（幂等）。首次真正访问数据库时才执行，避免模块导入即依赖 MySQL。"""
    global _init_done
    if _init_done:
        return
    with _init_lock:
        if _init_done:
            return
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id         INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL COMMENT '会话标识：前端临时 id（UUID）',
                    question   TEXT NOT NULL COMMENT '用户提问',
                    answer     MEDIUMTEXT COMMENT '问答结果',
                    `timestamp` DATETIME NOT NULL COMMENT '写入时间',
                    user_id    INT NULL COMMENT '登录用户 id；匿名会话为 NULL',
                    meta       VARCHAR(255) NULL COMMENT '引擎/耗时等展示信息',
                    KEY idx_conv_session (session_id, id),
                    KEY idx_conv_user (user_id, `timestamp`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                  COMMENT='对话记录：一行一轮问答'
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id VARCHAR(36) NOT NULL PRIMARY KEY,
                    title      VARCHAR(80) NOT NULL DEFAULT '新对话',
                    user_id    INT NULL COMMENT '会话归属用户；未登录为 NULL',
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    KEY idx_sess_updated (updated_at),
                    KEY idx_sess_user (user_id, updated_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                  COMMENT='会话元信息：标题与归属用户'
                """
            )
            _ensure_columns(conn, cur, "conversations", _EXTRA_COLUMNS)
            try:
                _backfill_sessions(cur)
            except Exception as e:
                # 回填失败只影响历史演示数据的可见性，不该阻断服务
                logger.warning(f"会话元信息回填失败（可忽略）: {e}")
        conn.commit()
        _init_done = True


def _backfill_sessions(cur) -> None:
    """
    兼容「先用演示 SQL 直接灌过 conversations」的场景：
    为没有元信息的会话补一条 chat_sessions 记录，否则侧边栏列表里看不到它们。

    INSERT IGNORE 保证幂等；已有记录不会被覆盖（标题、用户归属以 chat_sessions 为准）。
    """
    cur.execute(
        """
        INSERT IGNORE INTO chat_sessions (session_id, title, user_id, created_at, updated_at)
        SELECT c.session_id,
               COALESCE(LEFT((SELECT c2.question FROM conversations c2
                               WHERE c2.session_id = c.session_id
                               ORDER BY c2.id LIMIT 1), 24), '新对话') AS title,
               MAX(c.user_id) AS user_id,
               MIN(c.`timestamp`) AS created_at,
               MAX(c.`timestamp`) AS updated_at
        FROM conversations c
        GROUP BY c.session_id
        """
    )
    if cur.rowcount:
        logger.info(f"已从 conversations 回填 {cur.rowcount} 个会话的元信息")


# 若 conversations 表是由外部脚本（演示 SQL）先建好的，可能缺以下列。
# 这里只做「缺列补齐」，不改已有列的类型，也不动任何数据。
_EXTRA_COLUMNS = {
    "user_id": (
        "ALTER TABLE conversations ADD COLUMN user_id INT NULL "
        "COMMENT '登录用户 id；匿名会话为 NULL' AFTER `timestamp`"
    ),
    "meta": (
        "ALTER TABLE conversations ADD COLUMN meta VARCHAR(255) NULL "
        "COMMENT '引擎/耗时等展示信息' AFTER user_id"
    ),
}


def _ensure_columns(conn, cur, table: str, columns: Dict[str, str]) -> None:
    """补齐缺失列（幂等）。顺序按字典插入顺序，保证 AFTER 依赖成立。"""
    cur.execute(
        "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s",
        (table,),
    )
    existing = {row["COLUMN_NAME"] for row in cur.fetchall()}

    for name, ddl in columns.items():
        if name in existing:
            continue
        try:
            cur.execute(ddl)
            logger.info(f"会话表 {table} 缺少列 {name}，已自动补齐")
        except Exception as e:
            # 补列失败不阻断：后续写入会明确报错，便于定位
            logger.warning(f"为 {table} 补列 {name} 失败: {e}")


def init_db() -> None:
    """显式初始化表结构（服务启动时可调用，幂等）。"""
    with _db():
        pass
    logger.info("会话存储表检查完成（conversations / chat_sessions）")


def _new_session_id() -> str:
    """生成 36 位会话标识（UUID4，未登录用户的临时 id）。"""
    return str(uuid.uuid4())


def _normalize_sid(sid) -> str:
    """统一会话标识为字符串；非法值直接报错，避免脏查询。"""
    text = str(sid).strip() if sid is not None else ""
    if not text:
        raise ValueError("session_id 不能为空")
    return text[:SESSION_ID_LEN]


def _auto_title(question: str) -> str:
    """用首个问题自动生成会话标题。"""
    text = " ".join((question or "").split())
    return text[:24] + ("…" if len(text) > 24 else "")


def _session_row(row: Dict[str, Any], turns: int = 0) -> Dict[str, Any]:
    """把 chat_sessions 行转成前端/调用方需要的结构。

    id 与 session_id 都返回会话标识：前端一直用 conv.id 拼接口路径，
    这样只需改类型（int → str），不必改动前端逻辑。
    """
    sid = row["session_id"]
    return {
        "id": sid,
        "session_id": sid,
        "title": row.get("title") or "新对话",
        "user_id": row.get("user_id"),
        "created_at": _fmt(row.get("created_at")),
        "updated_at": _fmt(row.get("updated_at")),
        # 一轮问答 = 两条消息，沿用原前端的「N 条」口径
        "message_count": int(turns) * 2,
    }


# ==================== 会话 ====================

def create_conversation(title: str = "新对话", user_id: Optional[int] = None) -> Dict[str, Any]:
    """
    新建一个会话（只在 chat_sessions 里占位，首轮问答时才写入 conversations）。

    :param title:   会话标题，默认「新对话」
    :param user_id: 登录用户 id；不传表示匿名会话
    """
    title = (title or "新对话").strip() or "新对话"
    sid = _new_session_id()
    now = _now()

    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_sessions (session_id, title, user_id, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (sid, title[:80], user_id, now, now),
            )
    return {
        "id": sid,
        "session_id": sid,
        "title": title[:80],
        "user_id": user_id,
        "created_at": _fmt(now),
        "updated_at": _fmt(now),
        "message_count": 0,
    }


def list_conversations(limit: int = 100, user_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    按最近更新时间倒序列出会话。

    :param limit:   最多返回多少条
    :param user_id: 传了则只返回该用户的会话；None 表示不过滤（匿名 + 已登录）
    """
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.session_id, s.title, s.user_id, s.created_at, s.updated_at,
                       (SELECT COUNT(*) FROM conversations c
                         WHERE c.session_id = s.session_id) AS turns
                FROM chat_sessions s
                WHERE (%s IS NULL OR s.user_id = %s)
                ORDER BY s.updated_at DESC, s.created_at DESC
                LIMIT %s
                """,
                (user_id, user_id, int(limit)),
            )
            rows = cur.fetchall()
    return [_session_row(r, r["turns"]) for r in rows]


def get_conversation(sid) -> Optional[Dict[str, Any]]:
    """获取单个会话信息，不存在返回 None。"""
    sid = _normalize_sid(sid)
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT session_id, title, user_id, created_at, updated_at "
                "FROM chat_sessions WHERE session_id = %s",
                (sid,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            cur.execute(
                "SELECT COUNT(*) AS turns FROM conversations WHERE session_id = %s",
                (sid,),
            )
            turns = cur.fetchone()["turns"]
    return _session_row(row, turns)


def rename_conversation(sid, title: str) -> bool:
    """重命名会话，成功返回 True。"""
    sid = _normalize_sid(sid)
    title = (title or "").strip()
    if not title:
        return False
    with _db() as conn:
        with conn.cursor() as cur:
            rowcount = cur.execute(
                "UPDATE chat_sessions SET title = %s, updated_at = %s WHERE session_id = %s",
                (title[:80], _now(), sid),
            )
    return rowcount > 0


def delete_conversation(sid) -> bool:
    """删除会话及其全部问答记录，成功返回 True。"""
    sid = _normalize_sid(sid)
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM conversations WHERE session_id = %s", (sid,))
            deleted_rows = cur.rowcount
            cur.execute("DELETE FROM chat_sessions WHERE session_id = %s", (sid,))
            deleted_sessions = cur.rowcount
    return deleted_sessions > 0 or deleted_rows > 0


def bind_user(sid, user_id: int) -> bool:
    """
    登录后把匿名会话绑定到用户，历史记录随之保留。

    只回填 user_id 为空的行，避免把已有归属的会话改挂到别人名下。
    """
    sid = _normalize_sid(sid)
    if not user_id:
        return False
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET user_id = %s "
                "WHERE session_id = %s AND (user_id IS NULL OR user_id = %s)",
                (user_id, sid, user_id),
            )
            affected = cur.rowcount
            cur.execute(
                "UPDATE conversations SET user_id = %s "
                "WHERE session_id = %s AND user_id IS NULL",
                (user_id, sid),
            )
    return affected > 0


def purge_expired(anonymous_days: int = 1, user_days: int = 30, dry_run: bool = False) -> Dict[str, int]:
    """
    清理过期会话：匿名（临时）会话保留 anonymous_days 天，登录用户会话保留 user_days 天。

    判定依据是会话的最后更新时间 chat_sessions.updated_at，
    以「会话」为单位整体删除，避免出现只剩半段问答的脏数据。

    :param dry_run: True 时只统计不删除，用于上线前预览
    :return: {"sessions": 删除的会话数, "turns": 删除的问答轮数}
    """
    anon_cutoff = _now() - timedelta(days=max(int(anonymous_days), 0))
    user_cutoff = _now() - timedelta(days=max(int(user_days), 0))

    # 匿名会话按短周期、登录会话按长周期，两个条件合成一次删除
    session_cond = (
        "(user_id IS NULL AND updated_at < %s) "
        "OR (user_id IS NOT NULL AND updated_at < %s)"
    )
    join_cond = (
        "(s.user_id IS NULL AND s.updated_at < %s) "
        "OR (s.user_id IS NOT NULL AND s.updated_at < %s)"
    )
    params = (anon_cutoff, user_cutoff)

    with _db() as conn:
        with conn.cursor() as cur:
            if dry_run:
                cur.execute(f"SELECT COUNT(*) AS n FROM chat_sessions WHERE {session_cond}", params)
                sessions = cur.fetchone()["n"]
                cur.execute(
                    "SELECT COUNT(*) AS n FROM conversations c "
                    f"JOIN chat_sessions s ON s.session_id = c.session_id WHERE {join_cond}",
                    params,
                )
                turns = cur.fetchone()["n"]
                return {"sessions": int(sessions), "turns": int(turns)}

            cur.execute(
                "DELETE c FROM conversations c "
                f"JOIN chat_sessions s ON s.session_id = c.session_id WHERE {join_cond}",
                params,
            )
            turns = cur.rowcount
            cur.execute(f"DELETE FROM chat_sessions WHERE {session_cond}", params)
            sessions = cur.rowcount
    return {"sessions": int(sessions), "turns": int(turns)}


def set_title_if_default(sid, question: str) -> Optional[str]:
    """会话仍为默认标题时，用第一个问题自动命名。返回最新标题。"""
    sid = _normalize_sid(sid)
    auto_title = _auto_title(question)
    if not auto_title:
        return None

    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_sessions SET title = %s "
                "WHERE session_id = %s AND (title IS NULL OR title = '' OR title = '新对话')",
                (auto_title, sid),
            )
            cur.execute("SELECT title FROM chat_sessions WHERE session_id = %s", (sid,))
            row = cur.fetchone()
    return row["title"] if row else None


# ==================== 对话内容 ====================

def add_exchange(
    sid,
    question: str,
    answer: str,
    meta: Optional[str] = None,
    user_id: Optional[int] = None,
) -> Optional[str]:
    """
    写入一轮「一问一答」（单事务），同时刷新会话时间与默认标题。

    :param sid:     会话标识
    :param meta:    展示用元信息，如「FAQ · 耗时 0.12 s」
    :param user_id: 登录用户 id，匿名传 None
    :return: 最新会话标题
    """
    sid = _normalize_sid(sid)
    now = _now()
    auto_title = _auto_title(question)

    with _db() as conn:
        with conn.cursor() as cur:
            # 容错：前端可能带着一个还没有元信息的 session_id 直接提问
            cur.execute(
                "INSERT IGNORE INTO chat_sessions (session_id, title, user_id, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (sid, auto_title or "新对话", user_id, now, now),
            )
            cur.execute(
                "INSERT INTO conversations "
                "(session_id, question, answer, `timestamp`, user_id, meta) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (sid, question, answer, now, user_id, meta),
            )
            cur.execute(
                "UPDATE chat_sessions "
                "SET updated_at = %s, "
                "    user_id = COALESCE(user_id, %s), "
                "    title = CASE WHEN title IS NULL OR title = '' OR title = '新对话' "
                "                 THEN %s ELSE title END "
                "WHERE session_id = %s",
                (now, user_id, auto_title or "新对话", sid),
            )
            cur.execute("SELECT title FROM chat_sessions WHERE session_id = %s", (sid,))
            row = cur.fetchone()
    return row["title"] if row else None


def import_conversation(
    sid,
    title: str,
    exchanges: List[tuple],
    user_id: Optional[int] = None,
    created_at=None,
    updated_at=None,
) -> int:
    """
    导入一个已存在的会话（供 SQLite 历史迁移 / 测试数据回放使用）。

    与 add_exchange 的区别：可指定原始时间戳，且一次事务写入整段会话；
    会话已存在时直接跳过，保证脚本可以重复执行。

    :param exchanges: [(question, answer, timestamp, meta), ...]
    :return: 实际写入的问答行数；会话已存在返回 0
    """
    sid = _normalize_sid(sid)
    if not exchanges:
        return 0

    times = [item[2] for item in exchanges if item[2]]
    created = created_at or (min(times) if times else _now())
    updated = updated_at or (max(times) if times else created)

    with _db() as conn:
        with conn.cursor() as cur:
            inserted = cur.execute(
                "INSERT IGNORE INTO chat_sessions "
                "(session_id, title, user_id, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (sid, (title or "新对话")[:80], user_id, created, updated),
            )
            if inserted == 0:
                # 会话已存在：视为已迁移过，跳过
                return 0
            rows = [
                (sid, q, a, t or created, user_id, meta)
                for q, a, t, meta in exchanges
            ]
            cur.executemany(
                "INSERT INTO conversations "
                "(session_id, question, answer, `timestamp`, user_id, meta) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                rows,
            )
    return len(rows)


def list_messages(sid) -> List[Dict[str, Any]]:
    """
    按时间顺序取出某个会话的全部消息。

    库里一轮问答只占一行，这里拆成 user / assistant 两条，
    保持与前端既有渲染逻辑（按 role 逐条渲染）一致。
    """
    sid = _normalize_sid(sid)
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, question, answer, `timestamp`, meta "
                "FROM conversations WHERE session_id = %s ORDER BY id ASC",
                (sid,),
            )
            rows = cur.fetchall()

    messages: List[Dict[str, Any]] = []
    for row in rows:
        created = _fmt(row["timestamp"])
        messages.append({
            "id": row["id"] * 2,
            "conversation_id": sid,
            "role": "user",
            "content": row["question"],
            "meta": None,
            "created_at": created,
        })
        messages.append({
            "id": row["id"] * 2 + 1,
            "conversation_id": sid,
            "role": "assistant",
            "content": row["answer"],
            "meta": row["meta"],
            "created_at": created,
        })
    return messages


def recent_history(sid, turns: int = 5) -> List[Dict[str, str]]:
    """
    取某个会话最近 N 轮问答，供 RAG 注入上下文。

    :param sid:   会话标识
    :param turns: 取最近多少轮
    :return: [{"question": "...", "answer": "..."}, ...]，按时间正序
    """
    if turns <= 0:
        return []
    sid = _normalize_sid(sid)

    # 倒序取最近 turns 行，避免把整个会话读进内存
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT question, answer FROM conversations "
                "WHERE session_id = %s ORDER BY id DESC LIMIT %s",
                (sid, int(turns)),
            )
            rows = cur.fetchall()

    pairs: List[Dict[str, str]] = []
    for row in reversed(rows):
        question = (row["question"] or "").strip()
        answer = (row["answer"] or "").strip()
        if question and answer:
            pairs.append({"question": question, "answer": answer})
    return pairs


# ==================== 命令行入口 ====================
# 自测：建表 → 写入 → 查询 → 绑定 → 清理
#   python mysql_qa/utils/chat_store.py
# 过期清理（临时会话 1 天 / 登录用户 30 天，可在 config.ini [chat] 调整）
#   python mysql_qa/utils/chat_store.py --purge --dry-run    # 只统计
#   python mysql_qa/utils/chat_store.py --purge              # 真正删除
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="会话存储自测 / 过期清理")
    parser.add_argument("--purge", action="store_true", help="清理过期会话")
    parser.add_argument("--dry-run", action="store_true", help="配合 --purge，只统计不删除")
    parser.add_argument("--anon-days", type=int, default=1, help="临时会话保留天数（默认 1）")
    parser.add_argument("--user-days", type=int, default=30, help="登录用户会话保留天数（默认 30）")
    args = parser.parse_args()

    if args.purge:
        result = purge_expired(args.anon_days, args.user_days, dry_run=args.dry_run)
        if args.dry_run:
            print(f"[预览] 将清理会话 {result['sessions']} 个、问答 {result['turns']} 轮"
                  f"（临时 {args.anon_days} 天 / 登录 {args.user_days} 天）")
        else:
            print(f"[完成] 已清理会话 {result['sessions']} 个、问答 {result['turns']} 轮")
        raise SystemExit(0)

    logger.info("=== 会话存储 MySQL 自测开始 ===")
    init_db()

    # 1. 建会话
    conv = create_conversation()
    sid = conv["session_id"]
    print(f"[1] 新建匿名会话: {sid} (user_id={conv['user_id']})")

    # 2. 写入两轮问答
    add_exchange(sid, "什么是 RAG 技术？", "RAG 是检索增强生成。", meta="RAG · 耗时 1.2 s")
    add_exchange(sid, "文本切分重要吗？", "非常重要，影响召回质量。", meta="FAQ · 耗时 0.1 s")
    print(f"[2] 写入 2 轮问答完成，当前标题: {get_conversation(sid)['title']}")

    # 3. 查询会话与消息
    print(f"[3] 会话列表条数（未过滤 user_id）: {len(list_conversations())}")
    messages = list_messages(sid)
    for m in messages:
        print(f"    {m['role']:<9} | {m['content'][:24]}")

    # 4. 模拟登录后绑定用户
    print(f"[4] 绑定到 user_id=1: {bind_user(sid, 1)}")
    print(f"    该用户名下会话数: {len(list_conversations(user_id=1))}")

    # 5. RAG 取历史
    print(f"[5] 最近 5 轮历史: {recent_history(sid, turns=5)}")

    # 6. 重命名 + 删除
    print(f"[6] 重命名: {rename_conversation(sid, 'RAG 入门')} → {get_conversation(sid)['title']}")
    print(f"    删除会话: {delete_conversation(sid)}，剩余: {get_conversation(sid)}")

    logger.info("=== 会话存储 MySQL 自测结束 ===")
