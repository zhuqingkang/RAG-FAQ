# mysql_qa/utils/migrate_sqlite_history.py
# 一次性迁移：把旧 SQLite 会话库（mysql_qa/data/chat_history.db）里的对话搬到 MySQL。
#
# 用法（在 integrated_qa_system 目录下执行）：
#   python mysql_qa/utils/migrate_sqlite_history.py             # 只预览，不写库
#   python mysql_qa/utils/migrate_sqlite_history.py --apply     # 真正写入 MySQL
#
# 幂等设计：旧会话 id 通过 uuid5 映射成固定 session_id，
# 重复执行时已迁移过的会话会被 import_conversation 跳过，不会写出重复数据。
import argparse
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Dict, List

# 保证直接运行本文件时能导入项目包
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from base import logger
from mysql_qa.utils import chat_store

# 旧库默认位置：mysql_qa/data/chat_history.db
DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "chat_history.db"

# 固定命名空间：同一个旧会话每次都能算出同一个 session_id，迁移才可重复执行
NAMESPACE = uuid.UUID("2f1d9b6c-5a3e-4c7d-8b1a-0e4f6d2c9a71")


def map_session_id(old_cid) -> str:
    """旧的自增 id → 新的 36 位 UUID（确定性映射）。"""
    return str(uuid.uuid5(NAMESPACE, f"sqlite-session-{old_cid}"))


def read_old_conversations(db_path: Path) -> List[Dict]:
    """
    读取旧库并把 messages 配对成「一问一答」。

    :return: [{old_id, title, created_at, updated_at, exchanges, orphans}, ...]
             exchanges = [(question, answer, timestamp, meta), ...]
             orphans   = 只有提问、没有回答的消息条数（迁移时跳过）
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    result: List[Dict] = []
    conversations = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations ORDER BY id"
    ).fetchall()

    for conv in conversations:
        messages = conn.execute(
            "SELECT role, content, meta, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY id ASC",
            (conv["id"],),
        ).fetchall()

        exchanges: List[tuple] = []
        orphans = 0
        pending = None
        for msg in messages:
            if msg["role"] == "user":
                # 连续两条 user（上一问没答案）时，丢弃旧的那条并计数
                if pending is not None:
                    orphans += 1
                pending = msg
            elif msg["role"] == "assistant" and pending is not None:
                exchanges.append((
                    pending["content"],
                    msg["content"],
                    msg["created_at"] or pending["created_at"],
                    msg["meta"],
                ))
                pending = None
        if pending is not None:
            orphans += 1

        result.append({
            "old_id": conv["id"],
            "title": conv["title"],
            "created_at": conv["created_at"],
            "updated_at": conv["updated_at"],
            "exchanges": exchanges,
            "orphans": orphans,
        })

    conn.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="SQLite 会话历史 → MySQL 迁移")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="旧 SQLite 库路径")
    parser.add_argument("--apply", action="store_true", help="真正写入 MySQL（默认只预览）")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"未找到旧库：{db_path}\n可能已经迁移过，或路径需要手动指定（--db）")
        return

    conversations = read_old_conversations(db_path)
    total_turns = sum(len(c["exchanges"]) for c in conversations)
    total_orphans = sum(c["orphans"] for c in conversations)

    print(f"旧库：{db_path}")
    print(f"会话数：{len(conversations)}，可迁移问答轮次：{total_turns}，残缺消息：{total_orphans} 条（跳过）")
    print("-" * 72)

    if not args.apply:
        for conv in conversations:
            print(f"[预览] 旧#{conv['old_id']:<3} → {map_session_id(conv['old_id'])}  "
                  f"{len(conv['exchanges'])} 轮  {conv['created_at']}  {conv['title']}")
        print("-" * 72)
        print("以上为预览，未写入任何数据。加 --apply 参数执行迁移。")
        return

    # 建表（幂等）
    chat_store.init_db()

    migrated_sessions = 0
    migrated_turns = 0
    skipped_sessions = 0

    for conv in conversations:
        if not conv["exchanges"]:
            skipped_sessions += 1
            continue
        written = chat_store.import_conversation(
            map_session_id(conv["old_id"]),
            conv["title"],
            conv["exchanges"],
            created_at=conv["created_at"],
            updated_at=conv["updated_at"],
        )
        if written:
            migrated_sessions += 1
            migrated_turns += written
            print(f"[写入] 旧#{conv['old_id']} → {written} 轮  {conv['title']}")
        else:
            skipped_sessions += 1
            print(f"[跳过] 旧#{conv['old_id']} 已存在同名会话（幂等）")

    print("-" * 72)
    print(f"完成：迁移会话 {migrated_sessions} 个、问答 {migrated_turns} 轮；"
          f"跳过 {skipped_sessions} 个（空会话或已迁移）")
    logger.info(f"SQLite 历史迁移完成：{migrated_sessions} 会话 / {migrated_turns} 轮")


if __name__ == "__main__":
    main()
