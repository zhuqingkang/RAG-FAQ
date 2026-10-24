# mysql_qa/utils/qa_service.py
# 问答系统单例：供 app_async 的三条链路共用，保证只初始化一次
import sys
import threading
from pathlib import Path

# 只保证项目根在 sys.path；包内导入用相对导入，外部一律用 mysql_qa.xxx
_PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from base import logger
# 同包模块直接用相对导入，不再需要 try/except 兼容「包导入 / 裸导入」两种方式
from .mysql_qa_system import MySQLQASystem

_qa_system = None
_init_lock = threading.Lock()


def get_qa_system() -> MySQLQASystem:
    """获取（惰性初始化）问答系统单例。线程安全。"""
    global _qa_system
    if _qa_system is None:
        with _init_lock:
            if _qa_system is None:
                logger.info("正在初始化 MySQL 问答系统 ...")
                _qa_system = MySQLQASystem()
                logger.info("MySQL 问答系统初始化完成")
    return _qa_system


def ensure_ready() -> MySQLQASystem:
    """预热：提前加载 jieba 词典并建好 BM25 索引，避免首个请求耗时抖动。"""
    import jieba

    jieba.initialize()
    return get_qa_system()
