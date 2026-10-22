# utils/qa_service.py
# 问答系统单例：Flask 版和 FastAPI 异步版共用，保证只初始化一次
import sys
import threading
from pathlib import Path

# ===== 路径处理：允许直接运行/从任意目录导入 =====
_UTILS_DIR = Path(__file__).resolve().parent
_MYSQL_QA_DIR = _UTILS_DIR.parent
_PROJECT_DIR = _MYSQL_QA_DIR.parent
for _p in (str(_MYSQL_QA_DIR), str(_PROJECT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from base import logger

try:
    # 作为包导入时（from utils.qa_service import ...）
    from .mysql_qa_system import MySQLQASystem
except ImportError:
    # 直接运行本文件时
    from mysql_qa_system import MySQLQASystem

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
