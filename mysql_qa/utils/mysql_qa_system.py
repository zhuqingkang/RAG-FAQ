# utils/mysql_qa_system.py
# 问答系统核心：组合 MySQL 客户端、Redis 客户端与 BM25 检索
import os.path
import sys
from pathlib import Path

# ===== 路径处理：保证从任意目录导入/运行时都能找到 mysql_qa 与项目根 =====
_UTILS_DIR = Path(__file__).resolve().parent
_MYSQL_QA_DIR = _UTILS_DIR.parent
_PROJECT_DIR = _MYSQL_QA_DIR.parent
for _p in (str(_MYSQL_QA_DIR), str(_PROJECT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from db.MySQLClient import MySQLClient
# 导入 Redis 客户端
from cache.RedisClient import RedisClient

# 导入 BM25 搜索
from retrieval.bm25_search import BM25Search

# 导入日志
from base import logger

# 导入时间库
import time

class MySQLQASystem:
    def __init__(self):
        # 初始化日志
        self.logger = logger
        # 初始化 MySQL 客户端
        self.mysql_client = MySQLClient()
        # 初始化 Redis 客户端
        self.redis_client = RedisClient()
        # 初始化 BM25 搜索
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)

    def query(self, query) -> tuple:
        """
        走 FAQ 链路检索答案。
        :return: (answer, need_rag)
                 answer 为 None 表示 FAQ 没有可用答案；
                 need_rag 为 True 表示上层应转入 RAG 链路。
        """
        logger.debug("FAQ 问答开始。。。")

        # 查询 MySQL 系统: 开始时间
        start = time.time()
        logger.info(f"query: {query}")
        # 解包 BM25 的双返回值：答案 + 是否需要转 RAG
        answer, need_rag = self.bm25_search.search(query)
        logger.info(f'FAQ 耗时是 {time.time() - start } s')
        # 返回答案与调度标志
        return answer, need_rag

def main():
    # 初始化 MySQL 系统
    mysql_system = MySQLQASystem()
    # 打印欢迎信息
    print("\n欢迎使用 MySQL 问答系统！")
    print("输入查询进行问答，输入 'exit' 退出。")
    # 循环
    while True:
        # 获取用户输入
        query = input('请输入：')
        if query == 'exit':
            break
        # 执行查询
        answer, need_rag = mysql_system.query(query)
        if answer:
            # 打印答案
            print(answer)
        elif need_rag:
            # FAQ 拿不到可用答案，真实服务里这里转 RAG 链路
            print('FAQ 未命中，需要转 RAG 链路')



if __name__ == "__main__":
    # 运行主程序
    main()
