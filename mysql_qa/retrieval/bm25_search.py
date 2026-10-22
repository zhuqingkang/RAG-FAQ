# retrieval/bm25_search.py
# 导入 BM25 算法
import time
import threading

from rank_bm25 import BM25Okapi
# 导入数值计算库
import numpy as np
# 导入redis
from mysql_qa.cache import RedisClient
# 导入mysql
from mysql_qa.db import MySQLClient
# 导入日志
from base import logger
# 导入文本预处理
import jieba
from mysql_qa.utils import preprocess_text



class BM25Search:
    def __init__(self, redis_client, mysql_client):
        # 初始化日志

        # 初始化 Redis 客户端
        self.redis_client = redis_client
        # 初始化 MySQL 客户端
        self.mysql_client = mysql_client
        # 初始化 BM25 模型
        self.bm25 = None
        # 初始化问题列表
        self.questions = None
        # 初始化原始问题
        self.original_questions = None

        # 每 30 分钟从 MySQL 强制同步一次到 Redis
        self._update_interval = 30 * 60
        self._sync_lock = threading.Lock()
        # 启动时先同步一次
        self.sync_from_mysql()
        self._start_sync_thread()

    def _start_sync_thread(self):
        """后台线程：每隔 30 分钟把 MySQL 问题列表同步到 Redis，并重建 BM25。"""
        def _loop():
            while True:
                time.sleep(self._update_interval)
                try:
                    self.sync_from_mysql()
                except Exception as e:
                    logger.error(f"定时同步 MySQL -> Redis 失败: {e}")

        t = threading.Thread(target=_loop, name="mysql-redis-sync", daemon=True)
        t.start()
        logger.info(f"已启动 MySQL->Redis 定时同步，间隔 {self._update_interval} 秒")

    def sync_from_mysql(self):
        """以 MySQL 为准，覆盖 Redis 中的问题列表和分词结果，并重建 BM25。"""
        with self._sync_lock:
            original_key = "qa_original_questions"
            tokenized_key = "qa_tokenized_questions"

            questions = self.mysql_client.fetch_questions()
            if not questions:
                logger.info("查询无数据! 请先加载数据")
                return False

            self.original_tokenizers = [jieba.lcut(i[0]) for i in questions]
            self.original_questions = [i[0] for i in questions]
            self.questions = self.original_questions

            # 覆盖写入 Redis，保证和 MySQL 一致
            self.redis_client.set_data(original_key, self.original_questions)
            self.redis_client.set_data(tokenized_key, self.original_tokenizers)

            self.bm25 = BM25Okapi(self.original_tokenizers)
            logger.info(f"MySQL -> Redis 同步完成，问题数: {len(self.original_questions)}")
            return True

    def __load_data(self):
        # 兼容旧调用：改为强制同步
        return self.sync_from_mysql()

    def __softmax(self, scores):
        # 计算 Softmax 分数
        exp_scores = np.exp(scores - np.max(scores))
        # 返回归一化分数
        return exp_scores / exp_scores.sum()

    def search(self, query, threshold=0.85) -> str|None:
        """
            :param query:用户的问题
            :threshold ; 查询阈值
            :return answer
        """
        # 搜索查询 ->判断
        if not query:
            logger.info('查询为空,无效查询，请重新输入')
            return None

        # 检查 Redis 缓存:
        answer = self.redis_client.get_answer(query)
        # 返回缓存答案
        if answer:
            logger.info('redis 数据查询成功')
            return answer

        # 查询-> 分词
        tokenizer = jieba.lcut(query)
        # 计算 BM25 分数
        scores = self.bm25.get_scores(tokenizer)
        # 计算 Softmax 分数
        scores = self.__softmax(scores)
        # 获取最高分索引
        max_index = np.argmax(scores)
        # print(max_index)
        # 获取最高分
        score_max = scores[max_index]
        # 检查是否超过阈值
        if score_max > threshold:
            # 获取原始问题
            beast_query = self.original_questions[max_index]
            # 获取答案:mysql
            answer = self.mysql_client.fetch_answer(beast_query)
            # 缓存答案 key: answer:{query}
            self.redis_client.set_data(f'answer:{query}',answer)
            # 记录搜索成功
            logger.info('检索成功')
            # 返回答案
            return answer

        # 记录无可靠答案
        logger.info('无可靠答案，请重新查询！')
        # 返回 None
        return None

if __name__ == '__main__':
    m25search = BM25Search(RedisClient(), MySQLClient())
    m25search.search('为什么isalnum函数会通过中文呢？')