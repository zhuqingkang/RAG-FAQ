# mysql_qa/db/MySQLClient.py
# 导入 MySQL 连接库
import time
import threading
from pathlib import Path

import pymysql
# 导入pandas
import pandas as pd

# 导入配置和日志
from base import Config,logger

class MySQLClient:
    def __init__(self):
        logger.info('Mysql 初始化。。。')
        # 同一个连接/游标被多线程共享，读写必须串行化
        self._lock = threading.Lock()
        # ctrl + alt +t
        try:
            self.connection = pymysql.connect(
                host=Config().MYSQL_HOST,
                user=Config().MYSQL_USER,
                password=Config().MYSQL_PASSWORD,
                database=Config().MYSQL_DATABASE,
                port=3306
            )
            # 创建游标对象（操作数据库）
            self.cursor = self.connection.cursor()
            logger.info('Mysql 初始化成功。')
        except Exception as e:
            logger.error(f'MySQL 链接异常，日志是: {e}')

    def create_table(self):
        try:
            # 表名 knowledge（原 jpkb 已改名）。
            # 单学科知识对体量很大时可按 subject_name 分区，
            # 但主键需改为 (id, subject_name)，写法见 readme/开发文档-详细版.md §8.5.1
            sql = """
                create table if not exists knowledge (
                    id int auto_increment primary key,
                    subject_name varchar(100) ,
                    question varchar(1000) ,
                    answer varchar(5000)
                    );
            """
            with self._lock:
                self.cursor.execute(sql)
            print('成功创建 !')
        except Exception as e:
            logger.info(f'创建异常{e}')

    def insert_data(self, csv_path):
        try:
            logger.info('数据插入开始。。。')
            df = pd.read_csv(csv_path)

            sql = """
                    insert into knowledge (subject_name, question, answer) values (%s, %s, %s);
                    """

            with self._lock:
                for i, item in df.iterrows():
                    self.cursor.execute(sql,(item['学科名称'],item['问题'],item['答案']))

                self.connection.commit()
            logger.info('数据插入成功！')
        except Exception as e:
            logger.info(f'数据插入失败，异常日志是： {e}')

    # 查询所有问题列表
    def fetch_questions(self):

        # 获取所有问题
        logger.info(f'Mysql 查询问列表开始。。。')
        try:
            sql ='''
            select question from knowledge;
            '''
            with self._lock:
                self.cursor.execute(sql)
                res = self.cursor.fetchall()
            logger.info(f'Mysql 查询问列表成功！')
            return res
        except Exception as e:
            logger.info(f'Mysql 查询问题列表异常，日志是：{e}')

    def fetch_answer(self, question):
        logger.info(f'Mysql 根据问题查询问答案开始。。。')
        try:
            # 获取指定问题的答案
            sql = '''
            select answer from knowledge where question = %s;'''
            with self._lock:
                self.cursor.execute(sql, (question,))
                res = self.cursor.fetchone()
            logger.info(f'Mysql 根据问题查询问答案成功！')
            # fetchone() 返回的是 (answer,) 元组，这里统一成字符串，避免上层拿到元组/列表
            return res[0] if res else None
        except Exception as e:
            logger.info(f'Mysql 根据问题查询问异常，日志是：{e}')
    def close(self):
        try:
            self.cursor.close()
            self.connection.close()
            logger.info(f'游标及链接关闭成功！')
        except Exception as e:
            logger.info(f'游标关闭失败异常，日志是：{e}')


if __name__ == '__main__':
    start = time.time()
    sql = MySQLClient()
    # sql.create_table()
    # sql.insert_data(Path(__file__).parent.parent/'data'/'JP_Knowledge.csv')
    # print(sql.fetch_questions())
    print(sql.fetch_answer('为什么isalnum函数会通过中文'))
    sql.close()
    print(f'总耗时：{time.time() - start}')
