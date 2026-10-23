# -*-coding:utf-8-*-
# core/rag_system.py
import os
#   导入 time 模块，用于计算时间
import time

from base import logger, Config
# core 内部模块统一用相对导入，避免与顶层同名模块（prompts / vector_store）串包
from .prompts import RAGPrompts
from .query_classifier import QueryClassifier  #   导入查询分类器
from .strategy_selector import StrategySelector  #   导入策略选择器
from .vector_store import VectorStore  # 导入向量数据库对象

# 模型权重目录要用到 rag_qa 包路径（不是 sys.path，只是拼路径）
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_qa_path = os.path.dirname(current_dir)

conf = Config()

# ===== 对话历史注入配置 =====
HISTORY_TURNS = 5              # 最多带几轮问答
HISTORY_ITEM_MAX_CHARS = 200   # 单条问题 / 答案最多保留多少字
HISTORY_MAX_CHARS = 1500       # 历史整体预算，超出就丢弃更早的轮次

#   定义 RAGSystem 类，封装 RAG 系统的核心逻辑
class RAGSystem:
    #   初始化方法，设置 RAG 系统的基本参数
    def __init__(self, vector_store, llm):
        #   设置向量数据库对象
        self.vector_store = vector_store
        #   设置大语言模型调用函数
        self.llm = llm
        #   获取 RAG 提示模板
        self.rag_prompt = RAGPrompts.rag_prompt()
        #   初始化查询分类器
        classifier_path = os.path.join(rag_qa_path, 'models', 'bert_query_classifier')
        self.query_classifier = QueryClassifier(model_path=classifier_path)
        #   初始化策略选择器
        self.strategy_selector = StrategySelector()

        self.tools = {
            '子查询检索': self._retrieve_with_subqueries,
            '回溯问题检索': self._retrieve_with_backtracking,
            '假设问题检索': self._retrieve_with_hyde,
            '直接检索': self.vector_store.hybrid_search_with_rerank,
        }

    # 把单条文本压成一行并截断，避免长答案把 prompt 撑爆
    @staticmethod
    def _clip(text, limit=HISTORY_ITEM_MAX_CHARS):
        text = " ".join(str(text).split())
        return text[:limit] + ("..." if len(text) > limit else "")

    # 把最近几轮问答拼成 prompt 里可直接使用的历史片段
    def _build_history_context(self, history):
        # 不是 list 一律视为没带历史
        if not history or not isinstance(history, list):
            return ''

        # 从最近一轮往前累加，超出总长预算就停，保证留下的总是最新的对话
        blocks = []
        total = 0
        for item in reversed(history[-HISTORY_TURNS:]):
            # 脏数据（不是 dict、缺字段、空内容）直接跳过
            if not isinstance(item, dict):
                continue
            question = item.get('question')
            answer = item.get('answer')
            if not question or not answer:
                continue
            block = f"Q:{self._clip(question)}\nA:{self._clip(answer)}"
            if total + len(block) > HISTORY_MAX_CHARS:
                break
            blocks.append(block)
            total += len(block)

        # 上面是倒着累加的，这里翻回来还原时间顺序
        blocks.reverse()
        return "\n".join(blocks)

    # 1. 生成答案
    def generate_answer(self, query, source_filter=None, history=None):
        #   耗时
        start_time = time.time()

        # 构造对话历史片段：格式非法一律按「无历史」处理，不让它影响主流程
        history_context = self._build_history_context(history)
        if history_context:
            logger.info(f'注入对话历史，长度 {len(history_context)} 字符')

        #   查询问题类型
        category = self.query_classifier.predict_category(query)
        #   如果查询属于“通用知识”类别，则直接使用 LLM 回答
        if category == '通用知识':
            # 设置提示词,变量替换
            prompt = self.rag_prompt.format(context='', question=query, phone=conf.CUSTOMER_SERVICE_PHONE,
                                            history=history_context)
            # 模型调用
            answer = self.llm(prompt)
            return answer

        #   否则，进行 RAG 检索并生成答案
        logger.info('查询为专业咨询,执行RAG流程....')
        #   选择检索策略
        # 格式化模版,替换变量
        prompt = self.strategy_selector.strategy_prompt_template.format(query=query)

        # LLM模型调用,由大模型帮我们选择具体的查询策略
        strategy = self.llm(prompt)
        # 去掉模型可能输出的多余空白/换行,否则下面精确匹配会落空并退回“直接检索”
        strategy = (strategy or '').strip()
        logger.info(f'选用策略：{strategy}')
        # 根据检索策略,选择具体的查询策略,来执行后续的处理
        # 检索相关文档: retrieve_and_merge
        docs = self.retrieve_and_merge(query=query, source_filter=source_filter, strategy=strategy)
        # 遍历文档, 使用换行符拼接文档 -> 把final_docs 里面的多条语句,通过\n,拼接成一个context
        if docs:
            context = '\n'.join([i.page_content for i in docs])
            logger.info(f"构建上下文完成，包含 {len(docs)} 个文档块")
        else:
            context = ""
            logger.info("未检索到相关文档，上下文为空")

        # 构造 Prompt，调用大语言模型生成答案
        prompt = self.rag_prompt.format(context=context,
                                        question=query,
                                        history=history_context,
                                        phone=conf.CUSTOMER_SERVICE_PHONE)
        # 模型调用
        answer = self.llm(prompt)

        #   记录查询处理完成的日志
        end_time = time.time()
        logger.info(f'查询增强耗时为:{end_time - start_time :.2f}')
        # 返回结果
        return answer


    # 2.策略选择,根据传输的策略,来选择具体的查询策略,并做后续的向量数据检索
    # 并基于具体的查询策略,从向量库里面返回Top-k
    def retrieve_and_merge(self, query, source_filter=None, strategy=None) -> list:
        if strategy is None:
            logger.info('查询策略为空，请重新查询')
            return []

        # 策略名 -> 检索方法的映射表（在 __init__ 中构建）
        # 模型输出可能带引号/解释文字而无法精确匹配，取不到时退回「直接检索」
        retrieve = self.tools.get(strategy, self.tools['直接检索'])
        docs = retrieve(query = query, source_filter=source_filter)

        # ---- 旧的 if/elif 分派：改用 self.tools 字典分派后注释保留 ----
        # if strategy == '回溯问题检索':
        #     docs = self._retrieve_with_backtracking(query, source_filter)
        # elif strategy == '子查询检索':
        #     docs = self._retrieve_with_subqueries(query, source_filter)
        # elif strategy == '假设问题检索':
        #     docs = self._retrieve_with_hyde(query, source_filter)
        # else:
        #     docs = self.vector_store.hybrid_search_with_rerank(query, source_filter=source_filter)

        # 返回数据
        return docs


    # 3.1 假设文档提示词处理
    #   定义私有方法，使用假设文档进行检索（HyDE）
    def _retrieve_with_hyde(self, query, source_filter) -> list:
        logger.info(f"使用 HyDE 策略进行检索 (查询: '{query}')")

        #   获取假设问题生成的 Prompt 模板
        prompt = RAGPrompts.hyde_prompt().format(query = query)
        #   调用大语言模型生成假设答案
        hyde_answer = self.llm(prompt)
        #   使用假设答案进行检索，并返回检索结果
        docs = self.vector_store.hybrid_search_with_rerank(hyde_answer , source_filter = source_filter)

        return docs

    # 3.2 提示词:子查询进行检索
    #   定义类似私有方法，使用子查询进行检索
    def _retrieve_with_subqueries(self, query, source_filter):
        # pass
        #   获取子查询生成的 Prompt 模板
        prompt = RAGPrompts.subquery_prompt().format(query = query)
        #   调用大语言模型生成子查询列表,并按照\n分割
        prompt = self.llm(prompt)
        # 按照\n 把字符串分割成两个子查询 - > []
        queries = prompt.split('\n')

        #   判断子查询是否有返回数据,如果无,返回[]
        if queries is  None:
            logger.info(f'子查询为空，请重新查询...')
            return []
        #   初始化空列表，用于存储所有子查询的检索结果
        all_docs = []

        #   遍历每个子查询
        for query in queries:
            # 使用子查询混合检索,并添加入列表
            all_docs.extend(self.vector_store.hybrid_search_with_rerank(query, source_filter = source_filter))
        #  去重
        #   遍历 -> dict , key = 文档内容
        dict_docs = {i.page_content:i for i in all_docs}
        #  获取字典values ,并转列表
        docs_ls = list(dict_docs.values())
        # 返回文档
        return docs_ls


    # 3.3 提示词:回溯问题
    #   定义类似私有方法，使用回溯问题进行检索
    def _retrieve_with_backtracking(self, query, source_filter):
        # pass
        #   获取回溯问题生成的 Prompt 模板
        prompt = RAGPrompts.backtracking_prompt()
        # 格式化模板,参数替换
        prompt = prompt.format(query = query)
        #   调用大语言模型生成回溯问题
        opt_query = self.llm(prompt)
        #   使用回溯问题进行混合检索，并返回检索结果
        docs = self.vector_store.hybrid_search_with_rerank(opt_query , source_filter=source_filter)

        return docs

