# -*-coding:utf-8-*-
# core/strategy_selector.py
# base 是项目根下的顶层包，需要项目根在 sys.path（入口层负责、rag_service 也做了兜底）
# 导入 LangChain 提示模板
try:
    # 新版 langchain（>=0.2）中 PromptTemplate 位于 langchain_core
    from langchain_core.prompts import PromptTemplate
except ImportError:
    # 兼容旧版 langchain
    from langchain.prompts import PromptTemplate
# 导入日志和配置
from base import logger, Config
# 导入 OpenAI
from openai import OpenAI


class StrategySelector:
    def __init__(self):
        # 初始化 OpenAI 兼容客户端（DashScope / DeepSeek / 本地 vLLM 同一套接口）
        self.client = OpenAI(api_key=Config().LLM_API_KEY,
                             base_url=Config().LLM_BASE_URL
                             )
        # 获取策略选择提示模板
        self.strategy_prompt_template = self._get_strategy_prompt()


    def call_dashscope(self, prompt):
        # 调用大模型（方法名沿用历史，实际已不限于 DashScope）
        try:
            # 联网搜索是 DashScope 的专有参数，其它供应商带上会直接 400，
            # 是否发送由 Config().LLM_ENABLE_SEARCH 决定（[llm] web_search = auto/on/off）
            kwargs = {}
            if Config().LLM_ENABLE_SEARCH:
                kwargs["extra_body"] = {"enable_search": True}

            # 创建聊天完成请求
            completion = self.client.chat.completions.create(
                model=Config().LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                **kwargs
            )

            # 返回完成结果
            return completion.choices[0].message.content if completion.choices else "直接检索"
        except Exception as e:
            # 记录 API 调用失败
            logger.error(f"LLM 调用失败: {e}")
            # 默认返回直接检索
            return "直接检索"

    def _get_strategy_prompt(self):
        #   定义类似私有方法，获取策略选择 Prompt 模板
        return PromptTemplate(
            template="""
            你是一个智能助手，负责分析用户查询 {query}，并从以下四种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。

            以下是几种检索增强策略及其适用场景：

            1.  **直接检索：**
                * 描述：对用户查询直接进行检索，不进行任何增强处理。
                * 适用场景：适用于查询意图明确，需要从知识库中检索**特定信息**的问题，例如：
                    * 示例：
                        * 查询：AI 学科学费是多少？
                        * 策略：直接检索
                    * 查询：JAVA的课程大纲是什么？
                        * 策略：直接检索
            2.  **假设问题检索：**
                * 描述：使用 LLM 生成一个假设的答案，然后基于假设答案进行检索。
                * 适用场景：适用于查询较为抽象，直接检索效果不佳的问题，例如：
                    * 示例：
                        * 查询：人工智能在教育领域的应用有哪些？
                        * 策略：假设问题检索
            3.  **子查询检索：**
                * 描述：将复杂的用户查询拆分为多个简单的子查询，分别检索并合并结果。
                * 适用场景：适用于查询涉及多个实体或方面，需要分别检索不同信息的问题，例如：
                    * 示例：
                        * 查询：比较 Milvus 和 Zilliz Cloud 的优缺点。
                        * 策略：子查询检索
            4.  **回溯问题检索：**
                * 描述：将复杂的用户查询转化为更基础、更易于检索的问题，然后进行检索。
                * 适用场景：适用于查询较为复杂，需要简化后才能有效检索的问题，例如：
                    * 示例：
                        * 查询：我有一个包含 100 亿条记录的数据集，想把它存储到 Milvus 中进行查询。可以吗？
                        * 策略：回溯问题检索

            根据用户查询 {query}，直接返回最适合的策略名称，例如 "直接检索"。不要输出任何分析过程或其他内容。
            """
            ,
            input_variables=["query"],
        )

    #   定义方法，选择检索策略
    def select_strategy(self, query):
        #   调用 LLM 获取检索策略
        strategy = self.call_dashscope(self.strategy_prompt_template.format(query=query))
        logger.info(f"为查询 '{query}' 选择的检索策略：{strategy}")
        return strategy

if __name__ == '__main__':
    strategySelector = StrategySelector()
    # print(f'strategySelector.clinet--->{strategySelector.client}')
    # result = strategySelector.call_dashscope(prompt="上海今天天气如何")
    # print(f'result--》{result}')
    strategy = strategySelector.select_strategy(query="Mysql数据库能不能支持100w个样本的插入")
    print(strategy)
