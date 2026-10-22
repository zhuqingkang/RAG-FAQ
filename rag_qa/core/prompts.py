# core/prompts.py
# 导入 PromptTemplate 类，用于创建 Prompt 模板
try:
    # 新版 langchain（>=0.2）中 PromptTemplate 位于 langchain_core
    from langchain_core.prompts import PromptTemplate
except ImportError:
    # 兼容旧版 langchain
    from langchain.prompts import PromptTemplate


# 定义 RAGPrompts 类，用于管理所有 Prompt 模板
class RAGPrompts:
    # 定义 RAG 提示模板
    @staticmethod
    def rag_prompt():
        # 创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""
            你是一个智能助手，帮助用户回答问题。
            如果提供了上下文，请基于上下文回答；如果没有上下文，请直接根据你的知识回答。
            如果答案来源于检索到的文档，请在回答中说明。

            上下文: {context}
            问题: {question}

            如果无法回答，请回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”
            回答:
            """,
            #   定义输入变量
            input_variables=["context", "question", "phone"],
        )

    # 定义假设问题生成的 Prompt 模板
    @staticmethod
    def hyde_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""
               假设你是用户，想了解以下问题，请生成一个简短的假设答案：
               问题: {query}
               假设答案:
               """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义子查询生成的 Prompt 模板
    @staticmethod
    def subquery_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""
               将以下复杂查询分解为多个简单子查询，每行一个子查询，最多生成两个子查询（只保留子查询问题，其他的文本都不需要）：
               eg:
               用户原始query："Milvus 和 Zilliz Cloud 在功能上有什么不同？
               子查询："Milvus 有哪些功能？"，"Zilliz Cloud 有哪些功能？"

               查询: {query}
               子查询:
               """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义回溯问题生成的 Prompt 模板
    @staticmethod
    def backtracking_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""
               将以下复杂查询简化为一个更简单的问题：
               查询: {query}
               简化问题:
               """,
            #   定义输入变量
            input_variables=["query"],
        )

if __name__ == '__main__':
    # rga_prompt = RAGPrompts.rag_prompt()
    # result = rga_prompt.format(context="黑马程序员", question="这个机构叫什么名称", phone="12345")
    # subquery_prompt = RAGPrompts.subquery_prompt()
    # result = subquery_prompt.format(query="姚明和王治郅都是NBA球员吗?")

    backtracking_prompt = RAGPrompts.backtracking_prompt()
    result = backtracking_prompt.format(query="我想自研一个腾讯会议,然后也能跟你们一样,自己设计这个系统靠谱吗?")
    print(f'result-->{result}')