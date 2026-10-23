# -*- coding:utf-8 -*-
# core/vector_store.py
# 重库统一改成“用到时才 import”：
#   torch / pymilvus / milvus_model(BGE-M3) / sentence-transformers(BGE-Reranker)
#   / langchain_core(Document) / document_processor
# 只 import 本文件时几乎瞬间完成，实例化 VectorStore 也不会立刻去读模型权重。
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值
import hashlib
import os

from base import logger, Config

# 模型权重目录要用到 rag_qa 包路径（只拼路径，不进 sys.path）
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_qa_path = os.path.dirname(current_dir)

conf = Config()

# Document 类缓存：langchain_core 依赖 pydantic，第一次构造文档对象时再 import
_document_cls = None


def get_document_class():
    """拿 Document 类（重库延迟导入，全局只 import 一次）"""
    global _document_cls
    if _document_cls is None:
        from langchain_core.documents import Document
        _document_cls = Document
    return _document_cls

'''
    实现: 向量数据的存储和检索
    1. 向量表的创建:字段,稠密(索引,ip),稀疏(索引:ip) ,索引的创建:聚簇的数量nlist, 查询的时候:nprobe
    2. 数据的存储(向量,普通字符): 子块编码->稠密和稀疏(BEG-M3) ,id(md5/uuid)
    3. 向量检索:
        3.1 粗排(混合检索 ,WeightedRanker(1.0,0.7))   性能快
        3.2 精排(bge-reranker-large),通过语义进行排名  性能慢
'''


# core/vector_store.py
# 定义 VectorStore 类，封装向量存储和检索功能
class VectorStore:
    # 初始化方法，设置向量存储的基本参数
    def __init__(self,
                 collection_name=conf.MILVUS_COLLECTION_NAME,
                 host=conf.MILVUS_HOST,
                 port=conf.MILVUS_PORT,
                 database=conf.MILVUS_DATABASE_NAME):
        logger.info('===='*20)
        # 设置 Milvus 集合名称
        self.collection_name = collection_name
        # 设置 Milvus 主机地址
        self.host = host
        # 设置 Milvus 端口号
        self.port = port
        # 设置 Milvus 数据库名称
        self.database = database
        # 设置日志记录器
        self.logger = logger
        # 设备也延迟判断：import torch 本身就要十几秒，没必要在实例化时就付这个代价
        self._device = None
        # 模型懒加载：bge-m3 / bge-reranker 的权重都有好几个 G，
        # 每次 new VectorStore() 都读一遍太浪费 IO，改成第一次真正用到时才加载
        self._embedding_function = None
        self._reranker = None
        # pymilvus 也是重库，真要连 Milvus 时才 import
        from pymilvus import MilvusClient
        # 初始化 Milvus 客户端，连接到指定主机和数据库
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)
        # 调用方法创建或加载 Milvus 集合
        self._create_or_load_collection()

    @property
    def device(self):
        """设备（cuda/cpu）：第一次真正要用模型时才 import torch 判断"""
        if self._device is None:
            import torch
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.logger.info(f"使用设置：{self._device}")
        return self._device

    @property
    def embedding_function(self):
        """BGE-M3 嵌入模型：第一次需要编码时才 import、才读取权重文件"""
        if self._embedding_function is None:
            from milvus_model.hybrid import BGEM3EmbeddingFunction
            m3_path = os.path.join(rag_qa_path, 'models', 'bge-m3')
            self.logger.info(f"加载 BGE-M3 模型:{m3_path}")
            # use_fp16 表示权重类型的精度,其中fp16只支持GPU
            self._embedding_function = BGEM3EmbeddingFunction(
                model_name_or_path=m3_path, use_fp16=(self.device == 'cuda'), device=self.device)
        return self._embedding_function

    @property
    def reranker(self):
        """BGE-Reranker 精排模型：第一次检索到需要精排时才 import、才读取权重文件"""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            reranker_path = os.path.join(rag_qa_path, 'models', 'bge-reranker-large')
            self.logger.info(f"加载 BGE-Reranker 模型:{reranker_path}")
            self._reranker = CrossEncoder(reranker_path, device=self.device)
        return self._reranker

    # 类私有化方法
    def _create_or_load_collection(self):
        # 建表要用字段类型，用到时才 import
        from pymilvus import DataType
        # 检查指定集合是否已经存在
        if not self.client.has_collection(self.collection_name):
            # 创建集合 Schema，禁用自动 ID，启用动态字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加 ID 字段，作为主键，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 添加文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定（懒加载，这里才真正读模型）
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT16_VECTOR,
                             dim=self.embedding_function.dim["dense"])
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 添加学科类别字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 添加时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            # 创建索引参数对象
            index_params = self.client.prepare_index_params()
            # 为稠密向量字段添加 IVF_FLAT 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128} # 簇
            )
            # 为稀疏向量字段添加 SPARSE_INVERTED_INDEX 索引，度量类型为内积 (IP)
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX", # 稀疏
                metric_type="IP",
                params={"drop_ratio_build": 0.2} # 搜索忽略,降噪20% ,忽略向量值最少得20%,目的是减少计算的数据集提升性能
            )

            # 创建 Milvus 集合，应用定义的 Schema 和索引参数
            self.client.create_collection(collection_name=self.collection_name, schema=schema,
                                          index_params=index_params)
            # 记录创建集合的日志
            logger.info(f"已创建集合 {self.collection_name}")
        # 如果集合已存在
        else:
            # 记录加载集合的日志
            logger.info(f"已加载集合 {self.collection_name}")
        # 将集合加载到内存，确保可立即查询
        self.client.load_collection(self.collection_name) # milvus底层是存储和查询分离的,load可以磁盘数据加载到内存里面,提升查询性能

    # 定义方法，向向量存储添加文档
    def add_documents(self, documents, batch_size=8, skip_existing=False):
        if not documents:
            return
        # 先算好每篇文档的 id（正文 md5），已经入库的直接跳过，不用再编码一遍
        ids = [hashlib.md5(doc.page_content.encode('utf-8')).hexdigest() for doc in documents]
        if skip_existing:
            existed = self._existing_ids(ids)
            if existed:
                documents = [doc for doc, doc_id in zip(documents, ids) if doc_id not in existed]
                ids = [doc_id for doc_id in ids if doc_id not in existed]
                logger.info(f"跳过已入库文档 {len(existed)} 个，本次待处理 {len(documents)} 个")
            if not documents:
                logger.info("所有文档都已在向量库中，无需重复写入")
                return
        # 按批编码。一次把全部文本送进模型会占满内存，触发换页，任务管理器里就会看到大量系统 IO。
        for start in range(0, len(documents), batch_size):
            batch = documents[start:start + batch_size]
            texts = [doc.page_content for doc in batch]
            embeddings = self.embedding_function(texts)
            data = []
            for i, doc in enumerate(batch):
                sparse_vector = {}
                row = embeddings["sparse"]._getrow(i)
                for idx, value in zip(row.indices, row.data):
                    sparse_vector[idx] = value
                data.append({
                    "id": ids[start + i],
                    "text": doc.page_content,
                    "dense_vector": embeddings["dense"][i],
                    "sparse_vector": sparse_vector,
                    "parent_id": doc.metadata["parent_id"],
                    "parent_content": doc.metadata["parent_content"],
                    "source": doc.metadata.get("source", "unknown"),
                    "timestamp": doc.metadata.get("timestamp", "unknown")
                })
            self.client.upsert(collection_name=self.collection_name, data=data)
            logger.info(f"已插入或更新 {start + len(data)}/{len(documents)} 个文档")

    def _existing_ids(self, ids, batch_size=200):
        """批量查询 Milvus，返回已经存在的 id 集合（in 表达式不能拼太长，分批查）"""
        existed = set()
        for start in range(0, len(ids), batch_size):
            batch = ids[start:start + batch_size]
            expr = 'id in [' + ','.join(f'"{item}"' for item in batch) + ']'
            rows = self.client.query(collection_name=self.collection_name,
                                     filter=expr, output_fields=["id"])
            existed.update(row["id"] for row in rows)
        return existed

    # 定义方法，执行混合检索并重排序
    def hybrid_search_with_rerank(self, query, k = conf.RETRIEVAL_K, source_filter=None):
        # 搜索请求、融合排序器用到时才 import
        from pymilvus import AnnSearchRequest, WeightedRanker
        # 使用 BGE-M3 嵌入函数生成查询的嵌入
        query_embeddings = self.embedding_function([query])
        # 获取查询的稠密向量
        dense_query_vector = query_embeddings["dense"][0]
        # print(f'dense_query_vector--》{dense_query_vector.shape}')
        # 初始化查询的稀疏向量字典
        sparse_query_vector = {}
        # 获取查询稀疏向量的第 0 行数据

        row = query_embeddings["sparse"]._getrow(0)
        # 获取稀疏向量的非零值索引
        indices = row.indices
        # 获取稀疏向量的非零值
        values = row.data
        # 将索引和值配对，填充稀疏向量字典
        for idx, value in zip(indices, values):
            sparse_query_vector[idx] = value
        # print(f'sparse_query_vector-->{sparse_query_vector}')
        # 初始化过滤表达式，默认不过滤
        filter_expr = f"source == '{source_filter}'" if source_filter else ""
        # print(f'filter_expr--》{filter_expr}')
        # 创建稠密向量搜索请求
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit = k,
            expr=filter_expr
        )
        # 创建稀疏向量搜索请求
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=filter_expr
        )

        # 创建加权排序器，稀疏向量权重 0.3，稠密向量权重 0.7
        ranker = WeightedRanker(0.7, 0.3)
        # 执行混合搜索，返回 Top-K 结果
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=k,
            output_fields=["text", "parent_id", "parent_content", "source", "timestamp"]
        )[0]

        # 将上述搜索到的结果进行Document对象封装，便于查询使用
        sub_chunks = [self._doc_from_hit(hit["entity"])for hit in results]

        # 从子块中提取去重的父文档
        parent_docs = self._get_unique_parent_docs(sub_chunks)
        # 如果只有1个文档或者没有，直接返回跳过重排序
        if len(parent_docs) < 2:
            return parent_docs[:conf.CANDIDATE_M]
        # 如果有父文档，进行重排序
        if parent_docs:
            # 创建查询与文档内容的配对列表
            pairs = [[query, doc.page_content] for doc in parent_docs]
            # 使用 BGE-Reranker 计算每个配对的得分
            scores = self.reranker.predict(pairs)
            # 根据得分从高到低排序文档
            ranked_parent_docs = [doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)]
        # 如果没有父文档，返回空列表
        else:
            ranked_parent_docs = []

        # 返回前 m 个重排序后的文档
        return ranked_parent_docs[:conf.CANDIDATE_M]

    def _get_unique_parent_docs(self, sub_chunks):
        # 用到时才拿 Document 类
        Document = get_document_class()
        # 初始化集合，用于存储已处理的父块内容（去重）
        parent_contents = set()
        # 初始化列表，用于存储唯一父文档
        unique_docs = []
        # 遍历所有子块
        for chunk in sub_chunks:
            # 获取子块的父块内容，默认为子块内容
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            # 检查父块内容是否非空且未重复
            if parent_content and parent_content not in parent_contents:
                # 创建新的 Document 对象，包含父块内容和元数据
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                # 将父块内容添加到去重集合
                parent_contents.add(parent_content)
            # 返回去重后的父文档列表
        return unique_docs

    # 定义类似私有方法，从 Milvus 查询结果创建 Document 对象
    def _doc_from_hit(self, hit):
        # 用到时才拿 Document 类
        Document = get_document_class()
        # 创建并返回 Document 对象，填充内容和元数据
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp")
            }
        )
