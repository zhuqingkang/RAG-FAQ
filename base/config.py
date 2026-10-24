# base/config.py
# 导入配置解析库
import configparser
from pathlib  import Path


# 项目根目录
Base_dir = Path(__file__).resolve().parent.parent

class Config:
    # 初始化配置，加载 config.ini 文件
    def __init__(self, config_file= Base_dir / 'config.ini'):
        # 创建配置解析器
        self.config = configparser.ConfigParser()
        # 读取配置文件
        self.config.read(config_file,encoding='utf-8')

        # 项目根目录
        self.BASE_DIR = Base_dir
        self.SHOW_DIR = Base_dir / 'rag_qa/data/ai_data'
        self.Sample_DIR = Base_dir / 'rag_qa/data/samples'

        # 预训练模型路径
        self.Pre_Model_Path = Base_dir / 'rag_qa/models'
        self.Pre_bge_m3 = self.Pre_Model_Path / 'bge-m3'
        self.Pre_bge_rerank = self.Pre_Model_Path / 'bge-reranker-large'
        self.Pre_bert = self.Pre_Model_Path / 'nlp_bert_document-segmentation_chinese-base'

        # MySQL 配置
        # MySQL 主机地址
        self.MYSQL_HOST = self.config.get('mysql', 'host', fallback='localhost')
        # MySQL 用户名
        self.MYSQL_USER = self.config.get('mysql', 'user', fallback='root')
        # MySQL 密码
        self.MYSQL_PASSWORD = self.config.get('mysql', 'password', fallback='mysql')
        # MySQL 数据库名
        self.MYSQL_DATABASE = self.config.get('mysql', 'database', fallback='subjects_kg')

        # Redis 配置
        # Redis 主机地址
        self.REDIS_HOST = self.config.get('redis', 'host', fallback='localhost')
        # Redis 端口
        self.REDIS_PORT = self.config.getint('redis', 'port', fallback=6379)
        # Redis 密码
        self.REDIS_PASSWORD = self.config.get('redis', 'password', fallback='1234')
        # Redis 数据库编号
        self.REDIS_DB = self.config.getint('redis', 'db', fallback=0)
        # 日志文件路径
        self.LOG_FILE = self.config.get('logger', 'log_file', fallback=Base_dir/'logs/app.log')

        # Milvus 配置
        # Milvus 主机地址
        self.MILVUS_HOST = self.config.get('milvus', 'host', fallback='localhost')
        # Milvus 端口
        self.MILVUS_PORT = self.config.get('milvus', 'port', fallback='19530')
        # Milvus 数据库名
        self.MILVUS_DATABASE_NAME = self.config.get('milvus', 'database_name', fallback='itcast')
        # Milvus 集合名
        self.MILVUS_COLLECTION_NAME = self.config.get('milvus', 'collection_name', fallback='edurag_final')

        # LLM 配置（统一走 OpenAI 兼容协议：换供应商只改 model / base_url / api_key）
        # 模型名
        self.LLM_MODEL = self.config.get('llm', 'model', fallback='qwen-plus')
        # API 密钥：优先通用键 api_key，回退旧的 dashscope_api_key
        api_key = (
            self.config.get('llm', 'api_key', fallback='')
            or self.config.get('llm', 'dashscope_api_key', fallback='')
        )
        if not api_key:
            # 保持原有语义：密钥缺失直接报错，而不是等到首个请求才失败
            raise configparser.NoOptionError('api_key', 'llm')
        self.LLM_API_KEY = api_key
        # API 地址：优先通用键 base_url，回退旧的 dashscope_base_url
        self.LLM_BASE_URL = (
            self.config.get('llm', 'base_url', fallback='')
            or self.config.get('llm', 'dashscope_base_url',
                               fallback='https://dashscope.aliyuncs.com/compatible-mode/v1')
        )
        # 单次生成超时（秒）：本地小模型可给 30，云端一般 60~120
        self.LLM_TIMEOUT = self.config.getfloat('llm', 'timeout', fallback=60)

        # 联网搜索：enable_search 是 DashScope 专有参数，DeepSeek / vLLM 等带上会 400
        # auto = 仅当 base_url 指向 DashScope 时启用；on / off 为强制开关
        web_search = self.config.get('llm', 'web_search', fallback='auto').strip().lower()
        if web_search == 'auto':
            _url = (self.LLM_BASE_URL or '').lower()
            self.LLM_ENABLE_SEARCH = ('dashscope' in _url) or ('aliyuncs' in _url)
        else:
            self.LLM_ENABLE_SEARCH = (web_search == 'on')

        # 兼容旧属性名：rag_qa/core/strategy_selector.py 等仍在用 DASHSCOPE_*
        self.DASHSCOPE_API_KEY = self.LLM_API_KEY
        self.DASHSCOPE_BASE_URL = self.LLM_BASE_URL

        # 缓存配置：把目录清单/解析文本/切块结果存成 JSON，减少重复的磁盘 IO
        # 缓存目录（默认 rag_qa/data/.cache）
        self.CACHE_DIR = str(self.config.get('cache', 'cache_dir',
                                            fallback=Base_dir / 'rag_qa/data/.cache'))
        # 是否逐个文件核对 mtime/size。默认 False：完全信任目录清单，最省 IO
        # 改成 True 后，文件内容被原地覆盖（目录 mtime 不变）也能被发现
        self.CACHE_VERIFY_FILES = self.config.getboolean('cache', 'verify_files', fallback=False)
        # 是否启用父子切块结果缓存
        self.CACHE_CHUNKS = self.config.getboolean('cache', 'chunks', fallback=True)

        # 会话保留策略
        # 未登录的临时会话保留天数（超过即清理）
        self.CHAT_ANON_RETENTION_DAYS = self.config.getint('chat', 'anonymous_retention_days', fallback=1)
        # 已登录用户的会话保留天数
        self.CHAT_USER_RETENTION_DAYS = self.config.getint('chat', 'user_retention_days', fallback=30)
        # 启动时是否自动执行一次清理
        self.CHAT_AUTO_PURGE = self.config.getboolean('chat', 'auto_purge', fallback=True)

        # 检索参数
        # 父块大小
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        # 子块大小
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        # 块重叠大小
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        # 检索返回数量
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        # 最终候选数量
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone')
        self.VALID_SOURCES = eval(
            self.config.get('app', 'valid_sources', fallback=["ai", "java", "test", "ops", "bigdata"]))
