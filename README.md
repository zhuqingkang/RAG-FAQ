# EduRAG 集成问答系统

单服务、双链路的教育领域问答系统：**FAQ**（MySQL + BM25 + Redis）优先，未命中转 **RAG**（Milvus + BGE 精排 + 大模型生成）。对外只暴露一套页面与一个问答接口。

- 详细开发文档：[readme/开发文档-详细版.md](readme/开发文档-详细版.md)
- 简略版副本：[readme/开发文档-简略版.md](readme/开发文档-简略版.md)
- 项目介绍页（单文件离线版，双击即可打开）：[readme/project-intro.html](readme/project-intro.html)
- 项目结构图：[readme/architecture.svg](readme/architecture.svg)　｜　请求流程图：[readme/flow.svg](readme/flow.svg)

![项目结构图](readme/architecture.svg)

---

## 快速启动

```bash
# 1) 安装依赖（版本锁定见 requirements.txt，来自 conda env edurag / Python 3.12.14）
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

# 2) 生成本地配置，并把 DashScope API Key 填进 config.ini
cp config.example.ini config.ini        # PowerShell: Copy-Item config.example.ini config.ini

# 3) 前置服务：MySQL / Redis / Milvus 已启动
# 4) 模型：按下方「模型准备」下载后放到 rag_qa/models/

python app.py            # Flask   → http://127.0.0.1:5000
python app_async.py      # FastAPI → http://127.0.0.1:8000
```

启动顺序：FAQ 预热（jieba + BM25 索引，秒级）→ RAG 预热（后台加载 BGE-M3 / reranker，数十秒）→ 接收请求。

> `config.ini` 含 API Key，已被 `.gitignore` 忽略，不会进入仓库；仓库内只提供脱敏的 `config.example.ini`。

---

## 模型准备（不随仓库分发）

模型权重合计约 **5.4 GB**，体积过大未提交，需自行下载后按目录名放置到 `rag_qa/models/` 下。

| 目录 | 模型 | 用途 | 官方下载 |
|---|---|---|---|
| `bge-m3` | BAAI/bge-m3 | 混合检索的 dense + sparse 编码 | [HuggingFace](https://huggingface.co/BAAI/bge-m3) ｜ [ModelScope](https://modelscope.cn/models/BAAI/bge-m3) |
| `bge-reranker-large` | BAAI/bge-reranker-large | 检索结果精排（CrossEncoder） | [HuggingFace](https://huggingface.co/BAAI/bge-reranker-large) ｜ [ModelScope](https://modelscope.cn/models/BAAI/bge-reranker-large) |
| `bert-base-chinese` | google-bert/bert-base-chinese | 意图分类的底座模型 | [HuggingFace](https://huggingface.co/google-bert/bert-base-chinese) ｜ [ModelScope](https://modelscope.cn/models/google-bert/bert-base-chinese) |
| `nlp_bert_document-segmentation_chinese-base` | damo/nlp_bert_document-segmentation_chinese-base | 文档语义分割（备用切分器） | [ModelScope](https://modelscope.cn/models/damo/nlp_bert_document-segmentation_chinese-base) |
| `bert_query_classifier` | **本项目自训练** | 意图分类微调权重（通用知识 / 专业咨询） | 无官方下载，复现方式见下 |

**下载方式一：ModelScope（国内网络更稳）**

```bash
pip install -U modelscope
modelscope download --model BAAI/bge-m3 --local_dir rag_qa/models/bge-m3
modelscope download --model BAAI/bge-reranker-large --local_dir rag_qa/models/bge-reranker-large
modelscope download --model google-bert/bert-base-chinese --local_dir rag_qa/models/bert-base-chinese
```

**下载方式二：HuggingFace（需 git-lfs）**

```bash
git lfs install
git clone https://huggingface.co/BAAI/bge-m3 rag_qa/models/bge-m3
git clone https://huggingface.co/BAAI/bge-reranker-large rag_qa/models/bge-reranker-large
git clone https://huggingface.co/google-bert/bert-base-chinese rag_qa/models/bert-base-chinese
```

**复现意图分类权重 `bert_query_classifier`：**

```python
# 训练数据：rag_qa/data/model/model_generic_2000.json（仓库内已提供）
from rag_qa.core.query_classifier import QueryClassifier
clf = QueryClassifier()                       # 首次会从 bert-base-chinese 初始化
clf.train_model("rag_qa/data/model/model_generic_2000.json")
clf.save_model()                              # 输出到 models/bert_query_classifier
```

> 若暂时不训练，也可跳过：`rag_system.py` 会回退加载 `bert-base-chinese` 并随机初始化分类头，
> 分类结果因此不可用，表现是所有问题都按「专业咨询」走 RAG 流程。生产环境建议补齐该权重。

---

## 架构一眼看懂

```
浏览器 → 唯一前端（templates/ + static/）
        → app.py / app_async.py        入口层
        → qa_router.py                 路由层（唯一判定入口）
             ├─ FAQ  mysql_qa/  → BM25 → Redis 缓存 → MySQL jpkb
             └─ RAG  rag_qa/    → 混合检索 → 父块去重 → rerank → LLM 生成
        → chat_store（SQLite）落库 → 返回 JSON
```

| 链路 | 数据源 | 检索 | 生成 | 耗时 | 触发条件 |
|---|---|---|---|---|---|
| FAQ | MySQL `jpkb` | BM25 + softmax，最高分 > 0.85 | 直接取标准答案 | 毫秒 | `auto` 下永远先试 |
| RAG | Milvus `edurag_final` | BGE-M3 dense(0.7)+sparse(0.3) → 父块去重 → reranker Top2 | DashScope 生成 | 秒 | FAQ 未命中或 `mode=rag` |

`mode` 三态：`auto`（默认，FAQ 优先）/ `faq`（只查 FAQ）/ `rag`（只走 RAG）。

---

## 目录速查

| 路径 | 作用 |
|---|---|
| `app.py` / `app_async.py` | Flask / FastAPI 入口，只负责 HTTP 与落库 |
| `qa_router.py` | 统一路由：`answer_query`、`ask_faq`、`ask_rag`、预热、能力描述 |
| `templates/index.html`、`static/*` | 唯一前端：会话列表、模式切换、学科过滤、引擎徽标 |
| `base/` | `Config`（读 config.ini）+ 单例 `logger` |
| `mysql_qa/` | FAQ 能力包：`utils`（单例/组合器/会话存储）、`retrieval`、`db`、`cache` |
| `rag_qa/` | RAG 能力包：`rag_service`、`core`（检索/生成/切块/缓存）、`edu_*` 加载与切分 |
| `readme/` | 详细版、简略版、结构图、流程图 |

---

## 接口速查

`POST /query`

```json
// 请求
{"query": "AI 学科的课程大纲", "session_id": 3, "mode": "auto", "source": "ai"}

// 响应
{"code": 0, "session_id": 3, "title": "AI 学科的课程大纲", "answer": "……",
 "found": true, "engine": "rag", "elapsed": 3.2145, "note": "FAQ 未命中，已转 RAG"}
```

| 字段 | 说明 |
|---|---|
| `mode` | `auto` / `faq` / `rag`，非法值回退 `auto` |
| `source` | 学科过滤，仅 RAG 生效，取值见 `valid_sources` |
| `engine` | `faq` / `rag` / `none`，真正产出答案的链路 |
| `found` | `false` 时 `answer` 为兜底文案 |
| `note` | 过程说明，前端做悬浮提示 |

其余接口：`GET /`、`GET /api/meta`、`GET|POST /api/conversations`、`GET|PATCH|DELETE /api/conversations/{id}`、`GET /health`。

---

## 关键实现要点

**FAQ 链路**
- 单例 + 预热：`mysql_qa/utils/qa_service.get_qa_system()` / `ensure_ready()`。
- 索引：启动时 `select question from jpkb` → jieba 分词 → BM25；daemon 线程每 30 分钟重新同步。
- 判定：Redis `answer:{query}` 命中即返回；否则 BM25 打分 → softmax → 最高分 > 0.85 才算命中。
- 命中后用最相似的历史问题去 MySQL 取标准答案，并回写缓存。

**RAG 链路**
- 单例 + 失败冷却 60 s：Milvus 或模型不可用时快速失败，避免每请求重连。
- 意图分流：BERT 分类器（`models/bert_query_classifier` 微调权重）判断问题类型，「通用知识」不检索直接作答，「专业咨询」才进检索。
- 策略分派：LLM 选策略 → `strategy.strip()` → `retrieve_and_merge` 分派到回溯 / 子查询 / HyDE / 直接检索（不匹配时退化为直接检索）。
- 混合检索：dense 0.7 + sparse 0.3（`IP`，`k=retrieval_k`）→ 子块按 `parent_content` 去重还原父块 → `bge-reranker` 精排取 `candidate_m` 条。
- 生成：`rag_prompt(context, question, phone)` → DashScope；无可靠上下文时输出含客服电话的兜底句。

**会话存储**
- SQLite + WAL，库文件 `mysql_qa/data/chat_history.db`；线程本地连接 + 全局写锁。
- `add_exchange` 单事务写"一问一答"，顺带把默认标题自动改成首个问题。
- `messages.meta` 形如 `FAQ · 耗时 0.0123 s`，前端据此还原引擎徽标。

**异步版差异**
- 阻塞调用全部 `run_sync` 进线程池；三套 `CapacityLimiter`：FAQ=`max(4, CPU)`、RAG=2、DB=32。
- 超时：FAQ 30 s、RAG 180 s（环境变量可覆盖）。
- single-flight：`(question, mode, source)` 相同则同刻只算一次。

---

## 技术栈

| 层 | 组件 |
|---|---|
| 服务 | Python 3.12.14 · Flask 3.1.3 · FastAPI 0.143 · Uvicorn 0.54 · anyio 4.15 · Jinja2 3.1 |
| 检索 | jieba 0.42 · rank-bm25 0.2 · pymilvus 3.0（dense + sparse 混合检索） |
| 模型 | milvus-model（BGE-M3）· sentence-transformers 6.1（bge-reranker-large）· torch 2.7.1+cu126 · transformers 5.17 |
| 生成 | openai SDK 3.16 → DashScope 兼容接口（qwen3.7-plus） |
| **LangChain** | **langchain 1.4.4 · langchain-core 1.6.9 · langchain-community 0.4.2 · langchain-text-splitters 1.1.3** |
| 存储 | MySQL（jpkb）· Redis 8.1 · Milvus（edurag_final）· SQLite（chat_history.db） |
| 解析 | pymupdf · python-docx · python-pptx · RapidOCR · opencv |

**LangChain 用在基础抽象层**：`PromptTemplate`（提示词模板）、`Document` / `BaseLoader`（文档对象与加载器基类）、`RecursiveCharacterTextSplitter`（中文切分）、`TextLoader`（.txt 读取）。检索、重排、生成、路由均为自研代码，未使用 LCEL / Chain / Agent / Retriever。

---

## 配置速查（config.ini）

> 仓库内提供脱敏的 `config.example.ini`；`config.ini` 含 DashScope API Key，已被 `.gitignore` 忽略，请复制样例并填写自己的 Key。

| 段 | 关键项 | 当前值 |
|---|---|---|
| `[mysql]` | host / user / password / database | localhost / root / 123 / subjects_kg |
| `[redis]` | host / port / password / db | localhost / 6379 / 1234 / 0 |
| `[milvus]` | host / port / database_name / collection_name | localhost / 19530 / test5 / edurag_final |
| `[llm]` | model / dashscope_base_url | qwen3.7-plus / 私有兼容网关 |
| `[retrieval]` | parent 1200 / child 300 / overlap 50 / k 3 / m 2 | 切块与检索参数 |
| `[cache]` | cache_dir / verify_files / chunks | `rag_qa/data/.cache` / False / True |
| `[app]` | valid_sources / customer_service_phone | ai,java,test,ops,bigdata / 12345678 |

> `dashscope_api_key` 与 `customer_service_phone` **没有默认回退**，删掉会导致 `Config()` 直接报错。

---

## 缓存现状（四级缓存对照）

| 层级 | 是否实现 | 位置 | 说明 |
|---|---|---|---|
| 响应缓存 | **部分实现（仅 FAQ）** | Redis `answer:{query}`，读写都在 `BM25Search.search` | 精确匹配原始查询字符串，无 TTL、无失效；命中即跳过分词 + BM25 + MySQL |
| 语义缓存 | 未实现 | — | 换种说法问同一件事会完整重跑 |
| 检索缓存 | 未实现 | — | 每次请求都真实执行 Milvus 混合检索 |
| 嵌入缓存 | 未实现 | — | 每次请求都用 BGE-M3 重新编码 query |

**一句话结论：** FAQ 是"带精确响应缓存的链路"，RAG 是"每次全量重算的链路"（编码 → 检索 → 精排 → 生成四步全跑）。

注意区分：`rag_qa/data/.cache/` 下的 `catalog/documents/chunks` 三级 JSON 属于**离线建库缓存**（省 OCR 与切块），不是在线四级缓存；异步版的 single-flight 只合并同一时刻的并发请求，不能跨时间复用。

---

## 已落地的优化（摘要）

| 优化项 | 类型 | 做法 | 状态 |
|---|---|---|---|
| 重库延迟加载 | 启动 | `torch` / `pymilvus` / BGE 相关库改为函数内 import | 已落地 |
| 启动预热 | 启动 | FAQ 同步预热；RAG 后台加载并主动触发 BGE / reranker 权重 | 已落地 |
| 单例 + 失败冷却 | 稳定性 | 双重检查锁；初始化失败 60 s 内快速失败 | 已落地 |
| 离线三级缓存 | 建库 | `catalog` / `documents` / `chunks` + 三重校验 + 原子写 + 版本号 | 已落地 |
| FAQ 响应缓存 | 延迟 | Redis `answer:{query}`，命中直接返回 | 已落地（精确匹配、无 TTL） |
| 索引常驻 + 定时同步 | 延迟 | BM25 常驻内存；30 分钟同步 MySQL | 已落地 |
| 并发去重 | 并发 | single-flight，同刻相同 key 只算一次 | 已落地 |
| 线程池 + 三套闸门 | 并发 | FAQ / RAG / DB 分别限流 + 分层超时 | 已落地 |
| 父子块 + 混合检索 + 精排 | 检索 | 子块 300 检索、父块 1200 喂 LLM；dense 0.7 + sparse 0.3；rerank Top2 | 已落地 |
| 意图分类修正 | 检索 | 分类器指向 `models/bert_query_classifier`，加载微调权重 | 已落地 |
| 四策略分派 | 检索 | 回溯 / 子查询 / HyDE / 直接检索按 LLM 策略真正分派执行 | 已落地 |
| 通用知识直答 | 检索 | 通用知识不检索，直接由 LLM 作答 | 已落地 |
| 幻觉四道闸 | 答案质量 | FAQ 阈值直答 → 上下文收窄 → prompt 兜底句 → 出口空答案兜底 | 已落地 |
| 问答判定收口 | 工程化 | Flask / FastAPI 共用 `qa_router` 一份判定 | 已落地 |

详见 [详细版 §19 开发难点与优化实践](readme/开发文档-详细版.md)。

---

## 新增学科三步

1. 建目录 `rag_qa/data/{source}_data/`，放入原始文档（pdf/docx/pptx/txt/图片）。
2. `config.ini` 的 `valid_sources` 追加 `"{source}"`。
3. 建库：`python -c "from rag_qa.rag_main import main; main(query_mode=False, directory_path='rag_qa/data/test_data')"`，然后重启服务。

---

## 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 首个 RAG 请求特别慢 | BGE 模型懒加载 | 预热已在后台执行；未完成时耐心等待 |
| FAQ 全部不命中 | 语料变大后 softmax 最高分被稀释 | 重新标定阈值，或改用其他归一化方式 |
| RAG 一直未命中且 note 有 Milvus 报错 | 向量库不可用 | 检查 Milvus、库名 `test5` 与集合名 |
| 改了文件但建库结果没变 | 目录 mtime 未变（原地覆盖） | `[cache] verify_files = True` |
| MySQL 重启后 FAQ 持续报错 | 单连接无重连 | 重启服务（或补重连逻辑） |
| 前端学科选了 `ops` 检索为空 | 无对应 `ops_data` 语料 | 补语料或从配置移除 |

---

## 后续可添加 / 可修改（摘要）

| 位置 | 现状 | 可添加 / 可修改 |
|---|---|---|
| FAQ 阈值 | 固定 0.85，与语料规模强耦合 | 改打分方式（min-max / top1-top2 间隔），阈值移到 `config.ini` |
| MySQL 连接 | 单连接 + 锁，无重连 | 加 `ping(reconnect=True)` 或连接池 |
| LLM 客户端 | 两处各自封装，参数不一致 | 抽公共 `llm_client`，统一 temperature / 超时 / 重试 |
| 调试输出 | 已注释停用：`print(max_index)`、两处 `pass`、旧 if/elif 分派（代码保留） | 确认后可删除 |
| 学科配置 | `valid_sources` 含 `ops`，但无语料 | 补 `ops_data/` 或从配置移除 |
| 密钥管理 | `config.ini` 明文存 API Key | 改为环境变量注入 |
| 观测字段 | 只返回 `engine` / `elapsed` / `note` | 补 `cached`、分段耗时、trace id |
| 子查询清洗 | `split('\n')` 未过滤空行与序号，`is None` 判断恒假 | 清洗后判空再检索 |

完整说明见 [详细版 §15](readme/开发文档-详细版.md)。

---

## 致谢与技术栈

本项目建立在以下开源项目与云服务之上，感谢作者与社区的无私贡献。

### 编排与模型

| 项目 | 在本项目中的作用 | 链接 |
|---|---|---|
| **LangChain** | 提示词模板（`PromptTemplate`）、文档对象（`Document` / `BaseLoader`）、文本切分 | https://github.com/langchain-ai/langchain |
| **BGE / FlagEmbedding**（BAAI） | bge-m3 混合编码、bge-reranker-large 精排 | https://github.com/FlagOpen/FlagEmbedding |
| **sentence-transformers** | CrossEncoder 精排封装 | https://github.com/UKPLab/sentence-transformers |
| **Hugging Face Transformers** | BERT 意图分类 | https://github.com/huggingface/transformers |
| **PyTorch** | 模型推理运行时（CUDA 12.6） | https://pytorch.org |
| **ModelScope 魔搭** | 模型托管与下载 | https://github.com/modelscope/modelscope |
| **RapidOCR** | PDF / 图片 OCR | https://github.com/RapidAI/RapidOCR |

### 检索与存储

| 项目 | 在本项目中的作用 | 链接 |
|---|---|---|
| **Milvus** | 向量数据库（dense + sparse 混合检索） | https://github.com/milvus-io/milvus |
| **pymilvus** | Milvus Python SDK | https://github.com/milvus-io/pymilvus |
| **jieba** | 中文分词（FAQ 检索与文档处理） | https://github.com/fxsjy/jieba |
| **rank_bm25** | BM25Okapi 稀疏检索 | https://github.com/dorianbrown/rank_bm25 |
| **Redis / redis-py** | 问题列表与答案缓存 | https://github.com/redis/redis ｜ https://github.com/redis/redis-py |
| **MySQL / PyMySQL** | FAQ 标准问答对存储 | https://www.mysql.com ｜ https://github.com/PyMySQL/PyMySQL |
| **SQLite** | 会话与消息持久化 | https://www.sqlite.org |

### 服务与工程

| 项目 | 在本项目中的作用 | 链接 |
|---|---|---|
| **Flask** | 同步服务入口 | https://github.com/pallets/flask |
| **FastAPI** | 异步服务入口 | https://github.com/fastapi/fastapi |
| **Uvicorn** | ASGI 服务器 | https://github.com/encode/uvicorn |
| **anyio** | 线程池与并发闸门 | https://github.com/agronholm/anyio |
| **Jinja2** | 页面模板渲染 | https://github.com/pallets/jinja |
| **python-docx / python-pptx** | Word / PPT 解析 | https://github.com/python-openxml/python-docx ｜ https://github.com/scanny/python-pptx |
| **PyMuPDF** | PDF 解析与渲染 | https://github.com/pymupdf/PyMuPDF |
| **OpenCV** | 图像预处理（旋转校正等） | https://github.com/opencv/opencv |

### 大模型服务

| 项目 | 在本项目中的作用 | 链接 |
|---|---|---|
| **通义千问 Qwen / 阿里云百炼（DashScope）** | 检索策略选择与答案生成 | https://github.com/QwenLM/Qwen ｜ https://help.aliyun.com/zh/model-studio/ |
| **OpenAI Python SDK** | 以 OpenAI 兼容协议调用 DashScope | https://github.com/openai/openai-python |

> 各依赖的具体版本见 [`requirements.txt`](requirements.txt)（从 conda env `edurag` 导出，共 371 个包）。
