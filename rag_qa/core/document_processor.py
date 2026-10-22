# 这个脚本讲义的代码架构图没有体现，需要进行补充
# 重库都改成“用到时才 import”：
#   TextLoader     -> langchain_community
#   OCRxxxLoader   -> paddleocr / pdf2image / python-docx 等
#   ChineseRecursiveTextSplitter -> modelscope + transformers
# 这样只 import 本文件时几毫秒就能完成，不会一启动就把这些依赖全拉起来。
import importlib
import os
import sys
import time
from datetime import datetime

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f'current_dir--》{current_dir}')
# 获取core文件所在的目录的绝对路径
rag_qa_path = os.path.dirname(current_dir)
# print(f'rag_qa_path--》{rag_qa_path}')
sys.path.insert(0, rag_qa_path)
# 获取根目录文件所在的绝对位置
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)

from base import logger, Config

# core 目录本身也加入 sys.path，方便直接 import 同目录下的 json_cache
sys.path.insert(0, current_dir)
from json_cache import CACHE_VERSION, files_signature, load_catalog, normalize_source, read_json, write_json

conf = Config()

'''
    1. 解析器 -> 字典 ; 根据文件后缀获取指定的文档解析对象
    2. 加载文档
        2.1 根据指定目录遍历文件夹
        2.2 获取文件后缀
        2.3 选择解析器,解析文档
    3. 父子切块
        3.1 先进行父块切分
        3.2 再基于父块进行子块切分
        3.3 最后封装数据(父块,子块,父块ID....)
'''

# 定义支持的文件类型及其对应的加载器。
# 这里只登记“去哪儿找加载器”，不在模块导入时就去 import：
#   - OCRPDFLoader 会拉起 paddleocr / pdf2image
#   - OCRDOCLoader 会拉起 python-docx
#   - ChineseRecursiveTextSplitter 会拉起 modelscope + transformers
# 都是启动慢、占内存大的重库，具体用到哪个后缀再 import 哪个。
LOADER_REGISTRY = {
    # 文本文件使用 TextLoader
    ".txt": ("langchain_community.document_loaders", "TextLoader"),
    # PDF 文件使用 OCRPDFLoader
    ".pdf": ("edu_document_loaders", "OCRPDFLoader"),
    # Word 文件使用 OCRDOCLoader
    ".docx": ("edu_document_loaders", "OCRDOCLoader"),
    # PPT 文件使用 OCRPPTLoader
    ".ppt": ("edu_document_loaders", "OCRPPTLoader"),
    # PPTX 文件使用 OCRPPTLoader
    ".pptx": ("edu_document_loaders", "OCRPPTLoader"),
    # JPG 文件使用 OCRIMGLoader
    ".jpg": ("edu_document_loaders", "OCRIMGLoader"),
    # PNG 文件使用 OCRIMGLoader
    ".png": ("edu_document_loaders", "OCRIMGLoader"),
}

# 已经 import 过的加载器类缓存，同一种后缀只 import 一次
_loader_cache = {}
# Document 类缓存，第一次要构造文档对象时才 import
_document_cls = None


def get_loader(file_ext):
    """按后缀拿加载器类；不支持的后缀返回 None，第一次用到才 import"""
    if file_ext not in LOADER_REGISTRY:
        return None
    if file_ext not in _loader_cache:
        module_name, class_name = LOADER_REGISTRY[file_ext]
        _loader_cache[file_ext] = getattr(importlib.import_module(module_name), class_name)
    return _loader_cache[file_ext]


def get_document_class():
    """拿 Document 类（langchain_core 依赖 pydantic，也延迟加载）"""
    global _document_cls
    if _document_cls is None:
        from langchain_core.documents import Document
        _document_cls = Document
    return _document_cls

# ==================== JSON 缓存 ====================
# 思路：第一次运行把“目录清单 + 解析文本 + 切块结果”写成 JSON，
#      后面启动直接读 JSON，不再 os.walk 扫盘、不再打开 PDF/Word。
# 缓存目录由 config.ini 的 [cache] cache_dir 指定，默认 rag_qa/data/.cache
CACHE_DIR = conf.CACHE_DIR
# 目录清单：每个文件的路径/mtime_ns/size，用来替代 os.walk 遍历
CATALOG_FILE = os.path.join(CACHE_DIR, 'catalog.json')
# 解析结果：所有文件的正文合并在一个 JSON 里，一次读完，不用逐文件开缓存
DOCUMENTS_FILE = os.path.join(CACHE_DIR, 'documents.json')
# 切块结果：父子切块结果按“文件指纹 + 切块参数”缓存
CHUNKS_FILE = os.path.join(CACHE_DIR, 'chunks.json')
# 切块缓存最多保留几份（改过参数会生成新的，旧的不再无限堆积）
CHUNKS_CACHE_KEEP = 3


def _docs_from_cache(items):
    # 把 JSON 里的正文还原成 Document 对象
    Document = get_document_class()
    return [Document(page_content=item.get('page_content', '')) for item in items]


def _docs_to_cache(docs):
    # 正文落盘，source/file_path/timestamp 每次加载时重新补
    return [{'page_content': doc.page_content} for doc in docs]


def _chunks_to_json(child_chunks):
    # 父块正文只存一份，子块只记 parent_id，避免 JSON 体积膨胀好几倍
    parents = {}
    children = []
    for doc in child_chunks:
        metadata = dict(doc.metadata)
        parent_id = metadata.get('parent_id')
        parent_content = metadata.pop('parent_content', '')
        if parent_id and parents.get(parent_id) != parent_content:
            parents[parent_id] = parent_content
        children.append({'page_content': doc.page_content, 'metadata': metadata})
    return {'parents': parents, 'children': children}


def _chunks_from_json(entry):
    # 还原子块，并把父块正文写回 metadata['parent_content']
    Document = get_document_class()
    parents = entry.get('parents') or {}
    child_chunks = []
    for item in entry.get('children', []):
        metadata = dict(item.get('metadata') or {})
        metadata['parent_content'] = parents.get(metadata.get('parent_id'), '')
        child_chunks.append(Document(page_content=item.get('page_content', ''), metadata=metadata))
    return child_chunks


# 定义函数，从指定文件夹加载多种类型文件并添加元数据
def load_documents_from_directory(directory_path, catalog=None, verify_files=None) -> list:
    # 初始化空列表，用于存储加载的文档
    documents = []
    # 检查目录是否存在，避免静默返回空结果
    if not os.path.isdir(directory_path):
        logger.error(f'目录不存在或不是文件夹:{directory_path}')
        return documents

    # 是否逐个文件核对 mtime/size，默认取 config.ini 里的配置
    if verify_files is None:
        verify_files = conf.CACHE_VERIFY_FILES

    # 第一步：拿文件清单
    # 命中 catalog.json 时只有一次 JSON 读取 + 几次目录 stat，不再遍历全部文件
    if catalog is None:
        catalog, hit = load_catalog(directory_path, CATALOG_FILE)
        if hit:
            logger.info(f'命中目录清单缓存:{CATALOG_FILE}, 文件数:{len(catalog.get("files", []))}')
        else:
            logger.info(f'目录清单已刷新:{CATALOG_FILE}, 文件数:{len(catalog.get("files", []))}')

    # 第二步：一次性读出所有文件的解析结果（一个大 JSON，代替 N 个小文件）
    doc_payload = read_json(DOCUMENTS_FILE, default={}) or {}
    cached_files = doc_payload.get('files', {}) if doc_payload.get('version') == CACHE_VERSION else {}
    doc_cache_changed = False
    catalog_changed = False

    # 第三步：按清单逐个处理文件
    for item in catalog.get('files', []):
        file_path = item['file_path']
        file_ext = item['ext']
        # 按后缀拿加载器，第一次遇到这种类型才 import 对应重库
        loader_class = get_loader(file_ext)
        if loader_class is None:
            logger.warning(f'不支持的文件类型:{file_ext}, 已跳过:{file_path}')
            continue
        # 清单里记的 mtime/size
        mtime_ns, size = item['mtime_ns'], item['size']
        try:
            # 需要严格校验时，额外 stat 一下文件本身
            if verify_files:
                stat = os.stat(file_path)
                mtime_ns, size = stat.st_mtime_ns, stat.st_size
                if (mtime_ns, size) != (item['mtime_ns'], item['size']):
                    item['mtime_ns'], item['size'] = mtime_ns, size
                    catalog_changed = True

            entry = cached_files.get(file_path)
            # 原文件没改过（mtime 和 size 都一致）就直接用缓存正文
            if entry and entry.get('mtime_ns') == mtime_ns and entry.get('size') == size:
                docs = _docs_from_cache(entry.get('documents', []))
                logger.info(f'命中解析缓存:{file_path}, 文档数:{len(docs)}')
            else:
                docs = loader_class(file_path).load()
                cached_files[file_path] = {
                    'mtime_ns': mtime_ns,
                    'size': size,
                    'documents': _docs_to_cache(docs),
                }
                doc_cache_changed = True
                logger.info(f'成功加载文件:{file_path}, 文档数:{len(docs)}')

            # 补充元数据：学科标识、文件路径、解析时间
            data_source = normalize_source(directory_path)
            for doc in docs:
                doc.metadata['source'] = data_source
                doc.metadata['file_path'] = file_path
                doc.metadata['timestamp'] = datetime.now().isoformat()
            documents.extend(docs)
        except Exception as e:  # 单个文件失败不影响其余文件
            logger.error(f'加载文件失败:{file_path}, 错误:{e}')

    # 有变化才写回；没变化就一次写盘都没有
    if doc_cache_changed:
        write_json(DOCUMENTS_FILE, {
            'version': CACHE_VERSION,
            'root': catalog.get('root', os.path.abspath(directory_path)),
            'files': cached_files,
        })
        logger.info(f'解析结果已写入缓存:{DOCUMENTS_FILE}')
    if catalog_changed:
        write_json(CATALOG_FILE, catalog)

    return documents

# 定义函数，处理文档并进行分层切分，返回子块结果
def process_documents(directory_path, parent_chunk_size=conf.PARENT_CHUNK_SIZE,
                     child_chunk_size=conf.CHILD_CHUNK_SIZE,
                     chunk_overlap=conf.CHUNK_OVERLAP,
                     use_cache=conf.CACHE_CHUNKS):
    # 第一步：先拿目录清单，顺便算出“文件指纹”
    # 指纹一致说明源文件没动过，切块结果可以直接复用，省掉解析 + 切块的全部开销
    catalog, _ = load_catalog(directory_path, CATALOG_FILE)
    catalog = catalog or {}
    signature = (f'{files_signature(catalog.get("files", []))}'
                 f'|{parent_chunk_size}|{child_chunk_size}|{chunk_overlap}')

    # 第二步：命中切块缓存就直接返回
    entries = {}
    if use_cache:
        chunks_payload = read_json(CHUNKS_FILE, default={}) or {}
        if chunks_payload.get('version') == CACHE_VERSION:
            entries = chunks_payload.get('entries', {})
        entry = entries.get(signature)
        if entry:
            child_chunks = _chunks_from_json(entry)
            logger.info(f'命中切块缓存:{CHUNKS_FILE}, 子块数:{len(child_chunks)}')
            return child_chunks

    # 第三步：没命中就正常走一遍 加载 -> 父块切分 -> 子块切分
    # 从指定目录加载所有文档（复用上面已经拿到的目录清单，不再重复扫描）
    docs = load_documents_from_directory(directory_path, catalog=catalog)
    # 记录加载的文档总数日志
    logger.info(f'文档总数为:{len(docs)}') # type: ignore
    # 切块器依赖 modelscope + transformers，很重，只有真正要切块时才 import
    from edu_text_spliter import ChineseRecursiveTextSplitter
    # 初始化父块和子块分割器
    parent_spliter = ChineseRecursiveTextSplitter(chunk_size = parent_chunk_size,chunk_overlap = chunk_overlap)
    child_spliter = ChineseRecursiveTextSplitter(chunk_size = child_chunk_size,chunk_overlap = chunk_overlap)
    # 初始化空列表，用于存储所有子块
    child_chunks = []
    #遍历每个原始文档，带上索引 i
    for i, doc in enumerate(docs):
        # 使用父块切分器将文档切分为父块
        parent_docs = parent_spliter.split_documents([doc])
        # 遍历每个父块，带上索引 j
        for j , parent_doc in enumerate(parent_docs):
            # 为父块生成唯一 ID，格式为 "doc_i_parent_j"
            parent_id = f'doc{i}_parent{j}'
            # 使用子块切分器将父块切分为子块
            child_docs = child_spliter.split_documents([parent_doc])
            # 遍历每个子块，为子块主要添加对应的父块文档
            for k, child_doc in enumerate(child_docs):
                # 为子块添加父块的ID
                child_doc.metadata['parent_id'] = parent_id
                # 为子块添加对应的父块文档
                child_doc.metadata['parent_content'] = parent_doc.page_content

                # 为子块生成一个唯一的ID,格式为 "parent_id_child_k"
                child_doc.metadata['child_id'] = f'{parent_id}_child_{k}'
                # 将子块添加到子块列表中
                child_chunks.append(child_doc)

    # 第四步：把结果写回缓存，下次同参数同文件直接复用
    if use_cache:
        entries[signature] = _chunks_to_json(child_chunks)
        # 只保留最近几份，避免反复改参数把缓存撑大
        while len(entries) > CHUNKS_CACHE_KEEP:
            entries.pop(next(iter(entries)))
        write_json(CHUNKS_FILE, {'version': CACHE_VERSION, 'entries': entries})
        logger.info(f'切块结果已写入缓存:{CHUNKS_FILE}')

    # 记录子块总数日志
    logger.info(f'子块文档总数是:{len(child_chunks)}')
    # 返回所有子块列表
    return child_chunks

if __name__ == '__main__':
    start_time = time.time()
    directory_path = os.path.join(rag_qa_path, 'data', 'ai_data')
    logger.info(f'开始处理目录:{directory_path}')
    # print(load_documents_from_directory(directory_path))
    process_documents(directory_path)
    print(f'cost time:{time.time() - start_time}')