# -*- coding:utf-8 -*-
'''
    统一的 JSON 缓存工具

    目的：把"遍历目录 / 打开原始文件 / 切块"的结果落盘成 JSON，
         下次启动直接读 JSON，不再重复做磁盘 IO。

    缓存文件（默认放在 rag_qa/data/.cache/ 下）：
        catalog.json    目录清单：文件路径 + mtime_ns + size，替代 os.walk 遍历
        documents.json  所有文件的解析文本，替代逐个打开 PDF/Word 做 OCR
        chunks.json     父子切块结果，替代重复切块

    有效性判断：
        1. 目录清单：比对每个目录的 mtime_ns。目录下增删/改名文件会改目录 mtime，
           所以 mtime 没变就直接用清单，不再 os.walk 扫盘。
           注意：文件内容被覆盖不会改目录 mtime，需要逐文件核对时把 verify_files 打开。
        2. 解析结果：比对单个文件的 mtime_ns + size，变了才重新解析。
        3. 切块结果：用"文件指纹 + 切块参数"算一个签名，签名一致直接复用。
'''
import hashlib
import json
import os

# 缓存结构版本：字段改了就把版本号 +1，旧缓存自动作废，避免读到不兼容的数据
CACHE_VERSION = 2


def read_json(path, default=None):
    """读 JSON。文件不存在或内容损坏时返回 default，不抛异常"""
    if not path or not os.path.isfile(path):
        return default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, payload):
    """
    原子写 JSON：先写 .tmp，再 os.replace 替换。
    这样即使写到一半程序崩了，也不会留下半个坏文件，下次还能读到旧缓存。
    """
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        # 不加 indent：缓存文件可能几 MB，缩进会让体积翻倍
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)


def normalize_source(root_path):
    """
    规范化数据源标识：如 ai_data -> ai
    可根据实际需求扩展
    """
    # 示例实现：去掉 _data 后缀
    base = os.path.basename(os.path.normpath(root_path))
    if base.endswith('_data'):
        return base[:-5]
    return base


def scan_directory(directory_path):
    """
    真正遍历目录一次（只在缓存失效或强制刷新时调用）
    返回: {'version','root','dir_mtimes','files'}
         dir_mtimes: {相对目录: mtime_ns}  用于快速判断目录有没有增删文件
         files     : [{file_path, rel_path, ext, mtime_ns, size, source}, ...]
    """
    root_path = os.path.abspath(directory_path)
    # 数据源标识：ai_data -> ai，后续写入向量库的 source 字段
    source = normalize_source(root_path)
    files = []
    dir_mtimes = {}
    # 遍历指定目录及其子目录
    for root, dirs, names in os.walk(root_path):
        # 排序保证每次扫描顺序一致，便于复现问题
        dirs.sort()
        names.sort()
        rel_root = os.path.relpath(root, root_path)
        # 目录数量远少于文件数量，记录目录 mtime 的成本很低
        dir_mtimes[rel_root] = os.stat(root).st_mtime_ns
        for name in names:
            # 跳过 Office 打开文件时生成的临时文件（如 ~$xxx.docx）
            if name.startswith('~$'):
                continue
            file_path = os.path.join(root, name)
            stat = os.stat(file_path)
            files.append({
                'file_path': os.path.abspath(file_path),
                'rel_path': os.path.normpath(os.path.join(rel_root, name)),
                # 扩展名统一转小写，避免 .PDF/.JPG 等被误判为不支持
                'ext': os.path.splitext(name)[1].lower(),
                'mtime_ns': stat.st_mtime_ns,
                'size': stat.st_size,
                'source': source,
            })
    return {
        'version': CACHE_VERSION,
        'root': root_path,
        'dir_mtimes': dir_mtimes,
        'files': files,
    }


def catalog_is_valid(catalog, directory_path):
    """目录清单还能不能用：版本、根目录、每个子目录的 mtime_ns 都对得上才算命中"""
    if not catalog or catalog.get('version') != CACHE_VERSION:
        return False
    if catalog.get('root') != os.path.abspath(directory_path):
        return False
    recorded = catalog.get('dir_mtimes') or {}
    current = {}
    # 只 stat 目录，不 stat 文件、不打开文件，所以很快
    for root, dirs, _ in os.walk(directory_path):
        rel_root = os.path.relpath(root, directory_path)
        current[rel_root] = os.stat(root).st_mtime_ns
    return current == recorded


def load_catalog(directory_path, catalog_file, force_rescan=False) -> tuple[dict, bool]:
    """
    获取目录清单：能命中缓存就读 JSON，命中不了才真正遍历一次并写回 JSON
    返回 (catalog, hit) ，hit 表示这次是否命中缓存
    """
    if not force_rescan:
        cached = read_json(catalog_file)
        if isinstance(cached, dict) and catalog_is_valid(cached, directory_path):
            return cached, True
    catalog = scan_directory(directory_path)
    write_json(catalog_file, catalog)
    return catalog, False


def files_signature(files):
    """
    文件清单指纹：路径 + mtime_ns + size 拼起来做 md5。
    文件增删改任何一个，指纹都会变，切块缓存随之失效。
    """
    raw = '|'.join(f"{item.get('rel_path')}:{item.get('mtime_ns')}:{item.get('size')}" for item in files)
    return hashlib.md5(raw.encode('utf-8')).hexdigest()
