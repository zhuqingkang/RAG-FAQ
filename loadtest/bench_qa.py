# loadtest/bench_qa.py
# 集成问答系统压测脚本：分别测 FAQ 链路与 RAG 链路的延迟分位数（p50/p90/p95/p99）
#
# 用法（在项目根目录下执行）：
#   python loadtest/bench_qa.py --scenario faq --requests 300 --concurrency 8
#   python loadtest/bench_qa.py --scenario rag --requests 40  --concurrency 2
#   python loadtest/bench_qa.py --scenario faq_hot --requests 200 --concurrency 8
#
# 说明：
#   1) 走真实 HTTP 接口 POST /query，测的是端到端延迟（RTT）。
#   2) 同时记录服务端自报的 elapsed 字段，用于拆分「框架/落库开销」与「检索生成开销」。
#   3) faq 场景从 MySQL knowledge 表读取真实问题，每个问题只请求一次，尽量避开 Redis 缓存；
#      faq_hot 场景固定同一问题，命中 Redis 缓存，用于对比冷热路径。
#   4) rag 场景强制 mode=rag，绕过 FAQ。
import argparse
import configparser
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
import pymysql
import requests

BASE_DIR = Path(__file__).resolve().parent          # .../integrated_qa_system/loadtest
PROJECT_DIR = BASE_DIR.parent                       # .../integrated_qa_system
RESULT_DIR = BASE_DIR / "results"

# FAQ 场景：能从 knowledge 读到的真实问题不够时，循环复用
# RAG 场景：专业咨询类问题，会走「意图分类 -> 检索 -> 精排 -> 生成」完整链路
RAG_QUESTIONS = [
    "请详细介绍Transformer中自注意力机制的计算过程",
    "解释一下BERT的MLM和NSP两个预训练任务的区别",
    "大模型微调中LoRA的原理是什么，为什么能省显存",
    "简述知识图谱在智能教育系统里的典型应用",
    "向量数据库的ANN近似检索是怎么实现的",
    "RAG系统里为什么要在召回之后再加一层重排序",
    "训练深度网络时梯度消失问题有哪些常见解决办法",
    "对比RNN和CNN处理文本序列时的优缺点",
    "什么是few-shot prompting，和微调有什么区别",
    "请说明BM25算法的打分原理和它的局限",
    "教育领域做知识点推荐一般用哪些召回策略",
    "如何评估一个问答系统的答案质量",
]


def read_mysql_config():
    """从项目根目录的 config.ini 读取 MySQL 连接参数。"""
    conf = configparser.ConfigParser()
    conf.read(PROJECT_DIR / "config.ini", encoding="utf-8")
    return {
        "host": conf.get("mysql", "host"),
        "user": conf.get("mysql", "user"),
        "password": conf.get("mysql", "password"),
        "database": conf.get("mysql", "database"),
        "charset": "utf8mb4",
    }


def load_faq_questions():
    """从 knowledge 表取全部问题，作为 FAQ 压测的输入集。"""
    conn = pymysql.connect(**read_mysql_config())
    try:
        with conn.cursor() as cursor:
            cursor.execute("select question from knowledge;")
            rows = cursor.fetchall()
    finally:
        conn.close()
    return [row[0] for row in rows if row[0]]


def build_jobs(scenario, total, variant=False):
    """构造待发请求队列：元素为 (序号, 问题文本)。"""
    if scenario in ("faq", "faq_hot"):
        questions = load_faq_questions()
        if not questions:
            raise RuntimeError("knowledge 表没有数据，无法进行 FAQ 压测")
        if scenario == "faq_hot":
            # 固定问同一个问题：第一次之后全部命中 Redis 缓存
            questions = [questions[0]]
        elif variant:
            # 改写问题文本（统一换成全角问号）：key 变新，确保走 BM25 而不是 Redis 缓存
            questions = [q.rstrip("？?") + "？" for q in questions]
    else:
        questions = RAG_QUESTIONS

    print(f"[准备] 场景={scenario} 问题池={len(questions)} 条 目标请求数={total}")
    return [(i, questions[i % len(questions)]) for i in range(total)]


def worker(job_queue, records, lock, args):
    """单个压测线程：循环取问题 -> 发请求 -> 记录一行结果。"""
    session = requests.Session()

    while True:
        try:
            index, question = job_queue.get_nowait()
        except queue.Empty:
            return

        payload = {"query": question, "mode": args.mode}
        t0 = time.perf_counter()
        row = {"index": index, "question": question}
        try:
            resp = session.post(f"{args.base_url}/query", json=payload, timeout=args.timeout)
            row["rtt"] = time.perf_counter() - t0
            row["status"] = resp.status_code
            body = resp.json()
            row["server_elapsed"] = body.get("elapsed")
            row["engine"] = body.get("engine")
            row["found"] = body.get("found")
            row["note"] = body.get("note", "")
            if resp.status_code != 200:
                row["error"] = str(body)[:200]
        except Exception as e:
            row["rtt"] = time.perf_counter() - t0
            row["status"] = -1
            row["server_elapsed"] = None
            row["engine"] = None
            row["found"] = False
            row["error"] = f"{type(e).__name__}: {e}"

        with lock:
            records.append(row)
        job_queue.task_done()


def stats(values):
    """返回一组延迟的分位数统计（单位：秒）。"""
    arr = np.array(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std(ddof=1)), 4) if arr.size > 1 else 0.0,
        "min": round(float(arr.min()), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "p99": round(float(np.percentile(arr, 99)), 4),
        "max": round(float(arr.max()), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="FAQ / RAG 链路压测")
    parser.add_argument("--scenario", choices=["faq", "faq_hot", "rag"], required=True)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--variant", action="store_true",
                        help="FAQ 场景改写问题文本，绕开 Redis 答案缓存，测纯 BM25 路径")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    args.mode = "faq" if args.scenario in ("faq", "faq_hot") else "rag"

    # 健康检查：服务没起就直接退出，避免打出一堆连接错误
    try:
        health = requests.get(f"{args.base_url}/health", timeout=5).json()
        print(f"[检查] 服务健康状态: {health}")
    except Exception as e:
        raise SystemExit(f"服务不可用: {e}")

    # 预热：不计入统计，主要避免首次请求的惰性初始化/缓存抖动
    print(f"[预热] 发送 {args.warmup} 条请求 ...")
    warmup_jobs = build_jobs(args.scenario, args.warmup)
    for _, question in warmup_jobs:
        try:
            requests.post(f"{args.base_url}/query",
                          json={"query": question, "mode": args.mode}, timeout=args.timeout)
        except Exception as e:
            print(f"  预热请求失败: {e}")

    jobs = build_jobs(args.scenario, args.requests, variant=args.variant)
    job_queue = queue.Queue()
    for job in jobs:
        job_queue.put(job)

    records = []
    lock = threading.Lock()
    threads = [
        threading.Thread(target=worker, args=(job_queue, records, lock, args), daemon=True)
        for _ in range(args.concurrency)
    ]

    print(f"[执行] 并发={args.concurrency} 请求={args.requests} 模式={args.mode}")
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_start

    records.sort(key=lambda r: r["index"])

    # 失败请求与「未命中」要分开统计：未命中是正常业务结果，不是错误
    ok_rows = [r for r in records if r["status"] == 200]
    err_rows = [r for r in records if r["status"] != 200]
    found_rows = [r for r in ok_rows if r.get("found")]

    rtt_all = [r["rtt"] for r in ok_rows]
    rtt_found = [r["rtt"] for r in found_rows]
    server_all = [r["server_elapsed"] for r in ok_rows if r.get("server_elapsed") is not None]

    result = {
        "scenario": args.scenario,
        "mode": args.mode,
        "tag": args.tag,
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "wall_time_s": round(wall, 3),
        "qps": round(len(ok_rows) / wall, 2) if wall > 0 else 0,
        "success": len(ok_rows),
        "errors": len(err_rows),
        "hit_rate": round(len(found_rows) / len(ok_rows), 4) if ok_rows else 0,
        "rtt_all": stats(rtt_all) if rtt_all else None,
        "rtt_found": stats(rtt_found) if rtt_found else None,
        "server_elapsed_all": stats(server_all) if server_all else None,
        "error_samples": [r.get("error", "") for r in err_rows[:3]],
        "engine_mix": {e: sum(1 for r in ok_rows if r.get("engine") == e)
                       for e in {r.get("engine") for r in ok_rows}},
        "raw": records,
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{args.scenario}{('_' + args.tag) if args.tag else ''}_c{args.concurrency}_n{args.requests}"
    out_path = RESULT_DIR / f"{name}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    def fmt(row, label):
        if not row:
            return f"  {label}: 无数据"
        return (f"  {label}: n={row['count']} mean={row['mean']}s p50={row['p50']}s "
                f"p90={row['p90']}s p95={row['p95']}s p99={row['p99']}s max={row['max']}s")

    print("\n================ 压测结果 ================")
    print(f"场景={args.scenario} 并发={args.concurrency} 请求={args.requests}")
    print(f"总耗时={round(wall, 2)}s  吞吐={result['qps']} req/s  成功={len(ok_rows)} 失败={len(err_rows)}")
    print(f"命中率={result['hit_rate']}  引擎分布={result['engine_mix']}")
    print(fmt(result["rtt_all"], "RTT(全部成功请求)"))
    print(fmt(result["rtt_found"], "RTT(仅命中答案)"))
    print(fmt(result["server_elapsed_all"], "服务端自报耗时"))
    if result["error_samples"]:
        print(f"错误示例: {result['error_samples']}")
    print(f"明细已写入: {out_path}")


if __name__ == "__main__":
    main()
