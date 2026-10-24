# mysql_qa/utils/smalltalk.py
# 身份 / 寒暄类问题的直答规则。
#
# 为什么单独拎出来：
#   这类问题（你是谁、在吗、谢谢）没有知识库检索的价值，走 FAQ 必然未命中，
#   再转 RAG 会让大模型自由发挥，既慢又容易胡编。用规则表直接命中即可，
#   耗时从秒级降到微秒级，也不会污染检索日志。
#
# 怎么改成自己公司的口径：
#   1) 改 COMPANY_NAME 为贵公司名称（或用环境变量注入）；
#   2) 按需改写下面 RULES 里的 reply —— 这些文案就是「贵公司的开场白」。
import os
import re
from typing import Dict, List, Optional

# 品牌名占位：上线前请替换成贵公司名称
COMPANY_NAME = os.getenv("SMALLTALK_COMPANY", "【贵公司名称】")

# 话题前缀：允许「请问你是谁」这类礼貌说法，先剥掉再匹配
_POLITE_PREFIX = re.compile(r"^(请问|请问一下|请|麻烦问一下|问一下|那个|嗯|你好)[，,、\s]*")

# 句尾标点：判等时去掉，避免「你是谁？」「谢谢！」漏命中
_TRAILING_PUNCT = re.compile(r"[。！？!?.,，、~～\s]+$")

# 问题太长就不当寒暄处理
MAX_LEN = 20

# 规则表：pattern 用全句等值匹配（见 match_smalltalk），
# 只命中「整句就是一个寒暄/身份问题」的情况，避免「你是谁开发的算法」被误伤。
# 文案即「贵公司的开场白」，按需改写。
RULES: List[Dict[str, str]] = [
    {
        # 开场白 —— 请修改成贵公司的问候语
        "pattern": r"(你好|您好|哈喽|嗨|hi|hello|早上好|中午好|下午好|晚上好)(呀|啊|呢|哦|啦)?",
        "reply": f"你好！我是 {COMPANY_NAME} 的智能问答助手，专注为你解答学习相关的问题，请直接输入你的问题～",
    },
    {
        # 身份 —— 请修改成贵公司的自我介绍
        "pattern": r"(你是谁|您是谁|你叫什么|你叫什么名字|你的名字|你是什么|who\s*are\s*you|"
                   r"介绍一下你自己|做个自我介绍|你是干什么的)(呀|啊|呢)?",
        "reply": f"我是 {COMPANY_NAME} 的智能问答助手，可以为你解答课程与学习相关的问题，也支持多轮追问。",
    },
    {
        "pattern": r"(在吗|在不在|有人吗|在么)(呀|啊|呢)?",
        "reply": f"我在的，我是 {COMPANY_NAME} 的智能问答助手，有什么问题直接问我就好～",
    },
    {
        "pattern": r"(你能做什么|你会什么|你有什么功能|有什么功能|能帮我什么|怎么用|使用说明|帮助)(呀|啊|呢)?",
        "reply": "我可以做两件事：① 直接命中常见问题的标准答案；② 没有现成答案时，检索课程资料并由大模型组织回答。"
                 "支持指定学科范围，也能记住同一会话里的上下文。",
    },
    {
        "pattern": r"(你是机器人|你是人工智能|你是人吗|你是ai|你是真人)(吗|嘛|么)?",
        "reply": f"我是 {COMPANY_NAME} 部署的智能问答助手，不是真人客服；遇到我答不好的问题，可以转人工咨询。",
    },
    {
        "pattern": r"(谁开发|谁做的|哪家公司|你们公司|你们是谁)(的)?",
        "reply": f"本系统由 {COMPANY_NAME} 提供与维护，基于检索增强生成（RAG）技术搭建。",
    },
    {
        "pattern": r"(谢谢|感谢|多谢|thanks|thank\s*you|辛苦了)(你|您|啦|了)?",
        "reply": "不客气～ 还有其他问题随时问我。",
    },
    {
        "pattern": r"(再见|拜拜|bye|goodbye|下次聊)(啦|了)?",
        "reply": "再见！需要的时候随时回来找我～",
    },
]


def match_smalltalk(question: str) -> Optional[str]:
    """
    判断是否为身份 / 寒暄类问题。

    判定方式：剥掉「请问」类前缀与句尾标点后，与规则做**全句**匹配 ——
    只有整句就是一句寒暄/身份提问时才直答，带具体诉求的长句仍走检索链路。

    :param question: 用户原始问题（调用方需保证非空）
    :return: 命中的直答文案；不命中返回 None，调用方继续走检索链路
    """
    text = " ".join((question or "").split())
    if not text or len(text) > MAX_LEN:
        return None

    # 依次尝试：原句 → 原句去句尾标点 → 剥掉礼貌前缀后再去标点
    # 保留原句是为了让「你好呀」这类带语气词的说法也能命中
    candidates = [_TRAILING_PUNCT.sub("", text)]
    stripped = _POLITE_PREFIX.sub("", text)
    if stripped and stripped != text:
        candidates.append(_TRAILING_PUNCT.sub("", stripped))

    for candidate in candidates:
        if not candidate:
            continue
        for rule in RULES:
            if re.fullmatch(rule["pattern"], candidate, re.IGNORECASE):
                return rule["reply"]
    return None


if __name__ == "__main__":
    # 自测：确认命中与不误伤
    samples = [
        "你是谁",
        "请问你是谁？",
        "你好",
        "你好呀",
        "在吗",
        "你能做什么",
        "谢谢",
        "谢谢！",
        "再见",
        "who are you",
        "Bert 的注意力机制怎么实现",        # 不应命中
        "你是谁开发的这个算法怎么实现",      # 不应命中（带具体诉求）
        "你好，帮我看看这段代码",            # 不应命中
        "谢谢，再问一个问题",                # 不应命中
    ]
    for q in samples:
        hit = match_smalltalk(q)
        print(f"{q:<24} -> {'直答: ' + hit if hit else '未命中，继续走检索'}")
