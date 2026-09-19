import json
import re
import sys

import requests
MAILTO = "zhuqg@mail.ustc.edu.cn"

#: 返回的摘要字符上限。实测模型只能靠标题判断相关性，判不出来就换措辞反复重搜
#: （单轮实验 54 次 search_paper、0 次 read_paper）；给一段摘要它才有依据决定"读哪篇"。
#: 摘要在提示词里是纯成本，600 字符够判断主题是否对得上，又不至于把结果撑爆。
_ABSTRACT_CHAR_LIMIT = 600


def _clean_abstract(raw) -> str | None:
    """清洗 Crossref 的 abstract 并截断。

    Crossref 返回的 abstract 常常是 JATS 片段（``<jats:p>...</jats:p>``，还带
    ``<jats:italic>`` 之类的内联标签），直接塞进提示词会让模型读到一堆尖括号；
    这里去标签、压空白后按上限截断。没有摘要（Crossref 大量条目如此）时返回 None。
    """
    if not raw:
        return None
    text = re.sub(r"<[^>]+>", " ", str(raw))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    if len(text) > _ABSTRACT_CHAR_LIMIT:
        text = text[:_ABSTRACT_CHAR_LIMIT].rstrip() + "..."
    return text


def search_paper(query: str, num: int=3) -> str:
    """按关键词检索论文，返回 DOI、标题、摘要等元数据（JSON 字符串）。

    ``num`` 默认 3 而不是先前的 10：每条结果现在带一段摘要，条数过多只是把
    提示词预算花在无关条目上；配合"不要换措辞反复重搜"的提示词规则，
    少量高相关信息更有用。
    """
    params = {
        "query": query,
        "filter": "type:journal-article,type:proceedings-article,type:posted-content",  #包含期刊和会议论文，预印本，学术论文
        # "sort": "is-referenced-by-count",
        "order": "desc",
        "rows": num,
        "mailto": MAILTO,          # 进入 polite pool，更稳定
        # 刻意不传 select：Crossref 的 select 白名单**不含 abstract**——写了也会被静默
        # 忽略（实测带 select 时返回的 item 键里没有 abstract，值恒为 None），而摘要正是
        # 判断相关性的唯一依据。改为取完整记录后只在 Python 侧抽取需要的字段，
        # 因此提示词开销不变，只是 HTTP 响应体大一点（rows 默认 3，量级可忽略）。
    }
    r = requests.get("https://api.crossref.org/works", params=params, timeout=30)
    r.raise_for_status()
    items = r.json()["message"]["items"]

    results = []
    for it in items:
        # # 进一步保险：双重校验 type 字段
        # if it.get("type") != "journal-article":
        #     continue

        # 年/月/日可能缺失（如 date-parts: [[]]），直接 [0][0] 会 IndexError 拖垮整个查询；
        # 链尾补 issued：Crossref 大量条目只有 issued 有日期（实测 ~5% 命中缺 print/online）
        dp = ((it.get("published-print") or it.get("published-online") or it.get("issued") or {})
              .get("date-parts") or [[]])
        year = dp[0][0] if dp and dp[0] else None

        results.append({
            "doi": it.get("DOI"),
            "title": (it.get("title") or [""])[0],
            "journal": (it.get("container-title") or [""])[0],
            "year": year,
            "citations": it.get("is-referenced-by-count", 0),
            "authors": [f"{a.get('given','')} {a.get('family','')}"
                        for a in it.get("author", [])[:5]],
            # 摘要可能为 None（Crossref 很多条目没有）：仍是固定键，
            # 便于下游/测试按固定 schema 解析。
            "abstract": _clean_abstract(it.get("abstract")),
        })

    # 注意：经 MCP stdio 运行时 stdout 被改道且块缓冲，print 不可见；
    # 调试输出走 stderr 才能显示到控制台。
    # print(results, file=sys.stderr)

    results = json.dumps(results, ensure_ascii=False)
    return results

if __name__ == "__main__":
    # 用法
    papers = search_paper("磁流变液", num=20)
    print(papers)
    # for p in papers:
    #     print(f"{p['doi']}  |  {p['title'][:60]}  |  被引{p['citations']}")