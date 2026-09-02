import json
import sys

import requests
MAILTO = "zhuqg@mail.ustc.edu.cn"

def search_paper(query: str, num: int=10) -> str:
    """
    只返回期刊论文(journal-article)的 DOI 及核心元数据
    """
    params = {
        "query": query,
        "filter": "type:journal-article,type:proceedings-article,type:posted-content",  #包含期刊和会议论文，预印本，学术论文
        # "sort": "is-referenced-by-count",
        "order": "desc",
        "rows": num,
        "mailto": MAILTO,          # 进入 polite pool，更稳定
        "select": "DOI,URL,title,author,container-title,published-print,is-referenced-by-count,link,license"
    }
    r = requests.get("https://api.crossref.org/works", params=params, timeout=30)
    r.raise_for_status()
    items = r.json()["message"]["items"]

    results = []
    for it in items:
        # # 进一步保险：双重校验 type 字段
        # if it.get("type") != "journal-article":
        #     continue

        results.append({
            "doi": it.get("DOI"),
            "title": (it.get("title") or [""])[0],
            "journal": (it.get("container-title") or [""])[0],
            "year": ((it.get("published-print") or it.get("published-online") or {})
                     .get("date-parts", [[None]])[0][0]),
            "citations": it.get("is-referenced-by-count", 0),
            "authors": [f"{a.get('given','')} {a.get('family','')}"
                        for a in it.get("author", [])[:5]]
        })

    # 注意：经 MCP stdio 运行时 stdout 被改道且块缓冲，print 不可见；
    # 调试输出走 stderr 才能显示到控制台。
    print(results, file=sys.stderr)
    results = json.dumps(results, ensure_ascii=False)
    return results

if __name__ == "__main__":
    # 用法
    papers = search_paper("磁流变液", num=20)
    print(papers)
    # for p in papers:
    #     print(f"{p['doi']}  |  {p['title'][:60]}  |  被引{p['citations']}")