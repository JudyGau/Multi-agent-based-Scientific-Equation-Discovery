import requests

def search_paper_dois(query, rows=10, since_year="2020-01-01", mailto="you@example.com"):
    """
    只返回期刊论文(journal-article)的 DOI 及核心元数据
    """
    params = {
        "query": query,
        "filter": f"type:journal-article,from-pub-date:{since_year}",
        "sort": "is-referenced-by-count",
        "order": "desc",
        "rows": rows,
        "mailto": mailto,          # 进入 polite pool，更稳定
        "select": "DOI,title,author,container-title,published-print,is-referenced-by-count"
    }
    r = requests.get("https://api.crossref.org/works", params=params, timeout=30)
    r.raise_for_status()
    items = r.json()["message"]["items"]

    results = []
    for it in items:
        # 进一步保险：双重校验 type 字段
        if it.get("type") != "journal-article":
            continue
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
    return results

if __name__ == "__main__":
    # 用法
    papers = search_paper_dois("Retrieval-Augmented Generation", rows=5)
    for p in papers:
        print(f"{p['doi']}  |  {p['title'][:60]}  |  被引{p['citations']}")