import requests

def search_doi(query, rows=10):
    resp = requests.get(
        "https://api.crossref.org/works",
        params={"query.bibliographic": query, "rows": rows}
    )
    items = resp.json()["message"]["items"]
    return [
        {
            "doi": it.get("DOI"),
            "title": it.get("title", [""])[0],
            "year": (it.get("published-print") or it.get("created", {}))
                      .get("date-parts", [[None]])[0][0],
            "cited_by": it.get("is-referenced-by-count", 0)
        }
        for it in items
    ]