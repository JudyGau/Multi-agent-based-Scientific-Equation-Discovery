import http.client
from dataclasses import dataclass
from typing import List, Literal
import json

import llm


@dataclass
class PaperChunk:
    title: str
    authors: List[str]          # ["Zhou Z.H.", "Alpaydin E."]
    year: int | None            # 2021
    source: str                 # "Google Scholar" / "Wanfang"
    venue: str                  # 期刊名 / 出版社，如 "MIT Press"
    snippet: str                # 摘要或 snippet（SerpAPI 给的就是 snippet）
    citation_count: int | None  # 被引数，SerpAPI 有，万方也有
    metadata: dict              # 挂 doi / page / result_id / wanfang_id 等溯源用

import os, requests
from typing import List

def search_google_scholar(query: str, num: int = 5, language: str="en") -> List[PaperChunk]:

    # url = "https://scraperapi.dataify.com/request"
    # headers = {
    #     "Authorization": "Bearer 4knql00t3oea8bnboq2pp5g81fpd2p9e",
    #     "Content-Type": "application/x-www-form-urlencoded",
    # }
    # data = {
    #     "engine": "google_scholar",
    #     "q": query,
    #     "json": "1",
    #     "hl": language,
    #     "start": "0",
    #     "num": str(num),
    #     "scisbd": "0",
    #     "filter": "1",
    #     "as_vis": "0",
    #     "as_rr": "0",
    # }

    url = "https://google.serper.dev/scholar"

    payload = {
        "q": query,
        "type": "scholar",
        "num": 10,
        "page": 1,
        "engine": "google"
    }
    headers = {
        'X-API-KEY': 'ac28c1aac4d446f3de5c8e79ea6d406727509455',
        'Content-Type': 'application/json'
    }

    response =  json.loads( requests.request("POST", url, headers=headers, json=payload).text )

    print(response)

    # response = requests.post(url, headers=headers, data=data)
    # print(response.text)

    # """SerpAPI 谷歌学术，每月 100 次免费"""
    # r = requests.get("https://serpapi.com/search", params={
    #     "engine": "google_scholar",
    #     "q": query,
    #     "hl": "en",
    #     "num": num,
    #     "api_key": os.getenv("SERPAPI_KEY"),
    # }).json()

    chunks = []
    for p in response.get("organic", [])[:num]:
        authors = [a["name"] for a in p.get("publication_info", {}).get("authors", [])]
        cited = p.get("inline_links", {}).get("cited_by", {}).get("total")
        # summary = p["publication_info"]["summary"]  # "ZH Zhou - 2021 - books.google.com"
        summary = p["publication_info"]  # "ZH Zhou - 2021 - books.google.com"
        # 从 summary 抠 year（粗暴但够用，干净做法是用 author API 单独拉）
        yr = None
        for tok in summary.split(" - "):
            if tok.isdigit() and len(tok) == 4:
                yr = int(tok); break

        chunks.append(PaperChunk(
            title=p["title"],
            authors=authors,
            year=yr,
            source="Google Scholar",
            venue=summary.split(" - ")[-1] if " - " in summary else "",
            snippet=p.get("snippet", ""),
            citation_count=int(cited) if cited else None,
            metadata={
                "result_id": p["result_id"],
                "link": p["link"],
                "serpapi_cite_link": p["inline_links"]["serpapi_cite_link"],
            }
        ))
    return chunks

def search_wanfang(query: str, num: int = 5) -> List[PaperChunk]:
    """万方智搜题录，个人开发者走 AppCode 简认"""
    r = requests.post(
        "https://api.wanfangdata.com.cn/openwanfang/getQuery",
        params={"appCode": os.getenv("WANFANG_APP_CODE")},
        json={
            "query": query,
            "size": num,
            "start": 0,
        },
        headers={"Content-Type": "application/json"}
    ).json()

    chunks = []
    # 万方返回结构按开放平台文档，这里按 common case 解；实际字段名以你控制台文档为准
    for p in r.get("data", {}).get("docs", [])[:num]:
        chunks.append(PaperChunk(
            title=p.get("title", ""),
            authors=p.get("creators", []),
            year=p.get("publishYear"),
            source="Wanfang",
            venue=p.get("periodicalTitle", "") or p.get("sourceDbs", [""])[0],
            snippet=p.get("abstracts", ""),
            citation_count=p.get("citedCount"),
            metadata={
                "doi": p.get("doi", ""),
                "page": p.get("page", ""),
                "wanfang_id": p.get("id", ""),
            }
        ))
    return chunks

def search_literature(query: str, source: Literal["both","en","zh"] = "both", num: int = 5) -> str:
    """
    返回格式化字符串，给 DeepSeek 当 tool 结果塞回。
    每条带 [idx] 和 metadata 摘要，方便模型标引用。
    """
    en = search_google_scholar(query, num, "en") if source in ("both","en") else []
    zh = search_google_scholar(query, num, "zh") if source in ("both", "zh") else []
    # zh = search_wanfang(query, num) if source in ("both","zh") else []

    all_chunks = en + zh

    lines = []
    for i, c in enumerate(all_chunks[:num*2]):
        author_tag = c.authors[0].split()[-1] if c.authors else "Anon"
        year_tag = c.year or "?"
        meta_str = " | ".join(f"{k}={v}" for k,v in c.metadata.items() if v)
        lines.append(
            f"[{i+1}] 《{c.title}》{', '.join(c.authors)} {c.year or ''} {c.venue}\n"
            f"    被引{c.citation_count or '?'} | {c.snippet}\n"
            f"    meta: {meta_str}"
        )
    return "\n\n".join(lines)


# SERPER_KEY = "ac28c1aac4d446f3de5c8e79ea6d406727509455"
# # ── 工具 1：Serper Google Scholar ───────────────────────
# def search_seprer_google_scholar(query: str, num: int = 10) -> str:
#     """
#     调 Serper 的 Google Scholar API，
#     返回『已摘选结构化』的 JSON 字符串，方便模型消费。
#     """
#     conn = http.client.HTTPSConnection("google.serper.dev")
#     payload = json.dumps({"q": query, "num": min(num, 20)})
#     headers = {
#         "X-API-KEY": SERPER_KEY,
#         "Content-Type": "application/json",
#     }
#     conn.request("POST", "/scholar", payload, headers)
#     resp = conn.getresponse()
#     if resp.status != 200:
#         raise RuntimeError(f"Serper 报错: {resp.status}")
#     raw = json.loads(resp.read().decode())
#
#     # 只留 organic 里有用的字段（跟上一轮你贴的结构对齐）
#     cleaned = []
#     for item in raw.get("organic", []):
#         cleaned.append({
#             "title": item.get("title"),
#             "link": item.get("link"),
#             "publicationInfo": item.get("publicationInfo"),
#             "snippet": item.get("snippet"),
#             "year": item.get("year"),
#             "citedBy": item.get("citedBy"),
#             "pdfUrl": item.get("pdfUrl"),
#         })
#
#     return json.dumps(cleaned, ensure_ascii=False)


def chat_with_lit(user_q: str):
    messages = [
        {"role": "system", "content": (
            "你是研究助手。回答学术问题时：\n"
            "1. 必须先调 search_literature 查文献，不可凭记忆答研究类问题\n"
            "2. 回答中每处引用必须标成 [AuthorYear] 格式，如 [Zhou2021][Wang2023]，"
            "并在段末用 [1][2] 锚回 search_literature 返回的编号\n"
            "3. 若文献里找不到直接证据，明确说'现有检索结果未覆盖'，不要编\n"
            "4. 中文文献作者标拼音姓+年，英文标姓+年"
        )},
        {"role": "user", "content": user_q}
    ]

    # 第一轮：让模型决定调不调 tool
    resp = client.chat.completions.create(
        model="deepseek-chat",  # V3.2；有 V4 权限可换 deepseek-v4-flash
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    msg = resp.choices[0].message

    # 如果调了 tool，执行后回传
    if msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = search_literature(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })
        # 第二轮：基于文献答
        final = client.chat.completions.create(
            model="deepseek-chat", messages=messages
        )
        return final.choices[0].message.content

    return msg.content

if __name__ == "__main__":
    import json
    from openai import OpenAI

    client = OpenAI(base_url="https://api.deepseek.com", api_key="sk-3970c8c4922f49fd89761fe3ad4a5eb5")
    tools = [{
        "type": "function",
        "function": {
            "name": "search_literature",
            "description": "在中英文学术库（Google Scholar + 万方）按关键词检索题录与摘要，返回带出处的文献片段。用户问到学术/论文/研究类问题时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词，如'CRISPR 作物育种 脱靶效应'"},
                    "source": {"type": "string", "enum": ["both", "en", "zh"],
                               "description": "both=中英都查, en=仅谷歌学术, zh=仅万方"}
                },
                "required": ["query"]
            }
        }
    }]

    user_q = "磁流变液的磁流变效应与颗粒几何形态之间的关系"

    messages = [
        {"role": "system", "content": (
            "你是研究助手。回答学术问题时：\n"
            "1. 必须先调 search_literature 查文献，不可凭记忆答研究类问题\n"
            "2. 回答中每处引用必须标成 [AuthorYear] 格式，如 [Zhou2021][Wang2023]，"
            "并在段末用 [1][2] 锚回 search_literature 返回的编号\n"
            "3. 若文献里找不到直接证据，明确说'现有检索结果未覆盖'，不要编\n"
            "4. 中文文献作者标拼音姓+年，英文标姓+年"
        )},
        {"role": "user", "content": user_q}
    ]

    # 第一轮：让模型决定调不调 tool
    resp = client.chat.completions.create(
        model="deepseek-v4-pro",  # V3.2；有 V4 权限可换 deepseek-v4-flash
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    msg = resp.choices[0].message

    # 如果调了 tool，执行后回传
    if msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = search_literature(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })
        # 第二轮：基于文献答
        final = client.chat.completions.create(
            model="deepseek-chat", messages=messages
        )
        print( final.choices[0].message.content)

    print( msg.content)