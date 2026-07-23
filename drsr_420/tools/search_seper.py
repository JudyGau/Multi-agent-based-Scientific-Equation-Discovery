import os
import json
import http.client
from dotenv import load_dotenv
from openai import OpenAI

from drsr_420.tools.tools_description import tools

load_dotenv()

# ── 客户端 ──────────────────────────────────────────────
deepseek = OpenAI(
    api_key="sk-3970c8c4922f49fd89761fe3ad4a5eb5",
    base_url="https://api.deepseek.com",        # 兼容 OpenAI 格式
)

SERPER_KEY = "ac28c1aac4d446f3de5c8e79ea6d406727509455"


# ── 工具 1：Serper Google Scholar ───────────────────────
def search_google_scholar(query: str, num: int = 10) -> str:
    """
    调 Serper 的 Google Scholar API，
    返回『已摘选结构化』的 JSON 字符串，方便模型消费。
    """
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({"q": query, "num": min(num, 20)})
    headers = {
        "X-API-KEY": SERPER_KEY,
        "Content-Type": "application/json",
    }
    conn.request("POST", "/scholar", payload, headers)
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"Serper 报错: {resp.status}")
    raw = json.loads(resp.read().decode())

    # 只留 organic 里有用的字段（跟上一轮你贴的结构对齐）
    cleaned = []
    for item in raw.get("organic", []):
        print(item+'\n')

        cleaned.append({
            "title": item.get("title"),
            "link": item.get("link"),
            "publicationInfo": item.get("publicationInfo"),
            "snippet": item.get("snippet"),
            "year": item.get("year"),
            "citedBy": item.get("citedBy"),
            "pdfUrl": item.get("pdfUrl"),
        })


    return json.dumps(cleaned, ensure_ascii=False)


# ── Tool Schema（DeepSeek 兼容 OpenAI tools 协议）───────


# ── Agent Loop ──────────────────────────────────────────
def agent_run(user_query: str, model: str = "deepseek-chat"):
    """
    deepseek-chat = V3.2 非思考模式
    deepseek-reasoner = V3.2 思考模式（tool call 时要回传 reasoning_content，见下方提示）
    """
    messages = [
        {"role": "system", "content": "你是一个学术辅助助手，擅长用 Google Scholar 检索论文并做综述。"},
        {"role": "user", "content": user_query},
    ]

    # 第一轮：让模型决定是否调工具
    resp = deepseek.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    msg = resp.choices[0].message

    # 如果模型发了 tool_calls
    if msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            args = json.loads(tc.function.arguments)
            result=""
            if fn_name == "search_google_scholar":
                result = search_google_scholar(**args)
            else:
                result = json.dumps({"error": "unknown tool"})

            # 把『模型发的调用』和『工具返回』都追加进上下文
            # messages.append(msg)          # assistant 那条（含 tool_calls）

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        # 第二轮：模型拿到 Scholar 结果后做自然语言回答
        final = deepseek.chat.completions.create(
            model=model,
            messages=messages,
        )
        return final.choices[0].message.content

    # 没触发工具，直接答
    return msg.content


# ── 试运行 ──────────────────────────────────────────────
if __name__ == "__main__":
    q = "Apple Inc 营销战略 学术论文"
    answer = agent_run(q)
    print("\n🧠 DeepSeek 回答：\n")
    print(answer)