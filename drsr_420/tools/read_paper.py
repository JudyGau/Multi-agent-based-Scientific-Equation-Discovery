import json
import http.client
from openai import OpenAI
import requests
import os
from tqdm import tqdm

# ── 客户端 ──────────────────────────────────────────────
deepseek = OpenAI(
    api_key="sk-3970c8c4922f49fd89761fe3ad4a5eb5",
    base_url="https://api.deepseek.com",        # 兼容 OpenAI 格式
)

SERPER_KEY = "ac28c1aac4d446f3de5c8e79ea6d406727509455"


# ── 工具 2：web visit ───────────────────────
def read_paper(pdf_links, save_dir="pdf_downloads") :
    """
    调 Serper 的 WebPage API，
    返回『已摘选结构化』的 JSON 字符串，方便模型消费。
    """
    """下载PDF文件并保存到本地（使用代理）"""
    # 代理配置
    proxyHost = "www.16yun.cn"
    proxyPort = "5445"
    proxyUser = "16QMSOML"
    proxyPass = "280651"

    # 构造代理字典
    proxies = {
        "http": f"http://{proxyUser}:{proxyPass}@{proxyHost}:{proxyPort}",
        "https": f"http://{proxyUser}:{proxyPass}@{proxyHost}:{proxyPort}"
    }

    # 请求头设置
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for title, pdf_url in tqdm(pdf_links, desc="下载PDF（代理版）"):
        try:
            # 使用代理发送请求
            response = requests.get(
                pdf_url,
                stream=True,
                proxies=proxies,
                headers=headers,
                timeout=30  # 设置超时时间
            )

            if response.status_code == 200:
                # 替换文件名中的非法字符
                safe_title = "".join(c if c.isalnum() else "_" for c in title)
                file_path = os.path.join(save_dir, f"{safe_title}.pdf")

                # 分块写入文件
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
            else:
                print(f"下载失败: {title} | 状态码: {response.status_code} | URL: {pdf_url}")
        except requests.exceptions.RequestException as e:
            print(f"请求异常: {title} | 错误: {e}")
        except Exception as e:
            print(f"未知错误: {title} | 错误: {e}")




    # # 只留 organic 里有用的字段（跟上一轮你贴的结构对齐）
    # cleaned = []
    # for item in raw.get("organic", []):
    #     cleaned.append({
    #         "title": item.get("title"),
    #         "link": item.get("link"),
    #         "publicationInfo": item.get("publicationInfo"),
    #         "snippet": item.get("snippet"),
    #         "year": item.get("year"),
    #         "citedBy": item.get("citedBy"),
    #         "pdfUrl": item.get("pdfUrl"),
    #     })
    #
    # return json.dumps(cleaned, ensure_ascii=False)


# ── Tool Schema（DeepSeek 兼容 OpenAI tools 协议）───────
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_google_scholar",
            "description": "用 Google Scholar 搜索中文学术 / 英文论文，返回标题、作者、年份、被引量等。当用户问到论文、研究、文献、某作者工作时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "学术搜索关键词，如 'Apple Inc international marketing 2024'",
                    },
                    "num": {
                        "type": "integer",
                        "description": "返回条数，默认 10，最大 20",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    }
]


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
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            args = json.loads(tc.function.arguments)

            if fn_name == "read_paper":
                result = read_paper(**args)
            else:
                result = json.dumps({"error": "unknown tool"})

            # 把『模型发的调用』和『工具返回』都追加进上下文
            messages.append(msg)          # assistant 那条（含 tool_calls）
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
    # q = "Apple Inc 营销战略 学术论文"
    # answer = agent_run(q)
    # print("\n🧠 DeepSeek 回答：\n")
    # print(answer)
    # pdf_links = fetch_pdf_links_from_arxiv(max_results=5)
    pdf_links=""
    read_paper(pdf_links)