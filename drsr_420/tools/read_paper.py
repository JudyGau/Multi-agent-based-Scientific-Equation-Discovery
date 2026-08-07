import json
import http.client
import re
from bs4 import BeautifulSoup
from openai import OpenAI
import requests
import os
from tqdm import tqdm
from drsr_420.tools.tools_description import tools

import fitz  # PyMuPDF

from drsr_420.tools.search_paper import search_paper

# ── 客户端 ──────────────────────────────────────────────
deepseek = OpenAI(
    api_key="xxx",
    base_url="https://api.deepseek.com",        # 兼容 OpenAI 格式
)

# SERPER_KEY = "ac28c1aac4d446f3de5c8e79ea6d406727509455"


# ── 工具 2：web visit ───────────────────────
def read_paper(title_doi: list[tuple[str, str]], save_dir="pdf_downloads") -> str :
    """
    调 Serper 的 WebPage API，
    返回『已摘选结构化』的 JSON 字符串，方便模型消费。
    """
    """下载PDF文件并保存到本地"""
    # # 代理配置
    # proxyHost = "www.16yun.cn"
    # proxyPort = "5445"
    # proxyUser = "16QMSOML"
    # proxyPass = "280651"

    # # 构造代理字典
    # proxies = {
    #     "http": f"http://{proxyUser}:{proxyPass}@{proxyHost}:{proxyPort}",
    #     "https": f"http://{proxyUser}:{proxyPass}@{proxyHost}:{proxyPort}"
    # }

    # 请求头设置
    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7,es-ES;q=0.6,es;q=0.5',
        'cache-control': 'no-cache',
        'pragma': 'no-cache',
        'priority': 'u=0, i',
        'sec-ch-ua': '"Not;A=Brand";v="8", "Chromium";v="150", "Microsoft Edge";v="150"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'none',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36 Edg/150.0.0.0',
    }

    cookies = {
        # '__ddgid_': 'ifJHThZoCclQh7Xb',
        # '__ddg2_': 'E0OKMRIZMDuyo1DA',
        # '__ddg1_': 'XrbNjpRf24gdh6Zngg9x',
        # '__ddgmark_': 'QQUozRDUpKWhbW1a',
        # 'session': '76a2e63503e1e0609c1b32dca810b4f5',
        # 'refresh': '1784796304.9299',
        # 'session': '76a2e63503e1e0609c1b32dca810b4f5',
        # 'refresh': '1784796304.9305',
        # '__ddg9_': '210.45.118.7',
        # '__ddg5_': 'Nleqmf5URrhPyOLL',
        # 'PHPSESSID': '217c33eecb13851160634129ab7d9396',
        # '__ddg8_': 'n5BALTtkm8YR3KBo',
        # '__ddg10_': '1784879941',

        '__ddg1_' : 'CeSMSD6vG9cbBUhNgEZC',
        '__ddgid_' : 'bWJgM74dLEQqdIsH',
        '__ddg9_' : '114.101.214.75',
        '__ddgmark_' : 'JDwRHa4WjGEWX81T',
        '__ddg5_' : 'ni93fRH1XSKYX6fg',
        '__ddg2_' : '3713k9Eb5ZCXTZbp',
        'ddg_last_challenge' : '1786093482052',
        'PHPSESSID' : '3a26b98a3c5eee422b4ed949f536e070',
        'session' : 'd8181b66b46f21c48320fd96d257e873',
        'refresh' : '1786093506.7948',
        '__ddg10_' : '1786093645',
        '__ddg8_' : 'wykJtdYSwp3mOozy',

    }


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)



    textlist = []

    for title, pdf_url in tqdm(title_doi, desc="下载PDF"):
        try:

            # 先尝试从本地文献库寻找文献
            # 替换文标题中的非法字符
            safe_title = "".join(c if c.isalnum() else "_" for c in title)
            file_path = os.path.join(save_dir, f"{safe_title}.pdf")

            if os.path.exists(file_path):
                print(f"在本地文献库找到文献: {file_path}")
                doc = fitz.open(file_path)
                full_text = ""
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    full_text += f"\n--- Page {page_num + 1} ---\n{text}"
                doc.close()
                print(f"文件读取成功: {file_path}")
                textlist.append(full_text)

                #分析下一个文献
                continue
            else:
                print(f"本地文献库不存在文献: title_url {(title, pdf_url)}")
                textlist.append(f"本地文献库不存在文献: title_url {(title, pdf_url)}")


            pdf_url = "https://sci-hub.st/" + pdf_url
            headers['referer'] = "https://sci-hub.st"
            # 发送请求
            response = requests.get(
                pdf_url,
                stream=True,
                # impersonate="chrome120",
                # proxies=proxies,
                headers=headers,
                cookies=cookies,
                timeout=30  # 设置超时时间
            )

            if response.status_code == 200:
                # 验证确实是 HTML
                if "<html" not in response.text[:500].lower():
                    print("[!] 返回的不是 HTML 页面，可能镜像失效或 DOI 未收录")
                    print(f"    响应前 300 字符: {response.text[:300]}")
                    # exit(1)

                # 解析 HTML，提取真实 PDF 地址 ──────────
                soup = BeautifulSoup(response.text, "html.parser")
                pdf_url = None

                # 找到 <meta name="citation_pdf_url" content="...">
                meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
                if meta and meta.get("content"):
                    pdf_url = meta["content"]
                    pdf_url = "https://sci-hub.st" + pdf_url
                    print(f"找到 meta name = citation_pdf_url, content = {pdf_url}")


                # sci-hub 未收录，尝试找到官网的pdf_url
                # 找到 <a translate="zh:here" href="...">
                if not pdf_url:
                    a = soup.find("a", attrs={"translate": "zh:here"})
                    if a and a.get("href"):
                        pdf_url = a.get("href")
                        print(f"找到 a translate = zh:here, href = {pdf_url}")

                if not pdf_url:
                    print("[!] 未找到 PDF 链接，页面结构可能已变化")
                    print(f"    HTML 字符:\n{response.text}")
                    # exit(1)
            else:
                print(f"下载失败: title_url {(title, pdf_url)} | 状态码: {response.status_code}")
                textlist.append(f"下载失败: title_url {(title, pdf_url)} | 状态码: {response.status_code}")
                continue


            response = requests.get(
                pdf_url,
                stream=True,
                # impersonate="chrome120",
                headers=headers,
                cookies=cookies,
                timeout=30  # 设置超时时间
            )

            if response.status_code == 200:

                print(f"下载成功: title_url {(title, pdf_url)} | 状态码: {response.status_code}")
                # 替换文件名中的非法字符
                safe_title = "".join(c if c.isalnum() else "_" for c in title)
                file_path = os.path.join(save_dir, f"{safe_title}.pdf")

                # 分块写入文件
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)

                print(f"文件保存成功: {file_path}")

                doc = fitz.open(file_path)
                full_text = ""
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    full_text += f"\n--- Page {page_num + 1} ---\n{text}"
                doc.close()
                print(f"文件读取成功: {file_path}")
                textlist.append(full_text)
            else:
                print(f"下载失败: title_url {(title, pdf_url)} | 状态码: {response.status_code}")
                textlist.append(f"下载失败: title_url {(title, pdf_url)} | 状态码: {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"请求异常: {title} | 错误: {e}")
        except Exception as e:
            print(f"未知错误: {title} | 错误: {e}")

    return json.dumps(textlist)




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


# ── Agent Loop ──────────────────────────────────────────
def agent_run(user_query: str, model: str = "deepseek-v4-pro"):
    """
    deepseek-chat = V3.2 非思考模式
    deepseek-reasoner = V3.2 思考模式（tool call 时要回传 reasoning_content，见下方提示）
    """
    messages = [
        {"role": "system", "content": "你是一个学术辅助助手，擅长检索论文并下载以及做总结。"},
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

    while True:
        # print("========================思考过程========================\n")
        # print(resp.get('reasoning_content', ''))
        # print("====================================================\n")

        tool_calls = msg.tool_calls
        messages.append(msg)

        # 如果调了 tool，执行后回传
        if tool_calls:
            print("调用了工具：", tool_calls)

            for tc in tool_calls:
                fn_name = tc.function.name
                args = json.loads(tc.function.arguments)
                result = ''
                if fn_name == "search_paper":
                    result = search_paper(**args)
                elif fn_name == "read_paper":
                    result = read_paper(**args)
                else:
                    result = json.dumps({"error": "unknown tool"})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })

            # 第二轮：模型拿到 Scholar 结果后做自然语言回答
            resp = deepseek.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            # return final.choices[0].message.content
        # 如果未调用，则跳出循环
        else:
            # responses.append(resp.get('content', ''))
            # think_responses.append(resp.get('reasoning_content', ''))
            return msg.content


# ── 试运行 ──────────────────────────────────────────────
if __name__ == "__main__":
    q = "MRF"
    answer = agent_run(q)
    print("\n🧠 DeepSeek 回答：\n")
    print(answer)


    # pdf_links="https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=11323465"
    # read_paper(pdf_links,"LLMSR")