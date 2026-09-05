import json
import http.client
import re
import sys
from typing import List

from bs4 import BeautifulSoup
import requests
import os
from tqdm import tqdm
from drsr_420.tools.tools_description import tools

import pymupdf  # PyMuPDF
import llm
from drsr_420.tools.search_paper import search_paper

# ── 客户端（懒加载，避免模块导入时创建客户端而崩溃）────
def _load_llm_config():
    """读取 ./llm.config。"""
    with open("./llm_summary.config", 'r', encoding='utf-8') as f:
        return json.load(f)


def _build_client(config):
    """基于 llm.config 构建项目自身的 LLM 客户端（复用 llm.ClientFactory）。

    llm.config 使用 'host' 键作为 base_url，这里映射为 ClientFactory 需要的 'base_url'。
    使用项目自研客户端（基于 requests），兼容 api_key 为空串的本地服务。
    """
    cfg = dict(config)
    if not cfg.get('base_url') and cfg.get('host'):
        cfg['base_url'] = cfg['host']
    return llm.ClientFactory.from_config(cfg)


_reader_client = None
_reader_config = None


def _get_reader():
    """懒加载文献阅读/总结客户端（首次调用时构建并缓存）。"""
    global _reader_client, _reader_config
    if _reader_client is None:
        _reader_config = _load_llm_config()
        model_name = _reader_config.get('model')
        if not model_name or '/' not in model_name:
            raise ValueError("缺少模型提供商：请在 llm.config 的 model 字段使用 'provider/model' 格式，例如 'CSTCloud/gpt-oss-120b'")
        _reader_client = _build_client(_reader_config)
    return _reader_client, _reader_config


_agent_client = None


def _get_agent_client():
    """懒加载 agent_run 使用的客户端（基于 llm.config 配置的模型）。"""
    global _agent_client
    if _agent_client is None:
        _agent_client = _build_client(_load_llm_config())
    return _agent_client

# SERPER_KEY = "ac28c1aac4d446f3de5c8e79ea6d406727509455"


# ── 工具 2：web visit ───────────────────────
def read_paper(title_doi: list[tuple[str, str]] | tuple[str, str], save_dir="pdf_downloads") -> str :
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
        'cache-control': 'max-age=0',
        'pragma': 'no-cache',
        'priority': 'u=0, i',
        'sec-ch-ua': '"Not=A?Brand";v="99", "Microsoft Edge";v="151", "Chromium";v="151"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'none',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36 Edg/151.0.0.0',
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
        '__ddg9_' : '112.32.136.114',
        '__ddgmark_' : 'Py3toAtBOyjZhvUH',
        '__ddg5_' : 'ZYmdRTIB54WpAp5z',
        '__ddg2_' : '3713k9Eb5ZCXTZbp',
        'ddg_last_challenge' : '1786093482052',
        'PHPSESSID' : '1cb151d12013f5a9d94ad21d84e21f3e',
        'session' : 'd8181b66b46f21c48320fd96d257e873',
        'refresh' : '1786093506.7948',
        '__ddg10_' : '1786500725',
        '__ddg8_' : 'FT9PAhB6baveZXHD',

    }


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)



    textlist = []

    if isinstance(title_doi, list) and all(isinstance(item, str) for item in title_doi):
        title_doi = [title_doi]

    for title, pdf_url in tqdm(title_doi, desc="下载PDF"):
        try:

            # 先尝试从本地文献库寻找文献
            # 替换文标题中的非法字符
            doi=pdf_url
            safe_doi = "".join('' if c == '/' else c for c in doi)
            file_path = os.path.join(save_dir, f"{safe_doi}.pdf")

            if os.path.exists(file_path):
                print(f"在本地文献库找到文献: {file_path}", file=sys.stderr)
                doc = pymupdf.open(file_path)
                full_text = ""
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    full_text += f"\n--- Page {page_num + 1} ---\n{text}"
                doc.close()
                print(f"文件读取成功: {file_path}", file=sys.stderr)
                # 2. 发起聊天请求（使用项目自身 LLM 客户端，兼容空 api_key 的本地服务）
                client, cfg = _get_reader()
                client.kwargs.update({
                    'temperature': 0.4,
                    'frequency_penalty': 0.1,
                    'top_p': 0.9,
                    'max_completion_tokens': cfg.get("max_completion_tokens"),
                })
                response = client.chat([
                    {"role": "system", "content": "You are a helpful assistant, you need to read literature and summarize."},
                    {"role": "user", "content": f"{full_text}"}
                ])

                # 3. 打印回复
                summary = response["content"]
                # print("======总结结果======")
                # print(summary)
                # print("==================")
                textlist.append(summary)

                #分析下一个文献
                continue
            else:

                print(f"本地文献库不存在文献: file_path {file_path}, title_url {(title, pdf_url)}", file=sys.stderr)
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
                    print("[!] 返回的不是 HTML 页面，可能镜像失效或 DOI 未收录",file=sys.stderr)
                    print(f"    响应前 300 字符: {response.text[:300]}",file=sys.stderr)
                    # exit(1)

                # 解析 HTML，提取真实 PDF 地址 ──────────
                soup = BeautifulSoup(response.text, "html.parser")
                pdf_url = None

                # 找到 <meta name="citation_pdf_url" content="...">
                meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
                if meta and meta.get("content"):
                    pdf_url = meta["content"]
                    pdf_url = "https://sci-hub.st" + pdf_url
                    print(f"找到 meta name = citation_pdf_url, content = {pdf_url}",file=sys.stderr)


                # sci-hub 未收录，尝试找到官网的pdf_url
                # 找到 <a translate="zh:here" href="...">
                if not pdf_url:
                    a = soup.find("a", attrs={"translate": "zh:here"})
                    if a and a.get("href"):
                        pdf_url = a.get("href")
                        print(f"找到 a translate = zh:here, href = {pdf_url}",file=sys.stderr)

                if not pdf_url:
                    print("[!] 未找到 PDF 链接，页面结构可能已变化")
                    print(f"    HTML 字符:\n{response.text}")
                    # exit(1)
            else:
                print(f"下载失败: title_url {(title, pdf_url)} | 状态码: {response.status_code}",file=sys.stderr)
                textlist.append(f"下载失败: title_url {(title, pdf_url)} | 状态码: {response.status_code}",file=sys.stderr)
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

                print(f"下载成功: title_url {(title, pdf_url)} | 状态码: {response.status_code}",file=sys.stderr)
                # 替换文件名中的非法字符
                # safe_title = "".join(c if c.isalnum() else "_" for c in title)
                # file_path = os.path.join(save_dir, f"{safe_title}.pdf")
                safe_doi = "".join('' if c == '/' else c for c in doi)
                file_path = os.path.join(save_dir, f"{safe_doi}.pdf")

                # 分块写入文件
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)

                print(f"文件保存成功: {file_path}",file=sys.stderr)

                doc = pymupdf.open(file_path)
                full_text = ""
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    full_text += f"\n--- Page {page_num + 1} ---\n{text}"
                doc.close()
                print(f"文件读取成功: {file_path}", file=sys.stderr)
                textlist.append(full_text)
            else:
                print(f"下载失败: title_url {(title, pdf_url)} | 状态码: {response.status_code}",file=sys.stderr)
                textlist.append(f"下载失败: title_url {(title, pdf_url)} | 状态码: {response.status_code}",file=sys.stderr)

        except requests.exceptions.RequestException as e:
            print(f"请求异常: {title} | 错误: {e}",file=sys.stderr)
        except Exception as e:
            print(f"未知错误: {title} | 错误: {e}",file=sys.stderr)

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

    client = _get_agent_client()
    client.model = model  # 允许调用方指定模型名

    # 第一轮：让模型决定是否调工具
    resp = client.chat(messages)
    msg = resp

    while True:
        # print("========================思考过程========================\n")
        # print(resp.get('reasoning_content', ''))
        # print("====================================================\n")

        tool_calls = msg.get('tool_calls') or []
        messages.append(msg)

        # 如果调了 tool，执行后回传
        if tool_calls:
            print("调用了工具：", tool_calls)

            for tc in tool_calls:
                fn_name = tc.get('function', {}).get('name')
                args = json.loads(tc.get('function', {}).get('arguments', '{}') or '{}')
                result = ''
                if fn_name == "search_paper":
                    result = search_paper(**args)
                elif fn_name == "read_paper":
                    result = read_paper(**args)
                else:
                    result = json.dumps({"error": "unknown tool"})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get('id', ''),
                    "content": result
                })

            # 第二轮：模型拿到 Scholar 结果后做自然语言回答
            resp = client.chat(messages)
            msg = resp
        # 如果未调用，则跳出循环
        else:
            return msg.get('content', '')


# ── 试运行 ──────────────────────────────────────────────
if __name__ == "__main__":
    q = "MRF"
    answer = agent_run(q)
    print("\n🧠 DeepSeek 回答：\n")
    print(answer)


    # pdf_links="https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=11323465"
    # read_paper(pdf_links,"LLMSR")