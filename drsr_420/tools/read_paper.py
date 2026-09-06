import json
import http.client
import os
import random
import re
import sys
from typing import List
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
from drsr_420.tools.tools_description import tools

import pymupdf  # PyMuPDF
import llm
from drsr_420.tools.search_paper import search_paper

# ── 客户端（懒加载，避免模块导入时创建客户端而崩溃）────
def _load_llm_config():
    """读取 LLM 配置文件（如 glm_glm-5.3-flash.config）。"""
    return llm.load_llm_config("glm_glm-5.3-flash.config")


def _build_client(config):
    """基于模型配置文件（如 glm_glm-5.3-flash.config）构建项目自身的 LLM 客户端（复用 llm.ClientFactory）。

    使用项目自研客户端（基于 requests），兼容 api_key 为空串的本地服务；
    host/base_url 双键与 scheme 补齐由 ClientFactory 内部统一规范化。
    """
    return llm.ClientFactory.from_config(config)


_reader_client = None
_reader_config = None


def _get_reader():
    """懒加载文献阅读/总结客户端（首次调用时构建并缓存）。"""
    global _reader_client, _reader_config
    if _reader_client is None:
        _reader_config = _load_llm_config()
        model_name = _reader_config.get('model')
        if not model_name or '/' not in model_name:
            raise ValueError("缺少模型提供商：请在配置文件的 model 字段使用 'provider/model' 格式，例如 'CSTCloud/gpt-oss-120b'")
        _reader_client = _build_client(_reader_config)
    return _reader_client, _reader_config


def _summarize_text(client, cfg, full_text):
    """调用 LLM 对论文全文做摘要，返回摘要文本。

    本地文献库与新下载两条路径共用，保证工具恒返回摘要而非原文，
    避免长文本直接回传给 agent 撑爆上下文。
    """
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
    return response["content"]


_agent_client = None


def _get_agent_client():
    """懒加载 agent_run 使用的客户端（基于模型配置文件配置的模型）。"""
    global _agent_client
    if _agent_client is None:
        _agent_client = _build_client(_load_llm_config())
    return _agent_client

# ── Sci-Hub 反爬规避 ────────────────────────────────
# 镜像会不定期失效或被封，多镜像依次尝试，命中反爬/失效自动切换
SCI_HUB_MIRRORS = [
    "https://sci-hub.st",
    "https://sci-hub.se",
    "https://sci-hub.ru",
    "https://sci-hub.wf",
    "https://sci-hub.ee",
    "https://sci-hub.ren",
]

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0',
]


def _is_blocked_page(text: str) -> bool:
    """检测是否命中反爬/挑战/拦截页（Cloudflare / DataDome / Altcha / PoW 等）。"""
    low = (text or "")[:3000].lower()
    markers = (
        "just a moment", "cloudflare", "cf-challenge", "cf_chl",
        "datadome", "__ddg", "captcha", "attention required",
        "access denied", "challenge-platform", "403 forbidden",
        # JS 挑战页（requests 无法通过，识别后跳过该镜像）
        "checking your browser", "altcha", "проверка на робота",
        "proof-of-work", "proof of work", "verify you are human",
    )
    return any(m in low for m in markers)


def _extract_pdf_url(html: str, base_url: str) -> str | None:
    """从 sci-hub 结果页提取真实 PDF 直链（meta / iframe / embed / link 兜底）。"""
    soup = BeautifulSoup(html, "html.parser")
    meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
    if meta and meta.get("content"):
        return urljoin(base_url, meta["content"].strip())
    for tag in soup.find_all(["iframe", "embed", "object"]):
        src = tag.get("src") or tag.get("data")
        if src and ".pdf" in src.lower():
            return urljoin(base_url, src.strip())
    a = soup.find("a", attrs={"translate": "zh:here"})
    if a and a.get("href"):
        return urljoin(base_url, a["href"].strip())
    for a in soup.find_all("a", href=True):
        if a["href"].strip().lower().endswith(".pdf"):
            return urljoin(base_url, a["href"].strip())
    return None


def _download_to(file_path: str, url: str, session=None, timeout: int = 30) -> bool:
    """下载 url 到 file_path，用 %PDF 魔数校验内容，成功返回 True。

    用独立 session 时（sci-hub）可复用其中的 cookie/header；默认用 requests 直接下载。
    """
    try:
        fetcher = session if session is not None else requests
        resp = fetcher.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        first = next(resp.iter_content(1024), b"")
        if not first.startswith(b"%PDF"):
            return False
        with open(file_path, "wb") as f:
            f.write(first)
            for chunk in resp.iter_content(1024):
                f.write(chunk)
        return True
    except Exception:
        return False


def _try_unpaywall(doi: str, file_path: str, timeout: int = 30) -> bool:
    """通过 Unpaywall 查询开放获取（OA）PDF 直链并下载（合法、无 JS 挑战）。"""
    email = os.environ.get("UNPAYWALL_EMAIL", "drsr.rag.download@outlook.com")
    try:
        resp = requests.get(
            f"https://api.unpaywall.org/v2/{doi}?email={email}",
            timeout=timeout,
            headers={'user-agent': random.choice(USER_AGENTS)},
        )
        if resp.status_code != 200:
            return False
        best = (resp.json().get("best_oa_location") or {})
        pdf_url = best.get("url_for_pdf") or best.get("url")
        if not pdf_url:
            return False
        return _download_to(file_path, pdf_url, timeout=timeout)
    except Exception:
        return False


def _download_pdf_by_doi(doi: str, save_dir: str, timeout: int = 30) -> str:
    """按 DOI 下载 PDF 到 save_dir，返回本地文件路径。

    下载渠道按优先级依次尝试：
      1) Unpaywall 开放获取直链（合法，无 JS 反爬）；
      2) Sci-Hub 多镜像 failover（命中反爬/失效/无直链自动切换，跳过 JS 挑战页）。
    下载前用 %PDF 魔数校验，避免把挑战页 HTML 当 PDF 保存；全部失败则抛异常。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    safe_doi = "".join('' if c == '/' else c for c in doi)
    file_path = os.path.join(save_dir, f"{safe_doi}.pdf")

    # 渠道 1：Open Access（合法渠道，优先）
    if _try_unpaywall(doi, file_path, timeout):
        print(f"下载成功 (Unpaywall OA): {file_path}", file=sys.stderr)
        return file_path

    # 渠道 2：Sci-Hub 多镜像 failover
    last_err = None
    for mirror in SCI_HUB_MIRRORS:
        try:
            with requests.Session() as session:
                session.headers.update({
                    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                    'accept-language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
                    'user-agent': random.choice(USER_AGENTS),
                })
                # 1) 打开 sci-hub 结果页
                page_resp = session.get(f"{mirror}/{doi}", timeout=timeout)
                if page_resp.status_code != 200 or _is_blocked_page(page_resp.text):
                    continue
                # 2) 提取真实 PDF 直链
                pdf_url = _extract_pdf_url(page_resp.text, mirror)
                if not pdf_url:
                    continue
                # 3) 下载 PDF 并校验魔数
                if _download_to(file_path, pdf_url, session=session, timeout=timeout):
                    print(f"下载成功 (Sci-Hub {mirror}): {file_path}", file=sys.stderr)
                    return file_path
                raise RuntimeError(f"下载内容不是 PDF（可能被拦截）: {pdf_url}")
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"所有下载渠道失败（Unpaywall 无 OA + Sci-Hub 全部不可达/被反爬）: {last_err}")

# SERPER_KEY = "ac28c1aac4d446f3de5c8e79ea6d406727509455"


# ── 工具 2：web visit ───────────────────────
def read_paper(title_doi: list[tuple[str, str]] | tuple[str, str], save_dir="pdf_downloads") -> str :
    """
    调 Serper 的 WebPage API，
    返回『已摘选结构化』的 JSON 字符串，方便模型消费。
    """
    """下载PDF文件并保存到本地"""
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
                summary = _summarize_text(client, cfg, full_text)
                textlist.append(summary)

                #分析下一个文献
                continue
            else:
                print(f"本地文献库不存在文献: {file_path}，尝试 Sci-Hub 下载 ...", file=sys.stderr)
                try:
                    file_path = _download_pdf_by_doi(doi, save_dir)
                except Exception as e:
                    print(f"下载失败: {title} | 错误: {e}", file=sys.stderr)
                    textlist.append(f"下载失败: {title} | 错误: {e}")
                    continue

                # 读取全文并做 LLM 摘要（与本地库路径一致，恒返回摘要而非原文）
                doc = pymupdf.open(file_path)
                full_text = ""
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    full_text += f"\n--- Page {page_num + 1} ---\n{text}"
                doc.close()
                print(f"文件读取成功: {file_path}", file=sys.stderr)
                client, cfg = _get_reader()
                summary = _summarize_text(client, cfg, full_text)
                textlist.append(summary)

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
        {"role": "system", "content": "You are an academic assistant skilled at searching for papers, downloading them, and summarizing them."},
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