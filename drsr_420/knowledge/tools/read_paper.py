import contextlib
import json
import http.client
import os
import random
import re
import sys
from pathlib import Path
from typing import List
from urllib.parse import quote, urljoin

if __package__ in (None, ""):
    # 直接以脚本运行（python drsr_420/tools/read_paper.py）时项目根不在 sys.path，
    # 下面的 drsr_420.* 绝对导入会 ModuleNotFoundError；补上根目录使两种方式均可用。
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from bs4 import BeautifulSoup
import pymupdf  # PyMuPDF
import requests
from tqdm import tqdm
import drsr_420.llm as llm
from drsr_420.knowledge.tools.search_paper import search_paper

# ── 客户端（懒加载，避免模块导入时创建客户端而崩溃）────
def _load_llm_config():
    """读取 ``summary`` 角色的 LLM 档案。

    本模块跑在 **MCP 服务器子进程**里（由 ``knowledge/tool_runner`` 以 stdio 拉起），
    拿不到父进程的解析结果，只能靠环境变量 ``DRSR_ROLE_CONFIG_SUMMARY`` 继承父进程
    解析出的档案路径；该变量缺失时按 ``config/agents.config.json`` 自行解析。
    档案选择不接受硬编码文件名——库代码里出现字面量文件名有专门的护栏拦。
    """
    return llm.load_role_config("summary")


def _build_client(config):
    """构建 ``summary`` 角色客户端（含该角色声明的私有参数）。

    使用项目自研客户端（基于 requests），兼容 api_key 为空串的本地服务；
    base_url 的 scheme 补齐由 ClientFactory 内部统一规范化（``host`` 拼写已废弃）；
    ``temperature`` / ``top_p`` 之类的角色参数来自配置注册表，不再写死在本文件里。
    """
    return llm.ClientFactory.from_config(
        config, task_params={"summary": llm.resolve_params("summary")})


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


#: MCP 服务器子进程的标记环境变量，由 ``knowledge/tools/mcp_server._mark_server_process``
#: 在启动时写入本进程。读它就能判断"当前是不是跑在 MCP 子进程里"——见 ``_stats_to_stderr``。
MCP_SERVER_ENV = "DRSR_MCP_SERVER"


@contextlib.contextmanager
def _stats_to_stderr():
    """在 MCP 服务器子进程里，把 LLM 客户端的统计打印改道到 stderr。

    文献总结要调 LLM，``llm.client`` 每次请求都会 print 一段
    ``[provider][model] 第N次 / 本次 tokens / 累计 tokens / 本次用时``。
    这段文字在 MCP 子进程里**永远看不到**，原因有两层：

      1. 子进程启动时 stdout 就是 MCP 的 JSON-RPC 管道，解释器据此把 ``sys.stdout``
         设成块缓冲（8KB）；MCP SDK 之后只把 fd 1 重定向到 stderr，Python 层的缓冲
         模式不会跟着变——统计文字一直躺在缓冲区里，子进程被强杀收尾时整块丢弃。
      2. 若是在"fd 1 就是协议通道"的 SDK 版本里，直接 print 还会污染协议
         （客户端把非 JSON 行丢弃并记 warning）。

    stderr 是行缓冲的，改道后立即落到控制台/``run.err``，也与本模块其余诊断输出
    （下载成功、文件读取成功、下载失败……）的流向一致。改道只作用于本进程内的
    ``sys.stdout`` 变量，不碰 fd 1，MCP 传输层用的是启动时私有的那份 dup，不受影响。

    只在子进程里改道：在本进程（CLI/测试）直接调用 ``read_paper`` 时摘要统计仍走
    stdout，能被 ``cli.main`` 的输出 tee 正常收进 ``run.out``。
    """
    if not os.environ.get(MCP_SERVER_ENV):
        yield
        return
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout


def _summarize_text(client, cfg, full_text):
    """调用 LLM 对论文全文做摘要，返回摘要文本。

    本地文献库与新下载两条路径共用，保证工具恒返回摘要而非原文，
    避免长文本直接回传给 agent 撑爆上下文。
    """
    # 输出上限优先读 max_completion_tokens，缺失则回退 max_tokens：
    # 直接 cfg.get("max_completion_tokens") 会把 None 写进 kwargs，
    # 而 llm 的 glm 分支会用该 None 覆盖已配置好的 max_tokens（请求 400）。
    # （temperature / top_p / frequency_penalty 曾在此写死，已迁到
    #   config/agents.config.json 的 roles.summary.params）
    max_out = cfg.get("max_completion_tokens") or cfg.get("max_tokens")
    overrides = {}
    if isinstance(max_out, int) and max_out > 0:
        overrides['max_completion_tokens'] = max_out
    if overrides:
        client.kwargs.update(overrides)
    # 在 MCP 子进程里把统计打印改道 stderr（否则整段 token/耗时统计会被块缓冲吞掉）
    with _stats_to_stderr():
        response = client.chat([
            {"role": "system", "content": "You are a helpful assistant, you need to read literature and summarize."},
            {"role": "user", "content": f"{full_text}"}
        ])
    # drsr_420.llm.client 对每个请求都附带 tools + tool_choice=auto：摘要模型偶尔会"回答"成
    # 工具调用而 content 为空——必须显式报错（由调用方记入返回列表），
    # 否则空字符串会被当成合法摘要静默入库。
    if response.get("tool_calls"):
        raise RuntimeError("摘要模型返回了工具调用而非文本摘要")
    content = (response.get("content") or "").strip()
    if not content:
        raise RuntimeError("摘要模型返回空内容")
    return content


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
    先写入 .part 临时文件、校验并下载完整后才原子替换到目标路径：避免网络中断
    留下半截 PDF——本地缓存分支会永久命中这个坏文件（pymupdf 打开失败或读到空文本），
    该文献在此后每次运行中都"下载成功但读取失败"。
    """
    tmp_path = file_path + ".part"
    try:
        fetcher = session if session is not None else requests
        resp = fetcher.get(url, stream=True, timeout=timeout)
    except Exception:
        return False
    stream = None
    try:
        resp.raise_for_status()
        stream = resp.iter_content(1024)
        first = next(stream, b"")
        if not first.startswith(b"%PDF"):
            return False
        with open(tmp_path, "wb") as f:
            f.write(first)
            for chunk in stream:
                f.write(chunk)
        os.replace(tmp_path, file_path)
        return True
    except Exception:
        return False
    finally:
        # 先关生成器再关响应：魔数校验失败提前 return 时生成器仍挂起，
        # 直接 resp.close() 会在 GC 时抛 "ignored GeneratorExit" 噪音进 stderr
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass
        try:
            resp.close()
        except Exception:
            pass
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _try_unpaywall(doi: str, file_path: str, timeout: int = 30) -> bool:
    """通过 Unpaywall 查询开放获取（OA）PDF 直链并下载（合法、无 JS 挑战）。"""
    email = os.environ.get("UNPAYWALL_EMAIL", "drsr.rag.download@outlook.com")
    try:
        # DOI 可能含 URL 保留字符（如括号），必须转义，否则请求路径被破坏
        resp = requests.get(
            f"https://api.unpaywall.org/v2/{quote(str(doi), safe='')}?email={email}",
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


def _doi_filename(doi: str) -> str:
    """DOI → 安全本地文件名：只保留 [A-Za-z0-9._-]，其余字符（含 '/'）直接删除。

    DOI 来自 LLM 生成的检索结果，不可信。旧实现只删 '/'，`\\` 与 `:` 原样保留：
    `..\\..\\evil` 会穿越出 save_dir，绝对路径 `C:\\...` 会让 os.path.join 直接
    丢弃 save_dir —— 等于给 LLM 开了任意文件写入口。典型 DOI（如 10.1016/j.x）
    在新旧规则下文件名一致，缓存向后兼容。
    """
    return re.sub(r"[^A-Za-z0-9._-]", "", str(doi)) or "unnamed"


def _download_pdf_by_doi(doi: str, save_dir: str, timeout: int = 30) -> str:
    """按 DOI 下载 PDF 到 save_dir，返回本地文件路径。

    下载渠道按优先级依次尝试：
      1) Unpaywall 开放获取直链（合法，无 JS 反爬）；
      2) Sci-Hub 多镜像 failover（命中反爬/失效/无直链自动切换，跳过 JS 挑战页）。
    下载前用 %PDF 魔数校验，避免把挑战页 HTML 当 PDF 保存；全部失败则抛异常。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, f"{_doi_filename(doi)}.pdf")

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

    if isinstance(title_doi, tuple):
        title_doi = [title_doi]
    if isinstance(title_doi, list) and len(title_doi) == 2 and all(isinstance(item, str) for item in title_doi):
        # 单对 (title, doi) 被平铺成字符串列表的情况
        title_doi = [title_doi]

    for item in tqdm(title_doi, desc="下载PDF"):
        try:
            # 非法条目单独报错回传，不让整批调用崩溃。
            # 注意：2 个字符的字符串也能解包出 2 个变量（逐字符），必须先排除 str。
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError
            title, pdf_url = item
        except (TypeError, ValueError):
            textlist.append(f"非法条目（期望 [title, doi] 二元组）: {item!r}")
            continue
        try:

            # 先尝试从本地文献库寻找文献
            # 文件名按 DOI 净化（防穿越，见 _doi_filename）
            doi=pdf_url
            file_path = os.path.join(save_dir, f"{_doi_filename(doi)}.pdf")

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
            # 错误必须进返回列表：只 print 会让结果无声变短，
            # agent 拿到的数组与请求的论文对不上号，误以为"该文无内容"
            textlist.append(f"请求异常: {title} | 错误: {e}")
        except Exception as e:
            print(f"未知错误: {title} | 错误: {e}",file=sys.stderr)
            textlist.append(f"未知错误: {title} | 错误: {e}")

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
def agent_run(user_query: str, model: str | None = None):
    """
    deepseek-chat = V3.2 非思考模式
    deepseek-reasoner = V3.2 思考模式（tool call 时要回传 reasoning_content，见下方提示）

    :param model: 覆盖模型名；``None`` 表示沿用 ``summary`` 角色档案里配置的模型。
        曾经这里写死 ``"deepseek-v4-pro"``——它会**覆盖**档案里的选择，于是"给文献摘要
        换模型"这件事在配置层完全失效（且那个模型名已随档案一起下线）。
    """
    messages = [
        {"role": "system", "content": "You are an academic assistant skilled at searching for papers, downloading them, and summarizing them."},
        {"role": "user", "content": user_query},
    ]

    client = _get_agent_client()
    if model:
        client.model = model  # 允许调用方显式指定模型名

    # 第一轮：让模型决定是否调工具（统计打印同 _summarize_text，见 _stats_to_stderr）
    with _stats_to_stderr():
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
                try:
                    args = json.loads(tc.get('function', {}).get('arguments', '{}') or '{}')
                except json.JSONDecodeError as e:
                    # LLM 偶发产出非法工具参数 JSON：把错误回传给模型自行纠正，
                    # 而不是让整个循环崩溃
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get('id', ''),
                        "content": json.dumps({"error": f"invalid tool arguments: {e}"})
                    })
                    continue
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
            with _stats_to_stderr():
                resp = client.chat(messages)
            msg = resp
        # 如果未调用，则跳出循环
        else:
            return msg.get('content', '')


# ── 试运行 ──────────────────────────────────────────────
if __name__ == "__main__":
    q = "MRF"
    answer = agent_run(q)
    print("\n[DeepSeek 回答]\n")
    print(answer)


    # pdf_links="https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=11323465"
    # read_paper(pdf_links,"LLMSR")