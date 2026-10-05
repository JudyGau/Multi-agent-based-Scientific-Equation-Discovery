"""PDF 元数据抽取：从 PDF 文本里认出**标题与 DOI**，并剔除文件名伪装的假 DOI。

角色归属
--------
``knowledge`` 层 RAG 子系统的**书目识别内核**，从 :mod:`drsr_420.knowledge.rag_kb` 拆出。

为什么单独成模块
----------------
这是本项目里唯一"跟正则和 PDF 排版打交道的"地方，也是缺陷密度最高的一处：
知识库里不少条目的 DOI 是从**文件名回推**出来的（形如 ``10.216561000/-0887.380021``），
因此必须能把"真 DOI / 从文件名猜的假 DOI / 占位标题（N/A、untitled、期刊页眉）"
分开——否则参考文献清单里会混进编造的条目。这一堆启发式（20 个正则 + 10 个函数）
需要能被单独测试、单独调参；与检索/持久化混在一起时没人敢动。

公开面
------
:func:`extract_pdf_text` / :func:`doi_from_pdf_text` 是数据入口；
:func:`first_page_title` / :func:`is_placeholder_title` / :func:`resolve_title` /
:func:`resolve_doi` / :func:`same_doi_text` / :func:`safe_id` 供
:class:`~drsr_420.knowledge.rag_kb.RagKB` 组装元数据（**公开名**：跨模块引用私有名
等于声明一个不存在于任何文档里的接口，由 test_architecture 的护栏守护）。
"""
from __future__ import annotations

import re

# ── 文本处理 ──────────────────────────────────────────────────────────
def extract_pdf_text(path: str) -> str:
    """用 pymupdf 逐页提取 PDF 全文。"""
    import pymupdf
    doc = pymupdf.open(path)
    parts = []
    try:
        for i in range(doc.page_count):
            parts.append(f"\n--- Page {i + 1} ---\n{doc.load_page(i).get_text()}")
    finally:
        doc.close()
    return "".join(parts)


#: 论文正文里印出来的 DOI（含页脚 "http://dx.doi.org/10.xxxx/yyy" 形态）。
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>()\[\],;]+")

#: 找 DOI 时扫描的字符数上限（配合"参考文献之前"截断，见 :func:`doi_from_pdf_text`）。
#: 历史上是 3000，实测不够：IOP/Springer 把 DOI 印在首页页眉/页脚，文本落到
#: 3748–5612 字符处（如 ``10.1088/0964-1726/24/12/125005``、``10.1122/1.3479045``），
#: 于是 13 篇的 DOI 一直是空的；8000 覆盖了实测的全部命中位置。
PDF_DOI_SCAN_CHARS = 8000

#: 参考文献小节的起始行（其后的 DOI 属于被引论文，不是本文的）。
_REF_HEAD_RE = re.compile(r"^\s*(?:references|bibliography|参考文献)\s*$",
                          re.IGNORECASE | re.MULTILINE)


def _looks_like_doi(name: str) -> bool:
    """字符串是不是 DOI/DOI 去掉斜杠后的形态（用来拒绝把 DOI 当标题）。"""
    return bool(re.match(r"^10\.\d{4,9}", str(name or "").strip()))


def _doi_key(doi: str) -> str:
    """DOI 的比较键（忽略大小写与 ``/``、``.``、``#`` 之类的分隔差异）。

    只用于"印刷 DOI 与 DOI 形态文件名是否指向同一篇"的核对：文件名抹掉的是分隔符，
    去分隔符后应完全一致。**不能**用它判断"值有没有变"——``10.1122/1.3479045`` 与
    被猜错的 ``10.11221/.3479045`` 去分隔符后完全一样，那是两个不同的 DOI。
    """
    return re.sub(r"[^0-9a-z]", "", str(doi or "").lower())


def same_doi_text(a: str, b: str) -> bool:
    """两个 DOI 串是否只是大小写差异（用于"不必改写库"的判断）。"""
    return str(a or "").strip().lower() == str(b or "").strip().lower()


def doi_from_pdf_text(text: str, scan_chars: int = PDF_DOI_SCAN_CHARS) -> str | None:
    """从 PDF 文本里取**印出来的 DOI**，取不到返回 ``None``。

    这是恢复 DOI 的权威来源。去斜杠文件名无法唯一还原（``10.216561000-0887.380021``
    实际是 ``10.21656/1000-0887.380021``），而论文自己印的 DOI 没有歧义。只看开头
    若干字符：正文靠前的 DOI 属于本文，参考文献里的 DOI 是别人的。
    """
    if not text:
        return None
    body = str(text)
    ref = _REF_HEAD_RE.search(body)     # 参考文献之后的 DOI 属于被引论文，先截掉
    if ref:
        body = body[:ref.start()]
    m = _DOI_RE.search(body[:int(scan_chars)])
    return m.group(0).rstrip(".,;)#]") if m else None


def _recover_doi(stem: str) -> str | None:
    """从去斜杠文件名恢复 DOI（如 10.1016j.jmmm.2020.166652 -> 10.1016/j.jmmm.2020.166652）。

    只在**无歧义**时恢复：文件名抹掉的是哪一个 ``/``（注册号几位）无从判断。两条拒绝：
    后缀里出现 ``-``（``10.216561000-0887.380021``、``10.10880964-17262412125005``），
    或后缀**以数字/点开头**（``10.11429789812771209_0109`` 猜成 ``10.114297898/12771209_0109``、
    ``10.10631.4907603`` 猜成 ``10.10631/.4907603``，实测这两种都进了知识库，且指向错误文章）
    ——这类切分法无从确定注册号，猜出来的 DOI 多半是错的，宁可留空。

    后缀以字母开头时可唯一还原（实测 ``10.1007bf00856879`` → ``10.1007/bf00856879``；
    ``10.2139ssrn.4480902``、``10.1360cjcp2006.19(2).126.5`` 同理，注册号就是前 4
    位数字）。
    """
    stem = str(stem or "").strip()
    m = re.match(r"^10\.\d{4,9}", stem)
    if not m:
        return None
    prefix, rest = m.group(0), stem[m.end():]
    if not rest or "-" in rest or not re.match(r"^[A-Za-z]", rest):
        return None
    return prefix + "/" + rest


#: 排版软件/扫描件痕迹的文件名后缀（PDF 首页最大字号行常是这类留下的大字行）
_ARTIFACT_SUFFIX_RE = re.compile(
    r"\.(?:dvi|fm|tex|docx?|pmd|qxd|indd|tiff?|eps|ps|jpe?g|png)\s*$", re.IGNORECASE)
#: 出版社内部编号（Elsevier 的 ``PII: 0031-9201(82)90121-2``）
_PII_RE = re.compile(r"^pii:\s*\S+", re.IGNORECASE)
#: 带 DOI 标签的整行（``doi:10.1016/...``，实测被当成标题进了知识库）
_DOI_LABEL_RE = re.compile(r"^(?:doi|https?://(?:dx\.)?doi\.org/)\s*:?\s*10\.", re.IGNORECASE)
#: arXiv 预印本戳（``arXiv:2310.02737v2  [nlin.SI]  7 Feb 2024``）
_ARXIV_STAMP_RE = re.compile(r"^arxiv:\s*\S+", re.IGNORECASE)
#: PDF 私用区字形（页眉装饰，如 ``\ue929 Online \ue92d``）
_DECORATION_RE = re.compile(r"[\ue000-\uf8ff]")
#: 版面导航/样板词（整行只有这些词时不是论文标题）
_NAV_TITLES = frozenset({
    "abstract", "online", "view online", "export citation", "citation", "title",
    "untitled", "标题", "preprint not peer reviewed", "keywords", "contents",
    "full text", "download details",
})

#: 首页"封面块"的分隔标记：其后的内容都是下载/引用信息，不属于标题。
#: 实测 AIP 下载页把标题与引用信息排成同一字号，取出来是
#: ``Structure-enhanced yield stress of magnetorheological fluids Citation: Journal of
#: Applied Physics 87, 2634 (2000); doi: ... View online: ... Published by ...``；
#: 在标记处切掉即得真标题（该标记不可能出现在标题里）。
_COVER_MARKER_RE = re.compile(
    r"\s*(?:Citation:|To cite this article|View online|View Table of Contents|"
    r"Download details|Published by|Articles you may be interested|You may also like|"
    r"This content has been downloaded|Contents lists available)",
    re.IGNORECASE)


def is_placeholder_title(text: str) -> bool:
    """判断一段"标题"其实是占位物：排版软件痕迹、文件名、DOI、版面导航/样板行。

    旧判据只认 ``Microsoft Word``/``*.pdf``/``*.doc``/DOI 形态，实测漏掉一大类
    （16 篇，全部由 PDF 元数据或首页最大字号行给出）：``05[41-47]-HJ Choi.fm``、
    ``jae1371cc.dvi``、``full-tpl13.dvi``、``ComTech2104009Ponomarenko.fm``、
    ``doi:10.1016/j.actamat.2006.01.007``、``PII: 0304-8853(93)91037-8``、
    ``arXiv:2310.02737v2 ...``、``Preprint not peer reviewed``、``Abstract``、
    ``标题``、``803_1.tif``（扫描件）、``\ue929 Online \ue92d``（装饰字形拼出的
    "AIP 页面导航"）。
    判为占位物后由 :func:`resolve_title` 的回退链继续往下找（首页字号 → 文件名 → 空串）。
    """
    s = str(text or "").strip()
    if not s:
        return True
    low = s.lower()
    if (low.startswith("microsoft word") or low.endswith(".pdf")
            or low.endswith(".doc") or _looks_like_doi(s)):
        return True
    cleaned = _DECORATION_RE.sub("", s).strip()   # 去掉装饰字形后再看剩下什么
    if len(cleaned) <= 1:                         # 整行只是装饰（如 "\ue92d"）
        return True
    if (_ARTIFACT_SUFFIX_RE.search(s) or _PII_RE.match(s)
            or _DOI_LABEL_RE.match(s) or _ARXIV_STAMP_RE.match(s)):
        return True
    return cleaned.lower() in _NAV_TITLES


def first_page_title(pdf_path: str, limit: int = 300) -> str:
    """用首页**最大字号**的那几行当标题（元数据 title 常为空或是排版软件文件名）。

    与 ``knowledge/tools/read_paper._first_page_title_candidate`` 同类启发式；这里另起
    一份是为了不让知识库模块反向依赖 knowledge/tools（那边顶层 import 了 requests/llm，
    而本模块的既有约定是重依赖一律懒加载）。

    占位行（版面导航/装饰字形/arXiv 戳等，见 :func:`is_placeholder_title`）不参与
    字号比较：实测 AIP 的下载页把 "View Online / Export Citation" 排成最大字号，
    取出来的"标题"是 ``\\ue929 Online \\ue92d``；滤掉这些行后同一页最大字号落到真正的
    标题上（``Effect of particle shape in magnetorheology``）。滤行只影响"最大字号
    恰好是占位行"的那几篇，对正常论文没有影响。
    """
    try:
        import pymupdf
        doc = pymupdf.open(pdf_path)
    except Exception:
        return ""
    try:
        data = doc.load_page(0).get_text("dict")
    except Exception:
        return ""
    finally:
        doc.close()
    largest, lines = 0.0, []
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans") or []
            text = "".join((s.get("text") or "") for s in spans).strip()
            if not text or is_placeholder_title(text):
                continue
            size = max((s.get("size") or 0.0) for s in spans) if spans else 0.0
            if size > largest + 0.1:
                largest, lines = size, [text]
            elif lines and abs(size - largest) <= 0.1:
                lines.append(text)
    joined = " ".join(lines).strip()
    # 同一字号里混进了封面块（引用/下载信息）时按标记切掉，只留标题部分
    return _COVER_MARKER_RE.split(joined, maxsplit=1)[0].strip()[:limit]


def resolve_title(explicit: str, meta_title: str, page_title: str, stem: str) -> str:
    """标题回退链：显式传入 → PDF 元数据 → 首页最大字号 → 文件名（不像 DOI 时）→ 空串。

    **绝不用 DOI 冒充标题**：历史实现在两者都取不到时退化成 ``doi or stem``，于是
    report.md 的参考文献出现 ``10.216561000-0887.380021.pdf`` 这种"标题"（实测），
    读者无从判断是哪篇文献，模型也无法据此判断相关性。
    """
    for cand in (explicit, meta_title, page_title):
        if not is_placeholder_title(cand):
            return str(cand).strip()
    if stem and not _looks_like_doi(stem):
        return stem
    return ""


def resolve_doi(text: str, stem: str, old: str = "") -> str:
    """DOI 推断链：正文印刷值 → 无歧义文件名恢复 → ``old``（原值）→ 空串。

    两条护栏：
    * 文件名是 DOI 形态、但正文印刷的 DOI 与它对不上时，**不信印刷值**——实测
      ``10.10631.4907603 .pdf`` 里印的是 ``10.1088/1361-665X/aa549c``（IOP 下载包装页
      上另一篇文章的 DOI），此前的实现会把它当成本文 DOI 写进知识库；
    * 恢复文件名 DOI 时只在无歧义时给值（见 :func:`_recover_doi`），否则留空——
      宁可没有 DOI，也不要一个指向别处的假 DOI（实测 ``10.114297898/12771209_0109``
      是从 ``10.11429789812771209_0109.pdf`` 猜出来的，真值是
      ``10.1142/9789812771209_0109``）。
    """
    stem = str(stem or "").strip()
    printed = doi_from_pdf_text(text)
    if printed and _looks_like_doi(stem) and _doi_key(printed) != _doi_key(stem):
        printed = None
    if printed:
        return printed
    if old and old.replace("/", "").strip() == stem:
        return _recover_doi(stem) or ""
    return old or (_recover_doi(stem) or "")


def safe_id(name: str) -> str:
    """把任意字符串转成 Chroma 合法 id 片段。"""
    return re.sub(r"[^0-9A-Za-z_-]", "_", name)

