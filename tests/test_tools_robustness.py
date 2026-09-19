"""tools 包健壮性回归测试（第 8 轮加固）。

覆盖：
- read_paper._download_to 原子写入：半截 PDF 不得污染本地缓存；
- read_paper._try_unpaywall 的 DOI URL 转义；
- read_paper.read_paper 对非法条目的逐个报错（不再整批崩溃、不触发网络）；
- search_paper.search_paper 对残缺 date-parts 的年份提取健壮性。
"""
import json
import os
import tempfile
import unittest
from unittest import mock

import requests

from drsr_420.knowledge.tools import read_paper as rp
from drsr_420.knowledge.tools import search_paper as sp


class _FakeResp:
    """伪 requests.Response：可控状态码 + 流式内容 + 中途断流。"""

    def __init__(self, chunks, status=200, fail_after=None):
        self._chunks = list(chunks)
        self.status_code = status
        self._fail_after = fail_after
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code}")

    def iter_content(self, chunk_size):
        for i, c in enumerate(self._chunks):
            if self._fail_after is not None and i >= self._fail_after:
                raise OSError("connection reset mid-stream")
            yield c

    def close(self):
        self.closed = True


class _FakeSession:
    def __init__(self, resp):
        self._resp = resp
        self.get_kwargs = None

    def get(self, url, **kwargs):
        self.get_kwargs = kwargs
        return self._resp


class DownloadToTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "paper.pdf")

    def tearDown(self):
        self.tmp.cleanup()

    def test_success_writes_full_file(self):
        sess = _FakeSession(_FakeResp([b"%PDF-1.7\n", b"body"]))
        ok = rp._download_to(self.path, "http://x/p.pdf", session=sess)
        self.assertTrue(ok)
        with open(self.path, "rb") as f:
            self.assertEqual(f.read(), b"%PDF-1.7\nbody")
        self.assertFalse(os.path.exists(self.path + ".part"))

    def test_mid_stream_failure_leaves_no_cache_poison(self):
        """断流时目标路径与 .part 临时文件都必须不存在（旧实现会留半截 PDF）。"""
        sess = _FakeSession(_FakeResp([b"%PDF-1.7\n", b"a", b"b"], fail_after=1))
        ok = rp._download_to(self.path, "http://x/p.pdf", session=sess)
        self.assertFalse(ok)
        self.assertFalse(os.path.exists(self.path))
        self.assertFalse(os.path.exists(self.path + ".part"))

    def test_non_pdf_magic_rejected(self):
        sess = _FakeSession(_FakeResp([b"<!DOCTYPE html>"]))
        self.assertFalse(rp._download_to(self.path, "http://x/p.pdf", session=sess))
        self.assertFalse(os.path.exists(self.path))

    def test_http_error_rejected(self):
        sess = _FakeSession(_FakeResp([b"%PDF"], status=403))
        self.assertFalse(rp._download_to(self.path, "http://x/p.pdf", session=sess))
        self.assertFalse(os.path.exists(self.path))


class UnpaywallQuotingTest(unittest.TestCase):
    def test_doi_is_url_quoted(self):
        captured = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            raise requests.exceptions.ConnectionError("stop here")

        with mock.patch.object(rp.requests, "get", side_effect=fake_get):
            ok = rp._try_unpaywall("10.1234/j.ab(cd)", "ignored.pdf")
        self.assertFalse(ok)
        self.assertIn("10.1234%2Fj.ab%28cd%29", captured["url"])


class ReadPaperInvalidEntriesTest(unittest.TestCase):
    def test_bad_entries_reported_per_item_without_network(self):
        """非法条目逐个回传错误；不得抛异常，也不得走进下载路径。"""
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(rp, "_download_pdf_by_doi",
                                   side_effect=AssertionError("must not download")):
                out = rp.read_paper(
                    [1, "ab", ["x"], ["a", "b", "c"], (b"t",)],
                    save_dir=tmp,
                )
        items = json.loads(out)
        self.assertEqual(len(items), 5)
        for msg in items:
            self.assertTrue(str(msg).startswith("非法条目"), msg)


class SearchPaperYearTest(unittest.TestCase):
    def _query_with(self, items):
        resp = mock.Mock()
        resp.raise_for_status = mock.Mock()
        resp.json.return_value = {"message": {"items": items}}
        with mock.patch.object(sp.requests, "get", return_value=resp):
            return json.loads(sp.search_paper("q"))

    def test_missing_or_malformed_date_parts_no_crash(self):
        out = self._query_with([
            {"DOI": "10.1/a", "title": ["A"], "published-print": {"date-parts": [[]]}},
            {"DOI": "10.1/b", "title": ["B"], "published-print": {"date-parts": []}},
            {"DOI": "10.1/c", "title": ["C"]},
            {"DOI": "10.1/d", "title": ["D"],
             "published-print": {"date-parts": [[2021, 5, 1]]}},
        ])
        self.assertEqual([r["year"] for r in out], [None, None, None, 2021])

    def test_title_and_authors_defaults(self):
        out = self._query_with([{"DOI": "10.1/a"}])
        self.assertEqual(out[0]["title"], "")
        self.assertEqual(out[0]["authors"], [])


class SearchPaperAbstractTest(unittest.TestCase):
    """摘要字段：它能决定模型是否升级去 read_paper，不能是摆设。

    实测背景：只有标题时模型判断不了相关性，于是反复换措辞重搜（单轮 54 次
    search_paper、0 次 read_paper）。修复要点有两个：把 abstract 真正取回来，
    并让返回结构固定带这个键。
    """

    def _capture_query(self, items):
        """跑一次 search_paper，返回 (解析后的结果, 实际发出的 query 参数)。"""
        captured = {}

        def fake_get(url, params=None, timeout=None):
            captured.update(params or {})
            resp = mock.Mock()
            resp.raise_for_status = mock.Mock()
            resp.json.return_value = {"message": {"items": items}}
            return resp

        with mock.patch.object(sp.requests, "get", side_effect=fake_get):
            out = json.loads(sp.search_paper("q"))
        return out, captured

    def test_select_is_not_sent_because_crossref_drops_abstract(self):
        """回归：写 select 时 Crossref 会静默忽略 abstract（实测返回的键里没有它）。

        所以请求里不得带 select，否则摘要恒为 None，"判断相关性"这一步永远走不通。
        """
        _out, captured = self._capture_query([])
        self.assertNotIn("select", captured)
        self.assertEqual(captured["rows"], 3)      # 默认条数收紧到 3

    def test_jats_abstract_is_detagged_and_truncated(self):
        raw = "<jats:p>Hello <jats:italic>world</jats:italic>.</jats:p>" + "x" * 1000
        out, _ = self._capture_query([{"DOI": "10.1/a", "title": ["A"], "abstract": raw}])
        abstract = out[0]["abstract"]
        self.assertTrue(abstract.startswith("Hello world ."))
        self.assertNotIn("<", abstract)
        self.assertLessEqual(len(abstract), sp._ABSTRACT_CHAR_LIMIT + 3)
        self.assertTrue(abstract.endswith("..."))

    def test_missing_abstract_keeps_the_key(self):
        out, _ = self._capture_query([{"DOI": "10.1/a"}])
        self.assertIn("abstract", out[0])
        self.assertIsNone(out[0]["abstract"])

    def test_clean_abstract_basics(self):
        self.assertEqual(sp._clean_abstract("<jats:p>short</jats:p>"), "short")
        self.assertIsNone(sp._clean_abstract(None))
        self.assertIsNone(sp._clean_abstract("   "))


if __name__ == "__main__":
    unittest.main()
