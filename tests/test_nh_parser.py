from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from bs4 import BeautifulSoup

from data_get.collector import (
    CollectionConfig,
    CollectionItem,
    CollectionRequestError,
    NHAdapter,
    _field_names,
)


def sample_html(classes: str = "tag-container field-name svelte-example") -> str:
    """Synthetic, non-content fixture for the public field markup contract."""
    return f"""
        <h1 class="title"><span class="pretty">Example fallback title</span></h1>
        <h2 class="title"><span class="pretty">Example title</span></h2>
        <div class="{classes}">Tags:
            <a><span class="name extra">sample tag</span><span class="count">123</span></a>
            <a><span class="name">another tag</span></a>
        </div>
        <div class="{classes}">Artists: <a><span class="name">Example artist</span></a></div>
        <div class="{classes}">Groups: <a><span class="name">Example group</span></a></div>
        <div class="{classes}">Languages: <a><span class="name">chinese</span></a></div>
        <div class="{classes}">Pages: <a><span class="name">24</span></a></div>
        <div class="{classes}">Uploaded:
            <time datetime="2026-09-27T01:02:03+00:00">some time ago</time>
        </div>
        <div id="cover"><img data-src="//images.example.test/sample.webp"></div>
    """


class NHParserTests(unittest.TestCase):
    def setUp(self):
        self.adapter = NHAdapter(CollectionConfig(mode="nh-online", proxy="").resolved())
        self.item = CollectionItem("NH123", "https://example.test/g/123/")

    def parse(self, html: str):
        self.adapter._get = Mock(return_value=SimpleNamespace(text=html))
        return self.adapter.fetch_detail(self.item)

    def test_field_class_tokens_allow_extra_classes_and_any_order(self):
        for classes in (
            "tag-container field-name",
            "tag-container field-name svelte-example",
            "svelte-example field-name unrelated tag-container",
        ):
            with self.subTest(classes=classes):
                parsed = self.parse(sample_html(classes))
                self.assertEqual(parsed.info.as_csv_row(), {
                    "ID": "NH123",
                    "链接": "https://example.test/g/123/",
                    "标题": "Example title",
                    "标签": "sample tag, another tag",
                    "作者": "Example artist",
                    "团队": "Example group",
                    "语言": "chinese",
                    "页数": "24",
                    "上传日期": "2026-09-27",
                })
                self.assertEqual(parsed.thumbnail_url, "https://images.example.test/sample.webp")
                self.assertTrue(NHAdapter.is_info_complete(parsed.info.as_csv_row()))

    def test_field_label_is_not_matched_inside_a_tag_name(self):
        soup = BeautifulSoup("""
            <div class="tag-container field-name">Tags: <span class="name">Pages: example</span></div>
            <div class="field-name tag-container">Pages: <span class="name">24</span></div>
        """, "html.parser")
        self.assertEqual(_field_names(soup, "Pages:"), "24")

    def test_title_only_response_is_retryable_parse_failure(self):
        with self.assertRaises(CollectionRequestError) as caught:
            self.parse('<h2 class="title"><span class="pretty">Example title</span></h2>')
        self.assertTrue(caught.exception.retryable)
        self.assertIn("页数", str(caught.exception))
        self.assertIn("上传日期", str(caught.exception))

    def test_optional_tag_fields_can_be_absent(self):
        parsed = self.parse("""
            <h1 class="title">Example title</h1>
            <div class="tag-container field-name">Pages: 24</div>
            <div class="tag-container field-name">Uploaded: <time datetime="2024-02-29"></time></div>
        """)
        self.assertEqual(parsed.info.pages, "24")
        for key in ("标签", "作者", "团队", "语言"):
            self.assertEqual(parsed.info.as_csv_row()[key], "")
        self.assertFalse(NHAdapter.is_info_complete(parsed.info.as_csv_row()))
        self.assertEqual(set(parsed.unavailable_fields), {"标签", "作者", "团队", "语言"})
        NHAdapter.validate_detail(parsed)

    def test_empty_preferred_title_uses_nonempty_fallback(self):
        parsed = self.parse(sample_html().replace("Example title", ""))
        self.assertEqual(parsed.info.title, "Example fallback title")

    def test_missing_title_is_rejected(self):
        html = sample_html().replace("Example title", "").replace("Example fallback title", "")
        with self.assertRaises(CollectionRequestError) as caught:
            self.parse(html)
        self.assertIn("标题", str(caught.exception))

    def test_invalid_page_count_and_upload_date_are_rejected(self):
        for original, replacement, field in (
            ('class="name">24', 'class="name">0', "页数"),
            ('class="name">24', 'class="name">24 pages', "页数"),
            ("2026-09-27T01:02:03+00:00", "2026-02-30T01:02:03+00:00", "上传日期"),
            ("2026-09-27T01:02:03+00:00", "yesterday", "上传日期"),
        ):
            with self.subTest(replacement=replacement):
                with self.assertRaises(CollectionRequestError) as caught:
                    self.parse(sample_html().replace(original, replacement))
                self.assertIn(field, str(caught.exception))

    def test_existing_csv_validation_uses_the_same_mandatory_fields(self):
        row = self.parse(sample_html()).info.as_csv_row()
        for field in ("ID", "链接", "标题", "页数", "上传日期"):
            for value in ("", "  ", None):
                with self.subTest(field=field, value=value):
                    self.assertFalse(NHAdapter.is_info_complete({**row, field: value}))
        for pages in ("0", "000", "-1", "1.5", "1e2", "24,000", "twenty", "２４"):
            with self.subTest(pages=pages):
                self.assertFalse(NHAdapter.is_info_complete({**row, "页数": pages}))
        for date in (
            "2026-02-29", "0000-01-01", "2026-13-01", "2026-01-01junk",
            "2026/2/29", "2026/01-01", "26/1/1", "2026/001/01", "2026/1/1 12:00:00",
        ):
            with self.subTest(date=date):
                self.assertFalse(NHAdapter.is_info_complete({**row, "上传日期": date}))
        self.assertTrue(NHAdapter.is_info_complete({**row, "页数": " 24 ", "上传日期": " 2024-02-29 "}))

    def test_existing_spreadsheet_dates_remain_complete(self):
        row = self.parse(sample_html()).info.as_csv_row()
        for date in ("2026/4/20", "2026/04/20", "2026-4-20", "2026-1-01", "2024/2/29"):
            with self.subTest(date=date):
                self.assertTrue(NHAdapter.is_info_complete({**row, "上传日期": date}))
        self.assertEqual(row["上传日期"], "2026-09-27")


if __name__ == "__main__":
    unittest.main()
