from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from data_get.collector import (
    CSV_HEADERS,
    CollectionConfig,
    CollectionItem,
    CollectionRequestError,
    GalleryInfo,
    JMAdapter,
    NHAdapter,
    ParsedGallery,
)


def nh_html(*, pages: bool = True, date: bool = True) -> str:
    return """
        <h1 class="title">Synthetic gallery</h1>
        <div class="tag-container field-name">Tags: <span class="name">fixture</span></div>
        <div class="tag-container field-name">Languages: <span class="name">Chinese</span></div>
        <div id="cover"><img src="/fixture.png"></div>
    """ + ('<div class="tag-container field-name">Pages: 24</div>' if pages else '') + (
        '<div class="tag-container field-name">Uploaded: <time datetime="2026-09-01"></time></div>' if date else ''
    )


def jm_html(*, pages: bool = True, date: bool = True) -> str:
    return """
        <h1 id="book-name">Synthetic gallery</h1>
        <span itemprop="genre" data-type="tags"><a>中文</a><a>fixture</a></span>
        <span itemprop="author" data-type="author"><a>Fixture author</a></span>
        <meta property="og:image" content="/fixture.png">
    """ + ('<div>頁數：24</div>' if pages else '') + (
        '<meta itemprop="datePublished" content="2026-09-01T12:00:00+08:00">' if date else ''
    )


class MetadataCompletenessParserTests(unittest.TestCase):
    def parse(self, adapter_type, html: str):
        mode = "nh-online" if adapter_type is NHAdapter else "jm-online"
        adapter = adapter_type(CollectionConfig(mode=mode, proxy="").resolved())
        adapter._get = Mock(return_value=SimpleNamespace(text=html))
        return adapter.fetch_detail(CollectionItem("NH123" if adapter_type is NHAdapter else "JM123", "https://example.test/detail/123"))

    def test_every_raw_column_requires_a_check_for_both_sources(self):
        row = GalleryInfo("NH1", "https://example.test/g/1", "title", "tag", "author", "group", "Chinese", "24", "2026-09-01").as_csv_row()
        for adapter_type in (NHAdapter, JMAdapter):
            self.assertTrue(adapter_type.is_info_complete(row))
            for field in CSV_HEADERS:
                with self.subTest(source=adapter_type.__name__, field=field):
                    self.assertFalse(adapter_type.is_info_complete({**row, field: ""}))

    def test_nh_reports_genuinely_absent_optional_fields(self):
        parsed = self.parse(NHAdapter, nh_html())
        self.assertEqual(set(parsed.unavailable_fields), {"作者", "团队"})
        self.assertFalse(NHAdapter.is_info_complete(parsed.info.as_csv_row()))
        NHAdapter.validate_detail(parsed)

    def test_nh_can_report_missing_one_core_field_on_otherwise_rich_page(self):
        for missing in ("页数", "上传日期"):
            parsed = self.parse(NHAdapter, nh_html(pages=missing != "页数", date=missing != "上传日期"))
            self.assertIn(missing, parsed.unavailable_fields)
            NHAdapter.validate_detail(parsed)

    def test_nh_missing_both_core_fields_is_not_a_trusted_detail(self):
        with self.assertRaises(CollectionRequestError):
            self.parse(NHAdapter, nh_html(pages=False, date=False))

    def test_nh_missing_core_requires_cover_and_independent_sections(self):
        for html in (
            nh_html(pages=False).replace('<div id="cover"><img src="/fixture.png"></div>', ''),
            '<h1 class="title">title</h1><div class="tag-container field-name">Pages: 24</div>',
        ):
            with self.subTest(html=html), self.assertRaises(CollectionRequestError):
                self.parse(NHAdapter, html)

    def test_nh_unparsed_nonempty_optional_content_is_retryable(self):
        html = nh_html().replace('<span class="name">fixture</span>', '<a>unrecognized markup</a>')
        with self.assertRaises(CollectionRequestError) as caught:
            self.parse(NHAdapter, html)
        self.assertTrue(caught.exception.retryable)

    def test_jm_traditional_pages_and_content_date(self):
        parsed = self.parse(JMAdapter, jm_html())
        self.assertEqual(parsed.info.pages, "24")
        self.assertEqual(parsed.info.uploaded_date, "2026-09-01")
        self.assertEqual(parsed.info.tags, "fixture")
        self.assertEqual(parsed.info.languages, "中文")
        self.assertEqual(parsed.unavailable_fields, ("团队",))
        JMAdapter.validate_detail(parsed)

    def test_jm_missing_pages_and_date_are_accepted_only_on_structured_detail(self):
        parsed = self.parse(JMAdapter, jm_html(pages=False, date=False))
        self.assertEqual(set(parsed.unavailable_fields), {"团队", "页数", "上传日期"})
        JMAdapter.validate_detail(parsed)

    def test_jm_explicit_group_is_filled(self):
        parsed = self.parse(JMAdapter, jm_html() + '<span data-type="group"><a>Fixture team</a></span>')
        self.assertEqual(parsed.info.groups, "Fixture team")
        self.assertEqual(parsed.unavailable_fields, ())
        self.assertTrue(JMAdapter.is_info_complete(parsed.info.as_csv_row()))

    def test_jm_text_date_and_simplified_pages_remain_supported(self):
        html = jm_html().replace('頁數', '页数').replace(
            '<meta itemprop="datePublished" content="2026-09-01T12:00:00+08:00">',
            '<span itemprop="datePublished">2026-09-01</span>',
        )
        parsed = self.parse(JMAdapter, html)
        self.assertEqual(parsed.info.pages, "24")
        self.assertEqual(parsed.info.uploaded_date, "2026-09-01")

    def test_title_only_or_challenge_page_never_certifies_missing_fields(self):
        for adapter_type, title in ((NHAdapter, '<h1 class="title">title</h1>'), (JMAdapter, '<h1 id="book-name">title</h1>')):
            for html in (title, title + '<meta property="og:image" content="/cover.png">',
                         (nh_html() if adapter_type is NHAdapter else jm_html()) + '<form id="challenge-form"></form>'):
                with self.subTest(source=adapter_type.__name__, html=html), self.assertRaises(CollectionRequestError):
                    self.parse(adapter_type, html)

    def test_jm_malformed_nonempty_values_are_not_source_omissions(self):
        for before, after in (("頁數：24", "頁數：24junk"), ("頁數：24", "頁數：0"),
                              ("2026-09-01T12:00:00+08:00", "2026-02-30"),
                              ("2026-09-01T12:00:00+08:00", "unknown")):
            with self.subTest(after=after), self.assertRaises(CollectionRequestError):
                self.parse(JMAdapter, jm_html().replace(before, after))

    def test_invalid_omission_claims_cannot_bypass_validation(self):
        info = GalleryInfo("NH1", "https://example.test/g/1", "title", "tag", "author", "group", "Chinese", "24", "2026-09-01")
        for claim in (("ID",), ("标题",), ("unknown",), ("页数",), ("团队",)):
            for adapter_type in (NHAdapter, JMAdapter):
                with self.subTest(claim=claim, source=adapter_type.__name__), self.assertRaises(CollectionRequestError):
                    adapter_type.validate_detail(ParsedGallery(info, unavailable_fields=claim))


if __name__ == "__main__":
    unittest.main()
