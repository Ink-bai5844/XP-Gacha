"""Compatibility wrapper for the unified NH online collector.

New code should use ``python -m data_get.collector nh-online`` or import
``run_collection``.  The constants and ``main`` signature below keep existing
automation working without retaining a second crawler implementation.
"""

from __future__ import annotations

import os
import re
import sys
import time  # Retained for callers that patch the legacy module clock.
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from curl_cffi import requests

try:
    from data_get.collector import (
        CollectionConfig,
        CollectionItem,
        CollectionRequestError,
        NHAdapter,
        build_page_url,
        legacy_main,
        run_collection,
    )
    from data_get.proxy_config import build_proxies, configured_proxies
except ModuleNotFoundError:
    from collector import (
        CollectionConfig,
        CollectionItem,
        CollectionRequestError,
        NHAdapter,
        build_page_url,
        legacy_main,
        run_collection,
    )
    from proxy_config import build_proxies, configured_proxies


BASE_URL = "https://nhentai.net"
START_URL = f"{BASE_URL}/language/chinese/?sort=date"
IMG_DIR = "onlineimgtmp"
OUTPUT_CSV = "data/gallery_info_origin/NH_info_chinese.csv"
ERROR_LOG = "logs/collection/NH_info_chinese.errors.jsonl"
MAX_PAGE = 1
MAX_WORKERS = 10
LOOP_CRAWL = True
ID_COLUMN = "ID"
LINK_COLUMN = "链接"
CSV_HEADERS = [ID_COLUMN, LINK_COLUMN, "标题", "标签", "作者", "团队", "语言", "页数", "上传日期"]
ONLINE_COVER_PROXY = os.getenv("ONLINE_COVER_PROXY", "")
PROXIES = configured_proxies()


def extract_nh_id(url: str) -> str:
    match = re.search(r"/g/(\d+)/?", str(url or ""))
    return f"NH{match.group(1)}" if match else ""


def get_page_urls(page_num: int, retries: int = 3, start_url: str | None = None, base_url: str | None = None):
    """Small legacy helper retained for callers and proxy regression tests."""

    page_url = build_page_url(start_url or START_URL, page_num)
    target_base = (base_url or BASE_URL).rstrip("/") + "/"
    for _attempt in range(max(1, retries)):
        try:
            response = requests.get(
                page_url,
                impersonate="chrome120",
                proxies=PROXIES,
                timeout=30,
            )
            if response.status_code == 404:
                return None
            if response.status_code != 200:
                raise ValueError(f"HTTP {response.status_code}")
            soup = BeautifulSoup(response.text, "html.parser")
            items = []
            for gallery in soup.find_all("div", class_="gallery"):
                anchor = gallery.find("a", class_="cover")
                if not anchor or not anchor.get("href"):
                    continue
                detail_url = urljoin(target_base, str(anchor["href"]))
                gallery_id = extract_nh_id(detail_url)
                if not gallery_id:
                    continue
                image = gallery.find("img")
                thumb = ""
                if image:
                    thumb = str(image.get("data-src") or image.get("data-original") or image.get("src") or "")
                    thumb = urljoin(page_url, thumb)
                items.append({"id": gallery_id, "url": detail_url, "thumb_url": thumb})
            return items
        except Exception:
            continue
    return False


class _LegacyNHAdapter(NHAdapter):
    def discover_page(self, page: int) -> list[CollectionItem]:
        result = get_page_urls(page, retries=1, start_url=self.config.start_url, base_url=self.config.base_url)
        if result is False:
            raise CollectionRequestError(f"列表页请求失败: {build_page_url(self.config.start_url, page)}")
        if result is None:
            raise CollectionRequestError("列表页不存在", retryable=False, status_code=404)
        return [CollectionItem(item["id"], item["url"], item["thumb_url"], page) for item in result]


def main(
    max_page: int | None = None,
    start_url: str | None = None,
    base_url: str | None = None,
    output_csv: str | os.PathLike[str] | None = None,
    image_dir: str | os.PathLike[str] | None = None,
    error_log: str | os.PathLike[str] | None = None,
    max_workers: int | None = None,
    loop: bool | None = None,
):
    config = CollectionConfig(
        mode="nh-online",
        base_url=base_url or BASE_URL,
        start_url=start_url or START_URL,
        max_pages=max_page or MAX_PAGE,
        output_csv=output_csv or OUTPUT_CSV,
        image_dir=image_dir or IMG_DIR,
        error_log=error_log or ERROR_LOG,
        state_file=Path(error_log or ERROR_LOG).with_suffix(".state.jsonl"),
        workers=max_workers or MAX_WORKERS,
        request_attempts=1,
        max_rounds=0 if loop is not False else 1,
    )
    resolved = config.resolved()
    summary = run_collection(config, adapter=_LegacyNHAdapter(resolved))
    if summary.failed_pages and summary.discovered == 0:
        raise RuntimeError(f"列表页请求全部失败（{summary.failed_pages} 页）；请检查网络和 ONLINE_COVER_PROXY 配置")
    return summary


if __name__ == "__main__":
    raise SystemExit(legacy_main("nh-online", sys.argv[1:]))
