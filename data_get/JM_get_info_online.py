"""Thin compatibility wrapper for the unified JM online collector."""

from __future__ import annotations

import os
import re
import sys

try:
    from data_get.collector import CollectionConfig, legacy_main, run_collection
except ModuleNotFoundError:
    from collector import CollectionConfig, legacy_main, run_collection


MAX_WORKERS = 5
BASE_URL = "https://18comic.vip"
START_URL = "https://18comic.vip/search/photos?search_query=%E7%99%BE%E5%90%88"
MAX_PAGES = 80
OUTPUT_DIR = "onlineimgtmp"
CSV_PATH = "data/gallery_info_origin/JM_info_yuri.csv"
ERROR_LOG_PATH = "logs/collection/JM_info_yuri.errors.jsonl"
STATE_FILE = "logs/collection/JM_info_yuri.state.jsonl"
ID_COLUMN = "ID"
LINK_COLUMN = "链接"
CSV_HEADERS = [ID_COLUMN, LINK_COLUMN, "标题", "标签", "作者", "团队", "语言", "页数", "上传日期"]


def build_jm_id(raw_id: str) -> str:
    value = str(raw_id or "").strip()
    return value if value.startswith("JM") else (f"JM{value}" if value else "")


def extract_jm_id_from_url(url: str) -> str:
    match = re.search(r"/album/(\d+)/?", str(url or ""))
    return build_jm_id(match.group(1)) if match else ""


def scrape_18comic(
    *,
    base_url: str | None = None,
    start_url: str | None = None,
    max_pages: int | None = None,
    csv_path: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    workers: int | None = None,
    request_attempts: int = 3,
    max_rounds: int = 0,
    retry_backoff: float = 2.0,
    state_file: str | os.PathLike[str] | None = None,
    error_log: str | os.PathLike[str] | None = None,
):
    return run_collection(
        CollectionConfig(
            mode="jm-online",
            base_url=base_url or BASE_URL,
            start_url=start_url or START_URL,
            max_pages=max_pages or MAX_PAGES,
            output_csv=csv_path or CSV_PATH,
            image_dir=output_dir or OUTPUT_DIR,
            workers=workers or MAX_WORKERS,
            request_attempts=request_attempts,
            max_rounds=max_rounds,
            retry_backoff=retry_backoff,
            state_file=state_file or STATE_FILE,
            error_log=error_log or ERROR_LOG_PATH,
        )
    )


def main(**kwargs):
    return scrape_18comic(**kwargs)


if __name__ == "__main__":
    raise SystemExit(legacy_main("jm-online", sys.argv[1:]))
