"""Thin compatibility wrapper for ``data_get.collector nh-local-images``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from data_get.collector import CollectionConfig, legacy_main, run_collection
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from collector import CollectionConfig, legacy_main, run_collection


BASE_URL = "https://nhentai.net"
ROOT_DIR = "output"
INPUT_FILE = "data/local_data/NH_2.txt"
ERROR_LOG = "logs/collection/NH_images_local.errors.jsonl"
STATE_FILE = "logs/collection/NH_images_local.state.jsonl"
MAX_PAGE_LIMIT = 200
MAX_WORKERS = 4
REQUEST_INTERVAL_SECONDS = 0.0
PAGE_RETRY_TIMES = 3


def main(
    *,
    input_file: str | os.PathLike[str] | None = None,
    root_dir: str | os.PathLike[str] | None = None,
    error_log: str | os.PathLike[str] | None = None,
    state_file: str | os.PathLike[str] | None = None,
    max_pages: int | None = None,
    workers: int | None = None,
    request_attempts: int | None = None,
    max_rounds: int = 0,
    retry_backoff: float = 2.0,
    interval: float | None = None,
):
    return run_collection(
        CollectionConfig(
            mode="nh-local-images",
            base_url=BASE_URL,
            input_file=input_file or INPUT_FILE,
            output_dir=root_dir or ROOT_DIR,
            error_log=error_log or ERROR_LOG,
            state_file=state_file or STATE_FILE,
            max_pages=max_pages or MAX_PAGE_LIMIT,
            workers=workers or MAX_WORKERS,
            request_attempts=request_attempts or PAGE_RETRY_TIMES,
            max_rounds=max_rounds,
            retry_backoff=retry_backoff,
            interval=REQUEST_INTERVAL_SECONDS if interval is None else interval,
        )
    )


if __name__ == "__main__":
    raise SystemExit(legacy_main("nh-local-images", sys.argv[1:]))
