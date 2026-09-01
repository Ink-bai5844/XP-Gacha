"""Thin compatibility wrapper for ``data_get.collector nh-local-info``."""

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
INPUT_FILE = "data/local_data/NH_all.txt"
OUTPUT_CSV = "data/gallery_info_origin/NH_info_local.csv"
IMG_DIR = "onlineimgtmp"
ERROR_LOG = "logs/collection/NH_info_local.errors.jsonl"
STATE_FILE = "logs/collection/NH_info_local.state.jsonl"
MAX_WORKERS = 5
REQUEST_INTERVAL_SECONDS = 0.0


def main(
    *,
    input_file: str | os.PathLike[str] | None = None,
    output_csv: str | os.PathLike[str] | None = None,
    image_dir: str | os.PathLike[str] | None = None,
    error_log: str | os.PathLike[str] | None = None,
    state_file: str | os.PathLike[str] | None = None,
    workers: int | None = None,
    request_attempts: int = 3,
    max_rounds: int = 0,
    retry_backoff: float = 2.0,
    interval: float | None = None,
):
    return run_collection(
        CollectionConfig(
            mode="nh-local-info",
            base_url=BASE_URL,
            input_file=input_file or INPUT_FILE,
            output_csv=output_csv or OUTPUT_CSV,
            image_dir=image_dir or IMG_DIR,
            error_log=error_log or ERROR_LOG,
            state_file=state_file or STATE_FILE,
            workers=workers or MAX_WORKERS,
            request_attempts=request_attempts,
            max_rounds=max_rounds,
            retry_backoff=retry_backoff,
            interval=REQUEST_INTERVAL_SECONDS if interval is None else interval,
        )
    )


if __name__ == "__main__":
    raise SystemExit(legacy_main("nh-local-info", sys.argv[1:]))
