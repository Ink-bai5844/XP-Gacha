"""Deprecated compatibility entry for the former JM failure-log retry tool."""

from __future__ import annotations

import sys
from collections.abc import Iterable


def retry_failed_pages(argv: Iterable[str] | None = None) -> int:
    try:
        from data_get.collector import legacy_main
    except ModuleNotFoundError:
        from collector import legacy_main

    print("[弃用] JM 专用失败重试脚本已合并到统一采集器；本次直接恢复 JM 在线采集。", flush=True)
    return legacy_main("jm-online", list(argv) if argv is not None else sys.argv[1:])


def main(argv: Iterable[str] | None = None) -> int:
    return retry_failed_pages(argv)


if __name__ == "__main__":
    raise SystemExit(main())
