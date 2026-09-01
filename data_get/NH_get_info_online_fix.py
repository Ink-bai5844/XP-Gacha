"""Deprecated compatibility entry for the former NH failure-log retry tool.

Retrying is now intrinsic to :mod:`data_get.collector`; this module no longer
reads an error log or owns a second collection implementation.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable


def main(argv: Iterable[str] | None = None) -> int:
    try:
        from data_get.collector import legacy_main
    except ModuleNotFoundError:
        from collector import legacy_main

    print("[弃用] NH 专用失败重试脚本已合并到统一采集器；本次直接恢复 NH 在线采集。", flush=True)
    return legacy_main("nh-online", list(argv) if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
