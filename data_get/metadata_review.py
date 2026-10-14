"""Durable, row-bound receipts for fields a valid source page did not provide.

These receipts are not CSV data and never invent values for missing fields.
Losing/corrupting a receipt only causes another detail request, never data loss.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


class MetadataReviewStore:
    VERSION = 1

    def __init__(self, csv_path: Path, fields: Iterable[str]):
        self.path = csv_path.with_name(csv_path.name + ".metadata-checks.jsonl")
        if self.path.resolve().parent != csv_path.resolve().parent:
            raise ValueError("信息检查记录必须与 CSV 位于同一目录，不能链接到外部目录")
        self.fields = tuple(fields)
        self._records: dict[str, dict] = {}
        self._lock = threading.Lock()
        if self.path.exists():
            with self.path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(record, dict) or record.get("version") != self.VERSION:
                        continue
                    gallery_id = record.get("id")
                    missing = record.get("unavailable_fields")
                    if (not isinstance(gallery_id, str) or not gallery_id
                            or not isinstance(record.get("fingerprint"), str)
                            or not isinstance(missing, list) or not missing
                            or any(not isinstance(key, str) or key not in self.fields[3:] for key in missing)):
                        continue
                    self._records[gallery_id] = record

    def fingerprint(self, row: dict[str, str]) -> str:
        # Translations and user notes do not invalidate a source-field check.
        content = {key: str(row.get(key) or "").strip() for key in self.fields}
        raw = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def unavailable(self, row: dict[str, str]) -> tuple[str, ...]:
        with self._lock:
            record = self._records.get(str(row.get("ID") or ""))
            if record is None or record["fingerprint"] != self.fingerprint(row):
                return ()
            return tuple(record["unavailable_fields"])

    def record(self, row: dict[str, str], unavailable_fields: Iterable[str]) -> None:
        missing = tuple(dict.fromkeys(unavailable_fields))
        if not missing:
            return
        if any(key not in self.fields[3:] for key in missing):
            raise ValueError("不能将 ID、链接或标题标记为站点缺失")
        if any(not str(row.get(key) or "").strip() for key in self.fields[:3]):
            raise ValueError("信息检查记录缺少 ID、链接或标题")
        record = {
            "version": self.VERSION,
            "id": row["ID"],
            "fingerprint": self.fingerprint(row),
            "unavailable_fields": list(missing),
            "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        encoded = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with self._lock:
            # A torn tail remains available for inspection but cannot swallow
            # the next complete receipt. Do not truncate somebody else's data.
            with self.path.open("ab+") as handle:
                if handle.tell():
                    handle.seek(-1, os.SEEK_END)
                    if handle.read(1) != b"\n":
                        handle.write(b"\n")
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            self._records[row["ID"]] = record
