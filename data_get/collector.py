"""Unified, resumable NH/JM collection entry point.

The collector deliberately keeps the human-readable run log separate from its
machine-readable state.  CSV rows are keyed by ``ID`` and thumbnails are
written through a temporary file before an atomic replace, so rerunning or
terminating a job cannot create duplicate records or bless a partial image as
complete.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup

try:
    from data_get.proxy_config import build_proxies, configured_proxies
except ModuleNotFoundError:  # Direct execution from data_get/.
    from proxy_config import build_proxies, configured_proxies


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGIN_DIR = PROJECT_ROOT / "data" / "gallery_info_origin"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "onlineimgtmp"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "collection"
CSV_HEADERS = ["ID", "链接", "标题", "标签", "作者", "团队", "语言", "页数", "上传日期"]
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif"}
MODE_DEFAULTS: dict[str, dict[str, object]] = {
    "nh-online": {
        "base_url": "https://nhentai.net",
        "start_url": "https://nhentai.net/language/chinese/?sort=date",
        "max_pages": 1,
        "output_csv": ORIGIN_DIR / "NH_info_chinese.csv",
        "workers": 10,
    },
    "jm-online": {
        "base_url": "https://18comic.vip",
        "start_url": "https://18comic.vip/search/photos?search_query=%E7%99%BE%E5%90%88",
        "max_pages": 80,
        "output_csv": ORIGIN_DIR / "JM_info_yuri.csv",
        "workers": 5,
    },
    "nh-local-info": {
        "base_url": "https://nhentai.net",
        "input_file": PROJECT_ROOT / "data" / "local_data" / "NH_all.txt",
        "output_csv": ORIGIN_DIR / "NH_info_local.csv",
        "workers": 5,
    },
    "nh-local-images": {
        "base_url": "https://nhentai.net",
        "input_file": PROJECT_ROOT / "data" / "local_data" / "NH_2.txt",
        "output_dir": PROJECT_ROOT / "output",
        "max_pages": 200,
        "workers": 4,
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _project_path(value: str | os.PathLike[str] | None, default: Path | None = None) -> Path | None:
    if value is None or str(value).strip() == "":
        return default
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def build_page_url(start_url: str, page_number: int) -> str:
    if page_number <= 1:
        return start_url
    parts = urlsplit(start_url)
    query = parse_qsl(parts.query, keep_blank_values=True)
    replaced = False
    for index, (key, _value) in enumerate(query):
        if key == "page":
            query[index] = (key, str(page_number))
            replaced = True
            break
    if not replaced:
        query.append(("page", str(page_number)))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


class CollectionRequestError(RuntimeError):
    """A request or parse failure with an explicit retry classification."""

    def __init__(self, message: str, *, retryable: bool = True, status_code: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class StopRequested(RuntimeError):
    pass


@dataclass(slots=True)
class CollectionConfig:
    mode: str
    base_url: str = ""
    start_url: str = ""
    max_pages: int = 0
    output_csv: str | os.PathLike[str] | None = None
    image_dir: str | os.PathLike[str] | None = None
    input_file: str | os.PathLike[str] | None = None
    output_dir: str | os.PathLike[str] | None = None
    workers: int = 0
    request_attempts: int = 3
    max_rounds: int = 0
    retry_backoff: float = 2.0
    timeout: float = 30.0
    interval: float = 0.0
    state_file: str | os.PathLike[str] | None = None
    error_log: str | os.PathLike[str] | None = None
    proxy: str | None = None
    resume: bool = True

    def resolved(self) -> "ResolvedConfig":
        if self.mode not in MODE_DEFAULTS:
            raise ValueError(f"未知采集模式: {self.mode}")
        defaults = MODE_DEFAULTS[self.mode]
        output_csv = _project_path(self.output_csv, defaults.get("output_csv"))
        image_dir = _project_path(self.image_dir, DEFAULT_IMAGE_DIR)
        input_file = _project_path(self.input_file, defaults.get("input_file"))
        output_dir = _project_path(self.output_dir, defaults.get("output_dir"))
        max_pages = int(self.max_pages or defaults.get("max_pages", 1))
        workers = int(self.workers or defaults.get("workers", 5))
        if max_pages < 1:
            raise ValueError("max_pages 必须大于等于 1")
        if workers < 1:
            raise ValueError("workers 必须大于等于 1")
        if self.request_attempts < 1:
            raise ValueError("request_attempts 必须大于等于 1")
        if self.max_rounds < 0:
            raise ValueError("max_rounds 必须大于等于 0")
        if self.retry_backoff < 0 or self.timeout <= 0 or self.interval < 0:
            raise ValueError("timeout 必须为正数，backoff/interval 不得为负数")
        base_url = (self.base_url or str(defaults.get("base_url", ""))).rstrip("/")
        start_url = self.start_url or str(defaults.get("start_url", ""))
        identity = "|".join(
            [
                self.mode,
                base_url,
                start_url,
                str(max_pages),
                str(output_csv or ""),
                str(image_dir or ""),
                str(input_file or ""),
                str(output_dir or ""),
            ]
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        stem = (output_csv or output_dir or input_file or Path(self.mode)).stem
        state_file = _project_path(
            self.state_file,
            DEFAULT_LOG_DIR / f"{self.mode}-{stem}-{digest}.state.jsonl",
        )
        error_log = _project_path(
            self.error_log,
            DEFAULT_LOG_DIR / f"{self.mode}-{stem}-{digest}.errors.jsonl",
        )
        named_files = [
            ("output_csv", output_csv),
            ("input_file", input_file),
            ("state_file", state_file),
            ("error_log", error_log),
        ]
        seen_files: dict[str, str] = {}
        for name, path in named_files:
            if path is None:
                continue
            normalized = os.path.normcase(str(path.resolve()))
            previous = seen_files.get(normalized)
            if previous is not None:
                raise ValueError(f"{previous} 与 {name} 不能指向同一个文件: {path}")
            seen_files[normalized] = name
        for directory_name, directory in (("image_dir", image_dir), ("output_dir", output_dir)):
            if directory is None:
                continue
            normalized = os.path.normcase(str(directory.resolve()))
            file_role = seen_files.get(normalized)
            if file_role is not None:
                raise ValueError(f"文件参数 {file_role} 与目录参数 {directory_name} 不能指向同一路径: {directory}")
        return ResolvedConfig(
            mode=self.mode,
            base_url=base_url,
            start_url=start_url,
            max_pages=max_pages,
            output_csv=output_csv,
            image_dir=image_dir,
            input_file=input_file,
            output_dir=output_dir,
            workers=workers,
            request_attempts=int(self.request_attempts),
            max_rounds=int(self.max_rounds),
            retry_backoff=float(self.retry_backoff),
            timeout=float(self.timeout),
            interval=float(self.interval),
            state_file=state_file,
            error_log=error_log,
            proxy=self.proxy,
            resume=bool(self.resume),
            identity=digest,
        )


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    mode: str
    base_url: str
    start_url: str
    max_pages: int
    output_csv: Path | None
    image_dir: Path | None
    input_file: Path | None
    output_dir: Path | None
    workers: int
    request_attempts: int
    max_rounds: int
    retry_backoff: float
    timeout: float
    interval: float
    state_file: Path
    error_log: Path
    proxy: str | None
    resume: bool
    identity: str


@dataclass(slots=True)
class CollectionItem:
    id: str
    detail_url: str
    thumbnail_url: str = ""
    page: int = 0
    label: str = ""


@dataclass(slots=True)
class GalleryInfo:
    id: str
    link: str
    title: str
    tags: str = ""
    authors: str = ""
    groups: str = ""
    languages: str = ""
    pages: str = ""
    uploaded_date: str = ""

    def as_csv_row(self) -> dict[str, str]:
        return {
            "ID": self.id,
            "链接": self.link,
            "标题": self.title,
            "标签": self.tags,
            "作者": self.authors,
            "团队": self.groups,
            "语言": self.languages,
            "页数": self.pages,
            "上传日期": self.uploaded_date,
        }


@dataclass(slots=True)
class ParsedGallery:
    info: GalleryInfo
    thumbnail_url: str = ""


@dataclass(slots=True)
class BinaryPayload:
    content: bytes
    content_type: str = ""


@dataclass(slots=True)
class TaskState:
    item: CollectionItem
    info_ok: bool = False
    thumb_ok: bool = False
    terminal_info: bool = False
    terminal_thumb: bool = False

    @property
    def complete(self) -> bool:
        return self.info_ok and self.thumb_ok

    @property
    def terminal(self) -> bool:
        return (not self.info_ok and self.terminal_info) or (not self.thumb_ok and self.terminal_thumb)


@dataclass(slots=True)
class CollectionSummary:
    mode: str
    rounds: int = 0
    discovered: int = 0
    completed: int = 0
    pending: int = 0
    terminal: int = 0
    failed_pages: int = 0
    limit_reached: int = 0
    interrupted: bool = False
    output_csv: str = ""

    @property
    def success(self) -> bool:
        return (
            not self.interrupted
            and self.pending == 0
            and self.terminal == 0
            and self.failed_pages == 0
            and self.limit_reached == 0
        )

    @property
    def exit_code(self) -> int:
        if self.interrupted:
            return 130
        return 0 if self.success else 2

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["success"] = self.success
        result["exitCode"] = self.exit_code
        return result


class SiteAdapter(Protocol):
    def discover_page(self, page: int) -> list[CollectionItem]: ...

    def fetch_detail(self, item: CollectionItem) -> ParsedGallery: ...

    def fetch_thumbnail(self, url: str) -> BinaryPayload: ...


class JsonlWriter:
    def __init__(self, path: Path, *, flush_every: int = 100):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._handle = None
        self._buffered = 0
        self._flush_every = max(1, int(flush_every))

    def append(self, event: dict[str, object]) -> None:
        record = {"timestamp": _utc_now(), **event}
        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            if self._handle is None:
                self._handle = self.path.open("a", encoding="utf-8", newline="")
            self._handle.write(encoded)
            self._buffered += 1
            if self._buffered >= self._flush_every:
                self._handle.flush()
                self._buffered = 0

    def _close_locked(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None
            self._buffered = 0

    def close(self) -> None:
        with self._lock:
            self._close_locked()


@dataclass(slots=True)
class ReplayedState:
    tasks: dict[str, TaskState] = field(default_factory=dict)
    completed_pages: set[int] = field(default_factory=set)
    terminal_pages: set[int] = field(default_factory=set)
    limit_galleries: set[str] = field(default_factory=set)
    completed_galleries: set[str] = field(default_factory=set)
    local_discovery_complete: bool = False
    run_completed: bool = False


class Checkpoint(JsonlWriter):
    def replay(self, expected_identity: str | None = None) -> ReplayedState:
        """Replay only an active segment belonging to ``expected_identity``.

        A foreign ``run_start`` invalidates everything accumulated before it;
        a later matching ``run_start`` begins a fresh recoverable segment.  We
        therefore never resurrect stale state after a state file was reused by
        another collection identity.
        """

        replayed = ReplayedState()
        if not self.path.exists():
            return replayed
        try:
            handle = self.path.open("r", encoding="utf-8")
        except OSError:
            return replayed
        with handle:
            active_segment = expected_identity is None
            for line in handle:
                try:
                    event = json.loads(line)
                except (TypeError, json.JSONDecodeError):
                    continue  # Ignore a truncated tail left by a forced process stop.
                kind = event.get("event")
                if kind == "run_start":
                    recorded_identity = str(event.get("identity") or "")
                    matches = not expected_identity or not recorded_identity or recorded_identity == expected_identity
                    replayed = ReplayedState()
                    active_segment = matches
                    continue
                if not active_segment:
                    continue
                if kind == "run_resume":
                    recorded_identity = str(event.get("identity") or "")
                    if expected_identity and recorded_identity and recorded_identity != expected_identity:
                        active_segment = False
                        replayed = ReplayedState()
                elif kind == "page_complete":
                    replayed.completed_pages.add(int(event["page"]))
                elif kind == "page_terminal":
                    replayed.terminal_pages.add(int(event["page"]))
                elif kind == "gallery_limit":
                    replayed.limit_galleries.add(str(event["id"]))
                elif kind == "gallery_limit_clear":
                    replayed.limit_galleries.discard(str(event["id"]))
                elif kind == "gallery_complete":
                    replayed.completed_galleries.add(str(event["id"]))
                elif kind == "gallery_complete_clear":
                    replayed.completed_galleries.discard(str(event["id"]))
                elif kind == "local_discovery_complete":
                    replayed.local_discovery_complete = True
                elif kind == "task_state":
                    try:
                        raw_item = event["item"]
                        item = CollectionItem(**raw_item)
                        replayed.tasks[item.id] = TaskState(
                            item=item,
                            info_ok=bool(event.get("info_ok")),
                            thumb_ok=bool(event.get("thumb_ok")),
                            terminal_info=bool(event.get("terminal_info")),
                            terminal_thumb=bool(event.get("terminal_thumb")),
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
                elif kind == "task_delete":
                    replayed.tasks.pop(str(event.get("id") or ""), None)
                elif kind == "run_complete":
                    replayed.run_completed = True
        return replayed

    def task(self, state: TaskState) -> None:
        self.append(
            {
                "event": "task_state",
                "item": asdict(state.item),
                "info_ok": state.info_ok,
                "thumb_ok": state.thumb_ok,
                "terminal_info": state.terminal_info,
                "terminal_thumb": state.terminal_thumb,
            }
        )

    def delete_task(self, task_id: str) -> None:
        self.append({"event": "task_delete", "id": task_id})

    def compact(
        self,
        *,
        mode: str,
        identity: str,
        tasks: Iterable[TaskState],
        completed_pages: Iterable[int] = (),
        terminal_pages: Iterable[int] = (),
        limit_galleries: Iterable[str] = (),
        completed_galleries: Iterable[str] = (),
        local_discovery_complete: bool = False,
        run_completed: bool = False,
        summary: dict[str, object] | None = None,
    ) -> None:
        """Atomically replace an append-heavy checkpoint with its latest snapshot."""

        temp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")

        def write(handle, event: dict[str, object]) -> None:
            handle.write(json.dumps({"timestamp": _utc_now(), **event}, ensure_ascii=False, separators=(",", ":")) + "\n")

        with self._lock:
            self._close_locked()
            try:
                with temp.open("w", encoding="utf-8", newline="") as handle:
                    write(handle, {"event": "run_start", "mode": mode, "identity": identity})
                    for page in sorted(set(completed_pages)):
                        write(handle, {"event": "page_complete", "page": page})
                    for page in sorted(set(terminal_pages)):
                        write(handle, {"event": "page_terminal", "page": page})
                    for gallery_id in sorted(set(limit_galleries)):
                        write(handle, {"event": "gallery_limit", "id": gallery_id})
                    for gallery_id in sorted(set(completed_galleries)):
                        write(handle, {"event": "gallery_complete", "id": gallery_id})
                    if local_discovery_complete:
                        write(handle, {"event": "local_discovery_complete"})
                    for state in tasks:
                        write(
                            handle,
                            {
                                "event": "task_state",
                                "item": asdict(state.item),
                                "info_ok": state.info_ok,
                                "thumb_ok": state.thumb_ok,
                                "terminal_info": state.terminal_info,
                                "terminal_thumb": state.terminal_thumb,
                            },
                        )
                    write(
                        handle,
                        {
                            "event": "run_complete" if run_completed else "run_paused",
                            "summary": summary or {},
                        },
                    )
                os.replace(temp, self.path)
            finally:
                try:
                    temp.unlink()
                except FileNotFoundError:
                    pass


class CsvStore:
    """An ID-keyed UTF-8-SIG CSV store with duplicate-safe upserts."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.fieldnames = list(CSV_HEADERS)
        self.rows: dict[str, dict[str, str]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            self._rewrite_locked()
            return
        needs_rewrite = False
        with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fields = [name for name in (reader.fieldnames or []) if name]
            missing_required = [name for name in ("ID", "链接", "标题") if name not in existing_fields]
            if missing_required:
                missing = "、".join(missing_required)
                raise ValueError(f"非空 CSV 缺少必需表头: {missing}（不会修改原文件）")
            self.fieldnames = list(CSV_HEADERS) + [name for name in existing_fields if name not in CSV_HEADERS]
            try:
                for row in reader:
                    required = [row.get("ID"), row.get("链接"), row.get("标题")]
                    # DictReader uses None values/keys for truncated or over-wide
                    # rows.  Such a tail must never count as completed metadata.
                    if None in row or any(value is None or not str(value).strip() for value in required):
                        needs_rewrite = True
                        continue
                    gallery_id = str(row["ID"]).strip()
                    if gallery_id in self.rows:
                        needs_rewrite = True
                    self.rows[gallery_id] = {name: str(row.get(name, "") or "") for name in self.fieldnames}
            except csv.Error:
                needs_rewrite = True
        if needs_rewrite or existing_fields != self.fieldnames:
            self._rewrite_locked()

    def has(self, gallery_id: str) -> bool:
        with self._lock:
            return gallery_id in self.rows

    def upsert(self, info: GalleryInfo) -> bool:
        row = info.as_csv_row()
        with self._lock:
            old = self.rows.get(info.id)
            merged = {name: (row.get(name, old.get(name, "") if old else "")) for name in self.fieldnames}
            if old == merged:
                return False
            self.rows[info.id] = merged
            self._dirty = True
            return True

    def commit(self) -> None:
        """Commit all upserts as one atomic CSV replacement.

        A forced process stop can at worst leave an ignored ``.tmp`` file; it
        can never leave half a CSV row that would look complete on the next run.
        """

        with self._lock:
            if self._dirty:
                self._rewrite_locked()

    def _rewrite_locked(self) -> None:
        temp_path = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(self.rows.values())
            os.replace(temp_path, self.path)
            self._dirty = False
        finally:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


class ThumbnailStore:
    """A thumbnail index built with one directory scan, never one scan per task."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.ids: set[str] = set()
        with os.scandir(self.directory) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                suffix = Path(entry.name).suffix.lower()
                try:
                    valid = suffix in VALID_IMAGE_SUFFIXES and _validate_image_file(Path(entry.path))
                except OSError:
                    valid = False
                if valid:
                    self.ids.add(Path(entry.name).stem)

    def has(self, gallery_id: str) -> bool:
        with self._lock:
            return gallery_id in self.ids

    @staticmethod
    def _extension(url: str, content_type: str) -> str:
        suffix = Path(urlsplit(url).path).suffix.lower()
        if suffix in VALID_IMAGE_SUFFIXES:
            return ".jpg" if suffix == ".jpeg" else suffix
        mime = content_type.partition(";")[0].strip().lower()
        return {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/avif": ".avif",
        }.get(mime, ".jpg")

    def save(self, gallery_id: str, url: str, payload: BinaryPayload) -> Path:
        _validate_image_payload(payload)
        extension = self._extension(url, payload.content_type)
        target = self.directory / f"{gallery_id}{extension}"
        temp = self.directory / f".{gallery_id}.{uuid.uuid4().hex}.part"
        try:
            with temp.open("wb") as handle:
                handle.write(payload.content)
            os.replace(temp, target)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
        with self._lock:
            self.ids.add(gallery_id)
        return target


def _image_kind(content: bytes) -> str:
    if content.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "webp"
    if len(content) >= 12 and content[4:8] == b"ftyp" and content[8:12] in {b"avif", b"avis"}:
        return "avif"
    return ""


def _validate_image_file(path: Path) -> bool:
    """Validate one existing candidate without trusting its suffix or size."""

    try:
        if path.stat().st_size <= 0:
            return False
        with path.open("rb") as handle:
            prefix = handle.read(32)
        kind = _image_kind(prefix)
        if not kind:
            return False
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError, SyntaxError):
        return False
    except Exception:
        # Pillow raises format/plugin-specific exception classes.  Existing
        # files are an optimization only, so uncertainty must mean "missing".
        return False


def _validate_image_payload(payload: BinaryPayload) -> None:
    content = payload.content
    if not content:
        raise CollectionRequestError("图片响应内容为空")
    mime = payload.content_type.partition(";")[0].strip().lower()
    if mime == "text/html" or (mime and not mime.startswith("image/") and mime != "application/octet-stream"):
        raise CollectionRequestError(f"图片响应 Content-Type 非图片: {mime}")
    kind = _image_kind(content[:32])
    if not kind:
        raise CollectionRequestError("图片响应未通过标准图片魔数校验")
    try:
        from PIL import Image

        with Image.open(io.BytesIO(content)) as image:
            image.verify()
    except Exception as exc:
        raise CollectionRequestError(f"图片内容损坏或格式无效: {exc}") from exc


def _response_error(url: str, status_code: int) -> CollectionRequestError:
    retryable = status_code not in {400, 401, 404, 410}
    return CollectionRequestError(
        f"HTTP {status_code}: {url}", retryable=retryable, status_code=status_code
    )


def _field_names(soup: BeautifulSoup, field_name: str) -> str:
    for container in soup.find_all("div", class_="tag-container field-name"):
        if field_name in container.get_text(" ", strip=True):
            return ", ".join(span.get_text(strip=True) for span in container.find_all("span", class_="name"))
    return ""


def _nh_id(url: str) -> str:
    match = re.search(r"/g/(\d+)/?", url)
    return f"NH{match.group(1)}" if match else ""


def _jm_id(url: str) -> str:
    match = re.search(r"/album/(\d+)/?", url)
    return f"JM{match.group(1)}" if match else ""


class NHAdapter:
    def __init__(self, config: ResolvedConfig):
        self.config = config
        self.proxies = build_proxies(config.proxy) if config.proxy is not None else configured_proxies()

    def _get(self, url: str):
        from curl_cffi import requests

        response = requests.get(
            url,
            impersonate="chrome120",
            proxies=self.proxies,
            timeout=self.config.timeout,
        )
        if response.status_code != 200:
            raise _response_error(url, int(response.status_code))
        return response

    def discover_page(self, page: int) -> list[CollectionItem]:
        page_url = build_page_url(self.config.start_url, page)
        response = self._get(page_url)
        soup = BeautifulSoup(response.text, "html.parser")
        items: list[CollectionItem] = []
        for gallery in soup.find_all("div", class_="gallery"):
            anchor = gallery.find("a", class_="cover")
            if not anchor or not anchor.get("href"):
                continue
            detail_url = urljoin(self.config.base_url + "/", str(anchor["href"]))
            gallery_id = _nh_id(detail_url)
            if not gallery_id:
                continue
            image = gallery.find("img")
            thumbnail_url = ""
            if image:
                thumbnail_url = str(
                    image.get("data-src") or image.get("data-original") or image.get("src") or ""
                )
                thumbnail_url = urljoin(page_url, thumbnail_url)
            items.append(CollectionItem(gallery_id, detail_url, thumbnail_url, page))
        return items

    def fetch_detail(self, item: CollectionItem) -> ParsedGallery:
        response = self._get(item.detail_url)
        soup = BeautifulSoup(response.text, "html.parser")
        title_node = soup.find("h2", class_="title") or soup.find("h1", class_="title")
        if title_node and title_node.find("span", class_="pretty"):
            title = title_node.find("span", class_="pretty").get_text(strip=True)
        else:
            title = title_node.get_text(" ", strip=True) if title_node else ""
        if not title:
            raise CollectionRequestError(f"详情页未解析到标题: {item.detail_url}")
        upload_date = ""
        for container in soup.find_all("div", class_="tag-container field-name"):
            if "Uploaded:" in container.get_text(" ", strip=True):
                time_node = container.find("time")
                if time_node and time_node.get("datetime"):
                    upload_date = str(time_node["datetime"])[:10]
                    break
        cover = soup.select_one("#cover img")
        thumbnail_url = ""
        if cover:
            thumbnail_url = str(cover.get("data-src") or cover.get("src") or "")
        if not thumbnail_url:
            meta = soup.find("meta", attrs={"property": "og:image"})
            thumbnail_url = str(meta.get("content") or "") if meta else ""
        return ParsedGallery(
            GalleryInfo(
                id=item.id,
                link=item.detail_url,
                title=title,
                tags=_field_names(soup, "Tags:"),
                authors=_field_names(soup, "Artists:"),
                groups=_field_names(soup, "Groups:"),
                languages=_field_names(soup, "Languages:"),
                pages=_field_names(soup, "Pages:"),
                uploaded_date=upload_date,
            ),
            urljoin(item.detail_url, thumbnail_url) if thumbnail_url else "",
        )

    def fetch_thumbnail(self, url: str) -> BinaryPayload:
        response = self._get(url)
        headers = getattr(response, "headers", {}) or {}
        return BinaryPayload(bytes(response.content), str(headers.get("content-type", "")))


class JMAdapter:
    LANGUAGE_TAGS = {"中文", "英文", "日文"}

    def __init__(self, config: ResolvedConfig):
        self.config = config
        self.proxies = build_proxies(config.proxy) if config.proxy is not None else configured_proxies()
        self._local = threading.local()

    def _scraper(self):
        scraper = getattr(self._local, "scraper", None)
        if scraper is None:
            import cloudscraper

            scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "desktop": True}
            )
            self._local.scraper = scraper
        return scraper

    def _get(self, url: str):
        response = self._scraper().get(url, timeout=self.config.timeout, proxies=self.proxies)
        if response.status_code != 200:
            raise _response_error(url, int(response.status_code))
        return response

    def discover_page(self, page: int) -> list[CollectionItem]:
        page_url = build_page_url(self.config.start_url, page)
        response = self._get(page_url)
        soup = BeautifulSoup(response.text, "html.parser")
        items: list[CollectionItem] = []
        for node in soup.select("div.list-col"):
            anchor = node.select_one('a[href^="/album/"]')
            if not anchor:
                continue
            detail_url = urljoin(self.config.base_url + "/", str(anchor.get("href", "")))
            gallery_id = _jm_id(detail_url)
            if not gallery_id:
                continue
            image = anchor.select_one("img")
            thumbnail_url = ""
            if image:
                thumbnail_url = str(image.get("data-original") or image.get("src") or "")
                thumbnail_url = urljoin(page_url, thumbnail_url)
            items.append(CollectionItem(gallery_id, detail_url, thumbnail_url, page))
        return items

    def fetch_detail(self, item: CollectionItem) -> ParsedGallery:
        response = self._get(item.detail_url)
        soup = BeautifulSoup(response.text, "html.parser")
        title_node = soup.find("h1", id="book-name")
        title = title_node.get_text(" ", strip=True) if title_node else ""
        if not title:
            raise CollectionRequestError(f"详情页未解析到标题: {item.detail_url}")
        tags: list[str] = []
        languages: list[str] = []
        tag_container = soup.find("span", {"itemprop": "genre", "data-type": "tags"})
        if tag_container:
            for tag in tag_container.find_all("a"):
                value = tag.get_text(strip=True)
                if not value:
                    continue
                (languages if value in self.LANGUAGE_TAGS else tags).append(value)
        authors: list[str] = []
        author_container = soup.find("span", {"itemprop": "author", "data-type": "author"})
        if author_container:
            authors = [node.get_text(strip=True) for node in author_container.find_all("a") if node.get_text(strip=True)]
        date_node = soup.find("span", {"itemprop": "datePublished"})
        raw_date = date_node.get_text(strip=True) if date_node else ""
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", raw_date)
        text = soup.get_text("\n", strip=True)
        pages_match = re.search(r"页数\s*[:：]\s*(\d+)", text)
        thumbnail_url = ""
        meta = soup.find("meta", attrs={"property": "og:image"})
        if meta:
            thumbnail_url = str(meta.get("content") or "")
        return ParsedGallery(
            GalleryInfo(
                id=item.id,
                link=item.detail_url,
                title=title,
                tags=", ".join(dict.fromkeys(tags)),
                authors=", ".join(dict.fromkeys(authors)),
                languages=", ".join(dict.fromkeys(languages)),
                pages=pages_match.group(1) if pages_match else "",
                uploaded_date=date_match.group(0) if date_match else raw_date,
            ),
            urljoin(item.detail_url, thumbnail_url) if thumbnail_url else "",
        )

    def fetch_thumbnail(self, url: str) -> BinaryPayload:
        response = self._get(url)
        headers = getattr(response, "headers", {}) or {}
        return BinaryPayload(bytes(response.content), str(headers.get("content-type", "")))


def parse_local_links(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"未找到本地链接文件: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(text, "html.parser")
    candidates: list[tuple[str, str]] = []
    for anchor in soup.find_all("a", href=True):
        candidates.append((str(anchor["href"]).strip(), anchor.get_text(" ", strip=True)))
    for match in re.finditer(r"https?://[^\s\"'<>]+", text, flags=re.IGNORECASE):
        candidates.append((match.group(0).rstrip(".,;"), ""))
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for url, label in candidates:
        gallery_id = _nh_id(url)
        if gallery_id and gallery_id not in seen:
            seen.add(gallery_id)
            normalized = re.sub(r"(/g/\d+).*", r"\1/", url)
            result.append((normalized, label))
    return result


def _retryable(exc: BaseException) -> bool:
    return bool(getattr(exc, "retryable", True))


def _retry_delay(base: float, exponent: int, *, cap: float = 300.0) -> float:
    """Return a capped exponential delay without ever constructing 2**1024."""

    if base <= 0 or cap <= 0:
        return 0.0
    safe_exponent = min(max(0, int(exponent)), 30)
    return min(float(cap), float(base) * (2.0**safe_exponent))


class OnlineCollectionRunner:
    def __init__(
        self,
        config: ResolvedConfig,
        adapter: SiteAdapter,
        *,
        sleep_fn: Callable[[float], None] = time.sleep,
        stop_event: threading.Event | None = None,
    ):
        if config.output_csv is None or config.image_dir is None:
            raise ValueError(f"{config.mode} 需要 output_csv 和 image_dir")
        self.config = config
        self.adapter = adapter
        self.sleep_fn = sleep_fn
        self.stop_event = stop_event or threading.Event()
        self.csv_store = CsvStore(config.output_csv)
        self.thumbnail_store = ThumbnailStore(config.image_dir)
        self.checkpoint = Checkpoint(config.state_file)
        self.error_writer = JsonlWriter(config.error_log, flush_every=1)
        replayed = self.checkpoint.replay(config.identity) if config.resume else ReplayedState()
        has_resume_state = bool(
            replayed.tasks or replayed.completed_pages or replayed.terminal_pages
        )
        if replayed.run_completed or not config.resume or not has_resume_state:
            replayed = ReplayedState()
            self.checkpoint.append({"event": "run_start", "mode": config.mode, "identity": config.identity})
        else:
            self.checkpoint.append({"event": "run_resume", "mode": config.mode, "identity": config.identity})
        self.states = replayed.tasks
        self.completed_pages = replayed.completed_pages
        self.terminal_pages = replayed.terminal_pages
        for state in self.states.values():
            state.info_ok = self.csv_store.has(state.item.id)
            state.thumb_ok = self.thumbnail_store.has(state.item.id)

    def _wait(self, seconds: float) -> None:
        if self.stop_event.is_set():
            raise StopRequested()
        if seconds <= 0:
            return
        if self.sleep_fn is time.sleep:
            if self.stop_event.wait(seconds):
                raise StopRequested()
        else:
            self.sleep_fn(seconds)
            if self.stop_event.is_set():
                raise StopRequested()

    def _attempt(self, operation: Callable[[], Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.config.request_attempts + 1):
            if self.stop_event.is_set():
                raise StopRequested()
            try:
                return operation()
            except StopRequested:
                raise
            except Exception as exc:
                last_error = exc
                if not _retryable(exc) or attempt >= self.config.request_attempts:
                    break
                self._wait(_retry_delay(self.config.retry_backoff, attempt - 1, cap=60.0))
        assert last_error is not None
        raise last_error

    def _error(
        self,
        *,
        round_number: int,
        stage: str,
        exc: BaseException,
        page: int = 0,
        item: CollectionItem | None = None,
        url: str = "",
    ) -> None:
        self.error_writer.append(
            {
                "event": "collection_error",
                "mode": self.config.mode,
                "round": round_number,
                "stage": stage,
                "page": page or (item.page if item else 0),
                "id": item.id if item else "",
                "url": url,
                "errorType": type(exc).__name__,
                "statusCode": getattr(exc, "status_code", None),
                "message": str(exc),
                "retryable": _retryable(exc),
            }
        )

    def _merge_item(self, item: CollectionItem) -> TaskState:
        state = self.states.get(item.id)
        if state is None:
            state = TaskState(
                item=item,
                info_ok=self.csv_store.has(item.id),
                thumb_ok=self.thumbnail_store.has(item.id),
            )
            self.states[item.id] = state
        else:
            state.item.detail_url = item.detail_url or state.item.detail_url
            state.item.thumbnail_url = item.thumbnail_url or state.item.thumbnail_url
            state.item.page = item.page or state.item.page
            state.info_ok = self.csv_store.has(item.id)
            state.thumb_ok = self.thumbnail_store.has(item.id)
        self.checkpoint.task(state)
        return state

    def _discover_page(self, page: int, round_number: int) -> bool:
        try:
            def discover_nonempty() -> list[CollectionItem]:
                found = self.adapter.discover_page(page)
                if not found:
                    raise CollectionRequestError(
                        "列表页返回 200，但未解析到任何项目；可能是反爬页面或站点结构已变化"
                    )
                return found

            items = self._attempt(discover_nonempty)
        except StopRequested:
            raise
        except Exception as exc:
            self._error(
                round_number=round_number,
                stage="list",
                exc=exc,
                page=page,
                url=build_page_url(self.config.start_url, page),
            )
            if not _retryable(exc):
                if getattr(exc, "status_code", None) in {404, 410}:
                    if page == 1:
                        self.terminal_pages.add(page)
                        self.checkpoint.append({"event": "page_terminal", "page": page})
                    else:
                        self.completed_pages.add(page)
                        self.checkpoint.append({"event": "page_complete", "page": page, "empty": True})
                else:
                    self.terminal_pages.add(page)
                    self.checkpoint.append({"event": "page_terminal", "page": page})
                return True
            return False
        for item in items:
            self._merge_item(item)
        self.completed_pages.add(page)
        self.checkpoint.append({"event": "page_complete", "page": page, "items": len(items)})
        print(f"[采集] 第 {page} 页发现 {len(items)} 项", flush=True)
        return True

    def _process_item(self, state: TaskState, round_number: int) -> TaskState:
        item = state.item
        state.info_ok = self.csv_store.has(item.id)
        state.thumb_ok = self.thumbnail_store.has(item.id)
        if state.complete or state.terminal:
            return state

        needs_detail = not state.info_ok or (not state.thumb_ok and not item.thumbnail_url)
        if needs_detail:
            try:
                parsed = self._attempt(lambda: self.adapter.fetch_detail(item))
                if not state.info_ok:
                    self.csv_store.upsert(parsed.info)
                    state.info_ok = True
                    state.terminal_info = False
                if parsed.thumbnail_url:
                    item.thumbnail_url = parsed.thumbnail_url
            except StopRequested:
                raise
            except Exception as exc:
                purpose = "detail" if not state.info_ok else "thumbnail_url"
                self._error(
                    round_number=round_number,
                    stage=purpose,
                    exc=exc,
                    item=item,
                    url=item.detail_url,
                )
                if not _retryable(exc):
                    if not state.info_ok:
                        state.terminal_info = True
                    elif not state.thumb_ok:
                        state.terminal_thumb = True

        if not state.thumb_ok:
            if not item.thumbnail_url:
                exc = CollectionRequestError("未解析到缩略图 URL")
                self._error(
                    round_number=round_number,
                    stage="thumbnail",
                    exc=exc,
                    item=item,
                    url=item.detail_url,
                )
            else:
                try:
                    payload = self._attempt(lambda: self.adapter.fetch_thumbnail(item.thumbnail_url))
                    self.thumbnail_store.save(item.id, item.thumbnail_url, payload)
                    state.thumb_ok = True
                    state.terminal_thumb = False
                except StopRequested:
                    raise
                except Exception as exc:
                    self._error(
                        round_number=round_number,
                        stage="thumbnail",
                        exc=exc,
                        item=item,
                        url=item.thumbnail_url,
                    )
                    # A list-page CDN URL may expire or rotate.  Force the
                    # next round through fetch_detail() to obtain a fresh URL
                    # before treating the gallery itself as terminal.
                    item.thumbnail_url = ""
                    state.terminal_thumb = False
        self.checkpoint.task(state)
        status = "完成" if state.complete else "待重试"
        print(
            f"[采集] {item.id} {status}（信息={'OK' if state.info_ok else 'FAIL'}，缩略图={'OK' if state.thumb_ok else 'FAIL'}）",
            flush=True,
        )
        if self.config.interval:
            self._wait(self.config.interval)
        return state

    def _initial_local_items(self) -> None:
        if self.config.input_file is None:
            raise ValueError("nh-local-info 需要 input_file")
        links = parse_local_links(self.config.input_file)
        if not links and not self.states:
            self.checkpoint.close()
            self.error_writer.close()
            raise ValueError(f"输入文件未解析到任何 NH 图库链接: {self.config.input_file}")
        for url, label in links:
            self._merge_item(CollectionItem(_nh_id(url), url, page=0, label=label))

    def _summary(self, rounds: int, pending_pages: set[int], interrupted: bool = False) -> CollectionSummary:
        completed = sum(state.complete for state in self.states.values())
        terminal = sum(state.terminal and not state.complete for state in self.states.values()) + len(self.terminal_pages)
        pending = sum(not state.complete and not state.terminal for state in self.states.values())
        return CollectionSummary(
            mode=self.config.mode,
            rounds=rounds,
            discovered=len(self.states),
            completed=completed,
            pending=pending,
            terminal=terminal,
            failed_pages=len(pending_pages),
            interrupted=interrupted,
            output_csv=str(self.config.output_csv),
        )

    def run(self) -> CollectionSummary:
        is_online = self.config.mode in {"nh-online", "jm-online"}
        if not is_online:
            self._initial_local_items()
        pending_pages = (
            set(range(1, self.config.max_pages + 1)) - self.completed_pages - self.terminal_pages
            if is_online
            else set()
        )
        pending_ids = {
            gallery_id
            for gallery_id, state in self.states.items()
            if not state.complete and not state.terminal
        }
        rounds = 0
        try:
            while pending_pages or pending_ids:
                if self.config.max_rounds and rounds >= self.config.max_rounds:
                    break
                rounds += 1
                print(
                    f"[采集] 开始第 {rounds} 轮：失败页 {len(pending_pages)}，待完成项目 {len(pending_ids)}",
                    flush=True,
                )
                next_pages: set[int] = set()
                for page in sorted(pending_pages):
                    if not self._discover_page(page, rounds):
                        next_pages.add(page)

                candidates = {
                    gallery_id
                    for gallery_id, state in self.states.items()
                    if not state.complete and not state.terminal
                }
                if candidates:
                    executor = ThreadPoolExecutor(max_workers=self.config.workers)
                    futures = {
                        executor.submit(self._process_item, self.states[gallery_id], rounds): gallery_id
                        for gallery_id in sorted(candidates)
                    }
                    try:
                        for future in as_completed(futures):
                            future.result()
                    except BaseException:
                        self.stop_event.set()
                        for future in futures:
                            future.cancel()
                        executor.shutdown(wait=True, cancel_futures=True)
                        raise
                    else:
                        executor.shutdown(wait=True)

                self.csv_store.commit()

                pending_pages = next_pages
                pending_ids = {
                    gallery_id
                    for gallery_id, state in self.states.items()
                    if not state.complete and not state.terminal
                }
                if not pending_pages and not pending_ids:
                    break
                if self.config.max_rounds and rounds >= self.config.max_rounds:
                    break
                delay = _retry_delay(self.config.retry_backoff, rounds - 1)
                print(f"[采集] 本轮失败项将在 {delay:g} 秒后重试", flush=True)
                self._wait(delay)
        except (KeyboardInterrupt, StopRequested):
            self.stop_event.set()
            self.csv_store.commit()
            self.checkpoint.append({"event": "run_interrupted", "round": rounds})
            self.checkpoint.close()
            self.error_writer.close()
            return self._summary(rounds, pending_pages, interrupted=True)

        self.csv_store.commit()
        summary = self._summary(rounds, pending_pages)
        self.checkpoint.compact(
            mode=self.config.mode,
            identity=self.config.identity,
            tasks=self.states.values(),
            completed_pages=self.completed_pages,
            terminal_pages=self.terminal_pages,
            run_completed=summary.success,
            summary=summary.as_dict(),
        )
        self.error_writer.close()
        return summary


def _safe_folder_name(label: str, gallery_id: str) -> str:
    value = re.sub(r'[\\/*?:"<>|\x00-\x1f]', "_", label).strip(" .")
    if not value or value in {".", ".."}:
        return gallery_id
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if value.upper() in reserved:
        value = f"_{value}"
    # Link text is not unique.  Always include the stable gallery ID so two
    # equally titled galleries cannot share a folder or each other's page 1.
    return f"{gallery_id}_{value}"[:180]


class NHFullImageAdapter(NHAdapter):
    def fetch_full_image_url(self, page_url: str) -> str | None:
        try:
            response = self._get(page_url)
        except CollectionRequestError as exc:
            if exc.status_code in {404, 410}:
                return None
            raise
        soup = BeautifulSoup(response.text, "html.parser")
        image = soup.select_one("section#image-container img")
        if not image or not image.get("src"):
            raise CollectionRequestError(f"页面未解析到内页图片: {page_url}")
        return urljoin(page_url, str(image["src"]))


class FullImageStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._validity: dict[tuple[str, int], bool] = {}

    def has(self, folder: str, page: int) -> bool:
        key = (folder, page)
        with self._lock:
            cached = self._validity.get(key)
            if cached is not None:
                return cached
            target_dir = self.root / folder
            valid = False
            if target_dir.exists():
                for path in target_dir.glob(f"{page}.*"):
                    if path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES and _validate_image_file(path):
                        valid = True
                        break
            self._validity[key] = valid
            return valid

    def save(self, folder: str, page: int, url: str, payload: BinaryPayload) -> Path:
        _validate_image_payload(payload)
        target_dir = self.root / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        extension = ThumbnailStore._extension(url, payload.content_type)
        target = target_dir / f"{page}{extension}"
        temp = target_dir / f".{page}.{uuid.uuid4().hex}.part"
        try:
            with temp.open("wb") as handle:
                handle.write(payload.content)
            os.replace(temp, target)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
        with self._lock:
            self._validity[(folder, page)] = True
        return target


class LocalImagesRunner:
    """Resumable full-page downloader used by the legacy local-images mode."""

    def __init__(
        self,
        config: ResolvedConfig,
        adapter: Any,
        *,
        sleep_fn: Callable[[float], None] = time.sleep,
        stop_event: threading.Event | None = None,
    ):
        if config.input_file is None or config.output_dir is None:
            raise ValueError("nh-local-images 需要 input_file 和 output_dir")
        self.config = config
        self.adapter = adapter
        self.sleep_fn = sleep_fn
        self.stop_event = stop_event or threading.Event()
        self.store = FullImageStore(config.output_dir)
        self.checkpoint = Checkpoint(config.state_file)
        self.errors = JsonlWriter(config.error_log, flush_every=1)
        replayed = self.checkpoint.replay(config.identity) if config.resume else ReplayedState()
        has_resume_state = bool(
            replayed.tasks
            or replayed.local_discovery_complete
            or replayed.limit_galleries
            or replayed.completed_galleries
        )
        if replayed.run_completed or not config.resume or not has_resume_state:
            replayed = ReplayedState()
            self.checkpoint.append({"event": "run_start", "mode": config.mode, "identity": config.identity})
        else:
            self.checkpoint.append({"event": "run_resume", "mode": config.mode, "identity": config.identity})
        self.states = replayed.tasks
        self.discovery_complete = replayed.local_discovery_complete
        self.limit_galleries = replayed.limit_galleries
        self.completed_galleries = replayed.completed_galleries
        # Older checkpoints marked discovery complete even when a retryable
        # page failure had prevented continuous enumeration.  Re-open those
        # galleries at their first failed page.
        if any(not state.info_ok and not state.terminal_info for state in self.states.values()):
            self.discovery_complete = False

    def _wait(self, seconds: float) -> None:
        if self.stop_event.is_set():
            raise StopRequested()
        if seconds <= 0:
            return
        if self.sleep_fn is time.sleep:
            if self.stop_event.wait(seconds):
                raise StopRequested()
        else:
            self.sleep_fn(seconds)

    def _attempt(self, operation: Callable[[], Any]) -> Any:
        last: Exception | None = None
        for attempt in range(self.config.request_attempts):
            if self.stop_event.is_set():
                raise StopRequested()
            try:
                return operation()
            except StopRequested:
                raise
            except Exception as exc:
                last = exc
                if not _retryable(exc) or attempt + 1 >= self.config.request_attempts:
                    break
                self._wait(_retry_delay(self.config.retry_backoff, attempt, cap=60.0))
        assert last is not None
        raise last

    def _record_error(self, state: TaskState, stage: str, exc: BaseException, round_number: int) -> None:
        self.errors.append(
            {
                "event": "collection_error",
                "mode": self.config.mode,
                "round": round_number,
                "stage": stage,
                "id": state.item.id,
                "page": state.item.page,
                "url": state.item.detail_url if stage == "page" else state.item.thumbnail_url,
                "errorType": type(exc).__name__,
                "statusCode": getattr(exc, "status_code", None),
                "message": str(exc),
                "retryable": _retryable(exc),
            }
        )

    def _folder_and_gallery(self, state: TaskState) -> tuple[str, str]:
        raw_id = state.item.id.split(":", 1)[0]
        return _safe_folder_name(state.item.label, raw_id), state.item.detail_url.rsplit(str(state.item.page) + "/", 1)[0]

    def _process(self, state: TaskState, round_number: int) -> None:
        folder, _gallery = self._folder_and_gallery(state)
        if self.store.has(folder, state.item.page):
            state.info_ok = state.thumb_ok = True
            self.checkpoint.task(state)
            return
        if not state.info_ok:
            return  # Page discovery owns the continuous gallery cursor.
        if state.info_ok and not state.thumb_ok:
            try:
                payload = self._attempt(lambda: self.adapter.fetch_thumbnail(state.item.thumbnail_url))
                self.store.save(folder, state.item.page, state.item.thumbnail_url, payload)
                state.thumb_ok = True
            except StopRequested:
                raise
            except Exception as exc:
                self._record_error(state, "image", exc, round_number)
                # CDN URLs can expire or rotate.  Re-fetch the page URL in the
                # next round before declaring the image terminal.
                state.info_ok = False
                state.item.thumbnail_url = ""
                self.discovery_complete = False
        self.checkpoint.task(state)

    def _clear_limit(self, gallery_id: str) -> None:
        if gallery_id in self.limit_galleries:
            self.limit_galleries.discard(gallery_id)
            self.checkpoint.append({"event": "gallery_limit_clear", "id": gallery_id})

    def _complete_gallery(self, gallery_id: str) -> None:
        self._clear_limit(gallery_id)
        self.completed_galleries.add(gallery_id)
        self.checkpoint.append({"event": "gallery_complete", "id": gallery_id})

    def _reopen_gallery(self, gallery_id: str) -> None:
        if gallery_id in self.completed_galleries:
            self.completed_galleries.discard(gallery_id)
            self.checkpoint.append({"event": "gallery_complete_clear", "id": gallery_id})

    def _delete_gallery_tasks_from(self, gallery_id: str, page: int) -> None:
        prefix = f"{gallery_id}:"
        for task_id, state in list(self.states.items()):
            if task_id.startswith(prefix) and state.item.page >= page:
                self.states.pop(task_id, None)
                self.checkpoint.delete_task(task_id)

    def _discover_gallery(self, gallery_url: str, label: str, round_number: int) -> None:
        gallery_id = _nh_id(gallery_url)
        gallery_states = [
            state for task_id, state in self.states.items() if task_id.startswith(f"{gallery_id}:")
        ]
        retry_pages = sorted(
            state.item.page
            for state in gallery_states
            if not state.info_ok and not state.terminal_info
        )
        if gallery_id in self.completed_galleries and not retry_pages:
            return
        if gallery_id in self.limit_galleries and not retry_pages:
            return
        if retry_pages:
            cursor = retry_pages[0]
            self._reopen_gallery(gallery_id)
            self._delete_gallery_tasks_from(gallery_id, cursor + 1)
            self._clear_limit(gallery_id)
        else:
            cursor = max((state.item.page for state in gallery_states), default=0) + 1

        folder = _safe_folder_name(label, gallery_id)
        for page in range(cursor, self.config.max_pages + 1):
            page_url = f"{gallery_url.rstrip('/')}/{page}/"
            task_id = f"{gallery_id}:{page}"
            state = self.states.get(task_id) or TaskState(
                CollectionItem(task_id, page_url, page=page, label=label)
            )
            self.states[task_id] = state
            if self.store.has(folder, page):
                state.info_ok = state.thumb_ok = True
                self.checkpoint.task(state)
                continue
            if not state.info_ok:
                try:
                    image_url = self._attempt(lambda url=page_url: self.adapter.fetch_full_image_url(url))
                except StopRequested:
                    raise
                except Exception as exc:
                    self._record_error(state, "page", exc, round_number)
                    if not _retryable(exc):
                        state.terminal_info = True
                        state.terminal_thumb = True
                        self._complete_gallery(gallery_id)
                    self.checkpoint.task(state)
                    return  # Never enumerate beyond an unresolved page.
                if image_url is None:
                    if page == 1:
                        exc = CollectionRequestError("图库首页不存在", retryable=False, status_code=404)
                        state.terminal_info = True
                        state.terminal_thumb = True
                        self._record_error(state, "page", exc, round_number)
                        self.checkpoint.task(state)
                    else:
                        self._delete_gallery_tasks_from(gallery_id, page)
                    self._complete_gallery(gallery_id)
                    return
                state.item.thumbnail_url = image_url
                state.info_ok = True
                state.terminal_info = False
                self.checkpoint.task(state)
            if self.config.interval:
                self._wait(self.config.interval)

        if gallery_id not in self.limit_galleries:
            self.limit_galleries.add(gallery_id)
            self.errors.append(
                {
                    "event": "collection_limit",
                    "mode": self.config.mode,
                    "stage": "max_pages",
                    "id": gallery_id,
                    "url": gallery_url,
                    "maxPages": self.config.max_pages,
                    "message": "达到单本最大页数保护限制，尚未确认图库末页",
                    "retryable": False,
                }
            )
            self.checkpoint.append({"event": "gallery_limit", "id": gallery_id})

    def _discover_round(self, links: list[tuple[str, str]], round_number: int) -> None:
        for gallery_url, label in links:
            self._discover_gallery(gallery_url, label, round_number)
        gallery_ids = {_nh_id(url) for url, _label in links}
        self.discovery_complete = all(
            gallery_id in self.completed_galleries or gallery_id in self.limit_galleries
            for gallery_id in gallery_ids
        )
        if self.discovery_complete:
            self.checkpoint.append({"event": "local_discovery_complete"})

    def _summary(self, rounds: int, interrupted: bool = False) -> CollectionSummary:
        return CollectionSummary(
            mode=self.config.mode,
            rounds=rounds,
            discovered=len(self.states),
            completed=sum(state.complete for state in self.states.values()),
            pending=sum(not state.complete and not state.terminal for state in self.states.values()),
            terminal=sum(state.terminal and not state.complete for state in self.states.values()),
            limit_reached=len(self.limit_galleries),
            interrupted=interrupted,
            output_csv=str(self.config.output_dir),
        )

    def run(self) -> CollectionSummary:
        rounds = 0
        links = parse_local_links(self.config.input_file)
        if not links and not self.states:
            self.checkpoint.close()
            self.errors.close()
            raise ValueError(f"输入文件未解析到任何 NH 图库链接: {self.config.input_file}")
        try:
            while True:
                if self.config.max_rounds and rounds >= self.config.max_rounds:
                    break
                rounds += 1
                if not self.discovery_complete:
                    self._discover_round(links, rounds)
                image_pending = [
                    state
                    for state in self.states.values()
                    if state.info_ok and not state.thumb_ok and not state.terminal
                ]
                if image_pending:
                    with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
                        futures = [executor.submit(self._process, state, rounds) for state in image_pending]
                        for future in as_completed(futures):
                            future.result()
                retry_pending = any(
                    not state.complete and not state.terminal for state in self.states.values()
                )
                if self.discovery_complete and not retry_pending:
                    break
                if self.config.max_rounds and rounds >= self.config.max_rounds:
                    break
                self._wait(_retry_delay(self.config.retry_backoff, rounds - 1))
        except (KeyboardInterrupt, StopRequested):
            self.stop_event.set()
            self.checkpoint.append({"event": "run_interrupted", "round": rounds})
            self.checkpoint.close()
            self.errors.close()
            return self._summary(rounds, interrupted=True)
        summary = self._summary(rounds)
        self.checkpoint.compact(
            mode=self.config.mode,
            identity=self.config.identity,
            tasks=self.states.values(),
            limit_galleries=self.limit_galleries,
            completed_galleries=self.completed_galleries,
            local_discovery_complete=self.discovery_complete,
            run_completed=summary.success,
            summary=summary.as_dict(),
        )
        self.errors.close()
        return summary


def run_collection(
    config: CollectionConfig,
    *,
    adapter: SiteAdapter | Any | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    stop_event: threading.Event | None = None,
) -> CollectionSummary:
    """Run one collection job; no network is used when a fake adapter is injected."""

    resolved = config.resolved()
    if adapter is None:
        if resolved.mode == "jm-online":
            adapter = JMAdapter(resolved)
        elif resolved.mode == "nh-local-images":
            adapter = NHFullImageAdapter(resolved)
        else:
            adapter = NHAdapter(resolved)
    if resolved.mode == "nh-local-images":
        return LocalImagesRunner(
            resolved, adapter, sleep_fn=sleep_fn, stop_event=stop_event
        ).run()
    return OnlineCollectionRunner(
        resolved, adapter, sleep_fn=sleep_fn, stop_event=stop_event
    ).run()


def _add_retry_options(parser: argparse.ArgumentParser, *, default_workers: int) -> None:
    parser.add_argument("--workers", "--max-workers", type=int, default=default_workers, help="并发线程数")
    parser.add_argument("--request-attempts", type=int, default=3, help="每轮内单次请求尝试次数")
    parser.add_argument(
        "--max-rounds",
        "--retry-rounds",
        type=int,
        default=0,
        help="包含首轮的最大轮数；0 表示持续到全部成功",
    )
    parser.add_argument("--retry-backoff", type=float, default=2.0, help="轮次/请求指数退避基数秒数")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP 请求超时秒数")
    parser.add_argument("--interval", type=float, default=0.0, help="成功项目之间的额外间隔秒数")
    parser.add_argument("--state-file", help="结构化断点 JSONL；默认按任务参数生成")
    parser.add_argument("--error-log", help="结构化失败 JSONL；默认按任务参数生成")
    parser.add_argument("--proxy", help="显式 HTTP(S) 代理；空值使用 ONLINE_COVER_PROXY")
    parser.add_argument("--no-resume", action="store_true", help="忽略未完成断点并开始新一轮扫描")
    parser.add_argument("--once", action="store_true", help=argparse.SUPPRESS)


def _add_online_parser(subparsers, mode: str) -> None:
    defaults = MODE_DEFAULTS[mode]
    parser = subparsers.add_parser(mode, help=f"运行 {mode} 元数据和缩略图采集")
    parser.add_argument("legacy_max_pages", nargs="?", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--base-url", default=defaults["base_url"])
    parser.add_argument("--start-url", default=defaults["start_url"])
    parser.add_argument("-n", "--max-pages", "--max-page", type=int, default=None)
    parser.add_argument("--output-csv", default=str(defaults["output_csv"]))
    parser.add_argument("--image-dir", "--output-dir", default=str(DEFAULT_IMAGE_DIR))
    _add_retry_options(parser, default_workers=int(defaults["workers"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XP-Gacha 统一、可恢复的数据采集器")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    _add_online_parser(subparsers, "nh-online")
    _add_online_parser(subparsers, "jm-online")

    local_info = subparsers.add_parser("nh-local-info", help="从本地书签/URL 列表采集 NH 信息和缩略图")
    local_info.add_argument("--base-url", default=MODE_DEFAULTS["nh-local-info"]["base_url"])
    local_info.add_argument("--input-file", default=str(MODE_DEFAULTS["nh-local-info"]["input_file"]))
    local_info.add_argument("--output-csv", default=str(MODE_DEFAULTS["nh-local-info"]["output_csv"]))
    local_info.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    _add_retry_options(local_info, default_workers=int(MODE_DEFAULTS["nh-local-info"]["workers"]))

    local_images = subparsers.add_parser("nh-local-images", help="从本地书签/URL 列表下载 NH 完整内页")
    local_images.add_argument("--base-url", default=MODE_DEFAULTS["nh-local-images"]["base_url"])
    local_images.add_argument("--input-file", default=str(MODE_DEFAULTS["nh-local-images"]["input_file"]))
    local_images.add_argument("--output-dir", default=str(MODE_DEFAULTS["nh-local-images"]["output_dir"]))
    local_images.add_argument("--max-pages", "--max-page", type=int, default=int(MODE_DEFAULTS["nh-local-images"]["max_pages"]))
    _add_retry_options(local_images, default_workers=int(MODE_DEFAULTS["nh-local-images"]["workers"]))
    return parser


def config_from_args(args: argparse.Namespace) -> CollectionConfig:
    max_pages = getattr(args, "max_pages", None)
    if max_pages is None:
        max_pages = getattr(args, "legacy_max_pages", None) or 0
    max_rounds = 1 if getattr(args, "once", False) else args.max_rounds
    return CollectionConfig(
        mode=args.mode,
        base_url=getattr(args, "base_url", ""),
        start_url=getattr(args, "start_url", ""),
        max_pages=max_pages,
        output_csv=getattr(args, "output_csv", None),
        image_dir=getattr(args, "image_dir", None),
        input_file=getattr(args, "input_file", None),
        output_dir=getattr(args, "output_dir", None),
        workers=args.workers,
        request_attempts=args.request_attempts,
        max_rounds=max_rounds,
        retry_backoff=args.retry_backoff,
        timeout=args.timeout,
        interval=args.interval,
        state_file=args.state_file,
        error_log=args.error_log,
        proxy=args.proxy,
        resume=not args.no_resume,
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    summary = run_collection(config_from_args(args))
    print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2), flush=True)
    return summary.exit_code


def legacy_main(mode: str, argv: Iterable[str] | None = None) -> int:
    return main([mode, *(list(argv) if argv is not None else sys.argv[1:])])


if __name__ == "__main__":
    raise SystemExit(main())
