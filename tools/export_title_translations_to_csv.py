from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import unicodedata
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CSV_DIR = PROJECT_ROOT / "data" / "gallery_info"
DEFAULT_PATTERN = "*_full.csv"
ID_COLUMN = "ID"
TITLE_COLUMN = "标题"
TITLE_TRANSLATION_COLUMN = "标题译文"
TABLE_NAME = "gallery_info"

TranslationPair: TypeAlias = tuple[object, object]
TranslationSource: TypeAlias = Mapping[object, object] | Iterable[TranslationPair]
TranslationLoader: TypeAlias = Callable[[], TranslationSource]


class ExportError(RuntimeError):
    """Raised when translations cannot be exported without risking CSV data."""


@dataclass(frozen=True)
class FileExportStats:
    path: Path
    rows: int
    matched_rows: int
    filled_rows: int
    updated_rows: int
    unchanged_rows: int
    unmatched_rows: int
    column_added: bool
    column_reordered: bool

    @property
    def needs_write(self) -> bool:
        return self.column_added or self.column_reordered or self.filled_rows > 0 or self.updated_rows > 0


@dataclass
class ExportSummary:
    csv_dir: Path
    pattern: str
    dry_run: bool
    translation_count: int
    files_scanned: int = 0
    files_changed: int = 0
    files_written: int = 0
    rows: int = 0
    matched_rows: int = 0
    filled_rows: int = 0
    updated_rows: int = 0
    unchanged_rows: int = 0
    unmatched_rows: int = 0
    files: list[FileExportStats] = field(default_factory=list)

    def add_file(self, stats: FileExportStats) -> None:
        self.files.append(stats)
        self.files_scanned += 1
        self.files_changed += int(stats.needs_write)
        self.rows += stats.rows
        self.matched_rows += stats.matched_rows
        self.filled_rows += stats.filled_rows
        self.updated_rows += stats.updated_rows
        self.unchanged_rows += stats.unchanged_rows
        self.unmatched_rows += stats.unmatched_rows


@dataclass
class _PreparedFile:
    stats: FileExportStats
    temporary_path: Path | None


def normalize_id(value: object) -> str:
    """Normalize an ID for exact matching without changing the CSV value itself."""

    if value is None:
        return ""
    return unicodedata.normalize("NFKC", str(value)).strip().upper()


def normalize_translations(source: TranslationSource) -> dict[str, str]:
    """Return non-empty translations keyed by normalized exact gallery ID."""

    items = source.items() if isinstance(source, Mapping) else source
    normalized: dict[str, str] = {}
    for pair in items:
        try:
            raw_id, raw_translation = pair
        except (TypeError, ValueError) as exc:
            raise ExportError("标题译文数据必须由 (ID, 标题译文) 二元组组成") from exc

        item_id = normalize_id(raw_id)
        translation = "" if raw_translation is None else str(raw_translation)
        if not item_id or not translation.strip():
            continue

        existing = normalized.get(item_id)
        if existing is not None and existing != translation:
            raise ExportError(f"规范化 ID {item_id!r} 对应多个不同的非空标题译文")
        normalized[item_id] = translation
    return normalized


def load_translations_from_database(engine=None) -> list[TranslationPair]:
    """Read only ID and title translation from gallery_info.

    ``engine`` is injectable for tests. Importing the runtime engine lazily keeps
    pure CSV unit tests from opening the configured database.
    """

    from sqlalchemy import text

    if engine is None:
        from server.database import get_engine

        engine = get_engine()

    statement = text(f"SELECT `{ID_COLUMN}`, `{TITLE_TRANSLATION_COLUMN}` FROM `{TABLE_NAME}`")
    with engine.connect() as connection:
        rows = connection.execute(statement).all()
    return [(row[0], row[1]) for row in rows]


def _resolve_csv_dir(csv_dir: str | Path) -> Path:
    path = Path(csv_dir).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _validate_pattern(pattern: str) -> str:
    value = str(pattern or "").strip()
    pattern_path = Path(value)
    if not value:
        raise ExportError("CSV 文件匹配模式不能为空")
    if pattern_path.is_absolute() or ".." in pattern_path.parts or "/" in value or "\\" in value:
        raise ExportError("CSV 文件匹配模式只能是指定 CSV 目录内的文件名模式")
    return value


def _output_fieldnames(fieldnames: list[str], path: Path) -> tuple[list[str], bool, bool]:
    if len(fieldnames) != len(set(fieldnames)):
        raise ExportError(f"CSV 表头包含重复列：{path}")
    if ID_COLUMN not in fieldnames:
        raise ExportError(f"CSV 缺少 {ID_COLUMN} 列：{path}")
    if TITLE_COLUMN not in fieldnames:
        raise ExportError(f"CSV 缺少 {TITLE_COLUMN} 列：{path}")

    column_added = TITLE_TRANSLATION_COLUMN not in fieldnames
    without_translation = [name for name in fieldnames if name != TITLE_TRANSLATION_COLUMN]
    title_index = without_translation.index(TITLE_COLUMN)
    output = without_translation.copy()
    output.insert(title_index + 1, TITLE_TRANSLATION_COLUMN)
    column_reordered = not column_added and output != fieldnames
    return output, column_added, column_reordered


def _new_temporary_path(path: Path, suffix: str = ".tmp") -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=suffix,
    )
    os.close(descriptor)
    return Path(name)


def _unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _prepare_file(path: Path, translations: Mapping[str, str], dry_run: bool) -> _PreparedFile:
    temporary_path: Path | None = None
    output_stream = None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise ExportError(f"CSV 没有表头：{path}")
            fieldnames = list(reader.fieldnames)
            output_fieldnames, column_added, column_reordered = _output_fieldnames(fieldnames, path)

            writer = None
            if not dry_run:
                temporary_path = _new_temporary_path(path)
                output_stream = temporary_path.open("w", encoding="utf-8-sig", newline="")
                writer = csv.DictWriter(
                    output_stream,
                    fieldnames=output_fieldnames,
                    extrasaction="raise",
                    lineterminator="\n",
                )
                writer.writeheader()

            rows = 0
            matched_rows = 0
            filled_rows = 0
            updated_rows = 0
            unchanged_rows = 0
            unmatched_rows = 0

            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ExportError(f"CSV 第 {row_number} 行字段数超过表头：{path}")
                rows += 1
                current_translation = row.get(TITLE_TRANSLATION_COLUMN) or ""
                database_translation = translations.get(normalize_id(row.get(ID_COLUMN)))

                if database_translation is None:
                    unmatched_rows += 1
                else:
                    matched_rows += 1
                    if current_translation == database_translation:
                        unchanged_rows += 1
                    elif current_translation.strip():
                        updated_rows += 1
                    else:
                        filled_rows += 1
                    row[TITLE_TRANSLATION_COLUMN] = database_translation

                if writer is not None:
                    writer.writerow({name: row.get(name) or "" for name in output_fieldnames})

            if output_stream is not None:
                output_stream.flush()
                os.fsync(output_stream.fileno())
                output_stream.close()
                output_stream = None

        stats = FileExportStats(
            path=path,
            rows=rows,
            matched_rows=matched_rows,
            filled_rows=filled_rows,
            updated_rows=updated_rows,
            unchanged_rows=unchanged_rows,
            unmatched_rows=unmatched_rows,
            column_added=column_added,
            column_reordered=column_reordered,
        )
        if temporary_path is not None and not stats.needs_write:
            _unlink(temporary_path)
            temporary_path = None
        elif temporary_path is not None:
            try:
                shutil.copymode(path, temporary_path)
            except OSError:
                pass
        return _PreparedFile(stats=stats, temporary_path=temporary_path)
    except Exception:
        if output_stream is not None:
            output_stream.close()
        _unlink(temporary_path)
        raise


def _create_rollback_path(target: Path) -> Path:
    rollback_path = target.with_name(f".{target.name}.{uuid.uuid4().hex}.rollback.tmp")
    try:
        os.link(target, rollback_path)
    except OSError:
        try:
            shutil.copy2(target, rollback_path)
        except Exception:
            _unlink(rollback_path)
            raise
    return rollback_path


def _commit_prepared_files(prepared_files: list[_PreparedFile]) -> int:
    changed = [item for item in prepared_files if item.temporary_path is not None]
    if not changed:
        return 0

    rollback_paths: dict[Path, Path] = {}
    committed: list[Path] = []
    retained_rollbacks: set[Path] = set()
    try:
        for item in changed:
            rollback_paths[item.stats.path] = _create_rollback_path(item.stats.path)

        for item in changed:
            assert item.temporary_path is not None
            os.replace(item.temporary_path, item.stats.path)
            item.temporary_path = None
            committed.append(item.stats.path)
    except Exception as exc:
        rollback_errors: list[str] = []
        for target in reversed(committed):
            rollback_path = rollback_paths.get(target)
            if rollback_path is None:
                continue
            try:
                os.replace(rollback_path, target)
                rollback_paths.pop(target, None)
            except OSError as rollback_exc:
                retained_rollbacks.add(rollback_path)
                rollback_errors.append(
                    f"{target}: {rollback_exc}（原文件备份保留在 {rollback_path}）"
                )
        detail = f"；回滚失败：{'；'.join(rollback_errors)}" if rollback_errors else ""
        raise ExportError(f"原子替换 CSV 失败：{exc}{detail}") from exc
    finally:
        for item in changed:
            _unlink(item.temporary_path)
            item.temporary_path = None
        for rollback_path in rollback_paths.values():
            if rollback_path not in retained_rollbacks:
                _unlink(rollback_path)

    return len(changed)


def export_title_translations(
    csv_dir: str | Path = DEFAULT_CSV_DIR,
    pattern: str = DEFAULT_PATTERN,
    *,
    dry_run: bool = False,
    translations: TranslationSource | None = None,
    translation_loader: TranslationLoader | None = None,
) -> ExportSummary:
    """Fill matching CSV rows from database translations and atomically save files."""

    if translations is not None and translation_loader is not None:
        raise ExportError("translations 与 translation_loader 不能同时提供")

    resolved_dir = _resolve_csv_dir(csv_dir)
    validated_pattern = _validate_pattern(pattern)
    if not resolved_dir.is_dir():
        raise ExportError(f"CSV 目录不存在：{resolved_dir}")

    files = sorted(path for path in resolved_dir.glob(validated_pattern) if path.is_file())
    if not files:
        raise ExportError(f"未找到匹配 {validated_pattern!r} 的 CSV：{resolved_dir}")
    symbolic_links = [path for path in files if path.is_symlink()]
    if symbolic_links:
        raise ExportError(f"为避免替换符号链接，已拒绝处理：{symbolic_links[0]}")

    source = translations
    if source is None:
        source = (translation_loader or load_translations_from_database)()
    normalized_translations = normalize_translations(source)
    if not normalized_translations:
        raise ExportError("数据库或注入数据中没有非空标题译文")

    summary = ExportSummary(
        csv_dir=resolved_dir,
        pattern=validated_pattern,
        dry_run=bool(dry_run),
        translation_count=len(normalized_translations),
    )
    prepared_files: list[_PreparedFile] = []
    try:
        for path in files:
            prepared = _prepare_file(path, normalized_translations, bool(dry_run))
            prepared_files.append(prepared)
            summary.add_file(prepared.stats)
        if not dry_run:
            summary.files_written = _commit_prepared_files(prepared_files)
    except Exception:
        for prepared in prepared_files:
            _unlink(prepared.temporary_path)
            prepared.temporary_path = None
        raise
    return summary


def print_summary(summary: ExportSummary) -> None:
    mode = "预览" if summary.dry_run else "写入"
    for stats in summary.files:
        action = "无需更新"
        if stats.needs_write:
            action = "将更新" if summary.dry_run else "已更新"
        print(
            f"[{action}] {stats.path.name}: 行={stats.rows}, 匹配={stats.matched_rows}, "
            f"填充={stats.filled_rows}, 更新={stats.updated_rows}, "
            f"未匹配={stats.unmatched_rows}"
        )
    print(
        f"[{mode}汇总] 译文={summary.translation_count}, 文件={summary.files_scanned}, "
        f"需更新={summary.files_changed}, 已写入={summary.files_written}, 行={summary.rows}, "
        f"匹配={summary.matched_rows}, 填充={summary.filled_rows}, "
        f"更新={summary.updated_rows}, 未匹配={summary.unmatched_rows}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 gallery_info 数据库只读提取标题译文，并按 ID 原子更新 *_full.csv。"
    )
    parser.add_argument(
        "--csv-dir",
        default=str(DEFAULT_CSV_DIR.relative_to(PROJECT_ROOT)),
        help="CSV 目录，默认 data/gallery_info",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help='目录内文件匹配模式，默认 "*_full.csv"',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只读取数据库和 CSV 并输出统计，不写文件",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    try:
        summary = export_title_translations(
            csv_dir=args.csv_dir,
            pattern=args.pattern,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
