from __future__ import annotations

import csv
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.export_title_translations_to_csv import (
    ExportError,
    ExportSummary,
    export_title_translations,
    load_translations_from_database,
    main,
)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return list(self._rows)


class _FakeConnection:
    def __init__(self, rows):
        self.rows = rows
        self.statements: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, statement):
        self.statements.append(str(statement))
        return _FakeResult(self.rows)


class _FakeEngine:
    def __init__(self, rows):
        self.connection = _FakeConnection(rows)

    def connect(self):
        return self.connection


class TitleTranslationExportTests(unittest.TestCase):
    def test_adds_translation_after_title_and_preserves_rows_and_unknown_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            csv_dir = Path(temporary_dir)
            path = csv_dir / "catalog_full.csv"
            write_csv(
                path,
                ["ID", "标题", "未知列", "文件名"],
                [
                    {"ID": " nh1 ", "标题": "Title 1", "未知列": "含,逗号", "文件名": "A"},
                    {"ID": "JM2", "标题": "Title 2", "未知列": "保留值", "文件名": "B"},
                ],
            )

            summary = export_title_translations(
                csv_dir,
                translations={"ＮＨ１": "译文一", "JM2": None},
            )

            fieldnames, rows = read_csv(path)
            self.assertEqual(fieldnames, ["ID", "标题", "标题译文", "未知列", "文件名"])
            self.assertEqual([row["ID"] for row in rows], [" nh1 ", "JM2"])
            self.assertEqual([row["标题译文"] for row in rows], ["译文一", ""])
            self.assertEqual([row["未知列"] for row in rows], ["含,逗号", "保留值"])
            self.assertEqual(summary.files_scanned, 1)
            self.assertEqual(summary.files_written, 1)
            self.assertEqual(summary.rows, 2)
            self.assertEqual(summary.matched_rows, 1)
            self.assertEqual(summary.filled_rows, 1)
            self.assertEqual(summary.unmatched_rows, 1)
            self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_database_nonempty_values_update_but_empty_values_do_not_erase_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            csv_dir = Path(temporary_dir)
            path = csv_dir / "catalog_full.csv"
            write_csv(
                path,
                ["ID", "标题译文", "标题", "未知列"],
                [
                    {"ID": "NH1", "标题": "One", "标题译文": "旧译文", "未知列": "1"},
                    {"ID": "NH2", "标题": "Two", "标题译文": "保留我", "未知列": "2"},
                    {"ID": "NH3", "标题": "Three", "标题译文": "", "未知列": "3"},
                    {"ID": "NH4", "标题": "Four", "标题译文": "相同译文", "未知列": "4"},
                ],
            )

            summary = export_title_translations(
                csv_dir,
                translations={
                    "NH1": "新译文",
                    "NH2": "   ",
                    "NH3": "补充译文",
                    "NH4": "相同译文",
                },
            )

            fieldnames, rows = read_csv(path)
            self.assertEqual(fieldnames, ["ID", "标题", "标题译文", "未知列"])
            self.assertEqual(
                [row["标题译文"] for row in rows],
                ["新译文", "保留我", "补充译文", "相同译文"],
            )
            self.assertEqual(summary.matched_rows, 3)
            self.assertEqual(summary.updated_rows, 1)
            self.assertEqual(summary.filled_rows, 1)
            self.assertEqual(summary.unchanged_rows, 1)
            self.assertEqual(summary.unmatched_rows, 1)

    def test_dry_run_reports_changes_without_touching_file_or_creating_temps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            csv_dir = Path(temporary_dir)
            path = csv_dir / "catalog_full.csv"
            write_csv(path, ["ID", "标题", "额外"], [{"ID": "NH1", "标题": "One", "额外": "x"}])
            before = path.read_bytes()

            summary = export_title_translations(
                csv_dir,
                dry_run=True,
                translation_loader=lambda: [("NH1", "译文")],
            )

            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(summary.files_changed, 1)
            self.assertEqual(summary.files_written, 0)
            self.assertEqual(summary.filled_rows, 1)
            self.assertEqual([item.name for item in csv_dir.iterdir()], [path.name])

    def test_prepare_failure_leaves_all_original_files_unchanged_and_cleans_temps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            csv_dir = Path(temporary_dir)
            first = csv_dir / "a_full.csv"
            second = csv_dir / "b_full.csv"
            write_csv(first, ["ID", "标题"], [{"ID": "NH1", "标题": "One"}])
            write_csv(second, ["链接", "标题"], [{"链接": "https://example.test", "标题": "Bad"}])
            first_before = first.read_bytes()
            second_before = second.read_bytes()

            with self.assertRaisesRegex(ExportError, "缺少 ID 列"):
                export_title_translations(csv_dir, translations={"NH1": "译文"})

            self.assertEqual(first.read_bytes(), first_before)
            self.assertEqual(second.read_bytes(), second_before)
            self.assertEqual(sorted(item.name for item in csv_dir.iterdir()), [first.name, second.name])

    def test_replace_failure_keeps_original_and_cleans_output_and_rollback_temps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            csv_dir = Path(temporary_dir)
            path = csv_dir / "catalog_full.csv"
            write_csv(path, ["ID", "标题"], [{"ID": "NH1", "标题": "One"}])
            before = path.read_bytes()

            with (
                mock.patch(
                    "tools.export_title_translations_to_csv.os.replace",
                    side_effect=OSError("locked"),
                ),
                self.assertRaisesRegex(ExportError, "原子替换 CSV 失败"),
            ):
                export_title_translations(csv_dir, translations={"NH1": "译文"})

            self.assertEqual(path.read_bytes(), before)
            self.assertEqual([item.name for item in csv_dir.iterdir()], [path.name])

    def test_later_replace_failure_rolls_back_already_committed_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            csv_dir = Path(temporary_dir)
            first = csv_dir / "a_full.csv"
            second = csv_dir / "b_full.csv"
            write_csv(first, ["ID", "标题"], [{"ID": "NH1", "标题": "One"}])
            write_csv(second, ["ID", "标题"], [{"ID": "NH2", "标题": "Two"}])
            originals = {first: first.read_bytes(), second: second.read_bytes()}
            real_replace = __import__("os").replace

            def fail_second_commit(source, target):
                source_path = Path(source)
                target_path = Path(target)
                if target_path == second and ".rollback.tmp" not in source_path.name:
                    raise OSError("second file locked")
                return real_replace(source, target)

            with (
                mock.patch(
                    "tools.export_title_translations_to_csv.os.replace",
                    side_effect=fail_second_commit,
                ),
                self.assertRaisesRegex(ExportError, "原子替换 CSV 失败"),
            ):
                export_title_translations(
                    csv_dir,
                    translations={"NH1": "译文一", "NH2": "译文二"},
                )

            self.assertEqual(first.read_bytes(), originals[first])
            self.assertEqual(second.read_bytes(), originals[second])
            self.assertEqual(sorted(item.name for item in csv_dir.iterdir()), [first.name, second.name])

    def test_matching_file_with_correct_values_is_not_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            csv_dir = Path(temporary_dir)
            path = csv_dir / "catalog_full.csv"
            write_csv(
                path,
                ["ID", "标题", "标题译文", "额外"],
                [{"ID": "NH1", "标题": "One", "标题译文": "相同译文", "额外": "x"}],
            )
            before = path.read_bytes()

            summary = export_title_translations(csv_dir, translations={"NH1": "相同译文"})

            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(summary.files_changed, 0)
            self.assertEqual(summary.files_written, 0)
            self.assertEqual(summary.unchanged_rows, 1)
            self.assertEqual([item.name for item in csv_dir.iterdir()], [path.name])

    def test_conflicting_values_for_same_normalized_id_fail_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            csv_dir = Path(temporary_dir)
            path = csv_dir / "catalog_full.csv"
            write_csv(path, ["ID", "标题"], [{"ID": "NH1", "标题": "One"}])
            before = path.read_bytes()

            with self.assertRaisesRegex(ExportError, "多个不同"):
                export_title_translations(
                    csv_dir,
                    translations=[("NH1", "译文 A"), (" nh1 ", "译文 B")],
                )

            self.assertEqual(path.read_bytes(), before)

    def test_pattern_cannot_escape_or_scan_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            csv_dir = Path(temporary_dir)
            path = csv_dir / "catalog_full.csv"
            write_csv(path, ["ID", "标题"], [{"ID": "NH1", "标题": "One"}])

            for pattern in ("../*_full.csv", "nested/*_full.csv", "nested\\*_full.csv"):
                with self.subTest(pattern=pattern), self.assertRaisesRegex(
                    ExportError, "文件名模式"
                ):
                    export_title_translations(
                        csv_dir,
                        pattern=pattern,
                        translations={"NH1": "译文"},
                    )

    def test_database_loader_executes_only_the_two_column_select(self) -> None:
        engine = _FakeEngine([("NH1", "译文"), ("NH2", None)])

        rows = load_translations_from_database(engine)

        self.assertEqual(rows, [("NH1", "译文"), ("NH2", None)])
        self.assertEqual(
            engine.connection.statements,
            ["SELECT `ID`, `标题译文` FROM `gallery_info`"],
        )

    def test_main_returns_zero_for_success_and_one_for_operational_failure(self) -> None:
        summary = ExportSummary(
            csv_dir=Path("data/gallery_info"),
            pattern="*_full.csv",
            dry_run=True,
            translation_count=1,
        )
        with (
            mock.patch(
                "tools.export_title_translations_to_csv.export_title_translations",
                return_value=summary,
            ) as export,
            mock.patch("tools.export_title_translations_to_csv.print_summary") as print_result,
        ):
            return_code = main(["--dry-run"])

        self.assertEqual(return_code, 0)
        export.assert_called_once_with(
            csv_dir=str(Path("data") / "gallery_info"),
            pattern="*_full.csv",
            dry_run=True,
        )
        print_result.assert_called_once_with(summary)

        with (
            mock.patch(
                "tools.export_title_translations_to_csv.export_title_translations",
                side_effect=ExportError("test failure"),
            ),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
        ):
            return_code = main([])

        self.assertEqual(return_code, 1)
        self.assertIn("test failure", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
