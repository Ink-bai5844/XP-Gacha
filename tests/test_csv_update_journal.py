"""Existing-row metadata repair must survive interruption without full rewrites."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_get.collector import CSV_HEADERS, CsvStore, GalleryInfo


class CsvUpdateJournalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "origin.csv"
        self.fields = CSV_HEADERS + ["标题译文", "自定义"]
        self.original = {name: "" for name in self.fields}
        self.original.update({"ID": "NH1", "链接": "https://example.test/g/1/", "标题": "original", "标题译文": "译文", "自定义": "keep"})
        self.write_csv([self.original])

    def write_csv(self, rows: list[dict[str, str]]) -> None:
        with self.path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(rows)

    def read_csv(self) -> list[dict[str, str]]:
        with self.path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def info(self, **kwargs: str) -> GalleryInfo:
        values = dict(id="NH1", link="https://example.test/g/1/", title="repaired", tags="tag", pages="12", languages="chinese")
        values.update(kwargs)
        return GalleryInfo(**values)

    def test_existing_update_is_durable_before_commit_and_replays(self) -> None:
        original_bytes = self.path.read_bytes()
        store = CsvStore(self.path)
        with patch("data_get.collector.os.fsync", wraps=__import__("os").fsync) as fsync:
            self.assertTrue(store.upsert(self.info()))
            self.assertTrue(fsync.called)
        self.assertEqual(store.update_journal_path.name, "origin.csv.updates.jsonl")
        self.assertEqual(self.path.read_bytes(), original_bytes)
        self.assertEqual(store.write_count, 1)
        recovered = CsvStore(self.path)
        self.assertEqual(recovered.rows["NH1"]["页数"], "12")
        self.assertEqual(recovered.rows["NH1"]["标题译文"], "译文")
        self.assertEqual(recovered.write_count, 0)
        recovered.close()
        self.assertFalse(recovered.update_journal_path.exists())
        self.assertEqual(self.read_csv()[0]["页数"], "12")

    def test_multiple_updates_keep_extra_fields_and_ignore_empty_values(self) -> None:
        store = CsvStore(self.path)
        store.upsert(self.info(authors="author"))
        store.upsert(self.info(title="", tags="", authors="", pages="13"))
        self.assertEqual(store.write_count, 2)
        recovered = CsvStore(self.path)
        row = recovered.rows["NH1"]
        self.assertEqual((row["标题"], row["标签"], row["作者"], row["页数"]), ("repaired", "tag", "author", "13"))
        self.assertEqual(row["自定义"], "keep")
        recovered.close()
        self.assertEqual(len(self.read_csv()), 1)

    def test_commit_does_not_double_count_and_is_periodic_not_per_update(self) -> None:
        store = CsvStore(self.path)
        for number in range(24):
            store.upsert(self.info(pages=str(number + 1)))
        self.assertFalse(store.commit_if_due())
        self.assertEqual(self.read_csv()[0]["页数"], "")
        store.upsert(self.info(pages="25"))
        self.assertTrue(store.commit_if_due())
        self.assertEqual(store.write_count, 25)
        self.assertEqual(self.read_csv()[0]["页数"], "25")
        store.upsert(self.info(pages="26"))
        with patch("data_get.collector.time.monotonic", return_value=store._last_commit_time + 30):
            self.assertTrue(store.commit_if_due())
        self.assertEqual(store.write_count, 26)
        store.close()
        self.assertEqual(store.write_count, 26)

    def test_replace_then_failed_journal_cleanup_is_idempotent(self) -> None:
        store = CsvStore(self.path)
        store.upsert(self.info(pages="12"))
        store.upsert(self.info(pages="13"))
        real_unlink = Path.unlink

        def fail_journal(path: Path, *args: object, **kwargs: object) -> None:
            if path == store.update_journal_path:
                raise OSError("simulated crash after CSV replacement")
            real_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", fail_journal):
            with self.assertRaisesRegex(OSError, "simulated crash"):
                store.commit()
        self.assertEqual(self.read_csv()[0]["页数"], "13")
        self.assertTrue(store.update_journal_path.exists())
        recovered = CsvStore(self.path)
        recovered.close()
        self.assertEqual(len(self.read_csv()), 1)
        self.assertEqual(self.read_csv()[0]["页数"], "13")
        self.assertFalse(store.update_journal_path.exists())

    def test_known_intermediate_baseline_can_replay_to_final_update(self) -> None:
        store = CsvStore(self.path)
        store.upsert(self.info(pages="12"))
        intermediate = dict(store.rows["NH1"])
        store.upsert(self.info(pages="13"))
        self.write_csv([intermediate])
        recovered = CsvStore(self.path)
        self.assertEqual(recovered.rows["NH1"]["页数"], "13")
        recovered.close()

    def test_manual_edit_conflict_preserves_csv_and_journal(self) -> None:
        store = CsvStore(self.path)
        store.upsert(self.info())
        edited = dict(self.original, 标题="manual edit")
        self.write_csv([edited])
        before_csv = self.path.read_bytes()
        before_journal = store.update_journal_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "冲突"):
            CsvStore(self.path)
        with self.assertRaisesRegex(ValueError, "外部修改"):
            store.commit()
        self.assertEqual(self.path.read_bytes(), before_csv)
        self.assertEqual(store.update_journal_path.read_bytes(), before_journal)

    def test_missing_csv_with_journal_is_not_replaced_by_empty_file(self) -> None:
        store = CsvStore(self.path)
        store.upsert(self.info())
        self.path.unlink()
        before = store.update_journal_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "冲突"):
            CsvStore(self.path)
        self.assertFalse(self.path.exists())
        self.assertEqual(store.update_journal_path.read_bytes(), before)

    def test_truncated_tail_is_backed_up_and_previous_records_recovered(self) -> None:
        store = CsvStore(self.path)
        store.upsert(self.info())
        valid = store.update_journal_path.read_bytes()
        raw = valid + b'{"version":1,"row":'
        store.update_journal_path.write_bytes(raw)
        recovered = CsvStore(self.path)
        self.assertEqual(recovered.rows["NH1"]["页数"], "12")
        backups = list(self.path.parent.glob("*.updates.jsonl.recovery-*.bak"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), raw)
        self.assertEqual(store.update_journal_path.read_bytes(), valid)
        recovered.close()

    def test_corrupt_complete_record_is_not_treated_as_truncated_tail(self) -> None:
        store = CsvStore(self.path)
        store.upsert(self.info())
        before = store.update_journal_path.read_bytes() + b'{"broken":\n' + b'{"torn":'
        store.update_journal_path.write_bytes(before)
        csv_before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "第 2 行损坏"):
            CsvStore(self.path)
        self.assertEqual(store.update_journal_path.read_bytes(), before)
        self.assertEqual(self.path.read_bytes(), csv_before)
        self.assertFalse(list(self.path.parent.glob("*.bak")))

    def test_missing_final_newline_is_repaired_before_later_append(self) -> None:
        store = CsvStore(self.path)
        store.upsert(self.info())
        store.update_journal_path.write_bytes(store.update_journal_path.read_bytes().rstrip(b"\n"))
        recovered = CsvStore(self.path)
        recovered.upsert(self.info(pages="13"))
        reloaded = CsvStore(self.path)
        self.assertEqual(reloaded.rows["NH1"]["页数"], "13")
        reloaded.close()

    def test_invalid_journal_prevents_csv_repair_side_effects(self) -> None:
        store = CsvStore(self.path)
        store.upsert(self.info())
        store.update_journal_path.write_bytes(store.update_journal_path.read_bytes() + b"invalid\n")
        with self.path.open("ab") as handle:
            handle.write(b"NH2,partial")
        csv_before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "损坏"):
            CsvStore(self.path)
        self.assertEqual(self.path.read_bytes(), csv_before)

    def test_invalid_row_or_update_chain_is_rejected_without_mutation(self) -> None:
        store = CsvStore(self.path)
        store.upsert(self.info())
        raw = store.update_journal_path.read_bytes()
        record = json.loads(raw)
        record["row"]["ID"] = "other"
        bad = json.dumps(record).encode() + b"\n"
        store.update_journal_path.write_bytes(bad)
        with self.assertRaisesRegex(ValueError, "格式无效"):
            CsvStore(self.path)
        self.assertEqual(store.update_journal_path.read_bytes(), bad)
        store.update_journal_path.write_bytes(raw + raw)
        with self.assertRaisesRegex(ValueError, "更新链不连续"):
            CsvStore(self.path)

    def test_failed_fsync_does_not_publish_memory_or_clear_journal(self) -> None:
        store = CsvStore(self.path)
        with patch("data_get.collector.os.fsync", side_effect=OSError("disk error")):
            with self.assertRaisesRegex(OSError, "disk error"):
                store.upsert(self.info())
        self.assertEqual(store.rows["NH1"], self.original)
        self.assertEqual(store.write_count, 0)
        with self.assertRaisesRegex(OSError, "重启"):
            store.commit()
        recovered = CsvStore(self.path)
        self.assertEqual(recovered.rows["NH1"]["页数"], "12")
        recovered.close()

    def test_validation_helpers_return_copies(self) -> None:
        store = CsvStore(self.path)
        complete = lambda row: bool(row["页数"])
        self.assertTrue(store.has_complete("NH1"))
        self.assertFalse(store.has_complete("NH1", complete))
        self.assertFalse(store.has_complete("missing", complete))
        rows = store.incomplete_rows(complete)
        rows[0]["标题"] = "caller mutation"
        self.assertEqual(store.rows["NH1"]["标题"], "original")
        store.upsert(self.info())
        self.assertTrue(store.has_complete("NH1", complete))
        self.assertEqual(store.incomplete_rows(complete), [])
        store.close()

    def test_close_releases_append_handle_when_commit_fails(self) -> None:
        store = CsvStore(self.path)
        store.upsert(self.info(id="NH2"))
        handle = store._append_handle
        self.assertIsNotNone(handle)
        with patch.object(store, "commit", side_effect=ValueError("original failure")):
            with self.assertRaisesRegex(ValueError, "original failure"):
                store.close()
        self.assertTrue(handle.closed)
        self.assertIsNone(store._append_handle)

    def test_new_rows_still_append_immediately_and_empty_csv_can_initialize(self) -> None:
        self.path.write_bytes(b"")
        store = CsvStore(self.path)
        store.upsert(self.info())
        self.assertEqual(self.read_csv()[0]["页数"], "12")
        self.assertFalse(store.update_journal_path.exists())
        self.assertEqual(store.write_count, 1)
        store.close()


if __name__ == "__main__":
    unittest.main()
