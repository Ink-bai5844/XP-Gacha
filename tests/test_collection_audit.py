from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from tools.audit_collection_benchmark import audit_collection, csv_snapshot, state_snapshot


class CollectionAuditTests(unittest.TestCase):
    def csv_bytes(self, rows: list[list[str]]) -> bytes:
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(["ID", "标题"])
        writer.writerows(rows)
        return stream.getvalue().encode("utf-8-sig")

    def state_bytes(self, ids: list[str], *, page: int = 1, info: bool = True, image: bool = True) -> bytes:
        events = [{"event": "run_start", "identity": "same"}, {"event": "page_complete", "page": page}]
        events += [{"event": "task_state", "item": {"id": key}, "info_ok": info, "thumb_ok": image} for key in ids]
        return ("\n".join(json.dumps(event) for event in events) + "\n").encode("utf-8")

    def audit(self, root: Path, before_csv: bytes, after_csv: bytes, before_state: bytes, after_state: bytes, **kwargs):
        paths = {"before_csv": root / "before.csv", "after_csv": root / "after.csv",
                 "before_state": root / "before.jsonl", "after_state": root / "after.jsonl"}
        payloads = {"before_csv": before_csv, "after_csv": after_csv,
                    "before_state": before_state, "after_state": after_state}
        for key, path in paths.items():
            path.write_bytes(payloads[key])
        result = audit_collection(**paths, **kwargs)
        for key, path in paths.items():
            self.assertEqual(path.read_bytes(), payloads[key], "audit mutated input")
        return result

    def test_append_preserves_old_rows_flags_and_never_reports_titles(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.audit(Path(temporary), self.csv_bytes([["NH1", "private,title\nsecond line"]]),
                                self.csv_bytes([["NH1", "private,title\nsecond line"], ["NH2", "other-private"]]),
                                self.state_bytes(["NH1"]), self.state_bytes(["NH1", "NH2"]))
        self.assertTrue(result["ok"])
        self.assertTrue(result["checks"]["originalBytePrefixPreserved"])
        self.assertEqual(result["newCsvIds"], 1)
        self.assertNotIn("private", json.dumps(result))

    def test_changes_duplicate_ids_and_regressed_success_are_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.audit(Path(temporary), self.csv_bytes([["NH1", "secret-original"], ["NH2", "secret-removed"]]),
                                self.csv_bytes([["NH1", "secret-changed"], ["NH1", "secret-changed"]]),
                                self.state_bytes(["NH1", "NH2"]), self.state_bytes(["NH1"], page=2, info=False))
        self.assertFalse(result["ok"])
        self.assertEqual(result["anomalies"]["changedCsvIds"]["ids"], ["NH1"])
        self.assertEqual(result["anomalies"]["newDuplicateCsvIds"]["ids"], ["NH1"])
        self.assertEqual(result["anomalies"]["missingCsvIds"]["ids"], ["NH2"])
        self.assertEqual(result["anomalies"]["lostInformationSuccessIds"]["count"], 2)
        self.assertEqual(result["anomalies"]["lostCompletedPages"]["ids"], [1])
        self.assertNotIn("secret", json.dumps(result))

    def test_running_snapshot_can_ignore_only_incomplete_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_csv = self.csv_bytes([["NH1", "title"]])
            raw_state = self.state_bytes(["NH1"])
            result = self.audit(Path(temporary), raw_csv, raw_csv + b'NH2,"unfinished',
                                raw_state, raw_state + b'{"event":', allow_partial_tail=True)
            self.assertTrue(result["ok"])
            self.assertTrue(result["after"]["ignoredCsvPartialTail"])
            self.assertTrue(result["after"]["ignoredStatePartialTail"])
            with self.assertRaises(ValueError):
                csv_snapshot(Path(temporary) / "after.csv")
            with self.assertRaises(ValueError):
                state_snapshot(Path(temporary) / "after.jsonl")

    def test_invalid_middle_record_is_not_hidden_as_a_partial_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.jsonl"
            path.write_bytes(b'bad\n{"event":"run_start"}\n')
            with self.assertRaises(ValueError):
                state_snapshot(path, allow_partial_tail=True)
            path.write_bytes(b'ID,title\nNH1\nNH2,valid\n')
            with self.assertRaises(ValueError):
                csv_snapshot(path, allow_partial_tail=True)

    def test_running_csv_tail_can_end_mid_utf8_character(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "partial.csv"
            path.write_bytes(self.csv_bytes([["NH1", "old"]]) + b"NH2," + "中".encode("utf-8")[:1])
            result = csv_snapshot(path, allow_partial_tail=True)
            self.assertTrue(result["ignoredPartialTail"])
            self.assertEqual(list(result["rows"]), ["NH1"])

    def test_header_or_encoding_changes_fail_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            old_csv = self.csv_bytes([["NH1", "title"]])
            result = self.audit(Path(temporary), old_csv, old_csv[3:],
                                self.state_bytes(["NH1"]), self.state_bytes(["NH1"]))
        self.assertFalse(result["ok"])
        self.assertFalse(result["checks"]["utf8BomPreserved"])


if __name__ == "__main__":
    unittest.main()
