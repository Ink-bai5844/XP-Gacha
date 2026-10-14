from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_get.collector import CSV_HEADERS
from data_get.metadata_review import MetadataReviewStore


class MetadataReviewStoreTests(unittest.TestCase):
    def row(self):
        return dict(zip(CSV_HEADERS, ("NH1", "https://example.test/g/1/", "Fixture", "tag", "", "group", "Chinese", "12", "2026-09-01")))

    def test_torn_receipt_tail_does_not_swallow_next_durable_record(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "origin.csv"
            store = MetadataReviewStore(path, CSV_HEADERS)
            store.path.write_bytes(b'{"version":1,"id":"torn')
            store.record(self.row(), ("作者",))
            restored = MetadataReviewStore(path, CSV_HEADERS)
            self.assertEqual(restored.unavailable(self.row()), ("作者",))
            self.assertIn(b'"torn', store.path.read_bytes())

    def test_corrupt_wrong_version_and_identity_exemptions_are_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            store = MetadataReviewStore(Path(temp) / "origin.csv", CSV_HEADERS)
            store.path.write_text('[]\nnull\n{bad json}\n{"version":0}\n', encoding="utf-8")
            restored = MetadataReviewStore(store.path.with_name("origin.csv"), CSV_HEADERS)
            self.assertEqual(restored.unavailable(self.row()), ())
            with self.assertRaises(ValueError):
                store.record(self.row(), ("标题",))

    def test_fsync_failure_does_not_publish_in_memory_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            store = MetadataReviewStore(Path(temp) / "origin.csv", CSV_HEADERS)
            with patch("data_get.metadata_review.os.fsync", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    store.record(self.row(), ("作者",))
            self.assertEqual(store.unavailable(self.row()), ())

    def test_receipt_is_bound_to_all_raw_fields_but_not_user_translation(self):
        with tempfile.TemporaryDirectory() as temp:
            store = MetadataReviewStore(Path(temp) / "origin.csv", CSV_HEADERS)
            row = self.row()
            store.record(row, ("作者",))
            self.assertEqual(store.unavailable({**row, "标题译文": "译文"}), ("作者",))
            for key in CSV_HEADERS:
                with self.subTest(key=key):
                    self.assertEqual(store.unavailable({**row, key: "changed"}), ())


if __name__ == "__main__":
    unittest.main()
