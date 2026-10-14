"""Offline coverage for listing-scoped metadata repair and fair resume queues."""
from __future__ import annotations

import csv
import io
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from data_get.collector import (
    BinaryPayload, CSV_HEADERS, Checkpoint, CollectionConfig, CollectionItem,
    CsvStore, GalleryInfo, NHAdapter, OnlineCollectionRunner, ParsedGallery,
    TaskState, run_collection,
)
from data_get.metadata_review import MetadataReviewStore


def full_info(gallery_id):
    route = "album" if gallery_id.startswith("JM") else "g"
    return GalleryInfo(
        gallery_id, f"https://example.test/{route}/{gallery_id[2:]}/", "Fixture title",
        tags="Fixture tag", authors="Fixture author", groups="Fixture group",
        languages="Chinese", pages="12", uploaded_date="2026-09-01",
    )


def png():
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), (12, 34, 56)).save(stream, format="PNG")
    return stream.getvalue()


class ScopeAdapter:
    _incomplete_info_fields = staticmethod(NHAdapter._incomplete_info_fields)
    validate_detail = staticmethod(NHAdapter.validate_detail)

    def __init__(self, pages):
        self.pages = pages
        self.checked_ids = []
        self.detail_ids = []
        self.discovered_pages = []
        self.detail_count_at_list = {}
        self.lock = threading.Lock()

    def is_info_complete(self, row):
        self.checked_ids.append(row["ID"])
        return NHAdapter.is_info_complete(row)

    def discover_page(self, page):
        with self.lock:
            self.discovered_pages.append(page)
            self.detail_count_at_list[page] = len(self.detail_ids)
        # Intentionally omit page to verify the runner owns page provenance.
        return [CollectionItem(gallery_id, full_info(gallery_id).link)
                for gallery_id in self.pages[page]]

    def fetch_detail(self, item):
        with self.lock:
            self.detail_ids.append(item.id)
        return ParsedGallery(full_info(item.id), f"https://img.test/{item.id}.png")

    def fetch_thumbnail(self, url):
        return BinaryPayload(png(), "image/png")


class CollectionScopeTests(unittest.TestCase):
    def config(self, root, source="NH", **overrides):
        args = dict(
            mode=f"{source.lower()}-online", base_url="https://example.test",
            start_url="https://example.test/list", max_pages=2,
            output_csv=root / "origin.csv", image_dir=root / "images", workers=2,
            request_attempts=1, max_rounds=1, retry_backoff=0,
            state_file=root / "state.jsonl", error_log=root / "errors.jsonl",
        )
        args.update(overrides)
        return CollectionConfig(**args)

    def seed_rows(self, config, *ids):
        with config.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_HEADERS)
            writer.writeheader()
            writer.writerows(replace(full_info(gallery_id), authors="").as_csv_row()
                             for gallery_id in ids)
        config.image_dir.mkdir(exist_ok=True)
        for gallery_id in ids:
            (config.image_dir / f"{gallery_id}.png").write_bytes(png())

    def rows(self, config):
        with config.output_csv.open(encoding="utf-8-sig", newline="") as handle:
            return {row["ID"]: row for row in csv.DictReader(handle)}

    def checkpoint(self, config, entries, completed_pages=(), *, completed=False):
        resolved = config.resolved()
        checkpoint = Checkpoint(resolved.state_file)
        checkpoint.append({"event": "run_start", "mode": resolved.mode, "identity": resolved.identity})
        for page in completed_pages:
            checkpoint.append({"event": "page_complete", "page": page})
        for gallery_id, page in entries:
            checkpoint.task(TaskState(
                CollectionItem(gallery_id, full_info(gallery_id).link, page=page),
                info_ok=True, thumb_ok=True,
            ))
        if completed:
            checkpoint.append({"event": "run_complete"})
        checkpoint.close()

    def test_both_sources_only_repair_discovered_rows_and_never_scan_csv_completeness(self):
        for source in ("NH", "JM"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temporary:
                config = self.config(Path(temporary), source)
                ids = [f"{source}{number}" for number in (100, 101, 999)]
                self.seed_rows(config, *ids)
                original_outside = self.rows(config)[ids[2]]
                outside_image = (config.image_dir / f"{ids[2]}.png").read_bytes()
                adapter = ScopeAdapter({1: [ids[0]], 2: [ids[1]]})
                with patch.object(CsvStore, "incomplete_rows", side_effect=AssertionError("CSV-wide scan")):
                    result = run_collection(config, adapter=adapter)
                self.assertTrue(result.success)
                self.assertEqual(result.discovered, 2)
                self.assertCountEqual(adapter.detail_ids, ids[:2])
                self.assertNotIn(ids[2], adapter.checked_ids)
                self.assertEqual(self.rows(config)[ids[2]], original_outside)
                self.assertEqual((config.image_dir / f"{ids[2]}.png").read_bytes(), outside_image)

    def test_resume_filters_page_zero_out_of_range_and_preserves_valid_page_tasks(self):
        for source in ("NH", "JM"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temporary:
                config = self.config(Path(temporary), source)
                ids = [f"{source}{number}" for number in (100, 101, 999)]
                self.seed_rows(config, *ids)
                self.checkpoint(config, [(ids[0], 1), (ids[1], 0), (ids[2], 3)], [1, 99])
                adapter = ScopeAdapter({2: [ids[1]]})
                runner = OnlineCollectionRunner(config.resolved(), adapter)
                self.assertEqual(set(runner.states), {ids[0]})
                self.assertEqual(runner.completed_pages, {1})
                result = runner.run()
                self.assertTrue(result.success)
                self.assertEqual(adapter.discovered_pages, [2])
                self.assertCountEqual(adapter.detail_ids, ids[:2])
                self.assertNotIn(ids[2], adapter.checked_ids)
                replay = Checkpoint(config.state_file).replay(config.resolved().identity)
                self.assertEqual(set(replay.tasks), set(ids[:2]))
                self.assertEqual(replay.tasks[ids[1]].item.page, 2)

    def test_all_page_zero_pollution_is_removed_before_new_discovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary), max_pages=1)
            self.seed_rows(config, "NH100", "NH999")
            self.checkpoint(config, [("NH100", 0), ("NH999", 0)])
            adapter = ScopeAdapter({1: ["NH100"]})
            runner = OnlineCollectionRunner(config.resolved(), adapter)
            self.assertEqual(runner.states, {})
            self.assertEqual(adapter.checked_ids, [])
            self.assertTrue(runner.run().success)
            self.assertEqual(adapter.detail_ids, ["NH100"])
            self.assertEqual(self.rows(config)["NH999"]["作者"], "")

    def test_completed_run_rediscovers_range_and_reuses_confirmed_site_absence(self):
        for source in ("NH", "JM"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temporary:
                config = self.config(Path(temporary), source, max_pages=1)
                gallery_id, outside_id = f"{source}100", f"{source}999"
                self.seed_rows(config, gallery_id, outside_id)
                self.checkpoint(config, [(gallery_id, 1)], [1], completed=True)
                reviews = MetadataReviewStore(config.output_csv, CSV_HEADERS)
                reviews.record(self.rows(config)[gallery_id], ["作者"])
                adapter = ScopeAdapter({1: [gallery_id]})
                result = run_collection(config, adapter=adapter)
                self.assertTrue(result.success)
                self.assertEqual(adapter.discovered_pages, [1])
                self.assertEqual(adapter.detail_ids, [])
                self.assertNotIn(outside_id, adapter.checked_ids)

    def test_local_info_scope_follows_current_input_links_not_old_checkpoint_or_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_file = root / "links.txt"
            input_file.write_text("https://example.test/g/100/\n", encoding="utf-8")
            config = self.config(root, mode="nh-local-info", input_file=input_file)
            self.seed_rows(config, "NH100", "NH999")
            self.checkpoint(config, [("NH100", 0), ("NH999", 0)])
            adapter = ScopeAdapter({})
            result = run_collection(config, adapter=adapter)
            self.assertTrue(result.success)
            self.assertEqual(result.discovered, 1)
            self.assertEqual(adapter.detail_ids, ["NH100"])
            self.assertNotIn("NH999", adapter.checked_ids)
            self.assertEqual(self.rows(config)["NH999"]["作者"], "")

    def test_listing_is_not_starved_by_large_resume_backlog(self):
        for workers in (1, 3):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as temporary:
                config = self.config(Path(temporary), max_pages=3, workers=workers)
                backlog = [f"NH{number}" for number in range(100, 140)]
                self.checkpoint(config, [(gallery_id, 1) for gallery_id in backlog], [1])
                adapter = ScopeAdapter({2: ["NH200"], 3: ["NH300"]})
                result = run_collection(config, adapter=adapter)
                self.assertTrue(result.success)
                self.assertEqual(result.discovered, len(backlog) + 2)
                self.assertEqual(adapter.discovered_pages, [2, 3])
                self.assertLess(adapter.detail_count_at_list[2], len(backlog))
                self.assertLess(adapter.detail_count_at_list[3], len(backlog))
                self.assertCountEqual(adapter.detail_ids, backlog + ["NH200", "NH300"])


if __name__ == "__main__":
    unittest.main()
