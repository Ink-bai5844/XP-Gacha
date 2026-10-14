from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from data_get.collector import (
    CSV_HEADERS,
    Checkpoint,
    CollectionConfig,
    CollectionItem,
    CollectionRequestError,
    CsvStore,
    GalleryInfo,
    JMAdapter,
    NHAdapter,
    OnlineCollectionRunner,
    ParsedGallery,
    TaskState,
    run_collection,
)


def full_info(gallery_id: str = "NH100") -> GalleryInfo:
    route = "album" if gallery_id.startswith("JM") else "g"
    return GalleryInfo(
        gallery_id,
        f"https://example.test/{route}/{gallery_id[2:]}/",
        "Synthetic title",
        tags="synthetic-tag",
        authors="Synthetic author",
        groups="Synthetic group",
        languages="Chinese",
        pages="24",
        uploaded_date="2026-09-01",
    )


def detail_html(*, complete: bool = True) -> str:
    title = '<h2 class="title"><span class="pretty">Synthetic title</span></h2>'
    if not complete:
        return title
    return title + """
        <div class="tag-container field-name svelte-fixture">
          Artists: <a><span class="name">Synthetic author</span></a>
        </div>
        <div class="tag-container field-name svelte-fixture">
          Groups: <a><span class="name">Synthetic group</span></a>
        </div>
        <div class="tag-container field-name svelte-fixture">
          Tags: <a><span class="name">synthetic-tag</span></a>
        </div>
        <div class="svelte-fixture field-name tag-container">
          Languages: <a><span class="name">Chinese</span></a>
        </div>
        <div class="tag-container field-name svelte-fixture">
          Pages: <a><span class="name">24</span></a>
        </div>
        <div class="tag-container field-name svelte-fixture">
          Uploaded: <time datetime="2026-09-01T12:34:56+00:00">recently</time>
        </div>
        <div id="cover"><img src="https://images.example.test/test.png"></div>
    """


def valid_png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (2, 2), (10, 20, 30)).save(output, format="PNG")
    return output.getvalue()


class FixtureNHAdapter(NHAdapter):
    """Keep the production parser and completeness contract, but never network."""

    def __init__(self, config, *, complete: bool = True, discovered_ids=None):
        super().__init__(config)
        self.complete_fixture = complete
        self.discovered_ids = discovered_ids
        self.detail_calls: Counter[str] = Counter()
        self.discovery_calls = 0
        self.thumbnail_calls = 0

    def _get(self, url: str):
        self.detail_calls[url] += 1
        return SimpleNamespace(text=detail_html(complete=self.complete_fixture))

    def discover_page(self, page: int):
        self.discovery_calls += 1
        if self.discovered_ids is not None:
            return [CollectionItem(gallery_id, full_info(gallery_id).link, page=page)
                    for gallery_id in self.discovered_ids]
        raise AssertionError("completed discovery must not be repeated during metadata repair")

    def fetch_thumbnail(self, url: str):
        self.thumbnail_calls += 1
        raise AssertionError("an existing valid thumbnail must not be downloaded again")


class FixtureJMAdapter(JMAdapter):
    detail_calls = 0

    def fetch_detail(self, item: CollectionItem):
        self.detail_calls += 1
        return ParsedGallery(full_info(item.id))

    def discover_page(self, page: int):
        raise AssertionError("completed JM discovery must not be repeated")


class NHMetadataRepairTests(unittest.TestCase):
    def config(self, root: Path, **overrides) -> CollectionConfig:
        values = {
            "mode": "nh-online",
            "base_url": "https://example.test",
            "start_url": "https://example.test/list",
            "max_pages": 1,
            "output_csv": root / "NH_info_test.csv",
            "image_dir": root / "images",
            "workers": 1,
            "request_attempts": 1,
            "max_rounds": 1,
            "retry_backoff": 0,
            "state_file": root / "state.jsonl",
            "error_log": root / "errors.jsonl",
        }
        values.update(overrides)
        return CollectionConfig(**values)

    def write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(CSV_HEADERS)
        for row in rows:
            fields.extend(name for name in row if name not in fields)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def read_csv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def seed_checkpoint(self, config: CollectionConfig, *ids: str) -> None:
        resolved = config.resolved()
        checkpoint = Checkpoint(resolved.state_file)
        checkpoint.append({"event": "run_start", "mode": resolved.mode, "identity": resolved.identity})
        checkpoint.append({"event": "page_complete", "page": 1})
        for gallery_id in ids:
            checkpoint.task(TaskState(
                CollectionItem(gallery_id, full_info(gallery_id).link, page=1),
                info_ok=True,
                thumb_ok=True,
            ))
        checkpoint.close()

    def seed_images(self, root: Path, *ids: str) -> dict[str, bytes]:
        folder = root / "images"
        folder.mkdir(exist_ok=True)
        payload = valid_png()
        for gallery_id in ids:
            (folder / f"{gallery_id}.png").write_bytes(payload)
        return {gallery_id: payload for gallery_id in ids}

    def test_completeness_requires_every_original_metadata_field(self):
        adapter = NHAdapter(CollectionConfig(mode="nh-online").resolved())
        row = full_info().as_csv_row()
        self.assertTrue(adapter.is_info_complete(row))
        title_only = GalleryInfo("NH100", row["链接"], "Synthetic title").as_csv_row()
        self.assertFalse(adapter.is_info_complete(title_only))
        for field in CSV_HEADERS:
            with self.subTest(missing=field):
                self.assertFalse(adapter.is_info_complete({**row, field: ""}))
        for field, value in (("页数", ""), ("页数", "0"), ("页数", "-2"), ("页数", "unknown"), ("上传日期", ""), ("上传日期", "2026-99-40")):
            with self.subTest(field=field, value=value):
                incomplete = {**row, field: value}
                self.assertFalse(adapter.is_info_complete(incomplete))

    def test_extra_and_reordered_css_classes_preserve_required_metadata(self):
        adapter = FixtureNHAdapter(CollectionConfig(mode="nh-online").resolved())
        parsed = adapter.fetch_detail(CollectionItem("NH100", full_info().link))
        self.assertEqual(parsed.info.as_csv_row(), full_info().as_csv_row())
        self.assertEqual(parsed.thumbnail_url, "https://images.example.test/test.png")

    def test_title_only_html_is_a_retryable_parse_failure(self):
        adapter = FixtureNHAdapter(CollectionConfig(mode="nh-online").resolved(), complete=False)
        with self.assertRaises(CollectionRequestError) as failure:
            adapter.fetch_detail(CollectionItem("NH100", full_info().link))
        self.assertTrue(failure.exception.retryable)

    def test_false_success_checkpoint_is_corrected_and_existing_thumbnail_kept(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.config(root)
            row = GalleryInfo("NH100", full_info().link, "Old title").as_csv_row()
            row.update({"标题译文": "Saved translation", "自定义备注": "keep me"})
            self.write_csv(config.output_csv, [row])
            originals = self.seed_images(root, "NH100")
            self.seed_checkpoint(config, "NH100")
            adapter = FixtureNHAdapter(config.resolved())
            runner = OnlineCollectionRunner(config.resolved(), adapter, sleep_fn=lambda _: None)
            try:
                self.assertFalse(runner.states["NH100"].info_ok)
                self.assertTrue(runner.states["NH100"].thumb_ok)
                summary = runner.run()
            finally:
                runner.csv_store.close()
                runner.checkpoint.close()
                runner.error_writer.close()
            self.assertTrue(summary.success)
            self.assertEqual(sum(adapter.detail_calls.values()), 1)
            self.assertEqual(adapter.discovery_calls, 0)
            self.assertEqual(adapter.thumbnail_calls, 0)
            saved = self.read_csv(config.output_csv)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["语言"], "Chinese")
            self.assertEqual(saved[0]["页数"], "24")
            self.assertEqual(saved[0]["上传日期"], "2026-09-01")
            self.assertEqual(saved[0]["标题译文"], "Saved translation")
            self.assertEqual(saved[0]["自定义备注"], "keep me")
            self.assertEqual((root / "images" / "NH100.png").read_bytes(), originals["NH100"])

    def test_discovered_incomplete_row_is_repaired_but_outside_rows_are_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.config(root)
            incomplete = GalleryInfo("NH100", full_info().link, "Old title").as_csv_row()
            complete = full_info("NH200").as_csv_row()
            outside = GalleryInfo("NH400", full_info("NH400").link, "Outside title").as_csv_row()
            other_source = GalleryInfo("JM300", "https://example.test/album/300/", "JM title").as_csv_row()
            self.write_csv(config.output_csv, [incomplete, complete, outside, other_source])
            self.seed_images(root, "NH100", "NH200", "NH400", "JM300")
            adapter = FixtureNHAdapter(config.resolved(), discovered_ids=["NH100", "NH200"])
            summary = run_collection(config, adapter=adapter, sleep_fn=lambda _: None)
            self.assertTrue(summary.success)
            self.assertEqual(adapter.detail_calls, Counter({full_info().link: 1}))
            self.assertEqual(adapter.discovery_calls, 1)
            self.assertEqual(adapter.thumbnail_calls, 0)
            saved = {row["ID"]: row for row in self.read_csv(config.output_csv)}
            self.assertEqual(saved["NH100"], full_info().as_csv_row())
            self.assertEqual(saved["NH200"], complete)
            self.assertEqual(saved["NH400"], outside)
            self.assertEqual(saved["JM300"], other_source)

    def test_incomplete_csv_orphan_without_page_membership_is_not_repaired(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.config(root)
            orphan = GalleryInfo("NH100", full_info().link, "Outside title").as_csv_row()
            complete = full_info("NH200").as_csv_row()
            self.write_csv(config.output_csv, [orphan, complete])
            before = config.output_csv.read_bytes()
            self.seed_images(root, "NH100", "NH200")
            self.seed_checkpoint(config, "NH200")
            adapter = FixtureNHAdapter(config.resolved())
            summary = run_collection(config, adapter=adapter, sleep_fn=lambda _: None)
            self.assertTrue(summary.success)
            self.assertEqual(adapter.detail_calls, Counter())
            self.assertEqual(adapter.discovery_calls, 0)
            self.assertEqual(adapter.thumbnail_calls, 0)
            self.assertEqual(config.output_csv.read_bytes(), before)

    def test_completed_checkpoint_rediscovers_current_range_and_repairs_only_current_members(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.config(root)
            row = GalleryInfo("NH100", full_info().link, "Old title").as_csv_row()
            moved_out = GalleryInfo("NH200", full_info("NH200").link, "Outside current page").as_csv_row()
            self.write_csv(config.output_csv, [row, moved_out])
            self.seed_images(root, "NH100", "NH200")
            self.seed_checkpoint(config, "NH100", "NH200")
            checkpoint = Checkpoint(config.state_file)
            checkpoint.append({"event": "run_complete"})
            checkpoint.close()
            self.assertTrue(checkpoint.replay(config.resolved().identity).run_completed)
            adapter = FixtureNHAdapter(config.resolved(), discovered_ids=["NH100"])
            runner = OnlineCollectionRunner(config.resolved(), adapter, sleep_fn=lambda _: None)
            try:
                # A finished run is a fresh current-page scan, not a continuation
                # of stale membership from an earlier version of that page.
                self.assertEqual(runner.completed_pages, set())
                self.assertEqual(runner.states, {})
                # Flush the buffered run_start marker before reading its
                # persistent representation; close is safe to reopen later.
                runner.checkpoint.close()
                replayed = checkpoint.replay(config.resolved().identity)
                self.assertFalse(replayed.run_completed)
                self.assertEqual(replayed.completed_pages, set())
                self.assertEqual(replayed.tasks, {})
                result = runner.run()
            finally:
                runner.csv_store.close()
                runner.checkpoint.close()
                runner.error_writer.close()
            self.assertTrue(result.success)
            self.assertEqual(adapter.detail_calls, Counter({full_info().link: 1}))
            self.assertEqual(adapter.discovery_calls, 1)
            self.assertEqual(adapter.thumbnail_calls, 0)
            self.assertEqual(self.read_csv(config.output_csv), [full_info().as_csv_row(), moved_out])
            final_state = checkpoint.replay(config.resolved().identity)
            self.assertTrue(final_state.run_completed)
            self.assertEqual(final_state.completed_pages, {1})
            self.assertTrue(final_state.tasks["NH100"].info_ok)
            self.assertNotIn("NH200", final_state.tasks)

    def test_jm_title_only_completed_checkpoint_also_requires_metadata_repair(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.config(root, mode="jm-online")
            row = GalleryInfo("JM100", "https://example.test/album/100/", "JM title").as_csv_row()
            self.write_csv(config.output_csv, [row])
            self.seed_images(root, "JM100")
            self.seed_checkpoint(config, "JM100")
            adapter = FixtureJMAdapter(config.resolved())
            summary = run_collection(config, adapter=adapter)
            self.assertTrue(summary.success)
            self.assertEqual(adapter.detail_calls, 1)
            self.assertEqual(self.read_csv(config.output_csv), [full_info("JM100").as_csv_row()])

    def test_failed_repair_does_not_overwrite_existing_data_or_claim_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.config(root)
            row = GalleryInfo("NH100", full_info().link, "Saved title", authors="Saved author").as_csv_row()
            row["标题译文"] = "Saved translation"
            self.write_csv(config.output_csv, [row])
            before = config.output_csv.read_bytes()
            self.seed_images(root, "NH100")
            self.seed_checkpoint(config, "NH100")
            adapter = FixtureNHAdapter(config.resolved(), complete=False)
            summary = run_collection(config, adapter=adapter, sleep_fn=lambda _: None)
            self.assertFalse(summary.success)
            self.assertEqual(summary.pending, 1)
            self.assertEqual(config.output_csv.read_bytes(), before)
            state = Checkpoint(config.state_file).replay(config.resolved().identity).tasks["NH100"]
            self.assertFalse(state.info_ok)
            self.assertTrue(state.thumb_ok)
            self.assertFalse(state.terminal_info)
            errors = [json.loads(line) for line in config.error_log.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(error["stage"] == "detail" and error["retryable"] for error in errors))

    def test_safe_stop_preserves_repair_need_and_next_run_resumes_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.config(root)
            row = GalleryInfo("NH100", full_info().link, "Old title").as_csv_row()
            row["标题译文"] = "Saved translation"
            self.write_csv(config.output_csv, [row])
            before = config.output_csv.read_bytes()
            self.seed_images(root, "NH100")
            self.seed_checkpoint(config, "NH100")
            stop_event = threading.Event()
            stop_event.set()
            adapter = FixtureNHAdapter(config.resolved())
            stopped = run_collection(config, adapter=adapter, stop_event=stop_event)
            self.assertTrue(stopped.interrupted)
            self.assertEqual(sum(adapter.detail_calls.values()), 0)
            self.assertEqual(config.output_csv.read_bytes(), before)
            state = Checkpoint(config.state_file).replay(config.resolved().identity).tasks["NH100"]
            self.assertFalse(state.info_ok)
            self.assertTrue(state.thumb_ok)
            resumed = run_collection(config, adapter=adapter, sleep_fn=lambda _: None)
            self.assertTrue(resumed.success)
            self.assertEqual(sum(adapter.detail_calls.values()), 1)
            self.assertEqual(adapter.discovery_calls, 0)
            self.assertEqual(adapter.thumbnail_calls, 0)
            saved = self.read_csv(config.output_csv)
            self.assertEqual(saved[0]["页数"], "24")
            self.assertEqual(saved[0]["标题译文"], "Saved translation")

    def test_runner_rejects_incomplete_adapter_result_before_counting_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.config(root)
            row = GalleryInfo("NH100", full_info().link, "Saved title").as_csv_row()
            self.write_csv(config.output_csv, [row])
            before = config.output_csv.read_bytes()
            self.seed_images(root, "NH100")
            self.seed_checkpoint(config, "NH100")
            adapter = FixtureNHAdapter(config.resolved())
            # A future adapter regression must not bypass the runner's own
            # validation or contribute a false success to its controller.
            adapter.fetch_detail = lambda item: ParsedGallery(
                GalleryInfo(item.id, item.detail_url, "Invalid replacement")
            )
            runner = OnlineCollectionRunner(config.resolved(), adapter, sleep_fn=lambda _: None)
            recorded_successes = []
            runner.concurrency.record_success = lambda: recorded_successes.append(True)
            try:
                result = runner.run()
            finally:
                runner.csv_store.close()
                runner.checkpoint.close()
                runner.error_writer.close()
            self.assertFalse(result.success)
            self.assertFalse(runner.states["NH100"].info_ok)
            self.assertEqual(recorded_successes, [])
            self.assertEqual(config.output_csv.read_bytes(), before)

    def test_storage_failure_propagates_without_marking_success_or_network_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.config(root, max_rounds=2, request_attempts=3)
            row = GalleryInfo("NH100", full_info().link, "Saved title").as_csv_row()
            self.write_csv(config.output_csv, [row])
            before = config.output_csv.read_bytes()
            self.seed_images(root, "NH100")
            self.seed_checkpoint(config, "NH100")
            adapter = FixtureNHAdapter(config.resolved())
            runner = OnlineCollectionRunner(config.resolved(), adapter, sleep_fn=lambda _: None)
            try:
                with patch.object(runner.csv_store, "upsert", side_effect=OSError("simulated storage full")) as upsert:
                    with self.assertRaisesRegex(OSError, "simulated storage full"):
                        runner.run()
                upsert.assert_called_once()
                self.assertFalse(runner.states["NH100"].info_ok)
                self.assertTrue(runner.stop_event.is_set())
            finally:
                runner.csv_store.close()
                runner.checkpoint.close()
                runner.error_writer.close()
            self.assertEqual(adapter.detail_calls, Counter({full_info().link: 1}))
            self.assertEqual(config.output_csv.read_bytes(), before)
            if config.error_log.exists():
                self.assertEqual(config.error_log.read_text(encoding="utf-8"), "")

    def test_update_preserves_extra_columns_and_nonempty_old_optional_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "origin.csv"
            row = GalleryInfo("NH100", full_info().link, "Old title", authors="Saved author", groups="Saved group").as_csv_row()
            row.update({"标题译文": "Saved translation", "自定义备注": "saved notes"})
            self.write_csv(path, [row])
            store = CsvStore(path)
            try:
                self.assertTrue(store.upsert(replace(full_info(), authors="", groups="")))
            finally:
                store.close()
            saved = self.read_csv(path)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["作者"], "Saved author")
            self.assertEqual(saved[0]["团队"], "Saved group")
            self.assertEqual(saved[0]["标题译文"], "Saved translation")
            self.assertEqual(saved[0]["自定义备注"], "saved notes")
            self.assertEqual(saved[0]["语言"], "Chinese")

    def test_existing_row_update_survives_process_exit_without_close(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "origin.csv"
            row = GalleryInfo("NH100", full_info().link, "Old title", authors="Saved author").as_csv_row()
            row["标题译文"] = "Saved translation"
            self.write_csv(path, [row])
            child = subprocess.run(
                [sys.executable, "-c", (
                    "import os, sys\n"
                    "from pathlib import Path\n"
                    "from data_get.collector import CsvStore, GalleryInfo\n"
                    "store = CsvStore(Path(sys.argv[1]))\n"
                    "store.upsert(GalleryInfo('NH100', 'https://example.test/g/100/', "
                    "'Repaired title', languages='Chinese', pages='24', uploaded_date='2026-09-01'))\n"
                    "os._exit(0)\n"
                ), str(path)],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(child.returncode, 0, child.stderr)
            restored = CsvStore(path)
            try:
                self.assertEqual(restored.rows["NH100"]["页数"], "24")
                self.assertEqual(restored.rows["NH100"]["语言"], "Chinese")
                self.assertEqual(restored.rows["NH100"]["标题译文"], "Saved translation")
                self.assertEqual(restored.rows["NH100"]["作者"], "Saved author")
            finally:
                restored.close()
            saved = self.read_csv(path)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["标题"], "Repaired title")
            reopened = CsvStore(path)
            try:
                self.assertEqual(reopened.rows["NH100"], saved[0])
            finally:
                reopened.close()


if __name__ == "__main__":
    unittest.main()
