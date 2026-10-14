"""Offline regression tests for durable NH/JM missing-metadata decisions."""
from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from data_get.collector import (
    BinaryPayload,
    CSV_HEADERS,
    Checkpoint,
    CollectionConfig,
    CollectionItem,
    GalleryInfo,
    JMAdapter,
    NHAdapter,
    OnlineCollectionRunner,
    ParsedGallery,
    TaskState,
    run_collection,
)


def complete_info(source: str) -> GalleryInfo:
    route = "g" if source == "NH" else "album"
    return GalleryInfo(
        f"{source}100", f"https://example.test/{route}/100/", "Fixture title",
        tags="Fixture tag", authors="Fixture author", groups="Fixture group",
        languages="Chinese", pages="12", uploaded_date="2026-09-01",
    )


class FixtureAdapterMixin:
    """Retain the real validator, replace transport with deterministic results."""

    def __init__(self, config, *, info=None, missing=(), fail=False):
        super().__init__(config)
        self.info = info or complete_info("JM" if config.mode == "jm-online" else "NH")
        self.missing = tuple(missing)
        self.fail = fail
        self.detail_calls = 0
        self.discovery_calls = 0
        self.detail_urls = []
        self.thumbnail_url = ""

    def discover_page(self, page):
        self.discovery_calls += 1
        return [CollectionItem(self.info.id, self.info.link, page=page)]

    def fetch_detail(self, item):
        self.detail_calls += 1
        self.detail_urls.append(item.detail_url)
        if self.fail:
            raise TimeoutError("synthetic metadata timeout")
        return ParsedGallery(self.info, thumbnail_url=self.thumbnail_url, unavailable_fields=self.missing)

    def fetch_thumbnail(self, url):
        raise AssertionError("a valid existing thumbnail must be reused")


class FixtureNHAdapter(FixtureAdapterMixin, NHAdapter):
    pass


class FixtureJMAdapter(FixtureAdapterMixin, JMAdapter):
    pass


def fixture_adapter(config, **kwargs):
    cls = FixtureJMAdapter if config.mode == "jm-online" else FixtureNHAdapter
    return cls(config.resolved(), **kwargs)


def fixture_config(root: Path, source: str, **kwargs) -> CollectionConfig:
    values = dict(
        mode=f"{source.lower()}-online", base_url="https://example.test",
        start_url="https://example.test/list", max_pages=1,
        output_csv=root / "origin.csv", image_dir=root / "images",
        workers=1, request_attempts=1, max_rounds=1, retry_backoff=0,
        state_file=root / "state.jsonl", error_log=root / "errors.jsonl",
    )
    values.update(kwargs)
    return CollectionConfig(**values)


def write_rows(path: Path, rows):
    fields = list(CSV_HEADERS)
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def seed_fixture(config, source: str, row=None):
    info = complete_info(source)
    if row is None:
        row = replace(info, authors="", groups="").as_csv_row()
    write_rows(config.output_csv, [row])
    config.image_dir.mkdir(parents=True, exist_ok=True)
    image = io.BytesIO()
    Image.new("RGB", (2, 2), (10, 20, 30)).save(image, format="PNG")
    (config.image_dir / f"{info.id}.png").write_bytes(image.getvalue())
    seed_checkpoint(config, source)


def seed_checkpoint(config, source: str, *, completed=False):
    info = complete_info(source)
    resolved = config.resolved()
    checkpoint = Checkpoint(resolved.state_file)
    checkpoint.append({"event": "run_start", "mode": resolved.mode, "identity": resolved.identity})
    checkpoint.append({"event": "page_complete", "page": 1})
    checkpoint.task(TaskState(CollectionItem(info.id, info.link, page=1), info_ok=True, thumb_ok=True))
    if completed:
        checkpoint.append({"event": "run_complete"})
    checkpoint.close()


def receipts_path(config) -> Path:
    return config.output_csv.with_name(config.output_csv.name + ".metadata-checks.jsonl")


class MetadataRecheckTests(unittest.TestCase):
    def run_fixture(self, config, **kwargs):
        adapter = fixture_adapter(config, **kwargs)
        progress = []
        result = run_collection(config, adapter=adapter, sleep_fn=lambda _: None,
                                progress_callback=progress.append)
        return result, adapter, progress

    def test_each_missing_source_field_is_repaired_when_discovered_without_an_old_task(self):
        for source in ("NH", "JM"):
            for absent in CSV_HEADERS[1:]:
                with self.subTest(source=source, absent=absent), tempfile.TemporaryDirectory() as temporary:
                    config = fixture_config(Path(temporary), source)
                    row = complete_info(source).as_csv_row()
                    row[absent] = ""
                    seed_fixture(config, source, row)
                    checkpoint = Checkpoint(config.state_file)
                    checkpoint.append({"event": "run_start", "mode": config.mode,
                                       "identity": config.resolved().identity})
                    checkpoint.close()
                    result, adapter, _ = self.run_fixture(config)
                    self.assertTrue(result.success)
                    self.assertEqual(adapter.detail_calls, 1)
                    self.assertEqual(adapter.discovery_calls, 1)
                    self.assertEqual(read_rows(config.output_csv), [complete_info(source).as_csv_row()])

    def test_nh_local_info_uses_same_repair_and_persistent_site_absence_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_file = root / "links.txt"
            input_file.write_text("https://example.test/g/100/\n", encoding="utf-8")
            config = fixture_config(root, "NH", mode="nh-local-info", input_file=input_file)
            seed_fixture(config, "NH")
            partial = replace(complete_info("NH"), groups="")
            result, adapter, progress = self.run_fixture(config, info=partial, missing=("团队",))
            self.assertTrue(result.success)
            self.assertEqual(adapter.detail_calls, 1)
            self.assertEqual(adapter.discovery_calls, 0)
            self.assertEqual(progress[-1]["informationSiteMissing"], 1)
            resumed, second_adapter, _ = self.run_fixture(config, fail=True)
            self.assertTrue(resumed.success)
            self.assertEqual(second_adapter.detail_calls, 0)

    def test_missing_id_is_recovered_from_valid_source_link_and_metadata_is_repaired(self):
        for source in ("NH", "JM"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temporary:
                config = fixture_config(Path(temporary), source)
                row = replace(complete_info(source), authors="", groups="").as_csv_row()
                row["ID"] = ""
                row["标题译文"] = "Keep translation on recovered ID"
                seed_fixture(config, source, row)
                result, adapter, _ = self.run_fixture(config)
                self.assertTrue(result.success)
                self.assertEqual(adapter.detail_calls, 1)
                self.assertEqual(adapter.discovery_calls, 0)
                saved = read_rows(config.output_csv)
                self.assertEqual(len(saved), 1)
                self.assertEqual(saved[0]["ID"], f"{source}100")
                self.assertEqual(saved[0]["作者"], "Fixture author")
                self.assertEqual(saved[0]["标题译文"], row["标题译文"])

    def test_site_missing_metadata_does_not_skip_a_missing_thumbnail(self):
        for source in ("NH", "JM"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temporary:
                config = fixture_config(Path(temporary), source)
                seed_fixture(config, source)
                image = config.image_dir / f"{source}100.png"
                original = image.read_bytes()
                image.unlink()
                partial = replace(complete_info(source), authors="", groups="")
                adapter = fixture_adapter(config, info=partial, missing=("作者", "团队"))
                adapter.thumbnail_url = "https://image.example.test/100.png"
                requested = []

                def thumbnail(url):
                    requested.append(url)
                    return BinaryPayload(original, "image/png")

                adapter.fetch_thumbnail = thumbnail
                progress = []
                result = run_collection(config, adapter=adapter, sleep_fn=lambda _: None,
                                        progress_callback=progress.append)
                self.assertTrue(result.success)
                self.assertEqual(result.terminal, 0)
                self.assertEqual(result.pending, 0)
                self.assertEqual(requested, [adapter.thumbnail_url])
                self.assertEqual(image.read_bytes(), original)
                self.assertEqual(progress[-1]["informationSiteMissing"], 1)
                self.assertEqual(progress[-1]["imagesCompleted"], 1)

    def test_both_sources_resume_unfinished_false_success_without_rediscovering_pages(self):
        for source in ("NH", "JM"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temporary:
                config = fixture_config(Path(temporary), source)
                seed_fixture(config, source)
                # A per-item false success inside an unfinished run must be
                # rechecked without rediscovering its completed listing page.
                image = config.image_dir / f"{source}100.png"
                before = image.read_bytes()
                result, adapter, progress = self.run_fixture(config)
                self.assertTrue(result.success)
                self.assertEqual(adapter.detail_calls, 1)
                self.assertEqual(adapter.discovery_calls, 0)
                self.assertEqual(read_rows(config.output_csv), [complete_info(source).as_csv_row()])
                self.assertEqual(image.read_bytes(), before)
                self.assertEqual(progress[-1]["informationComplete"], 1)
                self.assertEqual(progress[-1]["informationSiteMissing"], 0)

    def test_verified_site_absence_is_persisted_and_separately_counted(self):
        for source in ("NH", "JM"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temporary:
                config = fixture_config(Path(temporary), source, max_rounds=3)
                seed_fixture(config, source)
                partial = replace(complete_info(source), authors="", groups="")
                result, adapter, progress = self.run_fixture(config, info=partial, missing=("作者", "团队"))
                self.assertTrue(result.success)
                self.assertEqual(adapter.detail_calls, 1)
                self.assertEqual(progress[-1]["informationComplete"], 0)
                self.assertEqual(progress[-1]["informationSiteMissing"], 1)
                self.assertEqual(progress[-1]["informationCompleted"], 1)
                self.assertEqual(progress[-1]["pending"], 0)
                receipt_lines = receipts_path(config).read_text(encoding="utf-8").splitlines()
                self.assertTrue(receipt_lines)
                self.assertTrue(all(isinstance(json.loads(line), dict) for line in receipt_lines))
                for state_file in (config.state_file, Path(temporary) / "another-state.jsonl"):
                    next_config = replace(config, state_file=state_file)
                    resumed, resumed_adapter, resumed_progress = self.run_fixture(next_config, fail=True)
                    self.assertTrue(resumed.success)
                    # The previous run finished: rediscover the current page
                    # membership, but reuse the durable site-absence receipt.
                    self.assertEqual(resumed_adapter.discovery_calls, 1)
                    self.assertEqual(resumed_adapter.detail_calls, 0)
                    self.assertEqual(resumed_progress[-1]["informationSiteMissing"], 1)

    def test_timeout_never_creates_an_absence_receipt(self):
        for source in ("NH", "JM"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temporary:
                config = fixture_config(Path(temporary), source)
                seed_fixture(config, source)
                before = config.output_csv.read_bytes()
                result, adapter, _ = self.run_fixture(config, fail=True)
                self.assertFalse(result.success)
                self.assertEqual(result.pending, 1)
                self.assertEqual(adapter.detail_calls, 1)
                self.assertEqual(config.output_csv.read_bytes(), before)
                receipt = receipts_path(config)
                self.assertFalse(receipt.exists() and receipt.read_text(encoding="utf-8").strip())
                resumed, retry_adapter, _ = self.run_fixture(config)
                self.assertTrue(resumed.success)
                self.assertEqual(retry_adapter.detail_calls, 1)

    def test_original_csv_field_change_invalidates_previous_absence_check(self):
        for source in ("NH", "JM"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temporary:
                config = fixture_config(Path(temporary), source)
                seed_fixture(config, source)
                partial = replace(complete_info(source), authors="", groups="")
                self.assertTrue(self.run_fixture(config, info=partial, missing=("作者", "团队"))[0].success)
                row = read_rows(config.output_csv)[0]
                row["标签"] = ""
                write_rows(config.output_csv, [row])
                result, adapter, progress = self.run_fixture(config)
                self.assertTrue(result.success)
                self.assertEqual(adapter.detail_calls, 1)
                self.assertEqual(read_rows(config.output_csv)[0]["标签"], "Fixture tag")
                self.assertEqual(progress[-1]["informationSiteMissing"], 0)

    def test_translation_and_note_edits_do_not_invalidate_an_absence_check(self):
        for source in ("NH", "JM"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temporary:
                config = fixture_config(Path(temporary), source)
                seed_fixture(config, source)
                partial = replace(complete_info(source), authors="", groups="")
                self.assertTrue(self.run_fixture(config, info=partial, missing=("作者", "团队"))[0].success)
                row = read_rows(config.output_csv)[0]
                row.update({"标题译文": "Saved translation", "自定义备注": "My note"})
                write_rows(config.output_csv, [row])
                result, adapter, _ = self.run_fixture(config, fail=True)
                self.assertTrue(result.success)
                self.assertEqual(adapter.detail_calls, 0)
                self.assertEqual(read_rows(config.output_csv), [row])

    def test_repair_merges_missing_site_fields_without_losing_saved_author_or_extras(self):
        for source in ("NH", "JM"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temporary:
                config = fixture_config(Path(temporary), source)
                row = replace(complete_info(source), authors="Saved author", groups="", tags="").as_csv_row()
                row.update({"标题译文": "Saved translation", "自定义备注": "Saved notes"})
                seed_fixture(config, source, row)
                partial = replace(complete_info(source), authors="", groups="")
                result, adapter, _ = self.run_fixture(config, info=partial, missing=("作者", "团队"))
                self.assertTrue(result.success)
                self.assertEqual(adapter.detail_calls, 1)
                saved = read_rows(config.output_csv)[0]
                self.assertEqual(saved["作者"], "Saved author")
                self.assertEqual(saved["标签"], "Fixture tag")
                self.assertEqual(saved["标题译文"], "Saved translation")
                self.assertEqual(saved["自定义备注"], "Saved notes")
                resumed, resumed_adapter, _ = self.run_fixture(config, fail=True)
                self.assertTrue(resumed.success)
                self.assertEqual(resumed_adapter.detail_calls, 0)

    def test_valid_id_with_missing_title_or_link_is_not_dropped_before_repair(self):
        # Excel whitespace must not hide a row discovered inside this page range.
        for source in ("NH", "JM"):
            with self.subTest(source=source, whitespace=True), tempfile.TemporaryDirectory() as temporary:
                config = fixture_config(Path(temporary), source)
                row = replace(complete_info(source), authors="").as_csv_row()
                row["ID"] = " " + row["ID"] + " "
                seed_fixture(config, source, row)
                checkpoint = Checkpoint(config.state_file)
                checkpoint.append({"event": "run_start", "mode": config.mode,
                                   "identity": config.resolved().identity})
                checkpoint.close()
                result, adapter, _ = self.run_fixture(config)
                self.assertTrue(result.success)
                self.assertEqual(adapter.detail_calls, 1)
                self.assertEqual(adapter.discovery_calls, 1)
                self.assertEqual(read_rows(config.output_csv)[0]["ID"], complete_info(source).id)
        for source in ("NH", "JM"):
            for absent in (("标题",), ("链接",), ("标题", "链接")):
                with self.subTest(source=source, absent=absent), tempfile.TemporaryDirectory() as temporary:
                    config = fixture_config(Path(temporary), source)
                    row = complete_info(source).as_csv_row()
                    row.update({name: "" for name in absent})
                    row["自定义备注"] = "must survive recovery"
                    seed_fixture(config, source, row)
                    result, adapter, _ = self.run_fixture(config)
                    self.assertTrue(result.success)
                    self.assertEqual(adapter.detail_calls, 1)
                    self.assertEqual(adapter.detail_urls, [complete_info(source).link])
                    saved = read_rows(config.output_csv)
                    self.assertEqual(len(saved), 1)
                    for field in CSV_HEADERS:
                        self.assertTrue(saved[0][field], field)
                    self.assertEqual(saved[0]["自定义备注"], row["自定义备注"])

    def test_unverified_partial_adapter_result_remains_pending(self):
        for source in ("NH", "JM"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temporary:
                config = fixture_config(Path(temporary), source)
                seed_fixture(config, source)
                partial = replace(complete_info(source), authors="", groups="")
                result, adapter, progress = self.run_fixture(config, info=partial)
                self.assertFalse(result.success)
                self.assertEqual(adapter.detail_calls, 1)
                self.assertEqual(result.pending, 1)
                self.assertEqual(progress[-1]["informationSiteMissing"], 0)
                receipt = receipts_path(config)
                self.assertFalse(receipt.exists() and receipt.read_text(encoding="utf-8").strip())

    def test_absence_decision_survives_exit_before_task_checkpoint_or_csv_close(self):
        for source in ("NH", "JM"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = fixture_config(root, source)
                seed_fixture(config, source)
                script = (
                    "import os, sys\n"
                    "from pathlib import Path\n"
                    "from dataclasses import replace\n"
                    "sys.path.insert(0, str(Path.cwd() / 'tests'))\n"
                    "from test_metadata_recheck import *\n"
                    "source = sys.argv[2]\n"
                    "config = fixture_config(Path(sys.argv[1]), source)\n"
                    "partial = replace(complete_info(source), authors='', groups='', title='Updated before crash')\n"
                    "adapter = fixture_adapter(config, info=partial, missing=('作者', '团队'))\n"
                    "runner = OnlineCollectionRunner(config.resolved(), adapter, sleep_fn=lambda _: None)\n"
                    "original_task = runner.checkpoint.task\n"
                    "def crash_before_task_checkpoint(state):\n"
                    "    if state.info_ok:\n"
                    "        os._exit(0)\n"
                    "    original_task(state)\n"
                    "runner.checkpoint.task = crash_before_task_checkpoint\n"
                    "runner.run()\n"
                    "raise RuntimeError('crash injection did not run')\n"
                )
                child = subprocess.run([sys.executable, "-c", script, temporary, source],
                                       cwd=Path(__file__).resolve().parents[1],
                                       capture_output=True, text=True, timeout=20)
                self.assertEqual(child.returncode, 0, child.stderr)
                self.assertTrue(receipts_path(config).is_file())
                result, adapter, progress = self.run_fixture(config, fail=True)
                self.assertTrue(result.success)
                self.assertEqual(adapter.detail_calls, 0)
                self.assertEqual(read_rows(config.output_csv)[0]["标题"], "Updated before crash")
                self.assertEqual(progress[-1]["informationSiteMissing"], 1)

    def test_failed_receipt_write_stops_and_does_not_claim_persistent_completion(self):
        for source in ("NH", "JM"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temporary:
                config = fixture_config(Path(temporary), source)
                seed_fixture(config, source)
                partial = replace(complete_info(source), authors="", groups="", title="Updated title")
                adapter = fixture_adapter(config, info=partial, missing=("作者", "团队"))
                runner = OnlineCollectionRunner(config.resolved(), adapter, sleep_fn=lambda _: None)
                try:
                    with patch.object(runner.metadata_reviews, "record", side_effect=OSError("receipt disk full")):
                        with self.assertRaisesRegex(OSError, "receipt disk full"):
                            runner.run()
                    self.assertFalse(runner.states[f"{source}100"].info_ok)
                finally:
                    runner.csv_store.close()
                    runner.checkpoint.close()
                    runner.error_writer.close()
                self.assertEqual(adapter.detail_calls, 1)
                resumed, second_adapter, _ = self.run_fixture(config, info=partial, missing=("作者", "团队"))
                self.assertTrue(resumed.success)
                self.assertEqual(second_adapter.detail_calls, 1)
                self.assertEqual(read_rows(config.output_csv)[0]["标题"], "Updated title")

    def test_torn_receipt_tail_is_not_trusted_and_does_not_swallow_next_receipt(self):
        for source in ("NH", "JM"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temporary:
                config = fixture_config(Path(temporary), source)
                seed_fixture(config, source)
                receipts_path(config).write_bytes(b'{"version":1,"id":"')
                partial = replace(complete_info(source), authors="", groups="")
                result, adapter, _ = self.run_fixture(config, info=partial, missing=("作者", "团队"))
                self.assertTrue(result.success)
                self.assertEqual(adapter.detail_calls, 1)
                resumed, second_adapter, _ = self.run_fixture(config, fail=True)
                self.assertTrue(resumed.success)
                self.assertEqual(second_adapter.detail_calls, 0)


if __name__ == "__main__":
    unittest.main()
