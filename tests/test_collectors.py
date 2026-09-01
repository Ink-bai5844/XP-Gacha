from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image

from data_get.collector import (
    BinaryPayload,
    Checkpoint,
    CollectionConfig,
    CollectionItem,
    CollectionRequestError,
    CsvStore,
    GalleryInfo,
    ParsedGallery,
    build_parser,
    config_from_args,
    parse_local_links,
    run_collection,
    _retry_delay,
)


def valid_png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (25, 80, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


class FlakyAdapter:
    def __init__(self, *, empty: bool = False, invalid_image: bool = False):
        self.empty = empty
        self.invalid_image = invalid_image
        self.discover_calls = 0
        self.detail_calls: Counter[str] = Counter()
        self.thumbnail_calls: Counter[str] = Counter()

    def discover_page(self, page: int) -> list[CollectionItem]:
        self.discover_calls += 1
        if self.empty:
            return []
        return [
            CollectionItem("NH100", "https://example.test/g/100/", "https://img.test/100.png", page),
            CollectionItem("NH200", "https://example.test/g/200/", "https://img.test/200.png", page),
        ]

    def fetch_detail(self, item: CollectionItem) -> ParsedGallery:
        self.detail_calls[item.id] += 1
        if item.id == "NH100" and self.detail_calls[item.id] == 1:
            raise TimeoutError("detail timeout")
        return ParsedGallery(
            GalleryInfo(item.id, item.detail_url, f"title-{item.id}"),
            f"https://img.test/{item.id.removeprefix('NH')}.png",
        )

    def fetch_thumbnail(self, url: str) -> BinaryPayload:
        gallery_id = "NH100" if "100" in url else "NH200"
        self.thumbnail_calls[gallery_id] += 1
        if self.invalid_image:
            return BinaryPayload(b"<html>captcha</html>", "text/html")
        if gallery_id == "NH200" and self.thumbnail_calls[gallery_id] == 1:
            raise ConnectionError("image connection reset")
        return BinaryPayload(valid_png(), "image/png")


class OneItemAdapter:
    def __init__(self, *, fail_detail: bool = False, forbid_discovery: bool = False):
        self.fail_detail = fail_detail
        self.forbid_discovery = forbid_discovery
        self.discover_calls = 0
        self.detail_calls = 0
        self.thumbnail_calls = 0

    def discover_page(self, page: int) -> list[CollectionItem]:
        self.discover_calls += 1
        if self.forbid_discovery:
            raise AssertionError("completed list page must not be rediscovered while resuming")
        return [CollectionItem("NH300", "https://example.test/g/300/", "https://img.test/300.png", page)]

    def fetch_detail(self, item: CollectionItem) -> ParsedGallery:
        self.detail_calls += 1
        if self.fail_detail:
            raise TimeoutError("temporary detail timeout")
        return ParsedGallery(GalleryInfo(item.id, item.detail_url, "title-300"), item.thumbnail_url)

    def fetch_thumbnail(self, url: str) -> BinaryPayload:
        self.thumbnail_calls += 1
        return BinaryPayload(valid_png(), "image/png")


class FullImageAdapter:
    def fetch_full_image_url(self, page_url: str) -> str | None:
        if page_url.endswith("/2/"):
            return None
        gallery_id = "111" if "/111/" in page_url else "222"
        return f"https://img.test/{gallery_id}/1.png"

    def fetch_thumbnail(self, url: str) -> BinaryPayload:
        return BinaryPayload(valid_png(), "image/png")


class InterruptingAdapter:
    def discover_page(self, page: int) -> list[CollectionItem]:
        raise KeyboardInterrupt()


class InterruptingFullImageAdapter:
    def fetch_full_image_url(self, page_url: str) -> str | None:
        raise KeyboardInterrupt()

    def fetch_thumbnail(self, url: str) -> BinaryPayload:
        raise AssertionError("thumbnail stage must not run")


class MissingPageAdapter:
    def __init__(self, *, first_page_missing: bool):
        self.first_page_missing = first_page_missing

    def discover_page(self, page: int) -> list[CollectionItem]:
        if (page == 1 and self.first_page_missing) or page > 1:
            raise CollectionRequestError("missing list page", retryable=False, status_code=404)
        return [CollectionItem("NH404", "https://example.test/g/404/", page=page)]

    def fetch_detail(self, item: CollectionItem) -> ParsedGallery:
        return ParsedGallery(
            GalleryInfo(item.id, item.detail_url, "existing page"),
            "https://img.test/404.png",
        )

    def fetch_thumbnail(self, url: str) -> BinaryPayload:
        return BinaryPayload(valid_png(), "image/png")


class RefreshingOnlineAdapter:
    def __init__(self):
        self.detail_calls = 0
        self.thumbnail_urls: list[str] = []

    def discover_page(self, page: int) -> list[CollectionItem]:
        return [CollectionItem("NH500", "https://example.test/g/500/", page=page)]

    def fetch_detail(self, item: CollectionItem) -> ParsedGallery:
        self.detail_calls += 1
        suffix = "old.png" if self.detail_calls == 1 else "fresh.png"
        return ParsedGallery(
            GalleryInfo(item.id, item.detail_url, "rotating thumbnail"),
            f"https://img.test/{suffix}",
        )

    def fetch_thumbnail(self, url: str) -> BinaryPayload:
        self.thumbnail_urls.append(url)
        if url.endswith("old.png"):
            raise CollectionRequestError("expired CDN URL", retryable=False, status_code=404)
        return BinaryPayload(valid_png(), "image/png")


class ContinuousLocalAdapter:
    def __init__(self, *, fail_page_two_once: bool = False):
        self.fail_page_two_once = fail_page_two_once
        self.page_two_calls = 0
        self.page_calls: list[int] = []
        self.thumbnail_urls: list[str] = []

    def fetch_full_image_url(self, page_url: str) -> str | None:
        page = int(page_url.rstrip("/").rsplit("/", 1)[1])
        self.page_calls.append(page)
        if page == 2:
            self.page_two_calls += 1
            if self.fail_page_two_once and self.page_two_calls == 1:
                raise TimeoutError("page 2 timeout")
        if page == 3:
            return None
        return f"https://img.test/{page}.png"

    def fetch_thumbnail(self, url: str) -> BinaryPayload:
        self.thumbnail_urls.append(url)
        return BinaryPayload(valid_png(), "image/png")


class MissingLocalGalleryAdapter:
    def fetch_full_image_url(self, page_url: str) -> str | None:
        return None

    def fetch_thumbnail(self, url: str) -> BinaryPayload:
        raise AssertionError("missing gallery must not download an image")


class RefreshingLocalAdapter:
    def __init__(self):
        self.page_one_calls = 0
        self.thumbnail_urls: list[str] = []

    def fetch_full_image_url(self, page_url: str) -> str | None:
        page = int(page_url.rstrip("/").rsplit("/", 1)[1])
        if page == 2:
            return None
        self.page_one_calls += 1
        suffix = "old.png" if self.page_one_calls == 1 else "fresh.png"
        return f"https://img.test/{suffix}"

    def fetch_thumbnail(self, url: str) -> BinaryPayload:
        self.thumbnail_urls.append(url)
        if url.endswith("old.png"):
            raise CollectionRequestError("expired full-image URL", retryable=False, status_code=404)
        return BinaryPayload(valid_png(), "image/png")


class CollectorTests(unittest.TestCase):
    def make_config(self, root: Path, **overrides) -> CollectionConfig:
        values = {
            "mode": "nh-online",
            "base_url": "https://example.test",
            "start_url": "https://example.test/list",
            "max_pages": 1,
            "output_csv": root / "origin" / "NH_info_test.csv",
            "image_dir": root / "images",
            "workers": 1,
            "request_attempts": 1,
            "max_rounds": 3,
            "retry_backoff": 0,
            "state_file": root / "state.jsonl",
            "error_log": root / "errors.jsonl",
        }
        values.update(overrides)
        return CollectionConfig(**values)

    def make_local_images_config(self, root: Path, **overrides) -> CollectionConfig:
        input_file = root / "links.html"
        if not input_file.exists():
            input_file.write_text('<a href="https://nhentai.net/g/111/">title</a>', encoding="utf-8")
        values = {
            "mode": "nh-local-images",
            "input_file": input_file,
            "output_dir": root / "pages",
            "max_pages": 5,
            "workers": 1,
            "request_attempts": 1,
            "max_rounds": 3,
            "retry_backoff": 0,
            "state_file": root / "pages.state.jsonl",
            "error_log": root / "pages.errors.jsonl",
        }
        values.update(overrides)
        return CollectionConfig(**values)

    def test_only_failed_stages_are_retried_until_info_and_thumbnail_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = FlakyAdapter()
            summary = run_collection(self.make_config(root), adapter=adapter, sleep_fn=lambda _seconds: None)

            self.assertTrue(summary.success)
            self.assertEqual(summary.rounds, 2)
            self.assertEqual(adapter.discover_calls, 1)
            self.assertEqual(adapter.detail_calls, Counter({"NH100": 2, "NH200": 2}))
            self.assertEqual(adapter.thumbnail_calls, Counter({"NH200": 2, "NH100": 1}))

            with (root / "origin" / "NH_info_test.csv").open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["ID"] for row in rows}, {"NH100", "NH200"})
            self.assertEqual(len(rows), 2)
            self.assertEqual({path.stem for path in (root / "images").iterdir()}, {"NH100", "NH200"})

            errors = [json.loads(line) for line in (root / "errors.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual({record["stage"] for record in errors}, {"detail", "thumbnail"})

    def test_existing_csv_row_with_missing_image_does_not_append_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            csv_path = root / "origin" / "NH_info_test.csv"
            csv_path.parent.mkdir(parents=True)
            with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["ID", "链接", "标题", "标签", "作者", "团队", "语言", "页数", "上传日期"])
                writer.writeheader()
                writer.writerow({"ID": "NH300", "链接": "https://example.test/g/300/", "标题": "existing"})

            adapter = OneItemAdapter()
            summary = run_collection(
                self.make_config(root, output_csv=csv_path, max_rounds=1),
                adapter=adapter,
                sleep_fn=lambda _seconds: None,
            )
            self.assertTrue(summary.success)
            self.assertEqual(adapter.detail_calls, 0)
            self.assertEqual(adapter.thumbnail_calls, 1)
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["标题"], "existing")

    def test_checkpoint_resume_retries_task_without_rescanning_completed_page(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.make_config(root, max_rounds=1)
            first = run_collection(config, adapter=OneItemAdapter(fail_detail=True), sleep_fn=lambda _seconds: None)
            self.assertFalse(first.success)
            self.assertEqual(first.pending, 1)

            resumed_adapter = OneItemAdapter(forbid_discovery=True)
            second = run_collection(config, adapter=resumed_adapter, sleep_fn=lambda _seconds: None)
            self.assertTrue(second.success)
            self.assertEqual(resumed_adapter.discover_calls, 0)
            self.assertEqual(resumed_adapter.detail_calls, 1)
            self.assertEqual(resumed_adapter.thumbnail_calls, 0)

    def test_html_or_invalid_image_is_not_marked_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = FlakyAdapter(invalid_image=True)
            summary = run_collection(
                self.make_config(root, max_rounds=1),
                adapter=adapter,
                sleep_fn=lambda _seconds: None,
            )
            self.assertFalse(summary.success)
            self.assertEqual(summary.pending, 2)
            self.assertEqual(list((root / "images").iterdir()), [])
            self.assertFalse(any(root.rglob("*.part")))
            errors = [json.loads(line) for line in (root / "errors.jsonl").read_text(encoding="utf-8").splitlines()]
            invalid_image_errors = [record for record in errors if "Content-Type" in record["message"]]
            self.assertEqual({record["id"] for record in invalid_image_errors}, {"NH100", "NH200"})
            self.assertTrue(all(record["stage"] == "thumbnail" for record in invalid_image_errors))

    def test_historical_html_with_image_suffix_is_not_treated_as_a_thumbnail(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_dir = root / "images"
            image_dir.mkdir()
            (image_dir / "NH300.jpg").write_bytes(b"<html>old captcha</html>")
            (image_dir / "NH300.avif").write_bytes(b"\x00\x00\x00\x18ftypavif")
            adapter = OneItemAdapter()

            summary = run_collection(
                self.make_config(root, image_dir=image_dir, max_rounds=1),
                adapter=adapter,
                sleep_fn=lambda _seconds: None,
            )

            self.assertTrue(summary.success)
            self.assertEqual(adapter.thumbnail_calls, 1)
            replacement = image_dir / "NH300.png"
            self.assertTrue(replacement.is_file())
            with Image.open(replacement) as image:
                image.verify()

    def test_historical_html_page_is_revalidated_and_downloaded(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "pages" / "NH111_title"
            folder.mkdir(parents=True)
            (folder / "1.jpg").write_bytes(b"<html>old captcha</html>")
            adapter = ContinuousLocalAdapter()

            summary = run_collection(
                self.make_local_images_config(root, max_rounds=1),
                adapter=adapter,
                sleep_fn=lambda _seconds: None,
            )

            self.assertTrue(summary.success)
            self.assertIn("https://img.test/1.png", adapter.thumbnail_urls)
            replacement = folder / "1.png"
            self.assertTrue(replacement.is_file())
            with Image.open(replacement) as image:
                image.verify()

    def test_empty_200_list_page_is_retryable_not_false_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = run_collection(
                self.make_config(root, max_rounds=1),
                adapter=FlakyAdapter(empty=True),
                sleep_fn=lambda _seconds: None,
            )
            self.assertFalse(summary.success)
            self.assertEqual(summary.failed_pages, 1)
            error = json.loads((root / "errors.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(error["stage"], "list")
            self.assertTrue(error["retryable"])

    def test_online_first_page_404_is_failure_but_later_404_is_end(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first_missing = run_collection(
                self.make_config(root, max_pages=1, max_rounds=1),
                adapter=MissingPageAdapter(first_page_missing=True),
                sleep_fn=lambda _seconds: None,
            )
            self.assertFalse(first_missing.success)
            self.assertEqual(first_missing.discovered, 0)
            self.assertEqual(first_missing.terminal, 1)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            later_missing = run_collection(
                self.make_config(root, max_pages=2, max_rounds=1),
                adapter=MissingPageAdapter(first_page_missing=False),
                sleep_fn=lambda _seconds: None,
            )
            self.assertTrue(later_missing.success)
            self.assertEqual(later_missing.discovered, 1)
            self.assertEqual(later_missing.completed, 1)

    def test_online_expired_thumbnail_url_is_reparsed_next_round(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = RefreshingOnlineAdapter()
            summary = run_collection(
                self.make_config(root, max_rounds=2),
                adapter=adapter,
                sleep_fn=lambda _seconds: None,
            )

            self.assertTrue(summary.success)
            self.assertEqual(summary.rounds, 2)
            self.assertEqual(adapter.detail_calls, 2)
            self.assertEqual(
                adapter.thumbnail_urls,
                ["https://img.test/old.png", "https://img.test/fresh.png"],
            )

    def test_retry_delay_is_capped_without_large_integer_overflow(self) -> None:
        self.assertEqual(_retry_delay(0, 100_000), 0)
        self.assertEqual(_retry_delay(2, 100_000), 300)

    def test_file_roles_and_directory_roles_must_not_collide(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            csv_path = root / "same.csv"
            with self.assertRaisesRegex(ValueError, "不能指向同一个文件"):
                self.make_config(root, output_csv=csv_path, state_file=csv_path).resolved()
            with self.assertRaisesRegex(ValueError, "不能指向同一个文件"):
                self.make_config(root, state_file=root / "same.jsonl", error_log=root / "same.jsonl").resolved()
            with self.assertRaisesRegex(ValueError, "不能指向同一个文件"):
                self.make_config(root, input_file=root / "input.txt", state_file=root / "input.txt").resolved()
            with self.assertRaisesRegex(ValueError, "不能指向同一路径"):
                self.make_config(root, state_file=root / "images", image_dir=root / "images").resolved()

    def test_empty_local_input_is_not_reported_as_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_file = root / "empty.txt"
            input_file.write_text("not a gallery link", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "未解析到任何 NH 图库链接"):
                run_collection(
                    self.make_config(
                        root,
                        mode="nh-local-info",
                        input_file=input_file,
                        max_rounds=1,
                    ),
                    adapter=OneItemAdapter(),
                    sleep_fn=lambda _seconds: None,
                )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_file = root / "links.html"
            input_file.write_text("<p>no gallery here</p>", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "未解析到任何 NH 图库链接"):
                run_collection(
                    self.make_local_images_config(root, input_file=input_file),
                    adapter=FullImageAdapter(),
                    sleep_fn=lambda _seconds: None,
                )

    def test_cli_defaults_and_local_link_formats(self) -> None:
        args = build_parser().parse_args(["nh-online"])
        resolved = config_from_args(args).resolved()
        self.assertEqual(resolved.output_csv.parent.name, "gallery_info_origin")
        self.assertEqual(resolved.output_csv.name, "NH_info_chinese.csv")
        self.assertEqual(resolved.max_rounds, 0)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "links.txt"
            path.write_text(
                '<a href="https://nhentai.net/g/123/">one</a>\nHTTPS://NHENTAI.NET/g/456/\nhttps://nhentai.net/g/123/',
                encoding="utf-8",
            )
            links = parse_local_links(path)
            self.assertEqual([url.lower() for url, _label in links], ["https://nhentai.net/g/123/", "https://nhentai.net/g/456/"])

        first_state = CollectionConfig(
            mode="nh-local-images", input_file="data/local_data/NH_1.txt", output_dir="output"
        ).resolved().state_file
        second_state = CollectionConfig(
            mode="nh-local-images", input_file="data/local_data/NH_2.txt", output_dir="output"
        ).resolved().state_file
        self.assertNotEqual(first_state, second_state)

    def test_local_images_same_title_uses_one_stable_id_prefix_per_gallery(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_file = root / "links.html"
            input_file.write_text(
                '<a href="https://nhentai.net/g/111/">same title</a>\n'
                '<a href="https://nhentai.net/g/222/">same title</a>',
                encoding="utf-8",
            )
            summary = run_collection(
                CollectionConfig(
                    mode="nh-local-images",
                    input_file=input_file,
                    output_dir=root / "pages",
                    max_pages=5,
                    workers=1,
                    request_attempts=1,
                    max_rounds=1,
                    retry_backoff=0,
                    state_file=root / "pages.state.jsonl",
                    error_log=root / "pages.errors.jsonl",
                ),
                adapter=FullImageAdapter(),
                sleep_fn=lambda _seconds: None,
            )
            self.assertTrue(summary.success)
            folders = {path.name for path in (root / "pages").iterdir() if path.is_dir()}
            self.assertEqual(folders, {"NH111_same title", "NH222_same title"})
            self.assertTrue((root / "pages" / "NH111_same title" / "1.png").is_file())
            self.assertTrue((root / "pages" / "NH222_same title" / "1.png").is_file())

    def test_local_discovery_stops_at_retryable_page_and_resumes_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = ContinuousLocalAdapter(fail_page_two_once=True)
            first = run_collection(
                self.make_local_images_config(root, max_rounds=1),
                adapter=adapter,
                sleep_fn=lambda _seconds: None,
            )
            self.assertFalse(first.success)
            self.assertEqual(first.pending, 1)
            self.assertEqual(adapter.page_calls, [1, 2])
            self.assertTrue((root / "pages" / "NH111_title" / "1.png").is_file())

            resumed = run_collection(
                self.make_local_images_config(root, max_rounds=3),
                adapter=adapter,
                sleep_fn=lambda _seconds: None,
            )
            self.assertTrue(resumed.success)
            self.assertEqual(adapter.page_calls, [1, 2, 2, 3])
            self.assertEqual(resumed.limit_reached, 0)
            self.assertTrue((root / "pages" / "NH111_title" / "2.png").is_file())
            replayed = Checkpoint(root / "pages.state.jsonl").replay(
                self.make_local_images_config(root).resolved().identity
            )
            self.assertEqual(set(replayed.tasks), {"NH111:1", "NH111:2"})
            self.assertEqual(replayed.completed_galleries, {"NH111"})

    def test_local_page_one_missing_is_terminal_not_zero_page_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = run_collection(
                self.make_local_images_config(root, max_rounds=1),
                adapter=MissingLocalGalleryAdapter(),
                sleep_fn=lambda _seconds: None,
            )
            self.assertFalse(summary.success)
            self.assertEqual(summary.discovered, 1)
            self.assertEqual(summary.terminal, 1)
            error = json.loads((root / "pages.errors.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(error["statusCode"], 404)
            self.assertFalse(error["retryable"])

    def test_local_expired_image_url_reparses_page_next_round(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = RefreshingLocalAdapter()
            summary = run_collection(
                self.make_local_images_config(root, max_rounds=2),
                adapter=adapter,
                sleep_fn=lambda _seconds: None,
            )

            self.assertTrue(summary.success)
            self.assertEqual(adapter.page_one_calls, 2)
            self.assertEqual(
                adapter.thumbnail_urls,
                ["https://img.test/old.png", "https://img.test/fresh.png"],
            )
            error = json.loads((root / "pages.errors.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(error["stage"], "image")
            self.assertEqual(error["statusCode"], 404)

    def test_checkpoint_replays_later_matching_identity_segment(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.jsonl"
            events = [
                {"event": "run_start", "identity": "foreign"},
                {
                    "event": "task_state",
                    "item": {
                        "id": "NH1",
                        "detail_url": "https://example.test/g/1/",
                        "thumbnail_url": "",
                        "page": 1,
                        "label": "",
                    },
                    "info_ok": True,
                    "thumb_ok": True,
                },
                {"event": "run_complete"},
                {"event": "run_start", "identity": "wanted"},
                {
                    "event": "task_state",
                    "item": {
                        "id": "NH2",
                        "detail_url": "https://example.test/g/2/",
                        "thumbnail_url": "",
                        "page": 2,
                        "label": "",
                    },
                    "info_ok": True,
                    "thumb_ok": False,
                },
                {"event": "run_complete"},
            ]
            path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

            replayed = Checkpoint(path).replay("wanted")
            self.assertEqual(set(replayed.tasks), {"NH2"})
            self.assertTrue(replayed.run_completed)

    def test_checkpoint_does_not_restore_matching_state_before_foreign_segment(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.jsonl"
            events = [
                {"event": "run_start", "identity": "wanted"},
                {
                    "event": "task_state",
                    "item": {
                        "id": "NH2",
                        "detail_url": "https://example.test/g/2/",
                        "thumbnail_url": "",
                        "page": 2,
                        "label": "",
                    },
                    "info_ok": True,
                    "thumb_ok": True,
                },
                {"event": "run_complete"},
                {"event": "run_start", "identity": "foreign"},
                {
                    "event": "task_state",
                    "item": {
                        "id": "NH9",
                        "detail_url": "https://example.test/g/9/",
                        "thumbnail_url": "",
                        "page": 9,
                        "label": "",
                    },
                    "info_ok": True,
                    "thumb_ok": True,
                },
            ]
            path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

            replayed = Checkpoint(path).replay("wanted")
            self.assertEqual(replayed.tasks, {})
            self.assertFalse(replayed.run_completed)

    def test_keyboard_interrupt_is_not_recorded_as_a_retryable_network_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            online = run_collection(
                self.make_config(root),
                adapter=InterruptingAdapter(),
                sleep_fn=lambda _seconds: None,
            )
            self.assertTrue(online.interrupted)
            self.assertEqual(online.exit_code, 130)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_file = root / "links.html"
            input_file.write_text('<a href="https://nhentai.net/g/111/">title</a>', encoding="utf-8")
            local = run_collection(
                CollectionConfig(
                    mode="nh-local-images",
                    input_file=input_file,
                    output_dir=root / "pages",
                    max_pages=2,
                    workers=1,
                    request_attempts=1,
                    state_file=root / "state.jsonl",
                    error_log=root / "errors.jsonl",
                ),
                adapter=InterruptingFullImageAdapter(),
                sleep_fn=lambda _seconds: None,
            )
            self.assertTrue(local.interrupted)
            self.assertEqual(local.exit_code, 130)

    def test_csv_store_discards_and_atomically_repairs_a_truncated_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "origin.csv"
            headers = ["ID", "链接", "标题", "标签", "作者", "团队", "语言", "页数", "上传日期"]
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=headers)
                writer.writeheader()
                writer.writerow({"ID": "NH1", "链接": "https://example.test/g/1/", "标题": "valid"})
            with path.open("a", encoding="utf-8", newline="") as handle:
                handle.write("NH2,https://example.test/g/2/,")

            store = CsvStore(path)
            self.assertTrue(store.has("NH1"))
            self.assertFalse(store.has("NH2"))
            store.upsert(GalleryInfo("NH1", "https://example.test/g/1/", "valid"))
            store.commit()

            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["ID"] for row in rows], ["NH1"])
            self.assertFalse(any(path.parent.glob(".*.tmp")))

    def test_csv_store_rejects_unrelated_nonempty_csv_without_modifying_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "unrelated.csv"
            original = "name,value\r\nalpha,1\r\n".encode("utf-8")
            path.write_bytes(original)

            with self.assertRaisesRegex(ValueError, "缺少必需表头"):
                CsvStore(path)

            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(any(path.parent.glob(".*.tmp")))


if __name__ == "__main__":
    unittest.main()
