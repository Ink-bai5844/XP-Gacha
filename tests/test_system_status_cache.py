from __future__ import annotations

import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import server.modules.system as system_module
from server.modules.system import SystemModule, _count_files


class SystemStatusCacheTest(unittest.TestCase):
    def test_default_requests_reuse_snapshot_and_refresh_replaces_it(self) -> None:
        module = SystemModule()
        first = {"csv": 1}
        second = {"csv": 2}

        with patch.object(module, "_build_counts", side_effect=[first, second]) as build:
            initial = module._counts()
            initial["csv"] = 99

            self.assertEqual(module._counts()["csv"], 1)
            self.assertEqual(build.call_count, 1)
            self.assertEqual(module._counts(refresh=True)["csv"], 2)
            self.assertEqual(module._counts()["csv"], 2)
            self.assertEqual(build.call_count, 2)

    def test_failed_refresh_keeps_last_successful_counts(self) -> None:
        module = SystemModule()
        with patch.object(
            module,
            "_build_counts",
            side_effect=[{"csv": 5}, RuntimeError("scan failed")],
        ):
            self.assertEqual(module._counts()["csv"], 5)
            with self.assertRaisesRegex(RuntimeError, "scan failed"):
                module._counts(refresh=True)
            self.assertEqual(module._counts()["csv"], 5)

    def test_concurrent_first_requests_build_only_one_snapshot(self) -> None:
        module = SystemModule()
        build_calls = 0
        calls_lock = threading.Lock()

        def build_counts() -> dict:
            nonlocal build_calls
            with calls_lock:
                build_calls += 1
            time.sleep(0.02)
            return {"csv": 7}

        with (
            patch.object(module, "_build_counts", side_effect=build_counts),
            ThreadPoolExecutor(max_workers=8) as pool,
        ):
            results = list(pool.map(lambda _index: module._counts(), range(16)))

        self.assertEqual(build_calls, 1)
        self.assertTrue(all(result["csv"] == 7 for result in results))

    def test_lightweight_status_fields_are_not_cached(self) -> None:
        module = SystemModule()
        first_database = {"available": False, "row_count": 0}
        second_database = {"available": True, "row_count": 12}
        capabilities = {
            "semantic": {"dependencies": {"model": {"ready": False}}},
            "cover": {"dependencies": {"model": {"ready": True}}},
        }

        with (
            patch.object(module, "_build_counts", return_value={"csv": 4}) as build_counts,
            patch.object(system_module, "database_status", side_effect=[first_database, second_database]) as database_status,
            patch.object(system_module, "_search_capabilities", return_value=capabilities) as search_capabilities,
            patch.object(system_module, "_file_status", return_value={}) as file_status,
        ):
            first = module.status()
            second = module.status()

        self.assertEqual(first["database"], first_database)
        self.assertEqual(second["database"], second_database)
        self.assertEqual(first["counts"], second["counts"])
        build_counts.assert_called_once_with()
        self.assertEqual(database_status.call_count, 2)
        self.assertEqual(search_capabilities.call_count, 2)
        self.assertEqual(file_status.call_count, 8)

    def test_file_count_is_non_recursive_and_respects_pattern(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xp-gacha-system-count-") as temp_dir:
            root = Path(temp_dir)
            (root / "one.txt").write_text("1", encoding="utf-8")
            (root / "two.csv").write_text("2", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "three.txt").write_text("3", encoding="utf-8")

            self.assertEqual(_count_files(str(root)), 2)
            self.assertEqual(_count_files(str(root), "*.txt"), 1)
            self.assertEqual(_count_files(str(root / "missing")), 0)


if __name__ == "__main__":
    unittest.main()
