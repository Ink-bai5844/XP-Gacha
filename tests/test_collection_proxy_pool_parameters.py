from __future__ import annotations

import io
import re
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import server.job_tasks as job_tasks


class CollectionProxyPoolParameterTests(unittest.TestCase):
    @contextmanager
    def capture_config(self, root: Path):
        captured: dict[str, object] = {}
        collector = ModuleType("data_get.collector")

        class FakeConfig:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        collector.CollectionConfig = FakeConfig
        collector.run_collection = Mock(return_value={"completed": True})
        with (
            patch.dict(sys.modules, {"data_get.collector": collector}),
            patch.object(job_tasks, "ROOT", root),
            patch.object(job_tasks, "start_collection_cancel_monitor", return_value=(None, None)),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            yield captured, collector.run_collection

    def test_all_modes_accept_camel_and_snake_case_pool_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            with self.capture_config(root) as (captured, run_collection):
                for script_id, mode in job_tasks.COLLECTION_MODES.items():
                    for alias in ("proxyPoolFile", "proxy_pool_file"):
                        with self.subTest(script_id=script_id, alias=alias):
                            self.assertTrue(job_tasks.run_collection_task(
                                script_id,
                                {"confirm": True, alias: " config/proxy-pool.json "},
                            ))
                            self.assertEqual(captured["mode"], mode)
                            self.assertEqual(captured["proxy_pool_file"], root / "config" / "proxy-pool.json")
                            self.assertIsNone(captured["proxy"])
                self.assertEqual(run_collection.call_count, len(job_tasks.COLLECTION_MODES) * 2)

    def test_missing_empty_and_whitespace_pool_paths_preserve_single_proxy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            with self.capture_config(root) as (captured, _run_collection):
                cases = [{}, *({alias: value} for alias in ("proxyPoolFile", "proxy_pool_file") for value in (None, "", "  "))]
                for parameters in cases:
                    with self.subTest(parameters=parameters):
                        job_tasks.run_collection_task(
                            "collection-nh-online",
                            {"confirm": True, "proxy": " http://127.0.0.1:7890 ", **parameters},
                        )
                        self.assertIsNone(captured["proxy_pool_file"])
                        self.assertEqual(captured["proxy"], "http://127.0.0.1:7890")

    def test_absolute_project_pool_path_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            with self.capture_config(root) as (captured, _run_collection):
                path = root / "config" / "proxy-pool.json"
                job_tasks.run_collection_task("collection-jm-online", {
                    "confirm": True, "proxyPoolFile": str(path), "proxy": "   ",
                })
                self.assertEqual(captured["proxy_pool_file"], path)
                self.assertIsNone(captured["proxy"])

    def test_pool_path_cannot_escape_project_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            with self.capture_config(root) as (_captured, run_collection):
                for alias in ("proxyPoolFile", "proxy_pool_file"):
                    for path in ("../proxy-pool.json", str(root.parent / "outside-proxy-pool.json")):
                        with self.subTest(alias=alias, path=path):
                            with self.assertRaisesRegex(ValueError, "项目目录内"):
                                job_tasks.run_collection_task("collection-nh-online", {
                                    "confirm": True, alias: path,
                                })
                run_collection.assert_not_called()

    def test_explicit_single_proxy_and_pool_are_rejected_before_collection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            with self.capture_config(root) as (_captured, run_collection):
                for script_id in job_tasks.COLLECTION_MODES:
                    for alias in ("proxyPoolFile", "proxy_pool_file"):
                        with self.subTest(script_id=script_id, alias=alias):
                            with self.assertRaisesRegex(ValueError, "不能同时填写"):
                                job_tasks.run_collection_task(script_id, {
                                    "confirm": True,
                                    "proxy": "http://127.0.0.1:7890",
                                    alias: "config/proxy-pool.json",
                                })
                run_collection.assert_not_called()

    def test_environment_default_does_not_conflict_with_pool_parameter(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            with (
                self.capture_config(root) as (captured, run_collection),
                patch.dict("os.environ", {"ONLINE_COVER_PROXY": "http://127.0.0.1:7890"}),
            ):
                job_tasks.run_collection_task("collection-nh-online", {
                    "confirm": True, "proxyPoolFile": "config/proxy-pool.json",
                })
                self.assertEqual(captured["proxy_pool_file"], root / "config" / "proxy-pool.json")
                self.assertIsNone(captured["proxy"])
                run_collection.assert_called_once()

    def test_frontend_all_four_forms_have_empty_pool_file_field_next_to_proxy(self):
        source = (Path(__file__).resolve().parents[1] / "web" / "src" / "data" / "scripts.ts").read_text(encoding="utf-8")
        collection_source = source.split("export const collectionScripts:", 1)[1]
        for script_id in job_tasks.COLLECTION_MODES:
            with self.subTest(script_id=script_id):
                block = collection_source.split(f'id: "{script_id}"', 1)[1].split('\n    id: "', 1)[0]
                self.assertEqual(block.count('text("proxyPoolFile"'), 1)
                self.assertRegex(block, re.compile(
                    r'text\("proxy",[^\n]+\),\s*text\("proxyPoolFile", "多节点代理配置文件", "",',
                ))
                self.assertIn("启用时请清空上方单代理", block)
                self.assertIn("不修改 Clash 当前节点或 TUN", block)


if __name__ == "__main__":
    unittest.main()
