"""Isolated import/CLI regressions; these tests never start network requests."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CollectorEntrypointTests(unittest.TestCase):
    def run_python(self, arguments: list[str], *, cwd: Path | None = None) -> str:
        result = subprocess.run(
            [sys.executable, *arguments],
            cwd=cwd or ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_short_and_package_imports_share_types_in_both_orders(self) -> None:
        for first in ("short", "package"):
            with self.subTest(first=first), tempfile.TemporaryDirectory() as temporary:
                first_import = (
                    f"sys.path.insert(0, {str(ROOT / 'data_get')!r}); import collector"
                    if first == "short"
                    else f"sys.path.insert(0, {str(ROOT)!r}); from data_get import collector"
                )
                program = "\n".join([
                    "import sys, threading, importlib",
                    first_import,
                    "short = importlib.import_module('collector')",
                    "packaged = importlib.import_module('data_get.collector')",
                    "assert short is packaged is collector",
                    "from data_get.concurrency_strategies import create_controller",
                    "for strategy in ('stable', 'adaptive-step', 'bandit'):",
                    "    control = create_controller(strategy, 2)",
                    "    assert isinstance(control, collector.AdaptiveConcurrency)",
                    "    cancelled = threading.Event(); cancelled.set()",
                    "    try:",
                    "        control.acquire(cancelled)",
                    "    except collector.StopRequested:",
                    "        pass",
                    "    else:",
                    "        raise AssertionError('cancelled gate did not stop')",
                    "print('identical modules and cancellation types')",
                ])
                output = self.run_python(["-I", "-c", program], cwd=Path(temporary))
                self.assertIn("identical modules", output)

    def test_module_entrypoint_delegates_to_existing_canonical_module(self) -> None:
        # Replace main before executing __main__: any second set of classes or
        # an attempt to launch a collector instead of delegating fails locally.
        program = "\n".join([
            "import runpy, sys, warnings",
            "from data_get import collector",
            "original = collector.StopRequested",
            "collector.main = lambda: 37",
            "with warnings.catch_warnings():",
            "    warnings.simplefilter('ignore', RuntimeWarning)",
            "    try:",
            "        runpy.run_module('data_get.collector', run_name='__main__', alter_sys=True)",
            "    except SystemExit as exc:",
            "        assert exc.code == 37, exc.code",
            "    else:",
            "        raise AssertionError('entrypoint did not delegate')",
            "assert sys.modules['data_get.collector'].StopRequested is original",
        ])
        self.run_python(["-c", program])

    def test_direct_and_legacy_entrypoints_expose_new_options_outside_project(self) -> None:
        entrypoints = (
            ("data_get/collector.py", ["nh-online"]),
            ("data_get/NH_get_info_online.py", []),
            ("data_get/JM_get_info_online.py", []),
            ("data_get/local/NH_get_info_local.py", []),
            ("data_get/local/NH_get_images_local.py", []),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for path, arguments in entrypoints:
                with self.subTest(path=path):
                    output = self.run_python(
                        ["-E", "-s", str(ROOT / path), *arguments, "--help"], cwd=Path(temporary),
                    )
                    self.assertIn("--strategy", output)
                    self.assertIn("--no-reuse-connections", output)
        output = self.run_python(["-m", "data_get.collector", "nh-online", "--help"])
        self.assertIn("--strategy", output)


if __name__ == "__main__":
    unittest.main()
