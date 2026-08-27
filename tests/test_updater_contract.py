from __future__ import annotations

import re
import os
import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_project_file(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8-sig")


class UpdaterDistributionContractTests(unittest.TestCase):
    def test_release_version_is_0_2_7(self) -> None:
        version_module = read_project_file("server/__init__.py")
        match = re.search(r'^__version__\s*=\s*"([^"]+)"', version_module, re.MULTILINE)

        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "0.2.7")

    def test_common_cmd_dispatches_to_the_powershell_updater(self) -> None:
        command = read_project_file("Update XP-Gacha.cmd")

        self.assertTrue((PROJECT_ROOT / "tools" / "update_xp_gacha.ps1").is_file())
        self.assertIn(r"tools\update_xp_gacha.ps1", command)
        self.assertIn("%*", command, "The CMD entry point must forward updater options.")
        self.assertRegex(command, r"(?i)exit\s+/b\s+%UPDATE_EXIT%")

    @unittest.skipUnless(os.name == "nt", "Windows CMD smoke test")
    def test_common_cmd_can_run_the_updater_self_test(self) -> None:
        environment = os.environ.copy()
        environment["XP_GACHA_NO_PAUSE"] = "1"
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", "call", "Update XP-Gacha.cmd", "-SelfTest"],
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("Updater self-tests passed", completed.stdout)

    def test_full_release_contains_the_common_update_entry_point(self) -> None:
        release_builder = read_project_file("scripts/build_portable_release.ps1")

        self.assertIn('"Update XP-Gacha.cmd"', release_builder)
        self.assertIn('"build_portable_update.ps1"', release_builder)
        self.assertIn("status --porcelain --untracked-files=all", release_builder)
        self.assertIn('"refs/tags/v$ReleaseVersion^{}"', release_builder)

    def test_update_builder_publishes_the_three_named_assets(self) -> None:
        update_builder = read_project_file("scripts/build_portable_update.ps1")

        self.assertIn('"$releaseName-update.zip"', update_builder)
        self.assertIn('"$releaseName-update.json"', update_builder)
        self.assertIn('$sidecarPath = "$packagePath.sha256"', update_builder)
        self.assertIn("runtimeCompatibility", update_builder)
        self.assertIn('"web/dist"', update_builder)
        self.assertIn("blankFirstStartVerified", update_builder)
        self.assertIn('"refs/tags/v$TargetVersion^{}"', update_builder)

        for protected_path in (
            "runtime",
            "data",
            "mysql",
            "config",
            "models",
            "manga_vectors",
            "dictionaries",
            "updates",
            ".env",
            ".env.local",
            "portable-settings.env",
        ):
            self.assertIn(f'"{protected_path}"', update_builder)

    def test_docs_describe_bootstrap_protection_and_all_update_assets(self) -> None:
        readme = read_project_file("README.md")
        portable_readme = read_project_file("portable/README_便携版.md")

        for document in (readme, portable_readme):
            self.assertIn("Update XP-Gacha.cmd", document)
            self.assertIn("v0.2.6", document)
            self.assertIn("v0.2.7", document)
            self.assertIn("updates/backups", document)
            self.assertIn("portable-settings.env", document)

        for suffix in ("update.json", "update.zip", "update.zip.sha256"):
            self.assertIn(f"XP-Gacha-v<version>-portable-win64-{suffix}", readme)

        self.assertIn("git merge --ff-only", readme)
        self.assertIn("runtimeCompatibility", read_project_file("scripts/build_portable_update.ps1"))

    def test_update_work_directory_is_git_ignored(self) -> None:
        ignored_paths = read_project_file(".gitignore").splitlines()

        self.assertIn("/updates/", ignored_paths)

    def test_windows_parser_check_includes_the_updater(self) -> None:
        parser_check = read_project_file("tests/check_windows_powershell_scripts.ps1")

        self.assertIn(r"tools\update_xp_gacha.ps1", parser_check)


if __name__ == "__main__":
    unittest.main()
