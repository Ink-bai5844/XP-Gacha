from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from server import local_folders


class OpenLocalFolderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.target = Path("/Users/reader/漫画/My book; $(touch unwanted)")

    def test_macos_passes_path_as_one_argument_and_waits_for_success(self) -> None:
        with (
            patch.object(local_folders.os, "name", "posix"),
            patch.object(local_folders.sys, "platform", "darwin"),
            patch.object(local_folders.subprocess, "run") as run,
        ):
            self.assertTrue(local_folders.open_local_folder(self.target, allowed=True))
        run.assert_called_once_with(["/usr/bin/open", str(self.target)], check=True)

    def test_macos_launch_failure_does_not_report_success(self) -> None:
        for failure in (
            FileNotFoundError("open unavailable"),
            subprocess.CalledProcessError(1, "/usr/bin/open"),
        ):
            with (
                self.subTest(failure=type(failure).__name__),
                patch.object(local_folders.os, "name", "posix"),
                patch.object(local_folders.sys, "platform", "darwin"),
                patch.object(local_folders.subprocess, "run", side_effect=failure),
                self.assertRaises(type(failure)),
            ):
                local_folders.open_local_folder(self.target, allowed=True)

    def test_windows_retains_startfile_behavior(self) -> None:
        with (
            patch.object(local_folders.os, "name", "nt"),
            patch.object(local_folders.os, "startfile", create=True) as startfile,
            patch.object(local_folders.subprocess, "run") as run,
        ):
            self.assertTrue(local_folders.open_local_folder(self.target, allowed=True))
        startfile.assert_called_once_with(str(self.target))
        run.assert_not_called()

    def test_disabled_setting_blocks_both_supported_platforms(self) -> None:
        for os_name, platform in (("nt", "win32"), ("posix", "darwin")):
            with (
                self.subTest(platform=platform),
                patch.object(local_folders.os, "name", os_name),
                patch.object(local_folders.sys, "platform", platform),
                patch.object(local_folders.os, "startfile", create=True) as startfile,
                patch.object(local_folders.subprocess, "run") as run,
            ):
                self.assertFalse(local_folders.open_local_folder(self.target, allowed=False))
            startfile.assert_not_called()
            run.assert_not_called()

    def test_unsupported_platform_reports_not_opened(self) -> None:
        with (
            patch.object(local_folders.os, "name", "posix"),
            patch.object(local_folders.sys, "platform", "linux"),
            patch.object(local_folders.os, "startfile", create=True) as startfile,
            patch.object(local_folders.subprocess, "run") as run,
        ):
            self.assertFalse(local_folders.open_local_folder(self.target, allowed=True))
        startfile.assert_not_called()
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
