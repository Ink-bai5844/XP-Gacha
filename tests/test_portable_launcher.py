from __future__ import annotations

import importlib.util
import subprocess
import unittest
from pathlib import Path
from unittest import mock


LAUNCHER_PATH = Path(__file__).resolve().parents[1] / "portable" / "portable_launcher.py"
SPEC = importlib.util.spec_from_file_location("portable_launcher_under_test", LAUNCHER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {LAUNCHER_PATH}")
portable_launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(portable_launcher)


class PortableMySqlAuthenticationTests(unittest.TestCase):
    def test_provisioning_can_reauthenticate_caching_sha2_accounts(self) -> None:
        config = {
            "databaseName": "xp_gacha",
            "databaseUser": "xp_gacha",
            "databasePassword": "AppPassword_123",
            "rootPassword": "RootPassword_123",
        }

        def mysql_result(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            if "--get-server-public-key" in command:
                return subprocess.CompletedProcess(command, 0, "")
            return subprocess.CompletedProcess(
                command,
                1,
                "ERROR 2061 (HY000): Authentication plugin 'caching_sha2_password' "
                "reported error: Authentication requires secure connection.",
            )

        with (
            mock.patch.object(portable_launcher.subprocess, "run", side_effect=mysql_result),
            mock.patch.object(portable_launcher.time, "monotonic", side_effect=[0, 0, 76]),
            mock.patch.object(portable_launcher.time, "sleep"),
        ):
            portable_launcher.provision_mysql(config, 3307, {})

    def test_shutdown_can_reauthenticate_caching_sha2_root(self) -> None:
        config = {"rootPassword": "RootPassword_123"}

        def mysqladmin_result(command: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
            return subprocess.CompletedProcess(
                command,
                0 if "--get-server-public-key" in command else 1,
                b"",
            )

        with (
            mock.patch.object(portable_launcher, "MYSQLADMIN_EXE", LAUNCHER_PATH),
            mock.patch.object(portable_launcher.subprocess, "run", side_effect=mysqladmin_result),
        ):
            self.assertTrue(portable_launcher.mysql_admin_shutdown(config, 3307, {}))


if __name__ == "__main__":
    unittest.main()
