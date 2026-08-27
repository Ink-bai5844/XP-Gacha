from __future__ import annotations

import importlib.util
import subprocess
import tempfile
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


class PortableJobVerificationTests(unittest.TestCase):
    def test_verify_job_runner_waits_for_child_process_completion(self) -> None:
        responses = [
            {"id": "job-1", "status": "queued"},
            {"id": "job-1", "status": "running", "returnCode": None, "lines": []},
            {
                "id": "job-1",
                "status": "completed",
                "returnCode": 0,
                "lines": ["[JOB] completed"],
            },
        ]

        with (
            mock.patch.object(portable_launcher, "request_json", side_effect=responses) as request_json,
            mock.patch.object(portable_launcher.time, "monotonic", return_value=0),
            mock.patch.object(portable_launcher.time, "sleep"),
        ):
            portable_launcher.verify_job_runner("http://127.0.0.1:8000")

        self.assertEqual(request_json.call_args_list[0].kwargs["method"], "POST")
        self.assertEqual(
            request_json.call_args_list[0].kwargs["payload"],
            {"scriptId": "cache-delete", "parameters": {"confirm": True, "targets": []}},
        )

    def test_verify_job_runner_reports_child_process_failure(self) -> None:
        responses = [
            {"id": "job-2", "status": "queued"},
            {
                "id": "job-2",
                "status": "failed",
                "returnCode": 1,
                "lines": ["ModuleNotFoundError: No module named 'server'"],
            },
        ]

        with (
            mock.patch.object(portable_launcher, "request_json", side_effect=responses),
            mock.patch.object(portable_launcher.time, "monotonic", return_value=0),
            mock.patch.object(portable_launcher.time, "sleep"),
        ):
            with self.assertRaisesRegex(RuntimeError, "ModuleNotFoundError"):
                portable_launcher.verify_job_runner("http://127.0.0.1:8000")


class PortableRootDataLayoutTests(unittest.TestCase):
    def test_stale_update_lock_is_removed_before_manual_start(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_file = Path(temporary) / "updates" / "update.lock"
            lock_file.parent.mkdir(parents=True)
            lock_file.write_text("", encoding="ascii")
            with mock.patch.object(portable_launcher, "UPDATE_LOCK_FILE", lock_file):
                portable_launcher.ensure_update_not_in_progress()

            self.assertFalse(lock_file.exists())

    def test_active_update_lock_blocks_manual_start(self) -> None:
        with (
            mock.patch.object(portable_launcher.os, "open", side_effect=PermissionError("locked")),
            self.assertRaisesRegex(RuntimeError, "一键更新正在进行"),
        ):
            portable_launcher.ensure_update_not_in_progress()

    def test_updater_owned_restart_can_bypass_its_lock(self) -> None:
        with (
            mock.patch.dict(portable_launcher.os.environ, {"XP_GACHA_UPDATE_RESTART": "1"}),
            mock.patch.object(portable_launcher.os, "open") as open_file,
        ):
            portable_launcher.ensure_update_not_in_progress()

        open_file.assert_not_called()

    def test_fresh_install_records_recovery_marker_before_config_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config_file = root / "config" / "portable.json"
            marker_file = root / "config" / ".initialization-pending.json"
            with (
                mock.patch.object(portable_launcher, "PORTABLE_CONFIG_FILE", config_file),
                mock.patch.object(portable_launcher, "INITIALIZATION_MARKER_FILE", marker_file),
                mock.patch.object(portable_launcher, "MYSQL_DATA_ROOT", root / "mysql" / "data"),
                mock.patch.object(portable_launcher.time, "time", return_value=1234.5),
            ):
                portable_launcher.begin_config_mysql_initialization()

            self.assertFalse(config_file.exists())
            self.assertEqual(
                portable_launcher.read_json(marker_file),
                {"schemaVersion": portable_launcher.SCHEMA_VERSION, "startedAt": 1234.5},
            )

    def test_config_only_with_recovery_marker_is_allowed_to_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config_file = root / "config" / "portable.json"
            marker_file = root / "config" / ".initialization-pending.json"
            config_file.parent.mkdir(parents=True)
            config_file.write_text("{}", encoding="utf-8")
            marker_file.write_text("{}", encoding="utf-8")
            with (
                mock.patch.object(portable_launcher, "PORTABLE_CONFIG_FILE", config_file),
                mock.patch.object(portable_launcher, "INITIALIZATION_MARKER_FILE", marker_file),
                mock.patch.object(portable_launcher, "MYSQL_DATA_ROOT", root / "mysql" / "data"),
            ):
                portable_launcher.validate_config_mysql_pair()

    def test_completed_initialization_clears_recovery_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config_file = root / "config" / "portable.json"
            marker_file = root / "config" / ".initialization-pending.json"
            config_file.parent.mkdir(parents=True)
            config_file.write_text("{}", encoding="utf-8")
            marker_file.write_text("{}", encoding="utf-8")
            mysql_data_root = root / "mysql" / "data"
            (mysql_data_root / "mysql").mkdir(parents=True)
            with (
                mock.patch.object(portable_launcher, "PORTABLE_CONFIG_FILE", config_file),
                mock.patch.object(portable_launcher, "INITIALIZATION_MARKER_FILE", marker_file),
                mock.patch.object(portable_launcher, "MYSQL_DATA_ROOT", mysql_data_root),
            ):
                portable_launcher.complete_config_mysql_initialization()

            self.assertFalse(marker_file.exists())

    def test_config_and_mysql_pair_allows_fresh_install(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            mysql_data_root = root / "mysql" / "data"
            mysql_data_root.mkdir(parents=True)
            with (
                mock.patch.object(portable_launcher, "PORTABLE_CONFIG_FILE", root / "config" / "portable.json"),
                mock.patch.object(
                    portable_launcher,
                    "INITIALIZATION_MARKER_FILE",
                    root / "config" / ".initialization-pending.json",
                ),
                mock.patch.object(portable_launcher, "MYSQL_DATA_ROOT", mysql_data_root),
            ):
                portable_launcher.validate_config_mysql_pair()

    def test_config_and_mysql_pair_allows_complete_migration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config_file = root / "config" / "portable.json"
            config_file.parent.mkdir(parents=True)
            config_file.write_text("{}", encoding="utf-8")
            mysql_data_root = root / "mysql" / "data"
            (mysql_data_root / "mysql").mkdir(parents=True)
            with (
                mock.patch.object(portable_launcher, "PORTABLE_CONFIG_FILE", config_file),
                mock.patch.object(
                    portable_launcher,
                    "INITIALIZATION_MARKER_FILE",
                    root / "config" / ".initialization-pending.json",
                ),
                mock.patch.object(portable_launcher, "MYSQL_DATA_ROOT", mysql_data_root),
            ):
                portable_launcher.validate_config_mysql_pair()

    def test_config_without_initialized_mysql_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config_file = root / "config" / "portable.json"
            config_file.parent.mkdir(parents=True)
            config_file.write_text("{}", encoding="utf-8")
            with (
                mock.patch.object(portable_launcher, "PORTABLE_CONFIG_FILE", config_file),
                mock.patch.object(
                    portable_launcher,
                    "INITIALIZATION_MARKER_FILE",
                    root / "config" / ".initialization-pending.json",
                ),
                mock.patch.object(portable_launcher, "MYSQL_DATA_ROOT", root / "mysql" / "data"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "config/portable.json.*mysql/data 尚未初始化.*成套迁移",
                ):
                    portable_launcher.validate_config_mysql_pair()

    def test_initialized_mysql_without_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            mysql_data_root = root / "mysql" / "data"
            (mysql_data_root / "mysql").mkdir(parents=True)
            with (
                mock.patch.object(portable_launcher, "PORTABLE_CONFIG_FILE", root / "config" / "portable.json"),
                mock.patch.object(
                    portable_launcher,
                    "INITIALIZATION_MARKER_FILE",
                    root / "config" / ".initialization-pending.json",
                ),
                mock.patch.object(portable_launcher, "MYSQL_DATA_ROOT", mysql_data_root),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "已初始化的 mysql/data.*缺少 config/portable.json.*成套迁移",
                ):
                    portable_launcher.validate_config_mysql_pair()

    def test_runtime_paths_are_siblings_of_the_package_root(self) -> None:
        root = portable_launcher.PACKAGE_ROOT

        self.assertEqual(portable_launcher.CONFIG_ROOT, root / "config")
        self.assertEqual(portable_launcher.RUN_ROOT, root / "run")
        self.assertEqual(portable_launcher.LOG_ROOT, root / "logs")
        self.assertEqual(portable_launcher.TMP_ROOT, root / "tmp")
        self.assertEqual(portable_launcher.MYSQL_DATA_ROOT, root / "mysql" / "data")
        self.assertNotIn("userdata", {part.lower() for part in portable_launcher.MYSQL_DATA_ROOT.parts})

    def test_base_environment_uses_source_compatible_root_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package_root = Path(temporary).resolve()
            with (
                mock.patch.object(portable_launcher, "PACKAGE_ROOT", package_root),
                mock.patch.object(portable_launcher, "DATA_ROOT", package_root),
                mock.patch.object(portable_launcher, "PYTHON_HOME", package_root / "runtime" / "python"),
                mock.patch.object(portable_launcher, "MYSQL_BIN", package_root / "runtime" / "mysql" / "bin"),
                mock.patch.object(portable_launcher, "TMP_ROOT", package_root / "tmp"),
                mock.patch.object(portable_launcher, "SETTINGS_FILE", package_root / "portable-settings.env"),
            ):
                env = portable_launcher.base_environment(
                    {},
                    8000,
                    3307,
                    {
                        "databaseName": "xp_gacha",
                        "databaseUser": "xp_gacha",
                        "databasePassword": "password",
                    },
                )

            self.assertEqual(env["XP_GACHA_DATA_ROOT"], str(package_root))
            self.assertEqual(env["XP_GACHA_BASE_DIR"], str(package_root / "library"))
            self.assertEqual(env["HF_HOME"], str(package_root / "models" / "cache" / "huggingface"))
            self.assertEqual(env["XDG_CACHE_HOME"], str(package_root / "models" / "cache" / "xdg"))
            self.assertEqual(env["CACHE_DIR"], str(package_root / "datacache"))
            self.assertEqual(env["DICTIONARY_DIR"], str(package_root / "dictionaries"))
            self.assertEqual(env["VECTOR_FILE"], str(package_root / "manga_vectors" / "manga_vectors_Qwen3.pkl"))

    def test_portable_paths_override_machine_level_path_variables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package_root = Path(temporary).resolve()
            with (
                mock.patch.dict(
                    portable_launcher.os.environ,
                    {
                        "CACHE_DIR": "C:/outside/cache",
                        "MODEL_DIR": "C:/outside/models",
                        "VECTOR_FILE": "C:/outside/vector.pkl",
                    },
                    clear=False,
                ),
                mock.patch.object(portable_launcher, "PACKAGE_ROOT", package_root),
                mock.patch.object(portable_launcher, "DATA_ROOT", package_root),
                mock.patch.object(portable_launcher, "PYTHON_HOME", package_root / "runtime" / "python"),
                mock.patch.object(portable_launcher, "MYSQL_BIN", package_root / "runtime" / "mysql" / "bin"),
                mock.patch.object(portable_launcher, "TMP_ROOT", package_root / "tmp"),
                mock.patch.object(portable_launcher, "SETTINGS_FILE", package_root / "portable-settings.env"),
            ):
                env = portable_launcher.base_environment(
                    {},
                    8000,
                    3307,
                    {
                        "databaseName": "xp_gacha",
                        "databaseUser": "xp_gacha",
                        "databasePassword": "password",
                    },
                )

            self.assertEqual(env["CACHE_DIR"], str(package_root / "datacache"))
            self.assertEqual(env["MODEL_DIR"], str(package_root / "models"))
            self.assertEqual(env["VECTOR_FILE"], str(package_root / "manga_vectors" / "manga_vectors_Qwen3.pkl"))

    def test_legacy_default_library_path_does_not_recreate_userdata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package_root = Path(temporary).resolve()
            with (
                mock.patch.object(portable_launcher, "PACKAGE_ROOT", package_root),
                mock.patch.object(portable_launcher, "DATA_ROOT", package_root),
                mock.patch.object(portable_launcher, "PYTHON_HOME", package_root / "runtime" / "python"),
                mock.patch.object(portable_launcher, "MYSQL_BIN", package_root / "runtime" / "mysql" / "bin"),
                mock.patch.object(portable_launcher, "TMP_ROOT", package_root / "tmp"),
                mock.patch.object(portable_launcher, "SETTINGS_FILE", package_root / "portable-settings.env"),
            ):
                env = portable_launcher.base_environment(
                    {"XP_GACHA_LIBRARY_PATH": "userdata/library"},
                    8000,
                    3307,
                    {
                        "databaseName": "xp_gacha",
                        "databaseUser": "xp_gacha",
                        "databasePassword": "password",
                    },
                )

            self.assertEqual(env["XP_GACHA_BASE_DIR"], str(package_root / "library"))
            self.assertFalse((package_root / "userdata").exists())

    def test_package_initialization_creates_only_root_level_data_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            python = root / "runtime" / "python" / "python.exe"
            mysql_bin = root / "runtime" / "mysql" / "bin"
            required = [
                python,
                mysql_bin / "mysqld.exe",
                mysql_bin / "mysql.exe",
                mysql_bin / "mysqladmin.exe",
                root / "server" / "main.py",
                root / "server" / "job_tasks.py",
                root / "web" / "dist" / "index.html",
            ]
            for path in required:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            with (
                mock.patch.object(portable_launcher, "PACKAGE_ROOT", root),
                mock.patch.object(portable_launcher, "PYTHON_EXE", python),
                mock.patch.object(portable_launcher, "MYSQLD_EXE", mysql_bin / "mysqld.exe"),
                mock.patch.object(portable_launcher, "MYSQL_EXE", mysql_bin / "mysql.exe"),
                mock.patch.object(portable_launcher, "MYSQLADMIN_EXE", mysql_bin / "mysqladmin.exe"),
                mock.patch.object(portable_launcher, "CONFIG_ROOT", root / "config"),
                mock.patch.object(portable_launcher, "RUN_ROOT", root / "run"),
                mock.patch.object(portable_launcher, "LOG_ROOT", root / "logs"),
                mock.patch.object(portable_launcher, "TMP_ROOT", root / "tmp"),
                mock.patch.object(portable_launcher, "MYSQL_DATA_ROOT", root / "mysql" / "data"),
            ):
                portable_launcher.ensure_package_layout()

            for name in (
                "data",
                "datacache",
                "b64_cache",
                "b64_tmp",
                "localimgtmp",
                "onlineimgtmp",
                "library",
                "manga_vectors",
                "models",
                "mysql",
                "config",
                "run",
                "logs",
                "tmp",
                "dictionaries",
            ):
                self.assertTrue((root / name).is_dir(), name)
            self.assertFalse((root / "userdata").exists())

    def test_package_initialization_stops_for_unmigrated_legacy_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            python = root / "runtime" / "python" / "python.exe"
            mysql_bin = root / "runtime" / "mysql" / "bin"
            required = [
                python,
                mysql_bin / "mysqld.exe",
                mysql_bin / "mysql.exe",
                mysql_bin / "mysqladmin.exe",
                root / "server" / "main.py",
                root / "server" / "job_tasks.py",
                root / "web" / "dist" / "index.html",
            ]
            for path in required:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            legacy_config = root / "userdata" / "config" / "portable.json"
            legacy_config.parent.mkdir(parents=True)
            legacy_config.write_text("{}", encoding="utf-8")

            with (
                mock.patch.object(portable_launcher, "PACKAGE_ROOT", root),
                mock.patch.object(portable_launcher, "PYTHON_EXE", python),
                mock.patch.object(portable_launcher, "MYSQLD_EXE", mysql_bin / "mysqld.exe"),
                mock.patch.object(portable_launcher, "MYSQL_EXE", mysql_bin / "mysql.exe"),
                mock.patch.object(portable_launcher, "MYSQLADMIN_EXE", mysql_bin / "mysqladmin.exe"),
            ):
                with self.assertRaisesRegex(RuntimeError, "v0.2.2 → v0.2.3"):
                    portable_launcher.ensure_package_layout()

    def test_release_verification_rejects_redirected_data_paths(self) -> None:
        root = portable_launcher.DATA_ROOT
        valid_status = {
            "paths": {
                "dataRoot": str(root),
                "library": str(root / "library"),
                "dictionaries": str(root / "dictionaries"),
            },
            "searchCapabilities": {
                "semantic": {
                    "dependencies": {
                        "model": {"path": str(root / "models" / "Qwen3-Embedding-0.6B")},
                        "vector": {"path": str(root / "manga_vectors" / "manga_vectors_Qwen3.pkl")},
                    }
                },
                "cover": {
                    "dependencies": {
                        "model": {"path": str(root / "models" / "clip-vit-base-patch32")},
                        "vector": {"path": str(root / "manga_vectors" / "clip_image_index.pkl")},
                    }
                },
            },
        }
        portable_launcher.verify_root_data_paths(valid_status)

        with self.assertRaisesRegex(RuntimeError, "dataRoot"):
            invalid_status = dict(valid_status)
            invalid_status["paths"] = {**valid_status["paths"], "dataRoot": str(root / "elsewhere")}
            portable_launcher.verify_root_data_paths(invalid_status)


if __name__ == "__main__":
    unittest.main()
