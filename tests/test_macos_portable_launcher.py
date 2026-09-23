from __future__ import annotations

import importlib.util
import os
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest import mock


LAUNCHER_PATH = Path(__file__).resolve().parents[1] / "portable" / "portable_launcher.py"
SPEC = importlib.util.spec_from_file_location("macos_portable_launcher_under_test", LAUNCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)

CONFIG = {
    "databaseName": "xp_gacha",
    "databaseUser": "xp_gacha",
    "databasePassword": "app-password",
    "rootPassword": "root-password",
}


@contextmanager
def package_paths(root: Path):
    paths = {
        "PACKAGE_ROOT": root,
        "DATA_ROOT": root,
        "PYTHON_HOME": root / "runtime" / "python",
        "PYTHON_EXE": root / "runtime" / "python" / "bin" / "python3",
        "MYSQL_HOME": root / "runtime" / "mysql",
        "MYSQL_BIN": root / "runtime" / "mysql" / "bin",
        "CONFIG_ROOT": root / "config",
        "RUN_ROOT": root / "run",
        "LOG_ROOT": root / "logs",
        "TMP_ROOT": root / "tmp",
        "MYSQL_DATA_ROOT": root / "mysql" / "data",
        "MYSQL_CONFIG_FILE": root / "config" / "my.cnf",
        "PORTABLE_CONFIG_FILE": root / "config" / "portable.json",
        "INITIALIZATION_MARKER_FILE": root / "config" / ".initialization.json",
        "STATE_FILE": root / "run" / "state.json",
        "STOP_REQUEST_FILE": root / "run" / "stop.request",
        "SETTINGS_FILE": root / "portable-settings.env",
    }
    for name, executable in (("MYSQLD_EXE", "mysqld"), ("MYSQL_EXE", "mysql"), ("MYSQLADMIN_EXE", "mysqladmin")):
        paths[name] = paths["MYSQL_BIN"] / executable
    for name in ("CONFIG_ROOT", "RUN_ROOT", "LOG_ROOT", "TMP_ROOT", "MYSQL_BIN", "PYTHON_HOME"):
        paths[name].mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(launcher, "IS_MACOS", True))
        for name, path in paths.items():
            stack.enter_context(mock.patch.object(launcher, name, path))
        yield


@unittest.skipIf(os.name == "nt", "macOS launcher uses POSIX primitives")
class MacOSPortableLauncherTests(unittest.TestCase):
    def test_darwin_uses_native_runtime_paths(self):
        with mock.patch.object(sys, "platform", "darwin"):
            spec = importlib.util.spec_from_file_location("darwin_path_test", LAUNCHER_PATH)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        self.assertEqual(module.PYTHON_EXE, module.RUNTIME_ROOT / "python/bin/python3")
        self.assertEqual(module.MYSQLD_EXE.name, "mysqld")
        self.assertEqual(module.MYSQL_EXE.name, "mysql")
        self.assertEqual(module.MYSQLADMIN_EXE.name, "mysqladmin")
        self.assertEqual(module.MYSQL_CONFIG_FILE.name, "my.cnf")

    def test_environment_removes_host_python_and_database_overrides(self):
        with tempfile.TemporaryDirectory(prefix="XP Gacha ") as temp, package_paths(Path(temp)):
            with mock.patch.dict(os.environ, {
                "PYTHONPATH": "/another/python",
                "PYTHONHOME": "/opt/homebrew",
                "__PYVENV_LAUNCHER__": "/another/python",
                "DYLD_LIBRARY_PATH": "/another/lib",
                "VIRTUAL_ENV": "/another/venv",
                "MYSQL_HOME": "/another/mysql",
                "MYSQL_UNIX_PORT": "/another/mysql.sock",
                "MYSQL_PWD": "external-password",
            }):
                env = launcher.base_environment({}, 8001, 3308, CONFIG)
            for name in ("PYTHONPATH", "PYTHONHOME", "__PYVENV_LAUNCHER__", "DYLD_LIBRARY_PATH", "VIRTUAL_ENV", "MYSQL_HOME", "MYSQL_UNIX_PORT", "MYSQL_PWD"):
                self.assertNotIn(name, env)
            self.assertEqual(env["PATH"].split(os.pathsep)[:2], [str(launcher.PYTHON_HOME / "bin"), str(launcher.MYSQL_BIN)])
            self.assertEqual(env["MYSQL_PORT"], "3308")
            self.assertEqual(env["XP_GACHA_PORT"], "8001")
            self.assertEqual(env["XP_GACHA_DATA_ROOT"], temp)

    def test_bound_port_is_skipped_without_windows_socket_option(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            self.assertFalse(launcher.port_is_free(port))
            self.assertNotEqual(launcher.choose_free_port(port, avoid={port + 1}), port)

    def test_flock_prevents_competing_launchers_and_releases_on_close(self):
        with tempfile.TemporaryDirectory() as temp, package_paths(Path(temp)):
            first, second = launcher.SingleInstanceMutex(), launcher.SingleInstanceMutex()
            try:
                self.assertTrue(first.acquire())
                self.assertFalse(second.acquire())
                first.close()
                self.assertTrue(second.acquire())
            finally:
                first.close()
                second.close()

    def test_root_start_is_rejected_before_any_initialization(self):
        with mock.patch.object(launcher, "IS_MACOS", True), mock.patch.object(os, "geteuid", return_value=0), mock.patch.object(launcher, "ensure_package_layout") as layout:
            with self.assertRaisesRegex(RuntimeError, "root 或 sudo"):
                launcher.run_start()
            layout.assert_not_called()

    def test_short_socket_directory_does_not_depend_on_long_package_path(self):
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(launcher, "PACKAGE_ROOT", Path(temp) / ("long folder " * 15)):
                directory = launcher.mysql_socket_directory()
                try:
                    self.assertLess(len(str(directory / "mysql.sock").encode()), 104)
                    self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
                    self.assertEqual(directory.stat().st_uid, os.getuid())
                finally:
                    directory.rmdir()

    def test_mysql_config_quotes_spaces_and_uses_private_socket(self):
        with tempfile.TemporaryDirectory(prefix="XP Gacha ") as temp, package_paths(Path(temp)):
            with mock.patch.object(launcher, "mysql_socket_directory", return_value=Path("/tmp/private-xpg")):
                launcher.write_mysql_config(3310)
            content = launcher.MYSQL_CONFIG_FILE.read_text()
            self.assertIn(f'basedir="{launcher.MYSQL_HOME.resolve()}"', content)
            self.assertIn('socket="/private/tmp/private-xpg/mysql.sock"' if sys.platform == "darwin" else 'socket="/tmp/private-xpg/mysql.sock"', content)
            self.assertIn("bind-address=127.0.0.1", content)
            self.assertIn("port=3310", content)

    def test_mysql_initialization_drops_windows_only_options(self):
        with tempfile.TemporaryDirectory(prefix="XP Gacha ") as temp, package_paths(Path(temp)):
            def initialize(command, **kwargs):
                (launcher.MYSQL_DATA_ROOT / "mysql").mkdir()
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(subprocess, "run", side_effect=initialize) as run:
                launcher.initialize_mysql({"PATH": "bundled"})
            command = run.call_args.args[0]
            self.assertIn("--initialize-insecure", command)
            self.assertNotIn("--console", command)
            self.assertNotIn("creationflags", run.call_args.kwargs)
            self.assertIn(f"--datadir={launcher.MYSQL_DATA_ROOT}", command)

    def test_mysql_start_uses_process_group_and_drops_windows_options(self):
        with tempfile.TemporaryDirectory(prefix="XP Gacha ") as temp, package_paths(Path(temp)):
            with mock.patch.object(launcher, "write_mysql_config"), mock.patch.object(subprocess, "Popen") as popen, mock.patch.object(launcher, "wait_for_tcp", return_value=True):
                process, handle = launcher.start_mysql({}, 3307, mock.Mock())
                handle.close()
            command = popen.call_args.args[0]
            self.assertNotIn("--console", command)
            self.assertNotIn("--no-monitor", command)
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertNotIn("creationflags", popen.call_args.kwargs)
            self.assertIs(process, popen.return_value)

    def test_failed_mysql_start_cleans_up_launched_process(self):
        with tempfile.TemporaryDirectory() as temp, package_paths(Path(temp)):
            with mock.patch.object(launcher, "write_mysql_config"), mock.patch.object(subprocess, "Popen") as popen, mock.patch.object(launcher, "wait_for_tcp", return_value=False), mock.patch.object(launcher, "stop_process_gracefully") as stop:
                with self.assertRaisesRegex(RuntimeError, "MySQL 启动失败"):
                    launcher.start_mysql({}, 3307, mock.Mock())
            stop.assert_called_once_with(popen.return_value)
            self.assertTrue(popen.call_args.kwargs["stdout"].closed)

    def test_app_start_uses_bundled_python_and_process_group(self):
        with tempfile.TemporaryDirectory(prefix="XP Gacha ") as temp, package_paths(Path(temp)):
            with mock.patch.object(subprocess, "Popen") as popen:
                _, handle = launcher.start_app({"MYSQL_PORT": "3308"}, 8010, mock.Mock())
                handle.close()
            self.assertEqual(popen.call_args.args[0][0], str(launcher.PYTHON_EXE))
            self.assertIn("8010", popen.call_args.args[0])
            self.assertEqual(popen.call_args.kwargs["cwd"], Path(temp))
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertNotIn("creationflags", popen.call_args.kwargs)

    def test_graceful_shutdown_signals_child_process_group(self):
        process = mock.Mock(pid=43210)
        process.poll.return_value = None
        with mock.patch.object(launcher, "IS_MACOS", True), mock.patch.object(os, "getpgid", return_value=43210), mock.patch.object(os, "killpg") as killpg:
            launcher.stop_process_gracefully(process)
        killpg.assert_called_once_with(43210, signal.SIGTERM)
        process.wait.assert_called_once_with(timeout=12)

    def test_graceful_shutdown_escalates_timed_out_process_group(self):
        process = mock.Mock(pid=43210)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("app", 12), 0]
        with mock.patch.object(launcher, "IS_MACOS", True), mock.patch.object(os, "getpgid", return_value=43210), mock.patch.object(os, "killpg") as killpg:
            launcher.stop_process_gracefully(process)
        self.assertEqual(killpg.call_args_list, [mock.call(43210, signal.SIGTERM), mock.call(43210, signal.SIGKILL)])

    def test_reused_pid_is_never_terminated(self):
        record = {"pid": 43210, "path": str(launcher.PYTHON_EXE), "created": 100}
        with mock.patch.object(launcher, "process_identity", return_value={**record, "created": 200}), mock.patch.object(os, "killpg") as killpg, mock.patch.object(os, "kill") as kill:
            self.assertFalse(launcher.terminate_verified_tree(record))
        killpg.assert_not_called()
        kill.assert_not_called()

    def test_external_process_is_never_terminated(self):
        record = {"pid": 43210, "path": "/usr/bin/python3", "created": 100}
        with mock.patch.object(launcher, "process_matches", return_value=True), mock.patch.object(os, "killpg") as killpg, mock.patch.object(os, "kill") as kill:
            self.assertFalse(launcher.terminate_verified_tree(record))
        killpg.assert_not_called()
        kill.assert_not_called()

    def test_descendant_snapshot_filters_external_and_reused_pids_deepest_first(self):
        python_path = str(launcher.PYTHON_EXE.resolve())
        root = {"pid": 100, "path": python_path, "created": 1000, "parentPid": 1}
        child = {"pid": 101, "path": python_path, "created": 1001, "parentPid": 100}
        grandchild = {"pid": 102, "path": python_path, "created": 1002, "parentPid": 101}
        external = {"pid": 103, "path": "/usr/bin/python3", "created": 1003, "parentPid": 100}
        reused = {"pid": 104, "path": python_path, "created": 1004, "parentPid": 999}
        older = {"pid": 105, "path": python_path, "created": 500, "parentPid": 100}
        identities = {record["pid"]: record for record in (root, child, grandchild, external, reused, older)}
        snapshot = "100 1\n101 100\n102 101\n103 100\n104 101\n105 100\n"
        with mock.patch.object(launcher, "IS_MACOS", True), mock.patch.object(launcher, "process_identity", side_effect=identities.get), mock.patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0, snapshot)):
            self.assertEqual(launcher.bundled_python_descendants(root), [grandchild, child])

    def test_descendants_stop_before_api_process_group(self):
        root = {"pid": 100, "path": str(launcher.PYTHON_EXE.resolve()), "created": 1000}
        children = [{"pid": 102}, {"pid": 101}]
        events = []
        process = mock.Mock(pid=100)
        process.poll.return_value = None
        with mock.patch.object(launcher, "IS_MACOS", True), mock.patch.object(launcher, "process_identity", return_value=root), mock.patch.object(launcher, "bundled_python_descendants", return_value=children), mock.patch.object(launcher, "terminate_verified_tree", side_effect=lambda record, **kwargs: events.append(record["pid"])), mock.patch.object(os, "getpgid", return_value=100), mock.patch.object(os, "killpg", side_effect=lambda pid, signum: events.append(pid)):
            launcher.stop_process_gracefully(process)
        self.assertEqual(events, [102, 101, 100])

    def test_descendant_identity_is_rechecked_before_termination(self):
        child = {"pid": 101, "path": str(launcher.PYTHON_EXE.resolve()), "created": 1000}
        with mock.patch.object(launcher, "bundled_python_descendants", return_value=[child]), mock.patch.object(launcher, "process_identity", return_value={**child, "created": 2000}), mock.patch.object(os, "killpg") as killpg, mock.patch.object(os, "kill") as kill:
            launcher.stop_bundled_python_descendants({"pid": 100})
        killpg.assert_not_called()
        kill.assert_not_called()

    def test_verified_launcher_receives_individual_signal_not_terminal_group(self):
        record = {"pid": 43210, "path": str(launcher.PYTHON_EXE), "created": 100}
        with mock.patch.object(launcher, "IS_MACOS", True), mock.patch.object(launcher, "stop_bundled_python_descendants"), mock.patch.object(launcher, "process_matches", side_effect=[True, True, False, False]), mock.patch.object(os, "getpgid", return_value=43000), mock.patch.object(os, "killpg") as killpg, mock.patch.object(os, "kill") as kill:
            self.assertTrue(launcher.terminate_verified_tree(record))
        kill.assert_called_once_with(43210, signal.SIGTERM)
        killpg.assert_not_called()

    def test_version_validation_executes_all_bundled_mysql_tools(self):
        with mock.patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "MySQL")) as run:
            launcher.validate_mysql_binaries()
        self.assertEqual([call.args[0][0] for call in run.call_args_list], [str(launcher.MYSQLD_EXE), str(launcher.MYSQL_EXE), str(launcher.MYSQLADMIN_EXE)])
        for call in run.call_args_list:
            self.assertIn("--version", call.args[0])
            self.assertNotIn("creationflags", call.kwargs)

    def test_version_validation_rejects_unusable_runtime(self):
        with mock.patch.object(subprocess, "run", side_effect=OSError("Bad CPU type in executable")):
            with self.assertRaisesRegex(RuntimeError, "Bad CPU type"):
                launcher.validate_mysql_binaries()

    def test_failing_configuration_releases_lock_and_restores_signal_handlers(self):
        with tempfile.TemporaryDirectory() as temp, package_paths(Path(temp)):
            signals = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
            with mock.patch.object(launcher, "validate_platform"), mock.patch.object(launcher, "ensure_update_not_in_progress"), mock.patch.object(launcher, "ensure_package_layout"), mock.patch.object(launcher, "load_or_create_config", side_effect=RuntimeError("invalid config")):
                with self.assertRaisesRegex(RuntimeError, "invalid config"):
                    launcher.run_start(no_browser=True)
            mutex = launcher.SingleInstanceMutex()
            try:
                self.assertTrue(mutex.acquire())
            finally:
                mutex.close()
            for number, handler in signals.items():
                self.assertEqual(signal.getsignal(number), handler)

    def test_shutdown_signal_stops_both_services_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as temp, package_paths(Path(temp)):
            app, mysql = mock.Mock(pid=43210), mock.Mock(pid=43211)
            app.poll.return_value = mysql.poll.return_value = None
            app_log, mysql_log = mock.Mock(), mock.Mock()

            def interrupt_wait(_delay):
                signal.getsignal(signal.SIGHUP)(signal.SIGHUP, None)

            patches = {
                "validate_platform": None,
                "ensure_update_not_in_progress": None,
                "ensure_package_layout": None,
                "load_or_create_config": CONFIG.copy(),
                "choose_free_port": 8000,
                "initialize_mysql": None,
                "complete_config_mysql_initialization": None,
                "start_mysql": (mysql, mysql_log),
                "provision_mysql": None,
                "start_app": (app, app_log),
                "wait_for_app": {"database": {"table_ready": True}},
                "build_state": {"packageRoot": temp},
                "mysql_admin_shutdown": True,
            }
            with ExitStack() as stack:
                for name, value in patches.items():
                    stack.enter_context(mock.patch.object(launcher, name, return_value=value))
                stop = stack.enter_context(mock.patch.object(launcher, "stop_process_gracefully"))
                stack.enter_context(mock.patch.object(launcher.time, "sleep", side_effect=interrupt_wait))
                self.assertEqual(launcher.run_start(no_browser=True), 0)
            stop.assert_called_once_with(app)
            mysql.wait.assert_called_once_with(timeout=20)
            app_log.close.assert_called_once()
            mysql_log.close.assert_called_once()
            self.assertFalse(launcher.STATE_FILE.exists())
            mutex = launcher.SingleInstanceMutex()
            try:
                self.assertTrue(mutex.acquire())
            finally:
                mutex.close()

    def test_windows_subprocess_flags_remain_unchanged(self):
        with mock.patch.object(os, "name", "nt"):
            self.assertEqual(launcher.subprocess_options(), {"creationflags": launcher.CREATE_NO_WINDOW})
            self.assertEqual(launcher.subprocess_options(new_group=True), {"creationflags": launcher.CREATE_NEW_PROCESS_GROUP})

    def test_interrupted_verify_returns_failure_and_cleans_up_services(self):
        with tempfile.TemporaryDirectory() as temp, package_paths(Path(temp)):
            app, mysql = mock.Mock(pid=43210), mock.Mock(pid=43211)
            app.poll.return_value = mysql.poll.return_value = None
            patches = {
                "validate_platform": None,
                "ensure_update_not_in_progress": None,
                "ensure_package_layout": None,
                "load_or_create_config": CONFIG.copy(),
                "choose_free_port": 8000,
                "validate_mysql_binaries": None,
                "initialize_mysql": None,
                "complete_config_mysql_initialization": None,
                "start_mysql": (mysql, mock.Mock()),
                "provision_mysql": None,
                "start_app": (app, mock.Mock()),
                "mysql_admin_shutdown": True,
            }
            with ExitStack() as stack:
                for name, value in patches.items():
                    stack.enter_context(mock.patch.object(launcher, name, return_value=value))
                stop = stack.enter_context(mock.patch.object(launcher, "stop_process_gracefully"))
                stack.enter_context(mock.patch.object(launcher, "wait_for_app", side_effect=KeyboardInterrupt))
                self.assertEqual(launcher.run_start(no_browser=True, verify=True), 130)
            stop.assert_called_once_with(app)
            mysql.wait.assert_called_once_with(timeout=20)

    def test_doctor_isolates_host_environment_without_creating_credentials(self):
        with tempfile.TemporaryDirectory() as temp, package_paths(Path(temp)):
            with mock.patch.dict(os.environ, {
                "XP_GACHA_DATA_ROOT": "/another/library",
                "CACHE_DIR": "/another/cache",
                "DATABASE_URL": "mysql+pymysql://external-db/other",
                "MYSQL_USER": "external-user",
                "PYTHONPATH": "/external/python",
            }):
                launcher.prepare_doctor_environment()
                self.assertEqual(os.environ["XP_GACHA_DATA_ROOT"], temp)
                self.assertEqual(os.environ["CACHE_DIR"], str(Path(temp) / "datacache"))
                self.assertIn("127.0.0.1:3307/xp_gacha", os.environ["DATABASE_URL"])
                self.assertEqual(os.environ["MYSQL_USER"], "xp_gacha")
                self.assertNotIn("PYTHONPATH", os.environ)
            self.assertFalse(launcher.PORTABLE_CONFIG_FILE.exists())
            self.assertFalse(launcher.INITIALIZATION_MARKER_FILE.exists())
            self.assertFalse(launcher.MYSQL_DATA_ROOT.exists())

    @unittest.skipUnless(sys.platform == "darwin", "requires macOS libproc")
    def test_real_process_identity_is_precise_and_stable(self):
        first = launcher.process_identity(os.getpid())
        second = launcher.process_identity(os.getpid())
        self.assertIsNotNone(first)
        self.assertEqual(first, second)
        self.assertTrue(Path(first["path"]).is_file())
        self.assertGreater(first["created"], 1_000_000_000_000_000)


if __name__ == "__main__":
    unittest.main()
