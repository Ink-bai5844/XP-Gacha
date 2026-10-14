from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from tools import clash_proxy_pool as tool


class ClashProxyPoolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.yaml"
        self.run = self.root / "sidecar"
        self.core = self.root / "core with spaces.exe"
        self.core.write_bytes(b"not an executable; tests never launch it")
        self.config = {
            "tun": {"enable": True}, "mixed-port": 7897, "external-controller": "127.0.0.1:9090",
            "secret": "controller-secret-do-not-print", "dns": {"enable": True},
            "proxies": [
                {"name": "🇯🇵 节点 A", "type": "ss", "server": "private.example", "port": 443, "cipher": "aes-128-gcm", "password": "private-node-secret"},
                {"name": "node B", "type": "trojan", "server": "second.example", "port": 443, "password": "second-secret"},
            ],
            "proxy-groups": [{"name": "default", "type": "select", "proxies": ["🇯🇵 节点 A"]}],
            "proxy-providers": {"provider": {"type": "http", "url": "https://private.example/subscription-secret"}},
            "rules": ["MATCH,default"],
        }
        self.save_source()

    def save_source(self) -> None:
        self.source.write_text(yaml.safe_dump(self.config, allow_unicode=True), encoding="utf-8")

    def prepare(self, **kwargs):
        return tool.prepare(self.source, self.run, ["🇯🇵 节点 A", "node B"], **kwargs)

    def state(self) -> dict:
        token = "1" * 32
        return {"schema": 1, "pid": 123456, "launch_id": token, "core": str(self.core.resolve()), "config": str(self.run.resolve() / "mihomo.yaml"), "home": str(self.run.resolve() / f"runtime-{token}"), "created": "123456789"}

    def save_state(self, state: dict | None = None) -> dict:
        value = state or self.state()
        self.run.mkdir(exist_ok=True)
        (self.run / "process.json").write_text(json.dumps(value), encoding="utf-8")
        return value

    def test_default_action_lists_only_names_types_without_creating_run_dir(self):
        output = io.StringIO()
        with redirect_stdout(output):
            result = tool.main(["--source", str(self.source), "--run-dir", str(self.run)])
        self.assertEqual(result, 0)
        nodes = json.loads(output.getvalue())
        self.assertEqual(nodes, [{"name": "🇯🇵 节点 A", "type": "ss"}, {"name": "node B", "type": "trojan"}])
        self.assertNotIn("secret", output.getvalue())
        self.assertFalse(self.run.exists())

    def test_prepare_does_not_mutate_source_and_only_generates_safe_configuration(self):
        before = self.source.read_bytes()
        result = self.prepare(interface="Ethernet")
        self.assertEqual(self.source.read_bytes(), before)
        self.assertTrue(result["contains_private_credentials"])
        config = yaml.safe_load((self.run / "mihomo.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["tun"], {"enable": False})
        self.assertEqual(config["dns"], {"enable": False})
        self.assertFalse(config["allow-lan"])
        self.assertEqual(config["mixed-port"], 0)
        self.assertEqual(config["rules"], ["MATCH,REJECT"])
        self.assertNotIn("external-controller", config)
        self.assertNotIn("secret", config)
        self.assertNotIn("proxy-providers", config)
        self.assertNotIn("proxy-groups", config)
        self.assertEqual([item["proxy"] for item in config["listeners"]], ["🇯🇵 节点 A", "node B"])
        self.assertTrue(all(item["listen"] == "127.0.0.1" and not item["udp"] for item in config["listeners"]))
        self.assertEqual(config["proxies"][0]["password"], "private-node-secret")
        self.assertEqual(config["proxies"][0]["interface-name"], "Ethernet")
        pool = json.loads((self.run / "proxy-pool.json").read_text())
        self.assertEqual(pool, {"proxies": [{"name": "node-1", "url": "http://127.0.0.1:17891"}, {"name": "node-2", "url": "http://127.0.0.1:17892"}], "per_proxy_interval": 3.2, "cooldown_seconds": 60, "connection_cooldown_seconds": 5})
        self.assertNotIn("secret", json.dumps(result))
        tool._verify_prepared(self.run)

    def test_prepare_requires_explicit_valid_nodes_ports_and_directory(self):
        for names in [[], ["default"], ["node B", "node B"], [str(i) for i in range(33)]]:
            with self.assertRaises(tool.PoolToolError):
                tool.prepare(self.source, self.run, names)
        for port in [0, 1023, 65535]:
            with self.assertRaises(tool.PoolToolError):
                self.prepare(base_port=port)
        with self.assertRaises(tool.PoolToolError):
            tool.prepare(self.source, self.root, ["node B"])
        with patch.object(tool, "default_source", return_value=self.source):
            with self.assertRaises(tool.PoolToolError):
                self.prepare()

    def test_unknown_network_protocol_and_missing_dialer_dependencies_are_rejected(self):
        self.config["proxies"][0]["type"] = "tailscale"
        self.save_source()
        with self.assertRaises(tool.PoolToolError):
            self.prepare()
        self.config["proxies"][0]["type"] = "ss"
        self.config["proxies"][0]["dialer-proxy"] = "default"
        self.save_source()
        with self.assertRaises(tool.PoolToolError):
            self.prepare()
        self.config["proxies"][0]["dialer-proxy"] = "node B"
        self.save_source()
        self.prepare()

    def test_pacing_settings_are_configurable_and_validated(self):
        for values in [{"per_proxy_interval": float("nan")}, {"per_proxy_interval": 0}, {"cooldown_seconds": 0}, {"cooldown_seconds": 86401}, {"connection_cooldown_seconds": True}, {"connection_cooldown_seconds": float("inf")}]:
            with self.assertRaises(tool.PoolToolError):
                self.prepare(**values)
        self.prepare(per_proxy_interval=4.0, cooldown_seconds=120, connection_cooldown_seconds=10)
        pool = json.loads((self.run / "proxy-pool.json").read_text())
        self.assertEqual((pool["per_proxy_interval"], pool["cooldown_seconds"], pool["connection_cooldown_seconds"]), (4.0, 120, 10))

    def test_prepare_refuses_overwrite_and_force_keeps_backups(self):
        self.prepare()
        before = (self.run / "mihomo.yaml").read_bytes()
        with self.assertRaises(tool.PoolToolError):
            self.prepare(base_port=18000)
        self.assertEqual((self.run / "mihomo.yaml").read_bytes(), before)
        self.prepare(base_port=18000, force=True)
        backups = list(self.run.glob("mihomo.yaml.*.bak"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), before)
        self.assertEqual(len(list(self.run.glob("*.bak"))), 3)

    def test_prepare_refuses_to_reconfigure_running_sidecar_even_with_force(self):
        self.prepare()
        self.save_state()
        before = (self.run / "mihomo.yaml").read_bytes()
        with patch.object(tool, "_windows_process", return_value={"alive": True}):
            with self.assertRaises(tool.PoolToolError):
                self.prepare(force=True)
        self.assertEqual((self.run / "mihomo.yaml").read_bytes(), before)

    def test_modified_config_or_unsafe_rehashed_configuration_cannot_start(self):
        self.prepare()
        config_path = self.run / "mihomo.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["tun"]["enable"] = True
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with self.assertRaises(tool.PoolToolError):
            tool._verify_prepared(self.run)
        manifest_path = self.run / "prepared.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["config_sha256"] = tool._digest(config_path)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(tool.PoolToolError):
            tool._verify_prepared(self.run)

    def test_exact_process_matching_rejects_reused_pid_other_config_and_extra_args(self):
        state = self.state()
        info = {"exe": state["core"], "created": state["created"], "args": [state["core"], "-d", state["home"], "-f", state["config"]]}
        self.assertTrue(tool._matches(info, state))
        for changed in [dict(info, created="other"), dict(info, exe=str(self.source)), dict(info, args=info["args"] + ["-v"]), dict(info, args=[state["core"], "-d", str(self.root), "-f", state["config"]])]:
            self.assertFalse(tool._matches(changed, state))

    def test_stop_requires_private_launch_home_and_verified_process(self):
        state = self.save_state()
        state["home"] = str(self.root)
        self.save_state(state)
        with patch.object(tool, "_windows_process") as inspect:
            with self.assertRaises(tool.PoolToolError):
                tool.stop(self.run)
            inspect.assert_not_called()
        state = self.save_state(self.state())
        with patch.object(tool, "_windows_process", side_effect=tool.PoolToolError("identity mismatch")) as inspect:
            with self.assertRaises(tool.PoolToolError):
                tool.stop(self.run)
            inspect.assert_called_once_with(state["pid"], terminate=True, expected=state)
        self.assertTrue((self.run / "process.json").exists())

    def test_status_and_stop_never_search_for_or_change_other_clash_processes(self):
        self.assertEqual(tool.status(self.run), {"running": False, "managed": False})
        state = self.save_state()
        with patch.object(tool, "_windows_process", return_value=None) as inspect:
            self.assertFalse(tool.status(self.run)["running"])
            self.assertTrue(tool.stop(self.run)["already_stopped"])
            self.assertEqual(inspect.call_count, 2)
        with patch.object(tool, "_windows_process", return_value={"verified": True}) as inspect:
            self.assertFalse(tool.stop(self.run)["already_stopped"])
            inspect.assert_called_once_with(state["pid"], terminate=True, expected=state)

    def test_start_validates_first_and_launches_only_isolated_no_window_core(self):
        self.prepare()
        child = MagicMock(pid=123456)
        child.poll.return_value = None
        with patch.object(tool, "_require_windows"), patch.object(tool, "_ports_available"), patch.object(tool, "_verify_listener_ownership") as ownership, patch.object(tool.subprocess, "run", return_value=MagicMock(returncode=0)) as validate, patch.object(tool.subprocess, "Popen", return_value=child) as launch, patch.object(tool.socket, "create_connection", return_value=MagicMock()), patch.object(tool, "_windows_process") as inspect:
            inspect.side_effect = lambda _pid: {"exe": str(self.core.resolve()), "created": "123456789", "args": list(launch.call_args.args[0])}
            result = tool.start(self.run, self.core)
        self.assertTrue(result["running"])
        validation_command = validate.call_args.args[0]
        command = launch.call_args.args[0]
        self.assertEqual(validation_command, command + ["-t"])
        self.assertEqual(command[0], str(self.core.resolve()))
        self.assertEqual(command[1], "-d")
        self.assertTrue(Path(command[2]).name.startswith("runtime-"))
        self.assertEqual(Path(command[2]).parent, self.run.resolve())
        self.assertEqual(command[3:], ["-f", str(self.run.resolve() / "mihomo.yaml")])
        self.assertEqual(launch.call_args.kwargs["creationflags"], getattr(tool.subprocess, "CREATE_NO_WINDOW", 0))
        ownership.assert_called_once_with(child.pid, {17891, 17892})
        self.assertEqual(tool._load_state(self.run)["pid"], child.pid)
        child.terminate.assert_not_called()

    def test_failed_validation_does_not_launch_any_process(self):
        self.prepare()
        with patch.object(tool, "_require_windows"), patch.object(tool, "_ports_available"), patch.object(tool.subprocess, "run", return_value=MagicMock(returncode=1)), patch.object(tool.subprocess, "Popen") as launch:
            with self.assertRaises(tool.PoolToolError):
                tool.start(self.run, self.core)
        launch.assert_not_called()

    def test_failed_postlaunch_identity_only_stops_retained_child_handle(self):
        self.prepare()
        child = MagicMock(pid=123456)
        child.poll.return_value = None
        with patch.object(tool, "_require_windows"), patch.object(tool, "_ports_available"), patch.object(tool.subprocess, "run", return_value=MagicMock(returncode=0)), patch.object(tool.subprocess, "Popen", return_value=child), patch.object(tool, "_windows_process", side_effect=tool.PoolToolError("unverifiable")):
            with self.assertRaises(tool.PoolToolError):
                tool.start(self.run, self.core)
        child.terminate.assert_called_once()
        child.wait.assert_called_once_with(timeout=5)

    def test_yaml_errors_and_unexpected_errors_do_not_echo_credentials(self):
        self.source.write_text("password: private-node-secret\ninvalid: [", encoding="utf-8")
        output = io.StringIO()
        with redirect_stderr(output):
            self.assertEqual(tool.main(["list", "--source", str(self.source)]), 1)
        self.assertNotIn("private-node-secret", output.getvalue())
        with patch.object(tool, "list_nodes", side_effect=ValueError("password=private-node-secret")), redirect_stderr(output):
            self.assertEqual(tool.main(["list"]), 1)
        self.assertNotIn("private-node-secret", output.getvalue())


if __name__ == "__main__":
    unittest.main()
