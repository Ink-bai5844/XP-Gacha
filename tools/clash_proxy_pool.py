"""Create an isolated, loopback-only Mihomo sidecar without changing Clash Verge.

The default action only lists node names/types. Generated YAML contains node
credentials: keep the entire run directory private and out of version control.
This tool never calls the existing controller or changes system proxy/TUN state.
Process management deliberately requires verifiable Windows process identity.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_get.collection_lock import collection_output_lock

DEFAULT_RUN_DIR = PROJECT_ROOT / "run" / "clash-pool"
NODE_TYPES = {"ss", "ssr", "vmess", "vless", "trojan", "hysteria", "hysteria2", "tuic", "snell", "socks5", "http", "anytls"}
GENERATED_FILES = ("mihomo.yaml", "proxy-pool.json", "prepared.json")


class PoolToolError(RuntimeError):
    """A safe, credential-free explanation suitable for console output."""


def default_source() -> Path:
    roaming = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
    return roaming / "io.github.clash-verge-rev.clash-verge-rev" / "clash-verge.yaml"


def _yaml():
    try:
        import yaml
    except ImportError as exc:
        raise PoolToolError("缺少 PyYAML，请先安装项目依赖。") from exc
    return yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = _yaml().safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError) as exc:
        raise PoolToolError("无法读取指定 YAML 文件。") from exc
    except _yaml().YAMLError as exc:
        # YAML parser exceptions often echo a source line containing a password.
        raise PoolToolError("YAML 格式无效；为保护节点凭据，不显示原始解析错误。") from exc
    if not isinstance(value, dict):
        raise PoolToolError("需要包含 proxies 列表的完整 Clash YAML。")
    return value


def list_nodes(source: Path) -> list[dict[str, str]]:
    return [{"name": node["name"], "type": node["type"]} for node in _nodes(source)]


def _nodes(source: Path) -> list[dict[str, Any]]:
    nodes = _load_yaml(source).get("proxies")
    if not isinstance(nodes, list):
        raise PoolToolError("源配置没有内联 proxies；请提供已展开节点的完整配置，工具不会下载订阅。")
    names: set[str] = set()
    result = []
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("name"), str) or not isinstance(node.get("type"), str):
            raise PoolToolError("源配置包含无效节点定义。")
        if not node["name"] or node["name"] in names:
            raise PoolToolError("源配置节点名称为空或重复，无法安全固定出口。")
        names.add(node["name"])
        result.append(node)
    return result


def _root(path: Path, source: Path | None = None) -> Path:
    root = path.resolve()
    clash_dir = default_source().parent.resolve()
    if root == clash_dir or clash_dir in root.parents or root == PROJECT_ROOT.resolve() or root == Path(root.anchor):
        raise PoolToolError("独立运行目录不能是工作区根目录、磁盘根目录或现有 Clash 配置目录。")
    if source is not None and root == source.resolve().parent:
        raise PoolToolError("独立运行目录不能直接使用源配置所在目录。")
    return root


@contextmanager
def _locked(root: Path):
    with collection_output_lock(SimpleNamespace(output_csv=root / "manager", state_file=None, output_dir=None)):
        yield


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PoolToolError("独立代理池状态文件缺失或损坏；不会操作任何现有 Clash 进程。") from exc
    if not isinstance(value, dict):
        raise PoolToolError("独立代理池状态格式无效。")
    return value


def _load_state(root: Path) -> dict[str, Any] | None:
    path = root / "process.json"
    if not path.exists():
        return None
    value = _read_json(path)
    token = value.get("launch_id")
    if (
        value.get("schema") != 1
        or not isinstance(token, str)
        or re.fullmatch(r"[0-9a-f]{32}", token) is None
        or type(value.get("pid")) is not int
        or value["pid"] <= 0
        or not isinstance(value.get("created"), str)
        or not value["created"].isdigit()
        or not isinstance(value.get("core"), str)
        or not isinstance(value.get("home"), str)
        or not isinstance(value.get("config"), str)
    ):
        raise PoolToolError("独立代理池进程身份记录无效，拒绝操作。")
    if Path(value["config"]).resolve() != root / "mihomo.yaml" or Path(value["home"]).resolve() != root / f"runtime-{token}":
        raise PoolToolError("进程记录不属于此独立运行目录，拒绝操作。")
    return value


def _same_path(first: str, second: str) -> bool:
    return os.path.normcase(str(Path(first).resolve())) == os.path.normcase(str(Path(second).resolve()))


def _matches(info: dict[str, Any], expected: dict[str, Any]) -> bool:
    # Exact argument vector: never substring-match a shared Clash executable.
    arguments = info.get("args", [])
    return (
        str(info.get("created")) == str(expected.get("created"))
        and _same_path(info.get("exe", ""), expected["core"])
        and len(arguments) == 5
        and _same_path(arguments[0], expected["core"])
        and arguments[1] == "-d"
        and _same_path(arguments[2], expected["home"])
        and arguments[3] == "-f"
        and _same_path(arguments[4], expected["config"])
    )


def _require_windows() -> None:
    if os.name != "nt":
        raise PoolToolError("自动 start/status/stop 当前仅支持 Windows；其他平台可使用 prepare 后手动管理独立核心。")


def _windows_process(pid: int, *, terminate: bool = False, expected: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Inspect and optionally terminate the SAME process handle, avoiding PID reuse."""
    _require_windows()
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    rights = 0x1000 | 0x100000 | (0x0001 if terminate else 0)
    handle = kernel.OpenProcess(rights, False, pid)
    if not handle:
        if ctypes.get_last_error() in {87, 1168}:
            return None
        raise PoolToolError("无法核实目标进程身份（可能权限不足）；不会结束任何进程。")
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise PoolToolError("无法查询独立核心状态，拒绝操作。")
        if code.value != 259:
            return None
        size = wintypes.DWORD(32768)
        executable = ctypes.create_unicode_buffer(size.value)
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.QueryFullProcessImageNameW(handle, 0, executable, ctypes.byref(size)) or not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            raise PoolToolError("无法核实核心路径或启动时间，拒绝操作。")
        # Only a numeric PID enters this fixed script; no paths or secrets are
        # interpolated into PowerShell. CommandLine contains only our -d/-f paths.
        script = f'[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); $p = Get-CimInstance Win32_Process -Filter "ProcessId={int(pid)}" -ErrorAction Stop; if ($null -eq $p) {{ exit 3 }}; @{{ CommandLine=$p.CommandLine }} | ConvertTo-Json -Compress'
        try:
            result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            command = json.loads(result.stdout).get("CommandLine") if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            raise PoolToolError("无法读取进程启动参数，拒绝操作；原 Clash 不受影响。") from exc
        if not isinstance(command, str) or not command:
            raise PoolToolError("无法核实进程启动参数，拒绝操作。")
        shell = ctypes.WinDLL("shell32", use_last_error=True)
        shell.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
        shell.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
        kernel.LocalFree.argtypes = [wintypes.HLOCAL]
        kernel.LocalFree.restype = wintypes.HLOCAL
        count = ctypes.c_int()
        argv = shell.CommandLineToArgvW(command, ctypes.byref(count))
        if not argv:
            raise PoolToolError("无法解析进程启动参数，拒绝操作。")
        try:
            arguments = [argv[index] for index in range(count.value)]
        finally:
            kernel.LocalFree(ctypes.cast(argv, wintypes.HLOCAL))
        info = {"exe": executable.value, "created": str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime), "args": arguments}
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:
            return None
        if expected is not None and not _matches(info, expected):
            raise PoolToolError("PID 已重用或目标不是本工具启动的独立核心；拒绝结束进程。")
        if terminate:
            if expected is None:
                raise PoolToolError("缺少可验证的启动身份，拒绝结束进程。")
            if not kernel.TerminateProcess(handle, 0) or kernel.WaitForSingleObject(handle, 5000) != 0:
                raise PoolToolError("独立核心停止未确认；保留身份记录，请检查后重试。")
        return info
    finally:
        kernel.CloseHandle(handle)


def _require_not_running(root: Path) -> None:
    state = _load_state(root)
    if state is not None and _windows_process(state["pid"], expected=state) is not None:
        raise PoolToolError("此目录的独立核心仍在运行；请先停止采集，再 stop 后重新 prepare/start。")


def prepare(source: Path, run_dir: Path, names: list[str], *, base_port: int = 17891, interface: str | None = None, force: bool = False, per_proxy_interval: float = 3.2, cooldown_seconds: float = 60, connection_cooldown_seconds: float = 5) -> dict[str, Any]:
    if not names or len(names) > 32 or len(set(names)) != len(names):
        raise PoolToolError("prepare 必须显式提供 1–32 个不重复的 --node；不会默认启用所有节点。")
    available = {node["name"]: node for node in _nodes(source)}
    if any(name not in available for name in names):
        raise PoolToolError("指定节点不存在；请先 list 确认名称。")
    selected = [copy.deepcopy(available[name]) for name in names]
    if any(node["type"] not in NODE_TYPES for node in selected):
        raise PoolToolError("所选项不是支持的普通代理节点；不启用策略组或可创建系统网络的协议。")
    if any(node.get("dialer-proxy") not in {None, "", "DIRECT", *names} for node in selected):
        raise PoolToolError("所选节点依赖未选中的 dialer-proxy；请显式选择依赖节点或改选独立节点。")
    if not 1024 <= base_port <= 65535 - len(selected) + 1:
        raise PoolToolError("端口必须位于 1024–65535，且连续范围足够容纳全部节点。")
    if interface is not None and (not interface.strip() or any(ord(char) < 32 for char in interface)):
        raise PoolToolError("物理网卡名称无效。")
    for name, value, minimum, maximum in (("per-proxy-interval", per_proxy_interval, 0.1, 3600), ("cooldown-seconds", cooldown_seconds, 1, 86400), ("connection-cooldown-seconds", connection_cooldown_seconds, 0.1, 3600)):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not minimum <= value <= maximum:
            raise PoolToolError(f"{name} 必须在 {minimum:g}–{maximum:g} 秒之间。")
    root = _root(run_dir, source)
    config: dict[str, Any] = {
        "mode": "rule", "allow-lan": False, "bind-address": "127.0.0.1", "log-level": "warning", "ipv6": False,
        "port": 0, "socks-port": 0, "mixed-port": 0, "redir-port": 0, "tproxy-port": 0,
        "tun": {"enable": False}, "dns": {"enable": False}, "ntp": {"enable": False},
        "profile": {"store-selected": False, "store-fake-ip": False}, "geo-auto-update": False,
        "proxies": selected, "listeners": [], "rules": ["MATCH,REJECT"],
    }
    if interface:
        config["interface-name"] = interface
        for node in selected:
            node["interface-name"] = interface
    pool = {"proxies": [], "per_proxy_interval": per_proxy_interval, "cooldown_seconds": cooldown_seconds, "connection_cooldown_seconds": connection_cooldown_seconds}
    for index, node in enumerate(selected):
        port = base_port + index
        config["listeners"].append({"name": f"collector-{index + 1}", "type": "mixed", "listen": "127.0.0.1", "port": port, "udp": False, "users": [], "proxy": node["name"]})
        pool["proxies"].append({"name": f"node-{index + 1}", "url": f"http://127.0.0.1:{port}"})
    with _locked(root):
        _require_not_running(root)
        existing = [root / name for name in GENERATED_FILES if (root / name).exists()]
        if existing and not force:
            raise PoolToolError("生成文件已存在；拒绝覆盖。需要备份后重建时显式使用 --force。")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        for path in existing:
            shutil.copy2(path, path.with_name(path.name + f".{stamp}.bak"))
        content = _yaml().safe_dump(config, allow_unicode=True, sort_keys=False).encode("utf-8")
        _atomic_write(root / "mihomo.yaml", content)
        _atomic_write(root / "proxy-pool.json", _json_bytes(pool))
        _atomic_write(root / "prepared.json", _json_bytes({"schema": 1, "created_at": stamp, "config_sha256": _digest(root / "mihomo.yaml"), "pool_sha256": _digest(root / "proxy-pool.json"), "nodes": [{"name": f"node-{index + 1}", "source_name": name} for index, name in enumerate(names)]}))
    return {"prepared": True, "nodes": len(selected), "config": str(root / "mihomo.yaml"), "pool": str(root / "proxy-pool.json"), "contains_private_credentials": True}


def _verify_prepared(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "prepared.json")
    try:
        valid = manifest.get("schema") == 1 and manifest.get("config_sha256") == _digest(root / "mihomo.yaml") and manifest.get("pool_sha256") == _digest(root / "proxy-pool.json")
    except OSError:
        valid = False
    if not valid:
        raise PoolToolError("生成文件缺失或被修改；拒绝启动，请重新 prepare。")
    config = _load_yaml(root / "mihomo.yaml")
    # Do not trust only a hash manifest when deciding whether to launch a core.
    allowed_keys = {"mode", "allow-lan", "bind-address", "log-level", "ipv6", "port", "socks-port", "mixed-port", "redir-port", "tproxy-port", "tun", "dns", "ntp", "profile", "geo-auto-update", "proxies", "listeners", "rules", "interface-name"}
    if set(config) - allowed_keys or config.get("ntp") != {"enable": False} or config.get("geo-auto-update") is not False:
        raise PoolToolError("独立配置包含未授权的全局配置项，拒绝启动。")
    if config.get("tun") != {"enable": False} or config.get("dns") != {"enable": False} or config.get("allow-lan") is not False:
        raise PoolToolError("独立配置不符合关闭 TUN/DNS/LAN 的安全约束，拒绝启动。")
    if any(key.startswith("external-controller") or key in {"proxy-providers", "rule-providers", "tunnels"} for key in config):
        raise PoolToolError("独立配置包含控制接口或外部资源，拒绝启动。")
    if any(config.get(key, 0) != 0 for key in ("port", "socks-port", "mixed-port", "redir-port", "tproxy-port")):
        raise PoolToolError("独立配置含有额外全局端口，拒绝启动。")
    listeners = config.get("listeners")
    if not isinstance(listeners, list) or not listeners or any(not isinstance(item, dict) or item.get("type") != "mixed" or item.get("listen") != "127.0.0.1" or item.get("udp") is not False for item in listeners):
        raise PoolToolError("独立配置的监听入口不安全，拒绝启动。")
    return config


def _verify_listener_ownership(pid: int, ports: set[int]) -> None:
    script = f'[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); @(Get-NetTCPConnection -State Listen -OwningProcess {int(pid)} -ErrorAction Stop | Where-Object LocalAddress -eq "127.0.0.1" | Select-Object -ExpandProperty LocalPort) | ConvertTo-Json -Compress'
    try:
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        values = json.loads(result.stdout) if result.returncode == 0 else []
        owned = {values} if type(values) is int else set(values or [])
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, TypeError) as exc:
        raise PoolToolError("无法核实入口端口归属；停止刚启动的独立核心，不接管其他程序端口。") from exc
    if not ports.issubset(owned):
        raise PoolToolError("入口端口不全部属于新启动的独立核心；拒绝使用其他程序的代理入口。")


def detect_core(requested: Path | None = None) -> Path:
    candidates = [requested] if requested else [Path("D:/Clash Verge/verge-mihomo.exe"), Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Clash Verge" / "verge-mihomo.exe"]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate.resolve()
    raise PoolToolError("找不到 Mihomo 核心；请用 --core 指定已安装的核心文件，本工具不会下载安装。")


def _ports_available(config: dict[str, Any]) -> None:
    held = []
    try:
        for item in config["listeners"]:
            handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            held.append(handle)
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                handle.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            handle.bind(("127.0.0.1", int(item["port"])))
    except (OSError, ValueError, KeyError) as exc:
        raise PoolToolError("独立入口端口已占用或无效；不会停止任何占用进程，请换端口重新 prepare。") from exc
    finally:
        for handle in held:
            handle.close()


def start(run_dir: Path, core: Path | None = None) -> dict[str, Any]:
    _require_windows()
    root = _root(run_dir)
    with _locked(root):
        _require_not_running(root)
        config = _verify_prepared(root)
        executable = detect_core(core)
        _ports_available(config)
        token = uuid.uuid4().hex
        home = root / f"runtime-{token}"
        home.mkdir(mode=0o700)
        command = [str(executable), "-d", str(home), "-f", str(root / "mihomo.yaml")]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            validation = subprocess.run(command + ["-t"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30, creationflags=flags)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PoolToolError("独立配置校验无法完成；没有启动核心，也没有修改现有 Clash。") from exc
        if validation.returncode != 0:
            raise PoolToolError("Mihomo 拒绝了独立配置；为保护凭据未打印内核解析信息，请核对核心版本和节点依赖。")
        try:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags, cwd=home)
        except OSError as exc:
            raise PoolToolError("无法启动独立核心；现有 Clash 未修改。") from exc
        try:
            info = _windows_process(child.pid)
            if info is None or child.poll() is not None:
                raise PoolToolError("独立核心启动后退出；现有 Clash 未修改。")
            state = {"schema": 1, "pid": child.pid, "launch_id": token, "core": str(executable), "home": str(home), "config": str(root / "mihomo.yaml"), "created": info["created"]}
            if not _matches(info, state):
                raise PoolToolError("新启动进程身份无法核实；不会接管其他进程。")
            _atomic_write(root / "process.json", _json_bytes(state))
            deadline = time.monotonic() + 5
            while True:
                if child.poll() is not None:
                    raise PoolToolError("独立核心未成功监听；现有 Clash 未修改。")
                ready = True
                for listener in config["listeners"]:
                    try:
                        with socket.create_connection(("127.0.0.1", listener["port"]), timeout=0.2):
                            pass
                    except OSError:
                        ready = False
                        break
                if ready:
                    _verify_listener_ownership(child.pid, {item["port"] for item in config["listeners"]})
                    break
                if time.monotonic() >= deadline:
                    raise PoolToolError("独立核心监听就绪超时；现有 Clash 未修改。")
                time.sleep(0.1)
        except BaseException:
            # Popen's retained Windows process handle identifies only the child
            # this call just created; it cannot target an unrelated reused PID.
            if child.poll() is None:
                try:
                    child.terminate()
                    child.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            raise
        return {"running": True, "pid": child.pid, "pool": str(root / "proxy-pool.json"), "nodes": len(config["listeners"])}


def status(run_dir: Path) -> dict[str, Any]:
    root = _root(run_dir)
    state = _load_state(root)
    if state is None:
        return {"running": False, "managed": False}
    info = _windows_process(state["pid"], expected=state)
    return {"running": info is not None, "managed": True, "pid": state["pid"]}


def stop(run_dir: Path) -> dict[str, Any]:
    root = _root(run_dir)
    with _locked(root):
        state = _load_state(root)
        if state is None:
            return {"stopped": True, "already_stopped": True}
        info = _windows_process(state["pid"], terminate=True, expected=state)
        # Keep the identity record for auditing; a later start replaces it only
        # after confirming the recorded process is gone.
        return {"stopped": True, "already_stopped": info is None, "pid": state["pid"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("list", "prepare", "start", "status", "stop"), default="list")
    parser.add_argument("--source", type=Path, default=default_source())
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--node", action="append", default=[])
    parser.add_argument("--base-port", type=int, default=17891)
    parser.add_argument("--per-proxy-interval", type=float, default=3.2)
    parser.add_argument("--cooldown-seconds", type=float, default=60)
    parser.add_argument("--connection-cooldown-seconds", type=float, default=5)
    parser.add_argument("--interface")
    parser.add_argument("--core", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "list":
            result = list_nodes(args.source)
        elif args.action == "prepare":
            result = prepare(args.source, args.run_dir, args.node, base_port=args.base_port, interface=args.interface, force=args.force, per_proxy_interval=args.per_proxy_interval, cooldown_seconds=args.cooldown_seconds, connection_cooldown_seconds=args.connection_cooldown_seconds)
        elif args.action == "start":
            result = start(args.run_dir, args.core)
        elif args.action == "status":
            result = status(args.run_dir)
        else:
            result = stop(args.run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except PoolToolError as exc:
        print(f"[代理池] {exc}", file=sys.stderr)
    except Exception:
        # Avoid accidentally echoing YAML/URI passwords from third-party errors.
        print("[代理池] 操作失败；为保护凭据未显示原始异常，现有 Clash 配置未被修改。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
