from __future__ import annotations

import argparse
import atexit
import ctypes
import hashlib
import importlib.util
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from ctypes import wintypes
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PACKAGE_ROOT
RUNTIME_ROOT = PACKAGE_ROOT / "runtime"
PYTHON_HOME = RUNTIME_ROOT / "python"
PYTHON_EXE = PYTHON_HOME / "python.exe"
MYSQL_HOME = RUNTIME_ROOT / "mysql"
MYSQL_BIN = MYSQL_HOME / "bin"
MYSQLD_EXE = MYSQL_BIN / "mysqld.exe"
MYSQL_EXE = MYSQL_BIN / "mysql.exe"
MYSQLADMIN_EXE = MYSQL_BIN / "mysqladmin.exe"

CONFIG_ROOT = PACKAGE_ROOT / "config"
RUN_ROOT = PACKAGE_ROOT / "run"
LOG_ROOT = PACKAGE_ROOT / "logs"
TMP_ROOT = PACKAGE_ROOT / "tmp"
MYSQL_DATA_ROOT = PACKAGE_ROOT / "mysql" / "data"
MYSQL_CONFIG_FILE = CONFIG_ROOT / "mysql.ini"
PORTABLE_CONFIG_FILE = CONFIG_ROOT / "portable.json"
INITIALIZATION_MARKER_FILE = CONFIG_ROOT / ".config-mysql-initialization-pending.json"
STATE_FILE = RUN_ROOT / "state.json"
STOP_REQUEST_FILE = RUN_ROOT / "stop.request"
SETTINGS_FILE = PACKAGE_ROOT / "portable-settings.env"

SCHEMA_VERSION = 1
DEFAULT_APP_PORT = 8000
DEFAULT_MYSQL_PORT = 3307
APP_START_TIMEOUT_SECONDS = 180
MYSQL_START_TIMEOUT_SECONDS = 120

CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000
ERROR_ALREADY_EXISTS = 183
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000
STILL_ACTIVE = 259


def _configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    if os.name == "nt":
        try:
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass


def log(message: str) -> None:
    print(f"[XP-Gacha] {message}", flush=True)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def tail_file(path: Path, max_lines: int = 35) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-max_lines:])


def path_inside(path: str | Path, parent: Path) -> bool:
    try:
        child = Path(path).resolve()
        root = parent.resolve()
        return child == root or root in child.parents
    except (OSError, RuntimeError, ValueError):
        return False


def parse_settings_file(path: Path) -> dict[str, str]:
    allowed = {
        "XP_GACHA_PORT",
        "MYSQL_PORT",
        "XP_GACHA_LIBRARY_PATH",
        "XP_GACHA_IMPORT_MAX_MB",
        "MAX_DISPLAY",
        "LM_STUDIO_API_BASE",
        "LM_STUDIO_API_KEY",
        "LM_STUDIO_MODEL",
        "ONLINE_API_BASE",
        "ONLINE_API_KEY",
        "ONLINE_MODEL",
        "ONLINE_COVER_PROXY",
        "ONLINE_COVER_FETCH_ENABLED",
    }
    result: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return result
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in allowed:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            try:
                decoded = json.loads(value)
                value = decoded if isinstance(decoded, str) else str(decoded)
            except json.JSONDecodeError:
                value = value[1:-1]
        elif len(value) >= 2 and value[0] == value[-1] == "'":
            value = value[1:-1]
        result[key] = value
    return result


def parse_port(value: str | int | None, fallback: int) -> int:
    try:
        port = int(value) if value not in (None, "") else fallback
    except (TypeError, ValueError):
        return fallback
    return port if 1024 <= port <= 65535 else fallback


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def choose_free_port(preferred: int, avoid: set[int] | None = None) -> int:
    avoid = avoid or set()
    candidates = [preferred, *range(preferred + 1, min(preferred + 200, 65536))]
    candidates.extend(range(18000, 18200))
    for port in candidates:
        if port not in avoid and port_is_free(port):
            return port
    raise RuntimeError("未找到可用的本机端口。")


def wait_for_tcp(port: int, process: subprocess.Popen[bytes], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            try:
                sock.connect(("127.0.0.1", port))
                return True
            except OSError:
                time.sleep(0.25)
    return False


def request_json(
    url: str,
    timeout: float = 2.0,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    try:
        body = None
        headers = {
            "Accept": "application/json",
            "User-Agent": "XP-Gacha-Portable/1",
        }
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
            return response_payload if isinstance(response_payload, dict) else None
    except (OSError, ValueError, urllib.error.URLError):
        return None


def fetch_json(url: str, timeout: float = 2.0) -> dict[str, Any] | None:
    return request_json(url, timeout=timeout)


def verify_job_runner(base_url: str, timeout: float = 30.0) -> None:
    started = request_json(
        base_url.rstrip("/") + "/api/jobs",
        timeout=timeout,
        method="POST",
        payload={"scriptId": "cache-delete", "parameters": {"confirm": True, "targets": []}},
    )
    job_id = started.get("id") if isinstance(started, dict) else None
    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError("便携版自检无法启动后台任务。")

    deadline = time.monotonic() + timeout
    last_job: dict[str, Any] | None = started
    while time.monotonic() < deadline:
        current = request_json(
            base_url.rstrip("/") + f"/api/jobs/{urllib.parse.quote(job_id)}?after=0",
            timeout=min(timeout, 5.0),
        )
        if current is not None:
            last_job = current
            status = current.get("status")
            if status == "completed" and current.get("returnCode") == 0:
                return
            if status in {"failed", "cancelled"}:
                lines = current.get("lines") if isinstance(current.get("lines"), list) else []
                detail = "\n".join(str(line) for line in lines[-20:])
                raise RuntimeError(f"便携版后台任务自检失败（状态：{status}）。\n{detail}")
        time.sleep(0.2)
    raise RuntimeError(f"便携版后台任务自检超时，最后状态：{last_job}")


def verify_root_data_paths(status: dict[str, Any]) -> None:
    paths = status.get("paths") if isinstance(status.get("paths"), dict) else {}
    expected = {
        "dataRoot": DATA_ROOT,
        "library": DATA_ROOT / "library",
        "dictionaries": DATA_ROOT / "dictionaries",
    }
    mismatches: list[str] = []
    for name, expected_path in expected.items():
        actual = paths.get(name)
        if not actual or os.path.normcase(str(Path(str(actual)).resolve())) != os.path.normcase(str(expected_path.resolve())):
            mismatches.append(f"{name}: {actual!r}（期望 {expected_path}）")

    capabilities = status.get("searchCapabilities") if isinstance(status.get("searchCapabilities"), dict) else {}
    semantic = capabilities.get("semantic") if isinstance(capabilities.get("semantic"), dict) else {}
    cover = capabilities.get("cover") if isinstance(capabilities.get("cover"), dict) else {}
    semantic_dependencies = semantic.get("dependencies") if isinstance(semantic.get("dependencies"), dict) else {}
    cover_dependencies = cover.get("dependencies") if isinstance(cover.get("dependencies"), dict) else {}
    search_paths = {
        "semantic.model": (semantic_dependencies.get("model"), DATA_ROOT / "models" / "Qwen3-Embedding-0.6B"),
        "semantic.vector": (semantic_dependencies.get("vector"), DATA_ROOT / "manga_vectors" / "manga_vectors_Qwen3.pkl"),
        "cover.model": (cover_dependencies.get("model"), DATA_ROOT / "models" / "clip-vit-base-patch32"),
        "cover.vector": (cover_dependencies.get("vector"), DATA_ROOT / "manga_vectors" / "clip_image_index.pkl"),
    }
    for name, (dependency, expected_path) in search_paths.items():
        actual = dependency.get("path") if isinstance(dependency, dict) else None
        if not actual or os.path.normcase(str(Path(str(actual)).resolve())) != os.path.normcase(str(expected_path.resolve())):
            mismatches.append(f"{name}: {actual!r}（期望 {expected_path}）")
    if mismatches:
        raise RuntimeError("便携版数据路径未指向发行包根目录：\n" + "\n".join(mismatches))


def open_browser(url: str) -> None:
    try:
        webbrowser.open(url, new=2)
    except Exception as exc:
        log(f"无法自动打开浏览器：{exc}")
        log(f"请手动打开 {url}")


def process_identity(pid: int) -> dict[str, Any] | None:
    if os.name != "nt" or pid <= 0:
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid)
    if not handle:
        return None
    try:
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return None

        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel_time = wintypes.FILETIME()
        user_time = wintypes.FILETIME()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            return None
        created = (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime)
        return {"pid": pid, "path": str(Path(buffer.value).resolve()), "created": created}
    finally:
        kernel32.CloseHandle(handle)


def process_matches(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    try:
        pid = int(record["pid"])
        expected_path = str(record["path"])
        expected_created = int(record["created"])
    except (KeyError, TypeError, ValueError):
        return False
    actual = process_identity(pid)
    if not actual:
        return False
    same_path = os.path.normcase(actual["path"]) == os.path.normcase(str(Path(expected_path).resolve()))
    return same_path and int(actual["created"]) == expected_created


def terminate_verified_tree(record: dict[str, Any] | None) -> bool:
    if not process_matches(record):
        return False
    if not path_inside(str(record.get("path", "")), PACKAGE_ROOT):
        return False
    result = subprocess.run(
        ["taskkill", "/PID", str(record["pid"]), "/T", "/F"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=CREATE_NO_WINDOW,
        check=False,
    )
    return result.returncode == 0


class SingleInstanceMutex:
    def __init__(self) -> None:
        self.handle: int | None = None
        digest = hashlib.sha256(str(PACKAGE_ROOT).lower().encode("utf-8")).hexdigest()[:20]
        self.name = f"Local\\XP_Gacha_Portable_{digest}"

    def acquire(self) -> bool:
        if os.name != "nt":
            return True
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        handle = kernel32.CreateMutexW(None, True, self.name)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self.handle = handle
        return ctypes.get_last_error() != ERROR_ALREADY_EXISTS

    def close(self) -> None:
        if self.handle and os.name == "nt":
            ctypes.windll.kernel32.CloseHandle(self.handle)
            self.handle = None


class WindowsJob:
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        pass

    EXTENDED_LIMIT_INFORMATION._fields_ = [
        ("BasicLimitInformation", BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]

    def __init__(self) -> None:
        self.handle: int | None = None
        if os.name != "nt":
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            return
        info = self.EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = self.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            handle,
            self.JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            kernel32.CloseHandle(handle)
            return
        self.handle = handle

    def assign(self, process: subprocess.Popen[bytes]) -> bool:
        if not self.handle or os.name != "nt":
            return False
        return bool(ctypes.windll.kernel32.AssignProcessToJobObject(self.handle, int(process._handle)))

    def close(self) -> None:
        if self.handle and os.name == "nt":
            ctypes.windll.kernel32.CloseHandle(self.handle)
            self.handle = None


def validate_config_mysql_pair() -> None:
    config_exists = PORTABLE_CONFIG_FILE.is_file()
    mysql_initialized = (MYSQL_DATA_ROOT / "mysql").is_dir()
    initialization_pending = INITIALIZATION_MARKER_FILE.is_file()
    if config_exists == mysql_initialized or (config_exists and initialization_pending):
        return

    if config_exists:
        mismatch = "检测到 config/portable.json，但 mysql/data 尚未初始化"
    else:
        mismatch = "检测到已初始化的 mysql/data，但缺少 config/portable.json"
    raise RuntimeError(
        f"{mismatch}。便携版配置凭据与数据库必须来自同一次完整备份，启动已中止。"
        "请关闭旧版后将 config 与 mysql 两个目录成套迁移；"
        "若要全新初始化，请先备份并移走当前残留项。"
    )


def begin_config_mysql_initialization() -> None:
    config_exists = PORTABLE_CONFIG_FILE.is_file()
    mysql_initialized = (MYSQL_DATA_ROOT / "mysql").is_dir()
    if config_exists or mysql_initialized or INITIALIZATION_MARKER_FILE.is_file():
        return
    atomic_write_json(
        INITIALIZATION_MARKER_FILE,
        {
            "schemaVersion": SCHEMA_VERSION,
            "startedAt": time.time(),
        },
    )


def complete_config_mysql_initialization() -> None:
    config_exists = PORTABLE_CONFIG_FILE.is_file()
    mysql_initialized = (MYSQL_DATA_ROOT / "mysql").is_dir()
    if not config_exists or not mysql_initialized:
        raise RuntimeError("便携版配置与 MySQL 尚未同时完成初始化，无法清除首次启动恢复标记。")
    INITIALIZATION_MARKER_FILE.unlink(missing_ok=True)


def ensure_package_layout() -> None:
    required = [
        PYTHON_EXE,
        MYSQLD_EXE,
        MYSQL_EXE,
        MYSQLADMIN_EXE,
        PACKAGE_ROOT / "server" / "main.py",
        PACKAGE_ROOT / "server" / "job_tasks.py",
        PACKAGE_ROOT / "web" / "dist" / "index.html",
    ]
    missing = [str(path.relative_to(PACKAGE_ROOT)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("发行包不完整，缺少：" + "、".join(missing))

    legacy_data_root = PACKAGE_ROOT / "userdata"
    if legacy_data_root.is_dir() and any(legacy_data_root.iterdir()):
        raise RuntimeError(
            "检测到旧版 userdata 数据目录。新版不会读取该目录，以免数据库与凭据错配。"
            "请先按 README 的 v0.2.2 → v0.2.3 迁移表，把 config 与 mysql 成套迁移，"
            "并将其他数据目录移动到发行包根目录。"
        )

    validate_config_mysql_pair()

    directories = [
        CONFIG_ROOT,
        RUN_ROOT,
        LOG_ROOT,
        TMP_ROOT,
        MYSQL_DATA_ROOT.parent,
        PACKAGE_ROOT / "data" / "gallery_info",
        PACKAGE_ROOT / "data" / "gallery_info_no_name",
        PACKAGE_ROOT / "data" / "local_data",
        PACKAGE_ROOT / "datacache" / "imports",
        PACKAGE_ROOT / "b64_cache",
        PACKAGE_ROOT / "b64_tmp",
        PACKAGE_ROOT / "localimgtmp",
        PACKAGE_ROOT / "onlineimgtmp",
        PACKAGE_ROOT / "library",
        PACKAGE_ROOT / "manga_vectors",
        PACKAGE_ROOT / "models",
        PACKAGE_ROOT / "models" / "cache",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    probe = RUN_ROOT / ".write-test"
    try:
        probe.write_text("ok", encoding="ascii")
        probe.unlink()
    except OSError as exc:
        raise RuntimeError(f"发行包目录不可写，请移动到普通可写目录后重试：{exc}") from exc

    dictionary_root = PACKAGE_ROOT / "dictionaries"
    dictionary_root.mkdir(parents=True, exist_ok=True)
    defaults = {
        "STOP_TAGS.txt": "",
        "SEMANTIC_MAP.json": "{}\n",
        "TITLE_STOP_WORDS.txt": "",
        "TITLE_SEMANTIC_MAP.json": "{}\n",
    }
    for name, default_content in defaults.items():
        destination = dictionary_root / name
        if not destination.exists():
            destination.write_text(default_content, encoding="utf-8")


def load_or_create_config(settings: dict[str, str]) -> dict[str, Any]:
    config = read_json(PORTABLE_CONFIG_FILE) or {}
    if int(config.get("schemaVersion", 0) or 0) != SCHEMA_VERSION:
        config = {}
    config.setdefault("schemaVersion", SCHEMA_VERSION)
    config.setdefault("databaseName", "xp_gacha")
    config.setdefault("databaseUser", "xp_gacha")
    config.setdefault("databasePassword", secrets.token_urlsafe(24))
    config.setdefault("rootPassword", secrets.token_urlsafe(28))
    config["preferredAppPort"] = parse_port(
        settings.get("XP_GACHA_PORT"), int(config.get("preferredAppPort", DEFAULT_APP_PORT))
    )
    config["preferredDatabasePort"] = parse_port(
        settings.get("MYSQL_PORT"), int(config.get("preferredDatabasePort", DEFAULT_MYSQL_PORT))
    )
    atomic_write_json(PORTABLE_CONFIG_FILE, config)
    return config


def base_environment(settings: dict[str, str], app_port: int, mysql_port: int, config: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    local_path = os.pathsep.join([str(PYTHON_HOME), str(PYTHON_HOME / "Scripts"), str(MYSQL_BIN)])
    env["PATH"] = local_path + os.pathsep + env.get("PATH", "")
    env.update(
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        XP_GACHA_ENV="production",
        XP_GACHA_HOST="127.0.0.1",
        XP_GACHA_PORT=str(app_port),
        XP_GACHA_DATA_ROOT=str(PACKAGE_ROOT),
        XP_GACHA_FRONTEND_DIST=str(PACKAGE_ROOT / "web" / "dist"),
        XP_GACHA_ALLOW_OPEN_LOCAL="1",
        XP_GACHA_RUNTIME_MODE="portable",
        XP_GACHA_SETTINGS_FILE=str(SETTINGS_FILE),
        ONLINE_IMG_DIR=str(DATA_ROOT / "onlineimgtmp"),
        IMG_CACHE_DIR=str(DATA_ROOT / "localimgtmp"),
        CACHE_DIR=str(DATA_ROOT / "datacache"),
        B64_CACHE_DIR=str(DATA_ROOT / "b64_cache"),
        MODEL_DIR=str(DATA_ROOT / "models"),
        DICTIONARY_DIR=str(DATA_ROOT / "dictionaries"),
        HISTORY_CACHE_FILE=str(DATA_ROOT / "datacache" / "recommendation_history.json"),
        VECTOR_FILE=str(DATA_ROOT / "manga_vectors" / "manga_vectors_Qwen3.pkl"),
        IMG_VECTOR_FILE=str(DATA_ROOT / "manga_vectors" / "clip_image_index.pkl"),
        LOCAL_MODEL_PATH=str(DATA_ROOT / "models" / "Qwen3-Embedding-0.6B"),
        CLIP_MODEL_PATH=str(DATA_ROOT / "models" / "clip-vit-base-patch32"),
        MYSQL_HOST="127.0.0.1",
        MYSQL_PORT=str(mysql_port),
        MYSQL_DATABASE=str(config["databaseName"]),
        MYSQL_USER=str(config["databaseUser"]),
        MYSQL_PASSWORD=str(config["databasePassword"]),
        DATABASE_URL=(
            "mysql+pymysql://"
            f"{urllib.parse.quote_plus(str(config['databaseUser']))}:"
            f"{urllib.parse.quote_plus(str(config['databasePassword']))}"
            f"@127.0.0.1:{mysql_port}/{urllib.parse.quote_plus(str(config['databaseName']))}"
            "?charset=utf8mb4"
        ),
        TEMP=str(TMP_ROOT),
        TMP=str(TMP_ROOT),
        TMPDIR=str(TMP_ROOT),
        HF_HOME=str(PACKAGE_ROOT / "models" / "cache" / "huggingface"),
        HUGGINGFACE_HUB_CACHE=str(PACKAGE_ROOT / "models" / "cache" / "huggingface" / "hub"),
        TRANSFORMERS_CACHE=str(PACKAGE_ROOT / "models" / "cache" / "transformers"),
        TORCH_HOME=str(PACKAGE_ROOT / "models" / "cache" / "torch"),
        XDG_CACHE_HOME=str(PACKAGE_ROOT / "models" / "cache" / "xdg"),
        STREAMLIT_BROWSER_GATHER_USAGE_STATS="false",
        TZ="Asia/Shanghai",
    )

    library_value = settings.get("XP_GACHA_LIBRARY_PATH", "library")
    library_path = Path(library_value).expanduser()
    legacy_library_parts = tuple(part.lower() for part in library_path.parts if part not in {"", "."})
    if not library_path.is_absolute() and legacy_library_parts == ("userdata", "library"):
        log("检测到旧版默认漫画路径，已自动改用根目录 library。")
        library_path = Path("library")
    elif not library_path.is_absolute() and legacy_library_parts[:1] == ("userdata",):
        raise RuntimeError("XP_GACHA_LIBRARY_PATH 不能再指向 userdata；请改为 library 或包外绝对路径。")
    if not library_path.is_absolute():
        library_path = PACKAGE_ROOT / library_path
    library_path.mkdir(parents=True, exist_ok=True)
    env["XP_GACHA_BASE_DIR"] = str(library_path.resolve())

    optional_defaults = {
        "LM_STUDIO_API_BASE": "http://127.0.0.1:1234/v1",
        "LM_STUDIO_API_KEY": "",
        "LM_STUDIO_MODEL": "local-model",
        "ONLINE_API_BASE": "",
        "ONLINE_API_KEY": "",
        "ONLINE_MODEL": "deepseek-v4-flash",
        "ONLINE_COVER_PROXY": "",
        "ONLINE_COVER_FETCH_ENABLED": "1",
        "XP_GACHA_IMPORT_MAX_MB": "1024",
        "MAX_DISPLAY": "500",
    }
    for key, default in optional_defaults.items():
        env[key] = settings.get(key, default)
    return env


def quote_mysql_ini(path: Path) -> str:
    return f'"{path.resolve().as_posix()}"'


def write_mysql_config(mysql_port: int) -> None:
    content = "\n".join(
        [
            "[client]",
            "protocol=tcp",
            "host=127.0.0.1",
            f"port={mysql_port}",
            "default-character-set=utf8mb4",
            "",
            "[mysqld]",
            f"basedir={quote_mysql_ini(MYSQL_HOME)}",
            f"datadir={quote_mysql_ini(MYSQL_DATA_ROOT)}",
            f"tmpdir={quote_mysql_ini(TMP_ROOT)}",
            f"plugin_dir={quote_mysql_ini(MYSQL_HOME / 'lib' / 'plugin')}",
            f"port={mysql_port}",
            "bind-address=127.0.0.1",
            "mysqlx=0",
            "ngram_token_size=2",
            "character-set-server=utf8mb4",
            "collation-server=utf8mb4_0900_ai_ci",
            "max_allowed_packet=1G",
            "innodb_buffer_pool_size=256M",
            "secure-file-priv=NULL",
            "skip-log-bin",
            "log_error_verbosity=2",
            f"log-error={quote_mysql_ini(LOG_ROOT / 'mysql-error.log')}",
            f"pid-file={quote_mysql_ini(RUN_ROOT / 'mysqld.pid')}",
            "",
        ]
    )
    MYSQL_CONFIG_FILE.write_text(content, encoding="utf-8", newline="\n")


def mysql_client_args(config: dict[str, Any], mysql_port: int, password: str | None) -> list[str]:
    args = [
        str(MYSQL_EXE),
        "--no-defaults",
        "--no-login-paths",
        "--protocol=TCP",
        "--host=127.0.0.1",
        f"--port={mysql_port}",
        "--user=root",
        "--connect-timeout=3",
        "--ssl-mode=DISABLED",
        "--get-server-public-key",
    ]
    args.append(f"--password={password}" if password else "--skip-password")
    return args


def initialize_mysql(env: dict[str, str]) -> None:
    initialized = (MYSQL_DATA_ROOT / "mysql").is_dir()
    if initialized:
        return
    if MYSQL_DATA_ROOT.exists() and any(MYSQL_DATA_ROOT.iterdir()):
        raise RuntimeError(
            f"MySQL 数据目录不完整且非空：{MYSQL_DATA_ROOT}。请先备份，再清空该目录后重试。"
        )
    MYSQL_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    init_log = LOG_ROOT / "mysql-initialize.log"
    log("首次启动：正在初始化包内 MySQL（不会注册系统服务）……")
    command = [
        str(MYSQLD_EXE),
        "--no-defaults",
        "--initialize-insecure",
        f"--basedir={MYSQL_HOME}",
        f"--datadir={MYSQL_DATA_ROOT}",
        "--console",
    ]
    with init_log.open("wb") as output:
        try:
            completed = subprocess.run(
                command,
                cwd=MYSQL_HOME,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=300,
                creationflags=CREATE_NO_WINDOW,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("MySQL 初始化超时，请查看 logs/mysql-initialize.log。") from exc
    if completed.returncode != 0 or not (MYSQL_DATA_ROOT / "mysql").is_dir():
        detail = tail_file(init_log)
        raise RuntimeError(f"MySQL 初始化失败（退出码 {completed.returncode}）。\n{detail}")
    log("包内 MySQL 初始化完成。")


def start_mysql(env: dict[str, str], mysql_port: int, job: WindowsJob) -> tuple[subprocess.Popen[bytes], Any]:
    write_mysql_config(mysql_port)
    mysql_log_handle = (LOG_ROOT / "mysql-console.log").open("ab", buffering=0)
    command = [
        str(MYSQLD_EXE),
        f"--defaults-file={MYSQL_CONFIG_FILE}",
        "--console",
        "--no-monitor",
    ]
    process = subprocess.Popen(
        command,
        cwd=MYSQL_HOME,
        env=env,
        stdout=mysql_log_handle,
        stderr=subprocess.STDOUT,
        creationflags=CREATE_NEW_PROCESS_GROUP,
    )
    job.assign(process)
    if not wait_for_tcp(mysql_port, process, MYSQL_START_TIMEOUT_SECONDS):
        detail = tail_file(LOG_ROOT / "mysql-error.log") or tail_file(LOG_ROOT / "mysql-console.log")
        mysql_log_handle.close()
        raise RuntimeError(f"MySQL 启动失败（退出码 {process.poll()}）。\n{detail}")
    return process, mysql_log_handle


def provision_mysql(config: dict[str, Any], mysql_port: int, env: dict[str, str]) -> None:
    database = str(config["databaseName"])
    user = str(config["databaseUser"])
    password = str(config["databasePassword"])
    root_password = str(config["rootPassword"])
    safe_token_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" )
    for value in (database, user, password, root_password):
        if not value or any(char not in safe_token_chars for char in value):
            raise RuntimeError("便携配置中的数据库凭据格式无效。")

    sql = "\n".join(
        [
            f"CREATE DATABASE IF NOT EXISTS `{database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;",
            f"CREATE USER IF NOT EXISTS '{user}'@'127.0.0.1' IDENTIFIED BY '{password}';",
            f"ALTER USER '{user}'@'127.0.0.1' IDENTIFIED BY '{password}';",
            f"GRANT ALL PRIVILEGES ON `{database}`.* TO '{user}'@'127.0.0.1';",
            f"CREATE USER IF NOT EXISTS '{user}'@'localhost' IDENTIFIED BY '{password}';",
            f"ALTER USER '{user}'@'localhost' IDENTIFIED BY '{password}';",
            f"GRANT ALL PRIVILEGES ON `{database}`.* TO '{user}'@'localhost';",
            f"ALTER USER 'root'@'localhost' IDENTIFIED BY '{root_password}';",
            "FLUSH PRIVILEGES;",
        ]
    )
    deadline = time.monotonic() + 75
    last_errors: dict[str, str] = {}
    candidates = [("配置的 root 密码", root_password), ("首次初始化空密码", None)]
    while time.monotonic() < deadline:
        for label, candidate in candidates:
            completed = subprocess.run(
                [*mysql_client_args(config, mysql_port, candidate), f"--execute={sql}"],
                cwd=PACKAGE_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                creationflags=CREATE_NO_WINDOW,
                check=False,
            )
            if completed.returncode == 0:
                return
            last_errors[label] = completed.stdout.strip()
        time.sleep(0.5)
    detail = "\n".join(f"{label}：{error}" for label, error in last_errors.items())
    raise RuntimeError(f"无法创建 XP-Gacha 数据库账户。\n{detail}")


def start_app(env: dict[str, str], app_port: int, job: WindowsJob) -> tuple[subprocess.Popen[bytes], Any]:
    app_log_handle = (LOG_ROOT / "app.log").open("ab", buffering=0)
    command = [
        str(PYTHON_EXE),
        "-X",
        "utf8",
        "-m",
        "uvicorn",
        "server.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(app_port),
        "--no-access-log",
    ]
    process = subprocess.Popen(
        command,
        cwd=PACKAGE_ROOT,
        env=env,
        stdout=app_log_handle,
        stderr=subprocess.STDOUT,
        creationflags=CREATE_NEW_PROCESS_GROUP,
    )
    job.assign(process)
    return process, app_log_handle


def wait_for_app(app_port: int, app_process: subprocess.Popen[bytes]) -> dict[str, Any]:
    health_url = f"http://127.0.0.1:{app_port}/api/health"
    deadline = time.monotonic() + APP_START_TIMEOUT_SECONDS
    last_payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        if app_process.poll() is not None:
            break
        payload = fetch_json(health_url)
        if payload:
            last_payload = payload
            database = payload.get("database") if isinstance(payload.get("database"), dict) else {}
            if payload.get("status") == "ok" and payload.get("frontend") is True and database.get("available") is True:
                return payload
        time.sleep(0.4)
    detail = tail_file(LOG_ROOT / "app.log")
    raise RuntimeError(
        f"应用启动失败（退出码 {app_process.poll()}，最后健康状态 {last_payload}）。\n{detail}"
    )


def mysql_admin_shutdown(config: dict[str, Any], mysql_port: int, env: dict[str, str]) -> bool:
    if not MYSQLADMIN_EXE.is_file():
        return False
    completed = subprocess.run(
        [
            str(MYSQLADMIN_EXE),
            "--no-defaults",
            "--no-login-paths",
            "--protocol=TCP",
            "--host=127.0.0.1",
            f"--port={mysql_port}",
            "--user=root",
            f"--password={config['rootPassword']}",
            "--connect-timeout=3",
            "--ssl-mode=DISABLED",
            "--get-server-public-key",
            "shutdown",
        ],
        cwd=PACKAGE_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=15,
        creationflags=CREATE_NO_WINDOW,
        check=False,
    )
    return completed.returncode == 0


def stop_process_gracefully(process: subprocess.Popen[bytes] | None, timeout: float = 12) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.terminate()
        process.wait(timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.terminate()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass


def existing_instance_url(wait_seconds: float = 30) -> str | None:
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        state = read_json(STATE_FILE)
        if state and os.path.normcase(str(state.get("packageRoot", ""))) == os.path.normcase(str(PACKAGE_ROOT)):
            url = str(state.get("url", ""))
            launcher_record = state.get("launcher") if isinstance(state.get("launcher"), dict) else None
            if url and process_matches(launcher_record):
                health = fetch_json(url.rstrip("/") + "/api/health")
                if health:
                    return url
        time.sleep(0.3)
    return None


def build_state(
    app_port: int,
    mysql_port: int,
    app_process: subprocess.Popen[bytes],
    mysql_process: subprocess.Popen[bytes],
) -> dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "instanceId": hashlib.sha256(str(PACKAGE_ROOT).lower().encode("utf-8")).hexdigest(),
        "packageRoot": str(PACKAGE_ROOT),
        "url": f"http://127.0.0.1:{app_port}",
        "appPort": app_port,
        "databasePort": mysql_port,
        "startedAt": time.time(),
        "launcher": process_identity(os.getpid()),
        "app": process_identity(app_process.pid),
        "database": process_identity(mysql_process.pid),
    }


def run_start(no_browser: bool = False, verify: bool = False) -> int:
    if os.name != "nt":
        raise RuntimeError("此发行包仅支持 Windows x64。")
    ensure_package_layout()
    mutex = SingleInstanceMutex()
    if not mutex.acquire():
        mutex.close()
        url = existing_instance_url()
        if url:
            log(f"当前发行包已在运行：{url}")
            if not no_browser:
                open_browser(url)
            return 0
        raise RuntimeError("另一个启动进程仍在初始化，请稍后再试。")

    job = WindowsJob()
    atexit.register(job.close)
    settings = parse_settings_file(SETTINGS_FILE)
    begin_config_mysql_initialization()
    config = load_or_create_config(settings)
    mysql_port = choose_free_port(parse_port(config.get("preferredDatabasePort"), DEFAULT_MYSQL_PORT))
    app_port = choose_free_port(
        parse_port(config.get("preferredAppPort"), DEFAULT_APP_PORT), avoid={mysql_port}
    )
    env = base_environment(settings, app_port, mysql_port, config)
    STOP_REQUEST_FILE.unlink(missing_ok=True)

    mysql_process: subprocess.Popen[bytes] | None = None
    app_process: subprocess.Popen[bytes] | None = None
    mysql_log_handle = None
    app_log_handle = None
    exit_code = 0
    try:
        initialize_mysql(env)
        complete_config_mysql_initialization()
        log(f"正在启动包内 MySQL：127.0.0.1:{mysql_port}")
        mysql_process, mysql_log_handle = start_mysql(env, mysql_port, job)
        provision_mysql(config, mysql_port, env)
        log(f"正在启动 XP-Gacha：127.0.0.1:{app_port}")
        app_process, app_log_handle = start_app(env, app_port, job)
        health = wait_for_app(app_port, app_process)
        url = f"http://127.0.0.1:{app_port}"
        state = build_state(app_port, mysql_port, app_process, mysql_process)
        atomic_write_json(STATE_FILE, state)
        log(f"启动完成：{url}")
        if health.get("database", {}).get("table_ready") is not True:
            log("当前是空数据库，请在“附录 → 一键导入”上传 CSV/ZIP。")
        if not no_browser:
            open_browser(url)

        if verify:
            meta = fetch_json(url + "/api/meta/options", timeout=30)
            status = fetch_json(url + "/api/system/status", timeout=30)
            if meta is None or status is None:
                raise RuntimeError("便携版自检未能访问核心 API。")
            verify_root_data_paths(status)
            verify_job_runner(url)
            log("便携版首启自检通过。")
            return 0

        log("保持此窗口开启即可使用；按 Ctrl+C 或双击 Stop XP-Gacha.cmd 可停止。")
        while True:
            if STOP_REQUEST_FILE.exists():
                log("收到停止请求。")
                break
            if app_process.poll() is not None:
                exit_code = 1
                raise RuntimeError(f"应用进程意外退出（{app_process.returncode}）。\n{tail_file(LOG_ROOT / 'app.log')}")
            if mysql_process.poll() is not None:
                exit_code = 1
                raise RuntimeError(
                    f"MySQL 进程意外退出（{mysql_process.returncode}）。\n"
                    f"{tail_file(LOG_ROOT / 'mysql-error.log') or tail_file(LOG_ROOT / 'mysql-console.log')}"
                )
            time.sleep(0.5)
    except KeyboardInterrupt:
        log("正在停止……")
    finally:
        STATE_FILE.unlink(missing_ok=True)
        STOP_REQUEST_FILE.unlink(missing_ok=True)
        stop_process_gracefully(app_process)
        if mysql_process is not None and mysql_process.poll() is None:
            if not mysql_admin_shutdown(config, mysql_port, env):
                stop_process_gracefully(mysql_process)
            else:
                try:
                    mysql_process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    stop_process_gracefully(mysql_process)
        if app_log_handle:
            app_log_handle.close()
        if mysql_log_handle:
            mysql_log_handle.close()
        job.close()
        mutex.close()
        log("XP-Gacha 已停止。")
    return exit_code


def run_stop() -> int:
    state = read_json(STATE_FILE)
    if not state:
        STOP_REQUEST_FILE.unlink(missing_ok=True)
        log("当前发行包没有正在运行的实例。")
        return 0
    if os.path.normcase(str(state.get("packageRoot", ""))) != os.path.normcase(str(PACKAGE_ROOT)):
        raise RuntimeError("状态文件不属于当前发行包，已拒绝停止。")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    STOP_REQUEST_FILE.write_text(str(time.time()), encoding="ascii")
    log("已发送停止请求，正在等待安全关闭……")
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        if not STATE_FILE.exists():
            log("XP-Gacha 已停止。")
            return 0
        time.sleep(0.4)

    state = read_json(STATE_FILE) or state
    config = read_json(PORTABLE_CONFIG_FILE) or {}
    settings = parse_settings_file(SETTINGS_FILE)
    mysql_port = parse_port(state.get("databasePort"), DEFAULT_MYSQL_PORT)
    app_port = parse_port(state.get("appPort"), DEFAULT_APP_PORT)
    if config.get("rootPassword"):
        try:
            env = base_environment(settings, app_port, mysql_port, config)
            mysql_admin_shutdown(config, mysql_port, env)
        except Exception:
            pass
    terminated = terminate_verified_tree(state.get("launcher"))
    if not terminated:
        terminate_verified_tree(state.get("app"))
        terminate_verified_tree(state.get("database"))
    time.sleep(1)
    STATE_FILE.unlink(missing_ok=True)
    STOP_REQUEST_FILE.unlink(missing_ok=True)
    log("已清理当前发行包的残留进程。")
    return 0


def run_status() -> int:
    state = read_json(STATE_FILE)
    if not state or not process_matches(state.get("launcher")):
        log("状态：未运行")
        return 1
    url = str(state.get("url", ""))
    health = fetch_json(url.rstrip("/") + "/api/health") if url else None
    if health:
        log(f"状态：运行中，地址 {url}")
        return 0
    log("状态：正在启动或发生异常")
    return 2


def run_doctor() -> int:
    problems: list[str] = []
    try:
        ensure_package_layout()
    except Exception as exc:
        problems.append(str(exc))
    if sys.maxsize <= 2**32:
        problems.append("Python 运行时不是 64 位。")
    required_imports = [
        "fastapi",
        "uvicorn",
        "pandas",
        "numpy",
        "scipy",
        "sqlalchemy",
        "pymysql",
        "streamlit",
        "janome",
        "PIL",
        "torch",
        "transformers",
        "sentence_transformers",
        "curl_cffi",
        "bs4",
        "cloudscraper",
        "cryptography",
    ]
    for name in required_imports:
        try:
            __import__(name)
        except Exception as exc:
            problems.append(f"Python 依赖 {name} 无法加载：{exc}")
    expected_job_module = (PACKAGE_ROOT / "server" / "job_tasks.py").resolve()
    try:
        job_spec = importlib.util.find_spec("server.job_tasks")
        job_origin = Path(job_spec.origin).resolve() if job_spec and job_spec.origin else None
        if job_origin != expected_job_module:
            problems.append(
                f"后台任务模块无法从当前发行包加载：期望 {expected_job_module}，实际 {job_origin}"
            )
    except Exception as exc:
        problems.append(f"后台任务模块无法加载：{exc}")
    if problems:
        log("自检失败：")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    log(f"自检通过：Python {sys.version.split()[0]} x64、MySQL 与前端文件均完整。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XP-Gacha portable Windows supervisor")
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start", help="start MySQL and XP-Gacha")
    start.add_argument("--no-browser", action="store_true")
    verify = subparsers.add_parser("verify", help="run a first-start smoke test and stop")
    verify.add_argument("--no-browser", action="store_true", default=True)
    subparsers.add_parser("stop", help="stop this package instance")
    subparsers.add_parser("status", help="show this package instance status")
    subparsers.add_parser("doctor", help="verify bundled files and imports")
    return parser


def main() -> int:
    _configure_console()
    os.chdir(PACKAGE_ROOT)
    args = build_parser().parse_args()
    try:
        if args.command == "start":
            return run_start(no_browser=args.no_browser)
        if args.command == "verify":
            return run_start(no_browser=True, verify=True)
        if args.command == "stop":
            return run_stop()
        if args.command == "status":
            return run_status()
        if args.command == "doctor":
            return run_doctor()
        return 2
    except Exception as exc:
        log(f"错误：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
