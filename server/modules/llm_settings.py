from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from pathlib import Path


LLM_ENV_DEFAULTS = {
    "LM_STUDIO_API_BASE": "http://127.0.0.1:1234/v1",
    "LM_STUDIO_API_KEY": "",
    "LM_STUDIO_MODEL": "local-model",
    "ONLINE_API_BASE": "",
    "ONLINE_API_KEY": "",
    "ONLINE_MODEL": "deepseek-v4-flash",
}
LLM_ENV_KEYS = tuple(LLM_ENV_DEFAULTS)
_ENV_LINE = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*=")
_SETTINGS_LOCK = threading.RLock()


def _default_value(key: str) -> str:
    if key == "LM_STUDIO_API_BASE" and os.getenv("XP_GACHA_RUNTIME_MODE", "").strip().lower() == "docker":
        return "http://host.docker.internal:1234/v1"
    return LLM_ENV_DEFAULTS[key]


def runtime_llm_connection(api_mode: str) -> tuple[str, str, str]:
    """Return one provider snapshot so concurrent saves cannot mix credentials."""

    if api_mode == "本地 (LM Studio)":
        keys = ("LM_STUDIO_API_BASE", "LM_STUDIO_API_KEY", "LM_STUDIO_MODEL")
    else:
        keys = ("ONLINE_API_BASE", "ONLINE_API_KEY", "ONLINE_MODEL")
    with _SETTINGS_LOCK:
        file_values = _read_env_file(LLMSettingsModule().settings_file())
        api_base, api_key, model = (
            os.getenv(key, file_values.get(key, _default_value(key))).strip()
            for key in keys
        )
        return api_base.rstrip("/"), api_key, model


def _decode_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, str) else str(decoded)
        except json.JSONDecodeError:
            return value[1:-1]
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1]
    return value


def _read_env_file(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (FileNotFoundError, IsADirectoryError):
        return {}
    values: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in LLM_ENV_DEFAULTS:
            values[key] = _decode_env_value(value)
    return values


def _render_updated_env(existing: str, updates: dict[str, str]) -> str:
    lines = existing.splitlines()
    seen: set[str] = set()
    rendered: list[str] = []
    for line in lines:
        match = _ENV_LINE.match(line)
        key = match.group(1) if match else ""
        if key in updates:
            if key not in seen:
                rendered.append(f"{key}={json.dumps(updates[key], ensure_ascii=False)}")
                seen.add(key)
        else:
            rendered.append(line)
    missing = [key for key in LLM_ENV_KEYS if key in updates and key not in seen]
    if missing:
        if rendered and rendered[-1].strip():
            rendered.append("")
        rendered.append("# LLM settings saved from the XP-Gacha assistant page.")
        rendered.extend(f"{key}={json.dumps(updates[key], ensure_ascii=False)}" for key in missing)
    return "\n".join(rendered).rstrip() + "\n"


def _write_env_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.replace(temporary, path)
        except OSError:
            # Docker bind-mounted files cannot be replaced as directory entries.
            # Keep the same mount point and flush the complete new content in place.
            with path.open("w", encoding="utf-8", newline="\n") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
    finally:
        temporary.unlink(missing_ok=True)
    if os.name != "nt":
        try:
            path.chmod(0o600)
        except OSError:
            pass


class LLMSettingsModule:
    def __init__(self, settings_file: str | Path | None = None) -> None:
        self._settings_file = Path(settings_file).resolve() if settings_file else None

    def settings_file(self) -> Path:
        configured = os.getenv("XP_GACHA_SETTINGS_FILE", "").strip()
        if self._settings_file:
            return self._settings_file
        if configured:
            return Path(configured).expanduser().resolve()
        return (Path(__file__).resolve().parents[2] / ".env").resolve()

    def _active_value(self, key: str, file_values: dict[str, str]) -> str:
        return os.getenv(key, file_values.get(key, _default_value(key))).strip()

    def status(self) -> dict:
        with _SETTINGS_LOCK:
            path = self.settings_file()
            file_values = _read_env_file(path)
            local_key = self._active_value("LM_STUDIO_API_KEY", file_values)
            online_key = self._active_value("ONLINE_API_KEY", file_values)
            runtime_mode = os.getenv("XP_GACHA_RUNTIME_MODE", "source").strip().lower() or "source"
            writable = (
                path.is_file() and os.access(path, os.W_OK)
                or not path.exists() and os.access(path.parent, os.W_OK)
            )
            return {
                "local": {
                    "apiBase": self._active_value("LM_STUDIO_API_BASE", file_values).rstrip("/"),
                    "model": self._active_value("LM_STUDIO_MODEL", file_values),
                    "apiKeyConfigured": bool(local_key),
                },
                "online": {
                    "apiBase": self._active_value("ONLINE_API_BASE", file_values).rstrip("/"),
                    "model": self._active_value("ONLINE_MODEL", file_values),
                    "apiKeyConfigured": bool(online_key),
                },
                "persistence": {
                    "runtimeMode": runtime_mode,
                    "fileName": path.name,
                    "writable": writable,
                    "restartRequired": False,
                },
            }

    def update(
        self,
        *,
        local_api_base: str,
        local_model: str,
        local_api_key: str | None,
        clear_local_api_key: bool,
        online_api_base: str,
        online_model: str,
        online_api_key: str | None,
        clear_online_api_key: bool,
    ) -> dict:
        if local_api_key is not None and clear_local_api_key:
            raise ValueError("不能同时设置并清除本地 API Key")
        if online_api_key is not None and clear_online_api_key:
            raise ValueError("不能同时设置并清除线上 API Key")
        path = self.settings_file()
        with _SETTINGS_LOCK:
            try:
                existing = path.read_text(encoding="utf-8-sig")
            except FileNotFoundError:
                existing = ""
            updates = {
                "LM_STUDIO_API_BASE": local_api_base.rstrip("/"),
                "LM_STUDIO_MODEL": local_model,
                "ONLINE_API_BASE": online_api_base.rstrip("/"),
                "ONLINE_MODEL": online_model,
            }
            if local_api_key is not None or clear_local_api_key:
                updates["LM_STUDIO_API_KEY"] = local_api_key or ""
            if online_api_key is not None or clear_online_api_key:
                updates["ONLINE_API_KEY"] = online_api_key or ""
            _write_env_file(path, _render_updated_env(existing, updates))
            os.environ.update(updates)
        return self.status()
