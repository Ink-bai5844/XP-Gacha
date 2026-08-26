from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import config


class PreferencesModule:
    def __init__(self) -> None:
        self.path = Path(config.CACHE_DIR) / "ui-preferences.json"
        self._lock = threading.RLock()

    def get(self) -> dict:
        with self._lock:
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                return payload if isinstance(payload, dict) else {}
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                return {}

    def update(self, payload: dict) -> dict:
        with self._lock:
            current = self.get()
            current.update(payload)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, self.path)
            return current
