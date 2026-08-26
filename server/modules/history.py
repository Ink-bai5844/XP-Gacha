from __future__ import annotations

import hashlib
import json
import threading

from server.modules.library import LibraryModule
from utils_history import (
    clear_history_entries,
    load_history_entries,
    record_recommendation_history,
    save_history_entries,
)


class HistoryModule:
    def __init__(self, library: LibraryModule) -> None:
        self.library = library
        self._lock = threading.RLock()

    @staticmethod
    def _public_entry(entry: dict, index: int) -> dict:
        digest = hashlib.md5(
            json.dumps(entry, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        action = entry.get("action", "")
        action_label = "打开本地目录" if action in {"local_folder", "打开本地目录"} else "打开网络来源"
        return {
            "key": f"{digest}-{index}",
            "openedAt": entry.get("opened_at", ""),
            "action": action_label,
            "itemId": entry.get("id", ""),
            "title": entry.get("title", ""),
            "author": entry.get("author", ""),
            "link": entry.get("link", ""),
            "localPath": entry.get("local_path", ""),
            "tags": entry.get("tags", []),
            "titleWords": entry.get("title_words", []),
        }

    def list(self) -> list[dict]:
        entries = load_history_entries()
        return [self._public_entry(entry, index) for index, entry in reversed(list(enumerate(entries)))]

    def record(self, item_id: str, action: str) -> list[dict]:
        detail = self.library.detail(item_id)
        if not detail:
            raise KeyError(item_id)
        raw = {
            "ID": detail["id"],
            "标题": detail["title"],
            "作者": detail["artist"],
            "链接": detail["link"],
            "本地目录": detail["localPath"],
            "解析后标签": detail["tags"],
            "标题特征词": detail.get("titleWords", []),
        }
        normalized_action = "local_folder" if action in {"local_folder", "打开本地目录"} else "network_link"
        with self._lock:
            record_recommendation_history(raw, normalized_action)
            self.library.clear_query_cache()
        return self.list()

    def delete(self, keys: list[str]) -> list[dict]:
        wanted = set(keys)
        with self._lock:
            entries = load_history_entries()
            kept = [
                entry
                for index, entry in enumerate(entries)
                if self._public_entry(entry, index)["key"] not in wanted
            ]
            save_history_entries(kept)
            self.library.clear_query_cache()
        return self.list()

    def clear(self) -> list[dict]:
        with self._lock:
            clear_history_entries()
            self.library.clear_query_cache()
        return []
