import json
import math
import os
import hashlib
import threading
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

from config import (
    HISTORY_CACHE_FILE,
    HISTORY_LINK_TRACKING_HOST,
    HISTORY_LINK_TRACKING_PORT,
    HISTORY_RECOMMENDATION_CACHE_SIZE,
)

_TRACKED_LINK_ITEMS = {}
_TRACKED_LINK_LOCK = threading.Lock()
_TRACKING_SERVER = None


def _coerce_list(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str):
        stripped_value = value.strip()
        if not stripped_value:
            return []
        return [item.strip() for item in stripped_value.split(",") if item.strip()]

    return []


def _unique_items(items):
    seen_items = set()
    unique_items = []
    for item in items:
        if item in seen_items:
            continue
        seen_items.add(item)
        unique_items.append(item)
    return unique_items


def _trim_entries(entries, max_entries=HISTORY_RECOMMENDATION_CACHE_SIZE):
    valid_entries = [entry for entry in entries if isinstance(entry, dict)]
    if max_entries <= 0:
        return []
    return valid_entries[-max_entries:]


def load_history_entries():
    if not os.path.exists(HISTORY_CACHE_FILE):
        return []

    try:
        with open(HISTORY_CACHE_FILE, "r", encoding="utf-8") as file:
            entries = json.load(file)
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(entries, list):
        return []

    return _trim_entries(entries)


def save_history_entries(entries):
    os.makedirs(os.path.dirname(HISTORY_CACHE_FILE), exist_ok=True)
    trimmed_entries = _trim_entries(entries)
    temp_file_path = f"{HISTORY_CACHE_FILE}.tmp"

    with open(temp_file_path, "w", encoding="utf-8") as file:
        json.dump(trimmed_entries, file, ensure_ascii=False, indent=2)

    os.replace(temp_file_path, HISTORY_CACHE_FILE)
    return trimmed_entries


def clear_history_entries():
    return save_history_entries([])


def build_history_entry(row_data, action):
    if hasattr(row_data, "to_dict"):
        row_data = row_data.to_dict()

    row_data = row_data or {}
    return {
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "action": str(action or "").strip(),
        "id": str(row_data.get("ID", "")).strip(),
        "title": str(row_data.get("标题", "")).strip(),
        "author": str(row_data.get("作者", "")).strip(),
        "link": str(row_data.get("链接", "")).strip(),
        "local_path": str(row_data.get("本地目录", "")).strip(),
        "tags": _unique_items(_coerce_list(row_data.get("解析后标签"))),
        "title_words": _unique_items(_coerce_list(row_data.get("标题特征词"))),
    }


def record_recommendation_history(row_data, action):
    entry = build_history_entry(row_data, action)
    if not entry["id"] and not entry["title"]:
        return load_history_entries()

    entries = load_history_entries()
    entries.append(entry)
    return save_history_entries(entries)


def _is_valid_web_link(link):
    normalized_link = str(link or "").strip().lower()
    return normalized_link.startswith(("http://", "https://"))


def _build_tracking_token(row_data):
    item_id = str(row_data.get("ID", "")).strip()
    if item_id:
        return item_id

    raw_token = json.dumps(
        {
            "title": row_data.get("标题", ""),
            "link": row_data.get("链接", ""),
            "local_path": row_data.get("本地目录", ""),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.md5(raw_token.encode("utf-8")).hexdigest()


def register_tracked_link_item(row_data):
    if hasattr(row_data, "to_dict"):
        row_data = row_data.to_dict()

    row_data = row_data or {}
    token = _build_tracking_token(row_data)
    with _TRACKED_LINK_LOCK:
        _TRACKED_LINK_ITEMS[token] = dict(row_data)
    return token


def build_tracked_link(row_data):
    if hasattr(row_data, "to_dict"):
        row_data = row_data.to_dict()

    row_data = row_data or {}
    target_link = str(row_data.get("链接", "")).strip()
    if not _is_valid_web_link(target_link):
        return target_link

    token = register_tracked_link_item(row_data)
    return (
        f"http://{HISTORY_LINK_TRACKING_HOST}:{HISTORY_LINK_TRACKING_PORT}/open"
        f"?token={quote(token)}&target={quote(target_link, safe='')}"
    )


class _HistoryTrackingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urlparse(self.path)
        if parsed_url.path != "/open":
            self.send_error(404)
            return

        query = parse_qs(parsed_url.query)
        token = query.get("token", [""])[0]
        target_link = query.get("target", [""])[0]

        with _TRACKED_LINK_LOCK:
            item_payload = _TRACKED_LINK_ITEMS.get(token)

        if item_payload:
            record_recommendation_history(item_payload, "network_link")

        if not _is_valid_web_link(target_link):
            target_link = str((item_payload or {}).get("链接", "")).strip()

        if not _is_valid_web_link(target_link):
            self.send_error(400, "Invalid target link")
            return

        self.send_response(302)
        self.send_header("Location", target_link)
        self.end_headers()

    def log_message(self, format, *args):
        return


def start_link_tracking_server():
    global _TRACKING_SERVER

    if _TRACKING_SERVER is not None:
        return _TRACKING_SERVER

    try:
        server = ThreadingHTTPServer(
            (HISTORY_LINK_TRACKING_HOST, HISTORY_LINK_TRACKING_PORT),
            _HistoryTrackingHandler,
        )
    except OSError:
        return None

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    _TRACKING_SERVER = server
    return server


def _count_history_features(history_entries, field_name):
    feature_counter = Counter()
    for entry in history_entries:
        if not isinstance(entry, dict):
            continue
        feature_counter.update(_unique_items(_coerce_list(entry.get(field_name))))
    return feature_counter


def _count_history_authors(history_entries):
    author_counter = Counter()
    for entry in history_entries:
        if not isinstance(entry, dict):
            continue
        author = str(entry.get("author", "")).strip()
        if author:
            author_counter[author] += 1
    return author_counter


def _build_rarity_bonus_map(history_counter, database_counter, bonus_scale):
    if not history_counter or bonus_scale <= 0:
        return {}

    total_database_occurrences = max(sum(database_counter.values()), 1)
    bonus_map = {}
    for feature_name, history_count in history_counter.items():
        database_count = max(int(database_counter.get(feature_name, 0)), 0)
        rarity_factor = math.log1p((total_database_occurrences + 1) / (database_count + 1))
        bonus_map[feature_name] = float(history_count) * rarity_factor * float(bonus_scale)
    return bonus_map


def build_history_preference_maps(
    history_entries,
    tag_freq,
    title_word_freq,
    artist_freq,
    tag_bonus_scale=1.0,
    title_bonus_scale=1.0,
    artist_bonus_scale=1.0,
):
    tag_history_counter = _count_history_features(history_entries, "tags")
    title_history_counter = _count_history_features(history_entries, "title_words")
    artist_history_counter = _count_history_authors(history_entries)

    return {
        "tags": _build_rarity_bonus_map(tag_history_counter, tag_freq, tag_bonus_scale),
        "title_words": _build_rarity_bonus_map(
            title_history_counter,
            title_word_freq,
            title_bonus_scale,
        ),
        "artists": _build_rarity_bonus_map(
            artist_history_counter,
            artist_freq,
            artist_bonus_scale,
        ),
    }
