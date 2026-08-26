"""Runtime configuration shared by the legacy Streamlit app and the API server.

This file intentionally contains no secrets.  Everything that may differ between
machines is read from the environment so the same checkout can run locally, in
Docker, or from a future desktop bundle.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("XP_GACHA_DATA_ROOT", str(PROJECT_ROOT))).expanduser().resolve()


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return int(default)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def runtime_path(env_name: str, relative_default: str) -> str:
    value = Path(os.getenv(env_name, relative_default)).expanduser()
    if not value.is_absolute():
        value = DATA_ROOT / value
    return str(value.resolve())


ONLINE_IMG_DIR = runtime_path("ONLINE_IMG_DIR", "onlineimgtmp")
IMG_CACHE_DIR = runtime_path("IMG_CACHE_DIR", "localimgtmp")
CACHE_DIR = runtime_path("CACHE_DIR", "datacache")
B64_CACHE_DIR = runtime_path("B64_CACHE_DIR", "b64_cache")
MODEL_DIR = runtime_path("MODEL_DIR", "models")
DICTIONARY_DIR = runtime_path("DICTIONARY_DIR", "dictionaries")

HISTORY_RECOMMENDATION_CACHE_SIZE = env_int("HISTORY_RECOMMENDATION_CACHE_SIZE", 50)
HISTORY_CACHE_FILE = runtime_path(
    "HISTORY_CACHE_FILE",
    str(Path("datacache") / "recommendation_history.json"),
)
HISTORY_LINK_TRACKING_HOST = os.getenv("HISTORY_LINK_TRACKING_HOST", "0.0.0.0")
HISTORY_LINK_TRACKING_PUBLIC_HOST = os.getenv("HISTORY_LINK_TRACKING_PUBLIC_HOST", "127.0.0.1")
HISTORY_LINK_TRACKING_PORT = env_int("HISTORY_LINK_TRACKING_PORT", 8765)

VECTOR_FILE = runtime_path("VECTOR_FILE", str(Path("manga_vectors") / "manga_vectors_Qwen3.pkl"))
IMG_VECTOR_FILE = runtime_path("IMG_VECTOR_FILE", str(Path("manga_vectors") / "clip_image_index.pkl"))
LOCAL_MODEL_PATH = runtime_path(
    "LOCAL_MODEL_PATH", str(Path("models") / "Qwen3-Embedding-0.6B")
)
CLIP_MODEL_PATH = runtime_path(
    "CLIP_MODEL_PATH", str(Path("models") / "clip-vit-base-patch32")
)

BASE_DIR = str(
    Path(os.getenv("XP_GACHA_BASE_DIR", str(DATA_ROOT / "library"))).expanduser().resolve()
)

LM_STUDIO_API_BASE = os.getenv("LM_STUDIO_API_BASE", "http://127.0.0.1:1234/v1").rstrip("/")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "local-model")
ONLINE_API_BASE = os.getenv("ONLINE_API_BASE", "").rstrip("/")
ONLINE_API_KEY = os.getenv("ONLINE_API_KEY", "")
ONLINE_MODEL = os.getenv("ONLINE_MODEL", "deepseek-v4-flash")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "你是‘地下金库’的智能助手。你的性格冷静、专业。"
    "‘地下金库’是一个储存大量漫画的数据库。我会为你提供一部分当前的库存数据作为参考。"
    "请结合这些数据回答问题。如果数据中没有相关内容，请基于你的通用知识库回答。"
    "如要推荐条目，要附上可点击的完整跳转链接。",
)

INITIAL_TAG_WEIGHTS = {
    "NTR(netorare)": env_float("INITIAL_TAG_WEIGHT_NTR", -2.0),
}

ONLINE_COVER_FETCH_ENABLED = env_bool("ONLINE_COVER_FETCH_ENABLED", True)
ONLINE_COVER_PROXY = os.getenv("ONLINE_COVER_PROXY", "")
ONLINE_COVER_FETCH_CONCURRENCY = env_int("ONLINE_COVER_FETCH_CONCURRENCY", 6)
SEMANTIC_SEARCH_TOP_K = env_int("SEMANTIC_SEARCH_TOP_K", 5000)
COVER_SEARCH_TOP_K = env_int("COVER_SEARCH_TOP_K", 5000)
MAX_DISPLAY = env_int("MAX_DISPLAY", 500)

for directory in [ONLINE_IMG_DIR, IMG_CACHE_DIR, CACHE_DIR, B64_CACHE_DIR, MODEL_DIR, DICTIONARY_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)


def load_text_config(filepath: str | Path) -> set[str]:
    try:
        content = Path(filepath).read_text(encoding="utf-8")
        quoted = set(re.findall(r"'(.*?)'", content))
        if quoted:
            return quoted
        return {
            line.strip().strip("\"'")
            for line in content.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    except FileNotFoundError:
        return set()


def load_json_config(filepath: str | Path) -> dict:
    try:
        return json.loads(Path(filepath).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


STOP_TAGS = load_text_config(Path(DICTIONARY_DIR) / "STOP_TAGS.txt")
SEMANTIC_MAP = load_json_config(Path(DICTIONARY_DIR) / "SEMANTIC_MAP.json")
TITLE_STOP_WORDS = load_text_config(Path(DICTIONARY_DIR) / "TITLE_STOP_WORDS.txt")
TITLE_SEMANTIC_MAP = load_json_config(Path(DICTIONARY_DIR) / "TITLE_SEMANTIC_MAP.json")
