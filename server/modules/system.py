from __future__ import annotations

import copy
import fnmatch
import os
import threading
from pathlib import Path

import config
from server.database import database_status


QWEN_MODEL_URL = "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B"
CLIP_MODEL_URL = "https://huggingface.co/openai/clip-vit-base-patch32"


def _count_files(path: str, pattern: str = "*") -> int:
    try:
        with os.scandir(path) as entries:
            return sum(
                1
                for entry in entries
                if fnmatch.fnmatch(entry.name, pattern) and entry.is_file()
            )
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return 0


def _file_status(label: str, path: str) -> dict:
    target = Path(path)
    return {
        "name": label,
        "path": str(target),
        "exists": target.is_file(),
        "sizeKb": round(target.stat().st_size / 1024, 1) if target.is_file() else 0,
    }


def _model_status(label: str, path: str, download_url: str, required_entries: tuple[str, ...] = ()) -> dict:
    target = Path(path)
    exists = target.is_dir()
    has_config = (target / "config.json").is_file() and (target / "config.json").stat().st_size > 0
    has_weight = any(
        candidate.is_file() and candidate.stat().st_size > 0
        for pattern in ("*.safetensors", "pytorch_model*.bin")
        for candidate in target.glob(pattern)
    ) if exists else False
    required_ready = all(
        (target / entry).is_file() and (target / entry).stat().st_size > 0
        for entry in required_entries
    )
    ready = exists and has_config and has_weight and required_ready
    return {
        "kind": "model",
        "label": label,
        "path": str(target),
        "exists": exists,
        "ready": ready,
        "state": "ready" if ready else ("incomplete" if exists else "missing"),
        "downloadUrl": download_url,
    }


def _vector_status(label: str, path: str) -> dict:
    target = Path(path)
    exists = target.is_file()
    ready = exists and target.stat().st_size > 0
    return {
        "kind": "vector",
        "label": label,
        "path": str(target),
        "exists": exists,
        "ready": ready,
        "state": "ready" if ready else ("incomplete" if exists else "missing"),
        "generatedLocally": True,
    }


def _search_capabilities() -> dict:
    semantic_model = _model_status(
        "Qwen3-Embedding-0.6B",
        config.LOCAL_MODEL_PATH,
        QWEN_MODEL_URL,
        required_entries=("modules.json", "tokenizer.json", "1_Pooling/config.json"),
    )
    semantic_vector = _vector_status("文本语义向量", config.VECTOR_FILE)
    clip_model = _model_status(
        "CLIP ViT-B/32",
        config.CLIP_MODEL_PATH,
        CLIP_MODEL_URL,
        required_entries=("preprocessor_config.json", "tokenizer.json"),
    )
    clip_vector = _vector_status("封面向量索引", config.IMG_VECTOR_FILE)

    semantic_missing = [
        kind for kind, dependency in (("model", semantic_model), ("vector", semantic_vector))
        if not dependency["ready"]
    ]
    cover_missing = [
        kind for kind, dependency in (("model", clip_model), ("vector", clip_vector))
        if not dependency["ready"]
    ]
    return {
        "semantic": {
            "label": "AI 语义检索",
            "ready": not semantic_missing,
            "missing": semantic_missing,
            "dependencies": {"model": semantic_model, "vector": semantic_vector},
            "setup": {
                "section": "cache",
                "scriptId": "text-vector",
                "actionLabel": "构建文本语义向量",
            },
        },
        "cover": {
            "label": "封面相似检索",
            "ready": not cover_missing,
            "idReady": clip_vector["ready"],
            "uploadReady": clip_model["ready"] and clip_vector["ready"],
            "missing": cover_missing,
            "dependencies": {"model": clip_model, "vector": clip_vector},
            "setup": {
                "section": "cache",
                "scriptId": "clip-vector",
                "actionLabel": "构建或刷新封面 CLIP 向量",
            },
        },
    }


class SystemModule:
    def __init__(self) -> None:
        self._counts_lock = threading.RLock()
        self._counts_cache: dict | None = None

    def health_status(self) -> dict:
        """Return service dependencies without scanning any cache directories."""
        return {"database": database_status()}

    @staticmethod
    def _build_counts() -> dict:
        return {
            "csv": _count_files(str(Path(config.DATA_ROOT) / "data" / "gallery_info"), "*.csv"),
            "onlineCovers": _count_files(config.ONLINE_IMG_DIR),
            "localThumbnails": _count_files(config.IMG_CACHE_DIR),
            "base64": _count_files(config.B64_CACHE_DIR, "*.txt"),
        }

    def _counts(self, *, refresh: bool = False) -> dict:
        """Return the cached directory counts, rebuilding only when requested."""
        with self._counts_lock:
            if self._counts_cache is None or refresh:
                latest = self._build_counts()
                self._counts_cache = latest
            return copy.deepcopy(self._counts_cache)

    def prime_counts(self) -> None:
        """Build the initial snapshot before catalogue warm-up starts."""
        self._counts()

    def status(self, *, refresh: bool = False) -> dict:
        """Return live lightweight state plus a cached directory-count snapshot.

        The first status request builds the counts synchronously so callers never
        receive misleading zero placeholders. Later requests, including browser
        reloads, reuse the counts; ``refresh=True`` explicitly rebuilds only them.
        Database, model, cache-file and path state remain lightweight and current.
        """
        db = database_status()
        search_capabilities = _search_capabilities()
        return {
            "database": db,
            "models": {
                "semantic": search_capabilities["semantic"]["dependencies"]["model"]["ready"],
                "clip": search_capabilities["cover"]["dependencies"]["model"]["ready"],
            },
            "searchCapabilities": search_capabilities,
            "counts": self._counts(refresh=refresh),
            "caches": [
                _file_status("预处理 DataFrame", str(Path(config.CACHE_DIR) / "preprocessed_df.pkl")),
                _file_status("预处理 Hash", str(Path(config.CACHE_DIR) / "data.hash")),
                _file_status("文本向量", config.VECTOR_FILE),
                _file_status("封面向量", config.IMG_VECTOR_FILE),
            ],
            "paths": {
                "dataRoot": str(config.DATA_ROOT),
                "library": config.BASE_DIR,
                "dictionaries": config.DICTIONARY_DIR,
            },
        }
