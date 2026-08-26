from __future__ import annotations

from pathlib import Path

import config
from server.database import database_status


def _count_files(path: str, pattern: str = "*") -> int:
    directory = Path(path)
    if not directory.exists():
        return 0
    return sum(1 for entry in directory.glob(pattern) if entry.is_file())


def _file_status(label: str, path: str) -> dict:
    target = Path(path)
    return {
        "name": label,
        "path": str(target),
        "exists": target.is_file(),
        "sizeKb": round(target.stat().st_size / 1024, 1) if target.is_file() else 0,
    }


class SystemModule:
    def status(self) -> dict:
        db = database_status()
        return {
            "database": db,
            "models": {
                "semantic": Path(config.LOCAL_MODEL_PATH).exists(),
                "clip": Path(config.CLIP_MODEL_PATH).exists(),
            },
            "counts": {
                "csv": _count_files(str(Path(config.DATA_ROOT) / "data" / "gallery_info"), "*.csv"),
                "onlineCovers": _count_files(config.ONLINE_IMG_DIR),
                "localThumbnails": _count_files(config.IMG_CACHE_DIR),
                "base64": _count_files(config.B64_CACHE_DIR, "*.txt"),
            },
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
