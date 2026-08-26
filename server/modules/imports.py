from __future__ import annotations

import importlib
import io
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

import pandas as pd
from sqlalchemy import inspect, text

import config
from data_processing.optimize_mysql_schema import optimize_gallery_schema
from server.database import get_engine, migration_lock
from server.settings import settings


ID_COLUMN = "ID"
LINK_COLUMN = "链接"
DB_COLUMNS = [
    ID_COLUMN,
    LINK_COLUMN,
    "文件名",
    "标题",
    "标题译文",
    "标签",
    "作者",
    "团队",
    "语言",
    "页数",
    "上传日期",
]
DICTIONARY_NAMES = {
    "STOP_TAGS.txt",
    "SEMANTIC_MAP.json",
    "TITLE_STOP_WORDS.txt",
    "TITLE_SEMANTIC_MAP.json",
}


def extract_gallery_id(url: object) -> str:
    import re

    value = str(url or "").strip()
    nh_match = re.search(r"/g/(\d+)/?", value)
    if nh_match:
        return f"NH{nh_match.group(1)}"
    jm_match = re.search(r"/album/(\d+)/?", value)
    if jm_match:
        return f"JM{jm_match.group(1)}"
    return ""


def normalize_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if LINK_COLUMN not in frame.columns:
        raise ValueError(f"CSV 缺少必要列：{LINK_COLUMN}")
    for column in DB_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame.fillna("")
    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str).str.strip()
    frame[LINK_COLUMN] = frame[LINK_COLUMN].astype(str).str.strip()
    missing = frame[ID_COLUMN] == ""
    frame.loc[missing, ID_COLUMN] = frame.loc[missing, LINK_COLUMN].apply(extract_gallery_id)
    frame["标题"] = frame.apply(
        lambda row: row.get("文件名", "") if str(row.get("标题", "")).strip() == "" else row.get("标题", ""),
        axis=1,
    )
    frame["页数"] = pd.to_numeric(frame["页数"], errors="coerce").fillna(0).astype(int)
    frame = frame[frame[ID_COLUMN] != ""]
    return frame.drop_duplicates(subset=[ID_COLUMN], keep="last")[DB_COLUMNS].reset_index(drop=True)


def read_csv_files(files: list[Path]) -> pd.DataFrame:
    frames = []
    errors = []
    for path in files:
        try:
            try:
                frames.append(pd.read_csv(path, encoding="utf-8-sig"))
            except UnicodeDecodeError:
                frames.append(pd.read_csv(path, encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    if not frames:
        detail = "；".join(errors) if errors else "未找到 CSV"
        raise ValueError(f"没有可导入的数据：{detail}")
    return normalize_dataframe(pd.concat(frames, ignore_index=True))


def _safe_extract(payload: bytes, destination: Path) -> list[Path]:
    destination = destination.resolve()
    extracted: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            relative = Path(info.filename.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"ZIP 包含不安全路径：{info.filename}")
            target = (destination / relative).resolve()
            if destination not in target.parents:
                raise ValueError(f"ZIP 包含越界路径：{info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted.append(target)
    return extracted


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{uuid.uuid4().hex}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _reload_dictionary_modules() -> None:
    import data_pipeline
    import utils_nlp

    config.STOP_TAGS = config.load_text_config(Path(config.DICTIONARY_DIR) / "STOP_TAGS.txt")
    config.SEMANTIC_MAP = config.load_json_config(Path(config.DICTIONARY_DIR) / "SEMANTIC_MAP.json")
    config.TITLE_STOP_WORDS = config.load_text_config(Path(config.DICTIONARY_DIR) / "TITLE_STOP_WORDS.txt")
    config.TITLE_SEMANTIC_MAP = config.load_json_config(Path(config.DICTIONARY_DIR) / "TITLE_SEMANTIC_MAP.json")
    data_pipeline.STOP_TAGS = config.STOP_TAGS
    data_pipeline.SEMANTIC_MAP = config.SEMANTIC_MAP
    utils_nlp.TITLE_STOP_WORDS = config.TITLE_STOP_WORDS
    utils_nlp.TITLE_SEMANTIC_MAP = config.TITLE_SEMANTIC_MAP


def import_dataframe(frame: pd.DataFrame, mode: str = "upsert") -> dict:
    engine = get_engine()
    dialect = engine.url.get_backend_name()
    with migration_lock():
        table_exists = inspect(engine).has_table("gallery_info")
        if mode == "replace" or not table_exists:
            frame.to_sql("gallery_info", engine, if_exists="replace", index=False, chunksize=2000)
        elif dialect == "mysql":
            optimize_gallery_schema(engine)
            staging = f"gallery_import_{uuid.uuid4().hex[:12]}"
            frame.to_sql(staging, engine, if_exists="replace", index=False, chunksize=2000)
            quoted_columns = ", ".join(f"`{column}`" for column in DB_COLUMNS)
            updates = ", ".join(
                (
                    "`标题译文` = COALESCE(NULLIF(VALUES(`标题译文`), ''), `标题译文`)"
                    if column == "标题译文"
                    else f"`{column}` = VALUES(`{column}`)"
                )
                for column in DB_COLUMNS
                if column != ID_COLUMN
            )
            try:
                with engine.begin() as connection:
                    connection.execute(
                        text(
                            f"INSERT INTO gallery_info ({quoted_columns}) "
                            f"SELECT {quoted_columns} FROM `{staging}` "
                            f"ON DUPLICATE KEY UPDATE {updates}"
                        )
                    )
            finally:
                with engine.begin() as connection:
                    connection.execute(text(f"DROP TABLE IF EXISTS `{staging}`"))
        else:
            existing = pd.read_sql("SELECT * FROM gallery_info", engine)
            combined = pd.concat([existing, frame], ignore_index=True)
            combined = normalize_dataframe(combined)
            combined.to_sql("gallery_info", engine, if_exists="replace", index=False)

        if dialect == "mysql":
            optimize_gallery_schema(engine)
        total = int(pd.read_sql("SELECT COUNT(*) AS count FROM gallery_info", engine).iloc[0]["count"])
    return {"imported": int(len(frame)), "total": total, "mode": mode}


class ImportModule:
    """Atomic dictionary and catalogue import behind one small interface."""

    def __init__(self, library) -> None:
        self.library = library
        self.import_root = Path(config.CACHE_DIR) / "imports"
        self.import_root.mkdir(parents=True, exist_ok=True)

    def import_project_data(self, mode: str = "upsert") -> dict:
        csv_dir = Path(config.DATA_ROOT) / "data" / "gallery_info"
        files = sorted(csv_dir.glob("*.csv")) if csv_dir.exists() else []
        frame = read_csv_files(files)
        result = import_dataframe(frame, mode=mode)
        self._after_import()
        return {**result, "csvFiles": len(files), "dictionaries": []}

    def import_bundle(
        self,
        filename: str,
        payload: bytes,
        mode: str = "upsert",
        include_dictionaries: bool = True,
    ) -> dict:
        if len(payload) > settings.import_max_mb * 1024 * 1024:
            raise ValueError(f"导入包超过 {settings.import_max_mb} MB 限制")
        with tempfile.TemporaryDirectory(prefix="xp-gacha-import-", dir=self.import_root) as temp:
            root = Path(temp)
            if filename.lower().endswith(".zip"):
                files = _safe_extract(payload, root)
            elif filename.lower().endswith(".csv"):
                target = root / Path(filename).name
                target.write_bytes(payload)
                files = [target]
            else:
                raise ValueError("只支持 .zip 或 .csv 导入包")

            dictionary_files = [path for path in files if path.name in DICTIONARY_NAMES]
            csv_files = [path for path in files if path.suffix.lower() == ".csv"]
            dictionary_result: list[str] = []
            if include_dictionaries and dictionary_files:
                backup_root = self.import_root / "backups" / time.strftime("%Y%m%d-%H%M%S")
                for source in dictionary_files:
                    destination = Path(config.DICTIONARY_DIR) / source.name
                    if destination.exists():
                        _atomic_copy(destination, backup_root / source.name)
                    _atomic_copy(source, destination)
                    dictionary_result.append(source.name)
                _reload_dictionary_modules()

            data_result = {"imported": 0, "total": None, "mode": mode}
            if csv_files:
                data_result = import_dataframe(read_csv_files(csv_files), mode=mode)
            if not csv_files and not dictionary_result:
                raise ValueError("导入包中没有识别到 CSV 或标准词典文件")

        self._after_import()
        return {
            **data_result,
            "csvFiles": len(csv_files),
            "dictionaries": dictionary_result,
        }

    def _after_import(self) -> None:
        for cache_name in ("preprocessed_df.pkl", "data.hash"):
            cache = Path(config.CACHE_DIR) / cache_name
            if cache.exists():
                cache.unlink()
        self.library.refresh()
