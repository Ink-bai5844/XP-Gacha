from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine


def _load_legacy_mysql_config() -> dict[str, str]:
    secrets_path = Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return {}
    try:
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib
        with secrets_path.open("rb") as stream:
            return dict(tomllib.load(stream).get("mysql", {}))
    except Exception:
        return {}


def get_database_url() -> str:
    explicit_url = os.getenv("DATABASE_URL", "").strip()
    if explicit_url:
        return explicit_url

    legacy = _load_legacy_mysql_config()
    user = os.getenv("MYSQL_USER", str(legacy.get("user", "xp_gacha")))
    password = os.getenv("MYSQL_PASSWORD", str(legacy.get("password", "xp_gacha")))
    host = os.getenv("MYSQL_HOST", str(legacy.get("host", "127.0.0.1")))
    port = os.getenv("MYSQL_PORT", str(legacy.get("port", "3306")))
    database = os.getenv("MYSQL_DATABASE", str(legacy.get("database", "xp_gacha")))
    return (
        f"mysql+pymysql://{quote_plus(user)}:{quote_plus(password)}@"
        f"{host}:{port}/{quote_plus(database)}?charset=utf8mb4"
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    url = get_database_url()
    options: dict = {"pool_pre_ping": True}
    if url.startswith("mysql"):
        options.update(pool_size=5, max_overflow=10, pool_recycle=3600)
    return create_engine(url, **options)


def database_status(engine: Engine | None = None) -> dict:
    engine = engine or get_engine()
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            tables = inspect(connection).get_table_names()
            row_count = 0
            if "gallery_info" in tables:
                row_count = int(connection.execute(text("SELECT COUNT(*) FROM gallery_info")).scalar() or 0)
        return {
            "available": True,
            "driver": engine.url.drivername,
            "database": engine.url.database or "",
            "table_ready": "gallery_info" in tables,
            "row_count": row_count,
            "error": None,
        }
    except Exception as exc:
        return {
            "available": False,
            "driver": engine.url.drivername,
            "database": engine.url.database or "",
            "table_ready": False,
            "row_count": 0,
            "error": str(exc),
        }


_migration_lock = threading.RLock()


def migration_lock() -> threading.RLock:
    return _migration_lock
