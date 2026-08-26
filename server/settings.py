from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


@dataclass(frozen=True)
class Settings:
    app_name: str = "XP-Gacha 地下金库"
    environment: str = os.getenv("XP_GACHA_ENV", "development")
    host: str = os.getenv("XP_GACHA_HOST", "0.0.0.0")
    port: int = int(os.getenv("XP_GACHA_PORT", "8000"))
    frontend_dist: Path = Path(
        os.getenv("XP_GACHA_FRONTEND_DIST", str(PROJECT_ROOT / "web" / "dist"))
    ).resolve()
    allow_open_local: bool = _bool_env("XP_GACHA_ALLOW_OPEN_LOCAL", True)
    import_max_mb: int = int(os.getenv("XP_GACHA_IMPORT_MAX_MB", "1024"))
    project_root: Path = PROJECT_ROOT


settings = Settings()
