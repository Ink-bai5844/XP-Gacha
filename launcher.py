from __future__ import annotations

import os
import threading
import webbrowser

import uvicorn

from server.settings import settings


def open_browser() -> None:
    webbrowser.open(f"http://127.0.0.1:{settings.port}")


def main() -> None:
    if os.getenv("XP_GACHA_NO_BROWSER", "").lower() not in {"1", "true", "yes"}:
        threading.Timer(1.2, open_browser).start()
    uvicorn.run("server.main:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
