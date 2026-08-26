"""Proxy configuration shared by the legacy online collection tasks."""

from __future__ import annotations

import os


def build_proxies(proxy_url: object) -> dict[str, str]:
    """Return an explicit proxy map; an empty setting forces direct access."""
    value = str(proxy_url or "").strip()
    if not value:
        # Both curl_cffi and requests otherwise inherit proxy variables from the process.
        return {"http": "", "https": ""}
    if "://" not in value:
        value = f"http://{value}"
    return {"http": value, "https": value}


def configured_proxies() -> dict[str, str]:
    return build_proxies(os.getenv("ONLINE_COVER_PROXY", ""))
