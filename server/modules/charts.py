from __future__ import annotations

from utils_charts import build_history_preference_chart_data
from utils_history import load_history_entries


def _normalize_chart_data(payload: dict) -> dict:
    normalized = {}
    for key, meta in payload.items():
        normalized[key] = {
            **meta,
            "top_15": [{"label": label, "value": value} for label, value in meta.get("top_15", [])],
            "top_150": [{"label": label, "value": value} for label, value in meta.get("top_150", [])],
        }
    return normalized


class ChartsModule:
    def __init__(self, library) -> None:
        self.library = library

    def global_charts(self) -> dict:
        return _normalize_chart_data(self.library.charts())

    def history_charts(self) -> dict:
        return _normalize_chart_data(build_history_preference_chart_data(load_history_entries()))
