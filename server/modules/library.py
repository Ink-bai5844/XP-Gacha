from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd

from config import COVER_SEARCH_TOP_K, HISTORY_CACHE_FILE, IMG_VECTOR_FILE, MAX_DISPLAY, SEMANTIC_SEARCH_TOP_K
from data_pipeline import (
    apply_dynamic_scores,
    get_row_indices_for_ids,
    load_base_data,
    search_gallery_candidate_ids,
)
from server.schemas import LibraryQuery
from utils_cv import search_similar_cover_items
from utils_history import build_history_preference_maps, load_history_entries


SORT_COLUMNS = {
    "score": "推荐评分",
    "keyword": "关键词相关度",
    "semantic": "AI相关度",
    "cover": "封面相关度",
    "id": "ID",
    "date": "上传日期",
    "titleZh": "标题译文",
    "title": "标题",
    "artist": "作者",
    "circle": "团队",
    "tags": "标签",
    "language": "语言",
    "pages": "页数",
    "localPath": "本地目录",
}

OPTION_PREVIEW_LIMIT = 80


def _as_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _as_float(value: Any) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _split_values(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in _as_text(value).replace("，", ",").split(",") if item.strip()]


def _cover_code(item_id: str) -> str:
    variants = ("circle", "slash", "frame", "type")
    digest = hashlib.md5(item_id.encode("utf-8")).digest()[0]
    return variants[digest % len(variants)]


def row_to_item(row: pd.Series | dict, include_internal: bool = False) -> dict:
    payload = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    item_id = _as_text(payload.get("ID")).strip()
    title = _as_text(payload.get("标题")).strip()
    title_zh = _as_text(payload.get("标题译文")).strip()
    parsed_tags = _split_values(payload.get("解析后标签")) or _split_values(payload.get("标签"))
    item = {
        "id": item_id,
        "titleZh": title_zh,
        "title": title,
        "artist": _as_text(payload.get("作者")).strip(),
        "circle": _as_text(payload.get("团队")).strip(),
        "tags": parsed_tags,
        "language": _as_text(payload.get("语言")).strip(),
        "pages": _as_int(payload.get("页数")),
        "uploadedAt": _as_text(payload.get("上传日期")).strip(),
        "baseScore": _as_int(payload.get("推荐评分")),
        "score": _as_int(payload.get("推荐评分")),
        "keywordRelevance": _as_float(payload.get("关键词相关度")),
        "aiRelevance": _as_float(payload.get("AI相关度")),
        "coverRelevance": _as_float(payload.get("封面相关度")),
        "localPath": _as_text(payload.get("本地目录")).strip() or "本地目录不存在",
        "link": _as_text(payload.get("链接")).strip(),
        "filename": _as_text(payload.get("文件名")).strip(),
        "summary": title_zh or title or f"馆藏条目 {item_id}",
        "coverCode": _cover_code(item_id),
        "coverUrl": f"/api/covers/{item_id}" if item_id else "",
    }
    if include_internal:
        item["titleWords"] = _split_values(payload.get("标题特征词"))
        item["rawTags"] = _split_values(payload.get("标签"))
    return item


class LibraryModule:
    """Owns the complete Streamlit-era search and scoring pipeline."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._base_cache: tuple | None = None
        self._query_cache: OrderedDict[str, dict] = OrderedDict()
        self._query_cache_limit = 8
        self._meta_cache: dict | None = None
        self._option_cache: dict[str, list[str]] = {}

    @staticmethod
    def _history_stamp() -> tuple[int, int]:
        try:
            stat = os.stat(HISTORY_CACHE_FILE)
            return stat.st_mtime_ns, stat.st_size
        except OSError:
            return 0, 0

    def _query_cache_key(self, request: LibraryQuery) -> str:
        payload = request.model_dump(mode="json", by_alias=True)
        return json.dumps(
            {"request": payload, "history": self._history_stamp()},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def clear_query_cache(self) -> None:
        with self._lock:
            self._query_cache.clear()

    def _cached_query(self, cache_key: str) -> dict | None:
        with self._lock:
            cached = self._query_cache.get(cache_key)
            if cached is not None:
                self._query_cache.move_to_end(cache_key)
            return cached

    def _remember_query(self, cache_key: str, payload: dict) -> dict:
        with self._lock:
            self._query_cache[cache_key] = payload
            self._query_cache.move_to_end(cache_key)
            while len(self._query_cache) > self._query_cache_limit:
                self._query_cache.popitem(last=False)
        return payload

    def _base(self):
        with self._lock:
            if self._base_cache is None:
                # Streamlit's cache_data deserializes this multi-hundred-thousand-row
                # payload on every bare Python call. The API process can safely keep
                # the already loaded tuple in memory until an import explicitly
                # refreshes it.
                self._base_cache = load_base_data()
            return self._base_cache

    def available(self) -> bool:
        df, *_ = self._base()
        return df is not None and not df.empty

    def meta(self) -> dict:
        with self._lock:
            if self._meta_cache is not None:
                return self._meta_cache
        df, tag_freq, artist_freq, title_freq, _, _ = self._base()
        option_cache = {
            "tags": sorted(tag_freq.keys()),
            "artists": sorted(artist_freq.keys()),
            "titleWords": sorted(title_freq.keys()),
        }
        payload = {
            "pageSize": MAX_DISPLAY,
            "tags": option_cache["tags"][:OPTION_PREVIEW_LIMIT],
            "artists": option_cache["artists"][:OPTION_PREVIEW_LIMIT],
            "titleWords": option_cache["titleWords"][:OPTION_PREVIEW_LIMIT],
            "languages": sorted(
                {_as_text(value).strip() for value in (df.get("语言", []) if df is not None else []) if _as_text(value).strip()}
            ),
            "metrics": {
                "items": int(len(df)) if df is not None else 0,
                "artists": len(artist_freq),
                "tags": len(tag_freq),
                "titleWords": len(title_freq),
            },
        }
        with self._lock:
            self._option_cache = option_cache
            self._meta_cache = payload
        return payload

    def search_options(
        self,
        kind: str,
        query: str = "",
        limit: int = OPTION_PREVIEW_LIMIT,
        offset: int = 0,
    ) -> dict:
        if kind not in {"tags", "artists", "titleWords"}:
            raise ValueError("不支持的选项类型")
        with self._lock:
            options = self._option_cache.get(kind)
        if options is None:
            self.meta()
            with self._lock:
                options = self._option_cache.get(kind, [])

        normalized_query = query.strip().casefold()
        matches = options if not normalized_query else [
            option for option in options if normalized_query in option.casefold()
        ]
        safe_limit = max(1, min(int(limit), 200))
        safe_offset = max(0, min(int(offset), len(matches)))
        end = safe_offset + safe_limit
        return {
            "items": matches[safe_offset:end],
            "total": len(matches),
            "offset": safe_offset,
            "hasMore": len(matches) > end,
        }

    def query(self, request: LibraryQuery) -> dict:
        cache_key = self._query_cache_key(request)
        cached = self._cached_query(cache_key)
        if cached is not None:
            return cached

        df, tag_freq, artist_freq, title_freq, _, score_cache = self._base()
        if df is None or df.empty or not score_cache:
            return self._empty_response(request, "数据库尚未导入 gallery_info 数据。")

        warnings: list[str] = []
        search_payload = None
        row_indices = None
        if request.keyword.strip():
            search_payload = search_gallery_candidate_ids(
                request.keyword,
                include_relevance=request.keyword_relevance,
            )
            row_indices = get_row_indices_for_ids(
                df,
                search_payload["ids"],
                score_cache.get("id_to_row"),
            )

        history_entries = load_history_entries()
        history_preference = None
        if request.weights.history > 0 and history_entries:
            history_preference = build_history_preference_maps(
                history_entries,
                tag_freq,
                title_freq,
                artist_freq,
                tag_bonus_scale=request.weights.tag,
                title_bonus_scale=request.weights.title,
                artist_bonus_scale=request.weights.artist,
            )

        result = apply_dynamic_scores(
            df,
            request.tag_weights,
            request.artist_weights,
            request.title_weights,
            tag_freq,
            artist_freq,
            title_freq,
            request.weights.tag,
            request.weights.artist,
            request.weights.title,
            score_cache=score_cache,
            history_preference=history_preference,
            global_history_w=request.weights.history,
            row_indices=row_indices,
        )

        if request.keyword_relevance and search_payload is not None and not result.empty:
            result["关键词相关度"] = result["ID"].astype(str).map(search_payload["score_map"]).fillna(0.0)

        if request.blocked_tags and not result.empty:
            blocked = set(request.blocked_tags)
            result = result[
                result["解析后标签"].apply(lambda tags: not any(tag in blocked for tag in (tags or [])))
            ]

        result = result[result["推荐评分"] >= request.min_score]
        if request.semantic_query.strip() and not result.empty:
            result = self._semantic_filter(result, request.semantic_query, warnings)
        if request.cover_matches and not result.empty:
            score_map = {str(item_id).upper(): float(score) for item_id, score in request.cover_matches.items()}
            result = result[result["ID"].astype(str).str.upper().isin(score_map)].copy()
            result["封面相关度"] = result["ID"].astype(str).str.upper().map(score_map)
            result = result.sort_values("封面相关度", ascending=False)
        elif request.cover_query.strip() and not result.empty:
            result = self._cover_filter(result, request.cover_query, warnings)

        sort_column = SORT_COLUMNS.get(request.sort, "推荐评分")
        if sort_column not in result.columns:
            sort_column = "推荐评分"
        if sort_column == "推荐评分":
            by = ["推荐评分"] + (["上传日期"] if "上传日期" in result.columns else [])
            result = result.sort_values(by=by, ascending=[not request.descending] * len(by), kind="stable")
        else:
            result = result.sort_values(by=sort_column, ascending=not request.descending, kind="stable")

        total = int(len(result))
        start = request.page * request.page_size
        page_df = result.iloc[start : start + request.page_size]
        recall = {
            "mode": (search_payload or {}).get("mode", "none"),
            "candidateCount": len((search_payload or {}).get("ids", [])),
            "usedFulltext": bool((search_payload or {}).get("used_fulltext", False)),
        }
        return self._remember_query(cache_key, {
            "items": [row_to_item(row) for _, row in page_df.iterrows()],
            "total": total,
            "page": request.page,
            "pageSize": request.page_size,
            "metrics": {
                "items": int(len(df)),
                "artists": len(artist_freq),
                "tags": len(tag_freq),
                "titleWords": len(title_freq),
            },
            "recall": recall,
            "warnings": warnings,
        })

    def detail(self, item_id: str) -> dict | None:
        df, tag_freq, artist_freq, title_freq, _, score_cache = self._base()
        if df is None or df.empty:
            return None
        indices = get_row_indices_for_ids(df, [item_id], score_cache.get("id_to_row") if score_cache else None)
        if not len(indices):
            return None
        scored = apply_dynamic_scores(
            df, {}, {}, {}, tag_freq, artist_freq, title_freq, 1.0, 1.0, 1.0,
            score_cache=score_cache, row_indices=indices,
        )
        return row_to_item(scored.iloc[0], include_internal=True)

    def rows_for_ids(self, item_ids: list[str]) -> pd.DataFrame:
        df, *_rest, score_cache = self._base()
        if df is None or df.empty:
            return pd.DataFrame()
        indices = get_row_indices_for_ids(df, item_ids, score_cache.get("id_to_row") if score_cache else None)
        return df.iloc[indices].copy() if len(indices) else pd.DataFrame()

    def charts(self) -> dict:
        _, _, _, _, chart_cache, _ = self._base()
        return chart_cache or {}

    def refresh(self) -> None:
        from data_pipeline import get_gallery_table_columns, has_gallery_fulltext_index

        for function in (load_base_data, get_gallery_table_columns, has_gallery_fulltext_index):
            clear = getattr(function, "clear", None)
            if callable(clear):
                clear()
        with self._lock:
            self._base_cache = None
            self._meta_cache = None
            self._option_cache.clear()
        self.clear_query_cache()

    def _semantic_filter(self, frame: pd.DataFrame, query: str, warnings: list[str]) -> pd.DataFrame:
        try:
            import torch
            from sentence_transformers import util
            from utils_nlp import load_semantic_engine

            model, corpus_embeddings, corpus_ids, id_to_index = load_semantic_engine()
            surviving_ids = frame["ID"].astype(str).tolist()
            indices = [id_to_index[item_id] for item_id in surviving_ids if item_id in id_to_index]
            if not indices:
                return pd.DataFrame()
            subset = corpus_embeddings[indices]
            subset_ids = [corpus_ids[index] for index in indices]
            query_embedding = model.encode([query], convert_to_tensor=True)
            subset = subset.to(query_embedding.device, dtype=query_embedding.dtype)
            scores = util.cos_sim(query_embedding, subset)[0]
            top_k = min(SEMANTIC_SEARCH_TOP_K, len(subset_ids))
            top = torch.topk(scores, k=top_k)
            matched = [subset_ids[index] for index in top[1].tolist()]
            score_map = dict(zip(matched, (top[0] * 100).tolist()))
            result = frame[frame["ID"].astype(str).isin(matched)].copy()
            result["AI相关度"] = result["ID"].astype(str).map(score_map)
            return result.sort_values("AI相关度", ascending=False)
        except Exception as exc:
            warnings.append(f"语义检索不可用：{exc}")
            return frame

    def _cover_filter(self, frame: pd.DataFrame, query_id: str, warnings: list[str]) -> pd.DataFrame:
        try:
            payload = search_similar_cover_items(
                query_item_id=query_id.strip().upper(),
                candidate_ids=frame["ID"].astype(str).tolist(),
                top_k=COVER_SEARCH_TOP_K,
            )
            score_map = {entry["item_id"]: entry["score"] for entry in payload["results"]}
            if not score_map:
                return pd.DataFrame()
            result = frame[frame["ID"].astype(str).isin(score_map)].copy()
            result["封面相关度"] = result["ID"].astype(str).map(score_map)
            return result.sort_values("封面相关度", ascending=False)
        except FileNotFoundError:
            warnings.append(f"封面检索不可用：未找到 {IMG_VECTOR_FILE}")
            return frame
        except Exception as exc:
            warnings.append(f"封面检索不可用：{exc}")
            return frame

    @staticmethod
    def _empty_response(request: LibraryQuery, warning: str) -> dict:
        return {
            "items": [],
            "total": 0,
            "page": request.page,
            "pageSize": request.page_size,
            "metrics": {"items": 0, "artists": 0, "tags": 0, "titleWords": 0},
            "recall": {"mode": "none", "candidateCount": 0, "usedFulltext": False},
            "warnings": [warning],
        }
