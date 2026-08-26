from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import MAX_DISPLAY


class APIModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class GlobalWeights(APIModel):
    tag: float = Field(1.0, ge=0, le=5)
    artist: float = Field(1.0, ge=0, le=5)
    title: float = Field(1.0, ge=0, le=5)
    history: float = Field(1.0, ge=0, le=5)


class LibraryQuery(APIModel):
    keyword: str = ""
    keyword_relevance: bool = Field(False, alias="keywordRelevance")
    semantic_query: str = Field("", alias="semanticQuery")
    cover_query: str = Field("", alias="coverQuery")
    cover_matches: dict[str, float] = Field(default_factory=dict, alias="coverMatches")
    weights: GlobalWeights = Field(default_factory=GlobalWeights)
    min_score: int = Field(0, alias="minScore")
    blocked_tags: list[str] = Field(default_factory=list, alias="blockedTags")
    tag_weights: dict[str, float] = Field(default_factory=dict, alias="tagWeights")
    artist_weights: dict[str, float] = Field(default_factory=dict, alias="artistWeights")
    title_weights: dict[str, float] = Field(default_factory=dict, alias="titleWeights")
    sort: str = "score"
    descending: bool = True
    page: int = Field(0, ge=0)
    page_size: int = Field(MAX_DISPLAY, alias="pageSize", ge=1, le=MAX_DISPLAY)


class HistoryRecordRequest(APIModel):
    item_id: str = Field(alias="itemId")
    action: Literal["local_folder", "network_link", "打开本地目录", "打开网络来源"]


class HistoryDeleteRequest(APIModel):
    keys: list[str]


class ChatRequest(APIModel):
    query: str = Field(min_length=1)
    api_mode: str = Field("本地 (LM Studio)", alias="apiMode")
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(4096, alias="maxTokens", ge=256, le=32768)
    context_ids: list[str] = Field(default_factory=list, alias="contextIds")
    context_count: int = Field(10, alias="contextCount", ge=0, le=500)


class JobStartRequest(APIModel):
    script_id: str = Field(alias="scriptId")
    parameters: dict[str, Any] = Field(default_factory=dict)


class ImportModeRequest(APIModel):
    mode: Literal["upsert", "replace"] = "upsert"
    include_dictionaries: bool = Field(True, alias="includeDictionaries")


class PreferencesRequest(APIModel):
    column_widths: dict[str, int] = Field(default_factory=dict, alias="columnWidths")

    @field_validator("column_widths")
    @classmethod
    def validate_widths(cls, value: dict[str, int]) -> dict[str, int]:
        return {str(key): max(40, min(int(width), 1600)) for key, width in value.items()}
