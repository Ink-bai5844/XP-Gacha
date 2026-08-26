from __future__ import annotations

from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from config import MAX_DISPLAY


def _contains_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


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
    api_mode: Literal["本地 (LM Studio)", "线上 API"] = Field("本地 (LM Studio)", alias="apiMode")
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(4096, alias="maxTokens", ge=256, le=32768)
    deep_thinking: bool = Field(False, alias="deepThinking")
    context_ids: list[str] = Field(default_factory=list, alias="contextIds")
    context_count: int = Field(10, alias="contextCount", ge=0, le=500)


class LLMSettingsRequest(APIModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    local_api_base: str = Field(alias="localApiBase", min_length=1, max_length=2048)
    local_model: str = Field(alias="localModel", min_length=1, max_length=300)
    local_api_key: SecretStr | None = Field(None, alias="localApiKey")
    clear_local_api_key: bool = Field(False, alias="clearLocalApiKey")
    online_api_base: str = Field("", alias="onlineApiBase", max_length=2048)
    online_model: str = Field(alias="onlineModel", min_length=1, max_length=300)
    online_api_key: SecretStr | None = Field(None, alias="onlineApiKey")
    clear_online_api_key: bool = Field(False, alias="clearOnlineApiKey")

    @field_validator("local_api_base", "online_api_base")
    @classmethod
    def validate_api_base(cls, value: str, info) -> str:
        if _contains_control(value) or any(character.isspace() for character in value):
            raise ValueError("API URL 不能包含空白或控制字符")
        normalized = value.strip().rstrip("/")
        if not normalized and info.field_name == "online_api_base":
            return ""
        try:
            parsed = urlsplit(normalized)
            hostname = parsed.hostname
            parsed.port
        except ValueError as exc:
            raise ValueError("API URL 格式无效") from exc
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or not hostname:
            raise ValueError("API URL 必须是完整的 http:// 或 https:// 地址")
        if parsed.username or parsed.password:
            raise ValueError("API URL 不能包含用户名或密码")
        if parsed.query or parsed.fragment:
            raise ValueError("API URL 不能包含查询参数或片段；API Key 请填写在单独字段")
        return normalized

    @field_validator("local_model", "online_model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or _contains_control(normalized):
            raise ValueError("模型名称不能为空或包含控制字符")
        return normalized

    @field_validator("local_api_key", "online_api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        normalized = value.get_secret_value().strip()
        if len(normalized) > 4096 or _contains_control(normalized):
            raise ValueError("API Key 不能超过 4096 字符或包含控制字符")
        return SecretStr(normalized) if normalized else None

    @model_validator(mode="after")
    def validate_key_actions(self) -> "LLMSettingsRequest":
        if self.local_api_key is not None and self.clear_local_api_key:
            raise ValueError("不能同时设置并清除本地 API Key")
        if self.online_api_key is not None and self.clear_online_api_key:
            raise ValueError("不能同时设置并清除线上 API Key")
        return self


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
