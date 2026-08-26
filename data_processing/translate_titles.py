from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from sqlalchemy import inspect, text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from server.database import get_database_url, get_engine

try:
    with contextlib.redirect_stdout(io.StringIO()):
        from config import LM_STUDIO_API_BASE, LM_STUDIO_MODEL
except Exception:
    LM_STUDIO_API_BASE = "http://localhost:1234/v1"
    LM_STUDIO_MODEL = "local-model"

SECRETS_FILE = PROJECT_ROOT / ".streamlit" / "secrets.toml"
TITLE_TRANSLATION_COLUMN = "标题译文"
DEFAULT_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_JSONL_OUTPUT = PROJECT_ROOT / "data_processing" / "title_translation_results.jsonl"
DEFAULT_FAILED_JSONL_OUTPUT = PROJECT_ROOT / "data_processing" / "title_translation_failed_results.jsonl"


class APIRequestError(RuntimeError):
    def __init__(self, status_code: int, url: str, body: str, retryable: bool):
        self.status_code = status_code
        self.url = url
        self.body = body
        self.retryable = retryable
        super().__init__(format_api_error_message(status_code, url, body))


class APIResponseFormatError(RuntimeError):
    pass


class APIContentRejectedError(RuntimeError):
    pass


def compact_text(value: object, limit: int = 1200) -> str:
    text_value = str(value or "").strip()
    if not text_value:
        return "<empty>"
    if len(text_value) > limit:
        return text_value[:limit] + "...[truncated]"
    return text_value


def format_api_error_message(status_code: int, url: str, body: str) -> str:
    normalized_body = str(body or "").lower()
    if status_code == 429 or "too many requests" in normalized_body:
        return f"{status_code} Too many requests"
    return f"{status_code} Client Error for url: {url}; response: {body}"


def is_content_rejection_text(text_value: object) -> bool:
    normalized = str(text_value or "").strip().lower()
    rejection_markers = [
        "high risk",
        "request was rejected",
        "was considered high risk",
        "content policy",
        "policy violation",
        "cannot comply",
        "can't assist",
        "cannot assist",
        "无法协助",
        "不能协助",
        "拒绝",
    ]
    return any(marker in normalized for marker in rejection_markers)


def load_db_uri():
    return get_database_url()


def normalize_api_url(api_url: str) -> str:
    api_url = str(api_url or "").strip().rstrip("/")
    if not api_url:
        raise ValueError("API URL 不能为空")
    if api_url.endswith("/chat/completions"):
        return api_url
    return f"{api_url}/chat/completions"


def resolve_api_url(api_url: str, lm_studio: bool) -> str:
    if lm_studio and str(api_url or "").strip() == DEFAULT_API_URL:
        return normalize_api_url(LM_STUDIO_API_BASE)
    return normalize_api_url(api_url)


def ensure_title_translation_column(conn) -> None:
    inspector = inspect(conn)
    if not inspector.has_table("gallery_info"):
        raise RuntimeError("未找到 gallery_info 表")

    columns = {column["name"] for column in inspector.get_columns("gallery_info")}
    if TITLE_TRANSLATION_COLUMN in columns:
        return

    if "标题" in columns:
        conn.execute(text("ALTER TABLE gallery_info ADD COLUMN `标题译文` VARCHAR(1024) AFTER `标题`;"))
    else:
        conn.execute(text("ALTER TABLE gallery_info ADD COLUMN `标题译文` VARCHAR(1024);"))
    print("已补充标题译文列。", flush=True)


def load_title_rows(engine, start_index: int, end_index: int | None) -> tuple[list[dict], int]:
    with engine.begin() as conn:
        ensure_title_translation_column(conn)
        rows = conn.execute(
            text(
                """
                SELECT `ID`, `标题`, `标题译文`
                FROM gallery_info
                WHERE `ID` IS NOT NULL AND `ID` != ''
                ORDER BY `ID`
                """
            )
        ).mappings().all()

    selected_rows = []
    skipped_translated = 0
    for sequence, row in enumerate(rows, start=1):
        if sequence < start_index:
            continue
        if end_index is not None and sequence > end_index:
            break

        title = str(row.get("标题") or "").strip()
        translated_title = str(row.get(TITLE_TRANSLATION_COLUMN) or "").strip()
        if not title:
            continue
        if translated_title:
            skipped_translated += 1
            continue

        selected_rows.append(
            {
                "sequence": sequence,
                "id": str(row["ID"]),
                "title": title,
            }
        )

    return selected_rows, skipped_translated


def chunk_rows(rows: list[dict], batch_size: int) -> list[list[dict]]:
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def resolve_output_path(path_value: str | Path) -> Path:
    path = Path(str(path_value).strip())
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def build_messages(batch: list[dict]) -> list[dict]:
    payload = [{"id": row["id"], "title": row["title"]} for row in batch]
    return [
        {
            "role": "system",
            "content": (
                "你是漫画标题翻译助手。请把输入标题翻译成自然、简洁的简体中文。"
                "保留作品名中的人名、专有名词、编号、卷数、括号、系列标记和必要的罗马字；"
                "不要添加解释，不要改写 ID。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请翻译下面这些标题，并且只返回 JSON。JSON 格式必须为(前后不要有```)："
                '{"translations":[{"id":"原ID","title_zh":"中文译文"}]}。'
                "字段名必须使用 title_zh，不要使用 title 或其他字段名。"
                "输入：\n"
                f"{json.dumps(payload, ensure_ascii=False)}"
            ),
        },
    ]


def extract_json(text_value: str) -> dict:
    text_value = str(text_value or "").strip()
    if text_value.startswith("```"):
        text_value = text_value.strip("`").strip()
        if text_value.lower().startswith("json"):
            text_value = text_value[4:].strip()

    try:
        return json.loads(text_value)
    except json.JSONDecodeError:
        start = text_value.find("{")
        end = text_value.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text_value[start : end + 1])


def normalize_translation_response(data: object) -> dict[str, str]:
    translations: dict[str, str] = {}
    if isinstance(data, dict) and isinstance(data.get("translations"), list):
        items = data["translations"]
    elif isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = [{"id": key, "title_zh": value} for key, value in data.items()]
    else:
        raise ValueError("LLM 返回内容不是可解析的 JSON 对象或数组")

    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        title_zh = str(
            item.get("title_zh")
            or item.get("translation")
            or item.get("translated_title")
            or item.get("zh")
            or item.get("title")
            or ""
        ).strip()
        if item_id and title_zh:
            translations[item_id] = title_zh
    return translations


def call_translation_api(
    batch: list[dict],
    api_url: str,
    api_key: str,
    model: str,
    timeout: int,
    max_retries: int,
    temperature: float,
) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": build_messages(batch),
        "temperature": temperature,
        "stream": False,
    }

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 2):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            if response.status_code >= 400:
                raise APIRequestError(
                    response.status_code,
                    api_url,
                    compact_text(response.text),
                    retryable=response.status_code == 429 or response.status_code >= 500,
                )
            try:
                response_data = response.json()
            except json.JSONDecodeError as exc:
                raise APIResponseFormatError(
                    f"接口 HTTP 响应不是 JSON：{compact_text(response.text)}"
                ) from exc
            try:
                content = response_data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise APIResponseFormatError(
                    "接口返回不是 OpenAI Chat Completions 格式："
                    f"{compact_text(json.dumps(response_data, ensure_ascii=False))}"
                ) from exc
            if not str(content or "").strip():
                raise APIResponseFormatError(
                    "接口返回的 choices[0].message.content 为空："
                    f"{compact_text(json.dumps(response_data, ensure_ascii=False))}"
                )
            try:
                response_json = extract_json(content)
            except json.JSONDecodeError as exc:
                if is_content_rejection_text(content):
                    raise APIContentRejectedError(f"接口内容安全拒绝：{compact_text(content)}") from exc
                raise APIResponseFormatError(
                    f"LLM message.content 不是可解析 JSON：{compact_text(content)}"
                ) from exc
            translations = normalize_translation_response(response_json)
            if not translations:
                raise APIResponseFormatError(
                    "LLM JSON 中没有可用译文字段，期望 title_zh；实际返回："
                    f"{compact_text(json.dumps(response_json, ensure_ascii=False))}"
                )
            return {
                "response_json": response_json,
                "translations": translations,
                "raw_content": content,
            }
        except Exception as exc:
            last_error = exc
            if isinstance(exc, APIRequestError) and not exc.retryable:
                break
            if isinstance(exc, APIContentRejectedError):
                break
            if attempt <= max_retries:
                sleep_seconds = min(2**attempt, 20)
                print(f"批次请求失败，第 {attempt} 次重试前等待 {sleep_seconds} 秒：{exc}", flush=True)
                time.sleep(sleep_seconds)
    raise RuntimeError(f"批次请求失败：{last_error}") from last_error


def update_translations(engine, translations: dict[str, str]) -> int:
    if not translations:
        return 0

    updated_count = 0
    with engine.begin() as conn:
        for item_id, title_zh in translations.items():
            result = conn.execute(
                text(
                    """
                    UPDATE gallery_info
                    SET `标题译文` = :title_zh
                    WHERE `ID` = :item_id
                      AND (`标题译文` IS NULL OR `标题译文` = '')
                    """
                ),
                {"item_id": item_id, "title_zh": title_zh},
            )
            updated_count += int(result.rowcount or 0)
    return updated_count


def append_jsonl_record(output_path: Path, record: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def translate_titles(args: argparse.Namespace) -> None:
    api_url = resolve_api_url(args.api_url, args.lm_studio)
    api_key = args.api_key or ("" if args.lm_studio else os.getenv("OPENAI_API_KEY", ""))
    model = args.model or (LM_STUDIO_MODEL if args.lm_studio else DEFAULT_MODEL)
    jsonl_output = resolve_output_path(args.jsonl_output)
    failed_jsonl_output = resolve_output_path(args.failed_jsonl_output)
    db_write_enabled = not args.jsonl_only
    engine = get_engine()

    rows, skipped_translated = load_title_rows(engine, args.start_index, args.end_index)
    batches = chunk_rows(rows, args.batch_size)
    print(
        f"待翻译 {len(rows)} 条，已跳过已有译文 {skipped_translated} 条，批次数 {len(batches)}。",
        flush=True,
    )
    if not batches:
        return

    total_updated = 0
    total_failed = 0
    print(f"成功 JSONL 结果将追加保存到：{jsonl_output}", flush=True)
    print(f"失败 JSONL 结果将追加保存到：{failed_jsonl_output}", flush=True)
    if args.lm_studio:
        print("已启用 LM Studio 本地 API 模式：强制单线程请求，API Key 可为空。", flush=True)
    if not db_write_enabled:
        print("已启用仅写 JSONL 模式，不会回写 gallery_info.标题译文。", flush=True)
    max_workers = 1 if args.lm_studio else args.concurrency
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(
                call_translation_api,
                batch,
                api_url,
                api_key,
                model,
                args.request_timeout,
                args.max_retries,
                args.temperature,
            ): batch
            for batch in batches
        }

        for completed_count, future in enumerate(as_completed(future_to_batch), start=1):
            batch = future_to_batch[future]
            batch_label = f"{batch[0]['sequence']}-{batch[-1]['sequence']}"
            try:
                api_result = future.result()
                translations = api_result["translations"]
                valid_ids = {row["id"] for row in batch}
                translations = {
                    item_id: title_zh
                    for item_id, title_zh in translations.items()
                    if item_id in valid_ids and title_zh
                }
                updated_count = update_translations(engine, translations) if db_write_enabled else 0
                append_jsonl_record(
                    jsonl_output,
                    {
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "success",
                        "batch_index": completed_count,
                        "batch_total": len(batches),
                        "sequence_start": batch[0]["sequence"],
                        "sequence_end": batch[-1]["sequence"],
                        "model": model,
                        "api_url": api_url,
                        "db_write_enabled": db_write_enabled,
                        "items": [
                            {
                                "sequence": row["sequence"],
                                "id": row["id"],
                                "title": row["title"],
                            }
                            for row in batch
                        ],
                        "response_json": api_result["response_json"],
                        "normalized_translations": translations,
                        "updated_count": updated_count,
                    },
                )
                total_updated += updated_count
                if db_write_enabled:
                    result_label = f"写入 {updated_count} 条"
                else:
                    result_label = f"仅保存 JSONL {len(translations)} 条"
                print(
                    f"[{completed_count}/{len(batches)}] 序号 {batch_label}：返回 {len(translations)} 条，{result_label}。",
                    flush=True,
                )
            except Exception as exc:
                total_failed += len(batch)
                append_jsonl_record(
                    failed_jsonl_output,
                    {
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "failed",
                        "batch_index": completed_count,
                        "batch_total": len(batches),
                        "sequence_start": batch[0]["sequence"],
                        "sequence_end": batch[-1]["sequence"],
                        "model": model,
                        "api_url": api_url,
                        "db_write_enabled": db_write_enabled,
                        "items": [
                            {
                                "sequence": row["sequence"],
                                "id": row["id"],
                                "title": row["title"],
                            }
                            for row in batch
                        ],
                        "error": str(exc),
                    },
                )
                print(f"[{completed_count}/{len(batches)}] 序号 {batch_label} 失败：{exc}", flush=True)

    if db_write_enabled:
        print(f"标题翻译完成：写入 {total_updated} 条，失败待重试 {total_failed} 条。", flush=True)
    else:
        print(f"标题翻译完成：仅写入 JSONL，未回写数据库，失败待重试 {total_failed} 条。", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate gallery_info titles into Chinese with an OpenAI-compatible API.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="OpenAI 兼容接口地址或 Base URL")
    parser.add_argument("--api-key", default="", help="API Key；为空时读取 OPENAI_API_KEY")
    parser.add_argument(
        "--model",
        default=None,
        help=f"模型名；普通模式默认 {DEFAULT_MODEL}，--lm-studio 默认读取 config.py 的 LM_STUDIO_MODEL",
    )
    parser.add_argument("--batch-size", type=int, default=20, help="每组标题数量")
    parser.add_argument("--concurrency", type=int, default=3, help="并发请求组数")
    parser.add_argument("--start-index", type=int, default=1, help="按 ID 排序后的起始序号，1 开始")
    parser.add_argument("--end-index", type=int, default=None, help="按 ID 排序后的结束序号，包含该序号")
    parser.add_argument("--request-timeout", type=int, default=120, help="单次 API 请求超时秒数")
    parser.add_argument("--max-retries", type=int, default=2, help="单组请求失败重试次数")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument(
        "--jsonl-output",
        default=str(DEFAULT_JSONL_OUTPUT.relative_to(PROJECT_ROOT)),
        help="保存成功批次 LLM 返回 JSON 的 JSONL 文件路径",
    )
    parser.add_argument(
        "--failed-jsonl-output",
        default=str(DEFAULT_FAILED_JSONL_OUTPUT.relative_to(PROJECT_ROOT)),
        help="保存失败批次错误信息的 JSONL 文件路径",
    )
    parser.add_argument(
        "--jsonl-only",
        "--no-db-write",
        action="store_true",
        help="只把翻译结果追加写入 JSONL，不回写 gallery_info.标题译文",
    )
    parser.add_argument(
        "--lm-studio",
        action="store_true",
        help="启用本地 LM Studio 兼容 API 模式，默认读取 config.py 的 LM_STUDIO_API_BASE / LM_STUDIO_MODEL，且强制单线程请求",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size 必须 >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency 必须 >= 1")
    if args.start_index < 1:
        raise ValueError("--start-index 必须 >= 1")
    if args.end_index is not None and args.end_index < args.start_index:
        raise ValueError("--end-index 必须大于等于 --start-index")
    return args


if __name__ == "__main__":
    translate_titles(parse_args())
