import html
import os
import re
import unicodedata
import uuid
from pathlib import Path

import pandas as pd


ID_COLUMN = "ID"
LINK_COLUMN = "链接"
FILE_NAME_COLUMN = "文件名"
TITLE_COLUMN = "标题"
TITLE_TRANSLATION_COLUMN = "标题译文"


def _cell_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _nonempty_text(value) -> str:
    text = _cell_text(value)
    return text if text.strip() else ""


def _normalize_id(value) -> str:
    return unicodedata.normalize("NFKC", _cell_text(value)).strip().upper()


def _normalize_link(value) -> str:
    return unicodedata.normalize("NFKC", _cell_text(value)).strip().rstrip("/")


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def _load_existing_translations(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load non-empty translations before the target file is replaced."""
    by_id: dict[str, str] = {}
    by_link: dict[str, str] = {}
    if not path.exists():
        return by_id, by_link

    existing = _read_csv(path)
    if TITLE_TRANSLATION_COLUMN not in existing.columns:
        return by_id, by_link

    for _, row in existing.iterrows():
        translation = _nonempty_text(row.get(TITLE_TRANSLATION_COLUMN, ""))
        if not translation:
            continue

        item_id = _normalize_id(row.get(ID_COLUMN, ""))
        link = _normalize_link(row.get(LINK_COLUMN, ""))
        if item_id:
            by_id[item_id] = translation
        if link:
            by_link[link] = translation

    return by_id, by_link


def _place_column_after(columns: list[str], column: str, after: str) -> list[str]:
    ordered = [name for name in columns if name != column]
    ordered.insert(ordered.index(after) + 1, column)
    return ordered


def process_gallery_data(csv_file, txt_file, output_file):
    input_path = Path(csv_file)
    txt_path = Path(txt_file)
    output_path = Path(output_file)

    try:
        df = _read_csv(input_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 {csv_file}")
        return

    missing_columns = [
        column for column in (LINK_COLUMN, TITLE_COLUMN) if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"CSV 缺少必要列: {' / '.join(missing_columns)}")

    # Parse the link export into {URL: local file name}.
    url_to_name = {}
    pattern = re.compile(r'HREF="(.*?)".*?>(.*?)</A>')
    try:
        with txt_path.open("r", encoding="utf-8") as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    url_to_name[match.group(1)] = html.unescape(match.group(2))
    except FileNotFoundError:
        print(f"错误：找不到文件 {txt_file}")
        return

    # Keep an already populated target authoritative. This prevents rerunning
    # the filename enrichment step from erasing database-exported translations.
    existing_by_id, existing_by_link = _load_existing_translations(output_path)
    input_translations = (
        df[TITLE_TRANSLATION_COLUMN].map(_nonempty_text)
        if TITLE_TRANSLATION_COLUMN in df.columns
        else pd.Series("", index=df.index, dtype="object")
    )

    translations = []
    for index, row in df.iterrows():
        item_id = _normalize_id(row.get(ID_COLUMN, ""))
        link = _normalize_link(row.get(LINK_COLUMN, ""))
        inherited = existing_by_id.get(item_id, "") if item_id else ""
        if not inherited and link:
            inherited = existing_by_link.get(link, "")
        translations.append(inherited or input_translations.at[index])
    df[TITLE_TRANSLATION_COLUMN] = translations

    mapped_file_names = df[LINK_COLUMN].map(url_to_name).fillna("")
    if FILE_NAME_COLUMN in df.columns:
        existing_file_names = df[FILE_NAME_COLUMN].map(_cell_text)
        mapped_file_names = mapped_file_names.where(
            mapped_file_names.astype(str).str.strip() != "", existing_file_names
        )
        df = df.drop(columns=[FILE_NAME_COLUMN])
    df.insert(1, FILE_NAME_COLUMN, mapped_file_names)

    df = df[_place_column_after(list(df.columns), TITLE_TRANSLATION_COLUMN, TITLE_COLUMN)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        df.to_csv(temporary_path, index=False, encoding="utf-8-sig")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"处理完成！已生成新表格：{output_file}")


if __name__ == "__main__":
    process_gallery_data(
        "data/gallery_info_origin/JM_info_yuri.csv",
        "data/local_data/NH_all.txt",
        "data/gallery_info/JM_info_yuri_full.csv",
    )
