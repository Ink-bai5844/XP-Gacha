import csv
from pathlib import Path


CSV_PATH = Path("data/gallery_info_no_name/JM_info_gender_bender.csv")
TAG_COLUMN = "标签"
LANG_COLUMN = "语言"
LANG_TAGS = {"英文", "日文"}


def split_items(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def merge_unique(items: list[str]) -> str:
    unique_items = []
    seen = set()

    for item in items:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)

    return ", ".join(unique_items)


def move_language_tags() -> None:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if TAG_COLUMN not in fieldnames or LANG_COLUMN not in fieldnames:
        raise ValueError(f"CSV 缺少必要列: {TAG_COLUMN} / {LANG_COLUMN}")

    updated_count = 0

    for row in rows:
        tags = split_items(row.get(TAG_COLUMN, ""))
        languages = split_items(row.get(LANG_COLUMN, ""))

        moved_languages = [tag for tag in tags if tag in LANG_TAGS]
        if not moved_languages:
            continue

        remaining_tags = [tag for tag in tags if tag not in LANG_TAGS]
        row[TAG_COLUMN] = ", ".join(remaining_tags)
        row[LANG_COLUMN] = merge_unique(languages + moved_languages)
        updated_count += 1

    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"已更新 {CSV_PATH}，共处理 {updated_count} 行。")


if __name__ == "__main__":
    move_language_tags()
