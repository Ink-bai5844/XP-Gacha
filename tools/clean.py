import csv
import re
from pathlib import Path


CSV_PATH = Path("data/gallery_info/JM_info_gender_bender_full.csv")
DATE_COLUMN = "上传日期"
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")


def normalize_date(value: str) -> str:
    match = DATE_PATTERN.search(str(value or "").strip())
    return match.group(1) if match else str(value or "").strip()


def clean_upload_dates() -> None:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if DATE_COLUMN not in fieldnames:
        raise ValueError(f"CSV 缺少必要列: {DATE_COLUMN}")

    updated_count = 0

    for row in rows:
        original_value = row.get(DATE_COLUMN, "")
        normalized_value = normalize_date(original_value)
        if normalized_value != original_value:
            row[DATE_COLUMN] = normalized_value
            updated_count += 1

    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"已更新 {CSV_PATH}，共清洗 {updated_count} 行上传日期。")


if __name__ == "__main__":
    clean_upload_dates()
