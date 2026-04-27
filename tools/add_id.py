import csv
import re
from pathlib import Path


CSV_DIR = Path("data/gallery_info_no_name")
LINK_COLUMN = "链接"
ID_COLUMN = "ID"
ID_PREFIX = "NH"


def extract_nh_id(url: str) -> str:
    """从 nhentai 链接中提取数字 ID，并加上 NH 前缀。"""
    match = re.search(r"https?://nhentai\.net/g/(\d+)/?", (url or "").strip())
    if not match:
        return ""
    return f"{ID_PREFIX}{match.group(1)}"


def update_csv(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if LINK_COLUMN not in fieldnames:
        raise ValueError(f"{csv_path} 中未找到列: {LINK_COLUMN}")

    for row in rows:
        row[ID_COLUMN] = extract_nh_id(row.get(LINK_COLUMN, ""))

    new_fieldnames = [name for name in fieldnames if name != ID_COLUMN]
    new_fieldnames.insert(0, ID_COLUMN)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> None:
    if not CSV_DIR.exists():
        raise FileNotFoundError(f"目录不存在: {CSV_DIR}")

    csv_files = sorted(CSV_DIR.glob("*.csv"))
    if not csv_files:
        print(f"未找到 CSV 文件: {CSV_DIR}")
        return

    total_rows = 0

    for csv_path in csv_files:
        row_count = update_csv(csv_path)
        total_rows += row_count
        print(f"已更新 {csv_path}，共处理 {row_count} 行。")

    print(f"完成，共更新 {len(csv_files)} 个 CSV 文件，处理 {total_rows} 行。")


if __name__ == "__main__":
    main()
