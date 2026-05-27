import argparse
import json
import shutil
from pathlib import Path


JSONL_PATH = Path("data_processing/title_translation_results.jsonl")


def clean_failed_entries(jsonl_path: Path = JSONL_PATH, keep_backup: bool = True) -> None:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"文件不存在: {jsonl_path}")

    kept_lines = []
    total_count = 0
    removed_count = 0
    invalid_count = 0

    with jsonl_path.open("r", encoding="utf-8") as file:
        for line in file:
            total_count += 1
            stripped_line = line.strip()
            if not stripped_line:
                kept_lines.append(line)
                continue

            try:
                record = json.loads(stripped_line)
            except json.JSONDecodeError:
                invalid_count += 1
                kept_lines.append(line)
                continue

            if isinstance(record, dict) and record.get("status") == "failed":
                removed_count += 1
                continue

            kept_lines.append(line)

    if keep_backup:
        backup_path = jsonl_path.with_suffix(jsonl_path.suffix + ".bak")
        shutil.copy2(jsonl_path, backup_path)
        print(f"已备份原文件: {backup_path}")

    with jsonl_path.open("w", encoding="utf-8", newline="") as file:
        file.writelines(kept_lines)

    print(
        f"清理完成：总行数 {total_count}，删除 failed {removed_count} 行，"
        f"保留 {len(kept_lines)} 行，无法解析保留 {invalid_count} 行。"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove failed entries from title translation JSONL.")
    parser.add_argument(
        "--jsonl-path",
        default=str(JSONL_PATH),
        help="要清理的标题翻译 JSONL 文件路径",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不生成 .bak 备份文件",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    clean_failed_entries(Path(args.jsonl_path), keep_backup=not args.no_backup)
