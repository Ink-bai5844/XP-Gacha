import argparse
import re
import sys
from pathlib import Path

from sqlalchemy import bindparam, create_engine, inspect, text
from sqlalchemy.engine import URL

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECRETS_FILE = PROJECT_ROOT / ".streamlit" / "secrets.toml"
DEFAULT_INPUT_FILE = PROJECT_ROOT / "tools" / "error.json"
TABLE_NAME = "gallery_info"
TITLE_TRANSLATION_COLUMN = "标题译文"
ID_PATTERN = re.compile(r'"id"\s*:\s*"([^"]+)"')

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_db_uri():
    if not SECRETS_FILE.exists():
        raise FileNotFoundError(f"未找到数据库配置文件：{SECRETS_FILE}")

    with SECRETS_FILE.open("rb") as f:
        secrets = tomllib.load(f)

    try:
        mysql_cfg = secrets["mysql"]
        return URL.create(
            "mysql+pymysql",
            username=str(mysql_cfg["user"]),
            password=str(mysql_cfg["password"]),
            host=str(mysql_cfg.get("host", "localhost")),
            port=int(mysql_cfg.get("port", 3306)),
            database=str(mysql_cfg["database"]),
            query={"charset": "utf8mb4"},
        )
    except KeyError as exc:
        raise KeyError(f"{SECRETS_FILE} 缺少 mysql.{exc.args[0]} 配置") from exc


def resolve_path(path_value: str | Path) -> Path:
    path = Path(str(path_value).strip())
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def extract_ids(input_file: Path) -> list[str]:
    if not input_file.exists():
        raise FileNotFoundError(f"未找到输入文件：{input_file}")

    content = input_file.read_text(encoding="utf-8", errors="replace")
    item_ids = []
    seen = set()
    for match in ID_PATTERN.finditer(content):
        item_id = match.group(1).strip()
        if item_id and item_id not in seen:
            item_ids.append(item_id)
            seen.add(item_id)
    return item_ids


def iter_chunks(items: list[str], chunk_size: int):
    for index in range(0, len(items), chunk_size):
        yield items[index : index + chunk_size]


def build_in_statement(sql: str):
    return text(sql).bindparams(bindparam("item_ids", expanding=True))


def preview_rows(conn, item_ids: list[str], limit: int) -> list[dict]:
    statement = build_in_statement(
        f"""
        SELECT `ID`, `标题`, `{TITLE_TRANSLATION_COLUMN}`
        FROM {TABLE_NAME}
        WHERE `ID` IN :item_ids
        ORDER BY `ID`
        LIMIT :limit
        """
    )
    return conn.execute(statement, {"item_ids": item_ids, "limit": limit}).mappings().all()


def count_matching_rows(conn, item_ids: list[str], chunk_size: int) -> tuple[int, int]:
    matched_count = 0
    nonempty_translation_count = 0
    count_statement = build_in_statement(
        f"""
        SELECT
            COUNT(*) AS matched_count,
            SUM(CASE WHEN `{TITLE_TRANSLATION_COLUMN}` IS NOT NULL
                      AND `{TITLE_TRANSLATION_COLUMN}` != ''
                     THEN 1 ELSE 0 END) AS nonempty_translation_count
        FROM {TABLE_NAME}
        WHERE `ID` IN :item_ids
        """
    )
    for chunk in iter_chunks(item_ids, chunk_size):
        row = conn.execute(count_statement, {"item_ids": chunk}).mappings().one()
        matched_count += int(row["matched_count"] or 0)
        nonempty_translation_count += int(row["nonempty_translation_count"] or 0)
    return matched_count, nonempty_translation_count


def clear_title_translations(
    input_file: Path,
    confirm: bool,
    empty_string: bool,
    preview_limit: int,
    chunk_size: int,
) -> None:
    item_ids = extract_ids(input_file)
    print(f"从 {input_file} 提取到 {len(item_ids)} 个唯一 ID。")
    if not item_ids:
        return

    engine = create_engine(load_db_uri())
    with engine.begin() as conn:
        inspector = inspect(conn)
        if not inspector.has_table(TABLE_NAME):
            raise RuntimeError(f"未找到 {TABLE_NAME} 表")

        table_columns = {column["name"] for column in inspector.get_columns(TABLE_NAME)}
        missing_columns = {"ID", "标题", TITLE_TRANSLATION_COLUMN} - table_columns
        if missing_columns:
            raise RuntimeError(f"{TABLE_NAME} 表缺少列：{', '.join(sorted(missing_columns))}")

        matched_count, nonempty_translation_count = count_matching_rows(conn, item_ids, chunk_size)
        print(f"数据库匹配到 {matched_count} 条记录，其中 {nonempty_translation_count} 条当前有标题译文。")

        rows = preview_rows(conn, item_ids, preview_limit)
        if rows:
            print(f"预览前 {len(rows)} 条：")
            for row in rows:
                print(
                    f"ID: {row['ID']} | 标题: {row.get('标题', '')} | "
                    f"标题译文: {row.get(TITLE_TRANSLATION_COLUMN, '')}"
                )

        if not confirm:
            print("当前为预览模式，未修改数据库。确认清空请追加 --confirm。")
            return

        replacement_sql = "''" if empty_string else "NULL"
        update_statement = build_in_statement(
            f"""
            UPDATE {TABLE_NAME}
            SET `{TITLE_TRANSLATION_COLUMN}` = {replacement_sql}
            WHERE `ID` IN :item_ids
              AND `{TITLE_TRANSLATION_COLUMN}` IS NOT NULL
              AND `{TITLE_TRANSLATION_COLUMN}` != ''
            """
        )

        updated_count = 0
        for chunk in iter_chunks(item_ids, chunk_size):
            result = conn.execute(update_statement, {"item_ids": chunk})
            updated_count += int(result.rowcount or 0)
        clear_value = "空字符串" if empty_string else "NULL"
        print(f"已将 {updated_count} 条记录的标题译文清空为 {clear_value}。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Extract all `"id": "NH..."` values from error.json and clear matching gallery_info title translations.'
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_FILE.relative_to(PROJECT_ROOT)),
        help="包含错误 JSON 文本的文件路径，默认 tools/error.json",
    )
    parser.add_argument("--confirm", action="store_true", help="确认执行 UPDATE；不加时只预览")
    parser.add_argument(
        "--empty-string",
        action="store_true",
        help="清空为 ''；默认清空为 NULL",
    )
    parser.add_argument("--preview-limit", type=int, default=30, help="预览匹配记录数量")
    parser.add_argument("--chunk-size", type=int, default=500, help="每批 SQL IN 的 ID 数量")
    args = parser.parse_args()

    if args.preview_limit < 1:
        raise ValueError("--preview-limit 必须 >= 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size 必须 >= 1")
    args.input = resolve_path(args.input)
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    clear_title_translations(
        input_file=parsed_args.input,
        confirm=parsed_args.confirm,
        empty_string=parsed_args.empty_string,
        preview_limit=parsed_args.preview_limit,
        chunk_size=parsed_args.chunk_size,
    )
