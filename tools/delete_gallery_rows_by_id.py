import argparse
import re
from pathlib import Path

from sqlalchemy import bindparam, create_engine, inspect, text
from sqlalchemy.engine import URL

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECRETS_FILE = PROJECT_ROOT / ".streamlit" / "secrets.toml"
TABLE_NAME = "gallery_info"
PREVIEW_COLUMNS = ["ID", "标题", "标题译文", "作者", "上传日期"]


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


def split_ids(raw_values: list[str]) -> list[str]:
    item_ids = []
    seen = set()
    for raw_value in raw_values:
        for item_id in re.split(r"[\s,，]+", str(raw_value or "")):
            item_id = item_id.strip()
            if item_id and item_id not in seen:
                item_ids.append(item_id)
                seen.add(item_id)
    return item_ids


def load_ids(args: argparse.Namespace) -> list[str]:
    raw_values = list(args.ids or [])
    if args.id_file:
        id_file = Path(args.id_file)
        if not id_file.is_absolute():
            id_file = PROJECT_ROOT / id_file
        raw_values.append(id_file.read_text(encoding="utf-8"))

    if not raw_values:
        raw_values.append(input("请输入要删除的 ID，多个 ID 可用空格或逗号分隔："))

    item_ids = split_ids(raw_values)
    if not item_ids:
        raise ValueError("未提供有效 ID")
    return item_ids


def build_preview_sql(columns: list[str]):
    selected_columns = ", ".join(f"`{column}`" for column in columns)
    return text(f"SELECT {selected_columns} FROM {TABLE_NAME} WHERE ID IN :item_ids").bindparams(
        bindparam("item_ids", expanding=True)
    )


def delete_gallery_rows(item_ids: list[str], confirm: bool = False) -> None:
    engine = create_engine(load_db_uri())

    with engine.begin() as conn:
        inspector = inspect(conn)
        if not inspector.has_table(TABLE_NAME):
            raise RuntimeError(f"未找到 {TABLE_NAME} 表")

        table_columns = {column["name"] for column in inspector.get_columns(TABLE_NAME)}
        preview_columns = [column for column in PREVIEW_COLUMNS if column in table_columns]
        if "ID" not in preview_columns:
            raise RuntimeError(f"{TABLE_NAME} 表缺少 ID 列")

        rows = conn.execute(build_preview_sql(preview_columns), {"item_ids": item_ids}).mappings().all()
        if not rows:
            print("没有匹配到任何条目。")
            return

        print(f"匹配到 {len(rows)} 条记录：")
        for row in rows:
            preview = " | ".join(f"{column}: {row.get(column, '')}" for column in preview_columns)
            print(preview)

        missing_ids = sorted(set(item_ids) - {str(row["ID"]) for row in rows})
        if missing_ids:
            print(f"未匹配到 {len(missing_ids)} 个 ID：{', '.join(missing_ids)}")

        if not confirm:
            print("当前为预览模式，未删除数据库记录。确认删除请追加 --confirm。")
            return

        delete_statement = text(f"DELETE FROM {TABLE_NAME} WHERE ID IN :item_ids").bindparams(
            bindparam("item_ids", expanding=True)
        )
        result = conn.execute(delete_statement, {"item_ids": [str(row["ID"]) for row in rows]})
        print(f"已删除 {int(result.rowcount or 0)} 条数据库记录。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete gallery_info rows by exact ID.")
    parser.add_argument("ids", nargs="*", help="要删除的 ID，支持空格或逗号分隔")
    parser.add_argument("--id-file", default="", help="从文本文件读取要删除的 ID")
    parser.add_argument("--confirm", action="store_true", help="确认执行 DELETE；不加时只预览")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    delete_gallery_rows(load_ids(args), confirm=args.confirm)
