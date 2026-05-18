from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECRETS_FILE = PROJECT_ROOT / ".streamlit" / "secrets.toml"
FULLTEXT_INDEX_NAME = "ft_gallery_search"


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


def get_engine():
    return create_engine(load_db_uri())


def get_index_names(inspector):
    return {index["name"] for index in inspector.get_indexes("gallery_info")}


def get_primary_key_columns(inspector):
    return inspector.get_pk_constraint("gallery_info").get("constrained_columns", [])


def execute_step(conn, sql, label):
    print(label)
    conn.execute(text(sql))


def load_column_meta(conn):
    rows = conn.execute(
        text(
            """
            SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'gallery_info'
            """
        )
    ).mappings().all()
    return {row["COLUMN_NAME"]: dict(row) for row in rows}


def ensure_column_type(conn, column_meta, column_name, expected_type, sql):
    meta = column_meta.get(column_name)
    if not meta:
        return
    current_type = str(meta["COLUMN_TYPE"]).lower()
    if current_type == expected_type.lower():
        print(f"{column_name} 类型已是 {expected_type}")
        return
    execute_step(conn, sql, f"正在优化 {column_name} 类型：{current_type} -> {expected_type}")


def ensure_fulltext_index(conn):
    index_exists = conn.execute(
        text(
            """
            SELECT COUNT(*)
            FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'gallery_info'
              AND INDEX_NAME = :index_name
            """
        ),
        {"index_name": FULLTEXT_INDEX_NAME},
    ).scalar()
    if index_exists:
        print(f"全文索引已存在：{FULLTEXT_INDEX_NAME}")
        return

    try:
        execute_step(
            conn,
            """
            ALTER TABLE gallery_info
            ADD FULLTEXT INDEX ft_gallery_search (`标题`, `标签`, `作者`, `团队`) WITH PARSER ngram;
            """,
            "正在创建 ngram FULLTEXT 索引：标题/标签/作者/团队",
        )
    except Exception as exc:
        print(f"ngram FULLTEXT 创建失败，尝试普通 FULLTEXT：{exc}")
        execute_step(
            conn,
            """
            ALTER TABLE gallery_info
            ADD FULLTEXT INDEX ft_gallery_search (`标题`, `标签`, `作者`, `团队`);
            """,
            "正在创建普通 FULLTEXT 索引：标题/标签/作者/团队",
        )


def optimize_gallery_schema(engine=None):
    engine = engine or get_engine()
    with engine.begin() as conn:
        inspector = inspect(conn)
        if not inspector.has_table("gallery_info"):
            raise RuntimeError("未找到 gallery_info 表")

        columns = {column["name"] for column in inspector.get_columns("gallery_info")}
        if "ID" not in columns:
            execute_step(conn, "ALTER TABLE gallery_info ADD COLUMN ID VARCHAR(32) FIRST;", "正在补充 ID 列")
            execute_step(
                conn,
                """
                UPDATE gallery_info
                SET ID = CASE
                    WHEN 链接 LIKE '%/g/%' THEN CONCAT('NH', TRIM(BOTH '/' FROM SUBSTRING_INDEX(链接, '/g/', -1)))
                    WHEN 链接 LIKE '%/album/%' THEN CONCAT('JM', TRIM(BOTH '/' FROM SUBSTRING_INDEX(链接, '/album/', -1)))
                    ELSE ''
                END
                WHERE ID IS NULL OR ID = '';
                """,
                "正在从链接回填 ID",
            )

        execute_step(conn, "UPDATE gallery_info SET ID = TRIM(ID) WHERE ID IS NOT NULL;", "正在清理 ID 空白")

        blank_count = conn.execute(
            text("SELECT COUNT(*) FROM gallery_info WHERE ID IS NULL OR ID = ''")
        ).scalar()
        if blank_count:
            raise RuntimeError(f"存在 {blank_count} 条空 ID，无法建立主键")

        column_meta = load_column_meta(conn)
        id_meta = column_meta.get("ID", {})
        if str(id_meta.get("COLUMN_TYPE", "")).lower() != "varchar(32)" or id_meta.get("IS_NULLABLE") != "NO":
            execute_step(conn, "ALTER TABLE gallery_info MODIFY COLUMN ID VARCHAR(32) NOT NULL;", "正在优化 ID 类型")
        else:
            print("ID 类型已是 VARCHAR(32) NOT NULL")

        inspector = inspect(conn)
        primary_key_columns = get_primary_key_columns(inspector)
        if primary_key_columns != ["ID"]:
            execute_step(conn, "ALTER TABLE gallery_info ADD PRIMARY KEY (ID);", "正在将 ID 设置为主键")

        inspector = inspect(conn)
        index_names = get_index_names(inspector)
        if "idx_id" in index_names:
            execute_step(conn, "DROP INDEX idx_id ON gallery_info;", "正在移除重复唯一索引 idx_id")

        column_meta = load_column_meta(conn)
        type_steps = [
            ("链接", "varchar(64)", "ALTER TABLE gallery_info MODIFY COLUMN `链接` VARCHAR(64);"),
            ("文件名", "varchar(256)", "ALTER TABLE gallery_info MODIFY COLUMN `文件名` VARCHAR(256);"),
            ("标题", "varchar(512)", "ALTER TABLE gallery_info MODIFY COLUMN `标题` VARCHAR(512);"),
            ("作者", "varchar(768)", "ALTER TABLE gallery_info MODIFY COLUMN `作者` VARCHAR(768);"),
            ("团队", "varchar(128)", "ALTER TABLE gallery_info MODIFY COLUMN `团队` VARCHAR(128);"),
            ("语言", "varchar(64)", "ALTER TABLE gallery_info MODIFY COLUMN `语言` VARCHAR(64);"),
            ("页数", "int unsigned", "ALTER TABLE gallery_info MODIFY COLUMN `页数` INT UNSIGNED;"),
            ("上传日期", "varchar(10)", "ALTER TABLE gallery_info MODIFY COLUMN `上传日期` VARCHAR(10);"),
        ]
        for column_name, expected_type, sql in type_steps:
            try:
                ensure_column_type(conn, column_meta, column_name, expected_type, sql)
            except Exception as exc:
                print(f"优化 {column_name} 类型失败，已跳过：{exc}")

        ensure_fulltext_index(conn)
        execute_step(conn, "ANALYZE TABLE gallery_info;", "正在刷新表统计信息")

    print("MySQL 表结构优化完成。")


if __name__ == "__main__":
    optimize_gallery_schema()
