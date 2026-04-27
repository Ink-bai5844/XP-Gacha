import os
import glob
import re
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECRETS_FILE = PROJECT_ROOT / ".streamlit" / "secrets.toml"
CSV_DIR = "data/gallery_info"
ID_COLUMN = "ID"
LINK_COLUMN = "链接"
DB_COLUMNS = [ID_COLUMN, LINK_COLUMN, "文件名", "标题", "标签", "作者", "团队", "语言", "页数", "上传日期"]


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
    except KeyError as e:
        raise KeyError(f"{SECRETS_FILE} 缺少 mysql.{e.args[0]} 配置") from e


engine = create_engine(load_db_uri())


def extract_gallery_id(url):
    url = str(url or "").strip()
    if not url:
        return ""

    nh_match = re.search(r"/g/(\d+)/?", url)
    if nh_match:
        return f"NH{nh_match.group(1)}"

    jm_match = re.search(r"/album/(\d+)/?", url)
    if jm_match:
        return f"JM{jm_match.group(1)}"

    return ""


def normalize_dataframe(df):
    df = df.copy()

    if LINK_COLUMN not in df.columns:
        raise ValueError(f"CSV 缺少必要列: {LINK_COLUMN}")

    if ID_COLUMN not in df.columns:
        df[ID_COLUMN] = ""

    for column in DB_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    df = df.fillna("")
    df[ID_COLUMN] = df[ID_COLUMN].astype(str).str.strip()
    df[LINK_COLUMN] = df[LINK_COLUMN].astype(str).str.strip()

    missing_id_mask = df[ID_COLUMN] == ""
    df.loc[missing_id_mask, ID_COLUMN] = df.loc[missing_id_mask, LINK_COLUMN].apply(extract_gallery_id)

    df["标题"] = df.apply(
        lambda row: row.get("文件名", "") if row.get("标题", "") == "" else row.get("标题", ""),
        axis=1,
    )
    df = df.drop_duplicates(subset=[ID_COLUMN]).copy()
    df = df[DB_COLUMNS]
    return df


def ensure_gallery_table_schema(conn):
    inspector = inspect(conn)
    if not inspector.has_table("gallery_info"):
        return

    columns = {column["name"] for column in inspector.get_columns("gallery_info")}

    if ID_COLUMN not in columns:
        conn.execute(text("ALTER TABLE gallery_info ADD COLUMN ID VARCHAR(64) FIRST;"))
        conn.execute(
            text(
                """
                UPDATE gallery_info
                SET ID = CASE
                    WHEN 链接 LIKE '%/g/%' THEN CONCAT('NH', TRIM(BOTH '/' FROM SUBSTRING_INDEX(链接, '/g/', -1)))
                    WHEN 链接 LIKE '%/album/%' THEN CONCAT('JM', TRIM(BOTH '/' FROM SUBSTRING_INDEX(链接, '/album/', -1)))
                    ELSE ''
                END
                WHERE ID IS NULL OR ID = '';
                """
            )
        )

    conn.execute(text("ALTER TABLE gallery_info MODIFY COLUMN ID VARCHAR(64);"))
    if LINK_COLUMN in columns:
        conn.execute(text("ALTER TABLE gallery_info MODIFY COLUMN 链接 VARCHAR(255);"))

    index_names = {index["name"] for index in inspector.get_indexes("gallery_info")}
    if "idx_link" in index_names:
        conn.execute(text("DROP INDEX idx_link ON gallery_info;"))
        index_names.remove("idx_link")
    if "idx_id" not in index_names:
        conn.execute(text("ALTER TABLE gallery_info ADD UNIQUE INDEX idx_id (ID);"))

def sync_csv_to_db():
    print("开始读取本地 CSV 文件...")
    all_dfs = []
    if os.path.exists(CSV_DIR):
        csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
        for file in csv_files:
            try:
                try:
                    temp_df = pd.read_csv(file, encoding='utf-8-sig')
                except UnicodeDecodeError:
                    temp_df = pd.read_csv(file, encoding='utf-8')
                all_dfs.append(temp_df)
            except Exception as e:
                print(f"读取表 {file} 时出错: {e}")
                
    if not all_dfs:
        print("未找到任何 CSV 数据！")
        return

    # 清洗合并本次读取到的数据
    df = pd.concat(all_dfs, ignore_index=True)
    df = normalize_dataframe(df)

    print(f"共解析到 {len(df)} 条数据，准备写入临时表...")

    # 将数据覆盖写入临时表 (temp_gallery_info)
    df.to_sql(name='temp_gallery_info', con=engine, if_exists='replace', index=False)

    # 执行原生 SQL 进行合并 (REPLACE INTO 会根据唯一索引自动判断是 UPDATE 还是 INSERT)
    print("正在执行数据库增量同步...")
    with engine.begin() as conn:
        ensure_gallery_table_schema(conn)

        inspector = inspect(conn)
        if not inspector.has_table("gallery_info"):
            conn.execute(text("CREATE TABLE gallery_info LIKE temp_gallery_info;"))
            conn.execute(text("ALTER TABLE gallery_info MODIFY COLUMN ID VARCHAR(64);"))
            conn.execute(text("ALTER TABLE gallery_info MODIFY COLUMN 链接 VARCHAR(255);"))
            conn.execute(text("ALTER TABLE gallery_info ADD UNIQUE INDEX idx_id (ID);"))

        # 有相同的'链接'就先删后插，没有就直接插入
        merge_sql = """
            REPLACE INTO gallery_info (`ID`, `链接`, `文件名`, `标题`, `标签`, `作者`, `团队`, `语言`, `页数`, `上传日期`)
            SELECT `ID`, `链接`, `文件名`, `标题`, `标签`, `作者`, `团队`, `语言`, `页数`, `上传日期`
            FROM temp_gallery_info;
        """
        conn.execute(text(merge_sql))
        
        # 销毁临时表
        conn.execute(text("DROP TABLE temp_gallery_info;"))

    print("🎉 增量同步完成！")

if __name__ == "__main__":
    sync_csv_to_db()
