import os
import glob
import re
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
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

def migrate_data():
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
                print(f"成功读取: {file}")
            except Exception as e:
                print(f"读取表 {file} 时出错: {e}")
                
    if not all_dfs:
        print("未找到CSV数据！")
        return

    # 合并与基础清洗
    df = pd.concat(all_dfs, ignore_index=True)
    df = normalize_dataframe(df)

    # 写入 MySQL (if_exists='replace' 会自动建表并覆盖已有数据)
    print("正在写入 MySQL 数据库，请稍候...")
    df.to_sql(name='gallery_info', con=engine, if_exists='replace', index=False)

    # 导入完成后修正字段类型，并为 ID 字段建立唯一索引。
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE gallery_info MODIFY COLUMN ID VARCHAR(64);"))
        conn.execute(text("ALTER TABLE gallery_info ADD UNIQUE INDEX idx_id (ID);"))

    print(f"迁移完成！共写入 {len(df)} 条数据。")

if __name__ == "__main__":
    migrate_data()
