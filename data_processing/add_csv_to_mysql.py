import os
import glob
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
    df = df.drop_duplicates(subset=['链接']).copy()
    df = df.fillna('')
    df['标题'] = df.apply(lambda row: row.get('文件名', '') if row.get('标题', '') == '' else row.get('标题', ''), axis=1)

    print(f"共解析到 {len(df)} 条数据，准备写入临时表...")

    # 将数据覆盖写入临时表 (temp_gallery_info)
    df.to_sql(name='temp_gallery_info', con=engine, if_exists='replace', index=False)

    # 执行原生 SQL 进行合并 (REPLACE INTO 会根据唯一索引自动判断是 UPDATE 还是 INSERT)
    print("正在执行数据库增量同步...")
    with engine.begin() as conn:
        # 有相同的'链接'就先删后插，没有就直接插入
        merge_sql = """
            REPLACE INTO gallery_info 
            SELECT * FROM temp_gallery_info;
        """
        conn.execute(text(merge_sql))
        
        # 销毁临时表
        conn.execute(text("DROP TABLE temp_gallery_info;"))

    print("🎉 增量同步完成！")

if __name__ == "__main__":
    sync_csv_to_db()
