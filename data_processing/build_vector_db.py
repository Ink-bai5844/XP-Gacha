import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# 允许直接运行 data_processing 下的脚本时，也能导入项目根目录模块。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import LOCAL_MODEL_PATH as CONFIG_LOCAL_MODEL_PATH
from config import VECTOR_FILE as CONFIG_VECTOR_FILE

# 默认优先读取 config.py；环境变量只用于临时覆盖。
LOCAL_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", CONFIG_LOCAL_MODEL_PATH)
VECTOR_FILE = os.getenv("VECTOR_FILE", CONFIG_VECTOR_FILE)
SECRETS_FILE = PROJECT_ROOT / ".streamlit" / "secrets.toml"


def get_model_display_name():
    return os.path.basename(os.path.normpath(LOCAL_MODEL_PATH)) or LOCAL_MODEL_PATH


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


def build_vectors():
    engine = create_engine(load_db_uri())
    print("正在从 MySQL 读取数据...")
    # 过滤掉没有链接的无效数据
    df = pd.read_sql("SELECT * FROM gallery_info WHERE 链接 != ''", con=engine)
    df = df.fillna("")

    documents = []
    ids = []

    print("正在拼接语义文本...")
    for _, row in df.iterrows():
        title = row.get("标题", "")
        artist = row.get("作者", "")
        tags = row.get("标签", "")
        team = row.get("团队", "")
        lang = row.get("语言", "")

        # 精简语义字符串，提高模型注意力集中度
        semantic_text = f"标题《{title}》，作者 {artist}，团队 {team}，语言 {lang}，元素标签：{tags}。"
        if len(semantic_text) > 800:
            semantic_text = semantic_text[:800]

        documents.append(semantic_text)
        ids.append(str(row["链接"]))

    model_name = get_model_display_name()
    print(f"正在唤醒向量模型：{model_name} ...")
    model = SentenceTransformer(
        LOCAL_MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )

    print(f"开始批量编码 {len(documents)} 条数据 ({model_name})...")
    embeddings = model.encode(
        documents,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    os.makedirs(os.path.dirname(VECTOR_FILE), exist_ok=True)
    print(f"编码完成！正在将矩阵序列化保存至 {VECTOR_FILE}...")
    with open(VECTOR_FILE, "wb") as f:
        pickle.dump({"ids": ids, "embeddings": embeddings}, f)

    print("🎉 向量引擎构建完毕！")


def build_qwen3_vectors():
    build_vectors()


if __name__ == "__main__":
    build_vectors()
