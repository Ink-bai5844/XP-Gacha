import os
import argparse
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


def get_model_display_name(model_path=None):
    model_path = str(model_path or LOCAL_MODEL_PATH)
    return os.path.basename(os.path.normpath(model_path)) or model_path


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


def build_vectors(
    model_path=None,
    vector_file=None,
    batch_size=16,
    max_text_length=800,
    sql_query="SELECT * FROM gallery_info WHERE ID != ''",
):
    model_path = str(model_path or LOCAL_MODEL_PATH)
    vector_file = str(vector_file or VECTOR_FILE)
    batch_size = int(batch_size)
    max_text_length = int(max_text_length)

    engine = create_engine(load_db_uri())
    print("正在从 MySQL 读取数据...")
    df = pd.read_sql(sql_query, con=engine)
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
        if max_text_length > 0 and len(semantic_text) > max_text_length:
            semantic_text = semantic_text[:max_text_length]

        documents.append(semantic_text)
        ids.append(str(row["ID"]))

    model_name = get_model_display_name(model_path)
    print(f"正在唤醒向量模型：{model_name} ...")
    model = SentenceTransformer(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    print(f"开始批量编码 {len(documents)} 条数据 ({model_name})...")
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    output_dir = os.path.dirname(vector_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    print(f"编码完成！正在将矩阵序列化保存至 {vector_file}...")
    with open(vector_file, "wb") as f:
        pickle.dump({"ids": ids, "embeddings": embeddings}, f)

    print("🎉 向量引擎构建完毕！")


def build_qwen3_vectors():
    build_vectors()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Build text semantic vectors for gallery_info.")
    parser.add_argument("--model-path", default=LOCAL_MODEL_PATH, help="Local embedding model path.")
    parser.add_argument("--vector-file", default=VECTOR_FILE, help="Output pickle file path.")
    parser.add_argument("--batch-size", type=int, default=16, help="Encoding batch size.")
    parser.add_argument("--max-text-length", type=int, default=800, help="Max semantic text length per row.")
    parser.add_argument(
        "--sql-query",
        default="SELECT * FROM gallery_info WHERE ID != ''",
        help="SQL query used to load source rows.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    build_vectors(
        model_path=args.model_path,
        vector_file=args.vector_file,
        batch_size=args.batch_size,
        max_text_length=args.max_text_length,
        sql_query=args.sql_query,
    )
