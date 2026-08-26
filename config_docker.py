import json
import os
import re


def env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return int(default)


def env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


# 目录配置
ONLINE_IMG_DIR = os.getenv("ONLINE_IMG_DIR", "onlineimgtmp")
IMG_CACHE_DIR = os.getenv("IMG_CACHE_DIR", "localimgtmp")
CACHE_DIR = os.getenv("CACHE_DIR", "datacache")
B64_CACHE_DIR = os.getenv("B64_CACHE_DIR", "b64_cache")
MODEL_DIR = os.getenv("MODEL_DIR", "models")

# 历史推荐偏好配置
HISTORY_RECOMMENDATION_CACHE_SIZE = env_int("HISTORY_RECOMMENDATION_CACHE_SIZE", 50)
HISTORY_CACHE_FILE = os.getenv(
    "HISTORY_CACHE_FILE",
    os.path.join(CACHE_DIR, "recommendation_history.json"),
)
HISTORY_LINK_TRACKING_HOST = os.getenv("HISTORY_LINK_TRACKING_HOST", "0.0.0.0")
HISTORY_LINK_TRACKING_PUBLIC_HOST = os.getenv("HISTORY_LINK_TRACKING_PUBLIC_HOST", "127.0.0.1")
HISTORY_LINK_TRACKING_PORT = env_int("HISTORY_LINK_TRACKING_PORT", 8765)

# 向量与模型
VECTOR_FILE = os.getenv("VECTOR_FILE", "manga_vectors/manga_vectors_Qwen3.pkl")
IMG_VECTOR_FILE = os.getenv("IMG_VECTOR_FILE", "manga_vectors/clip_image_index.pkl")
LOCAL_MODEL_PATH = os.getenv(
    "LOCAL_MODEL_PATH",
    os.path.join(MODEL_DIR, "Qwen3-Embedding-0.6B"),
)
CLIP_MODEL_PATH = os.getenv(
    "CLIP_MODEL_PATH",
    os.path.join(MODEL_DIR, "clip-vit-base-patch32"),
)

# Docker 中建议把宿主机漫画目录挂载到 /library。
BASE_DIR = os.getenv("XP_GACHA_BASE_DIR", "/library")

# LM Studio 配置。容器访问宿主机服务时用 host.docker.internal。
LM_STUDIO_API_BASE = os.getenv("LM_STUDIO_API_BASE", "http://host.docker.internal:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "local-model")

# 线上 AI 配置
ONLINE_API_BASE = os.getenv("ONLINE_API_BASE", "your_api_base")
ONLINE_API_KEY = os.getenv("ONLINE_API_KEY", "your_api_key")
ONLINE_MODEL = os.getenv("ONLINE_MODEL", "deepseek-v4-flash")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "你是‘地下金库’的智能助手。你的性格冷静、专业。"
    "‘地下金库’是一个储存大量漫画的数据库。我会为你提供一部分当前的库存数据作为参考。"
    "请结合这些数据回答问题。如果数据中没有相关内容，请基于你的通用知识库回答。"
    "如要推荐条目，要附上可点击的完整跳转链接。",
)

# 预设Tag权重配置
INITIAL_TAG_WEIGHTS = {
    "NTR(netorare)": env_float("INITIAL_TAG_WEIGHT_NTR", -2.0),
}

# 线上封面与 NH 采集。容器内如需走宿主机代理可设 ONLINE_COVER_PROXY=http://host.docker.internal:7890
ONLINE_COVER_FETCH_ENABLED = os.getenv("ONLINE_COVER_FETCH_ENABLED", "1").strip().lower() not in ("0", "false", "no")
ONLINE_COVER_PROXY = os.getenv("ONLINE_COVER_PROXY", "")
ONLINE_COVER_FETCH_CONCURRENCY = env_int("ONLINE_COVER_FETCH_CONCURRENCY", 6)

# 检索结果上限
SEMANTIC_SEARCH_TOP_K = env_int("SEMANTIC_SEARCH_TOP_K", 5000)
COVER_SEARCH_TOP_K = env_int("COVER_SEARCH_TOP_K", 5000)

# 每页条目显示上限
MAX_DISPLAY = env_int("MAX_DISPLAY", 500)

for directory in [ONLINE_IMG_DIR, IMG_CACHE_DIR, CACHE_DIR, B64_CACHE_DIR, MODEL_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_text_config(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            return set(re.findall(r"'(.*?)'", content))
    except FileNotFoundError:
        return set()


def load_json_config(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


STOP_TAGS = load_text_config("dictionaries/STOP_TAGS.txt")
SEMANTIC_MAP = load_json_config("dictionaries/SEMANTIC_MAP.json")
TITLE_STOP_WORDS = load_text_config("dictionaries/TITLE_STOP_WORDS.txt")
TITLE_SEMANTIC_MAP = load_json_config("dictionaries/TITLE_SEMANTIC_MAP.json")

print("Docker 配置已就绪！")
print(f"TITLE_STOP_WORDS 数量: {len(TITLE_STOP_WORDS)}")
print(f"TITLE_SEMANTIC_MAP 数量: {len(TITLE_SEMANTIC_MAP)}")
print(f"STOP_TAGS 数量: {len(STOP_TAGS)}")
print(f"SEMANTIC_MAP 数量: {len(SEMANTIC_MAP)}")
