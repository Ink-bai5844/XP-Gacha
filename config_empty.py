import os
import json
import re

# 目录配置
# 这些目录默认相对项目根目录。
ONLINE_IMG_DIR = "onlineimgtmp"
IMG_CACHE_DIR = "localimgtmp"
CACHE_DIR = "datacache"
B64_CACHE_DIR = "b64_cache"
MODEL_DIR = "models"

# 历史推荐偏好配置
# 最近 N 次通过应用打开的网络链接/本地目录会参与一条独立历史加权。
HISTORY_RECOMMENDATION_CACHE_SIZE = 50
HISTORY_CACHE_FILE = os.path.join(CACHE_DIR, "recommendation_history.json")
HISTORY_LINK_TRACKING_HOST = "127.0.0.1"
HISTORY_LINK_TRACKING_PORT = 8765

# 文本语义向量文件
VECTOR_FILE = "manga_vectors/manga_vectors_Qwen3.pkl"

# 封面图片向量文件
IMG_VECTOR_FILE = "manga_vectors/clip_image_index.pkl"

# 本地 embedding 模型目录
LOCAL_MODEL_PATH = os.path.join(MODEL_DIR, "Qwen3-Embedding-0.6B")

# 本地 CLIP 模型目录
CLIP_MODEL_PATH = os.path.join(MODEL_DIR, "clip-vit-base-patch32")

# 本地漫画根目录
BASE_DIR = r"D:\YourMangaLibrary"

# LM Studio 配置
LM_STUDIO_API_BASE = "http://localhost:1234/v1"
LM_STUDIO_MODEL = "local-model"

# 线上 AI 配置
ONLINE_API_BASE = "your_api_base" 
ONLINE_API_KEY = "your_api_key"
ONLINE_MODEL = "deepseek-v4-flash"
SYSTEM_PROMPT = (
    "你是‘地下金库’的智能助手。你的性格冷静、专业。 "
    "‘地下金库’是一个储存大量漫画的数据库。我会为你提供一部分当前的库存数据作为参考。 "
    "请结合这些数据回答问题。如果数据中没有相关内容，请基于你的通用知识库回答。"
    "如要推荐条目，要附上可点击的完整跳转链接。"
)

# 预设Tag权重配置
INITIAL_TAG_WEIGHTS = {
    'NTR(netorare)': -2.0
}

# 线上封面实时抓取
# 当 b64_cache、onlineimgtmp、本地目录都取不到封面时，直接从图源站点实时抓取并回写缓存
ONLINE_COVER_FETCH_ENABLED = True
# 实时抓取使用的代理（与爬虫一致）；设为 "" 表示直连
ONLINE_COVER_PROXY = "http://127.0.0.1:7890"
# 后台并发抓取的线程数（修改后需重启应用生效）
ONLINE_COVER_FETCH_CONCURRENCY = 6

# 检索结果上限
# 语义检索会在当前候选集中保留前 N 个文本最相似条目
SEMANTIC_SEARCH_TOP_K = 5000

# 封面检索会在当前候选集中保留前 N 个图片最相似条目
COVER_SEARCH_TOP_K = 5000

# 每页条目显示上限
MAX_DISPLAY = 500

# 初始化目录
for directory in [ONLINE_IMG_DIR, IMG_CACHE_DIR, CACHE_DIR, B64_CACHE_DIR, MODEL_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_text_config(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            return set(re.findall(r"'(.*?)'", content))
    except FileNotFoundError:
        return set()

def load_json_config(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# 预加载字典资源
STOP_TAGS = load_text_config('dictionaries/STOP_TAGS.txt')
SEMANTIC_MAP = load_json_config('dictionaries/SEMANTIC_MAP.json')
TITLE_STOP_WORDS = load_text_config('dictionaries/TITLE_STOP_WORDS.txt')
TITLE_SEMANTIC_MAP = load_json_config('dictionaries/TITLE_SEMANTIC_MAP.json')

print("配置已就绪！")
print(f"TITLE_STOP_WORDS 数量: {len(TITLE_STOP_WORDS)}")
print(f"TITLE_SEMANTIC_MAP 数量: {len(TITLE_SEMANTIC_MAP)}")
print(f"STOP_TAGS 数量: {len(STOP_TAGS)}")
print(f"SEMANTIC_MAP 数量: {len(SEMANTIC_MAP)}")
