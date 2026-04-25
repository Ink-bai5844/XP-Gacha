import os
import json
import re

# 目录配置
ONLINE_IMG_DIR = "onlineimgtmp"
IMG_CACHE_DIR = "localimgtmp"
CACHE_DIR = "datacache"
B64_CACHE_DIR = "b64_cache"
VECTOR_FILE = "manga_vectors/manga_vectors_Qwen3.pkl"
LOCAL_MODEL_PATH = r"D:\Python\XP-Gacha\Qwen3-Embedding-0.6B"
BASE_DIR = r"H:\1\12"

# LM Studio 配置
LM_STUDIO_API_BASE = "http://localhost:1234/v1"
LM_STUDIO_MODEL = "local-model"

# 线上 AI 配置
ONLINE_API_BASE = "http://api.bltcy.ai/v1" 
ONLINE_API_KEY = "你的API密钥"
ONLINE_MODEL = "grok-4-fast-non-reasoning"
SYSTEM_PROMPT = (
    "你是‘地下金库’的智能助手。你的性格冷静、专业。 "
    "‘地下金库’是一个储存大量漫画的数据库。我会为你提供一部分当前的库存数据作为参考。 "
    "请结合这些数据回答问题。如果数据中没有相关内容，请基于你的通用知识库回答。"
)

# 预设Tag权重配置
INITIAL_TAG_WEIGHTS = {
    'NTR(netorare)': -2.0
}

# 每页条目显示上限
MAX_DISPLAY = 500

# 初始化目录
for directory in [ONLINE_IMG_DIR, IMG_CACHE_DIR, CACHE_DIR, B64_CACHE_DIR]:
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