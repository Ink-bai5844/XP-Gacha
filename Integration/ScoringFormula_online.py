import os
import re
import json
import hashlib
import math
import pickle
import gc
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
from collections import Counter
from janome.tokenizer import Tokenizer
from scipy.sparse import csr_matrix
from sqlalchemy import create_engine, text

st.set_page_config(page_title="地下金库(Online)", layout="wide")

st.markdown(
    """
    <style>
    /* 针对表格图片放大悬浮窗进行向右平移 */
    div[id^="gdg-overlay-"] {
        margin-left: 100px !important; /* 向右平移 180 像素 */
        z-index: 99999 !important;     /* 确保在最上层 */
        
        /* 圆角和立体阴影 */
        border-radius: 8px !important;
        box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.4) !important;
        overflow: hidden !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 数据库连接
@st.cache_resource
def init_db_engine():
    # 使用 cache_resource 确保引擎只创建一次
    try:
        # 从 st.secrets 读取配置
        db_user = st.secrets["mysql"]["user"]
        db_pwd = st.secrets["mysql"]["password"]
        db_host = st.secrets["mysql"]["host"]
        db_port = st.secrets["mysql"]["port"]
        db_name = st.secrets["mysql"]["database"]
        
        # 构建连接字符串
        DB_URI = f"mysql+pymysql://{db_user}:{db_pwd}@{db_host}:{db_port}/{db_name}?charset=utf8mb4"
        
        # 创建引擎，并设置连接池参数
        # pool_size: 保持的连接数, max_overflow: 允许溢出的最大连接数
        engine = create_engine(
            DB_URI, 
            pool_size=5, 
            max_overflow=10, 
            pool_recycle=3600
        )
        return engine
    except KeyError as e:
        st.error(f"密钥配置缺失！请检查配置文件是否包含 {e}")
        st.stop()

engine = init_db_engine()

# 数据文件夹
CACHE_DIR = "datacache"
B64_CACHE_DIR = "b64_cache"

for directory in [CACHE_DIR, B64_CACHE_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# 初始标签权重
INITIAL_TAG_WEIGHTS = {
    'NTR(netorare)': -2.0
}

# 辅助读取函数
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

STOP_TAGS = load_text_config('dictionaries/STOP_TAGS.txt')
SEMANTIC_MAP = load_json_config('dictionaries/SEMANTIC_MAP.json')
TITLE_STOP_WORDS = load_text_config('dictionaries/TITLE_STOP_WORDS.txt')
TITLE_SEMANTIC_MAP = load_json_config('dictionaries/TITLE_SEMANTIC_MAP.json')

tokenizer = Tokenizer()

print("配置已就绪！")
print(f"TITLE_STOP_WORDS 数量: {len(TITLE_STOP_WORDS)}")
print(f"TITLE_SEMANTIC_MAP 数量: {len(TITLE_SEMANTIC_MAP)}")

# 缩略图字符串读取
def resolve_gallery_id(gallery_id="", url=""):
    if pd.notna(gallery_id) and str(gallery_id).strip():
        return str(gallery_id).strip()

    if pd.notna(url) and str(url).strip():
        url_str = str(url).strip()
        nh_match = re.search(r'/g/(\d+)/?', url_str)
        if nh_match:
            return f"NH{nh_match.group(1)}"

        jm_match = re.search(r'/album/(\d+)/?', url_str)
        if jm_match:
            return f"JM{jm_match.group(1)}"

    return None


def get_cover_base64(gallery_id="", url=""):
    gallery_id = resolve_gallery_id(gallery_id, url)

    if not gallery_id:
        return None

    # 直接寻找预先生成的 Base64 文本文件
    b64_file_path = os.path.join(B64_CACHE_DIR, f"{gallery_id}.txt")
    
    if os.path.exists(b64_file_path):
        try:
            # 极速读取文本，直接返回，0 编码开销
            with open(b64_file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None

    return None

def extract_and_map_title_words(title_raw):
    if not isinstance(title_raw, str):
        return []
    
    clean_title = re.sub(r'\[.*?\]', ' ', title_raw).strip()
    extracted_words = []
    
    for token in tokenizer.tokenize(clean_title):
        word = token.surface
        part_of_speech = token.part_of_speech.split(',')[0]
        
        if part_of_speech in ['名詞', '動詞', '形容詞'] and word not in TITLE_STOP_WORDS and len(word) >= 1:
            mapped_word = TITLE_SEMANTIC_MAP.get(word, word)
            extracted_words.append(mapped_word)
            
    return extracted_words

def get_data_hash():
    """根据配置文件和数据库实时状态生成哈希指纹"""
    hasher = hashlib.md5()
    
    # 监控配置文件
    config_files = ['dictionaries/STOP_TAGS.txt', 'dictionaries/SEMANTIC_MAP.json', 'dictionaries/TITLE_STOP_WORDS.txt', 'dictionaries/TITLE_SEMANTIC_MAP.json']
    for file in config_files:
        if os.path.exists(file):
            hasher.update(str(os.path.getmtime(file)).encode())
            
    # 监控数据库更新状态
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*), MAX(ID) FROM gallery_info")).fetchone()
            if result:
                hasher.update(str(result[0]).encode())
                hasher.update(str(result[1]).encode())
    except Exception as e:
        print(f"服务器数据库状态获取失败: {e}")
            
    return hasher.hexdigest()


def build_preference_chart_cache(tag_freq, artist_freq, title_word_freq):
    return {
        "tags": {
            "title": "Top 15 XP 标签分布",
            "top_15": tag_freq.most_common(15),
            "top_150": tag_freq.most_common(150),
            "label_col": "标签",
            "value_col": "频次",
            "table_label_col": "XP 标签",
            "table_value_col": "出现频次",
            "expander_label": "🔍 查看 Top 150 XP 标签",
        },
        "artists": {
            "title": "Top 15 核心作者分布",
            "top_15": artist_freq.most_common(15),
            "top_150": artist_freq.most_common(150),
            "label_col": "作者",
            "value_col": "频次",
            "table_label_col": "作者名",
            "table_value_col": "收录册数",
            "expander_label": "🔍 查看 Top 150 核心作者",
        },
        "title_words": {
            "title": "Top 15 标题高频词汇",
            "top_15": title_word_freq.most_common(15),
            "top_150": title_word_freq.most_common(150),
            "label_col": "词汇",
            "value_col": "频次",
            "table_label_col": "特征词汇",
            "table_value_col": "出现频次",
            "expander_label": "🔍 查看 Top 150 标题高频词汇",
        },
    }


def build_empty_base_data():
    empty_tag_frequencies = Counter()
    empty_artist_frequencies = Counter()
    empty_title_word_frequencies = Counter()
    chart_cache = build_preference_chart_cache(
        empty_tag_frequencies,
        empty_artist_frequencies,
        empty_title_word_frequencies,
    )
    return None, empty_tag_frequencies, empty_artist_frequencies, empty_title_word_frequencies, chart_cache, None


def build_multi_value_feature_cache(parsed_items_list, frequency_counter):
    feature_names = sorted(frequency_counter.keys())
    feature_index_map = {name: idx for idx, name in enumerate(feature_names)}
    row_count = len(parsed_items_list)
    row_indices = []
    col_indices = []
    values = []
    row_lengths = np.zeros(row_count, dtype=np.float32)

    for row_idx, items in enumerate(parsed_items_list):
        row_lengths[row_idx] = float(len(items))
        if not items:
            continue

        item_counter = Counter(items)
        for item_name, item_count in item_counter.items():
            feature_idx = feature_index_map.get(item_name)
            if feature_idx is None:
                continue
            row_indices.append(row_idx)
            col_indices.append(feature_idx)
            values.append(float(item_count))

    feature_matrix = csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(row_count, len(feature_names)),
        dtype=np.float32,
    )

    return {
        "names": feature_names,
        "index_map": feature_index_map,
        "base_scores": np.array(
            [math.log1p(frequency_counter[name]) * 10.0 for name in feature_names],
            dtype=np.float32,
        ),
        "matrix": feature_matrix,
        "row_norms": np.sqrt(np.maximum(row_lengths, 1.0)).astype(np.float32),
    }


def build_artist_feature_cache(df, artist_frequencies):
    artist_names = sorted(artist_frequencies.keys())
    artist_index_map = {name: idx for idx, name in enumerate(artist_names)}
    artist_codes = np.full(len(df), -1, dtype=np.int32)
    artists_series = (
        df['作者'].astype(str).str.strip()
        if '作者' in df.columns
        else pd.Series([''] * len(df), index=df.index, dtype='object')
    )

    for row_idx, artist_name in enumerate(artists_series):
        if artist_name:
            artist_codes[row_idx] = artist_index_map.get(artist_name, -1)

    return {
        "names": artist_names,
        "index_map": artist_index_map,
        "base_scores": np.array(
            [math.log1p(artist_frequencies[name]) * 10.0 for name in artist_names],
            dtype=np.float32,
        ),
        "codes": artist_codes,
    }


def build_score_cache(df, tag_frequencies, artist_frequencies, title_word_frequencies):
    parsed_tags_list = df['解析后标签'].tolist() if '解析后标签' in df.columns else [[] for _ in range(len(df))]
    parsed_title_words_list = df['标题特征词'].tolist() if '标题特征词' in df.columns else [[] for _ in range(len(df))]

    return {
        "tags": build_multi_value_feature_cache(parsed_tags_list, tag_frequencies),
        "artists": build_artist_feature_cache(df, artist_frequencies),
        "title_words": build_multi_value_feature_cache(parsed_title_words_list, title_word_frequencies),
    }


def render_ranked_bar_chart(items, label_col, value_col):
    if not items:
        return

    chart_df = pd.DataFrame(items, columns=[label_col, value_col])
    chart_df["排序标签"] = chart_df[label_col]

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                f"{label_col}:N",
                sort=chart_df["排序标签"].tolist(),
                axis=alt.Axis(
                    labelAngle=-45,
                    labelLimit=0,
                    labelOverlap=False,
                    labelFontSize=11,
                    title=None,
                ),
            ),
            y=alt.Y(f"{value_col}:Q", title=None),
            tooltip=[
                alt.Tooltip(f"{label_col}:N", title=label_col),
                alt.Tooltip(f"{value_col}:Q", title=value_col),
            ],
        )
        .properties(height=280)
    )

    st.altair_chart(chart, width="stretch")


def render_preference_chart_block(chart_meta):
    st.write(f"**{chart_meta['title']}**")
    render_ranked_bar_chart(
        chart_meta["top_15"],
        chart_meta["label_col"],
        chart_meta["value_col"],
    )

    with st.expander(chart_meta["expander_label"]):
        top_150_items = chart_meta["top_150"]
        if top_150_items:
            top_150_df = pd.DataFrame(
                top_150_items,
                columns=[chart_meta["table_label_col"], chart_meta["table_value_col"]],
            )
            st.dataframe(top_150_df, hide_index=True, width="stretch")


def render_global_preference_charts(chart_cache):
    st.markdown("---")
    st.subheader("全局偏好数据")
    chart_col1, chart_col2, chart_col3 = st.columns(3)

    with chart_col1:
        render_preference_chart_block(chart_cache["tags"])

    with chart_col2:
        render_preference_chart_block(chart_cache["artists"])

    with chart_col3:
        render_preference_chart_block(chart_cache["title_words"])


def normalize_cached_base_data(cached_payload, cache_data_file):
    if isinstance(cached_payload, tuple) and len(cached_payload) == 6:
        return cached_payload

    if isinstance(cached_payload, tuple) and len(cached_payload) == 5:
        df, tag_frequencies, artist_frequencies, title_word_frequencies, score_cache = cached_payload
        chart_cache = build_preference_chart_cache(
            tag_frequencies,
            artist_frequencies,
            title_word_frequencies,
        )
        normalized_payload = (
            df,
            tag_frequencies,
            artist_frequencies,
            title_word_frequencies,
            chart_cache,
            score_cache,
        )
        with open(cache_data_file, 'wb') as f:
            pickle.dump(normalized_payload, f)
        return normalized_payload

    if isinstance(cached_payload, tuple) and len(cached_payload) == 4:
        df, tag_frequencies, artist_frequencies, title_word_frequencies = cached_payload
        chart_cache = build_preference_chart_cache(
            tag_frequencies,
            artist_frequencies,
            title_word_frequencies,
        )
        score_cache = build_score_cache(
            df,
            tag_frequencies,
            artist_frequencies,
            title_word_frequencies,
        )
        normalized_payload = (
            df,
            tag_frequencies,
            artist_frequencies,
            title_word_frequencies,
            chart_cache,
            score_cache,
        )
        with open(cache_data_file, 'wb') as f:
            pickle.dump(normalized_payload, f)
        return normalized_payload

    raise ValueError("服务器预处理缓存文件格式无法识别，请删除 datacache 后重试。")

@st.cache_data(max_entries=1, show_spinner=False)
def load_base_data():
    current_hash = get_data_hash()
    cache_data_file = os.path.join(CACHE_DIR, "server_preprocessed_df.pkl")
    cache_hash_file = os.path.join(CACHE_DIR, "server_data.hash")

    # 命中缓存则极速拉取
    if os.path.exists(cache_data_file) and os.path.exists(cache_hash_file):
        with open(cache_hash_file, 'r') as f:
            saved_hash = f.read().strip()
        if saved_hash == current_hash:
            print("触发服务器级文件缓存，跳过计算！")
            with open(cache_data_file, 'rb') as f:
                cached_payload = pickle.load(f)
            return normalize_cached_base_data(cached_payload, cache_data_file)

    print("缓存失效或无缓存，执行数据库全量拉取与自然语言处理...")
    
    # 从 MySQL 拉取数据
    try:
        df = pd.read_sql("SELECT * FROM gallery_info", con=engine)
    except Exception as e:
        print(f"数据库读取失败: {e}")
        return build_empty_base_data()

    if df.empty:
        return build_empty_base_data()

    df = df.fillna('')
    # 修复可能存在的无标题情况
    df['标题'] = df.apply(lambda row: row.get('文件名', '') if row.get('标题', '') == '' else row.get('标题', ''), axis=1)

    all_tags = []
    parsed_tags_list = []
    all_title_words = []
    parsed_title_words_list = []

    # NLP 解析
    for _, row in df.iterrows():
        tags = row.get('标签', '')
        if tags:
            clean_tags = [t.strip() for t in str(tags).split(',') if t.strip() not in STOP_TAGS]
            mapped_tags = [SEMANTIC_MAP.get(t, t) for t in clean_tags]
            parsed_tags_list.append(mapped_tags)
            all_tags.extend(mapped_tags)
        else:
            parsed_tags_list.append([])
            
        title_words = extract_and_map_title_words(row.get('标题', ''))
        parsed_title_words_list.append(title_words)
        all_title_words.extend(title_words)

    df['解析后标签'] = parsed_tags_list
    df['标题特征词'] = parsed_title_words_list
    
    tag_frequencies = Counter(all_tags)
    artist_frequencies = Counter([str(a).strip() for a in df.get('作者', []) if str(a).strip()])
    title_word_frequencies = Counter(all_title_words)
    chart_cache = build_preference_chart_cache(
        tag_frequencies,
        artist_frequencies,
        title_word_frequencies,
    )
    score_cache = build_score_cache(
        df,
        tag_frequencies,
        artist_frequencies,
        title_word_frequencies,
    )

    result_tuple = (df, tag_frequencies, artist_frequencies, title_word_frequencies, chart_cache, score_cache)

    # 序列化保存结果
    with open(cache_data_file, 'wb') as f:
        pickle.dump(result_tuple, f)
    with open(cache_hash_file, 'w') as f:
        f.write(current_hash)

    # 清理中间产物内存
    del all_tags
    del all_title_words
    gc.collect()

    return result_tuple

with st.spinner('正在同步预处理缓存与计算引擎...'):
    df_base, tag_freq, artist_freq, title_word_freq, preference_chart_cache, score_cache = load_base_data()

# 侧边栏
st.sidebar.title("筛选与偏好设置")

if df_base is None:
    st.error("未找到 csv 文件！请确保 gallery_info 文件夹内有数据。")
    st.stop()

search_kw = st.sidebar.text_input("实时关键词搜索 (标题/标签/作者)：", placeholder="例如: elf...")

st.sidebar.markdown("---")
st.sidebar.subheader("全局评分权重分配")
st.sidebar.write("调整各维度在总分中的占比乘数：")

global_tag_weight = st.sidebar.slider("标签总分倍率", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
global_artist_weight = st.sidebar.slider("作者总分倍率", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
global_title_weight = st.sidebar.slider("标题总分倍率", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
st.sidebar.markdown("---")

all_unique_tags = sorted(list(tag_freq.keys()))
valid_default_tags = [t for t in INITIAL_TAG_WEIGHTS.keys() if t in all_unique_tags]

with st.sidebar.expander("屏蔽标签配置", expanded=False):
    blocked_tags = st.multiselect("选择要屏蔽的标签：", options=all_unique_tags, default=[])

with st.sidebar.expander("标签权重配置", expanded=True):
    selected_tags = st.multiselect("加权/降权标签列表：", options=all_unique_tags, default=valid_default_tags)
    dynamic_weights = {}
    for t in selected_tags:
        default_val = float(INITIAL_TAG_WEIGHTS.get(t, 1.0))
        val = st.number_input(f"「{t}」权重倍率", value=default_val, step=0.1, format="%.1f")
        dynamic_weights[t] = val

with st.sidebar.expander("作者喜爱倍数配置", expanded=False):
    all_artists = sorted(list(artist_freq.keys()))
    selected_artists = st.multiselect("需要特殊优待的作者：", options=all_artists, default=[])
    dynamic_artist_weights = {}
    for a in selected_artists:
        val = st.number_input(f"「{a}」倍率", value=5.0, step=0.5, format="%.1f")
        dynamic_artist_weights[a] = val

with st.sidebar.expander("标题关键词权重配置", expanded=False):
    all_title_words = sorted(list(title_word_freq.keys()))
    selected_title_words = st.multiselect("关键词列表：", options=all_title_words, default=[])
    dynamic_title_weights = {}
    for w in selected_title_words:
        val = st.number_input(f"词汇「{w}」权重", value=1.0, step=0.1, format="%.1f")
        dynamic_title_weights[w] = val

def build_weight_vector(feature_cache, dynamic_weights, default_value):
    weight_vector = np.full(len(feature_cache["names"]), default_value, dtype=np.float32)
    for feature_name, feature_weight in dynamic_weights.items():
        feature_idx = feature_cache["index_map"].get(feature_name)
        if feature_idx is not None:
            weight_vector[feature_idx] = float(feature_weight)
    return weight_vector


def apply_dynamic_scores(df, tag_weights, artist_weights, title_weights, tag_freq, artist_freq, title_word_freq, global_tag_w, global_artist_w, global_title_w, score_cache=None):
    if score_cache is None:
        score_cache = build_score_cache(df, tag_freq, artist_freq, title_word_freq)

    total_scores = np.zeros(len(df), dtype=np.float32)

    tag_cache = score_cache["tags"]
    if tag_cache["names"]:
        tag_weight_vector = build_weight_vector(tag_cache, tag_weights, 1.0)
        tag_effective_scores = tag_cache["base_scores"] * tag_weight_vector
        tag_score_sum = np.asarray(tag_cache["matrix"].dot(tag_effective_scores)).reshape(-1)
        total_scores += (tag_score_sum / tag_cache["row_norms"]) * float(global_tag_w) * 0.5

    artist_cache = score_cache["artists"]
    if artist_cache["names"]:
        artist_weight_vector = build_weight_vector(artist_cache, artist_weights, 5.0)
        valid_artist_mask = artist_cache["codes"] >= 0
        if valid_artist_mask.any():
            artist_codes = artist_cache["codes"][valid_artist_mask]
            artist_scores = (
                artist_cache["base_scores"][artist_codes]
                * artist_weight_vector[artist_codes]
                * float(global_artist_w)
            )
            total_scores[valid_artist_mask] += artist_scores

    title_cache = score_cache["title_words"]
    if title_cache["names"]:
        title_weight_vector = build_weight_vector(title_cache, title_weights, 1.0)
        title_effective_scores = title_cache["base_scores"] * title_weight_vector
        title_score_sum = np.asarray(title_cache["matrix"].dot(title_effective_scores)).reshape(-1)
        total_scores += (title_score_sum / title_cache["row_norms"]) * float(global_title_w)

    scored_df = df.copy()
    scored_df['推荐评分'] = total_scores.astype(np.int32)
    
    columns_order = ['封面', '推荐评分', 'ID', '上传日期', '标题', '作者', '团队', '标签', '语言', '页数', '链接', '文件名', '解析后标签', '标题特征词']
    if '上传日期' not in scored_df.columns:
        scored_df['上传日期'] = ''
        
    available_columns = [col for col in columns_order if col in scored_df.columns]
    scored_df = scored_df[available_columns]
    
    return scored_df

final_df = apply_dynamic_scores(
    df_base, 
    dynamic_weights, 
    dynamic_artist_weights, 
    dynamic_title_weights, 
    tag_freq, 
    artist_freq, 
    title_word_freq,
    global_tag_weight,
    global_artist_weight,
    global_title_weight,
    score_cache=score_cache
)

if blocked_tags:
    mask_not_blocked = final_df['解析后标签'].apply(lambda x: not any(t in blocked_tags for t in x))
    final_df = final_df[mask_not_blocked]

if '解析后标签' in final_df.columns:
    final_df = final_df.drop(columns=['解析后标签'])
if '标题特征词' in final_df.columns:
    final_df = final_df.drop(columns=['标题特征词'])

final_df = final_df.sort_values(by=['推荐评分', '上传日期'], ascending=[False, False]).reset_index(drop=True)

if not final_df.empty:
    min_possible_score = int(final_df['推荐评分'].min())
    max_possible_score = int(final_df['推荐评分'].max())
else:
    min_possible_score, max_possible_score = 0, 100

if min_possible_score >= max_possible_score:
    max_possible_score = min_possible_score + 1

default_min_slider = 0 if min_possible_score <= 0 <= max_possible_score else min_possible_score

min_score = st.sidebar.slider("最低推荐评分阈值：", min_value=min_possible_score, max_value=max_possible_score, value=default_min_slider)
filtered_df = final_df[final_df['推荐评分'] >= min_score]

if search_kw and not filtered_df.empty:
    kw_list = [kw.strip().lower() for kw in search_kw.replace('，', ',').split(',') if kw.strip()]
    for kw in kw_list:
        mask_search = (
            filtered_df['ID'].str.lower().str.contains(kw, regex=False, na=False) |
            filtered_df['标题'].str.lower().str.contains(kw, regex=False, na=False) |
            filtered_df['标签'].str.lower().str.contains(kw, regex=False, na=False) |
            filtered_df['作者'].str.lower().str.contains(kw, regex=False, na=False) |
            filtered_df['团队'].str.lower().str.contains(kw, regex=False, na=False)
        )
        filtered_df = filtered_df[mask_search]

# 释放用于过滤的全量 DataFrame 内存
del final_df
gc.collect()

# 页面展示

st.title("墨白的地下金库:P")

col1, col2, col3, col4 = st.columns(4)
col1.metric("当前显示条目数", f"{len(filtered_df)} 册")
col2.metric("总收录作者数", f"{len(artist_freq)} 位")
col3.metric("总标签种类", f"{len(tag_freq)} 种")
col4.metric("解析标题词汇数", f"{len(title_word_freq)} 种")

st.markdown("---")
st.subheader("库存列表")

if not filtered_df.empty:
    sort_columns = ['推荐评分', 'ID', '上传日期', '标题', '作者', '团队', '标签', '语言', '页数']
    
    col_sort1, col_sort2, col_page, col_empty = st.columns([1.5, 1, 1.5, 2])
    
    with col_sort1:
        global_sort_by = st.selectbox("全局排序依据：", options=sort_columns, index=0)
    with col_sort2:
        global_sort_order = st.radio("顺序：", options=["降序 ↓", "升序 ↑"], horizontal=True)
        
    is_ascending = (global_sort_order == "升序 ↑")
    
    if global_sort_by == '推荐评分':
        filtered_df = filtered_df.sort_values(by=['推荐评分', '上传日期'], ascending=[is_ascending, False]).reset_index(drop=True)
    else:
        filtered_df = filtered_df.sort_values(by=[global_sort_by], ascending=[is_ascending]).reset_index(drop=True)

    MAX_DISPLAY = 200
    total_items = len(filtered_df)
    total_pages = math.ceil(total_items / MAX_DISPLAY)
    
    page_options = []
    for i in range(total_pages):
        start_idx = i * MAX_DISPLAY
        end_idx = min((i + 1) * MAX_DISPLAY - 1, total_items - 1)
        page_options.append(f"{start_idx} ~ {end_idx}")
        
    with col_page:
        selected_page_label = st.selectbox("选择显示范围：", options=page_options)
    
    selected_page_index = page_options.index(selected_page_label)
    slice_start = selected_page_index * MAX_DISPLAY
    slice_end = (selected_page_index + 1) * MAX_DISPLAY
    
    display_df = filtered_df.iloc[slice_start:slice_end].copy()

    with st.spinner(f'正在加载 {selected_page_label} 范围的缩略图...'):
        display_df['封面'] = display_df.apply(
            lambda row: get_cover_base64(row.get('ID', ''), row.get('链接', '')),
            axis=1
        )

    display_df = display_df.drop(columns=['文件名'], errors='ignore')
    preferred_columns = ['封面', '推荐评分', 'ID', '上传日期', '标题', '作者', '团队', '标签', '语言', '页数', '链接']
    display_columns = [col for col in preferred_columns if col in display_df.columns]
    display_columns += [col for col in display_df.columns if col not in display_columns]
    display_df = display_df[display_columns]

    st.dataframe(
        display_df,
        column_config={
            "封面": st.column_config.ImageColumn("封面", help="云端缓存封面图"),
            "链接": st.column_config.LinkColumn("图库链接", display_text="网络来源"),
            "推荐评分": st.column_config.ProgressColumn(
                "推荐评分", 
                format="%d", 
                min_value=min_possible_score, 
                max_value=max_possible_score
            ),
            "ID": st.column_config.TextColumn("ID", help="唯一标识符"),
            "上传日期": st.column_config.TextColumn("上传日期", help="该漫画的上传时间")
        },
        hide_index=True,
        use_container_width=True,
        height=600 
    )

    # 显示完毕，安全销毁当前分页的重度 Base64 图片内存
    del display_df
    gc.collect()

else:
    st.info("没有可以显示的数据喔。")

render_global_preference_charts(preference_chart_cache)
