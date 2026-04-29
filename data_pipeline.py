import os
import math
import hashlib
import pickle
import gc
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter
from scipy.sparse import csr_matrix
from sqlalchemy import create_engine, text
from config import CACHE_DIR, STOP_TAGS, SEMANTIC_MAP
from utils_core import get_local_folders, match_local_folder
from utils_charts import build_preference_chart_cache
from utils_nlp import extract_and_map_title_words

@st.cache_resource
def init_db_engine():
    try:
        db_user = st.secrets["mysql"]["user"]
        db_pwd = st.secrets["mysql"]["password"]
        db_host = st.secrets["mysql"]["host"]
        db_port = st.secrets["mysql"]["port"]
        db_name = st.secrets["mysql"]["database"]
        
        DB_URI = f"mysql+pymysql://{db_user}:{db_pwd}@{db_host}:{db_port}/{db_name}?charset=utf8mb4"
        engine = create_engine(DB_URI, pool_size=5, max_overflow=10, pool_recycle=3600)
        return engine
    except KeyError as e:
        st.error(f"密钥配置缺失！请检查配置文件是否包含 {e}")
        st.stop()

engine = init_db_engine()


def build_empty_base_data():
    empty_tag_frequencies = Counter()
    empty_artist_frequencies = Counter()
    empty_title_word_frequencies = Counter()
    chart_cache = build_preference_chart_cache(
        empty_tag_frequencies,
        empty_artist_frequencies,
        empty_title_word_frequencies,
    )
    return (
        None,
        empty_tag_frequencies,
        empty_artist_frequencies,
        empty_title_word_frequencies,
        chart_cache,
        None,
    )


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

def get_data_hash():
    hasher = hashlib.md5()
    config_files = [
        'dictionaries/STOP_TAGS.txt',
        'dictionaries/SEMANTIC_MAP.json',
        'dictionaries/TITLE_STOP_WORDS.txt',
        'dictionaries/TITLE_SEMANTIC_MAP.json',
    ]
    for file in config_files:
        if os.path.exists(file):
            hasher.update(str(os.path.getmtime(file)).encode())
            
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*), MAX(ID) FROM gallery_info")).fetchone()
            if result:
                hasher.update(str(result[0]).encode()) 
                hasher.update(str(result[1]).encode()) 
    except Exception as e:
        print(f"数据库状态获取失败: {e}")
            
    return hasher.hexdigest()

@st.cache_data(max_entries=1, show_spinner=False)
def load_base_data():
    current_hash = get_data_hash()
    cache_data_file = os.path.join(CACHE_DIR, "preprocessed_df.pkl")
    cache_hash_file = os.path.join(CACHE_DIR, "data.hash")

    if os.path.exists(cache_data_file) and os.path.exists(cache_hash_file):
        with open(cache_hash_file, 'r') as f:
            saved_hash = f.read().strip()
        if saved_hash == current_hash:
            print("触发文件级缓存，跳过计算！")
            with open(cache_data_file, 'rb') as f:
                cached_payload = pickle.load(f)
            return normalize_cached_base_data(cached_payload, cache_data_file)

    print("缓存失效，执行数据库全量读取与自然语言处理...")
    try:
        df = pd.read_sql("SELECT * FROM gallery_info", con=engine)
    except Exception as e:
        print(f"数据库读取失败: {e}")
        return build_empty_base_data()

    if df.empty:
        return build_empty_base_data()

    df = df.fillna('')
    folder_map = get_local_folders()
    df['本地目录'] = df.apply(lambda row: match_local_folder(row.get('文件名', ''), folder_map), axis=1)

    all_tags = []
    parsed_tags_list = []
    all_title_words = []
    parsed_title_words_list = []

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

    result_tuple = (
        df,
        tag_frequencies,
        artist_frequencies,
        title_word_frequencies,
        chart_cache,
        score_cache,
    )

    with open(cache_data_file, 'wb') as f:
        pickle.dump(result_tuple, f)
    with open(cache_hash_file, 'w') as f:
        f.write(current_hash)

    del all_tags
    del all_title_words
    gc.collect()

    return result_tuple


def normalize_cached_base_data(cached_payload, cache_data_file):
    if isinstance(cached_payload, tuple) and len(cached_payload) == 6:
        return cached_payload

    if isinstance(cached_payload, tuple) and len(cached_payload) == 5:
        df, tag_frequencies, artist_frequencies, title_word_frequencies, chart_cache = cached_payload
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

    raise ValueError("预处理缓存文件格式无法识别，请删除 datacache 后重试。")

def build_weight_vector(feature_cache, dynamic_weights, default_value):
    weight_vector = np.full(len(feature_cache["names"]), default_value, dtype=np.float32)
    for feature_name, feature_weight in dynamic_weights.items():
        feature_idx = feature_cache["index_map"].get(feature_name)
        if feature_idx is not None:
            weight_vector[feature_idx] = float(feature_weight)
    return weight_vector


def apply_dynamic_scores(
    df,
    tag_weights,
    artist_weights,
    title_weights,
    tag_freq,
    artist_freq,
    title_word_freq,
    global_tag_w,
    global_artist_w,
    global_title_w,
    score_cache=None,
):
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
    
    columns_order = ['封面', '推荐评分', 'ID', '上传日期', '标题', '作者', '团队', '标签', '语言', '页数', '本地目录', '链接', '文件名', '解析后标签', '标题特征词']
    if '上传日期' not in scored_df.columns:
        scored_df['上传日期'] = ''
        
    available_columns = [col for col in columns_order if col in scored_df.columns]
    scored_df = scored_df[available_columns]
    return scored_df
