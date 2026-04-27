import os
import math
import hashlib
import pickle
import gc
import pandas as pd
import streamlit as st
from collections import Counter
from sqlalchemy import create_engine, text
from config import CACHE_DIR, STOP_TAGS, SEMANTIC_MAP
from utils_core import get_local_folders, match_local_folder
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
                return pickle.load(f)

    print("缓存失效，执行数据库全量读取与自然语言处理...")
    try:
        df = pd.read_sql("SELECT * FROM gallery_info", con=engine)
    except Exception as e:
        print(f"数据库读取失败: {e}")
        return None, Counter(), Counter(), Counter()

    if df.empty:
        return None, Counter(), Counter(), Counter()

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

    result_tuple = (df, tag_frequencies, artist_frequencies, title_word_frequencies)

    with open(cache_data_file, 'wb') as f:
        pickle.dump(result_tuple, f)
    with open(cache_hash_file, 'w') as f:
        f.write(current_hash)

    del all_tags
    del all_title_words
    gc.collect()

    return result_tuple

def apply_dynamic_scores(df, tag_weights, artist_weights, title_weights, tag_freq, artist_freq, title_word_freq, global_tag_w, global_artist_w, global_title_w):
    def calculate_score(row):
        score = 0.0
        tags = row['解析后标签']
        if tags:
            tag_score_sum = sum(math.log1p(tag_freq.get(t, 0)) * 10 * tag_weights.get(t, 1.0) for t in tags)
            base_tag_score = tag_score_sum / math.sqrt(len(tags))
            score += base_tag_score * global_tag_w * 0.5

        artist = row['作者'].strip()
        if artist:
            multiplier = artist_weights.get(artist, 5.0)
            base_artist_score = math.log1p(artist_freq.get(artist, 0)) * 10 * multiplier
            score += base_artist_score * global_artist_w 
            
        title_words = row.get('标题特征词', [])
        if title_words:
            title_score_sum = sum(
                math.log1p(title_word_freq.get(w, 0)) * 10 * title_weights.get(w, 1.0)
                for w in title_words
            )
            title_dilution = max(1, math.sqrt(len(title_words)))
            base_title_score = (title_score_sum / title_dilution)
            score += base_title_score * global_title_w 
            
        return int(score)

    scored_df = df.copy()
    scored_df['推荐评分'] = scored_df.apply(calculate_score, axis=1)
    
    columns_order = ['封面', '推荐评分', 'ID', '上传日期', '标题', '作者', '团队', '标签', '语言', '页数', '本地目录', '链接', '文件名', '解析后标签', '标题特征词']
    if '上传日期' not in scored_df.columns:
        scored_df['上传日期'] = ''
        
    available_columns = [col for col in columns_order if col in scored_df.columns]
    scored_df = scored_df[available_columns]
    return scored_df
