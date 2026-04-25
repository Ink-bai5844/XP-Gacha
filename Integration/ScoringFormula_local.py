import os
import re
import json
import glob
import hashlib
import base64
import difflib
import math
import pickle
import gc
import streamlit as st
import pandas as pd
from collections import Counter
from PIL import Image
from janome.tokenizer import Tokenizer
from sqlalchemy import create_engine
from sqlalchemy import text

ONLINE_IMG_DIR = "onlineimgtmp"
IMG_CACHE_DIR = "localimgtmp"
CACHE_DIR = "datacache"
B64_CACHE_DIR = "b64_cache"
VECTOR_FILE = "manga_vectors/manga_vectors.pkl"
BASE_DIR = r"H:\动漫资源\漫画集\HMAN"

# 预设Tag权重配置
INITIAL_TAG_WEIGHTS = {
    'TS/性转(gender bender)': 2.5,
    '变身(transformation)': 2.0,
    '身体改造(body modification)': 1.5,
    '后庭(anal)': 0.8,
    'NTR(netorare)': -2.0
}

st.set_page_config(page_title="地下金库(Local)", layout="wide")

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

@st.cache_resource(show_spinner="正在将 AI 语义矩阵载入内存...")
def load_semantic_engine():
    from sentence_transformers import SentenceTransformer
    import torch
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    with open(VECTOR_FILE, 'rb') as f:
        data = pickle.load(f)
        
    corpus_embeddings = torch.tensor(data['embeddings'])
    corpus_ids = [str(i) for i in data['ids']] # 确保全部转换为字符串
    
    # 核心魔法：建立一个字典，把“链接”映射到矩阵的“行号”，实现 O(1) 极速查找
    id_to_index = {link_id: idx for idx, link_id in enumerate(corpus_ids)}
    
    return model, corpus_embeddings, corpus_ids, id_to_index

# embed_model, corpus_embeddings, corpus_ids, id_to_index = load_semantic_engine()

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

STOP_TAGS = load_text_config('dictionaries/STOP_TAGS.txt')
SEMANTIC_MAP = load_json_config('dictionaries/SEMANTIC_MAP.json')
TITLE_STOP_WORDS = load_text_config('dictionaries/TITLE_STOP_WORDS.txt')
TITLE_SEMANTIC_MAP = load_json_config('dictionaries/TITLE_SEMANTIC_MAP.json')

tokenizer = Tokenizer()

print("配置已就绪！")
print(f"TITLE_STOP_WORDS 数量: {len(TITLE_STOP_WORDS)}")
print(f"TITLE_SEMANTIC_MAP 数量: {len(TITLE_SEMANTIC_MAP)}")

# 辅助函数

def sanitize_folder_name(name):
    if not isinstance(name, str):
        return ""
    illegal_chars = r'[\\/*?:"<>|]'
    return re.sub(illegal_chars, '_', name)

@st.cache_data(max_entries=1)
def get_local_folders():
    folder_map = {}
    if os.path.exists(BASE_DIR):
        for root, dirs, files in os.walk(BASE_DIR):
            for d in dirs:
                folder_map[d] = os.path.join(root, d)
    return folder_map

def match_local_folder(csv_filename, folder_map):
    if not csv_filename or not folder_map:
        return "本地目录不存在"
        
    sanitized_name = sanitize_folder_name(csv_filename)
    if sanitized_name in folder_map:
        return folder_map[sanitized_name]
        
    folder_names = list(folder_map.keys())
    matches = difflib.get_close_matches(sanitized_name, folder_names, n=1, cutoff=0.6)
    if matches:
        return folder_map[matches[0]]
        
    return "本地目录不存在"

def get_cover_base64(local_path, url=""):
    gallery_id = None
    if pd.notna(url) and str(url).strip():
        match = re.search(r'/g/(\d+)/?', str(url))
        if match:
            gallery_id = match.group(1)
            
    if not gallery_id:
        return None

    # 优先检查是否已有预渲染的 Base64 文本
    b64_file_path = os.path.join(B64_CACHE_DIR, f"{gallery_id}.txt")
    if os.path.exists(b64_file_path):
        try:
            with open(b64_file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass

    # 没有文本缓存，执行旧逻辑
    full_b64_string = None

    # 在线图片缓存检测
    online_img_pattern = os.path.join(ONLINE_IMG_DIR, f"{gallery_id}.*")
    matched_online_imgs = glob.glob(online_img_pattern)
    
    if matched_online_imgs:
        try:
            target_img = matched_online_imgs[0]
            with open(target_img, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            ext = target_img.split('.')[-1].lower()
            mime = f"image/{ext}" if ext in ['png', 'webp', 'gif'] else "image/jpeg"
            full_b64_string = f"data:{mime};base64,{encoded}"
        except Exception:
            pass

    # 本地目录压缩生成缩略图
    if not full_b64_string:
        if local_path == "本地目录不存在" or not isinstance(local_path, str) or not os.path.exists(local_path):
            return None
            
        cache_file = os.path.join(IMG_CACHE_DIR, f"{gallery_id}.jpg")
        
        if not os.path.exists(cache_file):
            escaped_path = glob.escape(local_path)
            search_pattern = os.path.join(escaped_path, "1.*")
            matched_files = glob.glob(search_pattern)
            
            valid_files = [f for f in matched_files if os.path.isfile(f)]
            if not valid_files:
                return None
                
            target_file = valid_files[0]
            
            try:
                with Image.open(target_file) as img:
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    img.thumbnail((150, 200)) 
                    img.save(cache_file, format="JPEG", quality=85)
            except Exception:
                return None 
                
        try:
            with open(cache_file, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            full_b64_string = f"data:image/jpeg;base64,{encoded}"
        except Exception:
            return None

    # 将 Base64 字符串存入 txt
    if full_b64_string:
        try:
            with open(b64_file_path, "w", encoding="utf-8") as f:
                f.write(full_b64_string)
        except Exception as e:
            print(f"写入 Base64 缓存失败: {e}")
            
    return full_b64_string

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

# 缓存文件系统
def get_data_hash():
    hasher = hashlib.md5()
    
    # 监控配置文件
    config_files = ['STOP_TAGS.txt', 'SEMANTIC_MAP.json', 'TITLE_STOP_WORDS.txt', 'TITLE_SEMANTIC_MAP.json']
    for file in config_files:
        if os.path.exists(file):
            hasher.update(str(os.path.getmtime(file)).encode())
            
    # 监控数据库更新
    # 依靠行数和最大链接字符来生成指纹
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*), MAX(链接) FROM gallery_info")).fetchone()
            if result:
                hasher.update(str(result[0]).encode()) # 行数
                hasher.update(str(result[1]).encode()) # 最新链接标识
    except Exception as e:
        print(f"数据库状态获取失败: {e}")
            
    return hasher.hexdigest()

@st.cache_data(max_entries=1, show_spinner=False)
def load_base_data():
    current_hash = get_data_hash()
    cache_data_file = os.path.join(CACHE_DIR, "preprocessed_df.pkl")
    cache_hash_file = os.path.join(CACHE_DIR, "data.hash")

    # 尝试读取本地序列化缓存 (加速 NLP 解析)
    if os.path.exists(cache_data_file) and os.path.exists(cache_hash_file):
        with open(cache_hash_file, 'r') as f:
            saved_hash = f.read().strip()
        if saved_hash == current_hash:
            print("触发文件级缓存，跳过计算！")
            with open(cache_data_file, 'rb') as f:
                return pickle.load(f)

    # 缓存失效或不存在，开始从 MySQL 读取并进行 NLP 处理
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

    # NLP 与特征提取逻辑保持不变
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

    # 写入二进制文件缓存
    with open(cache_data_file, 'wb') as f:
        pickle.dump(result_tuple, f)
    with open(cache_hash_file, 'w') as f:
        f.write(current_hash)

    # 主动清理内存
    del all_tags
    del all_title_words
    gc.collect()

    return result_tuple

with st.spinner('正在同步预处理缓存与计算引擎...'):
    df_base, tag_freq, artist_freq, title_word_freq = load_base_data()

# 侧边栏
st.sidebar.title("筛选与偏好设置")

if df_base is None:
    st.error("未找到 csv 文件！")
    st.stop()

search_kw = st.sidebar.text_input("实时关键词搜索 (标题/标签/作者)：", placeholder="例如: elf...")
vector_search_kw = st.sidebar.text_input("AI 语义检索 (自然语言)：", placeholder="例如: 猫娘X狐娘...")
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

# 计算逻辑
def apply_dynamic_scores(df, tag_weights, artist_weights, title_weights, tag_freq, artist_freq, title_word_freq, global_tag_w, global_artist_w, global_title_w):
    def calculate_score(row):
        score = 0.0
        tags = row['解析后标签']
        if tags:
            tag_score_sum = sum(math.log1p(tag_freq.get(t, 0)) * 10 * tag_weights.get(t, 1.0) for t in tags)
            base_tag_score = tag_score_sum / math.sqrt(len(tags))
            score += base_tag_score * global_tag_w 

        artist = row['作者'].strip()
        if artist:
            multiplier = artist_weights.get(artist, 5.0)
            base_artist_score = artist_freq.get(artist, 0) * multiplier
            score += base_artist_score * global_artist_w 
            
        title_words = row.get('标题特征词', [])
        if title_words:
            title_score_sum = sum(title_word_freq.get(w, 0) * title_weights.get(w, 1.0) for w in title_words)
            title_dilution = max(1, math.sqrt(len(title_words)))
            base_title_score = (title_score_sum / title_dilution) * 0.5 
            score += base_title_score * global_title_w 
            
        return int(score)

    scored_df = df.copy()
    scored_df['推荐评分'] = scored_df.apply(calculate_score, axis=1)
    
    columns_order = ['封面', '推荐评分', '上传日期', '标题', '作者', '团队', '标签', '语言', '页数', '本地目录', '链接', '文件名', '解析后标签', '标题特征词']
    if '上传日期' not in scored_df.columns:
        scored_df['上传日期'] = ''
        
    available_columns = [col for col in columns_order if col in scored_df.columns]
    scored_df = scored_df[available_columns]
    return scored_df

final_df = apply_dynamic_scores(
    df_base, dynamic_weights, dynamic_artist_weights, dynamic_title_weights, 
    tag_freq, artist_freq, title_word_freq, global_tag_weight, global_artist_weight, global_title_weight
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

min_score = st.sidebar.slider(
    "最低推荐评分阈值：", 
    min_value=min_possible_score, 
    max_value=max_possible_score, 
    value=default_min_slider
)
filtered_df = final_df[final_df['推荐评分'] >= min_score]

if search_kw and not filtered_df.empty:
    kw_list = [kw.strip().lower() for kw in search_kw.replace('，', ',').split(',') if kw.strip()]
    for kw in kw_list:
        mask_search = (
            filtered_df['标题'].str.lower().str.contains(kw, regex=False, na=False) |
            filtered_df['标签'].str.lower().str.contains(kw, regex=False, na=False) |
            filtered_df['作者'].str.lower().str.contains(kw, regex=False, na=False) |
            filtered_df['团队'].str.lower().str.contains(kw, regex=False, na=False)
        )
        filtered_df = filtered_df[mask_search]

# AI 语义二次过滤
if vector_search_kw and not filtered_df.empty:
    with st.spinner('正在唤醒 AI 引擎并载入矩阵空间 (首次唤醒需一段时间)...'):
        embed_model, corpus_embeddings, corpus_ids, id_to_index = load_semantic_engine()
    with st.spinner('正在当前结果集中进行 AI 语义碰撞...'):
        import torch
        from sentence_transformers import util
        # 找出剩下的所有链接
        surviving_links = filtered_df['链接'].astype(str).tolist()
        
        # 通过字典查出这些链接在总矩阵里对应的行号
        valid_indices = [id_to_index[link] for link in surviving_links if link in id_to_index]
        
        if valid_indices:
            # 切出只属于这批漫画的子矩阵
            sub_embeddings = corpus_embeddings[valid_indices]
            sub_ids = [corpus_ids[i] for i in valid_indices]
            
            # 将用户的搜索词转化为张量向量
            query_embedding = embed_model.encode([vector_search_kw], convert_to_tensor=True)

            # 让子矩阵转移至 query_embedding 所在的设备 (GPU/CPU)
            sub_embeddings = sub_embeddings.to(query_embedding.device)
            
            # 余弦相似度计算
            cos_scores = util.cos_sim(query_embedding, sub_embeddings)[0]
            
            # 提取前 1000 名
            top_k = min(1000, len(sub_ids))
            top_results = torch.topk(cos_scores, k=top_k)
            
            # 提取排序好的链接，同时提取出对应的分数转成百分制
            matched_links = [sub_ids[idx] for idx in top_results[1]]
            matched_scores = (top_results[0] * 100).tolist()  # 将余弦相似度放大为 0~100 的百分制
            score_map = dict(zip(matched_links, matched_scores))
            
            # 用这批链接过滤 DataFrame，并写入实体化的分数新列
            filtered_df = filtered_df[filtered_df['链接'].astype(str).isin(matched_links)].copy()
            filtered_df['AI相关度'] = filtered_df['链接'].astype(str).map(score_map)
            
            # 默认直接按 AI 相关度从高到低排序
            filtered_df = filtered_df.sort_values('AI相关度', ascending=False).reset_index(drop=True)
        else:
            # 索引未匹配，返回空
            filtered_df = pd.DataFrame()

# 主动释放无需再使用的全量内存
del final_df
gc.collect()

# UI 渲染
st.title("墨白的地下金库:P")

st.markdown("### 📂 打开本地漫画")
st.write("在下拉框中选择漫画打开本地文件夹 (已过滤屏蔽标签)")

if not filtered_df.empty:
    manga_options = dict(zip(filtered_df['标题'], filtered_df['本地目录']))
    selected_manga_title = st.selectbox("选择要阅读的漫画：", options=list(manga_options.keys()))
    
    selected_path = manga_options[selected_manga_title]
    
    col_btn, col_path = st.columns([1, 4])
    with col_btn:
        if st.button("打开本地文件夹"):
            if selected_path != "本地目录不存在" and os.path.exists(selected_path):
                os.startfile(selected_path)
                st.toast(f"已打开文件夹: {selected_path}", icon="✅")
            else:
                st.error(f"无法打开，请检查该目录是否存在：{selected_path}")
    with col_path:
        st.info(f"匹配路径: {selected_path}")
else:
    st.warning("当前筛选条件下没有匹配的漫画。")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.metric("当前显示条目数", f"{len(filtered_df)} 册")
col2.metric("总收录作者数", f"{len(artist_freq)} 位")
col3.metric("总标签种类", f"{len(tag_freq)} 种")
col4.metric("解析标题词汇数", f"{len(title_word_freq)} 种")

st.subheader("库存列表")

if not filtered_df.empty:
    sort_columns = ['推荐评分', '上传日期', '标题', '作者', '团队', '标签', '语言', '页数', '本地目录']

    # 如经过了 AI 检索，把 'AI相关度' 加入排序选项，并默认放在首位
    if 'AI相关度' in filtered_df.columns:
        sort_columns.insert(0, 'AI相关度')
    
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

    MAX_DISPLAY = 500
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
            lambda row: get_cover_base64(row['本地目录'], row.get('链接', '')), 
            axis=1
        )

    display_df = display_df.drop(columns=['文件名'], errors='ignore')

    cols = display_df.columns.tolist()
    if '封面' in cols:
        cols.remove('封面')
        cols.insert(0, '封面')
    display_df = display_df[cols]

    st.dataframe(
        display_df,
        column_config={
            "封面": st.column_config.ImageColumn("封面", help="本地文件夹中的 1.xxx 封面图"),
            "链接": st.column_config.LinkColumn("图库链接", display_text="网络来源"),
            "推荐评分": st.column_config.ProgressColumn(
                "推荐评分", 
                format="%d", 
                min_value=min_possible_score, 
                max_value=max_possible_score
            ),
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

st.markdown("---")
st.subheader("全局偏好数据")
chart_col1, chart_col2, chart_col3 = st.columns(3)

with chart_col1:
    st.write("**Top 15 XP 标签分布**")
    top_tags = tag_freq.most_common(15)
    if top_tags:
        tag_chart_df = pd.DataFrame(top_tags, columns=['标签', '频次']).set_index('标签')
        st.bar_chart(tag_chart_df)
        
    with st.expander("🔍 查看 Top 150 XP 标签"):
        top_150_tags = tag_freq.most_common(150)
        if top_150_tags:
            top_150_tags_df = pd.DataFrame(top_150_tags, columns=['XP 标签', '出现频次'])
            st.dataframe(top_150_tags_df, hide_index=True, use_container_width=True)

with chart_col2:
    st.write("**Top 15 核心作者分布**")
    top_artists = artist_freq.most_common(15)
    if top_artists:
        artist_chart_df = pd.DataFrame(top_artists, columns=['作者', '频次']).set_index('作者')
        st.bar_chart(artist_chart_df)
        
    with st.expander("🔍 查看 Top 150 核心作者"):
        top_150_artists = artist_freq.most_common(150)
        if top_150_artists:
            top_150_artists_df = pd.DataFrame(top_150_artists, columns=['作者名', '收录册数'])
            st.dataframe(top_150_artists_df, hide_index=True, use_container_width=True)

with chart_col3:
    st.write("**Top 15 标题高频词汇**")
    top_words = title_word_freq.most_common(15)
    if top_words:
        word_chart_df = pd.DataFrame(top_words, columns=['词汇', '频次']).set_index('词汇')
        st.bar_chart(word_chart_df)
        
    with st.expander("🔍 查看 Top 150 标题高频词汇"):
        top_150_words = title_word_freq.most_common(150)
        if top_150_words:
            top_150_df = pd.DataFrame(top_150_words, columns=['特征词汇', '出现频次'])
            st.dataframe(top_150_df, hide_index=True, use_container_width=True)