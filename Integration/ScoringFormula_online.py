import os
import re
import json
import hashlib
import math
import pickle
import gc
import streamlit as st
import pandas as pd
from collections import Counter
from janome.tokenizer import Tokenizer
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
                return pickle.load(f)

    print("缓存失效或无缓存，执行数据库全量拉取与自然语言处理...")
    
    # 从 MySQL 拉取数据
    try:
        df = pd.read_sql("SELECT * FROM gallery_info", con=engine)
    except Exception as e:
        print(f"数据库读取失败: {e}")
        return None, Counter(), Counter(), Counter()

    if df.empty:
        return None, Counter(), Counter(), Counter()

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

    result_tuple = (df, tag_frequencies, artist_frequencies, title_word_frequencies)

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
    df_base, tag_freq, artist_freq, title_word_freq = load_base_data()

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
    global_title_weight
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
