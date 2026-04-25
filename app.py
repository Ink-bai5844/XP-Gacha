import os
import math
import gc
import hashlib
import pandas as pd
import streamlit as st

from config import INITIAL_TAG_WEIGHTS, MAX_DISPLAY
from data_pipeline import load_base_data, apply_dynamic_scores
from utils_core import get_cover_base64
from utils_nlp import load_semantic_engine
from utils_chat import render_chat_interface

st.set_page_config(page_title="地下金库(Local)", layout="wide")
st.markdown(
    """
    <style>
    /* 针对表格图片放大悬浮窗进行向右平移 */
    div[id^="gdg-overlay-"] {
        margin-left: 100px !important;
        z-index: 99999 !important;
        border-radius: 8px !important;
        box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.4) !important;
        overflow: hidden !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def build_vector_search_signature(query, candidate_links):
    normalized_query = str(query).strip()
    joined_links = "\n".join(str(link) for link in candidate_links)
    raw = f"{normalized_query}\n{joined_links}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

with st.spinner('正在同步预处理缓存与计算引擎...'):
    df_base, tag_freq, artist_freq, title_word_freq = load_base_data()

# 侧边栏
st.sidebar.title("筛选与偏好设置")

if df_base is None:
    st.error("未找到数据文件！")
    st.stop()

search_kw = st.sidebar.text_input("实时关键词搜索 (标题/标签/作者)：", placeholder="例如: elf...")
vector_search_kw = st.sidebar.text_input("AI 语义检索 (自然语言)：", placeholder="例如: 猫娘X狐娘...")
st.sidebar.markdown("---")
st.sidebar.subheader("全局评分权重分配")

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

# 动态打分与过滤
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
    surviving_links = filtered_df['链接'].astype(str).tolist()
    current_vector_signature = build_vector_search_signature(vector_search_kw, surviving_links)
    cached_vector_signature = st.session_state.get("vector_search_signature")
    cached_vector_df = st.session_state.get("vector_search_result_df")

    if cached_vector_signature == current_vector_signature and cached_vector_df is not None:
        filtered_df = cached_vector_df.copy()
    else:
        with st.spinner('正在唤醒 AI 引擎并载入矩阵空间 (首次唤醒需一段时间)...'):
            embed_model, corpus_embeddings, corpus_ids, id_to_index = load_semantic_engine()
        with st.spinner('正在当前结果集中进行 AI 语义碰撞...'):
            import torch
            from sentence_transformers import util

            valid_indices = [id_to_index[link] for link in surviving_links if link in id_to_index]

            if valid_indices:
                sub_embeddings = corpus_embeddings[valid_indices]
                sub_ids = [corpus_ids[i] for i in valid_indices]

                query_embedding = embed_model.encode([vector_search_kw], convert_to_tensor=True)
                sub_embeddings = sub_embeddings.to(query_embedding.device)

                cos_scores = util.cos_sim(query_embedding, sub_embeddings)[0]
                top_k = min(5000, len(sub_ids))
                top_results = torch.topk(cos_scores, k=top_k)

                matched_links = [sub_ids[idx] for idx in top_results[1]]
                matched_scores = (top_results[0] * 100).tolist()
                score_map = dict(zip(matched_links, matched_scores))

                filtered_df = filtered_df[filtered_df['链接'].astype(str).isin(matched_links)].copy()
                filtered_df['AI相关度'] = filtered_df['链接'].astype(str).map(score_map)
                filtered_df = filtered_df.sort_values('AI相关度', ascending=False).reset_index(drop=True)
            else:
                filtered_df = pd.DataFrame()

        st.session_state["vector_search_signature"] = current_vector_signature
        st.session_state["vector_search_result_df"] = filtered_df.copy()

# 主动释放无需再使用的全量内存
del final_df
gc.collect()

# UI 渲染层
st.title("墨白的地下金库:P")

col1, col2, col3, col4 = st.columns(4)
col1.metric("当前显示条目数", f"{len(filtered_df)} 册")
col2.metric("总收录作者数", f"{len(artist_freq)} 位")
col3.metric("总标签种类", f"{len(tag_freq)} 种")
col4.metric("解析标题词汇数", f"{len(title_word_freq)} 种")

st.subheader("库存列表")

if not filtered_df.empty:
    sort_columns = ['推荐评分', '上传日期', '标题', '作者', '团队', '标签', '语言', '页数', '本地目录']
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
        width='stretch',
        height=600 
    )

    chat_context_df = display_df.drop(columns=['封面'], errors='ignore').copy()

    del display_df
    gc.collect()

else:
    st.info("没有可以显示的数据喔。")

# 传入纯文本上下文 df 进行渲染
if 'chat_context_df' in locals():
    render_chat_interface(chat_context_df)
else:
    render_chat_interface(None)

st.markdown("---")

st.markdown("### 📂 打开本地漫画")
st.write("在下拉框中选择漫画打开本地文件夹 (已过滤屏蔽标签)")

@st.fragment
def render_local_opener(filtered_df):
    if not filtered_df.empty:
        manga_options = dict(zip(filtered_df['标题'], filtered_df['本地目录']))
        selected_manga_title = st.selectbox(
            "选择要阅读的漫画：", 
            options=list(manga_options.keys()),
            key="manga_selector" # 显式指定 key 保证状态稳定
        )
        
        selected_path = manga_options[selected_manga_title]
        
        col_btn, col_path = st.columns([1, 4])
        with col_btn:
            if st.button("打开本地文件夹", use_container_width=True):
                if selected_path != "本地目录不存在" and os.path.exists(selected_path):
                    os.startfile(selected_path)
                    st.toast(f"已成功开启: {selected_path}", icon="✅")
                else:
                    st.error(f"路径失效：{selected_path}")
        with col_path:
            st.info(f"匹配路径: {selected_path}")
    else:
        st.warning("当前筛选条件下没有匹配的漫画。")

# 执行局部渲染函数
render_local_opener(filtered_df)

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
            st.dataframe(top_150_tags_df, hide_index=True, width="stretch")

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
            st.dataframe(top_150_artists_df, hide_index=True, width="stretch")

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
            st.dataframe(top_150_df, hide_index=True, width="stretch")
