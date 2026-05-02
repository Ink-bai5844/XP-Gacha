import os
import math
import gc
import hashlib
import pandas as pd
import streamlit as st

from config import (
    COVER_SEARCH_TOP_K,
    HISTORY_RECOMMENDATION_CACHE_SIZE,
    IMG_VECTOR_FILE,
    INITIAL_TAG_WEIGHTS,
    MAX_DISPLAY,
    SEMANTIC_SEARCH_TOP_K,
)
from data_pipeline import load_base_data, apply_dynamic_scores
from utils_charts import render_global_preference_charts, render_history_preference_charts
from utils_core import get_cover_base64
from utils_cv import search_similar_cover_items
from utils_history import (
    build_history_preference_maps,
    build_tracked_link,
    clear_history_entries,
    load_history_entries,
    record_recommendation_history,
    start_link_tracking_server,
)
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


def build_vector_search_signature(query, candidate_ids):
    normalized_query = str(query).strip()
    joined_ids = "\n".join(str(item_id) for item_id in candidate_ids)
    raw = f"{normalized_query}\n{joined_ids}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def build_cover_search_signature(query_item_id, query_image_bytes, candidate_ids):
    normalized_id = str(query_item_id).strip().upper()
    image_digest = hashlib.md5(query_image_bytes).hexdigest() if query_image_bytes else ""
    joined_ids = "\n".join(str(item_id) for item_id in candidate_ids)
    raw = f"{normalized_id}\n{image_digest}\n{joined_ids}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def apply_similarity_result(filtered_df, matched_ids, score_map, score_column):
    if filtered_df.empty or not matched_ids:
        return pd.DataFrame()

    normalized_matched_ids = [str(item_id) for item_id in matched_ids]
    result_df = filtered_df[filtered_df['ID'].astype(str).isin(normalized_matched_ids)].copy()
    result_df[score_column] = result_df['ID'].astype(str).map(score_map)
    result_df = result_df.sort_values(score_column, ascending=False).reset_index(drop=True)
    return result_df


def _get_item_label(item_payload):
    item_id = str(item_payload.get("ID", "")).strip()
    title = str(item_payload.get("标题", "")).strip()
    return f"{item_id} | {title}" if item_id and title else (title or item_id or "当前条目")


def open_local_history_item(item_payload):
    selected_path = str(item_payload.get("本地目录", "")).strip()
    if selected_path != "本地目录不存在" and os.path.exists(selected_path):
        record_recommendation_history(item_payload, "local_folder")
        os.startfile(selected_path)
        st.session_state["open_item_notice"] = f"已记录并打开本地目录：{_get_item_label(item_payload)}"
    else:
        st.session_state["open_item_error"] = f"路径失效：{selected_path}"


with st.spinner('正在同步预处理缓存与计算引擎...'):
    df_base, tag_freq, artist_freq, title_word_freq, preference_chart_cache, score_cache = load_base_data()

link_tracking_server = start_link_tracking_server()

open_item_notice = st.session_state.pop("open_item_notice", None)
if open_item_notice:
    st.toast(open_item_notice, icon="✅")

open_item_error = st.session_state.pop("open_item_error", None)
if open_item_error:
    st.toast(open_item_error, icon="⚠️")

# 侧边栏
st.sidebar.title("筛选与偏好设置")

if df_base is None:
    st.error("未找到数据文件！")
    st.stop()

search_kw = st.sidebar.text_input("实时关键词搜索 (标题/标签/作者)：", placeholder="例如: elf...")
vector_search_kw = st.sidebar.text_input("AI 语义检索 (自然语言)：", placeholder="例如: 猫娘X狐娘...")
with st.sidebar.expander("封面相似检索 (CLIP)", expanded=False):
    cover_query_id = st.text_input(
        "输入库内条目 ID：",
        placeholder="例如: JM1426534 / NH123456",
    )
    cover_query_file = st.file_uploader(
        "或上传一张图片：",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=False,
    )
    st.caption(
        f"上传图片优先于 ID；会在当前候选结果里按封面相似度筛到前 {COVER_SEARCH_TOP_K} 项。"
    )
    if cover_query_file is not None:
        st.image(cover_query_file, caption="当前上传图片", width="stretch")
st.sidebar.markdown("---")
st.sidebar.subheader("全局评分权重分配")

global_tag_weight = st.sidebar.slider("标签总分倍率", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
global_artist_weight = st.sidebar.slider("作者总分倍率", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
global_title_weight = st.sidebar.slider("标题总分倍率", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

history_entries = load_history_entries()
with st.sidebar.expander("历史偏好加权", expanded=False):
    global_history_weight = st.slider("历史偏好总分倍率", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    st.caption(f"缓存最近 {HISTORY_RECOMMENDATION_CACHE_SIZE} 次打开记录")
    st.caption(f"当前已记录 {len(history_entries)} 次打开。")
    if link_tracking_server is None:
        st.caption("网络链接追踪器未启动，表格链接会直接打开但不会写入历史")
    col_refresh_history, col_clear_history = st.columns(2)
    with col_refresh_history:
        if st.button("刷新记录", width="stretch"):
            st.session_state["open_item_notice"] = "已刷新历史偏好记录"
            st.rerun()
    with col_clear_history:
        if st.button("清空记录", width="stretch"):
            clear_history_entries()
            st.session_state["open_item_notice"] = "已清空历史偏好记录"
            st.rerun()

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

history_preference = (
    build_history_preference_maps(
        history_entries,
        tag_freq,
        title_word_freq,
        artist_freq,
        tag_bonus_scale=global_tag_weight,
        title_bonus_scale=global_title_weight,
        artist_bonus_scale=global_artist_weight,
    )
    if global_history_weight > 0 and history_entries
    else None
)

# 动态打分与过滤
final_df = apply_dynamic_scores(
    df_base, dynamic_weights, dynamic_artist_weights, dynamic_title_weights, 
    tag_freq, artist_freq, title_word_freq, global_tag_weight, global_artist_weight, global_title_weight,
    score_cache=score_cache,
    history_preference=history_preference,
    global_history_w=global_history_weight,
)

if blocked_tags:
    mask_not_blocked = final_df['解析后标签'].apply(lambda x: not any(t in blocked_tags for t in x))
    final_df = final_df[mask_not_blocked]

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
        if '搜索文本' in filtered_df.columns:
            mask_search = filtered_df['搜索文本'].str.contains(kw, regex=False, na=False)
        else:
            mask_search = (
                filtered_df['ID'].str.lower().str.contains(kw, regex=False, na=False) |
                filtered_df['标题'].str.lower().str.contains(kw, regex=False, na=False) |
                filtered_df['标签'].str.lower().str.contains(kw, regex=False, na=False) |
                filtered_df['作者'].str.lower().str.contains(kw, regex=False, na=False) |
                filtered_df['团队'].str.lower().str.contains(kw, regex=False, na=False)
            )
        filtered_df = filtered_df[mask_search]

# AI 语义二次过滤
if vector_search_kw and not filtered_df.empty:
    surviving_ids = filtered_df['ID'].astype(str).tolist()
    current_vector_signature = build_vector_search_signature(vector_search_kw, surviving_ids)
    cached_vector_signature = st.session_state.get("vector_search_signature")
    cached_vector_payload = st.session_state.get("vector_search_result_payload")

    if cached_vector_signature == current_vector_signature and cached_vector_payload is not None:
        filtered_df = apply_similarity_result(
            filtered_df,
            cached_vector_payload["matched_ids"],
            cached_vector_payload["score_map"],
            "AI相关度",
        )
    else:
        with st.spinner('正在唤醒 AI 引擎并载入矩阵空间 (首次唤醒需一段时间)...'):
            embed_model, corpus_embeddings, corpus_ids, id_to_index = load_semantic_engine()
        with st.spinner('正在当前结果集中进行 AI 语义碰撞...'):
            import torch
            from sentence_transformers import util

            valid_indices = [id_to_index[item_id] for item_id in surviving_ids if item_id in id_to_index]

            if valid_indices:
                sub_embeddings = corpus_embeddings[valid_indices]
                sub_ids = [corpus_ids[i] for i in valid_indices]

                query_embedding = embed_model.encode([vector_search_kw], convert_to_tensor=True)
                if query_embedding.device.type == "cpu":
                    query_embedding = query_embedding.float()
                    sub_embeddings = sub_embeddings.to(query_embedding.device, dtype=query_embedding.dtype)
                else:
                    sub_embeddings = sub_embeddings.to(query_embedding.device, dtype=query_embedding.dtype)

                cos_scores = util.cos_sim(query_embedding, sub_embeddings)[0]
                top_k = min(SEMANTIC_SEARCH_TOP_K, len(sub_ids))
                top_results = torch.topk(cos_scores, k=top_k)

                matched_ids = [sub_ids[idx] for idx in top_results[1]]
                matched_scores = (top_results[0] * 100).tolist()
                score_map = dict(zip(matched_ids, matched_scores))

                filtered_df = apply_similarity_result(
                    filtered_df,
                    matched_ids,
                    score_map,
                    "AI相关度",
                )
                st.session_state["vector_search_signature"] = current_vector_signature
                st.session_state["vector_search_result_payload"] = {
                    "matched_ids": matched_ids,
                    "score_map": score_map,
                }
            else:
                filtered_df = pd.DataFrame()
                st.session_state["vector_search_signature"] = current_vector_signature
                st.session_state["vector_search_result_payload"] = {
                    "matched_ids": [],
                    "score_map": {},
                }

cover_query_bytes = cover_query_file.getvalue() if cover_query_file is not None else None
cover_query_id = cover_query_id.strip().upper()

if (cover_query_bytes or cover_query_id) and not filtered_df.empty:
    surviving_ids = filtered_df['ID'].astype(str).tolist()
    current_cover_signature = build_cover_search_signature(
        cover_query_id,
        cover_query_bytes,
        surviving_ids,
    )
    cached_cover_signature = st.session_state.get("cover_search_signature")
    cached_cover_payload = st.session_state.get("cover_search_result_payload")

    if cached_cover_signature == current_cover_signature and cached_cover_payload is not None:
        filtered_df = apply_similarity_result(
            filtered_df,
            cached_cover_payload["matched_ids"],
            cached_cover_payload["score_map"],
            "封面相关度",
        )
    else:
        with st.spinner('正在进行封面向量相似检索...'):
            try:
                cover_search_payload = search_similar_cover_items(
                    query_item_id=cover_query_id,
                    query_image_bytes=cover_query_bytes,
                    candidate_ids=surviving_ids,
                    top_k=COVER_SEARCH_TOP_K,
                )
            except FileNotFoundError:
                st.warning(
                    f"封面向量检索暂时不可用：未找到向量文件 `{IMG_VECTOR_FILE}`。"
                )
            except ValueError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"封面向量检索失败：{exc}")
            else:
                cover_results = cover_search_payload["results"]
                if cover_results:
                    matched_ids = [item["item_id"] for item in cover_results]
                    score_map = {item["item_id"]: item["score"] for item in cover_results}

                    filtered_df = apply_similarity_result(
                        filtered_df,
                        matched_ids,
                        score_map,
                        "封面相关度",
                    )

                    st.session_state["cover_search_signature"] = current_cover_signature
                    st.session_state["cover_search_result_payload"] = {
                        "matched_ids": matched_ids,
                        "score_map": score_map,
                    }
                    st.session_state["cover_search_meta"] = cover_search_payload
                else:
                    filtered_df = pd.DataFrame()
                    st.session_state["cover_search_signature"] = current_cover_signature
                    st.session_state["cover_search_result_payload"] = {
                        "matched_ids": [],
                        "score_map": {},
                    }
                    st.info("当前候选结果里没有命中可用的封面向量。")
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
current_page_opener_df = pd.DataFrame()

if not filtered_df.empty:
    sort_columns = ['推荐评分', 'ID', '上传日期', '标题', '作者', '团队', '标签', '语言', '页数', '本地目录']
    if '封面相关度' in filtered_df.columns:
        sort_columns.insert(0, '封面相关度')
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
    opener_columns = [
        col
        for col in ['ID', '标题', '作者', '本地目录', '链接', '解析后标签', '标题特征词']
        if col in display_df.columns
    ]
    current_page_opener_df = display_df[opener_columns].copy()

    with st.spinner(f'正在加载 {selected_page_label} 范围的缩略图...'):
        display_df['封面'] = display_df.apply(
            lambda row: get_cover_base64(row['本地目录'], row.get('ID', ''), row.get('链接', '')), 
            axis=1
        )

    chat_context_df = display_df.drop(
        columns=['封面', '解析后标签', '标题特征词', '搜索文本'],
        errors='ignore',
    ).copy()

    table_df = display_df.drop(
        columns=['文件名', '解析后标签', '标题特征词', '搜索文本'],
        errors='ignore',
    )
    if link_tracking_server is not None and '链接' in table_df.columns:
        table_df['链接'] = display_df.apply(build_tracked_link, axis=1)

    preferred_columns = [
        '封面', '封面相关度', 'AI相关度', '推荐评分', 'ID', '上传日期',
        '标题', '作者', '团队', '标签', '语言', '页数',
        '本地目录', '链接'
    ]
    display_columns = [col for col in preferred_columns if col in table_df.columns]
    display_columns += [col for col in table_df.columns if col not in display_columns]
    table_df = table_df[display_columns]

    st.dataframe(
        table_df,
        column_config={
            "封面": st.column_config.ImageColumn("封面", help="本地文件夹中的 1.xxx 封面图"),
            "链接": st.column_config.LinkColumn("图库链接", display_text="网络来源"),
            "封面相关度": st.column_config.NumberColumn("封面相关度", format="%.2f"),
            "AI相关度": st.column_config.NumberColumn("AI相关度", format="%.2f"),
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
        width='stretch',
        height=600 
    )

    del table_df
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

def render_item_opener(filtered_df):
    if not filtered_df.empty:
        manga_options = {}
        for row_index, row in filtered_df.iterrows():
            item_payload = row.to_dict()
            manga_id = str(item_payload.get('ID', '')).strip() or f"row-{row_index}"
            manga_options[manga_id] = item_payload

        selected_manga_id = st.selectbox(
            "选择要阅读的漫画：",
            options=list(manga_options.keys()),
            format_func=lambda manga_id: _get_item_label(manga_options[manga_id]),
            key="manga_selector" # 显式指定 key 保证状态稳定
        )
        
        selected_item = manga_options[selected_manga_id]
        selected_path = str(selected_item.get("本地目录", "")).strip()
        col_btn, col_path = st.columns([1, 4])
        with col_btn:
            st.button(
                "打开本地文件夹",
                width="stretch",
                on_click=open_local_history_item,
                args=(selected_item,),
            )
        with col_path:
            st.info(f"匹配路径: {selected_path}")
    else:
        st.warning("当前筛选条件下没有匹配的漫画。")

render_item_opener(current_page_opener_df)

render_global_preference_charts(preference_chart_cache)
render_history_preference_charts(history_entries)
