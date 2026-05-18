import os
import math
import gc
import hashlib
import html
import json
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
from data_pipeline import (
    apply_dynamic_scores,
    fetch_gallery_rows_by_ids,
    get_row_indices_for_ids,
    load_base_data,
    search_gallery_candidate_ids,
)
from utils_charts import render_global_preference_charts, render_history_preference_charts
from utils_core import get_cover_base64
from utils_cv import search_similar_cover_items
from utils_history import (
    build_history_preference_maps,
    build_tracked_link,
    clear_history_entries,
    load_history_entries,
    record_recommendation_history,
    save_history_entries,
    start_link_tracking_server,
)
from utils_nlp import load_semantic_engine
from utils_chat import render_chat_interface
from ui_data_processing import render_data_processing_interface

SELECTED_MANGA_STATE_KEY = "selected_manga_id"

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
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    [data-testid="stMetric"] {
        border: 1px solid rgba(49, 51, 63, 0.14);
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        background: rgba(250, 250, 250, 0.72);
    }
    .manga-title {
        font-size: 1.45rem;
        font-weight: 750;
        margin-bottom: 0.1rem;
    }
    .muted-line {
        color: rgba(49, 51, 63, 0.70);
        margin-bottom: 0.55rem;
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


def build_score_state_signature(
    tag_weights,
    artist_weights,
    title_weights,
    global_weights,
    history_entries,
    candidate_ids=None,
):
    if candidate_ids is None:
        candidate_digest = "ALL"
    else:
        joined_ids = "\n".join(str(item_id) for item_id in candidate_ids)
        candidate_digest = hashlib.md5(joined_ids.encode("utf-8")).hexdigest()

    history_digest = hashlib.md5(
        json.dumps(history_entries, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    payload = {
        "tag_weights": tag_weights,
        "artist_weights": artist_weights,
        "title_weights": title_weights,
        "global_weights": global_weights,
        "history_digest": history_digest,
        "candidate_digest": candidate_digest,
    }
    return hashlib.md5(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def build_keyword_recall_signature(search_text, include_relevance):
    normalized_search = " ".join(str(search_text or "").strip().split())
    payload = {
        "search": normalized_search,
        "include_relevance": bool(include_relevance),
    }
    return hashlib.md5(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


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


def current_selected_manga_id():
    selected_manga_id = st.session_state.get(SELECTED_MANGA_STATE_KEY)
    return str(selected_manga_id) if selected_manga_id else ""


def make_selectable_table(table_df):
    editable_df = table_df.copy()
    selected_manga_id = current_selected_manga_id()
    editable_df["选中"] = editable_df["ID"].astype(str).eq(selected_manga_id)
    return editable_df


def apply_table_selection(edited_df, source_df):
    if "选中" not in edited_df.columns or "ID" not in edited_df.columns:
        return

    selected_rows = edited_df[edited_df["选中"].fillna(False)]
    if selected_rows.empty:
        return

    current_manga_id = current_selected_manga_id()
    newly_selected = selected_rows[selected_rows["ID"].astype(str) != current_manga_id]
    chosen_row = newly_selected.iloc[0] if not newly_selected.empty else selected_rows.iloc[0]
    chosen_manga_id = str(chosen_row["ID"])

    if chosen_manga_id != current_manga_id:
        matched_rows = source_df[source_df["ID"].astype(str) == chosen_manga_id]
        history_row = matched_rows.iloc[0] if not matched_rows.empty else chosen_row
        st.session_state[SELECTED_MANGA_STATE_KEY] = chosen_manga_id
        st.toast(f"已选中：{_get_item_label(history_row)}", icon="✅")
        st.rerun()


def _normalize_copy_value(value):
    try:
        is_missing = pd.isna(value)
    except TypeError:
        is_missing = False
    if isinstance(is_missing, bool) and is_missing:
        return ""
    text = str(value)
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    text = text.replace("\t", " ")
    return " ".join(text.split())


def build_current_page_copy_text(table_df):
    copy_df = table_df.drop(columns=["封面", "选中"], errors="ignore").copy()
    for column in copy_df.columns:
        copy_df[column] = copy_df[column].map(_normalize_copy_value)
    return copy_df.to_csv(sep="\t", index=False).rstrip()


def render_copy_page_button(copy_text, row_count):
    payload = json.dumps(copy_text, ensure_ascii=False).replace("</", "<\\/")
    disabled = "true" if not copy_text else "false"
    button_label = f"复制当前页 {row_count} 条" if row_count else "复制当前页"
    st.iframe(
        f"""
        <div class="copy-wrap">
            <button id="copy-current-page" type="button" {"disabled" if not copy_text else ""}>
                {html.escape(button_label)}
            </button>
            <span id="copy-status"></span>
            <textarea id="copy-payload" aria-hidden="true"></textarea>
        </div>
        <style>
            body {{
                margin: 0;
                font-family: "Source Sans Pro", sans-serif;
            }}
            .copy-wrap {{
                height: 42px;
                display: flex;
                justify-content: flex-end;
                align-items: center;
                gap: 8px;
            }}
            #copy-current-page {{
                border: 1px solid rgba(49, 51, 63, 0.22);
                border-radius: 8px;
                background: #ffffff;
                color: rgb(49, 51, 63);
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                line-height: 1;
                padding: 0.55rem 0.75rem;
                white-space: nowrap;
            }}
            #copy-current-page:hover:not(:disabled) {{
                border-color: rgba(255, 75, 75, 0.75);
                color: rgb(255, 75, 75);
            }}
            #copy-current-page:disabled {{
                cursor: not-allowed;
                opacity: 0.45;
            }}
            #copy-status {{
                color: rgba(49, 51, 63, 0.70);
                font-size: 13px;
                min-width: 42px;
            }}
            #copy-payload {{
                position: fixed;
                left: -9999px;
                top: -9999px;
                width: 1px;
                height: 1px;
                opacity: 0;
            }}
        </style>
        <script>
            const copyText = {payload};
            const disabled = {disabled};
            const button = document.getElementById("copy-current-page");
            const status = document.getElementById("copy-status");
            const payloadBox = document.getElementById("copy-payload");

            async function copyCurrentPage() {{
                if (disabled || !copyText) {{
                    return;
                }}
                try {{
                    if (navigator.clipboard && window.isSecureContext) {{
                        await navigator.clipboard.writeText(copyText);
                    }} else {{
                        payloadBox.value = copyText;
                        payloadBox.focus();
                        payloadBox.select();
                        document.execCommand("copy");
                    }}
                    status.textContent = "已复制";
                    window.setTimeout(() => {{
                        status.textContent = "";
                    }}, 1800);
                }} catch (error) {{
                    status.textContent = "复制失败";
                    console.error(error);
                }}
            }}

            button.addEventListener("click", copyCurrentPage);
        </script>
        """,
        height=46,
    )


def get_selected_manga(filtered_df):
    selected_manga_id = current_selected_manga_id()
    if not selected_manga_id or filtered_df.empty:
        return None

    matched_rows = filtered_df[filtered_df["ID"].astype(str) == selected_manga_id]
    if matched_rows.empty:
        return None

    return matched_rows.iloc[0]


def coerce_display_list(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]

    return []


def build_history_table(history_entries):
    rows = []
    for index, entry in enumerate(history_entries):
        if not isinstance(entry, dict):
            continue

        tags = coerce_display_list(entry.get("tags"))
        title_words = coerce_display_list(entry.get("title_words"))
        rows.append(
            {
                "删除": False,
                "序号": index + 1,
                "打开时间": str(entry.get("opened_at", "")).strip(),
                "动作": str(entry.get("action", "")).strip(),
                "ID": str(entry.get("id", "")).strip(),
                "标题": str(entry.get("title", "")).strip(),
                "作者": str(entry.get("author", "")).strip(),
                "本地目录": str(entry.get("local_path", "")).strip(),
                "链接": str(entry.get("link", "")).strip(),
                "标签": " | ".join(str(tag).strip() for tag in tags[:8] if str(tag).strip()),
                "标题词": " | ".join(str(word).strip() for word in title_words[:8] if str(word).strip()),
            }
        )

    return pd.DataFrame(rows)


def format_detail_value(value):
    if value is None:
        return ""

    if isinstance(value, (list, tuple, set)):
        return " | ".join(str(item).strip() for item in value if str(item).strip())

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def is_valid_web_link(link):
    normalized_link = str(link or "").strip().lower()
    return normalized_link.startswith(("http://", "https://"))


def open_local_history_item(item_payload):
    selected_path = str(item_payload.get("本地目录", "")).strip()
    if selected_path != "本地目录不存在" and os.path.exists(selected_path):
        record_recommendation_history(item_payload, "local_folder")
        os.startfile(selected_path)
        st.session_state["open_item_notice"] = f"已记录并打开本地目录：{_get_item_label(item_payload)}"
    else:
        st.session_state["open_item_error"] = f"路径失效：{selected_path}"


def render_manga_detail(manga):
    left, right = st.columns([1.1, 2.4], gap="large")
    manga_payload = manga.to_dict() if hasattr(manga, "to_dict") else dict(manga)

    with left:
        cover_data = get_cover_base64(
            manga_payload.get("本地目录", ""),
            manga_payload.get("ID", ""),
            manga_payload.get("链接", ""),
        )
        if cover_data:
            st.image(cover_data, width="stretch")
        else:
            st.info("暂时没有可用封面。")

        link = format_detail_value(manga_payload.get("链接"))
        if is_valid_web_link(link):
            detail_link = build_tracked_link(manga_payload) if link_tracking_server is not None else link
            st.link_button("打开网络来源", detail_link, width="stretch")

        local_path = format_detail_value(manga_payload.get("本地目录"))
        if local_path and local_path != "本地目录不存在" and os.path.exists(local_path):
            st.button(
                "打开本地目录",
                width="stretch",
                key=f"open-local-{manga_payload.get('ID')}",
                on_click=open_local_history_item,
                args=(manga_payload,),
            )
            st.caption("本地目录")
            st.code(local_path)
        else:
            st.warning(f"本地目录不可用：{local_path or '未匹配'}")

    with right:
        st.markdown(
            f"<div class='manga-title'>{html.escape(format_detail_value(manga_payload.get('标题')))}</div>",
            unsafe_allow_html=True,
        )
        subtitle_parts = [
            format_detail_value(manga_payload.get("ID")),
            format_detail_value(manga_payload.get("作者")),
            format_detail_value(manga_payload.get("团队")),
        ]
        subtitle = " · ".join(part for part in subtitle_parts if part)
        st.markdown(
            f"<div class='muted-line'>{html.escape(subtitle)}</div>",
            unsafe_allow_html=True,
        )

        meta_cols = st.columns(4)
        meta_cols[0].metric("推荐分", format_detail_value(manga_payload.get("推荐评分")) or "0")
        meta_cols[1].metric("上传日期", format_detail_value(manga_payload.get("上传日期")) or "-")
        meta_cols[2].metric("语言", format_detail_value(manga_payload.get("语言")) or "-")
        meta_cols[3].metric("页数", format_detail_value(manga_payload.get("页数")) or "-")

        priority_fields = [
            "推荐评分",
            "关键词相关度",
            "封面相关度",
            "AI相关度",
            "ID",
            "标题",
            "作者",
            "团队",
            "上传日期",
            "语言",
            "页数",
            "标签",
            "解析后标签",
            "标题特征词",
            "本地目录",
            "文件名",
            "链接",
            "搜索文本",
        ]
        ordered_fields = [field for field in priority_fields if field in manga_payload]
        ordered_fields += [field for field in manga_payload if field not in ordered_fields and field != "封面"]
        info_rows = [
            (field, format_detail_value(manga_payload.get(field)))
            for field in ordered_fields
            if field != "封面"
        ]
        info_df = pd.DataFrame(info_rows, columns=["字段", "内容"])
        st.dataframe(info_df, hide_index=True, width="stretch", height=560)


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
keyword_relevance_enabled = st.sidebar.toggle(
    "启用关键词相关度",
    value=False,
    help="开启后会显示并可按数据库全文相关度排序；关闭时只用关键词做候选召回。",
)
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
global_history_weight = st.sidebar.slider("历史偏好总分倍率", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

history_entries = load_history_entries()

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

# 数据库候选召回 + 动态打分
search_payload = None
candidate_row_indices = None
if search_kw:
    recall_signature = build_keyword_recall_signature(search_kw, keyword_relevance_enabled)
    cached_recall_payload = st.session_state.get("keyword_recall_payload")
    if (
        isinstance(cached_recall_payload, dict)
        and cached_recall_payload.get("signature") == recall_signature
        and cached_recall_payload.get("payload") is not None
    ):
        search_payload = cached_recall_payload["payload"]
    else:
        with st.spinner("正在从 MySQL 召回关键词候选集..."):
            search_payload = search_gallery_candidate_ids(
                search_kw,
                include_relevance=keyword_relevance_enabled,
            )
        st.session_state["keyword_recall_payload"] = {
            "signature": recall_signature,
            "payload": search_payload,
        }
    candidate_row_indices = get_row_indices_for_ids(
        df_base,
        search_payload["ids"],
        score_cache.get("id_to_row") if score_cache else None,
    )

score_signature = build_score_state_signature(
    dynamic_weights,
    dynamic_artist_weights,
    dynamic_title_weights,
    {
        "tag": global_tag_weight,
        "artist": global_artist_weight,
        "title": global_title_weight,
        "history": global_history_weight,
        "keyword_relevance": keyword_relevance_enabled,
    },
    history_entries,
    candidate_ids=([search_kw, *search_payload["ids"]] if search_payload is not None else None),
)
cached_score_payload = st.session_state.get("score_result_payload")

if (
    isinstance(cached_score_payload, dict)
    and cached_score_payload.get("signature") == score_signature
    and cached_score_payload.get("df") is not None
):
    final_df = cached_score_payload["df"].copy()
else:
    final_df = apply_dynamic_scores(
        df_base, dynamic_weights, dynamic_artist_weights, dynamic_title_weights,
        tag_freq, artist_freq, title_word_freq, global_tag_weight, global_artist_weight, global_title_weight,
        score_cache=score_cache,
        history_preference=history_preference,
        global_history_w=global_history_weight,
        row_indices=candidate_row_indices,
    )

    if keyword_relevance_enabled and search_payload is not None and not final_df.empty:
        final_df["关键词相关度"] = final_df["ID"].astype(str).map(search_payload["score_map"]).fillna(0.0)

    st.session_state["score_result_payload"] = {
        "signature": score_signature,
        "df": final_df.copy(),
    }

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

if search_payload is not None:
    st.caption(
        f"关键词召回：{search_payload['mode']} · "
        f"{len(search_payload['ids'])} 个候选 · "
        f"{'已使用全文索引' if search_payload['used_fulltext'] else '未使用全文索引'}"
    )

chat_context_df = (
    filtered_df.drop(
        columns=['封面', '解析后标签', '标题特征词', '搜索文本'],
        errors='ignore',
    ).copy()
    if not filtered_df.empty
    else None
)

tab_library, tab_llm, tab_detail, tab_history, tab_data_processing = st.tabs(
    ["库存列表", "LLM 助手", "漫画详情", "历史记录", "数据处理"]
)

with tab_library:
    title_col, copy_col = st.columns([4, 1.6])
    with title_col:
        st.subheader("库存列表")
    copy_button_slot = copy_col.empty()

    if not filtered_df.empty:
        sort_columns = ['推荐评分', 'ID', '上传日期', '标题', '作者', '团队', '标签', '语言', '页数', '本地目录']
        if '关键词相关度' in filtered_df.columns:
            sort_columns.insert(0, '关键词相关度')
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
            sorted_df = filtered_df.sort_values(
                by=['推荐评分', '上传日期'],
                ascending=[is_ascending, False],
            ).reset_index(drop=True)
        else:
            sorted_df = filtered_df.sort_values(
                by=[global_sort_by],
                ascending=[is_ascending],
            ).reset_index(drop=True)

        total_items = len(sorted_df)
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

        display_df = sorted_df.iloc[slice_start:slice_end].copy()
        page_ids = display_df["ID"].astype(str).tolist()
        fresh_page_df = fetch_gallery_rows_by_ids(page_ids)
        if not fresh_page_df.empty:
            fresh_page_df = fresh_page_df.set_index("ID").reindex(page_ids).reset_index()
            for column_name in fresh_page_df.columns:
                if column_name == "ID":
                    continue
                if column_name in display_df.columns:
                    fresh_values = fresh_page_df[column_name]
                    display_df[column_name] = fresh_values.where(fresh_values.notna(), display_df[column_name])

        with st.spinner(f'正在加载 {selected_page_label} 范围的缩略图...'):
            display_df['封面'] = display_df.apply(
                lambda row: get_cover_base64(row['本地目录'], row.get('ID', ''), row.get('链接', '')),
                axis=1
            )

        table_df = display_df.drop(
            columns=['文件名', '解析后标签', '标题特征词', '搜索文本'],
            errors='ignore',
        )
        if link_tracking_server is not None and '链接' in table_df.columns:
            table_df['链接'] = display_df.apply(build_tracked_link, axis=1)

        preferred_columns = [
            '封面', '选中', '封面相关度', 'AI相关度', '关键词相关度', '推荐评分', 'ID', '上传日期',
            '标题', '作者', '团队', '标签', '语言', '页数', '本地目录', '链接'
        ]
        table_df = make_selectable_table(table_df)
        display_columns = [col for col in preferred_columns if col in table_df.columns]
        display_columns += [col for col in table_df.columns if col not in display_columns]
        table_df = table_df[display_columns]
        with copy_button_slot.container():
            render_copy_page_button(build_current_page_copy_text(table_df), len(table_df))

        edited_table = st.data_editor(
            table_df,
            column_config={
                "封面": st.column_config.ImageColumn("封面", help="本地文件夹或线上缓存中的封面图"),
                "选中": st.column_config.CheckboxColumn("选中", width="small"),
                "链接": st.column_config.LinkColumn("图库链接", display_text="网络来源"),
                "封面相关度": st.column_config.NumberColumn("封面相关度", format="%.2f"),
                "AI相关度": st.column_config.NumberColumn("AI相关度", format="%.2f"),
                "关键词相关度": st.column_config.NumberColumn("关键词相关度", format="%.2f"),
                "推荐评分": st.column_config.ProgressColumn(
                    "推荐评分",
                    format="%d",
                    min_value=min_possible_score,
                    max_value=max_possible_score
                ),
                "ID": st.column_config.TextColumn("ID", help="唯一标识符"),
                "上传日期": st.column_config.TextColumn("上传日期", help="该漫画的上传时间"),
                "本地目录": st.column_config.TextColumn("本地目录", help="匹配到的本地漫画目录")
            },
            column_order=display_columns,
            disabled=[col for col in table_df.columns if col != "选中"],
            hide_index=True,
            width='stretch',
            height=650,
            key=f"library-select-{current_selected_manga_id()}-{selected_page_index}-{global_sort_by}-{global_sort_order}",
        )
        apply_table_selection(edited_table, display_df)

        del table_df
        del display_df
        del sorted_df
        gc.collect()

        render_global_preference_charts(preference_chart_cache)
    else:
        with copy_button_slot.container():
            render_copy_page_button("", 0)
        st.info("没有可以显示的数据喔。")

with tab_llm:
    render_chat_interface(chat_context_df)

with tab_detail:
    st.subheader("漫画详情")
    selected_manga = get_selected_manga(filtered_df)
    if selected_manga is None:
        if current_selected_manga_id():
            st.warning("当前选中的漫画不在筛选结果中，请调整筛选条件或在库存列表重新选中。")
        else:
            st.info("先在库存列表里勾选一部漫画，详情和本地打开入口会显示在这里。")
    else:
        render_manga_detail(selected_manga)

with tab_history:
    st.subheader("历史记录")
    st.caption(f"缓存最近 {HISTORY_RECOMMENDATION_CACHE_SIZE} 次来源链接或本地打开记录，当前已保存 {len(history_entries)} 条。")

    col_refresh_history, col_clear_history = st.columns(2)
    with col_refresh_history:
        if st.button("刷新记录", width="stretch", key="history-refresh"):
            st.rerun()
    with col_clear_history:
        if st.button("清空记录", width="stretch", key="history-clear"):
            clear_history_entries()
            st.toast("已清空历史偏好记录", icon="✅")
            st.rerun()

    history_table = build_history_table(history_entries)
    if history_table.empty:
        st.info("暂时还没有保存的历史记录。")
    else:
        edited_history = st.data_editor(
            history_table,
            column_config={
                "删除": st.column_config.CheckboxColumn("删除", width="small"),
                "序号": st.column_config.NumberColumn("序号", format="%d", width="small"),
                "打开时间": "打开时间",
                "动作": "动作",
                "ID": "ID",
                "标题": "标题",
                "作者": "作者",
                "本地目录": "本地目录",
                "链接": st.column_config.LinkColumn("链接", display_text="打开"),
                "标签": "标签",
                "标题词": "标题词",
            },
            column_order=["删除", "序号", "打开时间", "ID", "标题", "作者", "标签", "标题词", "动作", "本地目录", "链接"],
            disabled=[col for col in history_table.columns if col != "删除"],
            hide_index=True,
            width="stretch",
            height=560,
            key=f"history-record-editor-{len(history_entries)}",
        )

        rows_to_delete = edited_history[edited_history["删除"].fillna(False)]
        selected_delete_count = len(rows_to_delete)
        if st.button(
            f"删除选中的 {selected_delete_count} 条记录",
            width="stretch",
            disabled=selected_delete_count == 0,
            key="history-delete-selected",
        ):
            indices_to_delete = set((rows_to_delete["序号"].astype(int) - 1).tolist())
            remaining_entries = [
                entry for index, entry in enumerate(history_entries) if index not in indices_to_delete
            ]
            save_history_entries(remaining_entries)
            st.toast(f"已删除 {selected_delete_count} 条历史记录", icon="✅")
            st.rerun()

    render_history_preference_charts(history_entries)

with tab_data_processing:
    render_data_processing_interface()
