import altair as alt
import pandas as pd
import streamlit as st
from collections import Counter


def _coerce_history_list(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str):
        stripped_value = value.strip()
        if not stripped_value:
            return []
        return [item.strip() for item in stripped_value.split(",") if item.strip()]

    return []


def _unique_history_items(items):
    seen_items = set()
    unique_items = []
    for item in items:
        if item in seen_items:
            continue
        seen_items.add(item)
        unique_items.append(item)
    return unique_items


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


def build_history_preference_chart_data(history_entries):
    tag_counter = Counter()
    artist_counter = Counter()
    title_word_counter = Counter()

    for entry in history_entries:
        if not isinstance(entry, dict):
            continue

        tag_counter.update(_unique_history_items(_coerce_history_list(entry.get("tags"))))
        title_word_counter.update(_unique_history_items(_coerce_history_list(entry.get("title_words"))))

        author = str(entry.get("author", "")).strip()
        if author:
            artist_counter[author] += 1

    return {
        "tags": {
            "title": "Top 15 历史偏好标签",
            "top_15": tag_counter.most_common(15),
            "top_150": tag_counter.most_common(150),
            "label_col": "标签",
            "value_col": "打开频次",
            "table_label_col": "历史偏好标签",
            "table_value_col": "打开频次",
            "expander_label": "🔍 查看 Top 150 历史偏好标签",
        },
        "artists": {
            "title": "Top 15 历史偏好作者",
            "top_15": artist_counter.most_common(15),
            "top_150": artist_counter.most_common(150),
            "label_col": "作者",
            "value_col": "打开频次",
            "table_label_col": "历史偏好作者",
            "table_value_col": "打开频次",
            "expander_label": "🔍 查看 Top 150 历史偏好作者",
        },
        "title_words": {
            "title": "Top 15 历史偏好标题词",
            "top_15": title_word_counter.most_common(15),
            "top_150": title_word_counter.most_common(150),
            "label_col": "词汇",
            "value_col": "打开频次",
            "table_label_col": "历史偏好标题词",
            "table_value_col": "打开频次",
            "expander_label": "🔍 查看 Top 150 历史偏好标题词",
        },
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


def render_history_preference_charts(history_entries):
    chart_data = build_history_preference_chart_data(history_entries)

    st.markdown("---")
    st.subheader("用户历史偏好数据")
    st.caption("数据来源：datacache/recommendation_history.json")

    chart_col1, chart_col2, chart_col3 = st.columns(3)

    with chart_col1:
        render_preference_chart_block(chart_data["tags"])

    with chart_col2:
        render_preference_chart_block(chart_data["artists"])

    with chart_col3:
        render_preference_chart_block(chart_data["title_words"])
