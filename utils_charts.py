import altair as alt
import pandas as pd
import streamlit as st


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
