import requests
import json
import pandas as pd
import streamlit as st
from config import (
    LM_STUDIO_API_BASE, LM_STUDIO_MODEL,
    ONLINE_API_BASE, ONLINE_API_KEY, ONLINE_MODEL,
    SYSTEM_PROMPT
)

def get_ai_response_stream(user_query, context_df, api_mode="本地 (LM Studio)", temperature=0.7, max_tokens=4096):
    # 根据模式切换 API 源，支持流式返回
    if api_mode == "本地 (LM Studio)":
        url = f"{LM_STUDIO_API_BASE}/chat/completions"
        headers = {"Content-Type": "application/json"}
        model_name = LM_STUDIO_MODEL
    else:
        url = f"{ONLINE_API_BASE}/chat/completions"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {ONLINE_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)"
        }
        model_name = ONLINE_MODEL

    context_text = ""
    if not context_df.empty:
        context_text = "### 当前选中的参考库存数据：\n"
        for _, row in context_df.iterrows():
            link = row.get('链接', '无链接')
            context_text += f"- 标题: {row['标题']} | 作者: {row['作者']} | 标签: {row['标签']} | 评分: {row['推荐评分']} | 链接: {link}\n"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context_text}\n\n问题：{user_query}"}
        ],
        "temperature": temperature,  # 动态传入温度
        "max_tokens": max_tokens,    # 动态传入Token限制
        "stream": True  
    }

    try:
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data_json = json.loads(data_str)
                        if "choices" in data_json and len(data_json["choices"]) > 0:
                            delta = data_json["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        yield f"\n\n❌ API连接失败 ({api_mode}): {e}"

def render_chat_interface(chat_context_df):
    st.markdown("---")
    st.subheader("🤖 何不试试LLM呢？")

    col_mode, col_n, col_temp, col_tok = st.columns([1.5, 1.2, 1, 1])
    
    with col_mode:
        api_mode = st.radio("API 来源：", options=["本地 (LM Studio)", "线上 (Bltcy AI)"], horizontal=True)
    
    n_context = 0
    if chat_context_df is not None and not chat_context_df.empty:
        with col_n:
            max_n = min(len(chat_context_df), 500)
            n_context = st.number_input(
                f"随机注入条目数 (上限 {max_n})：", 
                min_value=0, max_value=max_n, value=min(10, max_n)
            )
    else:
        with col_n:
            # 数据为空时，占用位置并禁用输入框以保持UI整齐
            st.number_input("随机注入条目 (无数据)：", min_value=0, max_value=0, value=0, disabled=True)
        st.info("当前页面没有库存数据，AI 助手将仅能基于通用知识回答。")

    with col_temp:
        # 温度调节器 (0.0~2.0)
        temperature = st.number_input("温度 (Temp)：", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
        
    with col_tok:
        # Token长度调节器
        max_tokens = st.number_input("最大输出 (Tokens)：", min_value=256, max_value=32768, value=4096, step=512)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 折叠除最后两回合之外的所有记录
    if len(st.session_state.messages) > 2:
        with st.expander("📜 查看历史对话记录", expanded=False):
            for message in st.session_state.messages[:-2]:
                with st.chat_message(message["role"]):
                    content = message["content"]
                    if message["role"] == "assistant":
                        normalized = content.replace("Thinking Process:", "<think>")
                        if "<think>" in normalized and "</think>" in normalized:
                            parts = normalized.split("</think>")
                            answer_text = parts[1].strip()
                            st.markdown(f"*(已隐藏思考过程)*\n\n{answer_text}")
                        else:
                            st.markdown(content)
                    else:
                        st.markdown(content)

    # 显示最后一次交互
    for message in st.session_state.messages[-2:]:
        with st.chat_message(message["role"]):
            content = message["content"]
            if message["role"] == "assistant":
                normalized = content.replace("Thinking Process:", "<think>")
                if "<think>" in normalized and "</think>" in normalized:
                    parts = normalized.split("</think>")
                    think_text = parts[0].replace("<think>", "").strip()
                    answer_text = parts[1].strip()
                    with st.expander("LLM思考过程", expanded=False):
                        st.markdown(think_text)
                    st.markdown(answer_text)
                else:
                    st.markdown(content)
            else:
                st.markdown(content)

    if prompt := st.chat_input("问问LLM关于这些收藏的事..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if chat_context_df is not None and not chat_context_df.empty and n_context > 0:
            rag_context_df = chat_context_df.sample(n=int(n_context))
        else:
            rag_context_df = pd.DataFrame()

        with st.chat_message("assistant"):
            # 将前端收集到的 UI 参数传递给 API 接口
            stream_generator = get_ai_response_stream(
                user_query=prompt, 
                context_df=rag_context_df, 
                api_mode=api_mode,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            full_response = ""
            think_text = ""
            answer_text = ""
            think_placeholder = None
            answer_placeholder = st.empty()
            has_think_tag = False
            in_think_tag = False
            has_cleaned_prefix = False
            
            for chunk in stream_generator:
                full_response += chunk
                normalized_response = full_response.replace("Thinking Process:", "<think>")
                
                if "<think>" in normalized_response:
                    has_think_tag = True
                    if "</think>" in normalized_response:
                        in_think_tag = False
                        parts = normalized_response.split("</think>")
                        think_text = parts[0].replace("<think>", "").strip()
                        answer_text = parts[1].strip()
                    else:
                        in_think_tag = True
                        think_text = normalized_response.replace("<think>", "").strip()
                        answer_text = ""
                else:
                    has_think_tag = False
                    think_text = ""
                    answer_text = full_response

                if has_think_tag:
                    if not has_cleaned_prefix:
                        answer_placeholder.empty()
                        has_cleaned_prefix = True
                    if think_placeholder is None:
                        think_placeholder = st.expander("LLM思考过程", expanded=False).empty()
                    
                    if in_think_tag:
                        think_placeholder.markdown(think_text + "▌")
                    else:
                        think_placeholder.markdown(think_text) 
                        answer_placeholder.markdown(answer_text + "▌") 
                else:
                    answer_placeholder.markdown(answer_text + "▌")
                    
            if think_placeholder:
                think_placeholder.markdown(think_text)
            answer_placeholder.markdown(answer_text)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})
