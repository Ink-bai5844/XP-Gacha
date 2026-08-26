import requests
import json
import os
from urllib.parse import urlsplit
import pandas as pd
import streamlit as st
import config
from server.modules.llm_settings import runtime_llm_connection


def _runtime_setting(name: str, default: str = "") -> str:
    return os.getenv(name, str(getattr(config, name, default))).strip()


def _text_delta(value) -> str:
    """Normalize the text shapes used by OpenAI-compatible providers."""

    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text"):
            text = value.get(key)
            if isinstance(text, str):
                return text
        return ""
    if isinstance(value, list):
        return "".join(_text_delta(item) for item in value)
    return ""


class _TaggedThinkingParser:
    """Split fallback <think> blocks without leaking split tags into the answer."""

    _OPEN_MARKERS = ("<think>", "Thinking Process:")
    _CLOSE_MARKER = "</think>"

    def __init__(self, expose_reasoning: bool) -> None:
        self._buffer = ""
        self._in_reasoning = False
        self._expose_reasoning = expose_reasoning

    @staticmethod
    def _prefix_tail_length(value: str, markers: tuple[str, ...]) -> int:
        lowered = value.casefold()
        longest = 0
        for marker in markers:
            marker_lower = marker.casefold()
            limit = min(len(value), len(marker) - 1)
            for length in range(limit, 0, -1):
                if lowered.endswith(marker_lower[:length]):
                    longest = max(longest, length)
                    break
        return longest

    @staticmethod
    def _first_marker(value: str, markers: tuple[str, ...]) -> tuple[int, str] | None:
        lowered = value.casefold()
        matches = [
            (index, marker)
            for marker in markers
            if (index := lowered.find(marker.casefold())) >= 0
        ]
        return min(matches, key=lambda match: match[0]) if matches else None

    def _event(self, content: str) -> dict | None:
        if not content or self._in_reasoning and not self._expose_reasoning:
            return None
        return {
            "type": "reasoning" if self._in_reasoning else "content",
            "content": content,
        }

    def feed(self, content: str) -> list[dict]:
        self._buffer += content
        events: list[dict] = []
        while self._buffer:
            markers = (self._CLOSE_MARKER,) if self._in_reasoning else self._OPEN_MARKERS
            match = self._first_marker(self._buffer, markers)
            if match is not None:
                index, marker = match
                event = self._event(self._buffer[:index])
                if event:
                    events.append(event)
                self._buffer = self._buffer[index + len(marker):]
                self._in_reasoning = not self._in_reasoning
                continue

            held = self._prefix_tail_length(self._buffer, markers)
            emitted = self._buffer[:-held] if held else self._buffer
            self._buffer = self._buffer[-held:] if held else ""
            event = self._event(emitted)
            if event:
                events.append(event)
            break
        return events

    def finish(self) -> list[dict]:
        event = self._event(self._buffer)
        self._buffer = ""
        return [event] if event else []


def _thinking_request_fields(api_base: str, model_name: str, enabled: bool) -> dict:
    """Use provider-specific switches only where the wire format is known."""

    try:
        hostname = (urlsplit(api_base).hostname or "").casefold()
    except ValueError:
        hostname = ""
    model = model_name.casefold()
    if hostname == "api.deepseek.com":
        options: dict = {"thinking": {"type": "enabled" if enabled else "disabled"}}
        if enabled:
            options["reasoning_effort"] = "high"
        return options
    if hostname == "dashscope.aliyuncs.com" or hostname.endswith(".dashscope.aliyuncs.com"):
        return {"enable_thinking": enabled}
    if enabled and hostname == "api.openai.com" and model.startswith(("o1", "o3", "o4", "gpt-5")):
        return {"reasoning_effort": "high"}
    return {}


def get_ai_response_events(
    user_query,
    context_df,
    api_mode="本地 (LM Studio)",
    temperature=0.7,
    max_tokens=4096,
    deep_thinking=False,
):
    """Yield normalized reasoning/content events from an OpenAI-compatible SSE stream."""

    api_base, api_key, model_name = runtime_llm_connection(api_mode)
    url = f"{api_base}/chat/completions"
    if api_mode == "本地 (LM Studio)":
        headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)"
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    if not api_base:
        yield {"type": "error", "content": f"{api_mode} URL 尚未配置，请先在助手页保存 API 连接。"}
        return

    context_text = ""
    if not context_df.empty:
        context_text = "### 当前选中的参考库存数据：\n"
        for _, row in context_df.iterrows():
            item_id = row.get('ID', '无ID')
            link = row.get('链接', '无链接')
            title = row.get('标题', '')
            translated_title = row.get('标题译文', '')
            author = row.get('作者', '')
            tags = row.get('标签', '')
            score = row.get('推荐评分', '')
            context_text += (
                f"- ID: {item_id} | 标题: {title} | 标题译文: {translated_title} | "
                f"作者: {author} | 标签: {tags} | 评分: {score} | 链接: {link}\n"
            )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": _runtime_setting("SYSTEM_PROMPT", config.SYSTEM_PROMPT)},
            {"role": "user", "content": f"{context_text}\n\n问题：{user_query}"}
        ],
        "temperature": temperature,  # 动态传入温度
        "max_tokens": max_tokens,    # 动态传入Token限制
        "stream": True,
    }
    payload.update(_thinking_request_fields(api_base, model_name, bool(deep_thinking)))

    try:
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=(15, 300))
        response.raise_for_status()
        parser = _TaggedThinkingParser(bool(deep_thinking))
        try:
            lines = response.iter_lines(chunk_size=1, decode_unicode=True)
        except TypeError:
            # Keep compatibility with simple response fakes and older requests adapters.
            lines = response.iter_lines()
        for line in lines:
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].lstrip()
            if data_str == "[DONE]":
                break
            try:
                data_json = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = data_json.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            delta = choices[0].get("delta", {})
            if not isinstance(delta, dict):
                continue

            reasoning = ""
            for key in ("reasoning_content", "reasoning", "thinking"):
                reasoning = _text_delta(delta.get(key))
                if reasoning:
                    break
            if deep_thinking and reasoning:
                yield {"type": "reasoning", "content": reasoning}

            content = _text_delta(delta.get("content"))
            if content:
                yield from parser.feed(content)
        yield from parser.finish()
    except Exception as e:
        yield {"type": "error", "content": f"API 连接失败（{api_mode}）：{e}"}


def get_ai_response_stream(
    user_query,
    context_df,
    api_mode="本地 (LM Studio)",
    temperature=0.7,
    max_tokens=4096,
    deep_thinking=False,
):
    """Backward-compatible text stream used by the legacy Streamlit view."""

    reasoning_open = False
    for event in get_ai_response_events(
        user_query,
        context_df,
        api_mode=api_mode,
        temperature=temperature,
        max_tokens=max_tokens,
        deep_thinking=deep_thinking,
    ):
        event_type = event.get("type")
        content = event.get("content", "")
        if event_type == "reasoning":
            if not reasoning_open:
                reasoning_open = True
                yield "<think>"
            yield content
        elif event_type == "content":
            if reasoning_open:
                reasoning_open = False
                yield "</think>"
            yield content
        elif event_type == "error":
            if reasoning_open:
                reasoning_open = False
                yield "</think>"
            yield f"\n\n{content}"
    if reasoning_open:
        yield "</think>"

def render_chat_interface(chat_context_df):
    st.subheader("🤖 何不试试LLM呢？")

    col_mode, col_n, col_temp, col_tok = st.columns([1.5, 1.2, 1, 1])
    
    with col_mode:
        api_mode = st.radio("API 来源：", options=["本地 (LM Studio)", "线上API"], horizontal=True)
    
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
                if not isinstance(chunk, str) or not chunk:
                    continue
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
