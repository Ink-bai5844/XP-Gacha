from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    B64_CACHE_DIR,
    CACHE_DIR,
    CLIP_MODEL_PATH,
    IMG_CACHE_DIR,
    IMG_VECTOR_FILE,
    LOCAL_MODEL_PATH,
    LM_STUDIO_API_BASE,
    LM_STUDIO_MODEL,
    ONLINE_IMG_DIR,
    VECTOR_FILE,
)


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
DEFAULT_TIMEOUT = 3600


def project_path(path_value: str | Path) -> Path:
    path = Path(str(path_value).strip())
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def display_path(path_value: str | Path) -> str:
    path = Path(str(path_value).strip())
    if path.is_absolute():
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)
    return str(path)


def count_files(path_value: str | Path, pattern: str = "*") -> int:
    path = project_path(path_value)
    if not path.exists():
        return 0
    return sum(1 for item in path.glob(pattern) if item.is_file())


@st.cache_data(show_spinner=False, ttl=600)
def cached_count_files(path_text: str, pattern: str, cache_token: int) -> int:
    return count_files(path_text, pattern)


def get_optional_count(path_value: str | Path, pattern: str = "*") -> str:
    if not st.session_state.get("data_processing_stats_loaded", False):
        return "未统计"

    cache_token = int(st.session_state.get("data_processing_stats_token", 0))
    return str(cached_count_files(display_path(path_value), pattern, cache_token))


def path_exists_text(path_value: str | Path) -> str:
    return "存在" if project_path(path_value).exists() else "不存在"


def result_key(key: str) -> str:
    return f"data-process-result-{key}"


def run_command(
    command: list[str],
    timeout: int = DEFAULT_TIMEOUT,
    live_output=None,
    extra_env: dict[str, str] | None = None,
    display_command: list[str] | None = None,
) -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update({key: str(value) for key, value in extra_env.items() if value is not None})

    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
    except OSError as exc:
        return {
            "command": command,
            "display_command": display_command or command,
            "returncode": 1,
            "stdout": "",
            "stderr": str(exc),
            "timed_out": False,
            "run_id": str(time.time_ns()),
        }

    output_queue: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        try:
            if process.stdout is None:
                return
            for line in process.stdout:
                output_queue.put(line)
        finally:
            output_queue.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    output_parts: list[str] = []
    timed_out = False
    reader_done = False
    started_at = time.monotonic()
    last_rendered_text = ""

    if live_output is not None:
        live_output.code("实时输出\n\n等待脚本输出...", language="text")

    while True:
        if timeout and time.monotonic() - started_at > int(timeout) and process.poll() is None:
            timed_out = True
            process.kill()
            output_parts.append(f"\n[timeout] 任务超过 {int(timeout)} 秒，已停止等待。\n")

        try:
            item = output_queue.get(timeout=0.1)
        except queue.Empty:
            item = ""

        if item is None:
            reader_done = True
        elif item:
            output_parts.append(item)

        current_text = "".join(output_parts).strip() or "等待脚本输出..."
        if live_output is not None and current_text != last_rendered_text:
            live_output.code(f"实时输出\n\n{current_text}", language="text")
            last_rendered_text = current_text

        if reader_done and process.poll() is not None:
            break
        if timed_out and process.poll() is not None:
            break

    returncode = process.wait()
    reader.join(timeout=1)

    return {
        "command": command,
        "display_command": display_command or command,
        "returncode": returncode,
        "stdout": "".join(output_parts).strip(),
        "stderr": "",
        "timed_out": timed_out,
        "run_id": str(time.time_ns()),
    }


def run_python_code(code: str, timeout: int = DEFAULT_TIMEOUT, live_output=None) -> dict:
    return run_command([PYTHON, "-u", "-c", code], timeout=timeout, live_output=live_output)


def module_call_code(module_name: str, assignments: dict[str, object], body: str) -> str:
    assignment_parts = []
    for name, value in assignments.items():
        if isinstance(value, Path):
            assignment_parts.append(f"mod.{name} = Path({repr(str(value))})")
        else:
            assignment_parts.append(f"mod.{name} = {repr(value)}")
    assignment_lines = "\n".join(assignment_parts)
    return (
        "import sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {repr(str(PROJECT_ROOT))})\n"
        f"import {module_name} as mod\n"
        f"{assignment_lines}\n"
        f"{body}\n"
    )


def render_result(key: str, empty_label: str | None = None) -> None:
    result = st.session_state.get(result_key(key))
    if not result:
        if empty_label:
            st.info(empty_label)
        return

    display_command = result.get("display_command") or result["command"]
    command_text = " ".join(str(part) for part in display_command)
    with st.expander("脚本输出", expanded=True):
        st.code(command_text, language="powershell")
        if result["timed_out"]:
            st.warning("任务超时，已停止等待。")
        elif result["returncode"] == 0:
            st.success("任务完成。")
        else:
            st.error(f"任务退出码：{result['returncode']}")

        output_parts = []
        if result["stdout"]:
            output_parts.append(result["stdout"])
        if result["stderr"]:
            output_parts.append("[stderr]\n" + result["stderr"])
        output_text = "\n\n".join(output_parts) or "脚本没有输出。"
        output_key = f"{key}-script-output-{result.get('run_id', 'legacy')}"
        st.text_area("输出内容", output_text, height=320, key=output_key)


def save_inline_result(key: str, command_label: str, stdout: str, returncode: int = 0, stderr: str = "") -> None:
    st.session_state[result_key(key)] = {
        "command": [command_label],
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": False,
        "run_id": str(time.time_ns()),
    }


def submit_subprocess(
    button_label: str,
    key: str,
    command: list[str],
    timeout: int,
    disabled: bool = False,
    require_confirm: bool = True,
    extra_env: dict[str, str] | None = None,
    display_command: list[str] | None = None,
) -> None:
    if st.form_submit_button(button_label, width="stretch", disabled=disabled):
        if not require_confirm:
            st.warning("请先勾选确认项。")
            return
        with st.spinner("正在执行..."):
            live_output = st.empty()
            st.session_state[result_key(key)] = run_command(
                command,
                timeout=timeout,
                live_output=live_output,
                extra_env=extra_env,
                display_command=display_command,
            )
            live_output.empty()


def submit_python_code(
    button_label: str,
    key: str,
    code: str,
    timeout: int,
    disabled: bool = False,
    require_confirm: bool = True,
) -> None:
    if st.form_submit_button(button_label, width="stretch", disabled=disabled):
        if not require_confirm:
            st.warning("请先勾选确认项。")
            return
        with st.spinner("正在执行..."):
            live_output = st.empty()
            st.session_state[result_key(key)] = run_python_code(
                code,
                timeout=timeout,
                live_output=live_output,
            )
            live_output.empty()


def render_overview() -> None:
    col_refresh, col_hint = st.columns([1, 4])
    with col_refresh:
        if st.button("刷新统计", width="stretch", key="data-processing-refresh-stats"):
            st.session_state["data_processing_stats_loaded"] = True
            st.session_state["data_processing_stats_token"] = (
                int(st.session_state.get("data_processing_stats_token", 0)) + 1
            )
    with col_hint:
        st.caption("大目录文件数默认不自动扫描")

    status_cols = st.columns(5)
    status_cols[0].metric("CSV", get_optional_count("data/gallery_info", "*.csv"))
    status_cols[1].metric("线上封面", get_optional_count(ONLINE_IMG_DIR))
    status_cols[2].metric("本地缩略图", get_optional_count(IMG_CACHE_DIR))
    status_cols[3].metric("Base64", get_optional_count(B64_CACHE_DIR, "*.txt"))
    status_cols[4].metric("历史缓存", get_optional_count(CACHE_DIR))

    cache_rows = [
        ("预处理 DataFrame", "datacache/preprocessed_df.pkl"),
        ("预处理 Hash", "datacache/data.hash"),
        ("文本向量", VECTOR_FILE),
        ("封面向量", IMG_VECTOR_FILE),
    ]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "项目": name,
                    "路径": path,
                    "状态": path_exists_text(path),
                    "大小KB": round(project_path(path).stat().st_size / 1024, 1)
                    if project_path(path).exists()
                    else 0,
                }
                for name, path in cache_rows
            ]
        ),
        hide_index=True,
        width="stretch",
        height=180,
    )


def render_csv_tools() -> None:
    with st.expander("补全文件名列", expanded=False):
        with st.form("process-addname"):
            csv_file = st.text_input("CSV 文件", "data/gallery_info_origin/JM_info_yuri.csv")
            txt_file = st.text_input("本地链接 HTML/TXT", "data/local_data/NH_all.txt")
            output_file = st.text_input("输出 CSV", "data/gallery_info/JM_info_yuri_full.csv")
            timeout = st.number_input("超时秒数", 10, 7200, 600, 10, key="addname-timeout")
            code = module_call_code(
                "data_processing.addname",
                {},
                (
                    "mod.process_gallery_data("
                    f"{repr(csv_file)}, {repr(txt_file)}, {repr(output_file)}"
                    ")"
                ),
            )
            submit_python_code("执行补全", "addname", code, timeout)
        render_result("addname", "脚本输出会显示在这里。")

    with st.expander("NH 链接补 ID", expanded=False):
        with st.form("process-add-id"):
            csv_dir = st.text_input("CSV 目录", "data/gallery_info_origin", key="add-id-dir")
            prefix = st.text_input("ID 前缀", "NH", key="add-id-prefix")
            timeout = st.number_input("超时秒数", 10, 7200, 600, 10, key="add-id-timeout")
            code = module_call_code(
                "tools.add_id",
                {"CSV_DIR": Path(csv_dir), "ID_PREFIX": prefix},
                "mod.main()",
            )
            submit_python_code("补齐 ID", "add-id", code, timeout)
        render_result("add-id", "脚本输出会显示在这里。")

    with st.expander("迁移语言标签", expanded=False):
        with st.form("process-add-lang"):
            csv_path = st.text_input("CSV 文件", "data/gallery_info_origin/JM_info_gender_bender.csv")
            language_tags_raw = st.text_input("语言标签", "中文, 英文, 日文")
            timeout = st.number_input("超时秒数", 10, 7200, 600, 10, key="add-lang-timeout")
            language_tags = {
                item.strip()
                for item in language_tags_raw.replace("，", ",").split(",")
                if item.strip()
            }
            code = module_call_code(
                "tools.add_lang",
                {"CSV_PATH": Path(csv_path), "LANG_TAGS": language_tags},
                "mod.move_language_tags()",
            )
            submit_python_code("执行迁移", "add-lang", code, timeout)
        render_result("add-lang", "脚本输出会显示在这里。")

    with st.expander("清洗上传日期", expanded=False):
        with st.form("process-clean-date"):
            csv_path = st.text_input("CSV 文件", "data/gallery_info/JM_info_gender_bender_full.csv")
            timeout = st.number_input("超时秒数", 10, 7200, 600, 10, key="clean-date-timeout")
            code = module_call_code(
                "tools.clean",
                {"CSV_PATH": Path(csv_path)},
                "mod.clean_upload_dates()",
            )
            submit_python_code("清洗日期", "clean-date", code, timeout)
        render_result("clean-date", "脚本输出会显示在这里。")

    with st.expander("标题词频统计", expanded=False):
        with st.form("process-title-words"):
            input_csv = st.text_input("输入 CSV", "data/gallery_info/gallery_info_gender_bender_full.csv")
            output_csv = st.text_input("输出 CSV", "data_processing/title_words_frequency.csv")
            stop_words = st.text_input("停用词词典", "dictionaries/TITLE_STOP_WORDS.txt")
            semantic_map = st.text_input("标题语义映射", "dictionaries/TITLE_SEMANTIC_MAP.json")
            timeout = st.number_input("超时秒数", 10, 7200, 1200, 10, key="title-words-timeout")
            code = module_call_code(
                "data_processing.title_cut_set",
                {
                    "INPUT_CSV": project_path(input_csv),
                    "OUTPUT_CSV": project_path(output_csv),
                    "TITLE_STOP_WORDS_PATH": project_path(stop_words),
                    "TITLE_SEMANTIC_MAP_PATH": project_path(semantic_map),
                },
                "mod.process_title_words()",
            )
            submit_python_code("生成词频", "title-words", code, timeout)
        render_result("title-words", "脚本输出会显示在这里。")

    with st.expander("聚合未映射标签", expanded=False):
        with st.form("process-tag-set"):
            csv_dir = st.text_input("CSV 目录", "data/gallery_info")
            semantic_map = st.text_input("语义映射 JSON", "dictionaries/SEMANTIC_MAP.json")
            output_file = st.text_input("输出 TXT", "data_processing/aggregated_tags.txt")
            timeout = st.number_input("超时秒数", 10, 7200, 600, 10, key="tag-set-timeout")
            code = (
                "import sys\n"
                f"sys.path.insert(0, {repr(str(PROJECT_ROOT))})\n"
                "from data_processing import tag_set\n"
                f"semantic_map = tag_set.load_semantic_map({repr(semantic_map)})\n"
                f"tags = tag_set.get_aggregated_tags({repr(csv_dir)}, semantic_map)\n"
                f"exported = tag_set.export_tags_to_document(tags, {repr(output_file)})\n"
                "print(f'共找到 {len(tags)} 个不重复的新标签，已导出到：{exported}')\n"
                "for tag in sorted(tags)[:50]:\n"
                "    print(tag)\n"
            )
            submit_python_code("聚合标签", "tag-set", code, timeout)
        render_result("tag-set", "脚本输出会显示在这里。")

    with st.expander("语义映射补原名", expanded=False):
        with st.form("process-map-add-name"):
            input_file = st.text_input("输入 JSON", "data_processing/111.json")
            output_file = st.text_input("输出 JSON", "data_processing/222.json")
            timeout = st.number_input("超时秒数", 10, 7200, 600, 10, key="map-add-name-timeout")
            code = module_call_code(
                "data_processing.map_add_name",
                {"INPUT_FILE": input_file, "OUTPUT_FILE": output_file},
                "mod.transform_semantic_map()",
            )
            submit_python_code("转换映射", "map-add-name", code, timeout)
        render_result("map-add-name", "脚本输出会显示在这里。")


def render_database_tools() -> None:
    with st.expander("增量同步 CSV 到 MySQL", expanded=True):
        with st.form("process-db-sync"):
            csv_dir = st.text_input("CSV 目录", "data/gallery_info", key="db-sync-dir")
            timeout = st.number_input("超时秒数", 10, 14400, 1800, 10, key="db-sync-timeout")
            code = module_call_code(
                "data_processing.add_csv_to_mysql",
                {"CSV_DIR": csv_dir},
                "mod.sync_csv_to_db()",
            )
            submit_python_code("增量同步", "db-sync", code, timeout)
        render_result("db-sync", "脚本输出会显示在这里。")

    with st.expander("覆盖重建 MySQL 表", expanded=False):
        with st.form("process-db-rebuild"):
            csv_dir = st.text_input("CSV 目录", "data/gallery_info", key="db-rebuild-dir")
            confirm = st.checkbox("确认覆盖 gallery_info 表", value=False)
            timeout = st.number_input("超时秒数", 10, 14400, 1800, 10, key="db-rebuild-timeout")
            code = module_call_code(
                "data_processing.all_csv_to_mysql",
                {"CSV_DIR": csv_dir},
                "mod.migrate_data()",
            )
            submit_python_code("覆盖重建", "db-rebuild", code, timeout, require_confirm=confirm)
        render_result("db-rebuild", "脚本输出会显示在这里。")

    with st.expander("优化 MySQL 表结构与全文索引", expanded=False):
        with st.form("process-db-optimize"):
            confirm = st.checkbox("确认执行 ALTER TABLE 与 FULLTEXT 索引维护", value=False)
            timeout = st.number_input("超时秒数", 10, 14400, 3600, 10, key="db-optimize-timeout")
            code = module_call_code(
                "data_processing.optimize_mysql_schema",
                {},
                "mod.optimize_gallery_schema()",
            )
            submit_python_code("执行优化", "db-optimize", code, timeout, require_confirm=confirm)
        render_result("db-optimize", "脚本输出会显示在这里。")


def render_title_translation_tools() -> None:
    with st.expander("标题 AI 翻译", expanded=True):
        with st.form("process-title-translate"):
            lm_studio = st.checkbox(
                "LM Studio 本地单线程模式",
                value=False,
                help="默认读取 config.py 的 LM_STUDIO_API_BASE / LM_STUDIO_MODEL，强制单线程请求，API Key 可为空。",
            )
            api_url = st.text_input(
                "调用 URL / Base URL",
                LM_STUDIO_API_BASE
                if lm_studio
                else os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"),
                help="可填完整 /chat/completions 地址，也可填兼容 OpenAI 的 Base URL。",
                disabled=lm_studio,
            )
            api_key = st.text_input(
                "API Key",
                value="",
                type="password",
                help="为空时读取当前环境变量 OPENAI_API_KEY。",
            )
            model = st.text_input(
                "模型名",
                LM_STUDIO_MODEL if lm_studio else os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                disabled=lm_studio,
            )
            jsonl_output = st.text_input(
                "成功 JSONL 输出",
                "data_processing/title_translation_results.jsonl",
                help="每个成功批次会追加保存一行，包含输入标题和 LLM 返回的 JSON。",
            )
            failed_jsonl_output = st.text_input(
                "失败 JSONL 输出",
                "data_processing/title_translation_failed_results.jsonl",
                help="每个失败批次会追加保存一行，包含输入标题和错误原因。",
            )
            jsonl_only = st.checkbox(
                "仅写 JSONL，不回写数据库",
                value=False,
                help="适合先审查翻译质量；数据库中已有标题译文的条目仍会被跳过。",
            )
            batch_size = st.number_input("每组标题数量", 1, 200, 20, 1, key="title-translate-batch")
            concurrency = st.number_input(
                "并发组数",
                1,
                32,
                1 if lm_studio else 3,
                1,
                disabled=lm_studio,
                key="title-translate-concurrency",
            )
            start_index = st.number_input("起始序号", 1, 100000000, 1, 1, key="title-translate-start")
            end_index = st.number_input(
                "结束序号（0 表示不限制）",
                0,
                100000000,
                0,
                1,
                key="title-translate-end",
            )
            request_timeout = st.number_input(
                "单次请求超时秒数",
                10,
                3600,
                120,
                10,
                key="title-translate-request-timeout",
            )
            max_retries = st.number_input("失败重试次数", 0, 10, 2, 1, key="title-translate-retries")
            temperature = st.number_input(
                "Temperature",
                0.0,
                2.0,
                0.2,
                0.1,
                key="title-translate-temperature",
            )
            timeout = st.number_input(
                "整批任务超时秒数",
                10,
                86400,
                7200,
                60,
                key="title-translate-timeout",
            )
            confirm_label = (
                "确认调用 LLM 并只写入 JSONL"
                if jsonl_only
                else "确认调用 LLM 并写入 gallery_info.标题译文"
            )
            confirm = st.checkbox(confirm_label, value=False)

            command = [
                PYTHON,
                str(PROJECT_ROOT / "data_processing" / "translate_titles.py"),
                "--jsonl-output",
                jsonl_output,
                "--failed-jsonl-output",
                failed_jsonl_output,
                "--batch-size",
                str(int(batch_size)),
                "--concurrency",
                str(int(concurrency)),
                "--start-index",
                str(int(start_index)),
                "--request-timeout",
                str(int(request_timeout)),
                "--max-retries",
                str(int(max_retries)),
                "--temperature",
                str(float(temperature)),
            ]
            if not lm_studio:
                command.extend(["--api-url", api_url, "--model", model])
            if int(end_index) > 0:
                command.extend(["--end-index", str(int(end_index))])
            if jsonl_only:
                command.append("--jsonl-only")
            if lm_studio:
                command.append("--lm-studio")

            extra_env = {"OPENAI_API_KEY": api_key} if api_key else None
            submit_subprocess(
                "开始翻译标题",
                "title-translate",
                command,
                timeout,
                require_confirm=confirm,
                extra_env=extra_env,
            )
        render_result("title-translate", "脚本输出会显示在这里。")


def render_cache_tools() -> None:
    with st.expander("Base64 预编码", expanded=True):
        with st.form("process-b64"):
            source_options = {
                "线上封面": ONLINE_IMG_DIR,
                "本地缩略图": IMG_CACHE_DIR,
            }
            selected_sources = st.multiselect(
                "来源目录",
                options=list(source_options),
                default=list(source_options),
            )
            b64_cache_dir = st.text_input("主缓存目录", B64_CACHE_DIR)
            b64_tmp_dir = st.text_input("增量输出目录", "b64_tmp")
            timeout = st.number_input("超时秒数", 10, 14400, 1800, 10, key="b64-timeout")
            source_dirs = [source_options[name] for name in selected_sources]
            body_lines = [
                "import os",
                "os.makedirs(mod.B64_CACHE_DIR, exist_ok=True)",
                "os.makedirs(mod.B64_TMP_DIR, exist_ok=True)",
            ]
            if source_dirs:
                body_lines.extend(f"mod.process_directory({repr(directory)})" for directory in source_dirs)
            else:
                body_lines.append("print('未选择来源目录')")
            body = "\n".join(body_lines)
            code = module_call_code(
                "data_processing.b64_pre_encode",
                {"B64_CACHE_DIR": b64_cache_dir, "B64_TMP_DIR": b64_tmp_dir},
                body,
            )
            submit_python_code("生成 Base64", "b64", code, timeout)
        render_result("b64", "脚本输出会显示在这里。")

    with st.expander("文本语义向量", expanded=False):
        with st.form("process-text-vector"):
            model_path = st.text_input("模型目录", LOCAL_MODEL_PATH)
            vector_file = st.text_input("输出向量文件", VECTOR_FILE)
            batch_size = st.number_input("Batch Size", 1, 256, 16, 1, key="text-vector-batch")
            max_text_length = st.number_input("文本截断长度", 0, 5000, 800, 50)
            sql_query = st.text_area("SQL", "SELECT * FROM gallery_info WHERE ID != ''", height=80)
            timeout = st.number_input("超时秒数", 10, 86400, 7200, 60, key="text-vector-timeout")
            command = [
                PYTHON,
                str(PROJECT_ROOT / "data_processing" / "build_vector_db.py"),
                "--model-path",
                model_path,
                "--vector-file",
                vector_file,
                "--batch-size",
                str(int(batch_size)),
                "--max-text-length",
                str(int(max_text_length)),
                "--sql-query",
                sql_query,
            ]
            submit_subprocess("构建文本向量", "text-vector", command, timeout)
        render_result("text-vector", "脚本输出会显示在这里。")

    with st.expander("封面 CLIP 向量", expanded=False):
        action = st.radio("操作", ["构建/刷新", "统计"], horizontal=True, key="clip-action")
        with st.form("process-clip-vector"):
            model_path = st.text_input("CLIP 模型目录", CLIP_MODEL_PATH)
            index_path = st.text_input("索引文件", IMG_VECTOR_FILE)
            image_dirs_raw = st.text_input(
                "图片目录",
                f"{ONLINE_IMG_DIR}, {IMG_CACHE_DIR}",
                help="多个目录用逗号分隔",
            )
            batch_size = st.number_input("Batch Size", 1, 512, 64, 1, key="clip-batch")
            device = st.selectbox("设备", ["auto", "cpu", "cuda"], index=0)
            rebuild = st.checkbox("全量重建", value=False)
            timeout = st.number_input("超时秒数", 10, 86400, 7200, 60, key="clip-timeout")
            image_dirs = [
                item.strip()
                for item in image_dirs_raw.replace("，", ",").split(",")
                if item.strip()
            ]
            command = [
                PYTHON,
                str(PROJECT_ROOT / "data_processing" / "img_to_vector.py"),
                "build" if action == "构建/刷新" else "stats",
                "--model",
                model_path,
                "--index-path",
                index_path,
                "--batch-size",
                str(int(batch_size)),
                "--device",
                device,
            ]
            for image_dir in image_dirs:
                command.extend(["--image-dir", image_dir])
            if action == "构建/刷新" and rebuild:
                command.append("--rebuild")
            submit_subprocess("执行 CLIP 操作", "clip-vector", command, timeout)
        render_result("clip-vector", "脚本输出会显示在这里。")

    with st.expander("缓存清理", expanded=False):
        cache_targets = {
            "预处理 DataFrame": "datacache/preprocessed_df.pkl",
            "预处理 Hash": "datacache/data.hash",
            "文本向量": VECTOR_FILE,
            "封面向量": IMG_VECTOR_FILE,
        }
        selected_targets = st.multiselect("清理对象", list(cache_targets), default=[])
        confirm = st.checkbox("确认删除选中的缓存文件", value=False)
        if st.button("删除缓存", width="stretch", disabled=not confirm or not selected_targets):
            deleted = []
            for name in selected_targets:
                path = project_path(cache_targets[name])
                if path.exists() and path.is_file():
                    path.unlink()
                    deleted.append(str(path))
            skipped_count = len(selected_targets) - len(deleted)
            save_inline_result(
                "cache-delete",
                "删除缓存",
                "\n".join(
                    [
                        f"已删除 {len(deleted)} 个缓存文件，跳过 {skipped_count} 个不存在的文件。",
                        *deleted,
                    ]
                ),
            )
        render_result("cache-delete", "操作输出会显示在这里。")


def render_maintenance_tools() -> None:
    with st.expander("图片/缓存 ID 前缀修正", expanded=True):
        with st.form("process-prefix-rename"):
            target_kind = st.selectbox("对象", ["Base64 缓存 TXT", "本地缩略图"])
            target_dir = st.text_input(
                "目录",
                B64_CACHE_DIR if target_kind == "Base64 缓存 TXT" else IMG_CACHE_DIR,
            )
            prefix = st.text_input("前缀", "NH")
            timeout = st.number_input("超时秒数", 10, 7200, 600, 10, key="prefix-timeout")
            module_name = "tools.b64id" if target_kind == "Base64 缓存 TXT" else "tools.imgid"
            function_name = "rename_txt_files" if target_kind == "Base64 缓存 TXT" else "rename_images"
            code = module_call_code(
                module_name,
                {"TARGET_DIR": Path(target_dir), "PREFIX": prefix},
                f"mod.{function_name}()",
            )
            submit_python_code("执行重命名", "prefix-rename", code, timeout)
        render_result("prefix-rename", "脚本输出会显示在这里。")

    with st.expander("合并 Base64 增量缓存", expanded=False):
        tmp_dir = st.text_input("增量目录", "b64_tmp", key="merge-b64-tmp")
        cache_dir = st.text_input("主缓存目录", B64_CACHE_DIR, key="merge-b64-cache")
        overwrite = st.checkbox("覆盖同名文件", value=False)
        confirm = st.checkbox("确认合并", value=False)
        if st.button("合并缓存", width="stretch", disabled=not confirm):
            source_dir = project_path(tmp_dir)
            target_dir = project_path(cache_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            moved = 0
            skipped = 0
            if source_dir.exists():
                for file_path in source_dir.glob("*.txt"):
                    target_path = target_dir / file_path.name
                    if target_path.exists() and not overwrite:
                        skipped += 1
                        continue
                    shutil.move(str(file_path), str(target_path))
                    moved += 1
            save_inline_result(
                "merge-b64",
                "合并 Base64 增量缓存",
                f"已合并 {moved} 个文件，跳过 {skipped} 个文件。\n来源目录：{source_dir}\n目标目录：{target_dir}",
            )
        render_result("merge-b64", "操作输出会显示在这里。")

    with st.expander("从数据库中提取标题译文至 gallery_info CSV", expanded=False):
        with st.form("maintenance-export-title-translations"):
            csv_dir = st.text_input(
                "CSV 目录",
                "data/gallery_info",
                key="export-title-translations-csv-dir",
            )
            pattern = st.text_input(
                "文件匹配规则",
                "*_full.csv",
                help="只匹配当前目录内的文件名，不支持子目录或路径分隔符。",
                key="export-title-translations-pattern",
            )
            dry_run = st.checkbox(
                "仅预览，不写入 CSV",
                value=True,
                key="export-title-translations-dry-run",
            )
            timeout = st.number_input(
                "超时秒数",
                10,
                7200,
                600,
                10,
                key="export-title-translations-timeout",
            )
            confirm = st.checkbox(
                "确认批量更新匹配的 CSV 文件",
                value=False,
                disabled=dry_run,
                key="export-title-translations-confirm",
            )
            command = [
                PYTHON,
                str(PROJECT_ROOT / "tools" / "export_title_translations_to_csv.py"),
                "--csv-dir",
                csv_dir,
                "--pattern",
                pattern,
            ]
            if dry_run:
                command.append("--dry-run")
            submit_subprocess(
                "预览译文回填" if dry_run else "提取并写入标题译文",
                "export-title-translations",
                command,
                timeout,
                disabled=not csv_dir.strip() or not pattern.strip(),
                require_confirm=dry_run or confirm,
            )
        render_result("export-title-translations", "脚本输出会显示在这里。")

    with st.expander("清理标题翻译 JSONL failed 条目", expanded=False):
        with st.form("maintenance-clean-title-jsonl"):
            jsonl_path = st.text_input(
                "JSONL 文件",
                "data_processing/title_translation_results.jsonl",
                key="clean-title-jsonl-path",
            )
            keep_backup = st.checkbox("生成 .bak 备份", value=True, key="clean-title-jsonl-backup")
            timeout = st.number_input("超时秒数", 10, 7200, 600, 10, key="clean-title-jsonl-timeout")
            confirm = st.checkbox("确认清理 failed 条目", value=False, key="clean-title-jsonl-confirm")
            command = [
                PYTHON,
                str(PROJECT_ROOT / "tools" / "clean_failed_title_translation_jsonl.py"),
                "--jsonl-path",
                jsonl_path,
            ]
            if not keep_backup:
                command.append("--no-backup")
            submit_subprocess(
                "清理 failed 条目",
                "clean-title-jsonl",
                command,
                timeout,
                require_confirm=confirm,
            )
        render_result("clean-title-jsonl", "脚本输出会显示在这里。")

    with st.expander("按 ID 删除数据库条目", expanded=False):
        with st.form("maintenance-delete-gallery-rows"):
            ids_text = st.text_area(
                "ID 列表",
                "",
                height=100,
                help="多个 ID 可用空格、换行、英文逗号或中文逗号分隔。",
                key="delete-gallery-ids",
            )
            id_file = st.text_input(
                "ID 文件",
                "",
                help="可选：从文本文件读取 ID，路径相对项目根目录。",
                key="delete-gallery-id-file",
            )
            timeout = st.number_input("超时秒数", 10, 7200, 600, 10, key="delete-gallery-timeout")
            confirm_delete = st.checkbox(
                "确认执行 DELETE 删除整条数据库记录",
                value=False,
                key="delete-gallery-confirm",
            )
            command = [
                PYTHON,
                str(PROJECT_ROOT / "tools" / "delete_gallery_rows_by_id.py"),
            ]
            if ids_text.strip():
                command.append(ids_text)
            if id_file.strip():
                command.extend(["--id-file", id_file.strip()])
            if confirm_delete:
                command.append("--confirm")
            has_delete_input = bool(ids_text.strip() or id_file.strip())
            submit_subprocess(
                "预览/删除数据库条目",
                "delete-gallery-rows",
                command,
                timeout,
                disabled=not has_delete_input,
                require_confirm=True,
            )
        render_result("delete-gallery-rows", "脚本输出会显示在这里。")

    with st.expander("按 error.json 清空标题译文", expanded=False):
        with st.form("maintenance-clear-title-translation"):
            input_file = st.text_input(
                "错误 JSON 文件",
                "tools/error.json",
                help='会识别文件中所有 `"id": "NH..."` 文本段。',
                key="clear-title-translation-input",
            )
            preview_limit = st.number_input(
                "预览条数",
                1,
                1000,
                30,
                1,
                key="clear-title-translation-preview-limit",
            )
            chunk_size = st.number_input(
                "每批 ID 数量",
                1,
                5000,
                500,
                50,
                key="clear-title-translation-chunk-size",
            )
            empty_string = st.checkbox(
                "清空为空字符串（默认清空为 NULL）",
                value=False,
                key="clear-title-translation-empty-string",
            )
            timeout = st.number_input("超时秒数", 10, 7200, 600, 10, key="clear-title-translation-timeout")
            confirm_clear = st.checkbox(
                "确认清空这些 ID 的标题译文",
                value=False,
                key="clear-title-translation-confirm",
            )
            command = [
                PYTHON,
                str(PROJECT_ROOT / "tools" / "clear_title_translation_by_error_ids.py"),
                "--input",
                input_file,
                "--preview-limit",
                str(int(preview_limit)),
                "--chunk-size",
                str(int(chunk_size)),
            ]
            if empty_string:
                command.append("--empty-string")
            if confirm_clear:
                command.append("--confirm")
            submit_subprocess(
                "预览/清空标题译文",
                "clear-title-translation",
                command,
                timeout,
                disabled=not input_file.strip(),
                require_confirm=True,
            )
        render_result("clear-title-translation", "脚本输出会显示在这里。")


def render_collection_tools() -> None:
    mode = st.selectbox(
        "流程",
        [
            "NH 在线完整采集",
            "JM 在线完整采集",
            "NH 本地链接完整采集",
            "NH 本地链接抓取分册图片",
        ],
        key="collection-script",
    )
    st.caption("首轮结束后只重试上一轮未完成项；最多轮数为 0 时会持续到全部成功，或由用户终止进程。")

    if mode == "NH 在线完整采集":
        with st.form("process-nh-online"):
            base_url = st.text_input("站点 Base URL", "https://nhentai.net")
            start_url = st.text_input(
                "抓取起始网址",
                "https://nhentai.net/language/chinese/?sort=date",
                help="作为第 1 页；后续页会自动追加或替换 page 参数。",
            )
            max_pages = st.number_input("抓到多少页截止", 1, 100000, 1, 1)
            output_csv = st.text_input(
                "原始信息 CSV",
                "data/gallery_info_origin/NH_info_chinese.csv",
            )
            image_dir = st.text_input("缩略图保存目录", "onlineimgtmp")
            workers = st.number_input("并发线程数", 1, 64, 10, 1, key="nh-online-workers")
            request_attempts = st.number_input("单次请求尝试次数", 1, 20, 3, 1, key="nh-online-attempts")
            max_rounds = st.number_input(
                "最多轮数（含首轮，0 表示直至成功）",
                0,
                100000,
                0,
                1,
                key="nh-online-rounds",
            )
            retry_backoff = st.number_input("重试退避基数秒数", 0.0, 3600.0, 2.0, 0.5, key="nh-online-backoff")
            request_timeout = st.number_input("单次 HTTP 超时秒数", 1.0, 600.0, 30.0, 1.0, key="nh-online-http-timeout")
            interval = st.number_input("成功项目间隔秒数", 0.0, 60.0, 0.0, 0.5, key="nh-online-interval")
            proxy = st.text_input(
                "HTTP(S) 代理（可留空）",
                "",
                help="留空使用 ONLINE_COVER_PROXY 环境配置；填写后仅覆盖本次任务。",
                key="nh-online-proxy",
            )
            no_resume = st.checkbox(
                "忽略已有断点，从新一轮开始",
                value=False,
                help="启用后不恢复上次未完成状态。",
                key="nh-online-no-resume",
            )
            state_file = st.text_input(
                "断点状态 JSONL（可留空）",
                "",
                help="留空时按模式、网址和输出路径在 logs/collection 中生成唯一文件。",
            )
            error_log = st.text_input(
                "失败记录 JSONL（可留空）",
                "",
                help="留空时与断点参数一起生成唯一文件。",
            )
            confirm = st.checkbox("确认执行完整采集", value=False)
            command = [
                PYTHON,
                "-m",
                "data_get.collector",
                "nh-online",
                "--base-url",
                base_url,
                "--start-url",
                start_url,
                "--max-pages",
                str(int(max_pages)),
                "--output-csv",
                output_csv,
                "--image-dir",
                image_dir,
                "--workers",
                str(int(workers)),
                "--request-attempts",
                str(int(request_attempts)),
                "--max-rounds",
                str(int(max_rounds)),
                "--retry-backoff",
                str(float(retry_backoff)),
                "--timeout",
                str(float(request_timeout)),
                "--interval",
                str(float(interval)),
                "--state-file",
                state_file,
                "--error-log",
                error_log,
            ]
            if proxy.strip():
                command.extend(["--proxy", proxy.strip()])
            if no_resume:
                command.append("--no-resume")
            submit_subprocess("开始完整采集 NH", "collection-nh-online", command, 0, require_confirm=confirm)
        render_result("collection-nh-online", "脚本输出会显示在这里。")
        return

    if mode == "JM 在线完整采集":
        with st.form("process-jm-online"):
            base_url = st.text_input("站点 Base URL", "https://18comic.vip")
            start_url = st.text_input(
                "抓取起始网址",
                "https://18comic.vip/search/photos?search_query=%E7%99%BE%E5%90%88",
                help="作为第 1 页；后续页会自动追加或替换 page 参数。",
            )
            max_pages = st.number_input("抓到多少页截止", 1, 100000, 80, 1)
            output_csv = st.text_input(
                "原始信息 CSV",
                "data/gallery_info_origin/JM_info_yuri.csv",
            )
            image_dir = st.text_input("封面保存目录", "onlineimgtmp")
            workers = st.number_input("并发线程数", 1, 64, 5, 1, key="jm-online-workers")
            request_attempts = st.number_input("单次请求尝试次数", 1, 20, 3, 1, key="jm-online-attempts")
            max_rounds = st.number_input(
                "最多轮数（含首轮，0 表示直至成功）",
                0,
                100000,
                0,
                1,
                key="jm-online-rounds",
            )
            retry_backoff = st.number_input("重试退避基数秒数", 0.0, 3600.0, 2.0, 0.5, key="jm-online-backoff")
            request_timeout = st.number_input("单次 HTTP 超时秒数", 1.0, 600.0, 30.0, 1.0, key="jm-online-http-timeout")
            interval = st.number_input("成功项目间隔秒数", 0.0, 60.0, 0.0, 0.5, key="jm-online-interval")
            proxy = st.text_input(
                "HTTP(S) 代理（可留空）",
                "",
                help="留空使用 ONLINE_COVER_PROXY 环境配置；填写后仅覆盖本次任务。",
                key="jm-online-proxy",
            )
            no_resume = st.checkbox(
                "忽略已有断点，从新一轮开始",
                value=False,
                help="启用后不恢复上次未完成状态。",
                key="jm-online-no-resume",
            )
            state_file = st.text_input(
                "断点状态 JSONL（可留空）",
                "",
                help="留空时按模式、网址和输出路径在 logs/collection 中生成唯一文件。",
            )
            error_log = st.text_input(
                "失败记录 JSONL（可留空）",
                "",
                help="留空时与断点参数一起生成唯一文件。",
            )
            confirm = st.checkbox("确认执行完整采集", value=False)
            command = [
                PYTHON,
                "-m",
                "data_get.collector",
                "jm-online",
                "--base-url",
                base_url,
                "--start-url",
                start_url,
                "--max-pages",
                str(int(max_pages)),
                "--output-csv",
                output_csv,
                "--image-dir",
                image_dir,
                "--workers",
                str(int(workers)),
                "--request-attempts",
                str(int(request_attempts)),
                "--max-rounds",
                str(int(max_rounds)),
                "--retry-backoff",
                str(float(retry_backoff)),
                "--timeout",
                str(float(request_timeout)),
                "--interval",
                str(float(interval)),
                "--state-file",
                state_file,
                "--error-log",
                error_log,
            ]
            if proxy.strip():
                command.extend(["--proxy", proxy.strip()])
            if no_resume:
                command.append("--no-resume")
            submit_subprocess("开始完整采集 JM", "collection-jm-online", command, 0, require_confirm=confirm)
        render_result("collection-jm-online", "脚本输出会显示在这里。")
        return

    if mode == "NH 本地链接完整采集":
        with st.form("process-nh-local-info"):
            base_url = st.text_input("站点 Base URL", "https://nhentai.net", key="nh-local-info-base-url")
            input_file = st.text_input("本地链接 HTML/TXT", "data/local_data/NH_all.txt")
            output_csv = st.text_input(
                "原始信息 CSV",
                "data/gallery_info_origin/NH_info_local.csv",
            )
            image_dir = st.text_input("缩略图保存目录", "onlineimgtmp", key="nh-local-info-image-dir")
            workers = st.number_input("并发线程数", 1, 64, 5, 1, key="nh-local-info-workers")
            request_attempts = st.number_input("单次请求尝试次数", 1, 20, 3, 1, key="nh-local-info-attempts")
            request_timeout = st.number_input(
                "单次 HTTP 超时秒数",
                1.0,
                600.0,
                30.0,
                1.0,
                key="nh-local-info-http-timeout",
            )
            max_rounds = st.number_input(
                "最多轮数（含首轮，0 表示直至成功）",
                0,
                100000,
                0,
                1,
                key="nh-local-info-rounds",
            )
            retry_backoff = st.number_input("重试退避基数秒数", 0.0, 3600.0, 2.0, 0.5, key="nh-local-info-backoff")
            interval = st.number_input("成功项目间隔秒数", 0.0, 60.0, 0.0, 0.5, key="nh-local-info-interval")
            proxy = st.text_input(
                "HTTP(S) 代理（可留空）",
                "",
                help="留空使用 ONLINE_COVER_PROXY 环境配置；填写后仅覆盖本次任务。",
                key="nh-local-info-proxy",
            )
            no_resume = st.checkbox(
                "忽略已有断点，从新一轮开始",
                value=False,
                help="启用后不恢复上次未完成状态。",
                key="nh-local-info-no-resume",
            )
            state_file = st.text_input(
                "断点状态 JSONL（可留空）",
                "",
                help="留空时按输入和输出路径在 logs/collection 中生成唯一文件。",
            )
            error_log = st.text_input(
                "失败记录 JSONL（可留空）",
                "",
                help="留空时与断点参数一起生成唯一文件。",
            )
            confirm = st.checkbox("确认执行完整采集", value=False)
            command = [
                PYTHON,
                "-m",
                "data_get.collector",
                "nh-local-info",
                "--base-url",
                base_url,
                "--input-file",
                input_file,
                "--output-csv",
                output_csv,
                "--image-dir",
                image_dir,
                "--workers",
                str(int(workers)),
                "--request-attempts",
                str(int(request_attempts)),
                "--timeout",
                str(float(request_timeout)),
                "--max-rounds",
                str(int(max_rounds)),
                "--retry-backoff",
                str(float(retry_backoff)),
                "--interval",
                str(float(interval)),
                "--state-file",
                state_file,
                "--error-log",
                error_log,
            ]
            if proxy.strip():
                command.extend(["--proxy", proxy.strip()])
            if no_resume:
                command.append("--no-resume")
            submit_subprocess("开始完整采集本地链接", "collection-nh-local-info", command, 0, require_confirm=confirm)
        render_result("collection-nh-local-info", "脚本输出会显示在这里。")
        return

    if mode == "NH 本地链接抓取分册图片":
        with st.form("process-nh-local-images"):
            base_url = st.text_input("站点 Base URL", "https://nhentai.net", key="nh-local-images-base-url")
            input_file = st.text_input("本地链接 HTML/TXT", "data/local_data/NH_2.txt")
            output_dir = st.text_input("图片保存根目录", "output")
            max_page_limit = st.number_input("单本最大页数保护", 1, 10000, 200, 1)
            workers = st.number_input("并发线程数", 1, 64, 4, 1, key="nh-local-images-workers")
            request_attempts = st.number_input("单次请求尝试次数", 1, 20, 3, 1, key="nh-local-images-attempts")
            request_timeout = st.number_input(
                "单次 HTTP 超时秒数",
                1.0,
                600.0,
                30.0,
                1.0,
                key="nh-local-images-http-timeout",
            )
            max_rounds = st.number_input(
                "最多轮数（含首轮，0 表示直至成功）",
                0,
                100000,
                0,
                1,
                key="nh-local-images-rounds",
            )
            retry_backoff = st.number_input("重试退避基数秒数", 0.0, 3600.0, 2.0, 0.5, key="nh-local-images-backoff")
            interval = st.number_input("成功图片间隔秒数", 0.0, 60.0, 0.0, 0.5, key="nh-local-images-interval")
            proxy = st.text_input(
                "HTTP(S) 代理（可留空）",
                "",
                help="留空使用 ONLINE_COVER_PROXY 环境配置；填写后仅覆盖本次任务。",
                key="nh-local-images-proxy",
            )
            no_resume = st.checkbox(
                "忽略已有断点，从新一轮开始",
                value=False,
                help="启用后不恢复上次未完成状态。",
                key="nh-local-images-no-resume",
            )
            state_file = st.text_input(
                "断点状态 JSONL（可留空）",
                "",
                help="留空时按输入和输出路径在 logs/collection 中生成唯一文件。",
            )
            error_log = st.text_input(
                "失败记录 JSONL（可留空）",
                "",
                help="留空时与断点参数一起生成唯一文件。",
            )
            confirm = st.checkbox("确认执行完整采集", value=False)
            command = [
                PYTHON,
                "-m",
                "data_get.collector",
                "nh-local-images",
                "--base-url",
                base_url,
                "--input-file",
                input_file,
                "--output-dir",
                output_dir,
                "--max-pages",
                str(int(max_page_limit)),
                "--workers",
                str(int(workers)),
                "--request-attempts",
                str(int(request_attempts)),
                "--timeout",
                str(float(request_timeout)),
                "--max-rounds",
                str(int(max_rounds)),
                "--retry-backoff",
                str(float(retry_backoff)),
                "--interval",
                str(float(interval)),
                "--state-file",
                state_file,
                "--error-log",
                error_log,
            ]
            if proxy.strip():
                command.extend(["--proxy", proxy.strip()])
            if no_resume:
                command.append("--no-resume")
            submit_subprocess("开始完整抓取分册图片", "collection-nh-local-images", command, 0, require_confirm=confirm)
        render_result("collection-nh-local-images", "脚本输出会显示在这里。")
        return

    st.info("请选择一个采集流程。")


def render_data_processing_interface() -> None:
    st.subheader("数据处理")
    render_overview()

    tab_csv, tab_db, tab_title_translate, tab_cache, tab_maintenance, tab_collection = st.tabs(
        ["CSV 整理", "数据库同步", "标题AI翻译", "缓存与向量", "维护工具", "采集入口"]
    )

    with tab_csv:
        render_csv_tools()

    with tab_db:
        render_database_tools()

    with tab_title_translate:
        render_title_translation_tools()

    with tab_cache:
        render_cache_tools()

    with tab_maintenance:
        render_maintenance_tools()

    with tab_collection:
        render_collection_tools()
