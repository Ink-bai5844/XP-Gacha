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


def run_command(command: list[str], timeout: int = DEFAULT_TIMEOUT, live_output=None) -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

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

    command_text = " ".join(str(part) for part in result["command"])
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
            csv_file = st.text_input("CSV 文件", "data/gallery_info_no_name/JM_info_yuri.csv")
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
            csv_dir = st.text_input("CSV 目录", "data/gallery_info_no_name", key="add-id-dir")
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
            csv_path = st.text_input("CSV 文件", "data/gallery_info_no_name/JM_info_gender_bender.csv")
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


def render_collection_tools() -> None:
    mode = st.selectbox(
        "流程",
        [
            "NH 在线抓信息",
            "JM 在线抓信息",
            "NH 在线失败页重试",
            "JM 在线失败页重试",
            "NH 本地链接抓信息",
            "NH 本地链接抓图片",
        ],
        key="collection-script",
    )

    if mode == "NH 在线抓信息":
        with st.form("process-nh-online"):
            base_url = st.text_input("站点 Base URL", "https://nhentai.net")
            start_url = st.text_input(
                "抓取起始网址",
                "https://nhentai.net/language/chinese/?sort=date",
                help="作为第 1 页；后续页会自动追加或替换 page 参数。",
            )
            max_page = st.number_input("抓到多少页截止", 1, 100000, 1, 1)
            output_csv = st.text_input("保存 CSV 文件名及路径", "gallery_info_chinese.csv")
            image_dir = st.text_input("缩略图保存目录", "onlineimgtmp")
            error_log = st.text_input("错误日志路径", "logs/NH_error_log_online.txt")
            max_workers = st.number_input("并发线程数", 1, 64, 10, 1, key="nh-online-workers")
            timeout = st.number_input("超时秒数", 10, 86400, 1800, 60, key="nh-online-timeout")
            confirm = st.checkbox("确认执行采集脚本", value=False)
            code = module_call_code(
                "data_get.NH_get_info_online",
                {
                    "BASE_URL": base_url,
                    "START_URL": start_url,
                    "MAX_PAGE": int(max_page),
                    "OUTPUT_CSV": output_csv,
                    "IMG_DIR": image_dir,
                    "ERROR_LOG": error_log,
                    "MAX_WORKERS": int(max_workers),
                    "LOOP_CRAWL": False,
                },
                (
                    "mod.main("
                    "max_page=mod.MAX_PAGE, "
                    "start_url=mod.START_URL, "
                    "base_url=mod.BASE_URL, "
                    "output_csv=mod.OUTPUT_CSV, "
                    "image_dir=mod.IMG_DIR, "
                    "error_log=mod.ERROR_LOG, "
                    "max_workers=mod.MAX_WORKERS, "
                    "loop=mod.LOOP_CRAWL"
                    ")"
                ),
            )
            submit_python_code("开始抓取 NH 信息", "collection-nh-online", code, timeout, require_confirm=confirm)
        render_result("collection-nh-online", "脚本输出会显示在这里。")
        return

    if mode == "JM 在线抓信息":
        with st.form("process-jm-online"):
            base_url = st.text_input("站点 Base URL", "https://18comic.vip")
            start_url = st.text_input(
                "抓取起始网址",
                "https://18comic.vip/search/photos?search_query=%E7%99%BE%E5%90%88",
                help="作为第 1 页；后续页会自动追加或替换 page 参数。",
            )
            max_pages = st.number_input("抓到多少页截止", 1, 100000, 80, 1)
            csv_path = st.text_input("保存 CSV 文件名及路径", "JM_info_yuri.csv")
            output_dir = st.text_input("封面保存目录", "onlineimgtmp")
            max_workers = st.number_input("并发线程数", 1, 64, 5, 1, key="jm-online-workers")
            timeout = st.number_input("超时秒数", 10, 86400, 3600, 60, key="jm-online-timeout")
            confirm = st.checkbox("确认执行采集脚本", value=False)
            code = module_call_code(
                "data_get.JM_get_info_online",
                {
                    "BASE_URL": base_url,
                    "START_URL": start_url,
                    "MAX_PAGES": int(max_pages),
                    "CSV_PATH": csv_path,
                    "OUTPUT_DIR": output_dir,
                    "MAX_WORKERS": int(max_workers),
                },
                (
                    "import os\n"
                    "os.makedirs(os.path.dirname(mod.CSV_PATH) or '.', exist_ok=True)\n"
                    "os.makedirs(mod.OUTPUT_DIR, exist_ok=True)\n"
                    "mod.scrape_18comic()"
                ),
            )
            submit_python_code("开始抓取 JM 信息", "collection-jm-online", code, timeout, require_confirm=confirm)
        render_result("collection-jm-online", "脚本输出会显示在这里。")
        return

    if mode == "NH 本地链接抓信息":
        with st.form("process-nh-local-info"):
            input_file = st.text_input("本地链接 HTML/TXT", "data/local_data/NH_all.txt")
            output_csv = st.text_input("保存 CSV 文件名及路径", "gallery_info_local.csv")
            error_log = st.text_input("错误日志路径", "logs/NH_error_log_local.txt")
            interval = st.number_input("请求间隔秒数", 0.0, 60.0, 2.0, 0.5)
            timeout = st.number_input("超时秒数", 10, 86400, 1800, 60, key="nh-local-info-timeout")
            confirm = st.checkbox("确认执行采集脚本", value=False)
            code = module_call_code(
                "data_get.local.NH_get_info_local",
                {
                    "INPUT_FILE": input_file,
                    "OUTPUT_CSV": output_csv,
                    "ERROR_LOG": error_log,
                    "REQUEST_INTERVAL_SECONDS": float(interval),
                },
                "mod.main()",
            )
            submit_python_code("开始解析本地链接", "collection-nh-local-info", code, timeout, require_confirm=confirm)
        render_result("collection-nh-local-info", "脚本输出会显示在这里。")
        return

    if mode == "NH 本地链接抓图片":
        with st.form("process-nh-local-images"):
            input_file = st.text_input("本地链接 HTML/TXT", "data/local_data/NH_2.txt")
            root_dir = st.text_input("图片保存根目录", "output")
            error_log = st.text_input("错误日志路径", "logs/NH_error_log_images_local.txt")
            max_page_limit = st.number_input("单本最大页数保护", 1, 10000, 200, 1)
            interval = st.number_input("请求间隔秒数", 0.0, 60.0, 1.5, 0.5, key="nh-local-img-interval")
            retries = st.number_input("页面请求重试次数", 1, 20, 3, 1)
            timeout = st.number_input("超时秒数", 10, 86400, 3600, 60, key="nh-local-img-timeout")
            confirm = st.checkbox("确认执行采集脚本", value=False)
            code = module_call_code(
                "data_get.local.NH_get_images_local",
                {
                    "INPUT_FILE": input_file,
                    "ROOT_DIR": root_dir,
                    "ERROR_LOG": error_log,
                    "MAX_PAGE_LIMIT": int(max_page_limit),
                    "REQUEST_INTERVAL_SECONDS": float(interval),
                    "PAGE_RETRY_TIMES": int(retries),
                },
                "mod.main()",
            )
            submit_python_code("开始抓取本地链接图片", "collection-nh-local-images", code, timeout, require_confirm=confirm)
        render_result("collection-nh-local-images", "脚本输出会显示在这里。")
        return

    if mode == "NH 在线失败页重试":
        with st.form("process-nh-retry"):
            base_url = st.text_input("站点 Base URL", "https://nhentai.net", key="nh-retry-base-url")
            start_url = st.text_input(
                "重试页起始网址",
                "https://nhentai.net/language/chinese/?sort=date",
                help="用于根据错误页码重新拼接列表页 URL。",
                key="nh-retry-start-url",
            )
            source_error_log = st.text_input("读取的错误 Log 文件", "logs/NH_error_log_online.txt")
            retry_error_log = st.text_input("输出的错误报告文件", "logs/NH_error_log_online_fix.txt")
            output_csv = st.text_input("保存 CSV 文件名及路径", "gallery_info_chinese.csv")
            image_dir = st.text_input("缩略图保存目录", "onlineimgtmp")
            max_workers = st.number_input("并发线程数", 1, 64, 10, 1, key="nh-retry-workers")
            timeout = st.number_input("超时秒数", 10, 86400, 3600, 60, key="nh-retry-timeout")
            confirm = st.checkbox("确认执行失败页重试", value=False)
            code = module_call_code(
                "data_get.NH_get_info_online_fix",
                {
                    "BASE_URL": base_url,
                    "START_URL": start_url,
                    "SOURCE_ERROR_LOG": source_error_log,
                    "RETRY_ERROR_LOG": retry_error_log,
                    "OUTPUT_CSV": output_csv,
                    "IMG_DIR": image_dir,
                    "MAX_WORKERS": int(max_workers),
                },
                "mod.main()",
            )
            submit_python_code("开始 NH 失败页重试", "collection-nh-retry", code, timeout, require_confirm=confirm)
        render_result("collection-nh-retry", "脚本输出会显示在这里。")
        return

    if mode == "JM 在线失败页重试":
        with st.form("process-jm-retry"):
            base_url = st.text_input("站点 Base URL", "https://18comic.vip", key="jm-retry-base-url")
            start_url = st.text_input(
                "重试页起始网址",
                "https://18comic.vip/search/photos?search_query=%E7%99%BE%E5%90%88",
                help="用于根据错误页码重新拼接列表页 URL。",
                key="jm-retry-start-url",
            )
            source_error_log = st.text_input("读取的错误 Log 文件", "logs/getjm_errors_20260427_141729.log")
            retry_error_log = st.text_input("输出的错误日志文件", "logs/getjm_fix_retry.log")
            failed_pages_report = st.text_input("输出的错误报告 CSV", "logs/failed_pages_retry.csv")
            csv_path = st.text_input("保存 CSV 文件名及路径", "JM_info_yuri.csv")
            output_dir = st.text_input("封面保存目录", "onlineimgtmp")
            max_workers = st.number_input("并发线程数", 1, 64, 5, 1, key="jm-retry-workers")
            timeout = st.number_input("超时秒数", 10, 86400, 3600, 60, key="jm-retry-timeout")
            confirm = st.checkbox("确认执行失败页重试", value=False)
            code = module_call_code(
                "data_get.JM_get_info_online_fix",
                {
                    "BASE_URL": base_url,
                    "START_URL": start_url,
                    "SOURCE_ERROR_LOG": source_error_log,
                    "RETRY_ERROR_LOG": retry_error_log,
                    "FAILED_PAGES_REPORT_PATH": failed_pages_report,
                    "CSV_PATH": csv_path,
                    "OUTPUT_DIR": output_dir,
                    "MAX_WORKERS": int(max_workers),
                },
                "mod.retry_failed_pages()",
            )
            submit_python_code("开始 JM 失败页重试", "collection-jm-retry", code, timeout, require_confirm=confirm)
        render_result("collection-jm-retry", "脚本输出会显示在这里。")
        return

    st.info("请选择一个采集流程。")


def render_data_processing_interface() -> None:
    st.subheader("数据处理")
    render_overview()

    tab_csv, tab_db, tab_cache, tab_maintenance, tab_collection = st.tabs(
        ["CSV 整理", "数据库同步", "缓存与向量", "维护工具", "采集入口"]
    )

    with tab_csv:
        render_csv_tools()

    with tab_db:
        render_database_tools()

    with tab_cache:
        render_cache_tools()

    with tab_maintenance:
        render_maintenance_tools()

    with tab_collection:
        render_collection_tools()
