"""Explicit allow-listed adapter from web job forms to legacy processing modules."""

from __future__ import annotations

import base64
import importlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import config
from server.database import get_engine
from server.modules.imports import import_dataframe, read_csv_files


ROOT = Path(__file__).resolve().parents[1]


def project_path(value: object) -> Path:
    path = Path(str(value or "").strip())
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError(f"任务路径必须位于项目目录内：{resolved}")
    return resolved


def require_confirm(parameters: dict) -> None:
    if not parameters.get("confirm"):
        raise ValueError("该任务必须在界面中勾选确认项")


def assign(module, parameters: dict, mapping: dict[str, tuple[str, bool]]) -> None:
    for field, (attribute, is_path) in mapping.items():
        if field in parameters:
            value = project_path(parameters[field]) if is_path else parameters[field]
            setattr(module, attribute, value)


COLLECTION_MODES = {
    "collection-nh-online": "nh-online",
    "collection-jm-online": "jm-online",
    "collection-nh-local-info": "nh-local-info",
    "collection-nh-local-images": "nh-local-images",
}

COLLECTION_DEFAULTS = {
    "nh-online": {
        "base_url": "https://nhentai.net",
        "start_url": "https://nhentai.net/language/chinese/?sort=date",
        "max_pages": 1,
        "output_csv": "data/gallery_info_origin/NH_info_chinese.csv",
        "image_dir": "onlineimgtmp",
        "workers": 10,
        "state_file": None,
        "error_log": None,
    },
    "jm-online": {
        "base_url": "https://18comic.vip",
        "start_url": "https://18comic.vip/search/photos?search_query=%E7%99%BE%E5%90%88",
        "max_pages": 80,
        "output_csv": "data/gallery_info_origin/JM_info_yuri.csv",
        "image_dir": "onlineimgtmp",
        "workers": 5,
        "state_file": None,
        "error_log": None,
    },
    "nh-local-info": {
        "base_url": "https://nhentai.net",
        "max_pages": 1,
        "input_file": "data/local_data/NH_all.txt",
        "output_csv": "data/gallery_info_origin/NH_info_local.csv",
        "image_dir": "onlineimgtmp",
        "workers": 5,
        "state_file": None,
        "error_log": None,
    },
    "nh-local-images": {
        "base_url": "https://nhentai.net",
        "max_pages": 200,
        "input_file": "data/local_data/NH_2.txt",
        "output_dir": "output",
        "workers": 4,
        "state_file": None,
        "error_log": None,
    },
}


def parameter_value(parameters: dict, *names: str, default=None):
    """Return the first supplied form value while accepting legacy field aliases."""
    for name in names:
        if name in parameters and parameters[name] is not None:
            return parameters[name]
    return default


def parameter_path(parameters: dict, *names: str, default=None) -> Path | None:
    value = parameter_value(parameters, *names, default=default)
    if value is None or not str(value).strip():
        return None
    return project_path(value)


def parameter_bool(parameters: dict, *names: str, default: bool = False) -> bool:
    value = parameter_value(parameters, *names, default=default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"布尔参数 {names[0]} 的值无效：{value}")


def print_summary(summary: object) -> None:
    if callable(getattr(summary, "as_dict", None)):
        payload = summary.as_dict()
    elif is_dataclass(summary):
        payload = asdict(summary)
    elif isinstance(summary, dict):
        payload = summary
    else:
        payload = {"result": str(summary)}
    print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)


def run_collection_task(script_id: str, parameters: dict) -> bool:
    mode = COLLECTION_MODES.get(script_id)
    if not mode:
        return False
    require_confirm(parameters)

    from data_get.collector import CollectionConfig, run_collection

    defaults = COLLECTION_DEFAULTS[mode]
    image_field_names = ("imageDir", "outputDir", "image_dir") if mode == "jm-online" else ("imageDir", "image_dir")
    raw_proxy = parameter_value(parameters, "proxy", default=None)
    proxy = str(raw_proxy).strip() if raw_proxy is not None else ""
    config = CollectionConfig(
        mode=mode,
        base_url=str(parameter_value(parameters, "baseUrl", "base_url", default=defaults.get("base_url", ""))),
        start_url=str(parameter_value(parameters, "startUrl", "start_url", default=defaults.get("start_url", ""))),
        max_pages=int(parameter_value(parameters, "maxPages", "maxPage", "max_pages", default=defaults["max_pages"])),
        output_csv=parameter_path(parameters, "outputCsv", "csvPath", "output_csv", default=defaults.get("output_csv")),
        image_dir=parameter_path(parameters, *image_field_names, default=defaults.get("image_dir")),
        input_file=parameter_path(parameters, "inputFile", "input_file", default=defaults.get("input_file")),
        output_dir=parameter_path(parameters, "outputDir", "rootDir", "output_dir", default=defaults.get("output_dir")),
        workers=int(parameter_value(parameters, "workers", "maxWorkers", default=defaults["workers"])),
        request_attempts=int(parameter_value(parameters, "requestAttempts", "retries", "request_attempts", default=3)),
        max_rounds=int(parameter_value(parameters, "retryRounds", "maxRounds", "max_rounds", default=0)),
        retry_backoff=float(parameter_value(parameters, "retryBackoff", "retry_backoff", default=2.0)),
        timeout=float(parameter_value(parameters, "requestTimeout", "request_timeout", default=30.0)),
        interval=float(parameter_value(parameters, "interval", default=0.0)),
        state_file=parameter_path(parameters, "stateFile", "state_file", default=defaults.get("state_file")),
        error_log=parameter_path(parameters, "errorLog", "error_log", default=defaults.get("error_log")),
        proxy=proxy or None,
        resume=not parameter_bool(parameters, "noResume", "no_resume", default=False),
    )
    summary = run_collection(config)
    print_summary(summary)
    exit_code = int(getattr(summary, "exit_code", 0) or 0)
    if exit_code:
        raise SystemExit(exit_code)
    return True


def run_module(script_id: str, parameters: dict) -> bool:
    specs = {
        "addname": (
            "data_processing.addname", {},
            lambda mod: mod.process_gallery_data(
                str(project_path(parameters["csvFile"])),
                str(project_path(parameters["txtFile"])),
                str(project_path(parameters["outputFile"])),
            ),
        ),
        "add-id": (
            "tools.add_id", {"csvDir": ("CSV_DIR", True), "prefix": ("ID_PREFIX", False)},
            lambda mod: mod.main(),
        ),
        "add-lang": (
            "tools.add_lang", {"csvPath": ("CSV_PATH", True)},
            lambda mod: (setattr(mod, "LANG_TAGS", {x.strip() for x in str(parameters.get("languageTags", "")).replace("，", ",").split(",") if x.strip()}), mod.move_language_tags())[-1],
        ),
        "clean-date": (
            "tools.clean", {"csvPath": ("CSV_PATH", True)}, lambda mod: mod.clean_upload_dates(),
        ),
        "title-words": (
            "data_processing.title_cut_set",
            {
                "inputCsv": ("INPUT_CSV", True), "outputCsv": ("OUTPUT_CSV", True),
                "stopWords": ("TITLE_STOP_WORDS_PATH", True), "semanticMap": ("TITLE_SEMANTIC_MAP_PATH", True),
            },
            lambda mod: mod.process_title_words(),
        ),
        "map-add-name": (
            "data_processing.map_add_name", {"inputFile": ("INPUT_FILE", True), "outputFile": ("OUTPUT_FILE", True)},
            lambda mod: mod.transform_semantic_map(),
        ),
        "prefix-rename": (
            "tools.b64id" if parameters.get("targetKind") == "Base64 缓存 TXT" else "tools.imgid",
            {"targetDir": ("TARGET_DIR", True), "prefix": ("PREFIX", False)},
            lambda mod: mod.rename_txt_files() if hasattr(mod, "rename_txt_files") else mod.rename_images(),
        ),
    }
    spec = specs.get(script_id)
    if not spec:
        return False
    if script_id.startswith("collection-"):
        require_confirm(parameters)
    module_name, mapping, callback = spec
    module = importlib.import_module(module_name)
    assign(module, parameters, mapping)
    callback(module)
    return True


def run(script_id: str, parameters: dict) -> None:
    print(f"[JOB] {script_id} 参数校验完成", flush=True)
    if run_collection_task(script_id, parameters):
        return
    if run_module(script_id, parameters):
        return

    if script_id in {"db-sync", "db-rebuild"}:
        if script_id == "db-rebuild":
            require_confirm(parameters)
        csv_dir = project_path(parameters.get("csvDir", "data/gallery_info"))
        frame = read_csv_files(sorted(csv_dir.glob("*.csv")))
        result = import_dataframe(frame, "replace" if script_id == "db-rebuild" else "upsert")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    if script_id == "db-optimize":
        require_confirm(parameters)
        from data_processing.optimize_mysql_schema import optimize_gallery_schema
        optimize_gallery_schema(get_engine())
        return
    if script_id == "tag-set":
        module = importlib.import_module("data_processing.tag_set")
        semantic_map = module.load_semantic_map(str(project_path(parameters["semanticMap"])))
        tags = module.get_aggregated_tags(str(project_path(parameters["csvDir"])), semantic_map)
        exported = module.export_tags_to_document(tags, str(project_path(parameters["outputFile"])))
        print(f"共导出 {len(tags)} 个标签到 {exported}", flush=True)
        return
    if script_id == "b64":
        module = importlib.import_module("data_processing.b64_pre_encode")
        module.B64_CACHE_DIR = str(project_path(parameters.get("cacheDir", "b64_cache")))
        module.B64_TMP_DIR = str(project_path(parameters.get("tmpDir", "b64_tmp")))
        source_map = {"线上封面": config.ONLINE_IMG_DIR, "本地缩略图": config.IMG_CACHE_DIR}
        for source in parameters.get("sources", []):
            module.process_directory(source_map[source])
        return
    if script_id == "cache-delete":
        require_confirm(parameters)
        targets = {
            "预处理 DataFrame": Path(config.CACHE_DIR) / "preprocessed_df.pkl",
            "预处理 Hash": Path(config.CACHE_DIR) / "data.hash",
            "文本向量": Path(config.VECTOR_FILE),
            "封面向量": Path(config.IMG_VECTOR_FILE),
        }
        for label in parameters.get("targets", []):
            target = targets.get(label)
            if target and target.is_file():
                target.unlink()
                print(f"已删除 {label}: {target}", flush=True)
        return
    if script_id == "merge-b64":
        require_confirm(parameters)
        source = project_path(parameters.get("tmpDir", "b64_tmp"))
        target = project_path(parameters.get("cacheDir", "b64_cache"))
        target.mkdir(parents=True, exist_ok=True)
        for item in source.glob("*.txt"):
            destination = target / item.name
            if destination.exists() and not parameters.get("overwrite"):
                continue
            shutil.move(str(item), destination)
        return
    if script_id == "title-translate":
        require_confirm(parameters)
        command = [
            sys.executable, str(ROOT / "data_processing" / "translate_titles.py"),
            "--jsonl-output", str(project_path(parameters.get("jsonlOutput", "data_processing/title_translation_results.jsonl"))),
            "--failed-jsonl-output", str(project_path(parameters.get("failedJsonlOutput", "data_processing/title_translation_failed_results.jsonl"))),
            "--batch-size", str(int(parameters.get("batchSize", 20))),
            "--concurrency", str(int(parameters.get("concurrency", 3))),
            "--start-index", str(int(parameters.get("startIndex", 1))),
            "--request-timeout", str(int(parameters.get("requestTimeout", 120))),
            "--max-retries", str(int(parameters.get("maxRetries", 2))),
            "--temperature", str(float(parameters.get("temperature", 0.2))),
        ]
        if int(parameters.get("endIndex", 0)) > 0:
            command.extend(["--end-index", str(int(parameters["endIndex"]))])
        if parameters.get("jsonlOnly"):
            command.append("--jsonl-only")
        if parameters.get("lmStudio"):
            command.append("--lm-studio")
        else:
            command.extend(["--api-url", str(parameters.get("apiUrl", "")), "--model", str(parameters.get("model", ""))])
        child_env = os.environ.copy()
        if parameters.get("apiKey"):
            child_env["OPENAI_API_KEY"] = str(parameters["apiKey"])
        subprocess.run(command, cwd=ROOT, env=child_env, check=True)
        return
    if script_id == "text-vector":
        command = [
            sys.executable, str(ROOT / "data_processing" / "build_vector_db.py"),
            "--model-path", str(project_path(parameters.get("modelPath", "models/Qwen3-Embedding-0.6B"))),
            "--vector-file", str(project_path(parameters.get("vectorFile", "manga_vectors/manga_vectors_Qwen3.pkl"))),
            "--batch-size", str(int(parameters.get("batchSize", 16))),
            "--max-text-length", str(int(parameters.get("maxTextLength", 800))),
            "--sql-query", str(parameters.get("sql", "SELECT * FROM gallery_info WHERE ID != ''")),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        return
    if script_id == "clip-vector":
        action = "build" if parameters.get("action", "构建/刷新") == "构建/刷新" else "stats"
        command = [
            sys.executable, str(ROOT / "data_processing" / "img_to_vector.py"), action,
            "--model", str(project_path(parameters.get("modelPath", "models/clip-vit-base-patch32"))),
            "--index-path", str(project_path(parameters.get("indexPath", "manga_vectors/clip_image_index.pkl"))),
            "--batch-size", str(int(parameters.get("batchSize", 64))),
            "--device", str(parameters.get("device", "auto")),
        ]
        for image_dir in str(parameters.get("imageDirs", "onlineimgtmp, localimgtmp")).replace("，", ",").split(","):
            if image_dir.strip():
                command.extend(["--image-dir", str(project_path(image_dir.strip()))])
        if action == "build" and parameters.get("rebuild"):
            command.append("--rebuild")
        subprocess.run(command, cwd=ROOT, check=True)
        return
    if script_id == "clean-title-jsonl":
        require_confirm(parameters)
        command = [
            sys.executable, str(ROOT / "tools" / "clean_failed_title_translation_jsonl.py"),
            "--jsonl-path", str(project_path(parameters.get("jsonlPath", "data_processing/title_translation_results.jsonl"))),
        ]
        if not parameters.get("keepBackup", True):
            command.append("--no-backup")
        subprocess.run(command, cwd=ROOT, check=True)
        return
    if script_id == "delete-gallery-rows":
        require_confirm(parameters)
        command = [sys.executable, str(ROOT / "tools" / "delete_gallery_rows_by_id.py")]
        ids = str(parameters.get("ids", "")).strip()
        if ids:
            command.append(ids)
        if str(parameters.get("idFile", "")).strip():
            command.extend(["--id-file", str(project_path(parameters["idFile"]))])
        command.append("--confirm")
        subprocess.run(command, cwd=ROOT, check=True)
        return
    if script_id == "clear-title-translation":
        require_confirm(parameters)
        command = [
            sys.executable, str(ROOT / "tools" / "clear_title_translation_by_error_ids.py"),
            "--input", str(project_path(parameters.get("inputFile", "tools/error.json"))),
            "--preview-limit", str(int(parameters.get("previewLimit", 30))),
            "--chunk-size", str(int(parameters.get("chunkSize", 500))),
            "--confirm",
        ]
        if parameters.get("emptyString"):
            command.append("--empty-string")
        subprocess.run(command, cwd=ROOT, check=True)
        return
    if script_id == "export-title-translations":
        csv_dir = project_path(parameters.get("csvDir", "data/gallery_info"))
        pattern = str(parameters.get("pattern", "*_full.csv")).strip()
        if not pattern:
            raise ValueError("文件匹配模式不能为空")
        if ".." in pattern or "/" in pattern or "\\" in pattern:
            raise ValueError("文件匹配模式只能是当前 CSV 目录内的文件名模式")
        dry_run = bool(parameters.get("dryRun"))
        if not dry_run:
            require_confirm(parameters)
        from tools.export_title_translations_to_csv import export_title_translations

        print_summary(export_title_translations(csv_dir=csv_dir, pattern=pattern, dry_run=dry_run))
        return
    raise ValueError(f"脚本 {script_id} 尚未建立安全执行适配器")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m server.job_tasks <script-id> <base64-json>")
        return 2
    parameters = json.loads(base64.urlsafe_b64decode(sys.argv[2].encode("ascii")).decode("utf-8"))
    run(sys.argv[1], parameters)
    print("[JOB] completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
