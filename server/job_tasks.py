"""Explicit allow-listed adapter from web job forms to legacy processing modules."""

from __future__ import annotations

import base64
import importlib
import json
import os
import shutil
import subprocess
import sys
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
        "collection-nh-online": (
            "data_get.NH_get_info_online",
            {"baseUrl": ("BASE_URL", False), "startUrl": ("START_URL", False), "maxPage": ("MAX_PAGE", False), "outputCsv": ("OUTPUT_CSV", True), "imageDir": ("IMG_DIR", True), "errorLog": ("ERROR_LOG", True), "workers": ("MAX_WORKERS", False)},
            lambda mod: mod.main(max_page=mod.MAX_PAGE, start_url=mod.START_URL, base_url=mod.BASE_URL, output_csv=str(mod.OUTPUT_CSV), image_dir=str(mod.IMG_DIR), error_log=str(mod.ERROR_LOG), max_workers=int(mod.MAX_WORKERS), loop=False),
        ),
        "collection-jm-online": (
            "data_get.JM_get_info_online",
            {"baseUrl": ("BASE_URL", False), "startUrl": ("START_URL", False), "maxPages": ("MAX_PAGES", False), "csvPath": ("CSV_PATH", True), "outputDir": ("OUTPUT_DIR", True), "workers": ("MAX_WORKERS", False)},
            lambda mod: mod.scrape_18comic(),
        ),
        "collection-nh-retry": (
            "data_get.NH_get_info_online_fix",
            {"baseUrl": ("BASE_URL", False), "startUrl": ("START_URL", False), "sourceLog": ("SOURCE_ERROR_LOG", True), "retryLog": ("RETRY_ERROR_LOG", True), "outputCsv": ("OUTPUT_CSV", True), "imageDir": ("IMG_DIR", True), "workers": ("MAX_WORKERS", False)},
            lambda mod: mod.main(),
        ),
        "collection-jm-retry": (
            "data_get.JM_get_info_online_fix",
            {"baseUrl": ("BASE_URL", False), "startUrl": ("START_URL", False), "sourceLog": ("SOURCE_ERROR_LOG", True), "retryLog": ("RETRY_ERROR_LOG", True), "failedReport": ("FAILED_PAGES_REPORT_PATH", True), "csvPath": ("CSV_PATH", True), "outputDir": ("OUTPUT_DIR", True), "workers": ("MAX_WORKERS", False)},
            lambda mod: mod.main(),
        ),
        "collection-nh-local-info": (
            "data_get.local.NH_get_info_local",
            {"inputFile": ("INPUT_FILE", True), "outputCsv": ("OUTPUT_CSV", True), "errorLog": ("ERROR_LOG", True), "interval": ("REQUEST_INTERVAL_SECONDS", False)},
            lambda mod: mod.main(),
        ),
        "collection-nh-local-images": (
            "data_get.local.NH_get_images_local",
            {"inputFile": ("INPUT_FILE", True), "rootDir": ("ROOT_DIR", True), "errorLog": ("ERROR_LOG", True), "maxPages": ("MAX_PAGE_LIMIT", False), "interval": ("REQUEST_INTERVAL_SECONDS", False), "retries": ("PAGE_RETRY_TIMES", False)},
            lambda mod: mod.main(),
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
