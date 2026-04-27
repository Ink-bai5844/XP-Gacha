import csv
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from bs4 import BeautifulSoup

from JM_get_info_online import (
    BASE_URL,
    CSV_HEADERS,
    CSV_PATH,
    ERROR_LOG_PATH,
    FAILED_PAGES_REPORT_PATH,
    LOG_DIR,
    MAX_WORKERS,
    OUTPUT_DIR,
    START_URL,
    collect_page_tasks,
    ensure_csv_schema,
    filter_pending_tasks,
    format_task_progress,
    init_failed_pages_report,
    is_http_403,
    load_existing_csv_ids,
    logger,
    request_with_retry,
    write_failed_page_report,
    fetch_comic,
)


SOURCE_ERROR_LOG = os.path.join(LOG_DIR, "getjm_errors_20260427_141729.log")
RETRY_ERROR_LOG = os.path.join(LOG_DIR, "getjm_fix_retry.log")
PAGE_PATTERN = re.compile(r"\[第(\d+)页\s*/")


def load_error_pages(log_path):
    """从错误日志中提取失败页码，并按首次出现顺序去重。"""
    if not os.path.exists(log_path):
        print(f"未找到错误日志文件: {log_path}")
        return []

    pages = []
    seen = set()

    with open(log_path, "r", encoding="utf-8") as log_file:
        for line in log_file:
            match = PAGE_PATTERN.search(line)
            if not match:
                continue

            page = int(match.group(1))
            if page not in seen:
                seen.add(page)
                pages.append(page)

    return pages


def build_page_url(page_number):
    """根据页码构造列表页 URL。"""
    if page_number <= 1:
        return START_URL

    parsed = urlparse(START_URL)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["page"] = str(page_number)
    return urlunparse(parsed._replace(query=urlencode(query)))


def get_page_tasks(page_number, start_index=0):
    """抓取指定页并解析出该页的全部漫画任务。"""
    page_url = build_page_url(page_number)
    response = request_with_retry(page_url, timeout=15, request_name=f"请求重试列表页 {page_url}")
    soup = BeautifulSoup(response.text, "html.parser")
    tasks = collect_page_tasks(soup, page_number, start_index)
    return page_url, tasks


def append_retry_error(message):
    with open(RETRY_ERROR_LOG, "a", encoding="utf-8") as log_file:
        log_file.write(message.rstrip("\n") + "\n")


def retry_failed_pages():
    error_pages = load_error_pages(SOURCE_ERROR_LOG)
    if not error_pages:
        print("错误日志中没有可重试的页码，程序结束。")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ensure_csv_schema(CSV_PATH)
    existing_ids = load_existing_csv_ids(CSV_PATH)
    init_failed_pages_report(FAILED_PAGES_REPORT_PATH)

    preview = ", ".join(str(page) for page in error_pages[:20])
    if len(error_pages) > 20:
        preview += ", ..."

    logger.info("修复模式：将按错误日志中的页码重试，共 %s 页。", len(error_pages))
    logger.info("待重试页码：%s", preview)
    logger.info("初始化查重：已在本地CSV发现 %s 条历史记录。", len(existing_ids))
    logger.info("原始错误日志：%s", SOURCE_ERROR_LOG)
    logger.info("修复追加日志：%s", RETRY_ERROR_LOG)

    csv_exists = os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0
    total_count = 0
    total_seen = 0

    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as csv_file:
        csv_writer = csv.writer(csv_file)
        if not csv_exists:
            csv_writer.writerow(CSV_HEADERS)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for current_index, page in enumerate(error_pages, 1):
                logger.info("========== [%s/%s] 开始重试第 %s 页 ==========", current_index, len(error_pages), page)

                try:
                    page_url, tasks = get_page_tasks(page, start_index=total_seen)
                except Exception as exc:
                    write_failed_page_report(page, build_page_url(page), exc)
                    if is_http_403(exc):
                        logger.warning("重试列表页被拒绝(403): %s", build_page_url(page))
                    else:
                        logger.error("重试列表页失败 %s: %s", build_page_url(page), exc, exc_info=True)
                    append_retry_error(f"[第{page}页] 列表页失败: {exc}")
                    time.sleep(2)
                    continue

                if not tasks:
                    logger.warning("第 %s 页未解析到漫画任务，跳过。URL: %s", page, page_url)
                    time.sleep(1)
                    continue

                total_seen += len(tasks)
                tasks, skipped_count = filter_pending_tasks(tasks, existing_ids)
                if skipped_count:
                    logger.info("第 %s 页因查重跳过 %s 条。", page, skipped_count)
                if not tasks:
                    logger.info("第 %s 页经查重后无需重爬。", page)
                    time.sleep(1)
                    continue

                logger.info("第 %s 页共需重试 %s 个漫画，开启 %s 个线程并发处理...", page, len(tasks), MAX_WORKERS)

                futures = {executor.submit(fetch_comic, task): task for task in tasks}
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        title, row = future.result()
                        csv_writer.writerow(row)
                        csv_file.flush()
                        existing_ids.add(row[0])
                        total_count += 1
                        logger.info("%s 重试成功: %s", format_task_progress(task), title)
                    except Exception as exc:
                        if is_http_403(exc):
                            logger.warning("%s 重试详情页仍被拒绝(403): %s", format_task_progress(task), task["detail_url"])
                        else:
                            logger.error("%s 重试详情页失败 %s: %s", format_task_progress(task), task["detail_url"], exc, exc_info=True)
                        append_retry_error(f"[第{page}页] 详情页失败: {task['detail_url']} | {exc}")

                logger.info("--- 第 %s 页重试完成 ---", page)
                time.sleep(1)

    logger.info("失败页重试完成！本次共新抓取 %s 条详情数据。", total_count)


if __name__ == "__main__":
    retry_failed_pages()
