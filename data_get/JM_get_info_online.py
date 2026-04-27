import csv
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.parse import urljoin

import cloudscraper
from bs4 import BeautifulSoup
from requests.exceptions import HTTPError


MAX_WORKERS = 5
BASE_URL = "https://18comic.vip"
START_URL = "https://18comic.vip/search/photos?main_tag=0&search_query=%E5%85%BD%E8%80%B3"
MAX_PAGES = 100
OUTPUT_DIR = "output1"
CSV_PATH = "JM_info_kemonomimi.csv"
LOG_DIR = "logs"
RETRY_TIMES = 3
BACKOFF_BASE_SECONDS = 1
ID_COLUMN = "ID"
LINK_COLUMN = "链接"
FILENAME_COLUMN = "文件名"
TAG_COLUMN = "标签"
LANGUAGE_COLUMN = "语言"
DATE_COLUMN = "上传日期"
ID_PREFIX = "JM"
LANGUAGE_TAGS = {"中文", "英文", "日文"}
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
PROXIES = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}
CSV_HEADERS = [ID_COLUMN, LINK_COLUMN, "标题", "标签", "作者", "团队", "语言", "页数", "上传日期"]
thread_local = threading.local()


def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_path = os.path.join(LOG_DIR, f"getjm_{run_id}.log")
    error_log_path = os.path.join(LOG_DIR, f"getjm_errors_{run_id}.log")
    failed_pages_report_path = os.path.join(LOG_DIR, f"failed_pages_{run_id}.csv")

    logger = logging.getLogger("getjm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    run_file_handler = logging.FileHandler(run_log_path, encoding="utf-8")
    run_file_handler.setLevel(logging.INFO)
    run_file_handler.setFormatter(formatter)

    error_file_handler = logging.FileHandler(error_log_path, encoding="utf-8")
    error_file_handler.setLevel(logging.WARNING)
    error_file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(run_file_handler)
    logger.addHandler(error_file_handler)
    return logger, run_log_path, error_log_path, failed_pages_report_path


logger, RUN_LOG_PATH, ERROR_LOG_PATH, FAILED_PAGES_REPORT_PATH = setup_logger()


def create_scraper():
    return cloudscraper.create_scraper(
        browser={
            "browser": "chrome",
            "platform": "windows",
            "desktop": True,
        }
    )


def get_scraper():
    scraper = getattr(thread_local, "scraper", None)
    if scraper is None:
        scraper = create_scraper()
        thread_local.scraper = scraper
    return scraper


def is_http_403(exc):
    return isinstance(exc, HTTPError) and exc.response is not None and exc.response.status_code == 403


def init_failed_pages_report(report_path):
    with open(report_path, "w", newline="", encoding="utf-8-sig") as report_file:
        writer = csv.writer(report_file)
        writer.writerow(["page_number", "page_url", "error_type", "error_message", "failed_at"])


def write_failed_page_report(page_number, page_url, exc):
    with open(FAILED_PAGES_REPORT_PATH, "a", newline="", encoding="utf-8-sig") as report_file:
        writer = csv.writer(report_file)
        writer.writerow(
            [
                page_number,
                page_url,
                type(exc).__name__,
                str(exc),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ]
        )


def format_task_progress(task):
    return f"[第{task['page_number']}页 / 总第{task['global_index']}个漫画]"


def request_with_retry(url, timeout, request_name):
    scraper = get_scraper()
    last_exc = None

    for attempt in range(RETRY_TIMES + 1):
        try:
            response = scraper.get(url, timeout=timeout, proxies=PROXIES)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_exc = exc
            if attempt == RETRY_TIMES:
                break

            wait_seconds = BACKOFF_BASE_SECONDS * (2 ** attempt)
            logger.warning(
                "%s 失败，准备第 %s/%s 次重试，%s 秒后重试: %s",
                request_name,
                attempt + 1,
                RETRY_TIMES,
                wait_seconds,
                exc,
            )
            time.sleep(wait_seconds)

    raise last_exc


def extract_page_count(detail_soup):
    detail_text = detail_soup.get_text("\n", strip=True)
    match = re.search(r"\u9875\u6570\s*[:\uff1a]\s*(\d+)", detail_text)
    return match.group(1) if match else ""


def build_jm_id(raw_id):
    raw_id = (raw_id or "").strip()
    if not raw_id:
        return ""
    if raw_id.startswith(ID_PREFIX):
        return raw_id
    return f"{ID_PREFIX}{raw_id}"


def extract_jm_id_from_url(url):
    match = re.search(r"/album/(\d+)/?", (url or "").strip())
    if not match:
        return ""
    return build_jm_id(match.group(1))


def split_csv_items(value):
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def join_unique_items(items):
    unique_items = []
    seen = set()

    for item in items:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)

    return ", ".join(unique_items)


def normalize_tags_and_language(tags_value, language_value):
    tags = split_csv_items(tags_value)
    languages = split_csv_items(language_value)

    moved_languages = [tag for tag in tags if tag in LANGUAGE_TAGS]
    remaining_tags = [tag for tag in tags if tag not in LANGUAGE_TAGS]
    merged_language = join_unique_items(languages + moved_languages)

    return ", ".join(remaining_tags), merged_language


def normalize_upload_date(date_value):
    match = DATE_PATTERN.search(str(date_value or "").strip())
    return match.group(1) if match else str(date_value or "").strip()


def ensure_csv_schema(csv_path):
    """兼容旧 CSV：补上 ID 列、移除文件名列，并规范语言标签与上传日期。"""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames:
        return

    if LINK_COLUMN not in fieldnames:
        raise ValueError(f"CSV 中未找到列: {LINK_COLUMN}")

    new_fieldnames = [name for name in fieldnames if name != FILENAME_COLUMN and name != ID_COLUMN]
    new_fieldnames.insert(0, ID_COLUMN)

    needs_rewrite = fieldnames != new_fieldnames
    for row in rows:
        jm_id = extract_jm_id_from_url(row.get(LINK_COLUMN, ""))
        if row.get(ID_COLUMN, "") != jm_id:
            row[ID_COLUMN] = jm_id
            needs_rewrite = True

        normalized_tags, normalized_language = normalize_tags_and_language(
            row.get(TAG_COLUMN, ""),
            row.get(LANGUAGE_COLUMN, ""),
        )
        if row.get(TAG_COLUMN, "") != normalized_tags:
            row[TAG_COLUMN] = normalized_tags
            needs_rewrite = True
        if row.get(LANGUAGE_COLUMN, "") != normalized_language:
            row[LANGUAGE_COLUMN] = normalized_language
            needs_rewrite = True

        normalized_date = normalize_upload_date(row.get(DATE_COLUMN, ""))
        if row.get(DATE_COLUMN, "") != normalized_date:
            row[DATE_COLUMN] = normalized_date
            needs_rewrite = True

    if not needs_rewrite:
        return

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def download_cover(comic_id, img_url, output_dir, progress=""):
    if not img_url:
        return ""

    request_name = f"{progress} 下载图片 {img_url}".strip()
    response = request_with_retry(img_url, timeout=10, request_name=request_name)

    ext = img_url.split("?")[0].split(".")[-1]
    if not ext or len(ext) > 4:
        ext = "jpg"

    filename = f"{build_jm_id(comic_id)}.{ext}"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "wb") as file_obj:
        file_obj.write(response.content)
    return filename


def parse_detail(detail_url, progress=""):
    request_name = f"{progress} 获取详情页 {detail_url}".strip()
    response = request_with_retry(detail_url, timeout=15, request_name=request_name)
    detail_soup = BeautifulSoup(response.text, "html.parser")

    title_tag = detail_soup.find("h1", id="book-name")
    title = title_tag.text.strip() if title_tag else ""

    tags = []
    language = ""
    tag_container = detail_soup.find("span", {"itemprop": "genre", "data-type": "tags"})
    if tag_container:
        for tag in tag_container.find_all("a"):
            tag_text = tag.text.strip()
            if tag_text:
                tags.append(tag_text)

    tags_str, language = normalize_tags_and_language(", ".join(tags), "")

    authors = []
    author_container = detail_soup.find("span", {"itemprop": "author", "data-type": "author"})
    if author_container:
        for author in author_container.find_all("a"):
            author_text = author.text.strip()
            if author_text:
                authors.append(author_text)
    author_str = ", ".join(authors)

    date_tag = detail_soup.find("span", {"itemprop": "datePublished"})
    upload_date = normalize_upload_date(date_tag.text if date_tag else "")

    pages = extract_page_count(detail_soup)

    return {
        "title": title,
        "tags_str": tags_str,
        "author_str": author_str,
        "team_str": "",
        "language": language,
        "pages": pages,
        "upload_date": upload_date,
    }


def fetch_comic(task):
    comic_id = task["comic_id"]
    jm_id = task["id"]
    detail_url = task["detail_url"]
    img_url = task["img_url"]
    progress = format_task_progress(task)

    if img_url:
        try:
            download_cover(comic_id, img_url, OUTPUT_DIR, progress=progress)
        except Exception as exc:
            if is_http_403(exc):
                logger.warning("%s 下载图片被拒绝(403): %s", progress, img_url)
            else:
                logger.warning("%s 下载图片失败 %s: %s", progress, img_url, exc, exc_info=True)

    detail_data = parse_detail(detail_url, progress=progress)
    row = [
        jm_id,
        detail_url,
        detail_data["title"],
        detail_data["tags_str"],
        detail_data["author_str"],
        detail_data["team_str"],
        detail_data["language"],
        detail_data["pages"],
        detail_data["upload_date"],
    ]
    return detail_data["title"], row


def collect_page_tasks(soup, page_number, start_index):
    tasks = []
    for page_item_index, item in enumerate(soup.select("div.list-col"), start=1):
        a_tag = item.select_one('a[href^="/album/"]')
        if not a_tag:
            continue

        href = a_tag.get("href", "")
        match = re.search(r"/album/(\d+)", href)
        if not match:
            continue

        comic_id = match.group(1)
        detail_url = urljoin(BASE_URL, f"/album/{comic_id}/")

        img_tag = a_tag.select_one("img")
        img_url = ""
        if img_tag:
            img_url = img_tag.get("data-original") or img_tag.get("src") or ""

        tasks.append(
            {
                "comic_id": comic_id,
                "id": build_jm_id(comic_id),
                "detail_url": detail_url,
                "img_url": img_url,
                "page_number": page_number,
                "page_item_index": page_item_index,
                "global_index": start_index + len(tasks) + 1,
            }
        )
    return tasks


def load_existing_csv_ids(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()

    existing_ids = set()
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            jm_id = (row.get(ID_COLUMN, "") or "").strip()
            if jm_id:
                existing_ids.add(jm_id)
    return existing_ids


def has_existing_thumbnail(jm_id):
    prefix = f"{jm_id}."
    for file_name in os.listdir(OUTPUT_DIR):
        if file_name.startswith(prefix):
            return True
    return False


def filter_pending_tasks(tasks, existing_ids):
    pending_tasks = []
    skipped_count = 0

    for task in tasks:
        if task["id"] in existing_ids and has_existing_thumbnail(task["id"]):
            skipped_count += 1
            logger.info("%s Skip existing entry: %s", format_task_progress(task), task["id"])
            continue
        pending_tasks.append(task)

    return pending_tasks, skipped_count


def scrape_18comic():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    current_url = START_URL
    page_count = 0
    total_comics_seen = 0
    ensure_csv_schema(CSV_PATH)
    existing_ids = load_existing_csv_ids(CSV_PATH)
    init_failed_pages_report(FAILED_PAGES_REPORT_PATH)

    logger.info("开始抓取，完整日志: %s", RUN_LOG_PATH)
    logger.info("错误日志: %s", ERROR_LOG_PATH)
    logger.info("页面失败报告: %s", FAILED_PAGES_REPORT_PATH)

    csv_exists = os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0
    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as csv_file:
        csv_writer = csv.writer(csv_file)
        if not csv_exists:
            csv_writer.writerow(CSV_HEADERS)

        while current_url and page_count < MAX_PAGES:
            page_count += 1
            logger.info("=== 正在抓取第 %s 页: %s ===", page_count, current_url)

            try:
                response = request_with_retry(current_url, timeout=15, request_name=f"请求列表页 {current_url}")
            except Exception as exc:
                write_failed_page_report(page_count, current_url, exc)
                if is_http_403(exc):
                    logger.error("请求页面被拒绝(403): %s", current_url)
                else:
                    logger.error("请求页面失败 %s: %s", current_url, exc, exc_info=True)
                break

            soup = BeautifulSoup(response.text, "html.parser")
            tasks = collect_page_tasks(soup, page_count, total_comics_seen)
            if not tasks:
                logger.warning("未找到漫画列表，可能已到末页或被反爬拦截。当前页: %s", current_url)
                break
            total_comics_seen += len(tasks)

            tasks, skipped_count = filter_pending_tasks(tasks, existing_ids)
            if skipped_count:
                logger.info("Skipped %s existing entries on this page.", skipped_count)
            if not tasks:
                logger.info("No pending tasks on this page after dedupe.")

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(fetch_comic, task): task for task in tasks}
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        title, row = future.result()
                        csv_writer.writerow(row)
                        csv_file.flush()
                        existing_ids.add(row[0])
                        logger.info("%s 成功获取数据: %s", format_task_progress(task), title)
                    except Exception as exc:
                        if is_http_403(exc):
                            logger.warning("%s 获取详情页被拒绝(403): %s", format_task_progress(task), task["detail_url"])
                        else:
                            logger.error("%s 获取详情页失败 %s: %s", format_task_progress(task), task["detail_url"], exc, exc_info=True)

            next_page_a = soup.find("a", string=re.compile("下一"))
            if next_page_a and "href" in next_page_a.attrs:
                current_url = urljoin(BASE_URL, next_page_a["href"])
            else:
                logger.info("没有更多页数。")
                break

    logger.info("=== 抓取完成，详情页使用 %s 个线程并发 ===", MAX_WORKERS)


if __name__ == "__main__":
    scrape_18comic()
