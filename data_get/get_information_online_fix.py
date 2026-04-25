import csv
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from get_information_online import (
    IMG_DIR,
    MAX_WORKERS,
    csv_lock,
    get_page_urls,
    load_existing_urls,
    log_lock,
    process_single_gallery,
)

OUTPUT_CSV = "gallery_info_chinese.csv"
SOURCE_ERROR_LOG = "error_log_online_4.txt"
RETRY_ERROR_LOG = "error_log_online_fix.txt"
PAGE_PATTERN = re.compile(r"\[页数:\s*(\d+)")


def load_error_pages(log_path):
    """从错误日志中提取失败页码，并按首次出现顺序去重。"""
    if not os.path.exists(log_path):
        print(f"未找到错误日志文件: {log_path}")
        return []

    pages = []
    seen = set()

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = PAGE_PATTERN.search(line)
            if not match:
                continue

            page = int(match.group(1))
            if page not in seen:
                seen.add(page)
                pages.append(page)

    return pages


def append_error_log(log_path, message):
    """线程安全地写入修复任务的错误日志。"""
    with log_lock:
        with open(log_path, "a", encoding="utf-8") as f_err:
            f_err.write(message.rstrip("\n") + "\n")


def main():
    error_pages = load_error_pages(SOURCE_ERROR_LOG)
    if not error_pages:
        print("错误日志中没有可重试的页码，程序结束。")
        return

    os.makedirs(IMG_DIR, exist_ok=True)
    processed_urls = load_existing_urls(OUTPUT_CSV)
    page_preview = ", ".join(str(page) for page in error_pages[:20])
    if len(error_pages) > 20:
        page_preview += ", ..."

    print(f"修复模式：将按错误日志中的页码重试，共 {len(error_pages)} 页。")
    print(f"待重试页码：{page_preview}")
    print(f"初始化查重：已在本地CSV发现 {len(processed_urls)} 条历史记录。")

    csv_headers = ["链接", "标题", "标签", "作者", "团队", "语言", "页数", "上传日期"]
    write_header = not os.path.exists(OUTPUT_CSV)
    total_count = 0

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8-sig") as f_csv:
        writer = csv.writer(f_csv)
        if write_header:
            with csv_lock:
                writer.writerow(csv_headers)
                f_csv.flush()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for current_index, page in enumerate(error_pages, 1):
                print(f"\n========== [{current_index}/{len(error_pages)}] 开始重试第 {page} 页 ==========")
                items = get_page_urls(page)

                if items is None:
                    error_msg = f"[页数: {page}] 页面不存在或返回 404"
                    print(f"⚠️ {error_msg}")
                    append_error_log(RETRY_ERROR_LOG, error_msg)
                    time.sleep(1)
                    continue

                if items is False:
                    error_msg = f"[页数: {page}] 页面列表获取失败"
                    print(f"❌ {error_msg}")
                    append_error_log(RETRY_ERROR_LOG, error_msg)
                    time.sleep(3)
                    continue

                if not items:
                    print(f"第 {page} 页数据提取为空，跳过。")
                    time.sleep(1)
                    continue

                print(f"第 {page} 页共提取到 {len(items)} 个图库项目。开启 {MAX_WORKERS} 个线程并发处理...")

                futures = []
                for item_index, item in enumerate(items, 1):
                    future = executor.submit(
                        process_single_gallery,
                        item,
                        item_index,
                        len(items),
                        page,
                        processed_urls,
                        writer,
                        f_csv,
                        RETRY_ERROR_LOG,
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    total_count += future.result()

                print(f"--- 第 {page} 页重试完成 ---")
                time.sleep(1)

    print("\n==============================")
    print(f"失败页重试完成！本次共新抓取 {total_count} 条详情数据。")
    print(f"如仍有失败记录，请查看 {RETRY_ERROR_LOG}。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，已手动停止。")
