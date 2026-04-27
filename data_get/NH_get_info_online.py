import os
import csv
import time
import threading
import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from curl_cffi import requests 
from bs4 import BeautifulSoup

BASE_URL = "https://nhentai.net"
IMG_DIR = "onlineimgtmp_c"
LINK_COLUMN = "链接"
ID_COLUMN = "ID"
ID_PREFIX = "NH"
GALLERY_URL_PATTERN = re.compile(r"/g/(\d+)/?")
CSV_HEADERS = [ID_COLUMN, LINK_COLUMN, '标题', '标签', '作者', '团队', '语言', '页数', '上传日期']
PROXIES = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890"
}

MAX_WORKERS = 10
csv_lock = threading.Lock()  # 保护 CSV 写入
log_lock = threading.Lock()  # 保护 Error Log 写入

def parse_max_page():
    """Read max page n from command line or interactive input."""
    parser = argparse.ArgumentParser(
        description="Loop crawl pages 1..n until manually stopped."
    )
    parser.add_argument(
        "max_page",
        nargs="?",
        type=int,
        help="Max page n. The crawler loops from page 1 to n until Ctrl+C.",
    )
    parser.add_argument(
        "-n",
        "--max-page",
        dest="max_page_option",
        type=int,
        help="Max page n. Takes priority over positional max_page.",
    )
    args = parser.parse_args()

    max_page = args.max_page_option if args.max_page_option is not None else args.max_page
    while max_page is None or max_page < 1:
        raw = input("请输入要循环爬取的最大页码 n（>= 1）：").strip()
        try:
            max_page = int(raw)
        except ValueError:
            print("页码必须是整数，请重新输入。")
            continue

        if max_page < 1:
            print("页码必须大于等于 1，请重新输入。")

    return max_page


def load_existing_ids(csv_path):
    """读取已有的 CSV 文件，返回所有已经爬取过的 NH ID 集合。"""
    existing_ids = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                gallery_id = (row.get(ID_COLUMN, "") or "").strip()
                if gallery_id:
                    existing_ids.add(gallery_id)
    return existing_ids


def extract_nh_id(url):
    """从 nhentai 链接中提取数字 ID，并补上 NH 前缀。"""
    match = GALLERY_URL_PATTERN.search((url or "").strip())
    if not match:
        return ""
    return f"{ID_PREFIX}{match.group(1)}"


def ensure_csv_has_id_column(csv_path):
    """若 CSV 还是旧格式，则自动补齐 ID 列并调整到表头首列。"""
    if not os.path.exists(csv_path):
        return

    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames:
        return

    if LINK_COLUMN not in fieldnames:
        raise ValueError(f"CSV 中未找到列: {LINK_COLUMN}")

    new_fieldnames = [name for name in fieldnames if name != ID_COLUMN]
    new_fieldnames.insert(0, ID_COLUMN)

    needs_rewrite = (fieldnames != new_fieldnames)
    for row in rows:
        nh_id = extract_nh_id(row.get(LINK_COLUMN, ""))
        if row.get(ID_COLUMN, "") != nh_id:
            row[ID_COLUMN] = nh_id
            needs_rewrite = True

    if not needs_rewrite:
        return

    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def extract_field_data(soup, field_name):
    """提取标签信息"""
    containers = soup.find_all('div', class_='tag-container field-name')
    for container in containers:
        if field_name in container.text:
            names = [span.text.strip() for span in container.find_all('span', class_='name')]
            return ", ".join(names)
    return ""

def extract_upload_date(soup):
    """提取上传日期"""
    containers = soup.find_all('div', class_='tag-container field-name')
    for container in containers:
        if 'Uploaded:' in container.text:
            time_tag = container.find('time')
            if time_tag and time_tag.has_attr('datetime'):
                return time_tag['datetime'][:10]
    return ""

def download_thumbnail(img_url, nh_id, retries=3):
    """下载缩略图并保存到指定文件夹"""
    if not img_url:
        return False, "缩略图链接为空"
        
    ext = img_url.split('.')[-1].split('?')[0] 
    file_path = os.path.join(IMG_DIR, f"{nh_id}.{ext}")
    
    if os.path.exists(file_path):
        return True, ""

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                img_url,
                impersonate="chrome120",
                proxies=PROXIES,
                timeout=20
            )
            
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                return True, ""
            else:
                raise ValueError(f"HTTP状态码: {response.status_code}")
                
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
            else:
                return False, str(e)
                
    return False, "未知错误"

def get_gallery_info(url, retries=3):
    """请求详情网页并解析所需的漫画信息"""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url,
                impersonate="chrome120",
                proxies=PROXIES,
                timeout=30
            )
            
            if response.status_code == 404:
                return None
                
            if response.status_code != 200:
                raise ValueError(f"HTTP状态码异常: {response.status_code}")
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            title = ""
            title_tag = soup.find('h2', class_='title')
            if title_tag:
                title = title_tag.text.strip()
                
            if not title:
                h1_tag = soup.find('h1', class_='title')
                if h1_tag:
                    pretty_tag = h1_tag.find('span', class_='pretty')
                    title = pretty_tag.text.strip() if pretty_tag else h1_tag.text.strip()
            
            return {
                "id": extract_nh_id(url),
                "url": url,
                "title": title,
                "tags": extract_field_data(soup, 'Tags:'),
                "artists": extract_field_data(soup, 'Artists:'),
                "groups": extract_field_data(soup, 'Groups:'),
                "languages": extract_field_data(soup, 'Languages:'),
                "pages": extract_field_data(soup, 'Pages:'),
                "uploaded_date": extract_upload_date(soup)
            }
            
        except Exception as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                return None

def get_page_urls(page_num, retries=3):
    """获取指定页码上的所有漫画链接及其对应的缩略图"""
    page_url = f"{BASE_URL}/language/chinese/?sort=date&page={page_num}"
    print(f"\n[{time.strftime('%H:%M:%S')}] 正在扫描列表页: {page_url}")
    
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                page_url,
                impersonate="chrome120",
                proxies=PROXIES,
                timeout=30
            )
            
            if response.status_code == 404:
                return None  
            if response.status_code != 200:
                raise ValueError(f"HTTP状态码异常: {response.status_code}")
                
            soup = BeautifulSoup(response.text, 'html.parser')
            galleries = soup.find_all('div', class_='gallery')
            
            if not galleries:
                return [] 
            
            items = []
            for gallery in galleries:
                a_tag = gallery.find('a', class_='cover')
                img_tag = gallery.find('img')
                
                if a_tag and a_tag.has_attr('href'):
                    href = a_tag['href']
                    if href.startswith('/g/'):
                        items.append({
                            'url': BASE_URL + href,
                            'id': extract_nh_id(BASE_URL + href),
                            'thumb_url': img_tag.get('src', img_tag.get('data-src', '')) if img_tag else ""
                        })
            return items
            
        except Exception as e:
            print(f"列表页请求失败 (尝试 {attempt}/{retries}): {str(e)}")
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                return False 

def process_single_gallery(item, index, total, page, processed_ids, writer, f_csv, error_log):
    """
    单个画廊的处理函数（交由线程池执行）
    返回新增记录的数量 (0 或 1)
    """
    nh_id = item['id']
    gallery_url = item['url']
    thumb_url = item['thumb_url']
    
    # 读操作不需要锁
    is_in_csv = nh_id in processed_ids
    
    thumb_exists = True
    if thumb_url:
        ext = thumb_url.split('.')[-1].split('?')[0]
        thumb_path = os.path.join(IMG_DIR, f"{nh_id}.{ext}")
        thumb_exists = os.path.exists(thumb_path)
    
    if is_in_csv and thumb_exists:
        print(f"  [页 {page} - {index}/{total}] ⏭️ 双重已存，跳过: {nh_id}")
        return 0
        
    print(f"  [页 {page} - {index}/{total}] ⚡ 正在处理: {nh_id} (CSV已存:{is_in_csv} | 图已存:{thumb_exists})")
    
    new_record_added = 0
    
    # 下载缩略图
    if not thumb_exists and thumb_url:
        success, err_msg = download_thumbnail(thumb_url, nh_id)
        if success:
            print(f"    -> 🖼️ 缩略图保存成功: {nh_id}")
        else:
            with log_lock:
                with open(error_log, 'a', encoding='utf-8') as f_err:
                    f_err.write(f"[页数: {page}, ID: {nh_id}] 缩图获取失败: {thumb_url} ({err_msg})\n")
    
    # 抓取详情并写入 CSV
    if not is_in_csv:
        info = get_gallery_info(gallery_url)
        if info:
            # 涉及文件写入和共享集合修改，必须加锁
            with csv_lock:
                if info['id'] not in processed_ids: # 二次检查，防止极端情况下的并发写入
                    writer.writerow([
                        info['id'], info['url'], info['title'], info['tags'], info['artists'], 
                        info['groups'], info['languages'], info['pages'], info['uploaded_date']
                    ])
                    f_csv.flush()
                    processed_ids.add(info['id'])
                    new_record_added = 1
                    print(f"    -> 📝 CSV保存成功: {info['title'][:25]}...")
        else:
            with log_lock:
                with open(error_log, 'a', encoding='utf-8') as f_err:
                    f_err.write(f"[页数: {page}, ID: {nh_id}] 详情页获取失败: {gallery_url}\n")
                    
    return new_record_added


def main():
    output_csv = 'gallery_info_chinese.csv'
    error_log = 'logs/NH_error_log_online.txt'
    max_page = parse_max_page()
    
    os.makedirs(IMG_DIR, exist_ok=True)
    ensure_csv_has_id_column(output_csv)
    processed_ids = load_existing_ids(output_csv)
    print(f"循环模式：将从第 1 页爬到第 {max_page} 页，完成后重新从第 1 页开始。按 Ctrl+C 手动停止。")
    print(f"初始化查重：已在本地CSV发现 {len(processed_ids)} 条历史记录。")
    
    csv_headers = CSV_HEADERS
    write_header = not os.path.exists(output_csv)
    
    with open(output_csv, 'a', newline='', encoding='utf-8-sig') as f_csv:
        writer = csv.writer(f_csv)
        if write_header:
            writer.writerow(csv_headers)
            
        page = 1
        total_count = 0
        cycle_count = 1
        
        # 启动线程池
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            while True:
                if page > max_page:
                    cycle_count += 1
                    page = 1
                    print(f"\n========== 开始第 {cycle_count} 轮循环爬取 1..{max_page} ==========")

                items = get_page_urls(page)
                
                if items is None:
                    print(f"\n第 {page} 页不存在或没有漫画数据 (404)，停止扫描。")
                    print(f"第 {page} 页不存在或返回 404，跳过该页，继续循环范围内的后续页。")
                    page += 1
                    time.sleep(3)
                    continue
                    
                if items is False:
                    error_msg = f"[页数: {page}] 页面列表获取失败\n"
                    print(f" ❌ {error_msg.strip()}，已写入日志，继续尝试下一页...")
                    with log_lock:
                        with open(error_log, 'a', encoding='utf-8') as f_err:
                            f_err.write(error_msg)
                    page += 1
                    time.sleep(3)
                    continue
                    
                if not items:
                    print(f"第 {page} 页数据提取为空，尝试扫描下一页...")
                    page += 1
                    time.sleep(1)
                    continue
                    
                print(f"第 {page} 页共提取到 {len(items)} 个图库项目。开启 {MAX_WORKERS} 个线程并发处理...")
                
                # 提交当前页的所有任务到线程池
                futures = []
                for index, item in enumerate(items, 1):
                    future = executor.submit(
                        process_single_gallery, 
                        item, index, len(items), page, 
                        processed_ids, writer, f_csv, error_log
                    )
                    futures.append(future)
                
                # 等待当前页的所有线程任务执行完毕，再翻页
                for future in as_completed(futures):
                    total_count += future.result()
                
                print(f"--- 第 {page} 页处理完毕 ---")
                page += 1
                time.sleep(1) # 翻页间歇，对服务器温柔一点

    print("\n==============================")
    print(f"所有任务处理完成！本次共新抓取 {total_count} 条详情数据。")
    print(f"缺失的缩略图已同步至 {IMG_DIR} 文件夹。")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，已手动停止。")
