import os
import re
import csv
import time
from curl_cffi import requests 
from bs4 import BeautifulSoup

PROXIES = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890"
}

def extract_field_data(soup, field_name):
    """
    通用信息提取函数：
    根据传入的 field_name (如 'Tags:', 'Artists:') 寻找对应的标签组，
    并提取出所有 <span class="name"> 中的文本内容，用逗号拼接。
    """
    containers = soup.find_all('div', class_='tag-container field-name')
    for container in containers:
        if field_name in container.text:
            names = [span.text.strip() for span in container.find_all('span', class_='name')]
            return ", ".join(names)
    return ""

def extract_upload_date(soup):
    """
    专门用于提取上传日期的函数，精确到日
    """
    containers = soup.find_all('div', class_='tag-container field-name')
    for container in containers:
        if 'Uploaded:' in container.text:
            time_tag = container.find('time')
            if time_tag and time_tag.has_attr('datetime'):
                # datetime_str 格式如 "2022-09-07T11:12:24.000Z"
                datetime_str = time_tag['datetime']
                # 截取前10位 YYYY-MM-DD
                return datetime_str[:10]
    return ""

def get_gallery_info(url, retries=3):
    """请求网页并解析所需的漫画信息"""
    print(f"正在获取: {url}")
    
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url,
                impersonate="chrome120",
                proxies=PROXIES,
                timeout=30
            )
            
            if response.status_code == 404:
                print(f"页面不存在 (404): {url}")
                return None
                
            if response.status_code != 200:
                raise ValueError(f"HTTP状态码异常: {response.status_code}")
                
            # 解析 HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取标题
            title_tag = soup.find('h2', class_='title')
            title = title_tag.text.strip() if title_tag else ""
            
            # 提取各类 Tag 标签
            tags = extract_field_data(soup, 'Tags:')
            artists = extract_field_data(soup, 'Artists:')
            groups = extract_field_data(soup, 'Groups:')
            languages = extract_field_data(soup, 'Languages:')
            pages = extract_field_data(soup, 'Pages:')
            
            # 提取上传日期
            uploaded_date = extract_upload_date(soup)
            
            return {
                "url": url,
                "title": title,
                "tags": tags,
                "artists": artists,
                "groups": groups,
                "languages": languages,
                "pages": pages,
                "uploaded_date": uploaded_date
            }
            
        except Exception as e:
            print(f"请求失败 (尝试 {attempt}/{retries}): {str(e)}")
            if attempt < retries:
                time.sleep(2 ** attempt)  # 指数退避重试
            else:
                return None

def main():
    input_file = 'data/local_data/all.txt'
    output_csv = 'gallery_info_local.csv'
    error_log = 'error_log_local.txt'
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"未找到 {input_file} 文件！")
        return

    # 读取包含链接的txt文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # 用正则提取所有 A HREF 中的链接
    pattern = r'<A HREF="(https://nhentai\.net/g/\d+/)"'
    urls = []
    for line in lines:
        match = re.search(pattern, line)
        if match:
            urls.append(match.group(1))
            
    print(f"共提取到 {len(urls)} 个图库链接。开始处理...")
    
    # CSV 表头
    csv_headers = ['链接', '标题', '标签', '作者', '团队', '语言', '页数', '上传日期']
    
    # 是新文件则写入表头（使用 utf-8-sig 防止 Excel 乱码）
    write_header = not os.path.exists(output_csv)
    with open(output_csv, 'a', newline='', encoding='utf-8-sig') as f_csv:
        writer = csv.writer(f_csv)
        if write_header:
            writer.writerow(csv_headers)
            
        for index, url in enumerate(urls, 1):
            print(f"\n[{index}/{len(urls)}] {'-'*40}")
            
            info = get_gallery_info(url)
            
            if info:
                # 写入到 CSV
                writer.writerow([
                    info['url'], 
                    info['title'], 
                    info['tags'], 
                    info['artists'], 
                    info['groups'], 
                    info['languages'], 
                    info['pages'],
                    info['uploaded_date']
                ])
                # 强制刷新缓冲区，防止意外中断导致数据丢失
                f_csv.flush()
                print(f"成功解析并保存: {info['title'][:30]}...")
            else:
                print(f"解析失败，已记录到 {error_log}")
                with open(error_log, 'a', encoding='utf-8') as f_err:
                    f_err.write(f"解析失败: {url}\n")
            
            time.sleep(2)

    print("\n==============================")
    print(f"所有任务处理完成！数据已保存至 {output_csv}")

if __name__ == "__main__":
    main()