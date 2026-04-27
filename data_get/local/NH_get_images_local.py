import os
import re
from curl_cffi import requests 
from bs4 import BeautifulSoup
import time

PROXIES = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890"
}

def sanitize_folder_name(name):
    """移除文件夹名中的非法字符"""
    illegal_chars = r'[\\/*?:"<>|]'
    return re.sub(illegal_chars, '_', name)

def log_error(folder_name, page_num, error_msg, error_file="return.txt"):
    """记录错误到日志文件"""
    with open(error_file, 'a', encoding='utf-8') as f:
        log_line = f"{folder_name} {page_num} {error_msg}\n"
        f.write(log_line)
    print(f"已记录错误: {log_line.strip()}")

def download_image(image_url, save_path, referer, retries=3):
    """下载图片并保存到指定路径，支持重试"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': referer
    }
    
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                image_url, 
                headers=headers, 
                impersonate="chrome120", 
                proxies=PROXIES,
                timeout=30
            )
            
            if response.status_code != 200:
                raise ValueError(f"HTTP状态码异常: {response.status_code}")
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        except Exception as e:
            error_msg = f"下载失败 (尝试 {attempt}/{retries}): {str(e)}"
            print(error_msg)
            if attempt < retries:
                wait_time = 2 ** attempt  
                print(f"等待 {wait_time}秒后重试...")
                time.sleep(wait_time)
    
    return False

def main():
    # 创建根输出目录
    root_dir = "output"
    os.makedirs(root_dir, exist_ok=True)
    
    # 初始化错误日志
    error_log = "logs/NH_error_log_images_local.txt"
    if os.path.exists(error_log):
        os.remove(error_log)
    
    # 读取2.txt文件
    with open('data/local_data/NH_2.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # 正则表达式解析每行内容
    pattern = r'<A HREF="(.*?)".*?>(.*?)</A>'
    
    for line in lines:
        match = re.search(pattern, line)
        if not match:
            continue
            
        gallery_url = match.group(1).strip()
        folder_name = sanitize_folder_name(match.group(2).strip())
        folder_path = os.path.join(root_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"开始处理: {folder_name}")
        print(f"图库链接: {gallery_url}")
        
        page_num = 1
        gallery_completed = False  # 标记图库是否完成
        
        while not gallery_completed:
            page_url = f"{gallery_url}{page_num}/"
            print(f"\n尝试页面 #{page_num}: {page_url}")
            
            try:
                # 获取图片页面
                page_found = False
                for retry in range(1, 4):  # 最多重试3次
                    try:
                        # 添加 impersonate 参数
                        response = requests.get(page_url, impersonate="chrome120", proxies=PROXIES, timeout=30)
                        
                        if response.status_code == 404:
                            print(f"页面不存在 (404): {page_url}")
                            gallery_completed = True  
                            break
                        
                        if response.status_code != 200:
                            raise requests.exceptions.HTTPError(f"HTTP {response.status_code}")
                            
                        page_found = True
                        break
                    except Exception as e:
                        if retry < 3:
                            print(f"请求页面失败 (尝试 {retry}/3): {str(e)}")
                            time.sleep(3 ** retry)  # 指数退避等待
                        else:
                            raise
                
                # 如果检测到图库结束，跳过后续处理
                if gallery_completed:
                    print(f"检测到页面不存在，停止处理: {folder_name}")
                    break
                
                # 如果页面请求失败（非404），记录错误并继续下一页
                if not page_found:
                    error_msg = "页面请求失败"
                    print(f"处理页面时出错: {error_msg}")
                    log_error(folder_name, page_num, error_msg, error_log)
                    page_num += 1
                    continue
                
                # 解析图片URL
                soup = BeautifulSoup(response.text, 'html.parser')
                img_container = soup.find('section', id='image-container')
                if not img_container:
                    raise Exception("未找到图片容器")
                
                img_tag = img_container.find('img')
                if not img_tag or 'src' not in img_tag.attrs:
                    raise Exception("未找到图片标签")
                
                # 获取实际图片URL
                img_url = img_tag['src']
                ext = os.path.splitext(img_url)[1].split('?')[0]  # 处理带参数的URL
                save_path = os.path.join(folder_path, f"{page_num}{ext}")
                
                # 下载图片
                print(f"下载图片 #{page_num}: {img_url}")
                if download_image(img_url, save_path, page_url):
                    print(f"图片下载成功: {save_path}")
                else:
                    raise Exception("图片下载失败")
                    
                page_num += 1
                time.sleep(1.5)  # 增加延时避免触发并发限制
                
            except Exception as e:
                error_msg = str(e)
                print(f"处理页面时出错: {error_msg}")
                
                # 记录错误到日志
                log_error(folder_name, page_num, error_msg, error_log)
                page_num += 1
                
                # 安全限制
                if page_num > 200:  
                    print(f"达到安全页码限制，停止处理: {folder_name}")
                    break

    print("\n所有任务处理完成！")
    if os.path.exists(error_log):
        print(f"错误日志已保存到: {error_log}")

if __name__ == "__main__":
    main()