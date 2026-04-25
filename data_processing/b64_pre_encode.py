import os
import glob
import base64

ONLINE_IMG_DIR = "onlineimgtmp"
IMG_CACHE_DIR = "localimgtmp"
B64_CACHE_DIR = "b64_cache"
B64_TMP_DIR = "b64_tmp"

if not os.path.exists(B64_CACHE_DIR):
    os.makedirs(B64_CACHE_DIR)
if not os.path.exists(B64_TMP_DIR):
    os.makedirs(B64_TMP_DIR)

def process_directory(directory):
    if not os.path.exists(directory):
        return

    # 匹配常见图片格式
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.webp', '*.gif')
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))

    count = 0
    for img_path in image_files:
        # 提取画廊 ID (假设文件名就是 ID.后缀)
        basename = os.path.basename(img_path)
        gallery_id = os.path.splitext(basename)[0]
        
        merged_b64_file_path = os.path.join(B64_CACHE_DIR, f"{gallery_id}.txt")
        tmp_b64_file_path = os.path.join(B64_TMP_DIR, f"{gallery_id}.txt")
        
        # 已合并到主缓存，或已存在于增量目录，都跳过
        if os.path.exists(merged_b64_file_path) or os.path.exists(tmp_b64_file_path):
            continue
            
        try:
            with open(img_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            
            ext = basename.split('.')[-1].lower()
            mime = f"image/{ext}" if ext in ['png', 'webp', 'gif'] else "image/jpeg"
            full_b64_string = f"data:{mime};base64,{encoded}"
            
            # 增量部分单独写入临时目录，后续手动合并
            with open(tmp_b64_file_path, "w", encoding="utf-8") as f:
                f.write(full_b64_string)
            
            count += 1
            if count % 100 == 0:
                print(f"已处理 {count} 张图片...")
                
        except Exception as e:
            print(f"处理 {img_path} 时出错: {e}")

    print(f"[{directory}] 目录处理完成，新增 {count} 个 Base64 缓存。")

if __name__ == "__main__":
    print("开始进行 Base64 预缓存...")
    process_directory(ONLINE_IMG_DIR)
    process_directory(IMG_CACHE_DIR)
    print("全部缓存完毕！")
