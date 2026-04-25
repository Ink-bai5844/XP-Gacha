import os
import re
import glob
import base64
import difflib
import pandas as pd
from PIL import Image
import streamlit as st
from config import BASE_DIR, B64_CACHE_DIR, ONLINE_IMG_DIR, IMG_CACHE_DIR

def sanitize_folder_name(name):
    if not isinstance(name, str):
        return ""
    illegal_chars = r'[\\/*?:"<>|]'
    return re.sub(illegal_chars, '_', name)

@st.cache_data(max_entries=1)
def get_local_folders():
    folder_map = {}
    if os.path.exists(BASE_DIR):
        for root, dirs, files in os.walk(BASE_DIR):
            for d in dirs:
                folder_map[d] = os.path.join(root, d)
    return folder_map

def match_local_folder(csv_filename, folder_map):
    if not csv_filename or not folder_map:
        return "本地目录不存在"
        
    sanitized_name = sanitize_folder_name(csv_filename)
    if sanitized_name in folder_map:
        return folder_map[sanitized_name]
        
    folder_names = list(folder_map.keys())
    matches = difflib.get_close_matches(sanitized_name, folder_names, n=1, cutoff=0.6)
    if matches:
        return folder_map[matches[0]]
        
    return "本地目录不存在"

def get_cover_base64(local_path, url=""):
    gallery_id = None
    if pd.notna(url) and str(url).strip():
        match = re.search(r'/g/(\d+)/?', str(url))
        if match:
            gallery_id = match.group(1)
            
    if not gallery_id:
        return None

    b64_file_path = os.path.join(B64_CACHE_DIR, f"{gallery_id}.txt")
    if os.path.exists(b64_file_path):
        try:
            with open(b64_file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass

    full_b64_string = None
    online_img_pattern = os.path.join(ONLINE_IMG_DIR, f"{gallery_id}.*")
    matched_online_imgs = glob.glob(online_img_pattern)
    
    if matched_online_imgs:
        try:
            target_img = matched_online_imgs[0]
            with open(target_img, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            ext = target_img.split('.')[-1].lower()
            mime = f"image/{ext}" if ext in ['png', 'webp', 'gif'] else "image/jpeg"
            full_b64_string = f"data:{mime};base64,{encoded}"
        except Exception:
            pass

    if not full_b64_string:
        if local_path == "本地目录不存在" or not isinstance(local_path, str) or not os.path.exists(local_path):
            return None
            
        cache_file = os.path.join(IMG_CACHE_DIR, f"{gallery_id}.jpg")
        
        if not os.path.exists(cache_file):
            escaped_path = glob.escape(local_path)
            search_pattern = os.path.join(escaped_path, "1.*")
            matched_files = glob.glob(search_pattern)
            
            valid_files = [f for f in matched_files if os.path.isfile(f)]
            if not valid_files:
                return None
                
            target_file = valid_files[0]
            try:
                with Image.open(target_file) as img:
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    img.thumbnail((150, 200)) 
                    img.save(cache_file, format="JPEG", quality=85)
            except Exception:
                return None 
                
        try:
            with open(cache_file, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            full_b64_string = f"data:image/jpeg;base64,{encoded}"
        except Exception:
            return None

    if full_b64_string:
        try:
            with open(b64_file_path, "w", encoding="utf-8") as f:
                f.write(full_b64_string)
        except Exception as e:
            print(f"写入 Base64 缓存失败: {e}")
            
    return full_b64_string