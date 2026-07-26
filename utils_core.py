import os
import re
import glob
import base64
import difflib
import threading
import time
import pandas as pd
from PIL import Image
import streamlit as st
from config import BASE_DIR, B64_CACHE_DIR, ONLINE_IMG_DIR, IMG_CACHE_DIR
from utils_online_cover import fetch_and_cache_online_cover

# onlineimgtmp 可达数十万文件，逐行 glob 等于每行全目录扫描（实测 0.19s/次）；
# 改为一次扫描建索引并短 TTL 复用。新抓的封面同时落 b64_cache（优先级更高），
# 因此索引短暂过期不影响显示。
_ONLINE_IMG_INDEX_TTL = 30.0
_online_img_index = None
_online_img_index_built_at = 0.0
_online_img_index_lock = threading.Lock()


def _get_online_img_index():
    global _online_img_index, _online_img_index_built_at
    now = time.monotonic()
    with _online_img_index_lock:
        if _online_img_index is not None and now - _online_img_index_built_at < _ONLINE_IMG_INDEX_TTL:
            return _online_img_index
    index = {}
    try:
        for entry in os.scandir(ONLINE_IMG_DIR):
            if entry.is_file():
                stem = os.path.splitext(entry.name)[0]
                index.setdefault(stem.upper(), entry.path)
    except OSError:
        pass
    with _online_img_index_lock:
        _online_img_index = index
        _online_img_index_built_at = now
    return index

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

def resolve_gallery_id(gallery_id="", url=""):
    if pd.notna(gallery_id) and str(gallery_id).strip():
        return str(gallery_id).strip()

    if pd.notna(url) and str(url).strip():
        url_str = str(url).strip()
        nh_match = re.search(r'/g/(\d+)/?', url_str)
        if nh_match:
            return f"NH{nh_match.group(1)}"

        jm_match = re.search(r'/album/(\d+)/?', url_str)
        if jm_match:
            return f"JM{jm_match.group(1)}"

    return None


def _local_folder_cover_b64(local_path, gallery_id):
    cache_file = os.path.join(IMG_CACHE_DIR, f"{gallery_id}.jpg")

    if not os.path.exists(cache_file):
        if local_path == "本地目录不存在" or not isinstance(local_path, str) or not os.path.exists(local_path):
            return None

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
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None


def get_cover_base64(local_path, gallery_id="", url="", allow_online=True):
    gallery_id = resolve_gallery_id(gallery_id, url)
            
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
    target_img = _get_online_img_index().get(str(gallery_id).upper())

    if target_img:
        try:
            with open(target_img, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            ext = target_img.split('.')[-1].lower()
            mime = f"image/{ext}" if ext in ['png', 'webp', 'gif'] else "image/jpeg"
            full_b64_string = f"data:{mime};base64,{encoded}"
        except Exception:
            pass

    if not full_b64_string:
        full_b64_string = _local_folder_cover_b64(local_path, gallery_id)

    if not full_b64_string:
        if allow_online:
            return fetch_and_cache_online_cover(gallery_id, wait_if_in_flight=True)
        return None

    if full_b64_string:
        try:
            with open(b64_file_path, "w", encoding="utf-8") as f:
                f.write(full_b64_string)
        except Exception as e:
            print(f"写入 Base64 缓存失败: {e}")
            
    return full_b64_string
