"""线上封面实时抓取：b64_cache、onlineimgtmp、本地目录都取不到封面时的最后兜底。

NH 源：缩略图 URL 使用 media_id 而不是画廊 ID，因此先经
``https://nhentai.net/api/v2/galleries/<画廊ID>`` 换取 media_id（顺便得到真实扩展名，
旧版 ``/api/gallery/`` 接口作为回退），再按
``https://t{1~5}.nhentai.net/galleries/<media_id>/thumb.<ext>`` 逐个组合尝试。
JM 源：专辑 ID 即 JM ID 本身，直接按
``https://cdn-msp{,1~5}.18comic.vip/media/albums/<专辑ID>.<ext>`` 逐个组合尝试。

找到第一个可访问的组合后原子写入 onlineimgtmp 与 b64_cache。失败分两类：
确定性失败（404、内容非法等）的 ID 本次会话内不再重试；网络性失败（超时/连接失败）
不拉黑 ID，网络恢复后可重试。每个图源各自独立熔断：连续多次网络性失败后暂停该源
一段时间再自动恢复。代理与直连每次请求内互为回退，成功的一侧成为下次的首选。
"""

import atexit
import base64
import io
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from config import B64_CACHE_DIR, ONLINE_IMG_DIR

try:
    from config import ONLINE_COVER_FETCH_ENABLED
except ImportError:
    ONLINE_COVER_FETCH_ENABLED = True

try:
    from config import ONLINE_COVER_PROXY
except ImportError:
    ONLINE_COVER_PROXY = ""

try:
    from config import ONLINE_COVER_FETCH_CONCURRENCY
except ImportError:
    ONLINE_COVER_FETCH_CONCURRENCY = 6

_REQUEST_TIMEOUT = 5
_MAX_CONSECUTIVE_NETWORK_FAILURES = 5
_SOURCE_COOLDOWN_SECONDS = 600
_MAX_IMAGE_BYTES = 1_500_000

_NH_HOSTS = [f"t{i}.nhentai.net" for i in (1, 2, 3, 4, 5)]
_NH_EXT_BY_TYPE = {"j": "jpg", "p": "png", "w": "webp", "g": "gif"}
_NH_FALLBACK_EXTS = ["webp", "jpg", "png"]
_JM_HOSTS = ["cdn-msp"] + [f"cdn-msp{i}" for i in (1, 2, 3, 4, 5)]
_JM_EXTS = ["jpg", "webp", "png"]

_MISS_RETRY_COOLDOWN_SECONDS = 120

_state_lock = threading.Lock()
_failed_ids = set()
_in_flight = set()
_queued = set()
_retry_after = {}
_nh_media_cache = {}
_prefer_proxy = bool(ONLINE_COVER_PROXY)
_source_failures = {"nh": 0, "jm": 0}
_source_disabled_until = {"nh": 0.0, "jm": 0.0}

_transport = None
_executor = None


def _get_transport():
    global _transport
    if _transport is None:
        try:
            from curl_cffi import requests as curl_requests
            _transport = ("curl_cffi", curl_requests)
        except ImportError:
            import requests as plain_requests
            _transport = ("requests", plain_requests)
    return _transport


def _do_get(url, headers, proxies):
    kind, req = _get_transport()
    kwargs = {"headers": headers, "proxies": proxies, "timeout": _REQUEST_TIMEOUT}
    if kind == "curl_cffi":
        kwargs["impersonate"] = "chrome120"
    return req.get(url, **kwargs)


def _note_network_failure(source):
    with _state_lock:
        _source_failures[source] += 1
        if _source_failures[source] >= _MAX_CONSECUTIVE_NETWORK_FAILURES:
            _source_disabled_until[source] = time.monotonic() + _SOURCE_COOLDOWN_SECONDS
            _source_failures[source] = 0


def _is_source_disabled(source):
    with _state_lock:
        return time.monotonic() < _source_disabled_until[source]


def _http_get(url, headers, source):
    """代理与直连互为回退的 GET：成功的一侧成为下次首选；两侧都失败才计一次网络失败。"""
    global _prefer_proxy
    proxy_cfg = None
    if ONLINE_COVER_PROXY:
        proxy_cfg = {"http": ONLINE_COVER_PROXY, "https": ONLINE_COVER_PROXY}

    if proxy_cfg is None:
        modes = [None]
    elif _prefer_proxy:
        modes = [proxy_cfg, None]
    else:
        modes = [None, proxy_cfg]

    for proxies in modes:
        try:
            resp = _do_get(url, headers, proxies)
        except Exception:
            continue
        with _state_lock:
            _prefer_proxy = proxies is not None
            _source_failures[source] = 0
        return resp

    _note_network_failure(source)
    return None


def online_cover_fetch_available():
    if not ONLINE_COVER_FETCH_ENABLED:
        return False
    return not (_is_source_disabled("nh") and _is_source_disabled("jm"))


def _normalize_gallery_id(gallery_id):
    """返回 (前缀, 数字) 如 ("NH", "123456")；格式不合法返回 (None, None)。"""
    gid = str(gallery_id).strip() if gallery_id is not None else ""
    match = re.fullmatch(r"(?i)(NH|JM)(\d+)", gid)
    if not match:
        return None, None
    return match.group(1).upper(), match.group(2)


def _sniff_image_ext(content):
    if not content or len(content) < 12:
        return None
    if content[:2] == b"\xff\xd8":
        return "jpg"
    if content[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "webp"
    if content[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return None


def _resolve_nh_thumb_info(numeric_id):
    """画廊 ID -> ((media_id, 缩略图扩展名或 None), 是否确定性结果)。

    只有拿到服务器响应（确定性）时才缓存结论；纯网络失败不缓存，恢复后可重试。
    """
    with _state_lock:
        if numeric_id in _nh_media_cache:
            return _nh_media_cache[numeric_id], True

    info = None
    definitive = False
    api_urls = (
        f"https://nhentai.net/api/v2/galleries/{numeric_id}",
        f"https://nhentai.net/api/gallery/{numeric_id}",
    )
    for api_url in api_urls:
        resp = _http_get(api_url, headers={"Referer": "https://nhentai.net/"}, source="nh")
        if resp is None:
            continue
        if resp.status_code == 404:
            # 画廊确实不存在才算确定性结果；429/403/5xx 等临时状态不缓存、不拉黑
            definitive = True
            continue
        if resp.status_code != 200:
            continue
        try:
            data = resp.json()
            media_id = str(data.get("media_id", "")).strip()
            thumb_ext = None
            thumbnail = data.get("thumbnail")
            if isinstance(thumbnail, dict):
                # API v2：thumbnail.path 形如 galleries/<media_id>/thumb.webp
                ext_match = re.search(r"\.(\w+)$", str(thumbnail.get("path", "")))
                if ext_match and ext_match.group(1).lower() in ("jpg", "png", "webp", "gif"):
                    thumb_ext = ext_match.group(1).lower()
            if thumb_ext is None:
                # 旧版 API：images.thumbnail.t 为类型码 j/p/w/g
                images = data.get("images")
                if isinstance(images, dict):
                    thumb_ext = _NH_EXT_BY_TYPE.get((images.get("thumbnail") or {}).get("t"))
            if media_id.isdigit():
                info = (media_id, thumb_ext)
                definitive = True
                break
        except Exception:
            continue

    if definitive:
        with _state_lock:
            _nh_media_cache[numeric_id] = info
    return info, definitive


def _try_fetch(urls_by_ext, headers, source):
    """按扩展名分组尝试，返回 ((字节, 扩展名) 或 None, 是否确定性结果)。

    404 说明该路径不存在（换扩展名），连接失败/封锁则换镜像。
    """
    definitive = False
    for _ext, urls in urls_by_ext:
        for url in urls:
            if _is_source_disabled(source):
                return None, definitive
            resp = _http_get(url, headers, source)
            if resp is None:
                continue
            if resp.status_code == 404:
                # 路径确实不存在才算确定性结果；429/403/5xx 或非图片响应视为临时故障
                definitive = True
                break
            if resp.status_code != 200:
                continue
            content = resp.content
            real_ext = _sniff_image_ext(content)
            if real_ext:
                return (content, real_ext), True
    return None, definitive


def _fetch_online_cover(gallery_id):
    """返回 ((图片字节, 扩展名) 或 None, 是否确定性失败可拉黑)。"""
    prefix, digits = _normalize_gallery_id(gallery_id)
    if prefix is None:
        return None, True

    if prefix == "NH":
        if _is_source_disabled("nh"):
            return None, False
        info, definitive = _resolve_nh_thumb_info(digits)
        if not info:
            return None, definitive
        media_id, known_ext = info
        exts = list(_NH_FALLBACK_EXTS)
        if known_ext:
            exts = [known_ext] + [e for e in exts if e != known_ext]
        urls_by_ext = [
            (ext, [f"https://{host}/galleries/{media_id}/thumb.{ext}" for host in _NH_HOSTS])
            for ext in exts
        ]
        return _try_fetch(urls_by_ext, headers={"Referer": "https://nhentai.net/"}, source="nh")

    if _is_source_disabled("jm"):
        return None, False
    urls_by_ext = [
        (ext, [f"https://{host}.18comic.vip/media/albums/{digits}.{ext}" for host in _JM_HOSTS])
        for ext in _JM_EXTS
    ]
    return _try_fetch(urls_by_ext, headers={"Referer": "https://18comic.vip/"}, source="jm")


def fetch_online_cover(gallery_id):
    """实时抓取封面，返回 (图片字节, 扩展名)；失败返回 None。"""
    result, _definitive = _fetch_online_cover(gallery_id)
    return result


def _shrink_if_needed(content, ext):
    if len(content) <= _MAX_IMAGE_BYTES:
        return content, ext
    try:
        from PIL import Image
        with Image.open(io.BytesIO(content)) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.thumbnail((450, 600))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue(), "jpg"
    except Exception:
        return content, ext


def _atomic_write(path, data, binary=False):
    tmp_path = f"{path}.tmp-{os.getpid()}-{threading.get_ident()}"
    try:
        if binary:
            with open(tmp_path, "wb") as f:
                f.write(data)
        else:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(data)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _read_cached_b64(b64_file_path):
    if not os.path.exists(b64_file_path):
        return None
    try:
        with open(b64_file_path, "r", encoding="utf-8") as f:
            cached = f.read()
        if cached.startswith("data:image/"):
            return cached
    except Exception:
        pass
    return None


def has_cached_cover(gallery_id):
    """该 ID 是否已有 b64 缓存文件。"""
    prefix, digits = _normalize_gallery_id(gallery_id)
    if prefix is None:
        return False
    return os.path.exists(os.path.join(B64_CACHE_DIR, f"{prefix}{digits}.txt"))


def _wait_for_in_flight(canonical_id, timeout=15.0):
    """等待其他线程完成同一 ID 的抓取，返回它落盘的结果（或 None）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.2)
        with _state_lock:
            if canonical_id not in _in_flight and canonical_id not in _queued:
                break
    return _read_cached_b64(os.path.join(B64_CACHE_DIR, f"{canonical_id}.txt"))


def fetch_and_cache_online_cover(gallery_id, wait_if_in_flight=False):
    """实时抓取封面并写入 onlineimgtmp 与 b64_cache，返回 data URI；失败返回 None。

    同一 ID 已在其他线程抓取中时：默认直接返回 None（后台任务互相让路），
    wait_if_in_flight=True 则等待其完成并返回落盘结果（详情页单条即时加载用）。
    """
    if not ONLINE_COVER_FETCH_ENABLED:
        return None
    prefix, digits = _normalize_gallery_id(gallery_id)
    if prefix is None:
        return None
    canonical_id = prefix + digits

    with _state_lock:
        if canonical_id in _failed_ids:
            return None
        already_in_flight = canonical_id in _in_flight
        if not already_in_flight:
            _in_flight.add(canonical_id)

    if already_in_flight:
        if wait_if_in_flight:
            return _wait_for_in_flight(canonical_id)
        return None

    try:
        b64_file_path = os.path.join(B64_CACHE_DIR, f"{canonical_id}.txt")
        cached = _read_cached_b64(b64_file_path)
        if cached:
            return cached

        result, definitive = _fetch_online_cover(canonical_id)
        if not result:
            if definitive:
                with _state_lock:
                    _failed_ids.add(canonical_id)
                    _retry_after.pop(canonical_id, None)
            return None

        content, ext = _shrink_if_needed(*result)

        try:
            os.makedirs(ONLINE_IMG_DIR, exist_ok=True)
            _atomic_write(os.path.join(ONLINE_IMG_DIR, f"{canonical_id}.{ext}"), content, binary=True)
        except Exception:
            pass

        mime = f"image/{ext}" if ext in ("png", "webp", "gif") else "image/jpeg"
        full_b64_string = f"data:{mime};base64,{base64.b64encode(content).decode('utf-8')}"
        try:
            os.makedirs(B64_CACHE_DIR, exist_ok=True)
            _atomic_write(b64_file_path, full_b64_string)
        except Exception as e:
            print(f"写入 Base64 缓存失败: {e}")
        with _state_lock:
            _retry_after.pop(canonical_id, None)
        return full_b64_string
    finally:
        with _state_lock:
            _in_flight.discard(canonical_id)


def _shutdown_executor():
    if _executor is not None:
        # 丢弃排队任务，只等当前在抓的几条收尾，避免 Ctrl+C 卡到队列跑完
        _executor.shutdown(wait=False, cancel_futures=True)


def _get_executor():
    global _executor
    with _state_lock:
        if _executor is None:
            try:
                workers = max(1, int(ONLINE_COVER_FETCH_CONCURRENCY))
            except (TypeError, ValueError):
                workers = 6
            _executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="online-cover")
            atexit.register(_shutdown_executor)
        return _executor


def _run_queued_fetch(canonical_id):
    try:
        result = fetch_and_cache_online_cover(canonical_id)
        if result is None:
            # 网络性失败（未拉黑）短冷却，避免轮询刷新触发无限重试
            with _state_lock:
                if canonical_id not in _failed_ids:
                    _retry_after[canonical_id] = time.monotonic() + _MISS_RETRY_COOLDOWN_SECONDS
    finally:
        with _state_lock:
            _queued.discard(canonical_id)


def submit_online_cover_fetches(gallery_ids):
    """异步提交线上封面抓取到常驻线程池（并发 ONLINE_COVER_FETCH_CONCURRENCY），立即返回。

    已拉黑、抓取中、排队中或冷却期内的 ID 自动跳过；返回实际提交数量。
    抓取结果由 fetch_and_cache_online_cover 落盘到 b64_cache，由调用方轮询感知。
    """
    if not ONLINE_COVER_FETCH_ENABLED:
        return 0
    submitted = 0
    now = time.monotonic()
    for gallery_id in gallery_ids:
        prefix, digits = _normalize_gallery_id(gallery_id)
        if prefix is None:
            continue
        canonical_id = prefix + digits
        with _state_lock:
            if canonical_id in _failed_ids or canonical_id in _queued or canonical_id in _in_flight:
                continue
            if now < _retry_after.get(canonical_id, 0.0):
                continue
            _queued.add(canonical_id)
        try:
            _get_executor().submit(_run_queued_fetch, canonical_id)
        except Exception:
            with _state_lock:
                _queued.discard(canonical_id)
            break
        submitted += 1
    return submitted


def is_online_cover_pending(gallery_id):
    """该 ID 是否仍在排队或抓取中（结束后无论成败都返回 False）。"""
    prefix, digits = _normalize_gallery_id(gallery_id)
    if prefix is None:
        return False
    canonical_id = prefix + digits
    with _state_lock:
        return canonical_id in _queued or canonical_id in _in_flight
