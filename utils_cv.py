import io
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from config import CLIP_MODEL_PATH, COVER_SEARCH_TOP_K, IMG_VECTOR_FILE


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def normalize_vectors(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vectors / norms


@st.cache_resource(show_spinner=False)
def load_clip_vector_index(index_path=IMG_VECTOR_FILE):
    resolved_path = resolve_project_path(index_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"未找到封面向量文件: {resolved_path}")

    with resolved_path.open("rb") as fh:
        payload = pickle.load(fh)

    records = payload.get("records", [])
    embeddings = payload.get("embeddings")
    if embeddings is None or len(records) == 0:
        raise ValueError(f"封面向量文件为空或格式不完整: {resolved_path}")

    embeddings = normalize_vectors(embeddings)
    grouped_vectors = {}
    for record, vector in zip(records, embeddings, strict=True):
        item_id = str(record.get("item_id", "")).strip()
        if not item_id:
            continue
        grouped_vectors.setdefault(item_id, []).append(vector)

    if not grouped_vectors:
        raise ValueError(f"封面向量文件中没有可用的 item_id: {resolved_path}")

    item_ids = sorted(grouped_vectors.keys())
    item_embeddings = []
    for item_id in item_ids:
        mean_vector = np.mean(grouped_vectors[item_id], axis=0, dtype=np.float32)
        item_embeddings.append(normalize_vectors(mean_vector)[0])

    matrix = np.vstack(item_embeddings).astype(np.float32)
    id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

    return {
        "index_path": str(resolved_path),
        "model_name": payload.get("model_name", ""),
        "built_at": payload.get("built_at", ""),
        "records": records,
        "item_ids": item_ids,
        "item_embeddings": matrix,
        "id_to_index": id_to_index,
    }


@st.cache_resource(show_spinner=False)
def load_clip_query_engine(model_path=CLIP_MODEL_PATH):
    import torch
    from transformers import AutoProcessor, CLIPModel

    resolved_path = resolve_project_path(model_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"未找到 CLIP 模型目录: {resolved_path}")

    processor = AutoProcessor.from_pretrained(
        str(resolved_path),
        local_files_only=True,
        use_fast=False,
    )
    model = CLIPModel.from_pretrained(
        str(resolved_path),
        local_files_only=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return {
        "processor": processor,
        "model": model,
        "device": device,
        "model_path": str(resolved_path),
    }


def embed_uploaded_image(image_bytes, model_path=CLIP_MODEL_PATH):
    import torch

    if not image_bytes:
        raise ValueError("未提供上传图片内容。")

    engine = load_clip_query_engine(model_path=model_path)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = engine["processor"](images=[image], return_tensors="pt")
    inputs = {key: value.to(engine["device"]) for key, value in inputs.items()}
    with torch.inference_mode():
        features = engine["model"].get_image_features(**inputs)
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
    return features.detach().cpu().numpy().astype(np.float32)[0]


def get_vector_for_item_id(query_item_id, index_path=IMG_VECTOR_FILE):
    normalized_id = str(query_item_id).strip().upper()
    if not normalized_id:
        raise ValueError("请输入有效的条目 ID。")

    index_data = load_clip_vector_index(index_path=index_path)
    matched_index = index_data["id_to_index"].get(normalized_id)
    if matched_index is None:
        raise ValueError(
            f"向量库里没有找到 ID={normalized_id} 的封面向量，请确认该 ID 已被 build 到 {index_data['index_path']}。"
        )
    return index_data["item_embeddings"][matched_index]


def search_similar_cover_items(
    query_item_id="",
    query_image_bytes=None,
    candidate_ids=None,
    top_k=COVER_SEARCH_TOP_K,
    index_path=IMG_VECTOR_FILE,
    model_path=CLIP_MODEL_PATH,
):
    index_data = load_clip_vector_index(index_path=index_path)

    if query_image_bytes:
        query_vector = embed_uploaded_image(query_image_bytes, model_path=model_path)
        query_mode = "upload"
        query_label = "uploaded-image"
    else:
        query_vector = get_vector_for_item_id(query_item_id, index_path=index_path)
        query_mode = "id"
        query_label = str(query_item_id).strip().upper()

    query_vector = normalize_vectors(query_vector)[0]

    if candidate_ids is None:
        candidate_list = index_data["item_ids"]
    else:
        candidate_list = []
        seen = set()
        for item_id in candidate_ids:
            normalized_id = str(item_id).strip().upper()
            if not normalized_id or normalized_id in seen:
                continue
            seen.add(normalized_id)
            if normalized_id in index_data["id_to_index"]:
                candidate_list.append(normalized_id)

    if not candidate_list:
        return {
            "query_mode": query_mode,
            "query_label": query_label,
            "index_path": index_data["index_path"],
            "results": [],
        }

    candidate_indices = np.array(
        [index_data["id_to_index"][item_id] for item_id in candidate_list],
        dtype=np.int32,
    )
    candidate_matrix = index_data["item_embeddings"][candidate_indices]
    scores = candidate_matrix @ query_vector

    top_k = max(1, min(int(top_k), len(candidate_list)))
    ranked_local = np.argpartition(scores, len(scores) - top_k)[-top_k:]
    ranked_local = ranked_local[np.argsort(scores[ranked_local])[::-1]]

    results = []
    for rank, local_idx in enumerate(ranked_local, start=1):
        item_id = candidate_list[int(local_idx)]
        results.append(
            {
                "rank": rank,
                "item_id": item_id,
                "score": float(scores[int(local_idx)] * 100.0),
            }
        )

    return {
        "query_mode": query_mode,
        "query_label": query_label,
        "index_path": index_data["index_path"],
        "results": results,
    }
