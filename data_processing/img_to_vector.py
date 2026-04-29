from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, CLIPModel


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIRS = [
    PROJECT_ROOT / "onlineimgtmp",
    PROJECT_ROOT / "localimgtmp",
]
DEFAULT_INDEX_PATH = PROJECT_ROOT / "datacache" / "clip_image_index.pkl"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "clip-vit-base-patch32"
DEFAULT_MODEL_NAME = str(DEFAULT_MODEL_DIR)
PROGRESS_VERSION = 1
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class ImageRecord:
    path: str
    source: str
    name: str
    item_id: str
    size: int
    mtime_ns: int


class BuildInterruptedError(RuntimeError):
    pass


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def denormalize_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def collect_image_records(image_dirs: Sequence[Path]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for image_dir in image_dirs:
        if not image_dir.exists():
            continue
        source = image_dir.name
        for path in sorted(image_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            stat = path.stat()
            records.append(
                ImageRecord(
                    path=normalize_path(path),
                    source=source,
                    name=path.name,
                    item_id=path.stem,
                    size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                )
            )
    return records


def chunked(items: Sequence[ImageRecord], size: int) -> Iterable[Sequence[ImageRecord]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


class ClipImageSearcher:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        index_path: Path = DEFAULT_INDEX_PATH,
        image_dirs: Sequence[Path] | None = None,
        batch_size: int = 64,
        device: str = "auto",
        local_files_only: bool = True,
    ) -> None:
        self.model_name = model_name
        self.index_path = Path(index_path)
        self.image_dirs = list(image_dirs or DEFAULT_IMAGE_DIRS)
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self.device = self._resolve_device(device)
        self._processor: AutoProcessor | None = None
        self._model: CLIPModel | None = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_model(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        model_path = Path(self.model_name)
        if not model_path.exists():
            raise RuntimeError(
                f"Local CLIP model directory not found: {model_path}. "
                f"Expected files under {DEFAULT_MODEL_DIR}."
            )
        try:
            self._processor = AutoProcessor.from_pretrained(
                str(model_path),
                local_files_only=True,
            )
            self._model = CLIPModel.from_pretrained(
                str(model_path),
                local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the local CLIP model directory. "
                f"model_dir={model_path}"
            ) from exc
        self._model.to(self.device)
        self._model.eval()

    def _embed_pil_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        self._ensure_model()
        assert self._processor is not None
        assert self._model is not None

        inputs = self._processor(images=list(images), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            image_features = self._model.get_image_features(**inputs)
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
        return image_features.detach().cpu().numpy().astype(np.float16)

    def embed_image_file(self, image_path: str | Path) -> np.ndarray:
        image = load_rgb_image(Path(image_path))
        return self._embed_pil_images([image])[0].astype(np.float32)

    def embed_pil_image(self, image: Image.Image) -> np.ndarray:
        rgb_image = image.convert("RGB")
        return self._embed_pil_images([rgb_image])[0].astype(np.float32)

    def _save_index(self, payload: dict) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        with temp_path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        temp_path.replace(self.index_path)

    def load_index(self) -> dict:
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        with self.index_path.open("rb") as fh:
            payload = pickle.load(fh)
        return payload

    def _progress_dir(self) -> Path:
        return Path(str(self.index_path) + ".progress")

    def _progress_plan_path(self) -> Path:
        return self._progress_dir() / "plan.pkl"

    def _progress_state_path(self) -> Path:
        return self._progress_dir() / "state.pkl"

    def _save_pickle(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        temp_path.replace(path)

    def _load_pickle(self, path: Path) -> dict:
        with path.open("rb") as fh:
            return pickle.load(fh)

    def _chunk_path(self, chunk_index: int) -> Path:
        return self._progress_dir() / f"chunk_{chunk_index:06d}.npz"

    def _cleanup_progress(self) -> None:
        progress_dir = self._progress_dir()
        if progress_dir.exists():
            shutil.rmtree(progress_dir)

    def _build_signature(
        self,
        records: Sequence[ImageRecord],
        rebuild: bool,
        old_payload: dict | None,
    ) -> str:
        hasher = hashlib.sha256()
        hasher.update(str(PROGRESS_VERSION).encode("utf-8"))
        hasher.update(str(Path(self.model_name).resolve()).encode("utf-8"))
        hasher.update(str(rebuild).encode("utf-8"))
        for image_dir in self.image_dirs:
            hasher.update(str(image_dir.resolve()).encode("utf-8"))
        if old_payload is not None:
            hasher.update(str(old_payload.get("built_at", "")).encode("utf-8"))
            hasher.update(str(len(old_payload.get("records", []))).encode("utf-8"))
        for record in records:
            hasher.update(record.path.encode("utf-8"))
            hasher.update(record.source.encode("utf-8"))
            hasher.update(str(record.size).encode("utf-8"))
            hasher.update(str(record.mtime_ns).encode("utf-8"))
        return hasher.hexdigest()

    def _init_or_resume_progress(
        self,
        records: Sequence[ImageRecord],
        rebuild: bool,
        old_payload: dict | None,
    ) -> tuple[dict, dict, bool]:
        progress_dir = self._progress_dir()
        plan_path = self._progress_plan_path()
        state_path = self._progress_state_path()
        signature = self._build_signature(records, rebuild=rebuild, old_payload=old_payload)

        if progress_dir.exists() and plan_path.exists() and state_path.exists():
            plan = self._load_pickle(plan_path)
            state = self._load_pickle(state_path)
            if plan.get("plan_signature") == signature:
                return plan, state, True
            print("[build] existing progress does not match current data, restarting progress")
            self._cleanup_progress()

        progress_dir.mkdir(parents=True, exist_ok=True)
        plan = {
            "version": PROGRESS_VERSION,
            "plan_signature": signature,
            "model_name": self.model_name,
            "image_dirs": [str(path) for path in self.image_dirs],
            "rebuild": rebuild,
            "records": [asdict(record) for record in records],
            "created_at": now_text(),
        }
        state = {
            "next_record_index": 0,
            "chunk_count": 0,
            "skipped_positions": [],
            "updated_at": now_text(),
        }
        self._save_pickle(plan_path, plan)
        self._save_pickle(state_path, state)
        return plan, state, False

    def _save_chunk(self, chunk_index: int, positions: np.ndarray, embeddings: np.ndarray) -> None:
        chunk_path = self._chunk_path(chunk_index)
        temp_path = chunk_path.with_suffix(".npz.tmp")
        with temp_path.open("wb") as fh:
            np.savez(fh, positions=positions, embeddings=embeddings)
        temp_path.replace(chunk_path)

    def _finalize_progress(self, plan: dict) -> dict:
        chunk_paths = sorted(self._progress_dir().glob("chunk_*.npz"))
        if not chunk_paths:
            raise RuntimeError("No progress chunks were saved.")

        all_positions: list[np.ndarray] = []
        all_embeddings: list[np.ndarray] = []
        for chunk_path in chunk_paths:
            with np.load(chunk_path) as data:
                all_positions.append(data["positions"])
                all_embeddings.append(data["embeddings"])

        positions = np.concatenate(all_positions, axis=0)
        embeddings = np.vstack(all_embeddings).astype(np.float16)
        order = np.argsort(positions)
        positions = positions[order]
        embeddings = embeddings[order]

        record_dicts = plan["records"]
        final_records = [record_dicts[int(position)] for position in positions.tolist()]
        payload = {
            "model_name": self.model_name,
            "image_dirs": [str(path) for path in self.image_dirs],
            "built_at": now_text(),
            "records": final_records,
            "embeddings": embeddings,
        }
        self._save_index(payload)
        self._cleanup_progress()
        return payload

    def _build_with_progress(
        self,
        records: Sequence[ImageRecord],
        rebuild: bool,
        old_payload: dict | None,
        unchanged_indices: dict[str, int],
    ) -> dict:
        plan, state, resumed = self._init_or_resume_progress(
            records=records,
            rebuild=rebuild,
            old_payload=old_payload,
        )
        if resumed:
            print(
                f"[build] resuming progress at {state['next_record_index']}/{len(records)}, "
                f"saved_chunks={state['chunk_count']}"
            )
        else:
            print(f"[build] progress started, images={len(records)}")

        old_embeddings = None if old_payload is None else old_payload["embeddings"]
        started_at = time.time()
        next_record_index = int(state["next_record_index"])
        chunk_count = int(state["chunk_count"])
        skipped_positions = set(int(pos) for pos in state.get("skipped_positions", []))
        total_records = len(records)
        total_batches = math.ceil(total_records / self.batch_size)

        try:
            for start in range(next_record_index, total_records, self.batch_size):
                batch_records = records[start : start + self.batch_size]
                batch_positions: list[int] = []
                batch_rows: list[np.ndarray] = []
                pending_images: list[Image.Image] = []
                pending_positions: list[int] = []

                for offset, record in enumerate(batch_records):
                    position = start + offset
                    old_index = unchanged_indices.get(record.path)
                    if old_index is not None:
                        assert old_embeddings is not None
                        batch_positions.append(position)
                        batch_rows.append(old_embeddings[old_index].astype(np.float16))
                        continue

                    try:
                        pending_images.append(load_rgb_image(denormalize_path(record.path)))
                        pending_positions.append(position)
                    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
                        skipped_positions.add(position)
                        print(f"[skip] {record.path} -> {exc}")

                pending_map: dict[int, np.ndarray] = {}
                if pending_images:
                    pending_embeddings = self._embed_pil_images(pending_images)
                    pending_map = {
                        position: embedding
                        for position, embedding in zip(pending_positions, pending_embeddings, strict=True)
                    }

                for offset, _record in enumerate(batch_records):
                    position = start + offset
                    if position in skipped_positions:
                        continue
                    if position in pending_map:
                        batch_positions.append(position)
                        batch_rows.append(pending_map[position])

                if batch_positions:
                    chunk_count += 1
                    self._save_chunk(
                        chunk_count,
                        np.asarray(batch_positions, dtype=np.int32),
                        np.vstack(batch_rows).astype(np.float16),
                    )

                state["next_record_index"] = start + len(batch_records)
                state["chunk_count"] = chunk_count
                state["skipped_positions"] = sorted(skipped_positions)
                state["updated_at"] = now_text()
                self._save_pickle(self._progress_state_path(), state)

                elapsed = time.time() - started_at
                current_batch = (start // self.batch_size) + 1
                print(
                    f"[build] batch {current_batch}/{total_batches}, "
                    f"processed {state['next_record_index']}/{total_records}, "
                    f"saved_chunks={chunk_count}, elapsed {elapsed:.1f}s"
                )
        except KeyboardInterrupt as exc:
            raise BuildInterruptedError(
                "Build interrupted. Progress has been saved and the next `build` will resume."
            ) from exc

        return self._finalize_progress(plan)

    def build_index(self, rebuild: bool = False) -> dict:
        records = collect_image_records(self.image_dirs)
        if not records:
            raise RuntimeError("No images found in the configured directories.")

        if rebuild or not self.index_path.exists():
            print(f"[build] full rebuild, images={len(records)}")
            return self._build_with_progress(
                records=records,
                rebuild=True,
                old_payload=None,
                unchanged_indices={},
            )

        payload = self.load_index()
        if payload.get("model_name") != self.model_name:
            print("[build] model changed, forcing full rebuild")
            return self.build_index(rebuild=True)

        old_records = payload.get("records", [])
        old_embeddings = payload.get("embeddings")
        if old_embeddings is None or len(old_records) != len(old_embeddings):
            print("[build] cache is inconsistent, forcing full rebuild")
            return self.build_index(rebuild=True)

        old_map = {record["path"]: (index, record) for index, record in enumerate(old_records)}

        changed_or_new: list[ImageRecord] = []
        unchanged_indices: dict[str, int] = {}
        deleted_count = 0
        for record in records:
            old_entry = old_map.get(record.path)
            if old_entry is None:
                changed_or_new.append(record)
                continue
            old_index, old_record = old_entry
            if (
                old_record["mtime_ns"] == record.mtime_ns
                and old_record["size"] == record.size
                and old_record["source"] == record.source
            ):
                unchanged_indices[record.path] = old_index
            else:
                changed_or_new.append(record)

        new_path_set = {record.path for record in records}
        deleted_count = sum(1 for record in old_records if record["path"] not in new_path_set)

        if not changed_or_new and deleted_count == 0:
            print(f"[build] cache is up to date, images={len(records)}")
            payload["built_at"] = now_text()
            self._save_index(payload)
            return payload

        print(
            "[build] incremental update, "
            f"keep={len(unchanged_indices)}, add_or_update={len(changed_or_new)}, delete={deleted_count}"
        )
        return self._build_with_progress(
            records=records,
            rebuild=False,
            old_payload=payload,
            unchanged_indices=unchanged_indices,
        )

    def _ensure_index(self, auto_build: bool = False) -> dict:
        if self.index_path.exists():
            return self.load_index()
        if not auto_build:
            raise FileNotFoundError(
                f"Index file not found: {self.index_path}. Run the build command first."
            )
        return self.build_index(rebuild=False)

    @staticmethod
    def _top_k(scores: np.ndarray, top_k: int) -> np.ndarray:
        top_k = max(1, min(top_k, scores.shape[0]))
        kth = scores.shape[0] - top_k
        candidate_indices = np.argpartition(scores, kth)[-top_k:]
        return candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

    def _search_with_vector(
        self,
        query_vector: np.ndarray,
        payload: dict,
        top_k: int = 10,
        source: str = "all",
    ) -> list[dict]:
        records = payload["records"]
        embeddings = payload["embeddings"].astype(np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise RuntimeError("Index is empty.")

        scores = embeddings @ query_vector.astype(np.float32)
        if source != "all":
            mask = np.array([record["source"] == source for record in records], dtype=bool)
            if not mask.any():
                return []
            filtered_scores = scores[mask]
            filtered_indices = np.flatnonzero(mask)
            ranked_local = self._top_k(filtered_scores, top_k)
            ranked_indices = filtered_indices[ranked_local]
        else:
            ranked_indices = self._top_k(scores, top_k)

        results: list[dict] = []
        for rank, index in enumerate(ranked_indices, start=1):
            record = records[int(index)]
            results.append(
                {
                    "rank": rank,
                    "score": float(scores[int(index)]),
                    "path": record["path"],
                    "abs_path": str(denormalize_path(record["path"])),
                    "source": record["source"],
                    "name": record["name"],
                    "item_id": record["item_id"],
                }
            )
        return results

    def search_by_path(
        self,
        query_image_path: str | Path,
        top_k: int = 10,
        source: str = "all",
        auto_build: bool = False,
    ) -> list[dict]:
        payload = self._ensure_index(auto_build=auto_build)
        query_vector = self.embed_image_file(query_image_path)
        return self._search_with_vector(query_vector, payload, top_k=top_k, source=source)

    def search_by_image(
        self,
        image: Image.Image,
        top_k: int = 10,
        source: str = "all",
        auto_build: bool = False,
    ) -> list[dict]:
        payload = self._ensure_index(auto_build=auto_build)
        query_vector = self.embed_pil_image(image)
        return self._search_with_vector(query_vector, payload, top_k=top_k, source=source)

    def stats(self) -> dict:
        payload = self.load_index()
        records = payload["records"]
        source_counts: dict[str, int] = {}
        for record in records:
            source_counts[record["source"]] = source_counts.get(record["source"], 0) + 1
        return {
            "index_path": str(self.index_path),
            "model_name": payload["model_name"],
            "built_at": payload["built_at"],
            "image_count": len(records),
            "embedding_shape": list(payload["embeddings"].shape),
            "source_counts": source_counts,
        }


def build_image_index(
    model_name: str = DEFAULT_MODEL_NAME,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    image_dirs: Sequence[str | Path] | None = None,
    batch_size: int = 64,
    device: str = "auto",
    rebuild: bool = False,
    local_files_only: bool = True,
) -> dict:
    searcher = ClipImageSearcher(
        model_name=model_name,
        index_path=Path(index_path),
        image_dirs=[Path(path) for path in (image_dirs or DEFAULT_IMAGE_DIRS)],
        batch_size=batch_size,
        device=device,
        local_files_only=local_files_only,
    )
    return searcher.build_index(rebuild=rebuild)


def search_similar_image_file(
    query_image_path: str | Path,
    top_k: int = 10,
    source: str = "all",
    model_name: str = DEFAULT_MODEL_NAME,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    image_dirs: Sequence[str | Path] | None = None,
    batch_size: int = 64,
    device: str = "auto",
    auto_build: bool = False,
    local_files_only: bool = True,
) -> list[dict]:
    searcher = ClipImageSearcher(
        model_name=model_name,
        index_path=Path(index_path),
        image_dirs=[Path(path) for path in (image_dirs or DEFAULT_IMAGE_DIRS)],
        batch_size=batch_size,
        device=device,
        local_files_only=local_files_only,
    )
    return searcher.search_by_path(
        query_image_path=query_image_path,
        top_k=top_k,
        source=source,
        auto_build=auto_build,
    )


def search_similar_pil_image(
    image: Image.Image,
    top_k: int = 10,
    source: str = "all",
    model_name: str = DEFAULT_MODEL_NAME,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    image_dirs: Sequence[str | Path] | None = None,
    batch_size: int = 64,
    device: str = "auto",
    auto_build: bool = False,
    local_files_only: bool = True,
) -> list[dict]:
    searcher = ClipImageSearcher(
        model_name=model_name,
        index_path=Path(index_path),
        image_dirs=[Path(path) for path in (image_dirs or DEFAULT_IMAGE_DIRS)],
        batch_size=batch_size,
        device=device,
        local_files_only=local_files_only,
    )
    return searcher.search_by_image(
        image=image,
        top_k=top_k,
        source=source,
        auto_build=auto_build,
    )


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Local CLIP model directory. Defaults to ./clip-vit-base-patch32.",
    )
    parser.add_argument(
        "--index-path",
        default=str(DEFAULT_INDEX_PATH),
        help="Where the cached image index will be stored.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Embedding device.",
    )
    parser.add_argument(
        "--image-dir",
        action="append",
        default=None,
        help="Override image directories. Can be used multiple times.",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a CLIP-based image index and search visually similar entries."
    )
    add_shared_arguments(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build or refresh the image index.")
    add_shared_arguments(build_parser)
    build_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore cache and rebuild the full index.",
    )

    search_parser = subparsers.add_parser("search", help="Search similar images by a query image.")
    add_shared_arguments(search_parser)
    search_parser.add_argument("--query", required=True, help="Path to the query image.")
    search_parser.add_argument("--top-k", type=int, default=10, help="Number of matches to return.")
    search_parser.add_argument(
        "--source",
        choices=["all", "onlineimgtmp", "localimgtmp"],
        default="all",
        help="Restrict results to one source folder.",
    )
    search_parser.add_argument(
        "--auto-build",
        action="store_true",
        help="Build the index automatically if it does not exist yet.",
    )

    stats_parser = subparsers.add_parser("stats", help="Show index statistics.")
    add_shared_arguments(stats_parser)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    image_dirs = [Path(path) for path in args.image_dir] if args.image_dir else DEFAULT_IMAGE_DIRS
    searcher = ClipImageSearcher(
        model_name=args.model,
        index_path=Path(args.index_path),
        image_dirs=image_dirs,
        batch_size=args.batch_size,
        device=args.device,
        local_files_only=True,
    )

    if args.command == "build":
        try:
            payload = searcher.build_index(rebuild=args.rebuild)
        except BuildInterruptedError as exc:
            print(str(exc))
            return
        print(
            json.dumps(
                {
                    "built_at": payload["built_at"],
                    "image_count": len(payload["records"]),
                    "embedding_shape": list(payload["embeddings"].shape),
                    "index_path": str(searcher.index_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "search":
        results = searcher.search_by_path(
            query_image_path=args.query,
            top_k=args.top_k,
            source=args.source,
            auto_build=args.auto_build,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    if args.command == "stats":
        print(json.dumps(searcher.stats(), ensure_ascii=False, indent=2))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
