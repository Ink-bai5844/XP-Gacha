from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

logging.getLogger("streamlit").setLevel(logging.ERROR)

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from server import __version__
from server.modules.charts import ChartsModule
from server.modules.history import HistoryModule
from server.modules.imports import ImportModule
from server.modules.jobs import JobsModule
from server.modules.library import LibraryModule
from server.modules.preferences import PreferencesModule
from server.modules.system import SystemModule
from server.schemas import (
    ChatRequest,
    HistoryDeleteRequest,
    HistoryRecordRequest,
    ImportModeRequest,
    JobStartRequest,
    LibraryQuery,
    PreferencesRequest,
)
from server.settings import settings
from utils_chat import get_ai_response_stream
from utils_core import get_cover_base64
from utils_history import load_history_entries
from utils_online_cover import has_cached_cover, is_online_cover_pending, submit_online_cover_fetches
from utils_cv import search_similar_cover_items


library = LibraryModule()
history = HistoryModule(library)
charts = ChartsModule(library)
imports = ImportModule(library)
jobs = JobsModule(library.refresh)
system = SystemModule()
preferences = PreferencesModule()


@asynccontextmanager
async def app_lifespan(_app: FastAPI):
    try:
        library.meta()
    except Exception:
        logging.getLogger(__name__).exception("Library metadata warm-up failed")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=app_lifespan,
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)
    if settings.environment == "development":
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[],
            allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    register_api_routes(app)
    register_frontend(app)
    return app


def register_api_routes(app: FastAPI) -> None:
    @app.get("/api/health")
    def health() -> dict:
        status = system.status()
        return {
            "status": "ok",
            "version": __version__,
            "database": status["database"],
            "frontend": settings.frontend_dist.exists(),
        }

    @app.get("/api/system/status")
    def system_status() -> dict:
        return system.status()

    @app.get("/api/meta/options")
    def meta_options() -> dict:
        return library.meta()

    @app.get("/api/meta/options/search")
    def search_meta_options(
        kind: str = Query(..., pattern="^(tags|artists|titleWords)$"),
        q: str = Query("", max_length=200),
        limit: int = Query(80, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> dict:
        return library.search_options(kind, q, limit, offset)

    @app.post("/api/library/query")
    def query_library(request: LibraryQuery) -> dict:
        return library.query(request)

    @app.get("/api/gallery/{item_id}")
    def gallery_detail(item_id: str) -> dict:
        item = library.detail(item_id.strip().upper())
        if not item:
            raise HTTPException(404, "未找到该馆藏条目")
        return item

    @app.get("/api/covers/{item_id}")
    def cover(item_id: str, online: bool = Query(True)) -> Response:
        item = library.detail(item_id.strip().upper())
        if not item:
            raise HTTPException(404, "未找到该馆藏条目")
        data_uri = get_cover_base64(
            item.get("localPath", ""), item.get("id", ""), item.get("link", ""), allow_online=online
        )
        if not data_uri or "," not in data_uri:
            raise HTTPException(404, "封面尚不可用")
        header, encoded = data_uri.split(",", 1)
        media_type = header.split(";")[0].replace("data:", "") or "image/jpeg"
        try:
            return Response(base64.b64decode(encoded), media_type=media_type, headers={"Cache-Control": "public, max-age=86400"})
        except ValueError as exc:
            raise HTTPException(500, "封面缓存格式损坏") from exc

    @app.get("/api/covers/status")
    def cover_status(ids: str = Query("")) -> dict:
        item_ids = [item.strip().upper() for item in ids.split(",") if item.strip()]
        return {
            "items": {
                item_id: {"cached": has_cached_cover(item_id), "pending": is_online_cover_pending(item_id)}
                for item_id in item_ids[:500]
            }
        }

    @app.post("/api/covers/refresh")
    def refresh_covers(item_ids: Annotated[list[str], Body()]) -> dict:
        normalized = [item.strip().upper() for item in item_ids if item.strip()][:500]
        submit_online_cover_fetches(normalized)
        return {"queued": len(normalized)}

    @app.post("/api/search/cover")
    async def search_cover(file: Annotated[UploadFile, File()]) -> dict:
        payload = await file.read(20 * 1024 * 1024 + 1)
        if len(payload) > 20 * 1024 * 1024:
            raise HTTPException(413, "封面图片不能超过 20 MB")
        try:
            return search_similar_cover_items(query_image_bytes=payload)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/history")
    def list_history() -> dict:
        return {"entries": history.list()}

    @app.post("/api/history")
    def record_history(request: HistoryRecordRequest) -> dict:
        try:
            return {"entries": history.record(request.item_id, request.action)}
        except KeyError as exc:
            raise HTTPException(404, "未找到要记录的馆藏条目") from exc

    @app.delete("/api/history")
    def delete_history(request: HistoryDeleteRequest) -> dict:
        return {"entries": history.delete(request.keys)}

    @app.delete("/api/history/all")
    def clear_history() -> dict:
        return {"entries": history.clear()}

    @app.post("/api/gallery/{item_id}/open-local")
    def open_local(item_id: str) -> dict:
        item = library.detail(item_id.strip().upper())
        if not item:
            raise HTTPException(404, "未找到馆藏条目")
        target = Path(item.get("localPath", ""))
        if str(target) == "本地目录不存在" or not target.exists():
            raise HTTPException(404, "本地目录不存在")
        history.record(item["id"], "local_folder")
        if settings.allow_open_local and os.name == "nt":
            os.startfile(str(target))
        return {"opened": settings.allow_open_local and os.name == "nt", "path": str(target)}

    @app.get("/api/track/{item_id}")
    def track_network(item_id: str) -> RedirectResponse:
        item = library.detail(item_id.strip().upper())
        if not item:
            raise HTTPException(404, "未找到馆藏条目")
        target = item.get("link", "")
        if urlparse(target).scheme not in {"http", "https"}:
            raise HTTPException(400, "来源链接无效")
        history.record(item["id"], "network_link")
        return RedirectResponse(target, status_code=302)

    @app.get("/api/charts/global")
    def global_charts() -> dict:
        return charts.global_charts()

    @app.get("/api/charts/history")
    def history_charts() -> dict:
        return charts.history_charts()

    @app.post("/api/chat/stream")
    def chat_stream(request: ChatRequest) -> StreamingResponse:
        context_ids = request.context_ids
        if not context_ids and request.context_count > 0:
            query_result = library.query(LibraryQuery(page_size=request.context_count))
            context_ids = [item["id"] for item in query_result["items"]]
        context = library.rows_for_ids(context_ids[: request.context_count])

        def event_stream():
            yield f"data: {json.dumps({'type': 'meta', 'contextIds': context_ids}, ensure_ascii=False)}\n\n"
            for chunk in get_ai_response_stream(
                request.query,
                context,
                api_mode=request.api_mode,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ):
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/scripts")
    def scripts() -> dict:
        return jobs.scripts()

    @app.post("/api/jobs", status_code=202)
    def start_job(request: JobStartRequest) -> dict:
        try:
            return jobs.start(request.script_id, request.parameters)
        except KeyError as exc:
            raise HTTPException(404, "未知脚本") from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, after: int = Query(0, ge=0)) -> dict:
        try:
            return jobs.get(job_id, after=after)
        except KeyError as exc:
            raise HTTPException(404, "任务不存在") from exc

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        try:
            return jobs.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(404, "任务不存在") from exc

    @app.post("/api/import/bundle")
    async def import_bundle(
        file: Annotated[UploadFile, File()],
        mode: Annotated[str, Form()] = "upsert",
        include_dictionaries: Annotated[bool, Form()] = True,
    ) -> dict:
        if mode not in {"upsert", "replace"}:
            raise HTTPException(422, "mode 必须是 upsert 或 replace")
        try:
            payload = await file.read(settings.import_max_mb * 1024 * 1024 + 1)
            return imports.import_bundle(file.filename or "bundle.zip", payload, mode, include_dictionaries)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/import/project")
    def import_project(request: ImportModeRequest) -> dict:
        try:
            return imports.import_project_data(request.mode)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/preferences")
    def get_preferences() -> dict:
        return preferences.get()

    @app.put("/api/preferences")
    def put_preferences(request: PreferencesRequest) -> dict:
        return preferences.update(request.model_dump(by_alias=True))


def register_frontend(app: FastAPI) -> None:
    dist = settings.frontend_dist
    assets = dist / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        requested = (dist / path).resolve()
        if path and requested.is_file() and dist in requested.parents:
            return FileResponse(requested, media_type=mimetypes.guess_type(requested.name)[0])
        index = dist / "index.html"
        if index.exists():
            return FileResponse(index)
        return JSONResponse(
            {
                "status": "backend-ready",
                "message": "web 尚未构建；开发时运行 pnpm --dir web dev，部署时执行 pnpm --dir web build。",
                "docs": "/api/docs",
            }
        )


app = create_app()
