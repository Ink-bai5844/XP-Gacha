FROM node:22-bookworm-slim AS frontend

WORKDIR /build/web
RUN corepack enable
COPY web/package.json web/pnpm-lock.yaml web/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY web/ ./
RUN pnpm build

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XP_GACHA_ENV=production \
    XP_GACHA_HOST=0.0.0.0 \
    XP_GACHA_PORT=8000

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install -r requirements.txt

COPY . .
COPY --from=frontend /build/web/dist /app/web/dist
RUN mkdir -p b64_cache b64_tmp data/gallery_info datacache dictionaries \
    localimgtmp logs manga_vectors models onlineimgtmp library

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
