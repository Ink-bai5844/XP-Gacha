FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install -r requirements.txt

COPY . .
COPY config_docker.py config.py

RUN mkdir -p \
    .streamlit \
    b64_cache \
    b64_tmp \
    data \
    datacache \
    dictionaries \
    localimgtmp \
    logs \
    manga_vectors \
    models \
    onlineimgtmp

EXPOSE 8501 8765

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
