#!/usr/bin/env sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT"
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required. Install Docker Desktop or Docker Engine first." >&2
  exit 1
fi
if [ -d .env ]; then
  echo ".env is a directory. Rename or remove it, then run this script again." >&2
  exit 1
fi
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example."
fi
docker compose up --build -d

echo "Waiting for XP-Gacha web service..."
attempt=0
until docker compose exec -T app curl -fsS --max-time 3 http://127.0.0.1:8000/ >/dev/null 2>&1; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 180 ]; then
    echo "XP-Gacha did not become available within 180 seconds." >&2
    docker compose ps
    docker compose logs --tail 120 app
    exit 1
  fi
  sleep 1
done

APP_ADDRESS=$(docker compose port app 8000)
docker compose ps
echo "XP-Gacha started: http://$APP_ADDRESS"
