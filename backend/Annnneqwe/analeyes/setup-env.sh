#!/usr/bin/env bash
# Create backend/Annnneqwe/analeyes/.env from .env.example without overwriting secrets.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE="${ROOT}/.env.example"
TARGET="${ROOT}/.env"

if [[ -f "${TARGET}" ]]; then
  echo "[setup-env] ${TARGET} already exists — leaving it untouched."
  echo "[setup-env] Edit it manually or remove it first if you want a fresh copy."
  exit 0
fi

if [[ ! -f "${EXAMPLE}" ]]; then
  echo "[setup-env] ERROR: missing ${EXAMPLE}" >&2
  exit 1
fi

cp "${EXAMPLE}" "${TARGET}"
chmod 600 "${TARGET}"
echo "[setup-env] Created ${TARGET} from .env.example"
echo "[setup-env] Fill in secrets (e.g. TELEGRAM_BOT_TOKEN) before docker compose up."
