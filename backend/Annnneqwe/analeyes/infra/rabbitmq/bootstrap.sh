#!/bin/sh
# Idempotent RabbitMQ application vhost + user bootstrap via Management HTTP API.
# Runs once after the broker is healthy (fresh OR existing data volume).
set -eu

RABBITMQ_HOST="${RABBITMQ_HOST:-rabbitmq}"
RABBITMQ_ADMIN_USER="${RABBITMQ_ADMIN_USER:-admin}"
RABBITMQ_ADMIN_PASSWORD="${RABBITMQ_ADMIN_PASSWORD:?RABBITMQ_ADMIN_PASSWORD is required}"
RABBITMQ_APP_USER="${RABBITMQ_APP_USER:-analeyes}"
RABBITMQ_APP_PASSWORD="${RABBITMQ_APP_PASSWORD:?RABBITMQ_APP_PASSWORD is required}"
RABBITMQ_VHOST="${RABBITMQ_VHOST:-analeyes}"

API="http://${RABBITMQ_HOST}:15672/api"
AUTH="${RABBITMQ_ADMIN_USER}:${RABBITMQ_ADMIN_PASSWORD}"

echo "[rabbitmq-bootstrap] waiting for management API at ${RABBITMQ_HOST}:15672 ..."
ready=0
for _ in $(seq 1 45); do
  if curl -sf -u "${AUTH}" "${API}/overview" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
if [ "${ready}" -ne 1 ]; then
  echo "[rabbitmq-bootstrap] ERROR: management API not ready" >&2
  exit 1
fi

echo "[rabbitmq-bootstrap] ensuring vhost=${RABBITMQ_VHOST}"
curl -sf -u "${AUTH}" -X PUT "${API}/vhosts/${RABBITMQ_VHOST}" \
  -H 'content-type: application/json' -d '{}' || true

echo "[rabbitmq-bootstrap] ensuring user=${RABBITMQ_APP_USER}"
curl -sf -u "${AUTH}" -X PUT "${API}/users/${RABBITMQ_APP_USER}" \
  -H 'content-type: application/json' \
  -d "{\"password\":\"${RABBITMQ_APP_PASSWORD}\",\"tags\":\"\"}"

echo "[rabbitmq-bootstrap] granting permissions on vhost=${RABBITMQ_VHOST}"
curl -sf -u "${AUTH}" -X PUT "${API}/permissions/${RABBITMQ_VHOST}/${RABBITMQ_APP_USER}" \
  -H 'content-type: application/json' \
  -d '{"configure":".*","write":".*","read":".*"}'

echo "[rabbitmq-bootstrap] complete user=${RABBITMQ_APP_USER} vhost=${RABBITMQ_VHOST}"
