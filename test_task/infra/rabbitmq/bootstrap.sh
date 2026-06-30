#!/bin/sh
# Idempotent RabbitMQ vhost/user bootstrap for test_task local broker.
set -eu

RABBITMQ_HOST="${RABBITMQ_HOST:-rabbitmq}"
RABBITMQ_ADMIN_USER="${RABBITMQ_ADMIN_USER:-admin}"
RABBITMQ_ADMIN_PASSWORD="${RABBITMQ_ADMIN_PASSWORD:-changeme}"
RABBITMQ_APP_USER="${RABBITMQ_APP_USER:-analeyes}"
RABBITMQ_APP_PASSWORD="${RABBITMQ_APP_PASSWORD:-changeme}"
RABBITMQ_VHOST="${RABBITMQ_VHOST:-analeyes}"

API="http://${RABBITMQ_HOST}:15672/api"
AUTH="${RABBITMQ_ADMIN_USER}:${RABBITMQ_ADMIN_PASSWORD}"

echo "[aeb-rabbitmq-bootstrap] waiting for management API ..."
ready=0
for _ in $(seq 1 45); do
  if curl -sf -u "${AUTH}" "${API}/overview" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
if [ "${ready}" -ne 1 ]; then
  echo "[aeb-rabbitmq-bootstrap] ERROR: management API not ready" >&2
  exit 1
fi

curl -sf -u "${AUTH}" -X PUT "${API}/vhosts/${RABBITMQ_VHOST}" -H 'content-type: application/json' -d '{}' || true
curl -sf -u "${AUTH}" -X PUT "${API}/users/${RABBITMQ_APP_USER}" \
  -H 'content-type: application/json' \
  -d "{\"password\":\"${RABBITMQ_APP_PASSWORD}\",\"tags\":\"\"}"
curl -sf -u "${AUTH}" -X PUT "${API}/permissions/${RABBITMQ_VHOST}/${RABBITMQ_APP_USER}" \
  -H 'content-type: application/json' \
  -d '{"configure":".*","write":".*","read":".*"}'

echo "[aeb-rabbitmq-bootstrap] complete user=${RABBITMQ_APP_USER} vhost=${RABBITMQ_VHOST}"
