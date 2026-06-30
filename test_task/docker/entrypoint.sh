#!/usr/bin/env bash
# A.E. Brain container entrypoint.
#
# Responsibilities:
#   * (optionally) wait for PostgreSQL + RabbitMQ to be reachable,
#   * (optionally) apply the DB schema on first boot,
#   * then exec the requested `ae-brain` subcommand (default: run).
#
# Any arguments are passed straight through to the `ae-brain` CLI, e.g.:
#   docker compose run --rm ae-brain train all --data /app/data/candles.parquet
set -euo pipefail

wait_for() {
    local host="$1" port="$2" name="$3" tries="${4:-60}"
    echo "[entrypoint] waiting for ${name} at ${host}:${port} ..."
    for _ in $(seq 1 "${tries}"); do
        if (echo > "/dev/tcp/${host}/${port}") >/dev/null 2>&1; then
            echo "[entrypoint] ${name} is up."
            return 0
        fi
        sleep 1
    done
    echo "[entrypoint] WARNING: ${name} not reachable after ${tries}s; continuing."
}

if [[ "${AEB_WAIT_FOR_DEPS:-true}" == "true" ]]; then
    wait_for "${AEB_DB_HOST:-aeb-postgres}" "${AEB_DB_PORT:-5432}" "aeb-postgres"
    wait_for "${AEB_AMQP_HOST:-rabbitmq}" "${AEB_AMQP_PORT:-5672}" "rabbitmq"
fi

if [[ "${AEB_AUTO_INIT_DB:-true}" == "true" ]]; then
    echo "[entrypoint] applying database schema (idempotent) ..."
    ae-brain init-db || echo "[entrypoint] init-db failed (continuing)."
fi

echo "[entrypoint] exec: ae-brain $*"
exec ae-brain "$@"
