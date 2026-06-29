# ENVIRONMENT.md — Setup & Operations

A.E. Brain targets **4× NVIDIA Tesla P100 (Pascal, sm_60)** with **fp16**
inference, PostgreSQL for logging, and RabbitMQ for messaging.

---

## 1. System prerequisites

| Component | Version / Notes |
|-----------|-----------------|
| Python    | **3.10 – 3.12** (do *not* use 3.13+; several ML wheels lag). |
| CUDA      | 11.x driver compatible with Pascal P100. |
| TA-Lib C  | `0.4.x` system library (the Python `TA-Lib==0.4.28` wheel binds to it). |
| PostgreSQL| 14+ |
| RabbitMQ  | 3.12+ |

> **HARD DEPENDENCY RULE:** `numpy < 2.0`. The pinned `TA-Lib 0.4.28` wheels are
> built against the numpy 1.x C-ABI; numpy 2.x breaks them at import. This is
> enforced in `requirements.txt` and `pyproject.toml`.

### Install the TA-Lib C library first

```bash
# Debian/Ubuntu
sudo apt-get install -y build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib
./configure --prefix=/usr && make && sudo make install

# Arch
sudo pacman -S ta-lib

# macOS
brew install ta-lib
```

---

## 2. Python environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 1) numpy<2 FIRST so TA-Lib binds to the correct ABI
pip install "numpy<2.0"

# 2) PyTorch CUDA build matching the P100 driver (example: CUDA 11.8)
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118

# 3) everything else
pip install -r requirements.txt

# editable install of the package + console script `ae-brain`
pip install -e .
```

### Minimal (CPU / no-GPU) install for development & tests

The pure-numerical core (EV gate, sizing, fusion, feature engineering) runs
without torch/sb3:

```bash
pip install "numpy<2.0" pandas scipy scikit-learn pydantic pydantic-settings \
            typer structlog orjson joblib pytest pytest-asyncio
pytest -q   # smoke tests in tests/test_smoke.py
```

---

## 3. Configuration

All settings are environment-driven (prefix `AEB_`) with sensible defaults; see
[`ae_brain/config.py`](ae_brain/config.py). Create a `.env` file:

```dotenv
AEB_ENV=prod
AEB_LOG_JSON=true

# PostgreSQL
AEB_DB_HOST=localhost
AEB_DB_USER=ae_brain
AEB_DB_PASSWORD=change-me
AEB_DB_NAME=ae_brain

# RabbitMQ
AEB_AMQP_URL=amqp://guest:guest@localhost:5672/
AEB_AMQP_CONSUME_QUEUE=data.candidates.ai
AEB_AMQP_PUBLISH_ROUTING_KEY=signal.final

# GPU / precision (4x P100)
AEB_GPU_ENABLED=true
AEB_GPU_DEVICE_IDS=[0,1,2,3]
AEB_GPU_USE_FP16=true
AEB_GPU_PREFER_ONNX=true

# Risk
AEB_RISK_ACCOUNT_EQUITY_USD=100000
AEB_RISK_MAX_LEVERAGE=5
AEB_RISK_KELLY_FRACTION=0.25
```

---

## 4. Infrastructure bring-up (Docker Compose — recommended)

`docker-compose.yml` defines three services: `postgres`, `rabbitmq`, and the
GPU `ae-brain` app (behind the `app` profile). The `Dockerfile` builds the app
on a **CUDA 11.8 / cuDNN8** base (Pascal P100 `sm_60`), compiles the TA-Lib C
library, installs the cu118 PyTorch wheel, then `numpy<2` + the rest.

```bash
cp .env.example .env            # adjust credentials/ports if needed

# Infra only (no GPU/NVIDIA runtime required) — Postgres + RabbitMQ.
docker compose up -d
docker compose ps               # both should report (healthy)

# The PostgreSQL schema is auto-applied on first init via
# /docker-entrypoint-initdb.d (ae_brain/data/schema.sql).
docker compose exec postgres psql -U ae_brain -d ae_brain -c "\dt"

# Build + run the GPU app service (requires the NVIDIA Container Toolkit
# installed on the host so Docker can expose the P100s):
docker compose --profile app up -d --build

# One-off jobs (e.g. training) reuse the same image:
docker compose run --rm ae-brain gen-data --rows 20000 --out /app/data/candles.csv
docker compose run --rm ae-brain train all --data /app/data/candles.csv

# Tear down (add -v to also drop the pgdata/rabbitmqdata volumes):
docker compose down
```

RabbitMQ management UI: http://localhost:15672 (default guest/guest).

#### Image build notes (two non-obvious gotchas, already handled)

1. **TA-Lib C library is built serially** (`make`, not `make -j`). The 0.4.0
   `gen_code` tool has a dependency-file race under parallel make that fails with
   `mv: cannot stat '.deps/gen_code-gen_code.Tpo'`.
2. **`PIP_CONSTRAINT` pins `numpy<2` for PEP 517 build isolation.** Installing
   `numpy<2` in the main env is not enough: pip builds the `TA-Lib==0.4.28`
   wheel in an isolated env that otherwise pulls numpy 2.x, causing C-API
   compile errors (`NPY_DEFAULT` / `NPY_C_CONTIGUOUS` undeclared,
   `PyArray_Descr.subarray` missing). The Dockerfile writes
   `/etc/pip-constraints.txt` and sets `PIP_CONSTRAINT` so isolated builds honour
   the pin. (Verified end-to-end: TA-Lib wheel builds against numpy 1.26.4 and
   the package + all deps import cleanly.)

> **GPU prerequisite:** the `ae-brain` service requests GPUs via
> `deploy.resources.reservations.devices`. The host must have the
> [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
> configured. Without it, run only the infra services (default `up`) and run the
> Python app directly on the host (Section 2/5).

### Plain `docker run` (alternative, no compose)

```bash
docker run -d --name aeb-pg  -e POSTGRES_USER=ae_brain -e POSTGRES_PASSWORD=ae_brain \
  -e POSTGRES_DB=ae_brain -p 5432:5432 postgres:16
docker run -d --name aeb-mq  -p 5672:5672 -p 15672:15672 rabbitmq:3.13-management
```

---

## 5. End-to-end quickstart

```bash
# 0) (optional) inspect the ~60-feature contract
ae-brain features

# 1) generate synthetic candles to validate plumbing
ae-brain gen-data --rows 20000 --out data/candles.parquet

# 2) apply the PostgreSQL schema (signal_feature_logs, candles, ...)
ae-brain init-db

# 3) train all three layers -> artifacts/
ae-brain train all --data data/candles.parquet --epochs 5 --timesteps 50000

# 4) evaluate a single candidate offline (crypto)
ae-brain gen-candidate --out examples/candidate.json
ae-brain evaluate --candidate examples/candidate.json

# 4b) non-crypto assets: derivatives fields (funding/OI/CVD/liquidations) are
#     emitted as null to exercise the null-handling path.
ae-brain gen-candidate --out examples/aapl.json  --symbol AAPL   --asset-class stock
ae-brain gen-candidate --out examples/xau.json   --symbol XAUUSD --asset-class metal
ae-brain gen-candidate --out examples/eur.json   --symbol EURUSD --asset-class forex
ae-brain evaluate --candidate examples/aapl.json

# 4c) simulate the backend UPDATE path (use a real pre-inserted row id)
ae-brain gen-candidate --out examples/c.json --asset-class crypto --signal-log-db-id 12345

# 5) run the live RabbitMQ loop (consume data.candidates.ai -> publish signal.final)
ae-brain run

# 6) (optional) debugging HTTP surface
ae-brain serve-api --port 8080     # POST /evaluate, GET /health
```

---

## 6. Multi-asset architecture & the hybrid UPDATE/INSERT workflow

### 6.1 Asset classes (PRD/ТЗ #2)

Every `TradeCandidate` carries an `asset_class`: `crypto`, `stock`, `metal`, or
`forex` (invalid/missing values default to `crypto`). Only **crypto** is treated
as a *derivative* asset that carries perpetual microstructure; the rest are
"traditional" assets.

For traditional assets the backend publishes derivatives-only fields as `null`:
`funding_rate`, `open_interest`, `taker_buy_volume` (and therefore `cvd`),
`long_liq_notional`, `short_liq_notional`, `basis`. These are mapped to **neutral
numeric fallbacks** (`0.0`, or `0.5·volume` for taker-buy) by a single
choke-point, `_coerce_series` in `features/engineering.py`, plus `_last_float`
in `inference/engine.py`. Funding is forced to `0.0` for non-derivative assets,
so the EV gate sees no phantom funding cost/credit. Net effect: the exact same
~60-feature contract and 4-layer pipeline run unchanged across asset classes —
no `ValueError`, no shape mismatch.

### 6.2 Hybrid UPDATE / INSERT logging

The decision-logging path is now hybrid, keyed on `signal_log_db_id`:

| `signal_log_db_id` | Path | When |
|--------------------|------|------|
| `> 0` (backend-supplied) | **`UPDATE signal_feature_logs SET ... WHERE id = $1`** | Production: the backend pre-INSERTs the row (raw inputs) and hands us its id; the ensemble writes its outputs back into that same row (no duplicate). |
| `0` / missing / null | **`INSERT` fallback** | Local/dev/API and **legacy crypto producers** that publish without an id (backward compatible — never raises). |

The `UPDATE` writes the full ensemble result set into the existing row:
calibrated layer probabilities, EV breakdown, final decision, fractional-Kelly
sizing (`kelly_fraction`), leverage, dynamic TP/SL, the `metrics` JSONB
component breakdown, `asset_class`, and `evaluated_at = now()`.

Schema additions (idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`, so
existing databases migrate cleanly on `ae-brain init-db`): `evaluated_at`,
`asset_class`, `kelly_fraction`, `metrics`.

---

## 7. GPU / fp16 notes

* Sequence models are cast to **fp16** and moved round-robin across the 4 P100s
  (`DeviceRouter`). cuDNN autotuner is enabled for the fixed inference shape.
* If an exported `sequence_model.onnx` exists and `AEB_GPU_PREFER_ONNX=true`,
  inference uses **ONNXRuntime-GPU** (`CUDAExecutionProvider`); otherwise native
  torch is used. Training auto-exports ONNX best-effort.
* Heavy inference is offloaded off the event loop: feature engineering →
  `ProcessPoolExecutor`; model inference → `ThreadPoolExecutor` (torch/ONNX and
  the GBDT C++ cores release the GIL).

---

## 8. Operational guarantees

* **Message acking:** every RabbitMQ handler is wrapped in `try/finally` so an
  `ack`/`nack` is always sent — no unbounded unacked window, no queue overflow.
* **No external LLM APIs** are used in the decision path. RAG (ChromaDB) is
  optional, off by default, and not in the live loop.
* **Audit trail:** every evaluated candidate (including `SKIP`) is logged to
  `signal_feature_logs` with full features + EV math.
