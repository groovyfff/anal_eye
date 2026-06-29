# AnalEyes Crypto V6 — Подробная документация

> **Версия проекта в каталоге:** V6 (в коде/контейнерах встречаются строки с маркером `V9` — это унаследованное именование UI/заголовков; функционально это V6).
> **Язык комментариев/логов:** преимущественно русский.

AnalEyes Crypto — асинхронный микросервисный комплекс для автоматического анализа фьючерсных крипторынков, генерации торговых сигналов с помощью технических индикаторов, свечных паттернов и ансамбля LLM-движков, их рассылки в Telegram и пост-отслеживания результатов. Архитектура построена на **RabbitMQ** (шина событий, topic-exchange `analeyes_exchange`), **PostgreSQL** (центральное хранилище `signal_feature_logs` и смежные таблицы), **asyncio** в каждом сервисе, и контейнеризуется через **docker-compose**.

---

## Содержание

1. [Архитектура и поток данных](#1-архитектура-и-поток-данных)
2. [Структура репозитория](#2-структура-репозитория)
3. [Запуск (docker-compose)](#3-запуск-docker-compose)
4. [Конфигурация (`config/settings.yml`)](#4-конфигурация-configsettingsyml)
5. [Переменные окружения (`.env`)](#5-переменные-окружения-env)
6. [Shared-пакет (`shared/`)](#6-shared-пакет-shared)
7. [collector-service](#7-collector-service)
8. [processing-service](#8-processing-service)
9. [ai-service](#9-ai-service)
10. [notification-service](#10-notification-service)
11. [tracker-service](#11-tracker-service)
12. [backtest-service](#12-backtest-service)
13. [api-gateway](#13-api-gateway)
14. [web-ui](#14-web-ui)
15. [Схема БД (ORM-модели)](#15-схема-бд-orm-модели)
16. [RabbitMQ-топология](#16-rabbitmq-топология)
17. [Безопасность и известные нюансы](#17-безопасность-и-известные-нюансы)
18. [Утилиты и вспомогательные скрипты](#18-утилиты-и-вспомогательные-скрипты)
19. [Диаграммы (Mermaid)](#19-диаграммы-mermaid)
20. [Runbook и troubleshooting](#20-runbook-и-troubleshooting)
21. [Developer onboarding](#21-developer-onboarding)
22. [Config reference (полный)](#22-config-reference-полный)
23. [Glossary терминов](#23-glossary-терминов)
24. [Performance и масштабирование](#24-performance-и-масштабирование)
25. [Тестирование](#25-тестирование)
26. [Roadmap / TODO](#26-roadmap--todo)
27. [Changelog и лицензия](#27-changelog-и-лицензия)

---

## 1. Архитектура и поток данных

### Сервисы

| Сервис | Назначение |
|---|---|
| **collector-service** | Сбор рыночных данных с Binance Futures (REST+WS), а также с OKX, CoinGecko, Coinalyze, TradingView (календарь), X/Twitter, btcparser.com. Публикует сырые данные пар в шину. |
| **processing-service** | Генерация ~60 признаков, распознавание свечных паттернов, индикаторные стратегии, составной рейтинг (composite score), фильтрация пар. Пишет `signal_feature_logs`. |
| **ai-service** | Ансамбль LLM (GPT, Grok, Gemini, Claude, DeepSeek, Qwen). Голосование за направление, усреднение TP/SL/уверенности. Пишет `ensemble_model_decisions`. |
| **notification-service** | Форматирование и отправка сигналов/исходов/входов в Telegram (форум-топики по моделям). Генерация графиков через mplfinance. |
| **tracker-service** | Конечный автомат отслеживания каждого сигнала (ожидание входа → вход → TP/SL/EXPIRE). Публикует исходы и события входа. Prometheus-метрики. |
| **backtest-service** | Реплей исторических сигналов из БД, классификация исхода (TP/SL/EXPIRED) по одной свече, запрос LLM-анализа неудач. |
| **api-gateway** | FastAPI: JWT-аутентификация по Telegram ID, REST-эндпоинты (сигналы, настройки, ключи, статистика, бэктесты), WebSocket `/ws/status`, раздача `web-ui/index.html`. |

### Инфраструктура

| Компонент | Назначение |
|---|---|
| **PostgreSQL 15** | Центральная БД. |
| **RabbitMQ 3.12** (с management-плагином) | Шина обмена сообщениями. |
| **rabbitscout** | Опциональный веб-UI для наблюдения за RabbitMQ. |
| **ChromaDB** (внешний, см. `rag.*`) | Векторное хранилище для RAG (похожие сигналы в прошлом). |

### Цикл конвейера

```
            ┌─────────────────── Binance REST/WS ──────────────────┐
            │   + OKX, CoinGecko, Coinalyze, TV, X, btcparser      │
            ▼                                                       │
  ┌──────────────────┐   data.raw.binance   ┌────────────────────┐ │
  │ collector-service├──────────────────────▶│ processing-service │ │
  └──────────────────┘                       │  (features+filter+ │ │
            ▲ status.collector               │   composite score) │ │
            │                                └─────────┬──────────┘ │
            │                              data.candidates.ai       │
            │                                          ▼            │
            │                                ┌────────────────────┐ │
            │                                │    ai-service      │ │
            │                                │ (LLM ансамбль +    │ │
            │                                │  голосование)      │ │
            │                                └─────────┬──────────┘ │
            │                                          │ signal.final│
            │                       ┌──────────────────┼────────────┼─▶ DB (ensemble_model_decisions)
            │                       │                  │            │
            │                       ▼                  ▼            │
            │          ┌──────────────────┐  ┌────────────────────┐ │
            │          │ notification-svc │  │  tracker-service   │ │
            │          │ (Telegram+график)│  │  (FSM TP/SL/expire)│ │
            │          └────────▲─────────┘  └───┬────────────┬───┘ │
            │    signal.outcome   │                │            │     │
            │    signal.entry_evt │   ◀────────────┘            │     │
            └─────────────────────┘   signal.outcome            │     │
                                                                 │     │
                                  ┌──────────────────────────────┘     │
                                  ▼   (status.tracker + ensemble_backtest_results)
                          DB (signal_feature_logs.tracker_*)

  api-gateway  ──▶  publishes control.backtest.start ──▶ backtest-service ──▶ ai.backtest.analyze ──▶ ai-service (analyze_failure mode)
  api-gateway  ──▶  consumes status.* (broadcast to WebSocket /ws/status)
```

Каждый сервис также публикует heartbeat на `status.<имя>` (каденс ~5с), который api-gateway собирает и раздаёт фронту.

---

## 2. Структура репозитория

```
AnalEyes_Crypto_V6/
├── docker-compose.yml          # вся оркестрация
├── entrypoint.sh               # ждёт PG + применяет Alembic-миграции
├── requirements.txt            # общий корневой requirements (legacy/union)
├── aistud.py                   # утилита «сплющивания» проекта в один слой (для AI-анализа)
├── README.md                   # (этот файл)
├── config/
│   ├── settings.yml            # основная YAML-конфигурация
│   └── prompts/                # шаблоны промптов для LLM-движков
├── shared/                     # переиспользуемый Python-пакет
│   ├── setup.py                # name="analeyes-shared", импорт как `shared.*`
│   ├── requirements.txt
│   └── src/shared/
│       ├── config.py
│       ├── database/{db_manager.py, models.py}
│       ├── utils/{pika_client.py, logger.py, rate_limiter.py, circuit_breaker.py}
│       ├── rag/rag_manager.py
│       ├── alembic.ini
│       └── alembic/{env.py, versions/...}
├── services/
│   ├── ai-service/{Dockerfile, requirements.txt, src/}
│   ├── api-gateway/{Dockerfile, requirements.txt, src/}
│   ├── backtest-service/{Dockerfile, requirements.txt, src/}
│   ├── collector-service/{Dockerfile, requirements.txt, src/}
│   ├── notification-service/{Dockerfile, requirements.txt, src/}
│   ├── processing-service/{Dockerfile, requirements.txt, src/}
│   └── tracker-service/{Dockerfile, requirements.txt, src/}
├── web-ui/index.html           # Single-page дашборд (раздаётся api-gateway)
└── logs/                       # точка монтирования логов контейнеров
```

> **Именование директорий сервисов содержит дефис** (`ai-service`, и т.д.). Из-за этого команда `python -m services.ai-service.src.main` в Dockerfile **технически некорректна** (дефис — недопустимый символ в имени Python-модуля). На практике это работает только потому, что `python -m` трактует аргумент как путь файловой системы, а не как идентификатор импорта. Учитывайте это при локальном запуске/дебаге.

---

## 3. Запуск (docker-compose)

`docker-compose.yml` поднимает:

**Инфраструктура:**
- `rabbitmq:3.12-management` (порты 5672 AMQP, 15672 management UI; healthcheck `rabbitmq-diagnostics ping`).
- `postgres:15-alpine` (порт из `POSTGRES_PORT`, по умолчанию 5432; healthcheck `pg_isready`).
- `rabbitscout` — веб-мониторинг RabbitMQ на порту 3000.

**Сервисы приложения:** `api-gateway`, `collector-service`, `processing-service`, `ai-service`, `notification-service`, `tracker-service`, `backtest-service`. Все в сети `analeyes-net`.

**Ключевые моменты compose:**
- Зависимости (`depends_on` с `condition: service_healthy`) обеспечивают порядок: `rabbitmq`+`postgres` первыми, затем сервисы.
- Сервисы монтируют `./services`, `./shared`, `./config` внутрь `/app` (живой код для hot-reload).
- `api-gateway`, `processing`, `notification`, `tracker`, `backtest` используют `entrypoint.sh` — он ждёт готовность Postgres и запускает `alembic upgrade head` (миграции из `shared/src/shared/alembic.ini`).
- `api-gateway` запускается через `uvicorn ... --reload`, монтирует `web-ui`.
- Жёстко задано `PYTHONPATH=/app` в AI/notification/tracker/backtest для разрешения импортов `services.*`.

**Запуск:**
```bash
cp .env.example .env   # заполнить ключи (см. раздел 5)
docker compose up --build
```

---

## 4. Конфигурация (`config/settings.yml`)

YAML ~471 строка. Точные значения см. в файле; ниже — смысловая карта секций.

| Секция | Что регулирует |
|---|---|
| `binance` | Режим (testnet), лимит REST-вызовов/мин (1200), WS-флаги, CircuitBreaker (`max_errors`, `reset_interval_s`). |
| `collector` | Интервалы (`pairs_check`, `excluded_symbols_check`, `exchange_info`, `oi_history_fetch`), `max_pairs_for_testing` (200), ротация пар, `min_klines_ws_ready` (50), интервалы доп-таймфреймов (15m/30m/1h/4h), `permanent_subscription_subset` (BTC/ETH/SOL/BNB). |
| `websocket` | URL, ретраи, буферы (`max_klines_per_symbol`=500, `max_trades_per_symbol`=1000, `max_liquidations_per_symbol_ws`=200), `num_connections` (5), параметры ротации, ping/pong, `connection_liveness_timeout_s`. |
| `features` | Окна индикаторов, `min_candles_for_ta`=50, `liquidation_window_hours`=3, окна CVD/OFI/order_flow_ratio/hidden_order, time_factors по сессиям, eth_btc_spread, bvol_proxy, флаги onchain/sentiment-прокси. |
| `filters` | Базовые пороги (`vol_z`=0.8, `price_pct`=0.18, `oi_z`=0.5, `liq_usd`=40k, `funding`=0.00015, `ofi_delta`=100k, `eis`=5.0, `vol_rel`=0.8, `price_pct_norm`=0.25, `spread_pct`=0.15), категории по капитализации, glitch-фильтр, gamma/OKX/X-heat триггеры. |
| `composite_score` | `threshold`=0.5, `weights` (vol_rel 0.22, acc_vol 0.15, price_pct_norm 0.15, acc_price 0.12, oi_z 0.10, ofi_delta 0.10, patterns 0.10, eis 0.07, order_flow_ratio 0.05, cvd 0, liq 0), `normalization_thresholds`. |
| `ai_engines` | Переключатели движков (`use_gpt/grok/gemini/qwen/deepseek/claude`), имена моделей, `*_base_url`, `prompt_paths`, `gpt_max_retries`=2, `composite_threshold`=0.78, `min_confidence`=0.60, `prompt_compress`. |
| `trend_impulse` (EIS) | `half_life_min`=2, `oi_weight_div`=5.0, `filter_min_abs_threshold`=5.0, `emoji_bands`. |
| `ml_proxy` / `rl_exit` | Заготовки LightGBM/PPO-агента (в основном выключены и не подключены в рантайме). |
| `coingecko_api`, `btc_parser`, `okx`, `economic_calendar`, `x_heat`, `session_regimes` | Внешние источники/режимы. |
| `telegram` | `group_id`, `ai_specific_topics` (маппинг модель → ID топика форума: gpt=4, gemini=2, claude=12, deepseek=6, qwen=10, grok=8, ensemble=15), `min_confidence`=0.7, `retry_attempts`=3, `signal_cooldown_minutes`=15. |
| `signal_tracker` | `expiration_hours`=24, `check_interval_s`=1, `send_outcome_alerts`. |
| `timeouts` | `rest_s`=10, `ai_s`=20. |
| `trading` | `default_initial_bank_usd`=500.0. |
| `indicators` | Параметры RSI(14, 30/70), MACD(12/26/9), EMA(8/21), BB(20, 2), ADX(14), ATR(14), price_slope(10), VWAP, supertrend(10, 3.0), ichimoku(9/26/52), stochastic(14/3/3, 20/80). |
| `patterns` | Список 18 свечных паттернов TA-Lib, веса strength, `recent_candles_check_window`=3, `consensus_strength_ratio_threshold`=1.5. |
| `logging` | Уровень (INFO), форматы с `%(shortname)`, ротация файлов (10MB, 3 копии). |
| `system` | `cycle_interval_s`=60, retry/health интервалы. |
| `coinalyze` | ВКЛ, эндпоинты, **13 захардкоженных API-ключей** (ротор). |
| `market_global_conditions` | Состояние рынка (TREND/FLAT/TRANSITION), активность (VERY_ACTIVE…VERY_PASSIVE), матрица корректировки порогов. |
| `database` | `pool_size`=5, `max_overflow`=10, `log_retention_days`=30. |
| `forced_trade_parameters` | Если включено — фиксированные TP=15% / SL=-5% переопределяют AI-предложения. |
| `history_snapshots` | Длина 3, список метрик для сжатия в промпт. |
| `backtest.ai_analysis` | Запрос LLM-разбора неудачных сделок. |
| `rag` | ChromaDB-хост, коллекция, embedding-модель, top-N. |

---

## 5. Переменные окружения (`.env`)

Обязательные/важные:

| Переменная | Назначение |
|---|---|
| `RABBITMQ_USER`, `RABBITMQ_PASS` | Доступ к RabbitMQ. |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_PORT`, `POSTGRES_HOST` | Доступ к PostgreSQL. |
| `DATABASE_URL` | Полный URL async-SQLAlchemy (`postgresql+asyncpg://...`). |
| `RABBITMQ_URL` | AMQP URL для `pika` (`amqp://user:pass@rabbitmq:5672/%2F`). |
| `API_SECRET_KEY` | Секрет для JWT в api-gateway (обязателен, иначе импорт падает). |
| `HEALTH_CHECK_TOKEN` | Bearer-токен для `GET /api/health/full`. |
| `BINANCE_API_KEY`, `BINANCE_API_SECRET` | Опц., для подписанных эндпоинтов Binance. |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ADMIN_ID` | Бот-нотификации. |
| `OPENAI_API_KEY`, `GROK_API_KEY` (или `GROK3_API_KEY`), `GEMINI_API_KEY`, `QWEN_API_KEY`, `DEEPSEEK_API_KEY`, `CLAUDE_API_KEY` | Ключи LLM-провайдеров. |
| `X_BEARER_TOKEN` | Для X/Twitter-скрейпинга (в config выключен по умолчанию). |
| `HTTP_PROXY`, `HTTPS_PROXY` | Подхватываются REST-клиентами (Binance/Coinalyze). |
| `PYTHONPATH=/app` | Для сервисов с импортами `services.*` (задаётся в compose). |

> ⚠️ В `settings.yml` зафиксированы **plaintext-секреты** (13 ключей Coinalyze, `qwen_api_key`). Их следует вынести в env и никогда не коммитить.

---

## 6. Shared-пакет (`shared/`)

Распространяется как `analeyes-shared` (см. `setup.py`), но **импортируется как `shared.*`** (layout `src/shared/...`). Устанавливается editable (`pip install -e ./shared`) в Docker-образах.

### `shared/config.py` — `Config`
- Загрузка `config/settings.yml` с fallback'ом на basename в CWD.
- `get(key, default)` — dot-нотация (`'ai_engines.use_grok'`), обходит отсутствующие сегменты.
- `set`, `save` (yaml-dump), `all` (deep-copy).
- Секционные геттеры слияния YAML + env: `get_binance_config`, `get_telegram_config`, `get_ai_config`, `get_database_config`, `get_feature_config`, `get_indicator_config`, `get_correlation_config`, `get_filter_config`, `get_composite_score_config`, `get_market_global_conditions_config`, `get_history_snapshots_config`, `get_trend_impulse_config`, `get_ml_proxy_config`, `get_derivatives_config`, `get_okx_config`, `get_x_heat_config`, `get_session_regimes_config`, `get_signal_tracker_config`, `get_collector_config`, `get_forced_trade_parameters_config`.
- **Нет** pydantic-валидации схемы — только YAML-парс и duck-typing.

### `shared/utils/pika_client.py`
- `DataEncoder(json.JSONEncoder)` — сериализует `datetime` (isoformat), numpy-массивы/числа, `set`→list, pydantic-модели (`.dict()`), fallback `str(obj)`. Любой объект с методами `isoformat`/`tolist`/`item`/`dict` будет молча приведён.
- `RabbitMQClient(amqp_url)` — async-обёртка над pika `AsyncioConnection`:
  - Авто-реконнект (5с) при обрывах connection/channel.
  - `publish(exchange, routing_key, body)` — объявляет topic-exchange `analeyes_exchange` (durable) и публикует с `delivery_mode=Persistent`, `content_type=application/json`.
  - `start_consuming(queue, cb, exchange, routing_key, durable, exclusive, auto_delete, prefetch_count)` — объявляет exchange/queue/binding, `basic_qos(prefetch_count)` (по умолчанию 1), запоминает setup для восстановления после реконнекта.
  - `close()` закрывает channel+connection.

### `shared/utils/logger.py`
- `ShortenNameFormatter` — отрезает префикс `src.` из имени логгера (`record.shortname`), маппит `__main__`→`main`. Подавляет шумные библиотеки (websockets/aiohttp/pika/httpx).
- `configure_logging_globally(config_dict)` — настраивает root-логгер (консоль + опционально ротируемый файл).
- `get_logger(name)` — passthrough.

### `shared/utils/circuit_breaker.py`
- `CircuitState(Enum)`: CLOSED/OPEN/HALF_OPEN.
- `CircuitBreaker(max_errors=10, reset_interval_s=60, half_open_attempts=3)`.
- `can_proceed()`, `record_success()`, `record_error(error_code=None)` (особое логирование для 418/429 — лимиты Binance), `get_stats()`, `wait_if_needed(max_calls_per_minute)` (naive fixed-window rate-limit).

### `shared/utils/rate_limiter.py`
- `RateLimiter(rate, per_seconds=1.0, burst_allowance=None)` — token-bucket на `asyncio.Lock`. `acquire(tokens)`, `can_proceed(tokens)`, `reset()`, async-свойство `available_tokens`.

### `shared/database/db_manager.py` — `DatabaseManager`
- Синглтон: class-level `_engine`, `_async_session_maker`.
- `initialize()` — читает `DATABASE_URL` (критический exit если не задан), `create_async_engine(url, pool_size, max_overflow, pool_recycle=3600, pool_pre_ping=True)`, тест коннекта, `async_sessionmaker(expire_on_commit=False)`, `Base.metadata.create_all`.
- `get_session()` — async-generator: commit на успехе / rollback на ошибке. Бросает `RuntimeError` без `initialize()`.
- `log_performance_metric(event_data)` — fault-tolerant запись в `performance_logs`.
- `close()` — dispose движка.

### `shared/rag/rag_manager.py` — `RAGManager`
- ChromaDB `HttpClient(host, port)` + OpenAI-эмбеддинги (через `rag.openai_base_url`).
- `add_signal_document(signal_log)` — индексирует завершённый сигнал.
- `query_similar_signals(features, top_n)` — возвращает список `signal_log_db_id` похожих сигналов.
- Самоотключается при отсутствии ключа/недоступности Chroma.
- **Не интегрирован в ai-service на текущем уровне** — `rag_context` нигде не наполняется; индексация выполняется из tracker-service на TP/SL.

### `shared/alembic/` + `alembic.ini`
- `env.py` грузит `.env.local`/`.env`, импортирует `Base.metadata`, в online-режиме подменяет URL на `DATABASE_URL` (меняя `+asyncpg` → `+psycopg2` для синхронного драйвера).
- Три миграции, базовая ревизия `3cbcf834c770` (полная начальная схема).

---

## 7. collector-service

**Роль:** единая точка сбора рыночных и контекстных данных, источник истины по текущему состоянию рынка для всего конвейера.

### Источники данных
| Источник | Метод | Что получает |
|---|---|---|
| **Binance Futures REST** (`fapi.binance.com`) | REST + HMAC-SHA256 для подписанных эндпоинтов | `exchangeInfo`, `ticker/24hr`, `ticker/price`, `premiumIndex` (ставка фандинга), `klines`, `openInterest`, `futures/data/openInterestHist`. |
| **Binance Futures WebSocket** (`wss://fstream.binance.com/stream`) | WS | `kline_1m/5m`, `depth5@100ms`, `aggTrade`, `markPrice@1s`, `!forceOrder@arr` (ликвидации), для permanent-пар — дополнительно `trade` и `bookTicker`. |
| **Coinalyze** | REST (ключи ротируются) | Исторические ликвидации (long/short USD, бакеты 5min, окно 1ч). |
| **OKX** | REST (`/api/v5/market/tickers?instType=SWAP`) | Эвристика объёмного всплеска: `volCcy24h > 50M USD` и `|Δцены|>15%`. |
| **CoinGecko** | REST `/global` | Изменение доли стейблкоинов (USDT/USDC/DAI/TUSD/FDUSD). |
| **TradingView econ calendar** (неофиц.) | REST с подделанными browser-headers | Макрособытия (US/EU/CN), `is_event_imminent()` — в течение `lookahead_minutes`=15. |
| **X/Twitter v2** (через `tweepy`) | API | Кэштеги из твитов заданных инфлюенсеров за 15мин; `hype_threshold`=3. Выключено по умолчанию (`X_BEARER_TOKEN`). |
| **btcparser.com** | HTML-скрейпинг | Транзакции BTC-китов с фильтрами по кошелькам (min_tx_count, unique_inputs, hold_time, balance, inflow, diff inflow/outflow). Фильтры **захардкожены**. |

### Ключевые классы

**`BinanceCollector`** (`collectors/binance_collector.py`, ~1845 строк) — центральный оркестратор:
- Создаёт N (по умолчанию 5) экземпляров `BinanceWebSocketClient` и распределяет пары round-robin.
- Управляет под-клиентами (Coinalyze/OKX/XHeat/CoinGecko/BTCParser/Calendar).
- `_make_rest_request` — единый REST-враппер: CircuitBreaker + RateLimiter, подпись, учёт `X-MBX-USED-WEIGHT-1M`, авто-исключение символов с `-4108`, performance-логирование (опционально).
- `stream_processed_pair_data()` — главный генератор цикла: возвращает `(symbol, data_dict)` для каждой пары. Реализует **каскад источников klines** с явными тегами `*_source`: `REST_Prefetched_Full → WebSocket_Sufficient → REST_Prefetched_Partial → WebSocket_Partial_NoRESTPrefetch → None`.
- `calculate_global_market_conditions()` — через `app_context.feature_generator` выводит глобальное состояние рынка (TREND/FLAT/TRANSITION) по прокси-символу (BTC) и активность (VERY_ACTIVE…VERY_PASSIVE) агрегацией per-pair режимов.

**`BinanceWebSocketClient`** (`collectors/binance_websocket.py`, ~1041 строк):
- Поддержка 3 зеркал Binance (`wss://fstream-mr.binance.com`, и т.п.), ping_interval=30, ping_timeout=40.
- Ротация символов: permanent-subset + top-50 priority + pool 150 с батчами по 50, ротация каждые 300с.
- `_resubscribe_manager` каждые 5с — переподписка проблемных потоков.
- Bounded in-memory хранилища (klines, depth, agg_trades, mark_price, liquidations).
- **⚠️ `_backfill_missing_data` реализован, но вызов закомментирован** — при разрыве WS дыры в данных НЕ заполняются через REST.

### Цикл (`main.py`)
1. Connect RabbitMQ → spawn `_status_updater` (5с, RK `status.collector`).
2. spawn `_websocket_restarter` (каждые `websocket_restart_interval_s`=300с — профилактический рестарт WS).
3. Опц. `perform_initial_historical_backfill()` — тяжёлый стартовый прогрев.
4. `start_websocket()` один раз.
5. Loop:
   - increment cycle, mint `current_cycle_signal_id_uuid`.
   - `async for symbol, data in collector.stream_processed_pair_data(): publish({symbol, data}, RK='data.raw.binance')`.
   - sleep `system.cycle_interval_s`=60с.

### RabbitMQ
- **Публикует:** `data.raw.binance` (пейлоуд `{symbol, data:{...}}`), `status.collector`.
- **Не потребляет** ничего.

### Нюансы
- `DatabaseManager` **не инстанцируется** в collector — хуки performance-логирования не активны (`db_manager=None`).
- `BtcTransaction` — plain dataclass без `.dict()` → `DataEncoder` stringify'ит его через `str()`. Возможен артефакт в JSON-пейлоуде.
- Прокси `HTTP(S)_PROXY` подхватывается REST-клиентами.
- Старая эвристика «early signal» собирается одним `asyncio.gather` параллельно для всех внешних клиентов.

---

## 8. processing-service

**Роль:** превращает сырые данные в структурированные признаки, фильтрует пары, считает составной рейтинг, пишет `signal_feature_logs` и отдаёт кандидатов в AI.

### Файлы
| Файл | Назначение |
|---|---|
| `src/main.py` | App: RabbitMQ consumer/publisher, orchestration. |
| `src/logic/feature_generator.py` (~2318 строк) | Все технические/микроструктурные/производные фичи. |
| `src/logic/pair_filter.py` (~787 строк) | Многоуровневая фильтрация с динамическими порогами. |
| `src/logic/risk_profile_manager.py` | Per-symbol профили риска из истории win-rate. |
| `src/logic/patterns.py` | Свечные паттерны TA-Lib + консенсус. |
| `src/logic/indicators.py` | 8 индикаторных стратегий + консенсус. |
| `src/logic/composite_score.py` | Взвешенный итоговый рейтинг. |
| `src/logic/ml/train_lstm.py` | Оффлайн-тренер LSTM/GRU (не в runtime). |
| `src/logic/ml/lstm_proxy.py` | Stub: всегда возвращает 0 (LSTM-фильтр неактивен). |

### `feature_generator.py` — что считается

**TA-Lib на primary klines (5m):** RSI(14), MACD(12/26/9), EMA(8/21), BBANDS(20,2) + `bb_width`, OBV, ADX(14), ATR(14) (нормализованный к цене и SMA), price slope (`scipy.stats.linregress`), VWAP (ручной), Supertrend (упрощённо — только полосы `median ± mult*ATR`), Ichimoku (Tenkan/Kijun/Senkou A,B **без смещения**), Stochastic(14/3/3).

**Уровни:** Fibonacci (23.6/38.2/50/61.8 %), Support/Resistance (min/max за 20), rolling-rank (percentile).

**Объём/OI/цена:** Volume Z (1m+5m, выбор большего по модулю; кэшируется внешним `update_rolling`), OI Z (USD), price_pct, price_pct_norm (`/ (atr/mark*100)`), delta_OI×price product, vol_rel (1m объём / средний за 24ч).

**Микроструктура:** CVD (Σ taker buy − sell за последние 6 свечей 5m), spread (из depth или fallback), trade_rate_per_min, large_trade_ratio, large_sweep_flag, OB imbalance, OFI delta, QPC delta, hidden_order_proxy, order_flow_ratio, whale_trades_proxy, retail_impulse_proxy (z-score count и avg_size).

**Производные:** ликвидации (longs/shorts USD), funding_rate, funding_momentum (Δ за 3ч), IVR_premium_rank (`(curr-min)/(max-min)`), RVR (`std_short/std_long` лог-ретёрнов), basis и funding_basis_product, implied_funding, BVOL_proxy_z (только для BTC), sentiment_skew_proxy, **`btc_whale_flow` (⚠️ содержит баг — ссылается на `self.app_context`, которого у FeatureGenerator нет; AttributeError при наличии `btc_transactions` в данных)**.

**Время:** day-of-week one-hot (7 ключей), time_of_day_factor (asiatic=0.8 / european=1.0 / us=1.1).

**Определение режима рынка** (`detect_market_state`) — голосование 4 классификаторов:
- ADX-based: <20 → WEAK_TREND(0.5 в FLAT), ≥25 → TREND(1.0), иначе TRANSITION.
- BB-width: `<0.02` FLAT, `>0.05` TREND.
- ATR ratio: `atr/atr_sma > 1.1` TREND, `< 0.9` FLAT.
- Price slope: нормированный наклон, пороги 0.05% / 0.1%.

Финальное состояние: только если есть однозначный победитель и `max_votes >= min_votes_for_clear_state` (2.0), иначе TRANSITION. Состояния: `FLAT`, `TREND`, `TRANSITION`.

**EIS (Exponential Impulse Score):** `λ = 0.5 ** (cycle_interval_s / (half_life_min*60))`; сигнал за цикл = `sign(price_pct) × sqrt(|vol_z|) × (1 + |oi_z|/oi_weight_div)`; `eis_now = eis_prev*λ + signal_val`.

**Снимки/ускорения:** `feature_snapshot_buffer` хранит последние N (3) снимка; `acc_vol`, `acc_price`, `acc_oi_z`, `acc_cvd` — разница последних двух.

**Гарантия схемы:** в конце `generate_features` все ~60 ключей из `expected_feature_keys` устанавливаются через `setdefault(None)` — стабильный контракт для downstream.

### `pair_filter.py` — многоуровневые пороги

Пайплайн корректировки порогов:
1. Базовые пороги из `filters.*`.
2. Категория по капитализации (`thresholds_by_coin_size.{large,medium,small}`).
3. Сессионные режимы (`session_regimes`).
4. Глобальная матрица `market_global_conditions.threshold_adjustment_matrix[state][activity]` — мультипликатор на каждый порог (`spread_pct` делится, остальные умножаются).
5. Динамические модификаторы: risk_profile.adjustment_factor (HIGH_TRUST 0.9, NEUTRAL 1.0, LOW_TRUST **1.2** — ослабление порогов для «плохих» монет), gamma-proxy (×0.8 при высоком фандинге), X-Heat (×0.85), time-of-day factor, RVR-корректировка.

**Фильтры (все должны пройти, если не сработал acceleration fast-pass):**
`is_not_already_tracked`, `glitch_spike` (защита от фантомных гигантских сделок), `lstm_trend_agreement` (stub, всегда pass), `volume_z_score`, `volume_relative`, `price_pct_normalized`, `price_percentage_change`, `open_interest_z_score`, `liquidations_or_funding_rate` (OR), `spread_percentage`, `ofi_delta`, `large_sweep` (опц.), `monotonic_history`, `eis_absolute_value`.

**Acceleration Fast-Pass:** если `|acc_price| > 0.18` или `|acc_vol| > 0.4` — все критерии авто-pass.

**⚠️ Нюанс:** LOW_TRUST получает **более мягкие** пороги (фактор 1.2) — система даёт шанс подопечным, а не отсекает их. Документировано, но контринтуитивно.

### `patterns.py`
- 18 свечных паттернов TA-Lib (см. `patterns.enabled`).
- Вес паттерна = `|talib_value|/100 × strength_weight`, clamped [0,1].
- Консенсус: порог `consensus_strength_ratio_threshold`=1.5 — если `long_sum > short_sum*1.5` → LONG; симметрично SHORT; нейтральные (Doji/DojiStar/HaramiCross) — `NEUTRAL_PRESENCE`; иначе `MIXED` или `NONE`.

### `indicators.py`
8 стратегий: EMA-cross, RSI conditions, MACD signals, BB breakouts, VWAP relation, Supertrend (упрощённо), Ichimoku TK-cross (с подтверждением облаком), Stochastic. Каждая возвращает `{signal, strength, details}`. Консенсус: большинство → LONG/SHORT; равенство → MIXED; нет сигналов → NONE.

### `composite_score.py`
- Веса (нормализуются к 1.0): vol_rel 0.22, acc_vol 0.15, price_pct_norm 0.15, acc_price 0.12, oi_z 0.10, ofi_delta 0.10, patterns 0.10, eis 0.07, order_flow_ratio 0.05, cvd 0.0, liq 0.0.
- `_normalize_metric`: для `order_flow_ratio` — отклонение от 1.0; для `spread_pct` — `1 - |v|/thresh`; для `ivr_premium_rank` — `|v-0.5|*2`; иначе `min(1, |v|/thresh)`.
- **Эвристический консенсус сигнала:** голосуют {patterns, indicators, price-direction (LONG если `price_pct > 0.3`, SHORT если `< -0.3`)}; большинство (≥2 из 3) побеждает.
- `get_top_candidates(score_results_map, threshold, max_candidates=10)` — топ-N кандидатов по убыванию.

### `main.py` — пайплайн
- **Потребляет:** `q_raw_data` ← `data.raw.*`; `q_tracker_status_for_processing` ← `status.tracker` (синхронизирует `tracked_symbols_cache`).
- **Публикует:** `data.candidates.ai` (пароль pass + composite ≥ `composite_threshold`=0.78); `status.processing` (5с).
- Для **каждой** пары: `feature_generator.generate_features` → обновить `global_market_state` (только для BTC) → `pair_filter._check_criteria` → `composite_score.calculate_score` → `_save_signal_to_db` (пишет `signal_feature_logs` **до** gate по composite) → если ≥ порога, publish в AI.

### ML-заглушки
- `train_lstm.py` — оффлайн-тренер Keras (LSTM/GRU на 30 1m-свечах). Требует `tensorflow` (нет в requirements). Сохраняет `lstm_trend_v1.keras` + scaler.
- `lstm_proxy.py` — stub, всегда возвращает 0. Модель не загружается. Фильтр никогда не ветируется.

### Нюансы
- Risk profiles загружаются **один раз при старте** (`update_profiles`) и не обновляются — возможно устаревание win-rate для долгоживущих процессов.
- `_calculate_btc_whale_flow` — **латентный AttributeError** (см. выше).
- `_calculate_eth_btc_spread_z` определён, но **не вызывается** из `generate_features` — всегда None.
- `_should_scan_pair` (адаптивный планировщик сканирования) — **мёртвый код** (определён, но не вызывается).
- global_market_state обновляется **только по BTC**; если поток BTC прерван, остальные символы используют устаревшее состояние.
- Cold start: `acc_*` = 0 первые 2 цикла на символ, может искажать рейтинг.
- Numpy pinned `<2.0` для совместимости с TA-Lib 0.4.28; C-lib собирается с `CFLAGS=-Wno-incompatible-pointer-types`.

---

## 9. ai-service

**Роль:** ансамбль LLM-движков, параллельный запрос, голосование за направление, усреднение параметров сделки.

### Файлы
| Файл | Назначение |
|---|---|
| `src/main.py` | `AIServiceApp`: RabbitMQ consumer/publisher, DB persistence. |
| `src/logic/ai/ai_analyzer.py` | `AIAnalyzer`: fan-out + consensus. |
| `src/logic/ai/prompt_builder.py` | `PromptBuilder`: общий блок рыночного контекста. |
| `src/logic/ai/gpt_client.py`, `grok_client.py`, `gemini_client.py`, `claude_client.py`, `deepseek_client.py`, `qwen_client.py` | LLM-клиенты. |
| `src/logic/ml/ml_proxy.py` | LightGBM-фильтр (определён, **не подключён** в main). |
| `src/logic/ml/train_lgbm.py` | Оффлайн-тренер LightGBM. |
| `src/logic/ml/rl_exit/agent.py` | PPO-агент раннего выхода (deps не установлены). |

### LLM-клиенты — сравнение

| Клиент | SDK / endpoint | Env | Деф. модель | Timeout | max_tokens | temp | JSON-enforce | Markdown-strip | Режим отказа |
|---|---|---|---|---|---|---|---|---|---|
| **GPTClient** | `openai.AsyncOpenAI` → **`https://gemini.dkms.cc/openai/v1`** (захардкоженный прокси) | `OPENAI_API_KEY` | `gpt-4-turbo-preview` | 60 (hard) | — | — | `response_format json_object` | нет | SKIP dict |
| **GrokClient** | `openai.AsyncOpenAI` → `https://api.x.ai/v1` | `GROK_API_KEY`/`GROK3_API_KEY` | `grok-3-latest` | 10 | 450 | 0.4 | json_object | да (```json) | SKIP dict |
| **GeminiClient** | `google.generativeai` 0.5.x | `GEMINI_API_KEY` | `gemini-1.5-flash-latest` | 20 (asyncio.wait_for) | 450 | 0.3 | `response_mime_type=application/json` | нет | SKIP dict |
| **ClaudeClient** | `openai.AsyncOpenAI` → YAML `claude_base_url` | `CLAUDE_API_KEY` | `claude-3-haiku-20240307` | 20 | 1024 | 0.3 | нет (закоментировано) | да | dict с error |
| **DeepSeekClient** | `openai.AsyncOpenAI` → `https://api.deepseek.com` | `DEEPSEEK_API_KEY` | `deepseek-chat` | 15 | 1024 | 0.3 | json_object | нет | dict с error |
| **QwenClient** | `openai.AsyncOpenAI` → `dashscope-intl.aliyuncs.com/...` (захардкожено) | `QWEN_API_KEY` | `qwen-max` | 15 | 450 | 0.3 | json_object | нет | SKIP dict |

**Унифицированный выходной JSON-контракт:**
```json
{
  "signal": "LONG|SHORT|SKIP",
  "confidence": 0.0,
  "abstain": false,
  "reason": "...",
  "override_direction": false,
  "override_reason": null,
  "entry_price": "market|number|null",
  "tp": {"p10":..,"p50":..,"p90":..},
  "sl": {"p10":..,"p50":..,"p90":..},
  "horizon_min": null,
  "leverage": null,
  "data_warnings": []
}
```
Парсеры валидируют `signal ∈ {LONG,SHORT,SKIP}`, числовой `confidence` (clamp [0,1]), строковый `reason`. При ошибке — возвращают error/SKIP dict, **никогда не бросают**.

### `AIAnalyzer.analyze_single`
- Все включённые клиенты запускаются **одним** `asyncio.gather(..., return_exceptions=True)` — отказ одного не влияет на других.
- Per-client timeout находится **внутри** клиентов (на уровне оркестратора — нет).
- Каждый ответ обогащается `signal_id`, `symbol`, `composite_score_data` (полный strategies-дикт).

### `_aggregate_ai_responses` (голосование)
- Кворум ≥2 валидных ответов.
- Плюральность LONG vs SHORT. **Ничья → SKIP** (сигнал отбрасывается).
- Числовое усреднение **только по моделям, совпавшим с победившим направлением** (проигравшие исключаются).
- `avg_confidence` (среднее, 2 знака), `avg_tp`/`avg_sl` (4 знака), `avg_leverage` (int), `tp_distribution`/`sl_distribution` (по квантилям).
- Синтетическая строка: `source='ensemble'`, `model='AnalEyes-Mix'`, `entry='market'`, `cost=0`.

### `main.py`
- **Потребляет:** `q_ai_candidates` ← `data.candidates.ai`.
- Для каждого валидного результата (signal ≠ SKIP): `_adapt_ai_result_to_ensemble_format` → `_save_ensemble_model_decision` (пишет `EnsembleModelDecision`, composite_id = `{signal_id}_{source}`) → publish `signal.final`.
- **Всегда basic_ack** в finally — poison-message-safe, но при персистентных ошибках сообщение теряется (нет DLX).
- Status: `status.ai` каждые 5с.

### Промпты (`config/prompts/`)
| Файл | Содержание |
|---|---|
| `gpt_prompt.txt` | Полный шаблон + JSON-схема + LONG-bias пример. |
| `grok_prompt.txt` | **Только `{PROMPT_BODY}`** — инструкции в коде клиента. |
| `gemini_prompt.txt` | **Только `{PROMPT_BODY}`** — инструкции инлайн. |
| `claude_prompt.txt` | Полный шаблон + JSON-схема + LONG-bias пример. |
| `deepseek_prompt.txt` | Полный шаблон + LONG-bias пример. |
| `qwen_prompt.txt` | **Только `{PROMPT_BODY}`** — инструкции инлайн. |

**⚠️ Асимметрия few-shot:** примеры TP/SL генерируются с разными смещениями по направлениям (GPT/Claude/DeepSeek/Qwen — LONG-bias, Grok/Gemini — SHORT-bias). Может влиять на голосование.

### Нюансы
- **Hardcoded OpenAI-прокси** `https://gemini.dkms.cc/openai/v1` в GPTClient — трафик и `OPENAI_API_KEY` идут через сторонний домен. Тот же домен фигурирует в RAGManager.
- **Gemini BLOCK_NONE** по всем 4 категориям safety — фильтры отключены полностью.
- **GPT `timeout=60` захардкожен** — игнорирует `timeouts.ai_s` (несогласованность с другими клиентами).
- **GPT без `max_tokens`** — ответ может быть длинным, стоимость не ограничена сверху.
- **Нет rate-limit/circuit-breaker** на уровне AI — деградирующий провайдер генерит ошибки бесконечно.
- **`analyze_failure` режим** реализован только в Gemini + Qwen (для post-mortem сделок).
- **RAG-планирование есть, но не наполнено** — `rag_context` проходит через сигнатуры, но `main.py`/`AIAnalyzer` никогда его не задают.
- **Markdown-fence inconsistence** — Grok/Claude чистят ```json; GPT/DeepSeek/Qwen/Gemini — нет (падают на fenced JSON, логируется как ошибка → SKIP).
- **Уверенность — невзвешенное среднее** — слабая модель (haiku/flash) весит столько же, сколько GPT-4.
- **`short_mode`** параметр нигде не вызывается — мёртвый путь.

---

## 10. notification-service

**Роль:** форматирование и доставка сигналов/исходов/входов в Telegram (форум-топики), генерация графиков.

### Файлы
| Файл | Назначение |
|---|---|
| `src/main.py` | `NotificationServiceApp`: 3 consumer'а RabbitMQ + DB-апдейты. |
| `src/logic/telegram/telegram_sender.py` (~1297 строк) | `TelegramSender`: форматирование, графики, retry/fallback. |

### `main.py`
- **Потребляет:**
  - `q_new_signals` ← `signal.final`.
  - `q_signal_outcomes` ← `signal.outcome`.
  - `q_signal_entry_events` ← `signal.entry_event`.
- **Публикует:** `status.notification` (5с).
- Routing: `topic_id = ai_specific_topics.get(source_ai)`; если нет маппинга или нет group_id — скип отправки, но DB всё равно апдейтится (`_update_signal_log_for_skipped_or_unnotified`).
- После отправки сигнала обновляет `signal_feature_logs`: `telegram_message_sent=True`, `ai_signal_type`, `ai_confidence`, `ai_reason_summary`, `ai_entry/tp/sl_price_suggestion`, `ai_leverage_suggestion`, `ai_consensus_achieved`, `ai_tp_p10/p50/p90`, `ai_sl_p10/p50/p90`.

### `telegram_sender.py`
- `markdown_protect(text)` — экранирование specials MarkdownV2.
- `_format_price(price)` — динамическая точность по величине (≥1000→2dp, ≥1→4dp, …).
- `_extract_and_validate_ai_recommendations` — парсинг и валидация TP/SL/entry/leverage (clamp 1–125). **Forced TP/SL override** если `forced_trade_parameters.enabled`: LONG `tp = entry*(1+(tp_pct/100)/lev)`, SL симметрично в обратную сторону.
- `_format_signal_message` — 3 куска: пользовательский (🚀 заголовок, монета, AI, направление, уверенность %, market state, EIS-блок с emoji-полосами, entry/SL/TP/leverage, AI-комментарий курсивом, ссылка на Binance Futures, UTC-время), signal-debug (детальные метрики), ai-debug (prompt+response превью per engine).
- `_generate_signal_chart_image` — mplfinance + matplotlib + pandas. Тёмная тема. Маркеры: suggested entry (белая точка), TP/SL (зелёный/красный пунктир), actual entry (золотая), exit (фиолетовая). Сохраняет PNG во `tempfile.gettempdir()`, DPI 200.
- `send_message` — чанкование по 4096 символов с учётом ```-блоков; retry на "group migrated" (парсинг нового chat_id), на markdown-ошибке fallback на plain text (strip specials), на network errors экспоненциальный backoff `2^retry + 1` до `retry_attempts`=3.
- `send_signal` — **⚠️ генерация графика для новых сигналов закомментирована/stub** — переменная инициализируется None, код заменён на `...`. Графики реально генерируются только для outcome и entry_event.
- `send_signal_outcome` — форматирование результата с emoji-картой статусов (TP_HIT ✅, SL_HIT ❌, EXPIRED ⏳, CLOSED_BY_RL 🤖, CANCELLED_UNFILLED 🚫), PnL-строка относительно `$500` (`default_bank_for_pnl_calc_usd`=500), длительность.
- `send_error_alert`, `send_status_update` — admin-only.
- `send_entry_event_notification` — NEW: уведомление о входе в сделку.

### Нюансы
- Списки `new_signal_channel_ids` и `outcome_alert_channel_ids` **не используются** в текущем flow — только `group_id` + forum topics. Списки — vestigial.
- `on_entry_event` лишён SKIP-гарда и DB-апдейта — чисто форвард в Telegram.
- Дубликат логики DB-апдейта между success-путём и `_update_signal_log_for_skipped_or_unnotified`.
- Dockerfile CMD `services.notification-service.src.main` — дефис в пути (работает через `-m`-filesystem).

---

## 11. tracker-service

**Роль:** конечный автомат (FSM) отслеживания каждого сигнала от входа до исхода (TP/SL/EXPIRE/CANCEL).

### Файлы
| Файл | Назначение |
|---|---|
| `src/main.py` | `SignalServiceApp`: lifecycle, batch processor, status publisher. |
| `src/metrics.py` | Prometheus-метрики. |
| `src/logic/signal_tracker.py` (~755 строк) | `SignalTracker`, `TrackedSignal`, `SignalState` FSM. |
| `src/collectors/binance_collector.py` | WS+REST ценник. |
| `src/collectors/binance_websocket.py` | `SimpleBinanceWebSocketClient`: depth5 + markPrice. |
| `src/logic/rl_exit/agent.py` | **Мёртвый код** — импортирует `stable_baselines3`, не в requirements. |

### `SignalState` enum
`WAITING_FOR_ENTRY`, `ENTERED`, `TP_HIT`, `SL_HIT`, `EXPIRED`, `CANCELLED_UNFILLED`.

### `TrackedSignal`
- Поля: `id` (UUID), `symbol`, `direction`, `entry_price` (float | `'market'`), `tp_price`, `sl_price`, `signal_time`, `expiration_time` (`+24h`), `entry_deadline` (`+entry_timeout_sec`=300с), `fill_price`, `actual_entry_time`, `hit_price`, `hit_time`, `duration_seconds`, `entry_notified`, `leverage`, `source_ai`, `signal_log_db_id`, `slippage_pct`=0.01, `max_age_ms`=4500, `prefer='tp'`.
- Валидация TP/SL относительно entry (warn при неверной стороне, skip для `'market'`).
- `get_current_price_from_pair_data` — каскад: markPrice WS → depth mid → REST ticker → ticker_24h.lastPrice. Проверка свежести (`max_age_ms`); источники записываются в `WS_PRICE_AGE_MS_HISTOGRAM`.

### `process_market_data` — FSM
**WAITING_FOR_ENTRY:**
- `now > entry_deadline` → CANCELLED_UNFILLED.
- `'market'` → вход сразу. LONG: `price ≤ entry*(1+slippage)`. SHORT: `price ≥ entry*(1-slippage)`. → переход в ENTERED, возврат `(None, current_price)` (transition event).

**ENTERED:**
- `now > expiration_time` → EXPIRED.
- TP/SL crossing: LONG `tp_crossed = current ≥ tp`, `sl_crossed = current ≤ sl`; SHORT — наоборот.
- **Collision (оба за один тик)** — warn, honour `self.prefer` (`'tp'` или `'sl'`).

### `SignalTracker`
- `active_tracked_signals: Dict[str, TrackedSignal]` — ключ `f"{signal_id}-{source_ai}"` (мульти-AI ансамбль — один signal_id трекается отдельно per model).
- `start_tracking_signal` — дедупликация по ключу (`DUPLICATE_SIGNAL_TOTAL_COUNTER.inc()`).
- `check_tracked_signals(market_data_map)` — каждый polling cycle (10с по умолчанию): итерация копии активных сигналов, вызов `process_market_data`, при переходе в ENTERED — `publish_entry_to_rabbitmq` (один раз, флаг `entry_notified`), при финальном исходе — `_handle_signal_completion`.

### `_handle_signal_completion`
- `_update_signal_log_in_db` — апдейт `signal_feature_logs.tracker_*`.
- `_save_ensemble_backtest_result` — вставка `EnsembleBacktestResult` (skip CANCELLED_UNFILLED; skip если PnL None).
- **RAG indexing** при TP_HIT/SL_HIT и `rag_manager.enabled` → `rag_manager.add_signal_document`.
- `outcome_callback` → `app.publish_outcome_to_rabbitmq` → publish `signal.outcome`.
- Инкремент Prometheus-счётчиков.

### `_calculate_pnl`
- LONG: `(exit-entry)/entry*leverage*100` (%), `(exit-entry)/entry*bank*leverage` (USD), `bank = trading.default_initial_bank_usd`=500.
- SHORT: знак развёрнут.

### Binance клиент трекера
- WS подписки: `{symbol}@depth5@100ms` и `{symbol}@markPrice@1s`.
- REST fallback `/fapi/v1/ticker/price` с кэшем TTL 2000мс, семафор 5 concurrent.
- Gap detection по `first_update_id > last_id + 1` → ресабскрайб.

### Prometheus-метрики
`tracker_active_signals` (Gauge), `tracker_duplicate_signal_total`, `ws_price_age_ms` (Histogram), `tracker_tp_hit_total`, `tracker_sl_hit_total`, `tracker_expired_signal_total`, `tracker_cancelled_unfilled_total`.

### Нюансы
- Дедуп-ключ изменился с `signal_id` на `{signal_id}-{source_ai}` — один сигнал трекается per-model.
- `SignalTracker.publish_outcome_to_rabbitmq` (метод трекера) — **мёртвый код**, затенён app-level callback.
- Config-ключи в трекере (`rabbitmq.exchange_name`, `rabbitmq.publisher_routing_key`) — **dead path** (app использует хардкод).
- `slippage_pct` default 0.001 в `SignalTracker`, но 0.01 в `TrackedSignal`-конструкторе — эффективно из config.
- `max_age_ms` default 5000 в трекере, 4500 в `TrackedSignal`-конструкторе — эффективно из config.
- `EnsembleBacktestResult` исключает CANCELLED_UNFILLED — только реально вошедшие.
- RAG индексирует только TP/SL (не EXPIRED) — база похожих сигналов смещена к решительным исходам.
- `agent.py` RL-exit — **битый** (импорт stable_baselines3, не в requirements).

---

## 12. backtest-service

**Роль:** детерминированный симулятор исходов по историческим сигналам.

### Файлы
| Файл | Назначение |
|---|---|
| `src/main.py` | `BacktestServiceApp`: consumer/publisher, lifecycle. |
| `src/logic/backtest_runner.py` | `BacktestRunner`: собственно симуляция. |

### `main.py`
- **Потребляет:**
  - `q_start_backtest` ← `control.backtest.start`.
  - `q_backtest_analysis_results` ← `ai.backtest.analysis.result` (LLM-разбор неудач).
- **Публикует:** `status.backtest` (2с, только в статусе RUNNING).
- **Single-flight** — один бэктест на процесс; конкурентные start-команды ack-and-drop (busy reject silent).

### `BacktestRunner.run`
- Парсит `parameters['start_date'/'end_date']`, опц. `symbols`.
- Один DB-сессия на весь прогон: `SELECT * FROM signal_feature_logs WHERE created_at BETWEEN start AND end [AND symbol IN (...)] ORDER BY created_at ASC`.
- Per-row `_simulate_trade`.

### `_simulate_trade` — модель одной свечи
- Требуемые поля: `ai_tp/sl/entry_price_suggestion`, `ai_signal_type`, `ohlcv_high/low_price`.
- LONG: SL если `low ≤ sl`; TP если `high ≥ tp`. SL проверяется **первым** (пессимистично).
- SHORT: SL если `high ≥ sl`; TP если `low ≤ tp`.
- Дефолтный outcome — `EXPIRED`, exit_price = close или entry.
- `pnl_with_leverage = (exit-entry)/entry*100*leverage`.
- Записывает `BacktestResult` с **хардкоженным `duration_minutes=5`**.
- На `SL_HIT` и `ai_analysis.enabled` — `_request_failure_analysis` → publish `ai.backtest.analyze` (sync pika-publish).

### ⚠️ Важные оговорки
- **Одна свеча исхода** — проверка TP/SL против high/low **той же свечи**, что породила сигнал (не следующей). Документация в коде вводит в заблуждение.
- Нет slippage, комиссий, фандинга, ликвидаций. Только множитель leverage.
- `processed_signals` инкрементируется до early-return → искажает знаменатель `win_rate`.
- Status heartbeat только в RUNNING — idle-инстанс ничего не шлёт → health-чек будет показывать UNAVAILABLE даже на здоровом idle.
- `/api/health/full` намеренно **исключает** `backtest` из `expected_services` (закоментировано).
- LightGBM/`ml_proxy` в Dockerfile и config присутствуют, но сервис их **не использует** — чистый replay-симулятор.

---

## 13. api-gateway

**Роль:** FastAPI — REST API, WebSocket статуса, JWT-аутентификация, раздача `web-ui`.

### Файлы
| Файл | Назначение |
|---|---|
| `src/main.py` | FastAPI app + `StatusMonitor` (WebSocket). |
| `src/dependencies.py` | JWT, `get_current_user`, `get_admin_user`, `get_db`. |
| `src/schemas.py` | Pydantic (camelCase) схемы. |
| `src/endpoints/health.py` | `/api/health/full`. |
| `src/endpoints/users.py` | `/api/auth/users/auth` — login. |
| `src/endpoints/signals.py` | `/api/signals`, `/api/signal-results`, `/{id}/chart-data`. |
| `src/endpoints/keys.py` | CRUD exchange API keys. |
| `src/endpoints/autotrade.py` | Настройки авто-торговли. |
| `src/endpoints/settings.py` | Subscription, language. |
| `src/endpoints/stats.py` | PnL-статистика. |
| `src/endpoints/admin.py` | Signal display settings. |
| `src/endpoints/backtest.py` | Trigger backtest, get results. |

### Аутентификация
- **JWT** через `python-jose` (HS256, 7 дней, секрет `API_SECRET_KEY` — **обязателен**, иначе ImportError при старте).
- **Без паролей.** `User` идентифицируется по `telegram_id` (BigInteger). `POST /api/auth/users/auth {telegramId, username?}` upsert'ит юзера (+ trial subscription на 3 дня + default UserSettings) и возвращает JWT `sub=telegram_id`.
- ⚠️ **Эндпоинт `/auth` публичный** — кто угодно может получить токен по любому `telegram_id`. Доверие предполагается на стороне Telegram (Telegram Login Widget) — **HMAC-проверка здесь не реализована**.
- `get_admin_user` — сравнение `int(admin_id) == current_user.telegram_id` (403 иначе).
- `verify_health_check_token` — static `HEALTH_CHECK_TOKEN` (constant-time compare).

### Endpoints
| Метод | Путь | Auth | Назначение |
|---|---|---|---|
| POST | `/api/auth/users/auth` | — | Upsert + JWT. |
| GET | `/api/signals` | user | Список сигналов; если нет активной подписки — один DEMO-сигнал. |
| GET | `/api/signals/signal-results` | — | Закрытые сделки с фильтрами (`apply_filters`=true использует CTE-SQL с `RANDOM()` — пагинация недетерминирована). |
| GET | `/api/signals/{signal_id}/chart-data` | — | Прокси к Binance `/fapi/v1/klines`. **⚠️ Баг:** `Depends(Config)` вызовет `Config()` без пути → FileNotFoundError в большинстве деплоев. |
| GET/POST/DELETE | `/api/keys` | user | CRUD exchange API keys. `_validate_api_key` — stub, всегда True. |
| GET/PUT | `/api/autotrade/settings` | user | Auto-trade настройки (DB-only, **не** публикует в RabbitMQ). |
| POST | `/api/autotrade/start` `/stop` | user | Toggle `auto_trade_enabled`. |
| GET | `/api/settings/subscription` | user | 404 если нет. |
| PUT | `/api/settings/settings` | user | Update language. |
| GET | `/api/stats/pnl` | user | Aggregation over `trades` by period (today/7d/30d/all). |
| GET/PUT | `/api/admin/signal-display-settings` | admin | Filter rules. |
| POST | `/api/backtest/run` | — | Создаёт `BacktestRun`, публикует `control.backtest.start`. **⚠️ Без auth** — любой может запустить тяжёлый бэктест. |
| GET | `/api/backtest/{run_uuid}` | — | Статус + результаты. |
| GET | `/api/health/full` | health_token | DB + RabbitMQ + per-service liveness (60с stale). `expected_services = ['collector','processing','ai','tracker','notification']` — **backtest исключён**. |
| WS | `/ws/status` | — | `StatusMonitor` транслирует `{type:"update",service,data}` всем подключённым. |

### `StatusMonitor`
- Создаёт **эксклюзивную auto-delete** очередь `q_service_statuses_gw_{uuid6}` с биндингом `status.*`.
- Каждая реплика gateway получает свою копию каждого status-сообщения (fan-out).
- При подключении WS-клиента шлёт snapshot `{type:"full_status", data:statuses}`.

### Прочее
- **Static mount:** `app.mount("/", StaticFiles(directory=<repo>/web-ui, html=True))` — SPA-раздача `index.html` с fallback на 404.
- **CORS не настроен** — web-ui должен быть same-origin (что и обеспечивается static-mount'ом).
- **Pydantic схемы** — camelCase (`CamelCaseModel`); `SignalResultResponse.prepare_data` централизованно нормализует Decimal→float и ORM→dict.
- `BacktestRun.parameters`/`summary` хранятся как `JSONB`; `model_dump(mode='json')` перед публикацией.
- Python 3.10 (vs 3.11 в backtest) — skew версий.

### Нюансы
- Backtest ID-путаница: опубликованное сообщение использует integer PK `new_run.id` как `run_id`; ответ возвращает UUID `new_run.run_id`; runner использует integer для `session.get`. Именование сбивает с толку.
- Trial subscription создаётся автоматически на 3 дня; expiry проверяется только в `/signals`.

---

## 14. web-ui

`web-ui/index.html` — single-page дашборд (Catppuccin-подобная тёмная тема, aurora-градиент, шрифт Inter). Без билд-системы (без React/Vue) — чистый HTML/CSS/JS, раздаётся api-gateway через StaticFiles.

Возможности (по структуре):
- Табы переключения контента (с анимацией fadeIn).
- Карточки сервисов с live-статусами из `/ws/status`.
- Таблица результатов сделок с пагинацией.
- PnL-окрашивание (profit/loss).
- Connection-status индикатор (WS connect/disconnect).
- Иконки направлений (LONG/SHORT), статусов (TP/SL).
- Кликабельные строки таблицы (детали сделки).

Связь с бэкендом:
- WebSocket `/ws/status` для статусов.
- REST `/api/signals/*`, `/api/stats/pnl` и т.п.

---

## 15. Схема БД (ORM-модели)

Все в `shared/database/models.py` (Postgres-диалект, `JSONB`, `PG_UUID`, `ARRAY`). `OrmBase`: `id` (PK), `created_at`, `updated_at`.

| Модель | Таблица | Назначение |
|---|---|---|
| `CycleLog` | `cycle_logs` | Per-cycle статистика (cycle_num uniq idx). |
| `ArtefactLog` | `artefact_logs` | Промежуточные LLM/сырые артефакты (FK→signal_feature_logs). |
| `ErrorLog` | `error_logs` | Ошибки/исключения (module/function/line/traceback). |
| `OrmPerformanceLog` | `performance_logs` | Латентности/API-метрики (composite idx по event_type/symbol). |
| **`SignalFeatureLog`** | `signal_feature_logs` | **Центральная таблица** (~80 `feat_*` колонок + OHLCV + `strat_*` + `ai_*` + `tracker_*` + `composite_score_value`). Idx `(symbol, initial_metric_timestamp)`. |
| `User` | `users` | PK telegram_id (BigInteger uniq), username, language. |
| `Subscription` | `subscriptions` | plan='trial', expires_at, is_active (1:1 с User). |
| `ExchangeAPIKey` | `exchange_api_keys` | exchange, api_key, **api_secret plaintext** (документировано как уязвимость), is_valid. |
| `UserSettings` | `user_settings` | auto_trade_enabled, risk_profile, leverage_cap=10, blacklisted_coins ARRAY, max_concurrent_trades=3. |
| `Trade` | `trades` | FK signal_id_uuid, direction LONG/SHORT, entry/exit, leverage, status active/closed/error, pnl, opened/closed_at. |
| `SignalDisplaySettings` | `signal_display_settings` | max_loss_percentage=-15, loss_ratio=0.20, is_active. |
| `BacktestRun` | `backtest_runs` | run_id UUID uniq, status PENDING/RUNNING/COMPLETED/FAILED, parameters/summary JSONB. |
| `BacktestResult` | `backtest_results` | FK run, FK original_signal_log, outcome TP/SL/EXPIRED, pnl_percentage, entry/exit, duration_minutes. |
| `BacktestAnalysis` | `backtest_analysis` | FK result_id, model_used, prompt_text, analysis_text. |
| `EnsembleModelDecision` | `ensemble_model_decisions` | Per-model vote (id=signal_uuid, idx (symbol,model,timestamp)). prompt/completion_tokens, cost_usd, raw_response, prompt_sent. |
| `EnsembleBacktestResult` | `ensemble_backtest_results` | Per-model backtest outcome (idx (signal_id,model), (symbol,model,closed_at)). |

---

## 16. RabbitMQ-топология

**Exchange:** `analeyes_exchange` (topic, durable).

| Routing key | Producer → Consumer(s) | Payload (summary) |
|---|---|---|
| `data.raw.binance` | collector → processing | `{symbol, data:{...сырые рыночные данные...}}` |
| `data.candidates.ai` | processing → ai | Feature snapshot + strategies + score |
| `signal.final` | ai → notification, tracker | `{symbol, signal_id, source_ai, decision, confidence, tp, sl, leverage, composite_score_data, signal_log_db_id}` |
| `signal.outcome` | tracker → notification | `{signal_id, status, direction, entry/hit/tp/sl, pnl_usdt, pnl_percent, duration_seconds, ...}` |
| `signal.entry_event` | tracker → notification | `{signal_id, symbol, direction, source_ai, fill_price, entry_price, leverage, tp/sl, entry_time, signal_time}` |
| `status.collector` | collector → (api-gateway `status.*`) | heartbeat каждые 5с |
| `status.processing` | processing → (api-gateway) | heartbeat |
| `status.ai` | ai → (api-gateway) | heartbeat |
| `status.tracker` | tracker → (api-gateway) + processing (`status.tracker`) | heartbeat + live_update array |
| `status.notification` | notification → (api-gateway) | heartbeat |
| `status.backtest` | backtest → (api-gateway) | heartbeat (только в RUNNING) |
| `control.backtest.start` | api-gateway → backtest | `{run_id, parameters}` |
| `ai.backtest.analyze` | backtest → ai | failure-analysis request |
| `ai.backtest.analysis.result` | ai → backtest | analysis_text, model, prompt |

**Очереди** объявляются потребителями явно: `q_raw_data`, `q_tracker_status_for_processing`, `q_ai_candidates`, `q_new_signals` (notification), `q_signal_outcomes`, `q_signal_entry_events`, `q_new_signals_for_tracker`, `q_start_backtest`, `q_backtest_analysis_results`, `q_service_statuses_gw_{uuid}`.

---

## 17. Безопасность и известные нюансы

### Секреты в репозитории
- **13 Coinalyze API-ключей** в plaintext в `config/settings.yml`.
- **`qwen_api_key`** в plaintext в YAML.
- Колонка `exchange_api_keys.api_secret` — **plaintext** (модель ORM это фиксирует в docstring как уязвимость).

### Аутентификация/авторизация
- `POST /api/auth/users/auth` **без какой-либо верификации** `telegram_id` — любой может получить токен. Требует доверительного фронта (Telegram Login Widget HMAC).
- `POST /api/backtest/run` **без auth** — DoS-вектор через тяжёлые бэктесты.
- `GET /api/signals/signal-results`, `/{id}/chart-data`, `/api/backtest/{uuid}`, WS `/ws/status` — публичные.

### Сетевые аспекты
- **Hardcoded OpenAI-прокси** `https://gemini.dkms.cc/openai/v1` в GPTClient (и `rag.openai_base_url`). `OPENAI_API_KEY` течёт через сторонний домен.
- Gemini `BLOCK_NONE` по всем safety-категориям.
- Spoofed browser-headers в TradingView/btcparser клиентах — неофициальные API, хрупкие.
- CORS не настроен — только same-origin через static-mount.

### Корректность логики
- collector: `_backfill_missing_data` закомментирован — дыры WS не закрываются REST.
- collector: DB performance-логирование подключено, но `db_manager=None` → inert.
- processing: `_calculate_btc_whale_flow` бросит `AttributeError` при наличии `btc_transactions`.
- processing: `_calculate_eth_btc_spread_z` не вызывается → всегда None.
- processing: risk profiles грузятся один раз, не обновляются.
- ai-service: markdown-fence inconsistence (GPT/DeepSeek/Qwen/Gemini упадут на ```json```).
- ai-service: невзвешенное усреднение уверенности — слабые модели перевешивают.
- notification: генерация графика для новых сигналов stub.
- tracker: `rl_exit/agent.py` битый (импорт stable_baselines3).
- backtest: модель одной свечи (противоречит docstring про "следующую").
- backtest: `expected_services` исключает backtest в health.
- api-gateway: баг `Depends(Config)` в `/{signal_id}/chart-data`.

### Эксплуатация
- Дефисы в путях `python -m services.ai-service.src.main` работают только благодаря `-m`-filesystem.
- Numpy pinned `<2.0` для TA-Lib 0.4.28; сборка требует `CFLAGS=-Wno-incompatible-pointer-types`.
- Python-версии сервисов разрознены (3.10 в gateway, 3.11 в backtest/ai/notification).
- Список `expected_services` в health намеренно без backtest.
- `EnsembleModelDecision` с `source_ai='ensemble'` сохраняется рядом с individual — downstream должен фильтровать.

---

## 18. Утилиты и вспомогательные скрипты

### `aistud.py`
Утилита для «сплющивания» Python-проекта: копирует все `.py`-файлы в одну директорию с именами, склеенными через `__` (например, `services__ai-service__src__main.py`). Удобно для загрузки всего кода в LLM-контекст. Имеет списки игнорирования (venv, `__pycache__`, тесты, `flattened_project_output`, и т.д.). Запуск: `python aistud.py [source_dir] [target_dir]`.

### `entrypoint.sh`
Скрипт Docker-entrypoint:
1. Ждёт готовность PostgreSQL (`pg_isready`).
2. Применяет Alembic-миграции: `alembic -c shared/src/shared/alembic.ini upgrade head`.
3. `exec "$@"` — запускает основную команду из compose.

### `flattened_project_output/`
Каталог вывода `aistud.py` (в `.gitignore`). Содержит «сплющенный» снапшот кода.

---

## 19. Диаграммы (Mermaid)

### 19.1 Полный цикл сигнала (sequence)

```mermaid
sequenceDiagram
    autonumber
    participant C as collector
    participant Q as RabbitMQ
    participant P as processing
    participant DB as PostgreSQL
    participant AI as ai-service
    participant T as tracker
    participant N as notification
    participant TG as Telegram

    C->>Q: data.raw.binance (каждые 60с)
    Q->>P: доставлено
    P->>P: feature_generator (~60 фичей)
    P->>P: pair_filter (мультиуровневые пороги)
    P->>P: composite_score (взвешенный рейтинг)
    P->>DB: INSERT signal_feature_logs
    alt composite ≥ 0.78
        P->>Q: data.candidates.ai
        Q->>AI: доставлено
        par параллельно
            AI->>AI: GPT
            AI->>AI: Grok
            AI->>AI: Gemini
        end
        AI->>AI: голосование (ничья → SKIP)
        AI->>DB: INSERT ensemble_model_decisions
        AI->>Q: signal.final
        Q->>T: доставлено
        Q->>N: доставлено
        N->>N: форматирование + график
        N->>TG: send_photo (forum topic)
        N->>DB: UPDATE signal_feature_logs (telegram_message_sent)
        T->>T: TrackedSignal FSM
        T->>DB: UPDATE tracker_status
        T->>Q: signal.entry_event (при входе)
        T->>Q: signal.outcome (при TP/SL/expire)
        Q->>N: доставлено
        N->>TG: send_photo (исход)
    end
```

### 19.2 FSM трекера (state)

```mermaid
stateDiagram-v2
    [*] --> WAITING_FOR_ENTRY: start_tracking_signal
    WAITING_FOR_ENTRY --> ENTERED: price пересёк entry (с учётом slippage) / 'market'
    WAITING_FOR_ENTRY --> CANCELLED_UNFILLED: now > entry_deadline (300с)
    ENTERED --> TP_HIT: price пересёк TP
    ENTERED --> SL_HIT: price пересёк SL
    ENTERED --> EXPIRED: now > expiration_time (24ч)
    TP_HIT --> [*]: _handle_signal_completion
    SL_HIT --> [*]: _handle_signal_completion
    EXPIRED --> [*]: _handle_signal_completion
    CANCELLED_UNFILLED --> [*]: _handle_signal_completion

    note right of ENTERED
        Collision (TP+SL за тик):
        honour self.prefer ('tp' по умолчанию)
    end note
```

### 19.3 Голосование AI-ансамбля

```mermaid
flowchart TD
    A[candidate из processing] --> B[asyncio.gather: GPT+Grok+Gemini+...]
    B --> C{≥2 валидных?}
    C -- нет --> X[SKIP: сигнал отбрасывается]
    C -- да --> D[голос LONG vs SHORT]
    D --> E{ничья?}
    E -- да --> X
    E -- нет --> F[фильтр моделей по победившему направлению]
    F --> G[усреднение confidence/TP/SL/lev]
    G --> H[publish signal.final source=ensemble]
```

### 19.4 ER-диаграмма БД

```mermaid
erDiagram
    cycle_logs ||--o{ signal_feature_logs : "cycle_log_id"
    signal_feature_logs ||--o{ artefact_logs : "id"
    signal_feature_logs ||--o{ trades : "signal_id_uuid"
    signal_feature_logs ||--o{ ensemble_model_decisions : "id (signal_uuid)"
    signal_feature_logs ||--o{ ensemble_backtest_results : "signal_id"
    users ||--|| subscriptions : "user_id"
    users ||--|| user_settings : "user_id"
    users ||--o{ exchange_api_keys : "user_id"
    users ||--o{ trades : "user_id"
    backtest_runs ||--o{ backtest_results : "run_id"
    signal_feature_logs ||--o{ backtest_results : "id"
    backtest_results ||--|| backtest_analysis : "result_id"

    signal_feature_logs {
        uuid signal_id_uuid PK
        string symbol
        jsonb feat_ "~80 колонок"
        jsonb ai_ "decision/conf/tp/sl"
        string tracker_status
        float composite_score_value
    }
    users {
        bigint telegram_id PK
        string username
        string language
    }
    ensemble_model_decisions {
        string id PK "signal_uuid"
        string model
        string decision
        float confidence
        float cost_usd
    }
```

### 19.5 RabbitMQ topology (flowchart)

```mermaid
flowchart LR
    C[collector] -->|data.raw.binance| X((analeyes_exchange\n/topic durable/))
    P[processing] -->|data.candidates.ai| X
    AI[ai-service] -->|signal.final| X
    T[tracker] -->|signal.outcome\nsignal.entry_event| X
    GW[api-gateway] -->|control.backtest.start| X
    BT[backtest-service] -->|ai.backtest.analyze| X
    AI -->|ai.backtest.analysis.result| X

    X -->|data.raw.*| P
    X -->|data.candidates.ai| AI
    X -->|signal.final| N[notification]
    X -->|signal.final| T
    X -->|signal.outcome\nsignal.entry_event| N
    X -->|control.backtest.start| BT
    X -->|ai.backtest.*| AI
    X -->|ai.backtest.analysis.result| BT
    X -->|status.*| GW

    C -.->|status.collector| X
    P -.->|status.processing| X
    AI -.->|status.ai| X
    T -.->|status.tracker| X
    N -.->|status.notification| X
    BT -.->|status.backtest| X
```

---

## 20. Runbook и troubleshooting

### 20.1 Где что лежит

| Ресурс | Путь |
|---|---|
| Логи сервисов (docker) | `logs/<service>/` (примонтировано в compose) |
| Логи БД | внутри `postgres_db` контейнера, `docker logs postgres_db` |
| RabbitMQ management UI | http://localhost:15672 (user/pass из env) |
| Rabbitscout | http://localhost:3000 |
| api-gateway | http://localhost:8001 (или `API_PORT`), раздаёт web-ui на `/` |
| WebSocket статусов | ws://localhost:8001/ws/status |
| Prometheus (tracker) | http://localhost:8000/metrics (порт `prometheus.port`) |
| PostgreSQL | `localhost:5432` (user/pass/db из env) |

### 20.2 Типовые симптомы → диагноз

| Симптом | Вероятная причина | Действие |
|---|---|---|
| Сервис молчит в `status.*` > 60с | Упал / завис event-loop | `docker logs <service> --tail 200`; рестарт `docker compose restart <service>` |
| `CircuitBreakerState.OPEN` в логах collector | 5+ подряд ошибок Binance (429/418/5xx) | Подождать `reset_interval_s`=30с; проверить IP-бан Binance, `rest_rate_limit_per_min` |
| Сигналы не доходят до Telegram | `topic_id=None` для `source_ai` или нет `group_id` | Проверить `telegram.ai_specific_topics` в YAML; ключи `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ADMIN_ID` |
| AI всегда SKIP | <2 моделей включены, либо ничья голосов | Проверить `ai_engines.use_*` (по умолчанию только gpt+gemini); поднять 3+ модели |
| `processed_signals` растёт, `signals_sent` нет | composite < 0.78 или фильтр отбрасывает | Опустить `composite_score.threshold` или пороги `filters.*` для отладки |
| `expected_service X is UNAVAILABLE` в `/api/health/full` | сервис не публиковал heartbeat 60с+ | см. «сервис молчит» |
| Backtest висит в `RUNNING` | single-flight; предыдущий run не завершился | рестарт `backtest-service`; статус `FAILED` через exception в `run()` |
| WS collector не обновляет цены по символу | symbol в `excluded_symbols` (-4108) или не подписан | ждать очистки каждые 1800с; проверить `permanent_subscription_subset` |
| `psycopg2.OperationalError: too many connections` | DB-pool исчерпан | поднять `database.max_overflow`; проверить утечку сессий |

### 20.3 Health-check контракт

`GET /api/health/full` (Bearer `HEALTH_CHECK_TOKEN`):
- **DB probe:** `SELECT 1`.
- **RabbitMQ probe:** `_connection.is_open`.
- **Per-service liveness:** последний heartbeat `status.<service>` не старше `SERVICE_STATUS_TIMEOUT_SECONDS`=60с.
- `expected_services = ['collector','processing','ai','tracker','notification']` — **backtest намеренно исключён** (шлёт heartbeat только в RUNNING).
- Роллап: OPERATIONAL если всё OK; DEGRADED если что-то UNAVAILABLE; UNAVAILABLE если DB или RabbitMQ недоступны.

### 20.4 Мониторинг

| Источник | Что смотреть |
|---|---|
| Prometheus (`tracker-service`) | `tracker_active_signals`, `tracker_tp_hit_total`, `tracker_sl_hit_total`, `tracker_expired_signal_total`, `tracker_cancelled_unfilled_total`, `ws_price_age_ms` (гистограмма свежести цен) |
| RabbitMQ management | глубина очередей (`q_ai_candidates`, `q_new_signals*`), throughput, unacked |
| PostgreSQL | размер `signal_feature_logs`, `ensemble_model_decisions`; долгие транзакции; connection-pool |
| web-ui `/` | live-карточки статусов сервисов, PnL-таблица |
| Логи collector | `X-MBX-USED-WEIGHT-1M`, `ws_messages_per_second`, `rest_calls_per_minute` |

**Где Prometheus НЕ собирает:** collector/processing/ai/notification/backtest метрик не экспортируют. Для полного покрытия добавить `prometheus_client` + `/metrics` эндпоинт по образцу tracker.

### 20.5 Бэкапы

- `postgres_data` — docker volume; бэкапить через `pg_dump` или snapshot volume.
- `config/settings.yml` — в git (но содержит plaintext-ключи — см. безопасность).
- ChromaDB — внешний хост (`rag.chroma_host`), бэкап — на стороне инсталляции Chroma.

---

## 21. Developer onboarding

### 21.1 Локальный запуск без Docker (один сервис)

```bash
# 0. PostgreSQL + RabbitMQ где-то запущены (или через docker compose up postgres rabbitmq)
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/analeyes"
export RABBITMQ_URL="amqp://user:pass@localhost:5672/%2F"

# 1. Установить shared-пакет (editable)
pip install -e ./shared

# 2. Применить миграции
cd shared/src/shared && alembic upgrade head && cd -

# 3. Запустить нужный сервис (пример: processing)
cd services/processing-service
PYTHONPATH=/path/to/repo python src/main.py
```

**Важно:** `PYTHONPATH` должен указывать на корень репо, чтобы работали импорты `services.*` (для ai/notification/tracker/backtest) или `shared.*` (для collector/processing/gateway — через editable install).

### 21.2 Как добавить новый LLM-клиент

1. Создать `services/ai-service/src/logic/ai/<provider>_client.py` по образцу `qwen_client.py`:
   - SDK `openai.AsyncOpenAI` (если провайдер OpenAI-compat) или нативный SDK.
   - Конструктор: read config, build client, load prompt template.
   - `analyze(symbol, features, ...)` → всегда возвращает dict с `{signal, confidence, reason, tp, sl, ...}`, **никогда не бросает** (на ошибке — SKIP/error dict).
   - Парсер: validate `signal ∈ {LONG,SHORT,SKIP}`, clamp confidence [0,1].
2. В `ai_analyzer.py` `__init__` добавить инстанцирование под флагом `use_<provider>`.
3. В `config/settings.yml` `ai_engines`: добавить `use_<provider>`, `<provider>_base_url`, `<provider>_model`, `prompt_paths.<provider>`.
4. В `config/prompts/`: добавить `<provider>_prompt.txt` (минимум `{PROMPT_BODY}` если инструкции инлайн в клиенте).
5. Env-вары: добавить в `Config.get_ai_config` секцию слияния.
6. В `notification` `telegram.ai_specific_topics`: добавить топик форума для нового `source_ai`.

### 21.3 Как добавить новый индикатор

1. В `feature_generator.py` `_calculate_technical_indicators` добавить вызов TA-Lib (или ручной расчёт).
2. Ключи результата добавить в `expected_feature_keys` (конец `generate_features`) — иначе останутся None.
3. (Опц.) Если нужна фильтрация — добавить критерий в `pair_filter.py` `_check_criteria`.
4. (Опц.) Если нужна нормализация для composite — добавить ветку в `composite_score.py` `_normalize_metric` + вес в `composite_score.weights`.
5. (Опц.) Для отображения в debug-сообщении Telegram — добавить в `telegram_sender._format_signal_message` signal-debug блок.
6. Для persistence — колонку в ORM `SignalFeatureLog` + Alembic-миграцию (см. ниже).

### 21.4 Как добавить DB-миграцию

```bash
cd shared/src/shared
# сгенерировать пустую ревизию
alembic revision -m "add feature X"
# отредактировать shared/src/shared/alembic/versions/<hash>_add_feature_x.py:
#   def upgrade(): op.add_column('signal_feature_logs', sa.Column('feat_x', sa.Float(), nullable=True))
#   def downgrade(): op.drop_column('signal_feature_logs', 'feat_x')
alembic upgrade head
```

Базовая ревизия: `3cbcf834c770` (полная начальная схема). `env.py` грузит `.env.local` (приоритет) или `.env`, подменяет URL на sync-драйвер `+psycopg2`.

### 21.5 Конвенции кода

- **Язык комментариев/логов:** русский. Имена классов/функций/переменных — английские.
- **Async везде:** все I/O через `asyncio`; sync-библиотеки (pika publish в backtest_runner) изолированы.
- **Message encoding:** `DataEncoder` через duck-typing — кастомные объекты с `.dict()`/`.isoformat()`/`.tolist()` будут приведены автоматически. Для dataclass'ов — добавлять `.dict()` либо явно сериализовать перед publish.
- **Logging:** `configure_logging_globally` + `get_logger(__name__)`; формат использует `%(shortname)` (отрезает `src.`).
- **Never raises из message handlers:** всегда `basic_ack` в finally; ошибки логируются.
- **Numpy < 2.0** project-wide (TA-Lib 0.4.28 ABI constraint).
- **Импорты shared:** через `from shared.config import Config` (editable install) ИЛИ `from shared.src.shared.config import Config` (sys.path-инъекция) — оба варианта встречаются, несогласованно.

### 21.6 Переключение движков AI без кода

Все тумблеры в `config/settings.yml` `ai_engines`:
```yaml
ai_engines:
  use_gpt: true
  use_grok: false
  use_gemini: true
  use_qwen: false
  use_deepseek: false
  use_claude: false
  composite_threshold: 0.78
  min_confidence: 0.60
```
Перезапуск сервиса подхватит изменения (на старте `AIAnalyzer.__init__` строит список клиентов).

---

## 22. Config reference (полный)

Подробный референс `config/settings.yml`. Мёртвые/устаревшие ключи помечены 🪦.

### `binance`
| Ключ | Дефолт | Описание |
|---|---|---|
| `use_websocket` | true | Использовать WS для klines/depth/trades. |
| `rest_rate_limit_per_min` | 1200 | Лимит REST-вызовов Binance (фактически `RateLimiter` token-bucket). |
| `test_mode` | false | Testnet endpoints. |
| `stream_mark_price` | true | Подписка на `@markPrice@1s`. |
| `circuit_breaker.max_errors` | 5 | Ошибок подряд до OPEN. |
| `circuit_breaker.reset_interval_s` | 30 | Пауза OPEN→HALF_OPEN. |

### `collector`
| Ключ | Дефолт | Описание |
|---|---|---|
| `pairs_check_interval_s` | 900 | Обновление топ-пар по объёму. |
| `max_pairs_for_testing` | 200 | Верхняя граница числа пар. |
| `use_pair_rotation` | true | Ротация pool-пар через WS. |
| `min_klines_ws_ready` | 50 | Минимум klines из WS для марки `WebSocket_Sufficient`. |
| `rest_backup_for_non_ws` | true | REST-fallback для символов без WS. |
| `use_binance_liquidations` | true | Подписка `!forceOrder@arr`. |
| `klines_backfill_on_startup` | true | Тяжёлый стартовый прогрев REST. |
| `permanent_subscription_subset` | [BTC,ETH,SOL,BNB] | Всегда подписаны + доп. потоки trade/bookTicker. |
| `additional_kline_intervals` | [15m,30m,1h,4h] | Доп. таймфреймы из REST-кэша. |
| `additional_kline_fetch_interval_s` | 1800 | Обновление доп-таймфреймов. |
| `oi_history_fetch_interval_s` | 900 | Обновление OI history. |
| `exchange_info_interval_s` | 3600 | Кэш exchangeInfo. |
| `excluded_symbols_check_interval_s` | 1800 | Очистка списка исключённых (-4108). |

### `websocket`
| Ключ | Дефолт | Описание |
|---|---|---|
| `urls` | fstream mirror list | Зеркала для round-robin. |
| `num_connections` | 5 | Число параллельных WS-клиентов. |
| `reconnect_delay_s` | 5 | Пауза реконнекта. |
| `max_reconnect_attempts` | 10 | 🪦 читается, но `run()` не использует как hard-cap. |
| `max_klines_per_symbol` | 500 | Bound хранилища. |
| `max_trades_per_symbol` | 1000 | Bound. |
| `max_agg_trades_history_minutes` | 10 | Окно хранения aggTrades. |
| `max_liquidations_per_symbol_ws` | 200 | Bound. |
| `rotation.priority_pairs` | 50 | Top-N всегда подписаны. |
| `rotation.pool_size` | 150 | Размер pool ротации. |
| `rotation.batch_active_size` | 50 | Активных из pool одновременно. |
| `rotation.rotation_interval_s` | 300 | Каденс ротации. |
| `ping_interval` / `ping_timeout` | 30 / 40 | websockets lib. |
| `connection_liveness_timeout_s` | 180 | Порог «мёртвой» подписки. |

### `features`
| Ключ | Дефолт | Описание |
|---|---|---|
| `primary_kline_interval` | 5m | Основной таймфрейм. |
| `secondary_kline_interval` | 1m | 🪦 код хардкодит «1m if primary==5m else 5m». |
| `min_candles_for_ta` | 50 | Минимум свечей для индикаторов. |
| `liquidation_window_hours` | 3 | Окно суммы ликвидаций. |
| `adaptive_volume.enabled` | true | Использовать vol_rel в фильтре. |
| `atr_norm.enabled` | true | Нормировать price_pct на ATR. |
| `acceleration_filter.enabled` | false | Fast-pass по acc_*. |
| `correlation.enabled` | true | BTC-корреляция. |
| `correlation.window` | 20 | Окно корреляции. |
| `correlation.btc_symbol` | BTCUSDT | Прокси для рынка. |
| `fibonacci_window` | 50 | Окно swing high/low. |
| `support_resistance_window` | 20 | Окно S/R. |
| `eth_btc_spread.window` | 200 | 🪦 eth_btc_spread_z определён, но не вызывается. |
| `bvol_proxy.window` | 288 | 24ч 5m свечей для BVOL. |
| `time_factors.*` | asiatic 0.8 / european 1.0 / us 1.1 | Множители порогов по сессии. |

### `filters`
| Ключ | Дефолт | Описание |
|---|---|---|
| `vol_z_threshold` | 0.8 | Базовый порог |vol_z|. |
| `price_pct_threshold` | 0.18 | Базовый порог |price_pct|. |
| `oi_z_threshold` | 0.5 | Базовый порог |oi_z|. |
| `liq_usd_threshold` | 40000 | Ликвидации USD. |
| `funding_rate_threshold` | 0.00015 | |funding|. |
| `ofi_delta_threshold` | 100000 | |OFI delta|. |
| `eis_min_abs_threshold` | 5.0 | |EIS| (если `eis_filter.enabled`). |
| `vol_rel_threshold` | 0.8 | vol_rel (если adaptive_volume). |
| `price_pct_norm_threshold` | 0.25 | |price_pct_norm| (если atr_norm). |
| `spread_pct_threshold` | 0.15 | Верхняя граница спреда. |
| `coin_size_categorization.*` | large $500M / medium $50M | Категории для overrides. |
| `glitch_spike_filter.*` | vol_z≥2.5, multiplier 10 | Защита от фантомных сделок. |
| `gamma_proxy.funding_rate_threshold_abs` | 0.001 | Высокий фандинг → ×0.8. |

### `composite_score`
| Ключ | Дефолт | Описание |
|---|---|---|
| `threshold` | 0.5 | ⚠️ в `main.py` processing используется `composite_threshold`=0.78 из `ai_engines`, не отсюда. |
| `weights.vol_rel` | 0.22 | |
| `weights.acc_vol` | 0.15 | |
| `weights.price_pct_norm` | 0.15 | |
| `weights.acc_price` | 0.12 | |
| `weights.oi_z` | 0.10 | |
| `weights.ofi_delta` | 0.10 | |
| `weights.patterns` | 0.10 | |
| `weights.eis` | 0.07 | |
| `weights.order_flow_ratio` | 0.05 | |
| `weights.cvd` | 0.0 | 🪦 веса 0 — не влияют. |
| `weights.liq_usd` | 0.0 | 🪦 веса 0. |
| `normalization_thresholds.*` | various | Пороги `min(1, |v|/thresh)`. |

### `ai_engines`
| Ключ | Дефолт | Описание |
|---|---|---|
| `use_gpt/grok/gemini/qwen/deepseek/claude` | gpt+gemini true | Тумблеры клиентов. |
| `<engine>_model` | various | Имя модели. |
| `<engine>_base_url` | various | Endpoint. GPT hardcode `gemini.dkms.cc`. |
| `timeout` | 60 | 🪦 GPT игнорирует, хардкод 60. |
| `gpt_max_retries` | 2 | Передаётся в openai SDK. |
| `composite_threshold` | 0.78 | Порог публикации в AI (используется processing). |
| `min_confidence` | 0.60 | Порог финальной уверенности (информативно). |
| `prompt_compress` | true | Сжатие history-snapshots в промпте. |
| `grok_supports_json_response_format` | — | 🪦 читается из config, но GrokClient не использует. |
| `qwen_api_key` | plaintext | 🪦 в YAML, должен быть в env. |

### `trend_impulse` (EIS)
| Ключ | Дефолт | Описание |
|---|---|---|
| `half_life_min` | 2 | Период полураспада EIS. |
| `oi_weight_div` | 5.0 | Делитель OI в сигнале. |
| `filter_min_abs_threshold` | 5.0 | Порог EIS в фильтре. |
| `emoji_bands` | various | Полосы для emoji в Telegram. |

### `ml_proxy` / `rl_exit`
🪦 **Оба выключены и не подключены** в рантайме (LSTM stub всегда 0; RL-exit импортирует отсутствующий `stable_baselines3`).
| Ключ | Дефолт | Описание |
|---|---|---|
| `ml_proxy.reject_threshold` | 0.63 | Порог LightGBM-фильтра. |
| `ml_proxy.model_path` | `src/ml/models/lgbm_proxy_v1.txt` | Путь к артефакту. |
| `rl_exit.min_conf_online` | 0.55 | Порог уверенности PPO. |

### `telegram`
| Ключ | Дефолт | Описание |
|---|---|---|
| `group_id` | — | ID форума-группы. |
| `ai_specific_topics` | gpt=4, gemini=2, ... | Маппинг модель→топик форума. |
| `min_confidence` | 0.7 | Порог отправки (информативно). |
| `retry_attempts` | 3 | Retry network errors. |
| `signal_cooldown_minutes` | 15 | 🪦 дедупlication по символу — не реализован в notification. |
| `explicit_outcome_channels_only` | true | 🪦 vestigial. |
| `new_signal_channels` / `outcome_alert_channels` | [] | 🪦 vestigial, не используются. |

### `signal_tracker`
| Ключ | Дефолт | Описание |
|---|---|---|
| `enabled` | true | |
| `expiration_hours` | 24 | Срок жизни ENTERED. |
| `entry_timeout_sec` | 300 | Окно входа. |
| `slippage_pct` | 0.001 | Допуск цены входа. |
| `max_age_ms` | 5000 | Свежесть источника цены. |
| `prefer` | tp | При collision TP/SL за тик. |
| `check_interval_s` | 1 | 🪦 в main loop default 10с, не отсюда. |

### `market_global_conditions`
| Ключ | Дефолт | Описание |
|---|---|---|
| `state_proxy_symbol` | BTCUSDT | Прокси для global state. |
| `activity_top_n_pairs` | 10 | N пар для активности. |
| `threshold_adjustment_matrix[state][activity]` | various | Мультипликаторы порогов фильтра. |

### `coinalyze`
| Ключ | Дефолт | Описание |
|---|---|---|
| `api_keys` | 13 plaintext 🪦 | Ротация round-robin. |
| `interval` | 5min | Бакеты. |
| `api_history_hours` | 1 | Окно. |
| `symbol_suffix` | _PERP.A | Маппинг символов. |

### Прочие секции
`coingecko_api`, `btc_parser`, `okx`, `economic_calendar`, `x_heat` (off), `session_regimes` (off), `indicators.*`, `patterns.*`, `history_snapshots` (length 3), `forced_trade_parameters` (TP 15% / SL -5%), `backtest.ai_analysis.enabled`, `rag.*` (ChromaDB host, collection `crypto_signals_v1`), `database.pool_size=5`, `database.max_overflow=10`, `trading.default_initial_bank_usd=500`.

---

## 23. Glossary терминов

| Термин | Расшифровка |
|---|---|
| **EIS** (Exponential Impulse Score) | Экспоненциально затухающий «импульс» цены, интегрирующий `sign(price_pct)×sqrt(|vol_z|)×(1+|oi_z|/div)` с полураспадом 2мин. |
| **OFI** (Order Flow Imbalance) | `(Σbid − Σask)_now − (Σbid − Σask)_prev` по топ-N уровням стакана. |
| **QPC** (Quantity-Price Change delta) | `bid_qty_change − ask_qty_change` по дельтам depth. |
| **CVD** (Cumulative Volume Delta) | `Σ(taker_buy_quote_vol − taker_sell_quote_vol)` за окно (6 свечей 5m). |
| **RVR** (Realized Volatility Ratio) | `std(logret_short) / std(logret_long)`, обычно 5m/30m. |
| **IVR** (Implied Volatility Rank) | Здесь — `(curr_premium − min)/(max − min)`, premium = `(mark−index)/index`. |
| **BVOL** | Прокси волатильности BTC через ATR; z-score против 24ч окна. |
| **OB imbalance** (Order Book Imbalance) | `(Σbid_qty − Σask_qty)/(Σbid_qty + Σask_qty)`. |
| **Gamma-proxy** | Эвристика: при `|funding| ≥ 0.001` применяется мультипликатор порогов ×0.8. |
| **X-Heat** | Активность кэштегов в X/Twitter среди списка инфлюенсеров; `hype_threshold`=3. |
| **Glitch-spike** | Защита: если kline `avg_vol_per_trade > 24h_avg × 10`, фильтр отбрасывает (фантомные сделки API). |
| **Risk profile** | Per-symbol категория доверия: HIGH_TRUST (win-rate ≥0.65, tighter ×0.9), NEUTRAL (×1.0), LOW_TRUST (looser ×1.2). |
| **Acceleration fast-pass** | Если `|acc_price|>0.18` или `|acc_vol|>0.4` — все критерии pair_filter авто-pass. |
| **Composite score** | Взвешенная сумма нормализованных метрик ∈ [0,1]; порог публикации 0.78. |
| **Heuristic signal consensus** | Голос {patterns, indicators, price-direction} — большинство определяет LONG/SHORT/MIXED/NONE. |
| **tp_distribution / sl_distribution** | Квантили `{p10, p50, p90}` от LLM для TP/SL — усредняются по ансамблю per-quantile. |
| **Whale trade proxy** | Сделки > median×20 (по USD), с распадом за 5мин. |
| **Retail impulse proxy** | Z-score count и avg_size трейдов последней 1m-свечи против 60-окна. |
| **Hidden order proxy** | Max число сделок по одной цене за 2с (прокси айсберга). |
| **Trend/Flat/Transition** | Режим рынка, 4-классификаторное голосование (ADX, BB-width, ATR ratio, slope). |

---

## 24. Performance и масштабирование

### 24.1 Пропускные характеристики

| Метрика | Значение |
|---|---|
| Длина цикла collector | 60с (`system.cycle_interval_s`) |
| Пар на цикл | до `max_pairs_for_testing`=200 |
| WS-соединений collector | 5 (`websocket.num_connections`) |
| Стримов на WS-клиент | ~40 пар × ~4 потока = ~160 (плюс permanent trade/bookTicker) |
| Частота AI-запросов | ≤ кандидатов × 6 моделей / цикл (GPT/Grok/Gemini/Claude/DeepSeek/Qwen) |
| Polling трекера | 10с (`signal_tracker.check_interval_s` default) |
| WebSocket heartbeat статус | 5с на сервис |

### 24.2 Узкие места

1. **`composite_score_data` целиком в каждом RabbitMQ-сообщении** — `signal.final` несёт полный strategies-дикт. Большие пейлоуды → нагрузка на RabbitMQ + network.
2. **`RANDOM()` в SQL-пагинации** `signal-results` — недетерминированные результаты между страницами (skip/дубли).
3. **Single-flight backtest** — один run на процесс; горизонтальное масштабирование требует external partitioning.
4. **Always-ack без DLX** во всех consumer'ах — poison-message теряется молча.
5. **Глобальный market_state обновляется только по BTC** — если поток BTC задержан, остальные пары используют устаревший режим.
6. **Cold start acc_*=0** — первые 2 цикла на символ искажают composite (acc_vol/acc_price).
7. **Risk profiles грузятся 1 раз** — долгоживущий процесс оперирует устаревшими win-rate.
8. **REST fallback tracker** — при отсутствии WS-данных по символу каждый polling-cycle дёргает REST (смягчается семафором 5 + кэш 2с).
9. **LightGBM/CircuitBreaker в collector не используются** — RateLimiter да, но AI-clients без circuit-breaker (деградирующий провайдер → бесконечные ошибки).
10. **Pika sync publish** в `backtest_runner._request_failure_analysis` — блокирует event-loop на время publish.

### 24.3 Масштабирование

- **Вертикально:** поднять `database.pool_size`/`max_overflow`, `websocket.num_connections`, `collector.max_concurrent_rest_fetches`.
- **Горизонтально:** stateless сервисы (api-gateway, ai-service, processing, notification) масштабируются репликами. **Stateful:** tracker (in-memory `active_tracked_signals` — реплика получит дубли), collector (WS-подписки — нужна координация).
- **БД:** при росте `signal_feature_logs` — партиционирование по `created_at`, индексы уже есть на `(symbol, initial_metric_timestamp)` и `(symbol,model,timestamp)` для ensemble.
- **RabbitMQ:** prefetch=1 по умолчанию — поднять для throughput; мониторить unacked.

---

## 25. Тестирование

### Текущее состояние
🚫 **Тестов нет.** Каталог `tests/` отсутствует; в `.gitignore` и ignore-листе `aistud.py` тесты исключены, но фактически их нет. CI/CD-конфигурации в репо тоже нет.

### Рекомендуемая стратегия

| Слой | Что покрывать | Инструменты |
|---|---|---|
| **Unit: feature_generator** | Каждый `_calculate_*` на синтетических данных (RSI на ровном ряде = 50, vol_z на константе = 0). | `pytest`, фикстуры с numpy-массивами. |
| **Unit: pair_filter** | `_adjust_thresholds` по категориям/сессиям/глобальному матриксу; glitch-spike; acceleration fast-pass. | pytest table-driven. |
| **Unit: composite_score** | `_normalize_metric` для edge-cases (order_flow_ratio <1/>1, spread=0); веса суммируются в 1.0. | pytest. |
| **Unit: signal_tracker FSM** | `process_market_data` для всех переходов (WAITING→ENTERED→TP/SL/EXPIRE/CANCELLED), collision, slippage, staleness. | pytest + mock pair_data. |
| **Unit: AI clients** | Парсинг валидного JSON, fenced JSON, пустого choices, невалидного signal enum, timeout. Мокировать `AsyncOpenAI`. | pytest + `respx`/`httpx` mock. |
| **Integration: RabbitMQ** | Сквозной flow collector→processing→ai→tracker→notification на test-exchange. | `pytest-asyncio` + testcontainers RabbitMQ. |
| **Integration: DB** | Миграции alembic up/down; CRUD моделей. | testcontainers PostgreSQL. |
| **Contract: prompt builder** | Снапшот промпта для фиксированного features-дикта — ловить дрейф. | pytest snapshot (`syrupy`). |
| **E2E: backtest** | Детерминированный replay известного сигнала → ожидаемый TP/SL/EXPIRED. | pytest. |

### Покрытие, prioritized
1. `feature_generator._calculate_*` (~30 методов) — высокий риск регресса.
2. `pair_filter._check_criteria` — сложная логика корректировок.
3. `signal_tracker.process_market_data` — критичный FSM.
4. AI-clients `analyze` — много путей отказа.
5. `composite_score._normalize_metric` — edge cases.

---

## 26. Roadmap / TODO

Приоритет: 🔴 security → 🟠 correctness → 🟡 cleanup → 🟢 enhancement.

### 🔴 Security
- [ ] Вынести 13 Coinalyze API-ключей из `settings.yml` в env (ротор).
- [ ] Вынести `qwen_api_key` из YAML в env.
- [ ] Шифровать `exchange_api_keys.api_secret` (или хотя бы не plaintext в ORM).
- [ ] Реализовать HMAC-проверку Telegram Login Widget в `/api/auth/users/auth`.
- [ ] Добавить auth на `POST /api/backtest/run` (admin gate).
- [ ] Рассмотреть ownership/доверие к hardcoded OpenAI-прокси `gemini.dkms.cc` (GPTClient + RAG).
- [ ] Сделать `OPENAI_API_KEY` override base_url через config (сейчас hardcode).

### 🟠 Correctness
- [ ] Реализовать `_backfill_missing_data` в collector WS (сейчас закомментирован — дыры не закрываются).
- [ ] Починить `_calculate_btc_whale_flow` (AttributeError на `self.app_context`).
- [ ] Подключить `_calculate_eth_btc_spread_z` в `generate_features` (сейчас всегда None).
- [ ] Обновлять risk profiles периодически, не только на старте.
- [ ] Починить `Depends(Config)` в `GET /api/signals/{id}/chart-data` (FileNotFoundError).
- [ ] Markdown-fence stripping во всех AI-clients (GPT/DeepSeek/Qwen/Gemini упадут на ```json```).
- [ ] weighted confidence в AI-ансамбле (учитывать надёжность модели, не plain mean).
- [ ] Backtest: модель следующей свечи, не той же (противоречит docstring).
- [ ] `processed_signals` инкремент после проверки полей в backtest (искажённый win_rate).
- [ ] DLX + retry для poison-messages во всех consumer'ах.
- [ ] Имплементировать `signal_cooldown_minutes` в notification (сейчас дедуп не реализован).
- [ ] `_validate_api_key` в `/api/keys` (сейчас stub always True).
- [ ] Включить backtest в `expected_services` health.

### 🟡 Cleanup / dead code
- [ ] Удалить или подключить `rl_exit/agent.py` (битый импорт stable_baselines3) в tracker и ai-service.
- [ ] Удалить или подключить `lstm_proxy.py` (stub всегда 0) в processing.
- [ ] Удалить или подключить `ml/ml_proxy.py` (не вызывается в ai-service main).
- [ ] Удалить мёртвый `SignalTracker.publish_outcome_to_rabbitmq` (затенён app-callback).
- [ ] Удалить `_should_scan_pair` (определён, не вызывается).
- [ ] Удалить vestigial `new_signal_channels` / `outcome_alert_channels` в telegram config.
- [ ] Удалить `short_mode` параметр в AI-clients (не вызывается).
- [ ] Подключить RAG (`rag_context`) в ai-service или убрать параметр.
- [ ] Согласовать `slippage_pct` / `max_age_ms` defaults между `SignalTracker` и `TrackedSignal`.
- [ ] Унифицировать Python-версии сервисов (3.10 vs 3.11).
- [ ] Унифицировать стиль импортов shared (`from shared.*` vs `from shared.src.shared.*`).
- [ ] Переименовать директории сервисов без дефиса ИЛИ явно документировать `-m`-filesystem workaround.
- [ ] Включить `secondary_kline_interval` в обработку (сейчас hardcode).
- [ ] Дедуплицировать DB-апдейт в notification (`on_new_signal` vs `_update_signal_log_for_skipped_or_unnotified`).
- [ ] Завести генерацию графика в `send_signal` (сейчас stub).

### 🟢 Enhancement
- [ ] Prometheus `/metrics` во всех сервисах (сейчас только tracker).
- [ ] Кеширование/дедуп в ai-service (повторные идентичные запросы).
- [ ] Per-client CircuitBreaker в ai-service.
- [ ] Cost/budget caps на AI (сейчас не ограничены).
- [ ] Sliding-window rate limit в CircuitBreaker (сейчас naive fixed-window).
- [ ] Партиционирование `signal_feature_logs` по времени.
- [ ] CI/CD pipeline (lint, type-check, tests).
- [ ] `.env.example` с всеми переменными.
- [ ] Helm/k8s manifests (сейчас только docker-compose).
- [ ] OpenAPI-документация эндпоинтов (FastAPI autodoc уже есть на `/docs`, добавить описания).
- [ ] Локализация web-ui (сейчас ru-only).

---

## 27. Changelog и лицензия

### Changelog
В репозитории отдельного `CHANGELOG.md` нет. Для автоматической генерации из git-коммитов использовать skill `changelog-generator`.

Кратко по видимым маркерам:
- **V6** — текущая каталоговая версия; микросервисная архитектура (7 сервисов + shared).
- **V9** — встречается в UI-заголовках и FastAPI title (`AnalEyes Crypto V9 - API Gateway`) — унаследованное маркетинговое именование, не отражает структурную версию.

### Лицензия
`LICENSE`-файла в репозитории **нет**. По умолчанию применяется режим «все права защищены» (proprietary). Для open-source-релиза — добавить `LICENSE` (MIT/Apache-2.0/Proprietary) и `NOTICE`.

---

### Итог

AnalEyes Crypto V6 — зрелый, но активно эволюционирующий конвейер: 7 asyncio-сервисов, координируемых через RabbitMQ и PostgreSQL, с многоуровневой ML/AI-обработкой рыночных данных, ансамблем из 6 LLM, Telegram-доставкой с графиками, FSM-трекером сигналов и REST/WebSocket API для пользователей. Основные риски сосредоточены в (а) plaintext-секретаx, (б) доверительной модели аутентификации по telegram_id, (в) hardcoded LLM-прокси, (г) ряде мёртвых/битых кодовых путей (RL-exit, lstm_proxy, eth_btc_spread_z, ws-backfill) и (д) упрощениях моделей (одна свеча в backtest, упрощённый Supertrend, невзвешенное голосование). Полный приоритизированный план рефакторинга — в разделе [Roadmap](#26-roadmap--todo).
