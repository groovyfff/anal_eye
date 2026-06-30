--
-- PostgreSQL database dump
--

\restrict XeEIfeLV3kGPRcWRZnjUHzLGjZNpceX5fjuKZ1BBD8jE9W4rMVHbtT74pAo1m2D

-- Dumped from database version 15.18
-- Dumped by pg_dump version 15.18

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: alembic_version; Type: TABLE; Schema: public; Owner: analeyes
--

CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);


ALTER TABLE public.alembic_version OWNER TO analeyes;

--
-- Name: ensemble_backtest_results; Type: TABLE; Schema: public; Owner: analeyes
--

CREATE TABLE public.ensemble_backtest_results (
    asset_class character varying(16) DEFAULT 'crypto'::character varying NOT NULL,
    signal_id character varying(64) NOT NULL,
    symbol character varying(64) NOT NULL,
    model character varying(64) NOT NULL,
    outcome character varying(32),
    pnl_percentage double precision,
    closed_at timestamp with time zone,
    id integer NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.ensemble_backtest_results OWNER TO analeyes;

--
-- Name: ensemble_backtest_results_id_seq; Type: SEQUENCE; Schema: public; Owner: analeyes
--

CREATE SEQUENCE public.ensemble_backtest_results_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ensemble_backtest_results_id_seq OWNER TO analeyes;

--
-- Name: ensemble_backtest_results_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: analeyes
--

ALTER SEQUENCE public.ensemble_backtest_results_id_seq OWNED BY public.ensemble_backtest_results.id;


--
-- Name: ensemble_model_decisions; Type: TABLE; Schema: public; Owner: analeyes
--

CREATE TABLE public.ensemble_model_decisions (
    asset_class character varying(16) DEFAULT 'crypto'::character varying NOT NULL,
    signal_id uuid NOT NULL,
    symbol character varying(64) NOT NULL,
    model character varying(64) NOT NULL,
    "timestamp" timestamp with time zone NOT NULL,
    decision character varying(16),
    confidence double precision,
    raw_response text,
    id integer NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.ensemble_model_decisions OWNER TO analeyes;

--
-- Name: ensemble_model_decisions_id_seq; Type: SEQUENCE; Schema: public; Owner: analeyes
--

CREATE SEQUENCE public.ensemble_model_decisions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ensemble_model_decisions_id_seq OWNER TO analeyes;

--
-- Name: ensemble_model_decisions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: analeyes
--

ALTER SEQUENCE public.ensemble_model_decisions_id_seq OWNED BY public.ensemble_model_decisions.id;


--
-- Name: signal_feature_logs; Type: TABLE; Schema: public; Owner: analeyes
--

CREATE TABLE public.signal_feature_logs (
    asset_class character varying(16) DEFAULT 'crypto'::character varying NOT NULL,
    symbol character varying(64) NOT NULL,
    display_name character varying(128),
    signal_id_uuid uuid NOT NULL,
    initial_metric_timestamp timestamp with time zone NOT NULL,
    trigger_reason character varying(128),
    heuristic_signal_consensus character varying(16),
    composite_score_value double precision,
    feat_current_price double precision,
    feat_price_pct double precision,
    feat_market_state character varying(32),
    feat_rsi double precision,
    feat_macd double precision,
    feat_macd_signal double precision,
    feat_macd_hist double precision,
    feat_ema_short double precision,
    feat_ema_long double precision,
    feat_ema_50 double precision,
    feat_ema_200 double precision,
    feat_adx double precision,
    feat_bb_upper double precision,
    feat_bb_middle double precision,
    feat_bb_lower double precision,
    feat_bb_width double precision,
    feat_atr double precision,
    feat_atr_pct double precision,
    feat_vwap double precision,
    feat_vol_rel double precision,
    feat_obv double precision,
    feat_support_nearest double precision,
    feat_resistance_nearest double precision,
    feat_sp500_correlation double precision,
    feat_dxy_correlation double precision,
    feat_bid_ask_spread_pips double precision,
    feat_funding_rate double precision,
    feat_open_interest_z double precision,
    feat_liquidations_long_usd double precision,
    feat_liquidations_short_usd double precision,
    feat_cvd double precision,
    ohlcv_open_price double precision,
    ohlcv_high_price double precision,
    ohlcv_low_price double precision,
    ohlcv_close_price double precision,
    ohlcv_volume double precision,
    strat_indicators_consensus character varying(16),
    strat_patterns_consensus character varying(16),
    historical_snapshots_json jsonb,
    historical_ohlcv_json jsonb,
    indicators_json jsonb,
    patterns_json jsonb,
    features_json jsonb,
    ai_signal_type character varying(16),
    ai_confidence double precision,
    ai_reason_summary text,
    ai_entry_price_suggestion double precision,
    ai_tp_price_suggestion double precision,
    ai_sl_price_suggestion double precision,
    ai_leverage_suggestion double precision,
    ai_consensus_achieved boolean,
    telegram_message_sent boolean DEFAULT false,
    tracker_status character varying(32),
    tracker_entry_price double precision,
    tracker_exit_price double precision,
    tracker_pnl_percent double precision,
    tracker_pnl_usdt double precision,
    tracker_duration_seconds integer,
    tracker_closed_at timestamp with time zone,
    id integer NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.signal_feature_logs OWNER TO analeyes;

--
-- Name: signal_feature_logs_id_seq; Type: SEQUENCE; Schema: public; Owner: analeyes
--

CREATE SEQUENCE public.signal_feature_logs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.signal_feature_logs_id_seq OWNER TO analeyes;

--
-- Name: signal_feature_logs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: analeyes
--

ALTER SEQUENCE public.signal_feature_logs_id_seq OWNED BY public.signal_feature_logs.id;


--
-- Name: trades; Type: TABLE; Schema: public; Owner: analeyes
--

CREATE TABLE public.trades (
    asset_class character varying(16) DEFAULT 'crypto'::character varying NOT NULL,
    signal_id_uuid uuid NOT NULL,
    symbol character varying(64) NOT NULL,
    direction character varying(8) NOT NULL,
    entry_price double precision,
    exit_price double precision,
    leverage double precision,
    status character varying(16) DEFAULT 'active'::character varying NOT NULL,
    pnl double precision,
    user_telegram_id bigint,
    opened_at timestamp with time zone,
    closed_at timestamp with time zone,
    id integer NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.trades OWNER TO analeyes;

--
-- Name: trades_id_seq; Type: SEQUENCE; Schema: public; Owner: analeyes
--

CREATE SEQUENCE public.trades_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.trades_id_seq OWNER TO analeyes;

--
-- Name: trades_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: analeyes
--

ALTER SEQUENCE public.trades_id_seq OWNED BY public.trades.id;


--
-- Name: ensemble_backtest_results id; Type: DEFAULT; Schema: public; Owner: analeyes
--

ALTER TABLE ONLY public.ensemble_backtest_results ALTER COLUMN id SET DEFAULT nextval('public.ensemble_backtest_results_id_seq'::regclass);


--
-- Name: ensemble_model_decisions id; Type: DEFAULT; Schema: public; Owner: analeyes
--

ALTER TABLE ONLY public.ensemble_model_decisions ALTER COLUMN id SET DEFAULT nextval('public.ensemble_model_decisions_id_seq'::regclass);


--
-- Name: signal_feature_logs id; Type: DEFAULT; Schema: public; Owner: analeyes
--

ALTER TABLE ONLY public.signal_feature_logs ALTER COLUMN id SET DEFAULT nextval('public.signal_feature_logs_id_seq'::regclass);


--
-- Name: trades id; Type: DEFAULT; Schema: public; Owner: analeyes
--

ALTER TABLE ONLY public.trades ALTER COLUMN id SET DEFAULT nextval('public.trades_id_seq'::regclass);


--
-- Data for Name: alembic_version; Type: TABLE DATA; Schema: public; Owner: analeyes
--

COPY public.alembic_version (version_num) FROM stdin;
001
\.


--
-- Data for Name: ensemble_backtest_results; Type: TABLE DATA; Schema: public; Owner: analeyes
--

COPY public.ensemble_backtest_results (asset_class, signal_id, symbol, model, outcome, pnl_percentage, closed_at, id, created_at, updated_at) FROM stdin;
\.


--
-- Data for Name: ensemble_model_decisions; Type: TABLE DATA; Schema: public; Owner: analeyes
--

COPY public.ensemble_model_decisions (asset_class, signal_id, symbol, model, "timestamp", decision, confidence, raw_response, id, created_at, updated_at) FROM stdin;
\.


--
-- Data for Name: signal_feature_logs; Type: TABLE DATA; Schema: public; Owner: analeyes
--

COPY public.signal_feature_logs (asset_class, symbol, display_name, signal_id_uuid, initial_metric_timestamp, trigger_reason, heuristic_signal_consensus, composite_score_value, feat_current_price, feat_price_pct, feat_market_state, feat_rsi, feat_macd, feat_macd_signal, feat_macd_hist, feat_ema_short, feat_ema_long, feat_ema_50, feat_ema_200, feat_adx, feat_bb_upper, feat_bb_middle, feat_bb_lower, feat_bb_width, feat_atr, feat_atr_pct, feat_vwap, feat_vol_rel, feat_obv, feat_support_nearest, feat_resistance_nearest, feat_sp500_correlation, feat_dxy_correlation, feat_bid_ask_spread_pips, feat_funding_rate, feat_open_interest_z, feat_liquidations_long_usd, feat_liquidations_short_usd, feat_cvd, ohlcv_open_price, ohlcv_high_price, ohlcv_low_price, ohlcv_close_price, ohlcv_volume, strat_indicators_consensus, strat_patterns_consensus, historical_snapshots_json, historical_ohlcv_json, indicators_json, patterns_json, features_json, ai_signal_type, ai_confidence, ai_reason_summary, ai_entry_price_suggestion, ai_tp_price_suggestion, ai_sl_price_suggestion, ai_leverage_suggestion, ai_consensus_achieved, telegram_message_sent, tracker_status, tracker_entry_price, tracker_exit_price, tracker_pnl_percent, tracker_pnl_usdt, tracker_duration_seconds, tracker_closed_at, id, created_at, updated_at) FROM stdin;
\.


--
-- Data for Name: trades; Type: TABLE DATA; Schema: public; Owner: analeyes
--

COPY public.trades (asset_class, signal_id_uuid, symbol, direction, entry_price, exit_price, leverage, status, pnl, user_telegram_id, opened_at, closed_at, id, created_at, updated_at) FROM stdin;
\.


--
-- Name: ensemble_backtest_results_id_seq; Type: SEQUENCE SET; Schema: public; Owner: analeyes
--

SELECT pg_catalog.setval('public.ensemble_backtest_results_id_seq', 1, false);


--
-- Name: ensemble_model_decisions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: analeyes
--

SELECT pg_catalog.setval('public.ensemble_model_decisions_id_seq', 1, false);


--
-- Name: signal_feature_logs_id_seq; Type: SEQUENCE SET; Schema: public; Owner: analeyes
--

SELECT pg_catalog.setval('public.signal_feature_logs_id_seq', 1, false);


--
-- Name: trades_id_seq; Type: SEQUENCE SET; Schema: public; Owner: analeyes
--

SELECT pg_catalog.setval('public.trades_id_seq', 1, false);


--
-- Name: alembic_version alembic_version_pkc; Type: CONSTRAINT; Schema: public; Owner: analeyes
--

ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);


--
-- Name: ensemble_backtest_results ensemble_backtest_results_pkey; Type: CONSTRAINT; Schema: public; Owner: analeyes
--

ALTER TABLE ONLY public.ensemble_backtest_results
    ADD CONSTRAINT ensemble_backtest_results_pkey PRIMARY KEY (id);


--
-- Name: ensemble_model_decisions ensemble_model_decisions_pkey; Type: CONSTRAINT; Schema: public; Owner: analeyes
--

ALTER TABLE ONLY public.ensemble_model_decisions
    ADD CONSTRAINT ensemble_model_decisions_pkey PRIMARY KEY (id);


--
-- Name: signal_feature_logs signal_feature_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: analeyes
--

ALTER TABLE ONLY public.signal_feature_logs
    ADD CONSTRAINT signal_feature_logs_pkey PRIMARY KEY (id);


--
-- Name: trades trades_pkey; Type: CONSTRAINT; Schema: public; Owner: analeyes
--

ALTER TABLE ONLY public.trades
    ADD CONSTRAINT trades_pkey PRIMARY KEY (id);


--
-- Name: ix_ensemble_model_decisions_signal_id; Type: INDEX; Schema: public; Owner: analeyes
--

CREATE INDEX ix_ensemble_model_decisions_signal_id ON public.ensemble_model_decisions USING btree (signal_id);


--
-- Name: ix_signal_feature_logs_symbol_ts; Type: INDEX; Schema: public; Owner: analeyes
--

CREATE INDEX ix_signal_feature_logs_symbol_ts ON public.signal_feature_logs USING btree (symbol, initial_metric_timestamp);


--
-- PostgreSQL database dump complete
--

\unrestrict XeEIfeLV3kGPRcWRZnjUHzLGjZNpceX5fjuKZ1BBD8jE9W4rMVHbtT74pAo1m2D

