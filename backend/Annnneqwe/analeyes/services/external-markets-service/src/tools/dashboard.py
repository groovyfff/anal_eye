import json
import threading
import time
import os
import sys
from datetime import datetime, timezone
from typing import Any

# Импорт для исправления ошибки ScriptRunContext
from streamlit.runtime.scriptrunner import add_script_run_ctx

# Add the project root to sys.path to ensure 'src' is found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pika
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.settings import load_settings
from src.utils.logger import setup_logging

# --- Configuration & Setup ---
st.set_page_config(
    page_title="AnalEyes Dashboard",
    page_icon="📈",
    layout="wide",
)

# Load settings for RabbitMQ
settings = load_settings("config/settings.yml")
rabbit_cfg = settings.get('rabbitmq', {})

# Inside Docker, we should use 'rabbitmq' hostname if defined, else fallback to settings
RABBIT_URL = os.environ.get('RABBITMQ_URL') or rabbit_cfg.get('url', 'amqp://guest:guest@localhost:5672/')

# Fix for docker networking: if localhost is in the URL and we're in docker, swap it
if 'localhost' in RABBIT_URL and os.path.exists('/.dockerenv'):
    RABBIT_URL = RABBIT_URL.replace('localhost', 'rabbitmq')

EXCHANGE = rabbit_cfg.get('exchange', 'analeyes_exchange')

# --- State Management ---
if 'live_prices' not in st.session_state:
    st.session_state.live_prices = {}
if 'candidates' not in st.session_state:
    st.session_state.candidates = []
if 'max_candidates' not in st.session_state:
    st.session_state.max_candidates = 50
if 'last_update_count' not in st.session_state:
    st.session_state.last_update_count = 0

# --- RabbitMQ Consumer ---
def rabbit_worker():
    """Background thread to consume RabbitMQ messages."""
    try:
        parameters = pika.URLParameters(RABBIT_URL)
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        channel.exchange_declare(exchange=EXCHANGE, exchange_type='topic', durable=True)

        # Temporary queues for the dashboard
        result = channel.queue_declare(queue='', exclusive=True)
        queue_name = result.method.queue

        # Bind to live prices and candidates
        channel.queue_bind(exchange=EXCHANGE, queue=queue_name, routing_key='data.live_prices.#')
        channel.queue_bind(exchange=EXCHANGE, queue=queue_name, routing_key='data.candidates.#')

        def callback(ch, method, properties, body):
            try:
                payload = json.loads(body)
                routing_key = method.routing_key
                
                # Используем блокировку или проверяем наличие session_state, 
                # так как поток фоновый
                if 'live_prices' in routing_key:
                    symbol = payload.get('symbol')
                    if symbol:
                        st.session_state.live_prices[symbol] = payload
                        st.session_state.last_update_count += 1
                
                elif 'candidates' in routing_key:
                    st.session_state.candidates.insert(0, payload)
                    # Keep only the last N candidates
                    max_c = st.session_state.get('max_candidates', 50)
                    if len(st.session_state.candidates) > max_c:
                        st.session_state.candidates = st.session_state.candidates[:max_c]
            except Exception as e:
                print(f"Error in callback: {e}")

        channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
        channel.start_consuming()
    except Exception as e:
        print(f"RabbitMQ Connection Error: {e}")
        time.sleep(5)
        rabbit_worker() # Simple retry

# --- Start RabbitMQ thread once (with Context) ---
if 'rabbit_thread' not in st.session_state:
    thread = threading.Thread(target=rabbit_worker, daemon=True)
    # ПРИВЯЗКА КОНТЕКСТА: это самое важное изменение
    add_script_run_ctx(thread)
    st.session_state.rabbit_thread = thread
    thread.start()

# --- Helpers ---
def fmt_time(ts_ms):
    if not ts_ms: return "-"
    try:
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        return dt.strftime("%H:%M:%S")
    except:
        return "-"

# --- UI Components ---
st.title("🛡️ AnalEyes External Markets Dashboard")

# Top Metrics Row
if st.session_state.live_prices:
    # Показываем первые 6 символов в метриках
    display_symbols = list(st.session_state.live_prices.keys())[:6]
    cols = st.columns(len(display_symbols))
    for i, symbol in enumerate(display_symbols):
        data = st.session_state.live_prices[symbol]
        with cols[i]:
            price = data.get('price', 0)
            st.metric(label=symbol, value=f"{price:.4f}")

tabs = st.tabs(["🔥 AI Candidates", "📊 Live Stream", "⚙️ Settings"])

with tabs[0]:
    st.subheader("Latest AI Signals")
    if not st.session_state.candidates:
        st.info("Waiting for signals from AI Service...")
    else:
        for cand in st.session_state.candidates:
            symbol = cand.get('symbol', 'UNKNOWN')
            reason = cand.get('trigger_reason', 'N/A')
            score = cand.get('composite_score', 0)
            ts = cand.get('ts', 0)
            
            with st.expander(f"[{fmt_time(ts)}] {symbol} - {reason} (Score: {score})", expanded=True):
                c1, c2, c3 = st.columns(3)
                
                features = cand.get('features', {})
                with c1:
                    st.write("**Price Action**")
                    st.write(f"Price: {cand.get('price', 'N/A')}")
                    st.write(f"RSI: {features.get('rsi_14', 'N/A')}")
                    st.write(f"ATR: {features.get('atr_14', 'N/A')}")
                
                with c2:
                    st.write("**Technical Indicators**")
                    st.write(f"EMA 8/21: {features.get('ema_8', 'N/A')}/{features.get('ema_21', 'N/A')}")
                    st.write(f"Supertrend: {features.get('supertrend', 'N/A')}")
                    st.write(f"ADX: {features.get('adx_14', 'N/A')}")

                with c3:
                    st.write("**Correlation & Context**")
                    st.write(f"SP500 Corr: {features.get('sp500_correlation', 'N/A')}")
                    st.write(f"DXY Corr: {features.get('dxy_correlation', 'N/A')}")
                    st.write(f"Rel Volume: {features.get('relative_volume', 'N/A')}")
                
                # Visual Score Bar
                st.progress(min(max(float(score) / 100, 0.0), 1.0), text=f"Composite Score: {score}")

with tabs[1]:
    st.subheader("Real-time Market Data")
    if not st.session_state.live_prices:
        st.info("Waiting for price updates...")
    else:
        df_prices = pd.DataFrame.from_dict(st.session_state.live_prices, orient='index')
        cols_to_show = ['symbol', 'asset_class', 'price', 'bid', 'ask', 'timestamp']
        existing_cols = [c for c in cols_to_show if c in df_prices.columns]
        st.dataframe(df_prices[existing_cols], use_container_width=True)

with tabs[2]:
    st.session_state.max_candidates = st.slider("Max candidates to keep in memory", 10, 500, 50)
    st.write(f"Connected to: `{RABBIT_URL}`")
    st.write(f"Exchange: `{EXCHANGE}`")
    if st.button("Clear State"):
        st.session_state.live_prices = {}
        st.session_state.candidates = []
        st.rerun()

# Auto-refresh every 1 second
time.sleep(1)
st.rerun()