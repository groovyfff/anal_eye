"""Dynamic trading-symbol support for the inference agent."""

from __future__ import annotations

import pytest

from ae_brain.contracts import Decision, FinalSignal, LayerProbabilities, TradeCandidate
from ae_brain.layers.fusion import FusionContext, FusionLayer
from ae_brain.messaging.candidate_normalizer import normalize_candidate
from ae_brain.messaging.rabbitmq import build_signal_final_payload
from ae_brain.symbols import extract_base_asset, normalize_symbol, require_symbol
from ae_brain.training.synthetic import generate_synthetic_candles
from tests.test_smoke import _components


@pytest.mark.parametrize(
    ("symbol", "expected_asset"),
    [
        ("BTCUSDT", "BTC"),
        ("ETHUSDT", "ETH"),
        ("SOLUSDT", "SOL"),
        ("bnbusdt", "BNB"),
        ("XRPUSDT", "XRP"),
        ("DOGEUSDT", "DOGE"),
    ],
)
def test_extract_base_asset(symbol: str, expected_asset: str) -> None:
    assert extract_base_asset(symbol) == expected_asset
    assert normalize_symbol(symbol) == symbol.upper()


def test_require_symbol_rejects_missing() -> None:
    with pytest.raises(ValueError, match="missing_symbol"):
        require_symbol(None)
    with pytest.raises(ValueError, match="missing_symbol"):
        require_symbol("  ")


def test_trade_candidate_from_message_requires_symbol() -> None:
    with pytest.raises(ValueError, match="missing_symbol"):
        TradeCandidate.from_message({"candles": []})


def test_normalize_candidate_rejects_missing_symbol() -> None:
    result = normalize_candidate({"candles": [{"close": 1.0}], "features": {"rsi": 50}})
    assert result.skip_reason == "missing_symbol"
    assert result.payload is None


def _candidate_payload(symbol: str, *, seed: int) -> dict:
    candles = generate_synthetic_candles(n=64, seed=seed)
    rows = candles.assign(ts=candles["ts"].astype(str)).to_dict(orient="records")
    return {
        "source": "binance",
        "market": "futures",
        "asset_class": "crypto",
        "symbol": symbol,
        "timeframe": "1h",
        "current_price": float(candles["close"].iloc[-1]),
        "market_state": "trend",
        "composite_score": 0.82,
        "features": {
            "current_price": float(candles["close"].iloc[-1]),
            "rsi": 55.4,
            "macd": 1.2,
            "macd_signal": 0.8,
            "macd_hist": 0.4,
            "adx": 28.0,
            "atr": 420.0,
            "ema_short": float(candles["close"].iloc[-1]) * 0.99,
            "ema_long": float(candles["close"].iloc[-1]) * 0.97,
            "ema_50": float(candles["close"].iloc[-1]) * 0.96,
            "ema_200": float(candles["close"].iloc[-1]) * 0.94,
            "volume_change": 0.35,
            "price_change_1h": 2.1,
        },
        "candles": rows,
    }


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
def test_normalize_candidate_preserves_symbol(symbol: str) -> None:
    result = normalize_candidate(_candidate_payload(symbol, seed=hash(symbol) % 1000))
    assert result.skip_reason is None
    assert result.payload is not None
    assert result.payload["symbol"] == symbol
    assert result.symbol == symbol


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
def test_agent_output_echoes_input_symbol(symbol: str) -> None:
    """End-to-end symbol passthrough: normalize -> candidate -> fusion -> publish payload."""
    norm = normalize_candidate(_candidate_payload(symbol, seed=hash(symbol) % 1000))
    assert norm.payload is not None
    candidate = TradeCandidate.from_message(norm.payload)

    _, _, _, _, fusion = _components()
    probs = LayerProbabilities(
        tabular_p_up=0.78,
        sequence_p_continuation=0.8,
        sequence_trend_sign=1.0,
        rl_target_exposure=0.7,
    )
    ctx = FusionContext(
        symbol=candidate.symbol,
        entry_price=100.0,
        atr=1.0,
        adv_usd=5_000_000,
        correlation_id=candidate.correlation_id,
    )
    signal: FinalSignal = fusion.decide(probs, ctx)
    signal.signal_id = candidate.signal_id
    signal.asset_class = candidate.asset_class

    out = signal.to_dict()
    assert out["symbol"] == symbol
    assert out["asset"] == extract_base_asset(symbol)
    assert symbol in out["reason_summary"]
    assert extract_base_asset(symbol) in out["reason_summary"]

    published = build_signal_final_payload(signal, candidate)
    assert published["symbol"] == symbol
    assert published["asset"] == extract_base_asset(symbol)
    assert published["decision"] in {d.value for d in Decision}
