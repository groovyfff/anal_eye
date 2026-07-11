"""Tests for top-200 universe discovery, env apply, promotion, runtime verify, training orchestrator."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ae_brain.artifact_verify import missing_runtime_artifacts, required_runtime_artifacts
from ae_brain.env_universe import count_csv_symbols, parse_env_file, update_env_file
from ae_brain.universe_top200 import (
    LEGACY_SIX_SYMBOLS,
    build_top200_universe,
    filter_trading_perpetual_usdt,
    load_universe_txt,
    write_universe_files,
)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_exchange_info() -> dict:
    symbols = []
    for i in range(250):
        sym = f"COIN{i:03d}USDT"
        symbols.append(
            {
                "symbol": sym,
                "quoteAsset": "USDT",
                "contractType": "PERPETUAL",
                "status": "TRADING",
            }
        )
    symbols.extend(
        [
            {"symbol": s, "quoteAsset": "USDT", "contractType": "PERPETUAL", "status": "TRADING"}
            for s in LEGACY_SIX_SYMBOLS
        ]
    )
    symbols.extend(
        [
            {"symbol": "BTCBUSD", "quoteAsset": "BUSD", "contractType": "PERPETUAL", "status": "TRADING"},
            {"symbol": "ETHUSD_PERP", "quoteAsset": "USD", "contractType": "PERPETUAL", "status": "TRADING"},
            {"symbol": "DELISTEDUSDT", "quoteAsset": "USDT", "contractType": "PERPETUAL", "status": "BREAK"},
            {"symbol": "DELIVERYUSDT", "quoteAsset": "USDT", "contractType": "CURRENT_QUARTER", "status": "TRADING"},
        ]
    )
    return {"symbols": symbols}


def _fake_tickers(exchange_info: dict) -> list[dict]:
    rows = []
    for item in exchange_info["symbols"]:
        sym = item["symbol"]
        if item.get("quoteAsset") != "USDT" or item.get("contractType") != "PERPETUAL":
            continue
        if item.get("status") != "TRADING":
            continue
        vol = 1_000_000 - (hash(sym) % 900_000)
        if sym in LEGACY_SIX_SYMBOLS:
            vol = 10_000
        rows.append({"symbol": sym, "quoteVolume": str(vol)})
    return rows


def test_top200_discovery_returns_exactly_200_symbols() -> None:
    ex = _fake_exchange_info()
    record = build_top200_universe(ex, _fake_tickers(ex), target_size=200)
    assert len(record.symbols) == 200
    assert len(set(record.symbols)) == 200


def test_existing_six_symbols_force_included() -> None:
    ex = _fake_exchange_info()
    record = build_top200_universe(ex, _fake_tickers(ex), target_size=200)
    for sym in LEGACY_SIX_SYMBOLS:
        assert sym in record.symbols


def test_non_usdt_non_perpetual_non_trading_filtered_out() -> None:
    ex = _fake_exchange_info()
    allowed = filter_trading_perpetual_usdt(ex)
    assert "BTCBUSD" not in allowed
    assert "DELISTEDUSDT" not in allowed
    assert "DELIVERYUSDT" not in allowed
    assert "BTCUSDT" in allowed


def test_env_updater_preserves_unrelated_keys_and_secrets(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "TELEGRAM_BOT_TOKEN=secret-token\nAEB_DB_PASSWORD=dbpass\nFOO=bar\n",
        encoding="utf-8",
    )
    symbols = [f"SYM{i}USDT" for i in range(200)]
    updates = {
        "SYMBOLS": ",".join(symbols),
        "BINANCE_SYMBOLS": ",".join(symbols),
        "ANAL_EYES_ALLOWED_SYMBOLS": ",".join(symbols),
        "AEB_ALLOWED_SYMBOLS": ",".join(symbols),
        "SYMBOL_LIMIT": "200",
        "AEB_ONLY_BTC": "false",
        "AEB_MIN_PUBLISH_CONFIDENCE": "0.70",
        "NOTIFICATION_MIN_CONFIDENCE": "0.70",
    }
    update_env_file(env_path, updates)
    text = env_path.read_text(encoding="utf-8")
    assert "TELEGRAM_BOT_TOKEN=secret-token" in text
    assert "AEB_DB_PASSWORD=dbpass" in text
    assert "FOO=bar" in text
    assert count_csv_symbols(parse_env_file(env_path)["SYMBOLS"]) == 200


def test_env_updater_writes_exactly_200_symbols(tmp_path: Path) -> None:
    symbols = [f"AAA{i:03d}USDT" for i in range(200)]
    env_path = tmp_path / ".env"
    env_path.write_text("# init\n", encoding="utf-8")
    update_env_file(env_path, {"AEB_ALLOWED_SYMBOLS": ",".join(symbols)})
    assert count_csv_symbols(parse_env_file(env_path)["AEB_ALLOWED_SYMBOLS"]) == 200


def test_promotion_backs_up_old_artifacts(tmp_path: Path) -> None:
    promote = _load_script("promote_top200_artifact")
    production = tmp_path / "artifacts"
    production.mkdir()
    (production / "tabular_model.joblib").write_bytes(b"old")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    for name in required_runtime_artifacts(candidate):
        (candidate / name).write_bytes(b"x")
    (candidate / "training_summary.json").write_text(
        json.dumps({"meta": {"meta_mode": "two_stage"}}),
        encoding="utf-8",
    )
    (candidate / "test_metrics.json").write_text(
        json.dumps(
            {
                "expected_ev_usd": 1.0,
                "net_pnl_usd": 1.0,
                "max_drawdown": 0.1,
                "trade_count": 30,
                "long_count": 10,
                "short_count": 10,
                "precision_at_conf_70": 0.5,
                "publishable_long_count_ge_70": 5,
                "publishable_short_count_ge_70": 5,
                "publishable_long_ev_ge_70": 1.0,
                "publishable_short_ev_ge_70": 1.0,
                "publishable_total_ev_ge_70": 2.0,
                "publishable_total_trade_count_ge_70": 10,
                "ev_by_confidence_bucket": {"0.70-0.80": 1.0},
                "per_symbol": {},
            }
        ),
        encoding="utf-8",
    )
    backups = tmp_path / "artifacts_backups"
    backup_dir = promote.backup_production_artifacts(production, backups)
    assert backup_dir.exists()
    assert (backup_dir / "tabular_model.joblib").read_bytes() == b"old"
    promote.promote_with_backup(candidate, production, backups, force=True)
    assert (production / "tabular_model.joblib").exists()
    assert missing_runtime_artifacts(production) == []


def test_runtime_verification_fails_on_btc_only_mode(tmp_path: Path) -> None:
    verify = _load_script("verify_top200_runtime")
    symbols = [f"SYM{i}USDT" for i in range(200)]
    env_path = tmp_path / ".env"
    csv = ",".join(symbols)
    env_path.write_text(
        "\n".join(
            [
                f"SYMBOLS={csv}",
                f"BINANCE_SYMBOLS={csv}",
                f"ANAL_EYES_ALLOWED_SYMBOLS={csv}",
                f"AEB_ALLOWED_SYMBOLS={csv}",
                "SYMBOL_LIMIT=200",
                "AEB_ONLY_BTC=true",
                "AEB_MIN_PUBLISH_CONFIDENCE=0.70",
                "NOTIFICATION_MIN_CONFIDENCE=0.70",
            ]
        ),
        encoding="utf-8",
    )
    errors = verify.verify_env_symbols(env_path, expected_count=200)
    assert any("AEB_ONLY_BTC" in err for err in errors)


def test_runtime_verification_fails_if_symbol_count_not_200(tmp_path: Path) -> None:
    verify = _load_script("verify_top200_runtime")
    env_path = tmp_path / ".env"
    env_path.write_text("AEB_ALLOWED_SYMBOLS=BTCUSDT,ETHUSDT\n", encoding="utf-8")
    errors = verify.verify_env_symbols(env_path, expected_count=200)
    assert any("200" in err for err in errors)


def test_training_orchestrator_supports_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_top200 = _load_script("run_top200_training")
    universe = tmp_path / "universe.txt"
    symbols = list(LEGACY_SIX_SYMBOLS) + [f"ALT{i:03d}USDT" for i in range(194)]
    universe.write_text("\n".join(symbols) + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_top200_training.py",
            "--dry-run",
            "--universe-txt",
            str(universe),
        ],
    )
    run_top200.main()

    artifact_dir = tmp_path / "artifacts_candidates"
    runs = list(artifact_dir.glob("top200_*"))
    assert runs == [], "dry-run should not persist artifact tree when cwd is isolated"

    # Batch dry-run should not invoke downloader.
    artifact_dir = tmp_path / "artifacts_candidates" / "top200_test"
    logger = run_top200.setup_logger(artifact_dir / "logs" / "train.log")
    state = {"downloaded_symbols": []}
    time_range = run_top200.parse_cli_time_range("2021-01-01", "2026-01-01")
    run_top200.download_symbols_batched(
        symbols[:5],
        raw_dir=tmp_path / "raw",
        time_range=time_range,
        batch_size=2,
        state=state,
        logger=logger,
        dry_run=True,
    )
    assert state["downloaded_symbols"] == []


def test_downloader_resume_does_not_redownload_existing_complete_files(tmp_path: Path) -> None:
    dm = _load_script("download_market_data")
    out_dir = tmp_path / "binance"
    sym_dir = out_dir / "BTCUSDT" / "1h"
    sym_dir.mkdir(parents=True)
    out_path = sym_dir / "klines.csv"
    end_ms = 1_800_000_000_000
    rows = []
    ts = end_ms - 149 * 3_600_000
    for i in range(150):
        row_ts = ts + i * 3_600_000
        rows.append(
            {
                "timestamp": pd.Timestamp(row_ts, unit="ms", tz="UTC").isoformat(),
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 10.0,
                "trades_count": 1,
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)
    time_range = dm.TimeRange(
        start_ms=ts,
        end_ms=end_ms,
        start_utc=pd.Timestamp(ts, unit="ms", tz="UTC"),
        end_utc=pd.Timestamp(end_ms, unit="ms", tz="UTC"),
    )
    run_top200 = _load_script("run_top200_training")
    assert run_top200.klines_complete(out_path, time_range, min_rows=100)

    calls: list[str] = []

    def _fake_download(*args, **kwargs):
        calls.append("download")
        return out_path

    with patch.object(dm, "download_symbol", side_effect=_fake_download):
        logger = run_top200.setup_logger(tmp_path / "train.log")
        state: dict = {"downloaded_symbols": []}
        run_top200.download_symbols_batched(
            ["BTCUSDT"],
            raw_dir=out_dir,
            time_range=time_range,
            batch_size=1,
            state=state,
            logger=logger,
            dry_run=False,
        )
    assert calls == []
    assert "BTCUSDT" in state["downloaded_symbols"]


def test_warn_publishable_sides_logs_without_raising(tmp_path: Path) -> None:
    run_top200 = _load_script("run_top200_training")
    logger = run_top200.setup_logger(tmp_path / "train.log")
    warnings = run_top200.warn_publishable_sides(
        {"publishable_long_count_ge_70": 0, "publishable_short_count_ge_70": 100},
        logger=logger,
    )
    assert len(warnings) == 1
    assert "LONG" in warnings[0]


def test_side_specialists_diagnostic_snapshot(tmp_path: Path) -> None:
    run_top200 = _load_script("run_top200_training")
    artifact_dir = tmp_path / "candidate"
    artifact_dir.mkdir()
    (artifact_dir / "side_specialists_report.json").write_text(
        json.dumps(
            {
                "second_pass_threshold_report": {
                    "per_threshold": {"0.70": {"publishable_LONG": 1, "publishable_SHORT": 2}}
                },
                "calibration_ceiling_summary": {"LONG": {}},
                "side_balance": {"label_counts": {}},
            }
        ),
        encoding="utf-8",
    )
    snap = run_top200._side_specialists_diagnostic_snapshot(artifact_dir)
    assert snap is not None
    assert snap["diagnostic_only"] is True
    assert snap["promotion_source_of_truth"] == "summary.json"
    assert snap["second_pass_threshold_0_70"]["publishable_LONG"] == 1


def test_write_universe_files_roundtrip(tmp_path: Path) -> None:
    ex = _fake_exchange_info()
    record = build_top200_universe(ex, _fake_tickers(ex), target_size=200)
    txt = tmp_path / "universe.txt"
    js = tmp_path / "universe.json"
    write_universe_files(txt, js, record)
    loaded = load_universe_txt(txt)
    assert len(loaded) == 200
    meta = json.loads(js.read_text(encoding="utf-8"))
    assert meta["count"] == 200
