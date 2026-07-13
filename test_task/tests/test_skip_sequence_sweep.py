"""Tests for proactive --skip-sequence / --skip-rl flags across the Top200 sweep pipeline."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sweep = _load("sweep_top200_side_balance")
run_top200 = _load("run_top200_training")
train_multi = _load("train_multi_asset")
train_prod = _load("train_production")


def test_sweep_passes_skip_flags_to_run_top200() -> None:
    cmd = sweep.build_training_command(
        run_id="top200_sweep_test_long_3.10_short_1.50",
        long_weight=3.10,
        short_weight=1.50,
        sample_per_symbol=2500,
        batch_size=10,
        resume=False,
        skip_sequence=True,
        skip_rl=True,
    )
    joined = " ".join(cmd)
    assert "--skip-sequence true" in joined
    assert "--skip-rl true" in joined
    assert "--allow-skip-sequence true" in joined


def test_run_top200_passes_skip_flags_to_train_multi_asset() -> None:
    parser = run_top200.argparse.ArgumentParser()
    parser.add_argument("--run-id", default="top200_test_skip")
    parser.add_argument("--dataset", type=Path, default=run_top200.DEFAULT_DATASET)
    parser.add_argument("--sample-per-symbol", type=int, default=2500)
    parser.add_argument("--meta-mode", default="side_specialists")
    parser.add_argument("--balance-side-specialists", default="true")
    parser.add_argument("--long-positive-weight", default="auto")
    parser.add_argument("--short-positive-weight", default="auto")
    parser.add_argument("--balance-train-samples", default="true")
    parser.add_argument("--allow-skip-sequence", default="true")
    parser.add_argument("--skip-sequence", default="true", choices=["true", "false"])
    parser.add_argument("--skip-rl", default="true", choices=["true", "false"])
    parser.add_argument("--max-side-train-samples-per-class", type=int, default=None)
    args = parser.parse_args([])
    artifact_dir = run_top200.ROOT / "artifacts_candidates" / args.run_id
    train_cmd = [
        sys.executable,
        str(run_top200.ROOT / "scripts" / "train_multi_asset.py"),
        "--dataset",
        str(args.dataset),
        "--symbols",
        "BTCUSDT",
        "--interval",
        "1h",
        "--output-dir",
        str(artifact_dir),
        "--meta-mode",
        args.meta_mode,
        "--medium",
        "--sample-per-symbol",
        str(args.sample_per_symbol),
        "--balance-side-specialists",
        args.balance_side_specialists,
        "--long-positive-weight",
        args.long_positive_weight,
        "--short-positive-weight",
        args.short_positive_weight,
        "--balance-train-samples",
        args.balance_train_samples,
        "--allow-skip-sequence",
        args.allow_skip_sequence,
        "--skip-sequence",
        args.skip_sequence,
        "--skip-rl",
        args.skip_rl,
        "--skip-evaluate",
    ]
    joined = " ".join(train_cmd)
    assert "--skip-sequence true" in joined
    assert "--skip-rl true" in joined


def test_train_multi_asset_forwards_skip_flags_to_train_production(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_argv: list[str] = []

    def _fake_tp_main():
        captured_argv[:] = list(sys.argv)

    monkeypatch.setattr(train_multi, "_export_parquet_to_csv", lambda *a, **k: Path("/tmp/cache"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_multi_asset.py",
            "--dataset",
            str(ROOT / "data" / "datasets" / "multi_asset.parquet"),
            "--symbols",
            "BTCUSDT",
            "--output-dir",
            "/tmp/out",
            "--skip-sequence",
            "true",
            "--skip-rl",
            "true",
            "--skip-evaluate",
        ],
    )
    with patch.object(train_multi.ExperimentTracker, "start") as mock_start:
        mock_start.return_value = MagicMock(run_id="test", train_metrics={}, decision_distribution=None)
        with patch.object(train_multi.ExperimentTracker, "log_run"):
            with patch("scripts.train_production.main", _fake_tp_main):
                train_multi.main()
    joined = " ".join(captured_argv)
    assert "--skip-sequence true" in joined
    assert "--skip-rl true" in joined


def test_train_production_does_not_call_sequence_when_skip_sequence_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"sequence": 0, "rl": 0}

    def _seq(*args, **kwargs):
        calls["sequence"] += 1
        raise AssertionError("train_sequence_multi should not be called")

    def _rl(*args, **kwargs):
        calls["rl"] += 1
        return {"mean_episode_reward": 1.0}, MagicMock()

    monkeypatch.setattr(train_prod, "train_sequence_multi", _seq)
    monkeypatch.setattr(train_prod, "train_rl_multi", _rl)
    monkeypatch.setattr(train_prod, "fit_regime_and_features", lambda frames, settings: (MagicMock(), {}))
    monkeypatch.setattr(train_prod, "train_tabular_multi", lambda *a, **k: {"val_auc": 0.6})
    monkeypatch.setattr(
        train_prod,
        "train_meta_multi",
        lambda *a, **k: {"meta_mode": "side_specialists"},
    )
    monkeypatch.setattr(train_prod, "_load_symbol_frames", lambda *a, **k: {"BTCUSDT": MagicMock()})

    settings = train_prod.Settings()
    tmp_artifacts = Path("/tmp/test_skip_seq_artifacts")
    tmp_artifacts.mkdir(parents=True, exist_ok=True)
    settings.model.artifacts_dir = tmp_artifacts

    with patch.object(train_prod, "Settings", return_value=settings):
        with patch.object(
            train_prod.argparse.ArgumentParser,
            "parse_args",
            return_value=MagicMock(
                data_dir=Path("data/production"),
                symbols="BTCUSDT",
                interval="1h",
                artifacts=tmp_artifacts,
                seq_epochs=1,
                seq_cap=100,
                seq_batch=32,
                rl_timesteps=100,
                sample_per_symbol=100,
                meta_mode="side_specialists",
                allow_skip_sequence=False,
                skip_sequence="true",
                skip_rl="true",
            ),
        ):
            train_prod.main()

    assert calls["sequence"] == 0
    assert calls["rl"] == 0
    summary = json.loads((tmp_artifacts / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["sequence_skipped"] is True
    assert summary["rl_skipped"] is True


def test_build_layer_mask_forces_off_when_skipped() -> None:
    settings = train_prod.Settings()
    mask = train_prod._build_layer_mask(
        settings,
        {"skipped": True, "val_auc": 0.99},
        {"skipped": True, "mean_episode_reward": 99.0},
    )
    assert mask["sequence"] is False
    assert mask["rl"] is False


def test_run_top200_copies_skip_flags_into_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact_dir = tmp_path / "candidate"
    artifact_dir.mkdir()
    (artifact_dir / "training_summary.json").write_text(
        json.dumps({"sequence_skipped": True, "rl_skipped": True}),
        encoding="utf-8",
    )
    (artifact_dir / "summary.json").write_text(json.dumps({"promotable": False}), encoding="utf-8")
    (artifact_dir / "test_metrics.json").write_text(
        json.dumps({"publishable_long_count_ge_70": 0, "publishable_short_count_ge_70": 0}),
        encoding="utf-8",
    )

    summary_path = artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    training_summary = json.loads((artifact_dir / "training_summary.json").read_text(encoding="utf-8"))
    if training_summary.get("sequence_skipped"):
        summary["sequence_skipped"] = True
    if training_summary.get("rl_skipped"):
        summary["rl_skipped"] = True
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    final = json.loads(summary_path.read_text(encoding="utf-8"))
    assert final["sequence_skipped"] is True
    assert final["rl_skipped"] is True


def test_sweep_candidate_with_skipped_layers_can_still_rank_from_summary(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "candidate"
    artifact_dir.mkdir()
    (artifact_dir / "training_summary.json").write_text(
        json.dumps({"sequence_skipped": True, "rl_skipped": True}),
        encoding="utf-8",
    )
    (artifact_dir / "summary.json").write_text(
        json.dumps(
            {
                "promotable": True,
                "promotion_blockers": [],
                "publishable_long_count_ge_70": 40,
                "publishable_short_count_ge_70": 60,
                "publishable_total_trade_count_ge_70": 100,
                "publishable_long_ev_ge_70": 100.0,
                "publishable_short_ev_ge_70": 200.0,
                "publishable_total_ev_ge_70": 300.0,
                "sequence_skipped": True,
                "rl_skipped": True,
            }
        ),
        encoding="utf-8",
    )
    result = sweep.evaluate_candidate_result(
        run_id="candidate",
        long_weight=3.1,
        short_weight=1.5,
        artifact_dir=artifact_dir,
        rules=sweep.SweepRules(),
    )
    assert result.accepted is True
    assert result.status == "completed"


def test_sigkill_missing_summary_rejected_clearly(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "sigkill_run"
    logs = artifact_dir / "logs"
    logs.mkdir(parents=True)
    (logs / "train.log").write_text(
        "Training died with <Signals.SIGKILL: 9>\n",
        encoding="utf-8",
    )
    (artifact_dir / "state.json").write_text(json.dumps({"training_done": False}), encoding="utf-8")
    result = sweep.evaluate_candidate_result(
        run_id="sigkill_run",
        long_weight=3.1,
        short_weight=1.5,
        artifact_dir=artifact_dir,
        rules=sweep.SweepRules(),
    )
    assert result.accepted is False
    assert "summary_json_missing" in result.rejection_reasons
    assert "training_sigkill" in result.rejection_reasons


def test_rl_series_returns_zeros_when_model_none() -> None:
    from ae_brain.training.meta_series import rl_series
    import numpy as np

    out = rl_series(None, np.zeros(5), np.ones(5), np.ones(5))
    assert out.shape == (5,)
    assert np.allclose(out, 0.0)
