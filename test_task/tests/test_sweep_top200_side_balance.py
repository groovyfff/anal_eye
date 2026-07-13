"""Tests for Top200 side-balance sweep runner."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_sweep():
    spec = importlib.util.spec_from_file_location(
        "sweep_top200_side_balance",
        ROOT / "scripts" / "sweep_top200_side_balance.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sweep_top200_side_balance"] = mod
    spec.loader.exec_module(mod)
    return mod


sweep = _load_sweep()


def _summary(
    *,
    promotable: bool,
    pub_long: int,
    pub_short: int,
    long_ev: float,
    short_ev: float,
    total_ev: float | None = None,
    blockers: list[str] | None = None,
) -> dict:
    total = pub_long + pub_short
    return {
        "promotable": promotable,
        "promotion_blockers": blockers or ([] if promotable else ["not_promotable"]),
        "publishable_long_count_ge_70": pub_long,
        "publishable_short_count_ge_70": pub_short,
        "publishable_total_trade_count_ge_70": total,
        "publishable_long_ev_ge_70": long_ev,
        "publishable_short_ev_ge_70": short_ev,
        "publishable_total_ev_ge_70": total_ev if total_ev is not None else long_ev + short_ev,
    }


def _side_report(pub_long: int, pub_short: int, total_ev: float) -> dict:
    return {
        "diagnostic_only": True,
        "second_pass_threshold_report": {
            "per_threshold": {
                "0.70": {
                    "publishable_LONG": pub_long,
                    "publishable_SHORT": pub_short,
                    "publishable_EV_total": total_ev,
                }
            }
        },
        "calibration_ceiling_summary": {"LONG": {"ceiling": 0.8}},
        "side_balance": {"label_counts": {"validation": {"LONG_profitable": pub_long}}},
    }


def test_generate_weight_grid() -> None:
    grid = sweep.generate_weight_grid()
    assert len(grid) == len(sweep.LONG_POSITIVE_WEIGHTS) * len(sweep.SHORT_POSITIVE_WEIGHTS)
    assert (3.05, 1.45) in grid
    assert (3.45, 1.75) in grid


def test_summary_first_ranking_prefers_balanced_over_side_report_only(tmp_path: Path) -> None:
    rules = sweep.SweepRules()
    balanced_dir = tmp_path / "balanced"
    balanced_dir.mkdir()
    (balanced_dir / "summary.json").write_text(
        json.dumps(_summary(promotable=True, pub_long=40, pub_short=60, long_ev=100.0, short_ev=200.0)),
        encoding="utf-8",
    )
    side_only_dir = tmp_path / "side_only"
    side_only_dir.mkdir()
    (side_only_dir / "side_specialists_report.json").write_text(
        json.dumps(_side_report(500, 900, 1_000_000.0)),
        encoding="utf-8",
    )

    balanced = sweep.evaluate_candidate_result(
        run_id="balanced",
        long_weight=3.1,
        short_weight=1.5,
        artifact_dir=balanced_dir,
        rules=rules,
    )
    side_only = sweep.evaluate_candidate_result(
        run_id="side_only",
        long_weight=3.2,
        short_weight=1.6,
        artifact_dir=side_only_dir,
        rules=rules,
    )
    ranked = sweep.rank_candidates([side_only, balanced])
    assert ranked[0].run_id == "balanced"
    assert ranked[0].accepted is True
    assert side_only.accepted is False
    assert "summary_json_missing" in side_only.rejection_reasons


def test_reject_missing_summary_json(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "missing"
    artifact_dir.mkdir()
    result = sweep.evaluate_candidate_result(
        run_id="missing",
        long_weight=3.1,
        short_weight=1.5,
        artifact_dir=artifact_dir,
        rules=sweep.SweepRules(),
    )
    assert result.status == "failed"
    assert result.accepted is False
    assert "summary_json_missing" in result.rejection_reasons


def test_reject_sigkill_missing_summary(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "sigkill"
    logs = artifact_dir / "logs"
    logs.mkdir(parents=True)
    (logs / "train.log").write_text("died with <Signals.SIGKILL: 9>\n", encoding="utf-8")
    result = sweep.evaluate_candidate_result(
        run_id="sigkill",
        long_weight=3.1,
        short_weight=1.5,
        artifact_dir=artifact_dir,
        rules=sweep.SweepRules(),
    )
    assert result.accepted is False
    assert "summary_json_missing" in result.rejection_reasons
    assert "training_sigkill" in result.rejection_reasons


def test_reject_side_report_only_candidate(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "side_only"
    artifact_dir.mkdir()
    (artifact_dir / "side_specialists_report.json").write_text(
        json.dumps(_side_report(112, 819, 1_457_312.0)),
        encoding="utf-8",
    )
    result = sweep.evaluate_candidate_result(
        run_id="side_only",
        long_weight=3.1,
        short_weight=1.5,
        artifact_dir=artifact_dir,
        rules=sweep.SweepRules(),
    )
    assert result.accepted is False
    assert "summary_json_missing" in result.rejection_reasons


def test_reject_long_only(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "long_only"
    artifact_dir.mkdir()
    (artifact_dir / "summary.json").write_text(
        json.dumps(_summary(promotable=False, pub_long=44, pub_short=5, long_ev=20_000.0, short_ev=1.0)),
        encoding="utf-8",
    )
    result = sweep.evaluate_candidate_result(
        run_id="long_only",
        long_weight=3.1,
        short_weight=1.5,
        artifact_dir=artifact_dir,
        rules=sweep.SweepRules(),
    )
    assert result.accepted is False
    assert any("short_count=" in r for r in result.rejection_reasons)


def test_reject_short_only(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "short_only"
    artifact_dir.mkdir()
    (artifact_dir / "summary.json").write_text(
        json.dumps(
            _summary(
                promotable=False,
                pub_long=0,
                pub_short=1117,
                long_ev=0.0,
                short_ev=500.0,
                blockers=["publishable_long_at_0.70=0"],
            )
        ),
        encoding="utf-8",
    )
    (artifact_dir / "side_specialists_report.json").write_text(
        json.dumps(_side_report(0, 1117, 500.0)),
        encoding="utf-8",
    )
    result = sweep.evaluate_candidate_result(
        run_id="short_only",
        long_weight=3.1,
        short_weight=1.5,
        artifact_dir=artifact_dir,
        rules=sweep.SweepRules(),
    )
    assert result.accepted is False
    assert any("long_count=" in r for r in result.rejection_reasons)


def test_reject_weak_side_count(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "weak_side"
    artifact_dir.mkdir()
    (artifact_dir / "summary.json").write_text(
        json.dumps(_summary(promotable=True, pub_long=29, pub_short=80, long_ev=10.0, short_ev=20.0)),
        encoding="utf-8",
    )
    result = sweep.evaluate_candidate_result(
        run_id="weak_side",
        long_weight=3.1,
        short_weight=1.5,
        artifact_dir=artifact_dir,
        rules=sweep.SweepRules(),
    )
    assert result.accepted is False
    assert any("long_count=29<30" in r for r in result.rejection_reasons)


def test_accept_balanced_positive_ev_candidate(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "good"
    artifact_dir.mkdir()
    (artifact_dir / "summary.json").write_text(
        json.dumps(_summary(promotable=True, pub_long=40, pub_short=60, long_ev=100.0, short_ev=200.0)),
        encoding="utf-8",
    )
    result = sweep.evaluate_candidate_result(
        run_id="good",
        long_weight=3.1,
        short_weight=1.5,
        artifact_dir=artifact_dir,
        rules=sweep.SweepRules(),
    )
    assert result.accepted is True
    assert result.long_share == pytest.approx(0.4)
    assert result.short_share == pytest.approx(0.6)
    assert result.rank_key is not None


def test_ranking_prefers_better_balance_then_ev(tmp_path: Path) -> None:
    rules = sweep.SweepRules()

    def _write(name: str, pub_long: int, pub_short: int, long_ev: float, short_ev: float) -> Path:
        d = tmp_path / name
        d.mkdir()
        (d / "summary.json").write_text(
            json.dumps(_summary(promotable=True, pub_long=pub_long, pub_short=pub_short, long_ev=long_ev, short_ev=short_ev)),
            encoding="utf-8",
        )
        return d

    balanced = sweep.evaluate_candidate_result(
        run_id="balanced",
        long_weight=3.1,
        short_weight=1.5,
        artifact_dir=_write("balanced", 40, 60, 100.0, 200.0),
        rules=rules,
    )
    high_ev_skewed = sweep.evaluate_candidate_result(
        run_id="skewed",
        long_weight=3.2,
        short_weight=1.6,
        artifact_dir=_write("skewed", 20, 80, 50.0, 900.0),
        rules=rules,
    )
    ranked = sweep.rank_candidates([high_ev_skewed, balanced])
    assert ranked[0].run_id == "balanced"


def test_writes_json_and_csv_summary(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "good"
    artifact_dir.mkdir()
    (artifact_dir / "summary.json").write_text(
        json.dumps(_summary(promotable=True, pub_long=40, pub_short=60, long_ev=100.0, short_ev=200.0)),
        encoding="utf-8",
    )
    candidate = sweep.evaluate_candidate_result(
        run_id="good",
        long_weight=3.1,
        short_weight=1.5,
        artifact_dir=artifact_dir,
        rules=sweep.SweepRules(),
    )
    sweep_id = "top200_sweep_test"
    json_path, csv_path = sweep.write_sweep_summary(
        sweep_id=sweep_id,
        candidates=[candidate],
        output_dir=tmp_path,
        rules=sweep.SweepRules(),
        grid_size=1,
        executed=1,
    )
    assert json_path.exists()
    assert csv_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["promotion_source_of_truth"] == "summary.json"
    assert payload["accepted_count"] == 1
    assert payload["best_candidate"]["run_id"] == "good"
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["accepted"] == "True"
    assert rows[0]["publishable_long_count_ge_70"] == "40"


def test_does_not_promote_automatically(capsys: pytest.CaptureFixture[str]) -> None:
    with patch.object(sys, "argv", ["sweep_top200_side_balance.py", "--dry-run"]):
        sweep.main()
    out = capsys.readouterr().out
    assert "pending_runs" in out
    promote_calls: list[str] = []

    def _fail_if_promote(*args, **kwargs):
        promote_calls.append("promote")
        raise AssertionError("auto-promote must not be called")

    with patch("subprocess.run", side_effect=_fail_if_promote):
        with patch.object(sweep, "generate_weight_grid", return_value=[]):
            with patch.object(sys, "argv", ["sweep_top200_side_balance.py"]):
                sweep.main()
    assert promote_calls == []


def test_resumable_skips_valid_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sweep_id = "top200_sweep_resume"
    done_dir = sweep.candidate_artifact_dir(sweep_id, 3.05, 1.45, candidates_dir=tmp_path)
    done_dir.mkdir(parents=True)
    (done_dir / "summary.json").write_text(
        json.dumps(_summary(promotable=True, pub_long=40, pub_short=60, long_ev=1.0, short_ev=1.0)),
        encoding="utf-8",
    )
    monkeypatch.setattr(sweep, "DEFAULT_CANDIDATES_DIR", tmp_path)
    assert sweep.should_skip_candidate(done_dir, retry_failed=False) is True
    assert sweep.should_skip_candidate(done_dir, retry_failed=True) is True

    failed_dir = sweep.candidate_artifact_dir(sweep_id, 3.10, 1.45, candidates_dir=tmp_path)
    failed_dir.mkdir(parents=True)
    (failed_dir / "state.json").write_text("{}", encoding="utf-8")
    assert sweep.should_skip_candidate(failed_dir, retry_failed=False) is True
    assert sweep.should_skip_candidate(failed_dir, retry_failed=True) is False


def test_parse_side_diagnostics_extracts_0_70_threshold(tmp_path: Path) -> None:
    path = tmp_path / "side_specialists_report.json"
    path.write_text(json.dumps(_side_report(10, 20, 30.0)), encoding="utf-8")
    diag = sweep.parse_side_diagnostics(path)
    assert diag["diagnostic_only"] is True
    assert diag["second_pass_threshold_0_70"]["publishable_LONG"] == 10
    assert diag["calibration_ceiling_summary"]["LONG"]["ceiling"] == 0.8
