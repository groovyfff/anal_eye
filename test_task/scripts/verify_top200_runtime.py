#!/usr/bin/env python3
"""Verify top-200 runtime configuration and production artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ae_brain.artifact_verify import verify_runtime_artifacts
from ae_brain.env_universe import count_csv_symbols, parse_env_file
from ae_brain.universe_top200 import LEGACY_SIX_SYMBOLS, load_universe_txt

DEFAULT_UNIVERSE_TXT = ROOT / "config" / "universe_top200_usdtm.txt"
BACKEND_ENV = REPO_ROOT / "backend" / "Annnneqwe" / "analeyes" / ".env"
TEST_TASK_ENV = ROOT / ".env"
PRODUCTION_ARTIFACTS = ROOT / "artifacts"

SYMBOL_ENV_KEYS = (
    "SYMBOLS",
    "BINANCE_SYMBOLS",
    "ANAL_EYES_ALLOWED_SYMBOLS",
    "AEB_ALLOWED_SYMBOLS",
)


def _parse_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(value: str | None, default: float) -> float:
    try:
        return float(str(value or "").strip())
    except ValueError:
        return default


def verify_env_symbols(path: Path, *, expected_count: int = 200) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"missing env file: {path}"]
    env = parse_env_file(path)
    for key in SYMBOL_ENV_KEYS:
        if key not in env:
            errors.append(f"{path.name}: missing {key}")
            continue
        count = count_csv_symbols(env[key])
        if count != expected_count:
            errors.append(f"{path.name}: {key} has {count} symbols, expected {expected_count}")
        for legacy in LEGACY_SIX_SYMBOLS:
            if legacy not in env[key]:
                errors.append(f"{path.name}: {key} missing legacy symbol {legacy}")
    if _parse_bool(env.get("AEB_ONLY_BTC")):
        errors.append(f"{path.name}: AEB_ONLY_BTC is enabled")
    if _parse_float(env.get("AEB_MIN_PUBLISH_CONFIDENCE"), 0.70) < 0.70:
        errors.append(f"{path.name}: AEB_MIN_PUBLISH_CONFIDENCE below 0.70")
    if _parse_float(env.get("NOTIFICATION_MIN_CONFIDENCE"), 0.70) < 0.70:
        errors.append(f"{path.name}: NOTIFICATION_MIN_CONFIDENCE below 0.70")
    limit = env.get("SYMBOL_LIMIT")
    if limit is not None and str(limit).strip() != str(expected_count):
        errors.append(f"{path.name}: SYMBOL_LIMIT={limit} expected {expected_count}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify top-200 runtime readiness")
    parser.add_argument("--universe-txt", type=Path, default=DEFAULT_UNIVERSE_TXT)
    parser.add_argument("--backend-env", type=Path, default=BACKEND_ENV)
    parser.add_argument("--test-task-env", type=Path, default=TEST_TASK_ENV)
    parser.add_argument("--artifacts-dir", type=Path, default=PRODUCTION_ARTIFACTS)
    parser.add_argument("--expected-count", type=int, default=200)
    args = parser.parse_args()

    errors: list[str] = []
    try:
        universe = load_universe_txt(args.universe_txt)
        if len(universe) != args.expected_count:
            errors.append(f"universe file has {len(universe)} symbols, expected {args.expected_count}")
    except FileNotFoundError:
        errors.append(f"missing universe file: {args.universe_txt}")

    try:
        verify_runtime_artifacts(args.artifacts_dir)
    except FileNotFoundError as exc:
        errors.append(str(exc))

    errors.extend(verify_env_symbols(args.backend_env, expected_count=args.expected_count))
    errors.extend(verify_env_symbols(args.test_task_env, expected_count=args.expected_count))

    if errors:
        print("VERIFY FAILED:", file=sys.stderr)
        for err in errors:
            print(f" - {err}", file=sys.stderr)
        sys.exit(1)

    print("Top-200 runtime verification passed.")
    print(f"Artifacts: {args.artifacts_dir}")
    print(f"Universe: {args.universe_txt}")


if __name__ == "__main__":
    main()
