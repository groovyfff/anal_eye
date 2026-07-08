#!/usr/bin/env python3
"""Apply top-200 universe symbols to runtime .env files without touching secrets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ae_brain.env_universe import count_csv_symbols, ensure_env_example_keys, update_env_file
from ae_brain.universe_top200 import LEGACY_SIX_SYMBOLS, load_universe_txt, symbols_csv

DEFAULT_UNIVERSE_TXT = ROOT / "config" / "universe_top200_usdtm.txt"
BACKEND_ENV = REPO_ROOT / "backend" / "Annnneqwe" / "analeyes" / ".env"
TEST_TASK_ENV = ROOT / ".env"
BACKEND_ENV_EXAMPLE = REPO_ROOT / "backend" / "Annnneqwe" / "analeyes" / ".env.example"
TEST_TASK_ENV_EXAMPLE = ROOT / ".env.example"

SYMBOL_KEYS = (
    "SYMBOLS",
    "BINANCE_SYMBOLS",
    "ANAL_EYES_ALLOWED_SYMBOLS",
    "AEB_ALLOWED_SYMBOLS",
)
THRESHOLD_KEYS = (
    ("SYMBOL_LIMIT", "200"),
    ("AEB_ONLY_BTC", "false"),
    ("AEB_MIN_PUBLISH_CONFIDENCE", "0.70"),
    ("NOTIFICATION_MIN_CONFIDENCE", "0.70"),
)


def build_updates(symbols: list[str]) -> dict[str, str]:
    csv = symbols_csv(symbols)
    updates = {key: csv for key in SYMBOL_KEYS}
    for key, value in THRESHOLD_KEYS:
        updates[key] = value
    return updates


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply top-200 universe to AnalEyes env files")
    parser.add_argument("--universe-txt", type=Path, default=DEFAULT_UNIVERSE_TXT)
    parser.add_argument("--backend-env", type=Path, default=BACKEND_ENV)
    parser.add_argument("--test-task-env", type=Path, default=TEST_TASK_ENV)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    symbols = load_universe_txt(args.universe_txt)
    if len(symbols) != 200:
        print(f"Expected 200 symbols in {args.universe_txt}, got {len(symbols)}", file=sys.stderr)
        sys.exit(1)

    missing_legacy = [s for s in LEGACY_SIX_SYMBOLS if s not in symbols]
    if missing_legacy:
        print(f"Legacy six symbols missing from universe: {missing_legacy}", file=sys.stderr)
        sys.exit(1)

    updates = build_updates(symbols)
    if args.dry_run:
        print("DRY RUN updates:")
        for key, value in updates.items():
            preview = value if key in SYMBOL_KEYS else value
            if key in SYMBOL_KEYS:
                print(f"{key}=<{count_csv_symbols(preview)} symbols>")
            else:
                print(f"{key}={preview}")
        return

    if args.backend_env.exists():
        update_env_file(args.backend_env, updates)
        print(f"Updated {args.backend_env}")
    else:
        print(f"Skip missing backend env: {args.backend_env}")

    if args.test_task_env.exists():
        update_env_file(args.test_task_env, updates)
        print(f"Updated {args.test_task_env}")
    else:
        print(f"Skip missing test_task env: {args.test_task_env}")

    example_updates = dict(updates)
    ensure_env_example_keys(BACKEND_ENV_EXAMPLE, example_updates)
    ensure_env_example_keys(TEST_TASK_ENV_EXAMPLE, example_updates)
    print("Ensured example env keys exist (without overwriting existing values).")


if __name__ == "__main__":
    main()
