"""Environment file helpers for universe symbol propagation."""

from __future__ import annotations

import re
from pathlib import Path

_ENV_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _ENV_LINE_RE.match(stripped)
        if match:
            values[match.group(1)] = match.group(2)
    return values


def update_env_file(path: Path, updates: dict[str, str], *, create: bool = True) -> None:
    """Update or append keys in a .env file without removing unrelated lines/secrets."""
    lines: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
    else:
        if not create:
            raise FileNotFoundError(path)

    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out.append(line)
            continue
        match = _ENV_LINE_RE.match(stripped)
        if not match:
            out.append(line)
            continue
        key = match.group(1)
        if key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(line)

    for key, value in updates.items():
        if key not in seen:
            out.append(f"{key}={value}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")


def ensure_env_example_keys(path: Path, example_updates: dict[str, str]) -> None:
    """Add missing safe example keys to .env.example without overwriting existing values."""
    if not path.exists():
        return
    existing = parse_env_file(path)
    missing = {k: v for k, v in example_updates.items() if k not in existing}
    if missing:
        update_env_file(path, missing, create=True)


def count_csv_symbols(value: str) -> int:
    return len([part for part in str(value or "").split(",") if part.strip()])
