"""A.E. Brain command-line interface.

Examples
--------
    ae-brain gen-data --rows 20000 --out data/candles.parquet
    ae-brain init-db
    ae-brain train all --data data/candles.parquet
    ae-brain evaluate --candidate examples/candidate.json
    ae-brain run                 # live RabbitMQ consume/publish loop
    ae-brain serve-api --port 8080
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer

from ae_brain.config import get_settings
from ae_brain.utils.logging import configure_logging, get_logger

app = typer.Typer(add_completion=False, help="A.E. Brain trading ensemble CLI")
train_app = typer.Typer(help="Train ensemble layers")
app.add_typer(train_app, name="train")

log = get_logger("ae_brain.cli")


def _settings():
    s = get_settings()
    configure_logging(s.log_level, s.log_json)
    return s


def _load_candles(path: Path):
    import pandas as pd

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".csv", ".txt"):
        return pd.read_csv(path)
    raise typer.BadParameter(f"unsupported data format: {path.suffix}")


# --------------------------------------------------------------------------- #
@app.command("gen-data")
def gen_data(
    rows: int = typer.Option(20_000, help="Number of synthetic candles"),
    out: Path = typer.Option(Path("data/candles.parquet"), help="Output path"),
    seed: int = typer.Option(7),
) -> None:
    """Generate synthetic candles for smoke-testing the pipeline."""
    from ae_brain.training.synthetic import generate_synthetic_candles

    _settings()
    df = generate_synthetic_candles(n=rows, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix in (".csv", ".txt"):
        df.to_csv(out, index=False)
    else:
        df.to_parquet(out)
    typer.echo(f"wrote {len(df)} candles -> {out}")


@app.command("gen-candidate")
def gen_candidate(
    out: Path = typer.Option(Path("examples/candidate.json"), help="Output JSON path"),
    symbol: str = typer.Option(..., help="Trading pair symbol (e.g. ETHUSDT, SOLUSDT)"),
    window: int = typer.Option(64, help="Number of candles to embed"),
    seed: int = typer.Option(11),
    asset_class: str = typer.Option(
        "crypto",
        "--asset-class",
        "-c",
        help="crypto | stock | metal | forex (non-crypto nulls derivatives fields)",
    ),
    signal_log_db_id: int = typer.Option(
        0,
        "--signal-log-db-id",
        "-id",
        help="Pre-inserted backend row id (0 => INSERT fallback for local/dev)",
    ),
) -> None:
    """Emit an example ``data.candidates.ai`` message for `evaluate`/tests.

    For non-crypto asset classes the derivatives-only microstructure fields
    (funding_rate, open_interest, taker_buy_volume, liquidations, basis) are set
    to ``null`` to mirror how a real backend publishes traditional assets.
    """
    from ae_brain.contracts import AssetClass
    from ae_brain.training.synthetic import generate_synthetic_candles

    _settings()
    try:
        asset_class = AssetClass(asset_class.lower()).value
    except ValueError:
        raise typer.BadParameter(
            f"asset_class must be one of {[a.value for a in AssetClass]}"
        )

    df = generate_synthetic_candles(n=window + 200, seed=seed).tail(window).copy()
    df["ts"] = df["ts"].astype(str)

    # Traditional assets do not carry perpetual-derivatives microstructure;
    # null those columns so the example exercises the null-handling path.
    if asset_class != AssetClass.CRYPTO.value:
        for col in (
            "funding_rate", "open_interest", "taker_buy_volume",
            "long_liq_notional", "short_liq_notional", "basis",
        ):
            if col in df:
                df[col] = None

    payload = {
        "symbol": symbol,
        "interval": "5m",
        # 0 => local/dev: engine INSERTs a new row. A real backend supplies the
        # id of the row it already inserted, and the engine UPDATEs that row.
        "signal_log_db_id": signal_log_db_id,
        "asset_class": asset_class,
        "correlation_id": f"example-{seed}",
        "meta": {"adv_usd": 5_000_000.0, "expected_holding_hours": 4.0},
        "candles": df.to_dict(orient="records"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    typer.echo(
        f"wrote example candidate ({len(df)} candles, asset_class={asset_class}) -> {out}"
    )


@app.command("init-db")
def init_db() -> None:
    """Apply the PostgreSQL schema (idempotent)."""
    from ae_brain.data.database import Database

    s = _settings()

    async def _run() -> None:
        db = Database(s.database)
        await db.connect()
        await db.apply_schema()
        await db.close()

    asyncio.run(_run())
    typer.echo("schema applied")


@train_app.command("tabular")
def train_tabular_cmd(data: Path = typer.Option(..., exists=True)) -> None:
    from ae_brain.training.trainers import train_tabular

    s = _settings()
    metrics = train_tabular(_load_candles(data), s)
    typer.echo(json.dumps(metrics, indent=2))


@train_app.command("sequence")
def train_sequence_cmd(
    data: Path = typer.Option(..., exists=True),
    epochs: int = typer.Option(5),
) -> None:
    from ae_brain.training.trainers import train_sequence

    s = _settings()
    metrics = train_sequence(_load_candles(data), s, epochs=epochs)
    typer.echo(json.dumps(metrics, indent=2))


@train_app.command("rl")
def train_rl_cmd(
    data: Path = typer.Option(..., exists=True),
    timesteps: int = typer.Option(50_000),
) -> None:
    from ae_brain.training.trainers import train_rl

    s = _settings()
    metrics = train_rl(_load_candles(data), s, total_timesteps=timesteps)
    typer.echo(json.dumps(metrics, indent=2))


@train_app.command("all")
def train_all_cmd(
    data: Path = typer.Option(..., exists=True),
    epochs: int = typer.Option(5),
    timesteps: int = typer.Option(50_000),
) -> None:
    """Train tabular + sequence + RL in one shot."""
    from ae_brain.training.trainers import train_rl, train_sequence, train_tabular

    s = _settings()
    candles = _load_candles(data)
    out = {"tabular": train_tabular(candles, s)}
    try:
        out["sequence"] = train_sequence(candles, s, epochs=epochs)
    except Exception as exc:  # pragma: no cover
        out["sequence"] = {"error": str(exc)}
    try:
        out["rl"] = train_rl(candles, s, total_timesteps=timesteps)
    except Exception as exc:  # pragma: no cover
        out["rl"] = {"error": str(exc)}
    typer.echo(json.dumps(out, indent=2, default=str))


@app.command("evaluate")
def evaluate_cmd(
    candidate: Path = typer.Option(..., exists=True, help="JSON candidate file"),
    use_db: bool = typer.Option(False, help="Connect to PostgreSQL for logging"),
) -> None:
    """Evaluate a single candidate JSON and print the final signal."""
    from ae_brain.contracts import TradeCandidate
    from ae_brain.runtime import LiveRuntime

    s = _settings()
    payload = json.loads(candidate.read_text())
    cand = TradeCandidate.from_message(payload)

    async def _run():
        return await LiveRuntime(s).evaluate_once(cand, use_db=use_db)

    signal = asyncio.run(_run())
    typer.echo(json.dumps(signal.to_dict(), indent=2, default=str))


@app.command("run")
def run_cmd() -> None:
    """Start the live RabbitMQ consume/publish loop."""
    from ae_brain.runtime import run_live

    s = _settings()
    try:
        asyncio.run(run_live(s))
    except KeyboardInterrupt:  # pragma: no cover
        typer.echo("interrupted")


@app.command("serve-api")
def serve_api(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run the optional FastAPI debugging surface."""
    import uvicorn

    _settings()
    uvicorn.run("ae_brain.api:create_app", host=host, port=port, factory=True)


@app.command("features")
def features_cmd() -> None:
    """Print the canonical feature schema."""
    from ae_brain.features.schema import FEATURE_SCHEMA, n_features

    _settings()
    for i, spec in enumerate(FEATURE_SCHEMA):
        typer.echo(f"{i:>2} [{spec.group:<11}] {spec.name:<22} {spec.description}")
    typer.echo(f"\ntotal features: {n_features()}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
