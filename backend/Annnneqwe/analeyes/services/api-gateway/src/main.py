from __future__ import annotations

import os
from typing import Any

import uuid

from fastapi import FastAPI, HTTPException, Query
from shared.config import Config
from shared.database.db_manager import DatabaseManager
from shared.database.models import SignalFeatureLog
from sqlalchemy import func, select

CONFIG_PATH = os.environ.get('ANALEYES_CONFIG', '/app/config/settings.yml')
app = FastAPI(title='AnalEyes API Gateway')
_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config(CONFIG_PATH)
    return _config


@app.on_event('startup')
async def startup() -> None:
    await DatabaseManager.initialize(database_url=os.environ.get('DATABASE_URL'))


@app.on_event('shutdown')
async def shutdown() -> None:
    await DatabaseManager.close()


@app.get('/api/health')
async def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/api/signals')
async def list_signals(
    asset_class: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> list[dict[str, Any]]:
    async for session in DatabaseManager.get_session():
        stmt = select(SignalFeatureLog).order_by(SignalFeatureLog.created_at.desc()).limit(limit)
        if asset_class:
            stmt = stmt.where(SignalFeatureLog.asset_class == asset_class)
        rows = (await session.execute(stmt)).scalars().all()
        return [
            {
                'id': row.id,
                'symbol': row.symbol,
                'asset_class': row.asset_class,
                'signal_id': str(row.signal_id_uuid),
                'composite_score': row.composite_score_value,
                'tracker_status': row.tracker_status,
                'tracker_pnl_percent': row.tracker_pnl_percent,
                'created_at': row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]
    return []


@app.get('/api/stats/pnl')
async def stats_pnl(asset_class: str | None = Query(default=None)) -> dict[str, Any]:
    async for session in DatabaseManager.get_session():
        stmt = select(
            SignalFeatureLog.asset_class,
            func.count(SignalFeatureLog.id),
            func.coalesce(func.sum(SignalFeatureLog.tracker_pnl_percent), 0.0),
        ).where(SignalFeatureLog.tracker_pnl_percent.is_not(None))
        if asset_class:
            stmt = stmt.where(SignalFeatureLog.asset_class == asset_class)
        stmt = stmt.group_by(SignalFeatureLog.asset_class)
        rows = (await session.execute(stmt)).all()
        return {
            'by_asset_class': [
                {'asset_class': row[0], 'count': row[1], 'total_pnl_percent': float(row[2] or 0.0)}
                for row in rows
            ]
        }
    return {'by_asset_class': []}


@app.get('/api/signals/{signal_id}/chart-data')
async def chart_data(signal_id: str) -> dict[str, Any]:
    # Исправление: явный Config с путём, без Depends(Config) который падал FileNotFoundError
    _ = get_config()
    try:
        signal_uuid = uuid.UUID(signal_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail='Invalid signal_id UUID') from exc
    async for session in DatabaseManager.get_session():
        stmt = select(SignalFeatureLog).where(SignalFeatureLog.signal_id_uuid == signal_uuid)
        row = (await session.execute(stmt)).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail='Signal not found')
        return {
            'symbol': row.symbol,
            'asset_class': row.asset_class,
            'ohlcv': row.historical_ohlcv_json or [],
            'snapshots': row.historical_snapshots_json or [],
        }
    raise HTTPException(status_code=404, detail='Signal not found')
