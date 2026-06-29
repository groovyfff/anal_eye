from __future__ import annotations

import logging
import os
import sys
from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from shared.database.models import OrmBase

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Async SQLAlchemy менеджер (синглтон на процесс)."""

    _engine: AsyncEngine | None = None
    _async_session_maker: async_sessionmaker[AsyncSession] | None = None

    @classmethod
    async def initialize(cls, database_url: str | None = None, pool_size: int = 5, max_overflow: int = 10) -> None:
        url = database_url or os.environ.get('DATABASE_URL')
        if not url:
            logger.error('DATABASE_URL не задан — запись в БД недоступна')
            raise RuntimeError('DATABASE_URL is required')
        cls._engine = create_async_engine(
            url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_recycle=3600,
            pool_pre_ping=True,
        )
        async with cls._engine.begin() as conn:
            await conn.run_sync(OrmBase.metadata.create_all)
        cls._async_session_maker = async_sessionmaker(cls._engine, expire_on_commit=False)
        logger.info('DatabaseManager инициализирован')

    @classmethod
    async def get_session(cls) -> AsyncGenerator[AsyncSession, None]:
        if cls._async_session_maker is None:
            raise RuntimeError('DatabaseManager.initialize() must be called first')
        session = cls._async_session_maker()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    @classmethod
    async def close(cls) -> None:
        if cls._engine is not None:
            await cls._engine.dispose()
            cls._engine = None
            cls._async_session_maker = None

    @classmethod
    def get_engine(cls) -> AsyncEngine | None:
        return cls._engine
