"""SQLAlchemy 2.x async engine, sessionmaker, and lifecycle management."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.common.logger import get_logger
from src.module8_database.config import db_settings

logger = get_logger("Module8.Connection")


class DatabaseSessionManager:
    """Manages async database engine lifecycle and session production."""

    def __init__(self, database_url: Optional[str] = None, echo: bool = False):
        self._url = database_url or db_settings.async_database_url
        self._echo = echo or db_settings.echo_sql

        engine_kwargs = {"echo": self._echo, "future": True}
        # SQLite in-memory / file drivers do not use pool_size / max_overflow
        if "sqlite" not in self._url:
            engine_kwargs.update({
                "pool_size": db_settings.pool_size,
                "max_overflow": db_settings.max_overflow,
                "pool_timeout": db_settings.pool_timeout,
            })

        self._engine: AsyncEngine = create_async_engine(self._url, **engine_kwargs)
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        return self._session_factory

    async def close(self) -> None:
        """Closes all connections in the connection pool."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database engine connections closed.")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Provides an asynchronous database session context."""
        session: AsyncSession = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception as exc:
            await session.rollback()
            logger.error(f"Database session rolled back due to error: {exc}")
            raise
        finally:
            await session.close()


# Default singleton instance using application settings
db_manager = DatabaseSessionManager()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI-compatible async dependency generator for acquiring database sessions."""
    async with db_manager.session() as session:
        yield session
