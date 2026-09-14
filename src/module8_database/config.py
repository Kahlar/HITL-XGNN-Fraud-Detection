"""Database settings and configuration management using Pydantic Settings."""

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL and SQLAlchemy async database configuration."""
    postgres_user: str = Field(default="postgres", alias="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", alias="POSTGRES_PASSWORD")
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="hitl_fraud_detection", alias="POSTGRES_DB")

    # Optional direct database URL override (e.g. for testing with sqlite+aiosqlite)
    database_url_override: Optional[str] = Field(default=None, alias="DATABASE_URL")

    # Connection pooling parameters
    pool_size: int = Field(default=10, alias="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, alias="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, alias="DB_POOL_TIMEOUT")
    echo_sql: bool = Field(default=False, alias="DB_ECHO_SQL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def async_database_url(self) -> str:
        """Returns the async database connection URI."""
        if self.database_url_override:
            return self.database_url_override
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        """Returns the sync database connection URI for Alembic migrations."""
        if self.database_url_override:
            # Replace asyncpg / aiosqlite with standard drivers if needed
            return self.database_url_override.replace("+asyncpg", "").replace("+aiosqlite", "")
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


db_settings = DatabaseSettings()
