"""FastAPI application configuration and environment settings."""

from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    """Configuration for the FastAPI async service."""
    app_title: str = "HITL-XGNN Financial Fraud Detection API"
    app_description: str = (
        "High-performance async API for Graph Neural Network Bitcoin transaction fraud detection, "
        "GNNExplainer post-hoc attributions, and Human-in-the-Loop triage workflows."
    )
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    host: str = Field(default="0.0.0.0", alias="API_HOST")
    port: int = Field(default=8000, alias="API_PORT")
    debug: bool = Field(default=False, alias="API_DEBUG")

    # CORS configuration
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173", "*"],
        alias="CORS_ALLOWED_ORIGINS",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


api_settings = ApiSettings()
