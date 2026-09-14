"""FastAPI dependency injection providers for database sessions and services."""

from typing import AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.module6_api.services.analytics_service import AnalyticsService
from src.module6_api.services.graph_service import GraphService
from src.module6_api.services.inference_service import InferenceService
from src.module6_api.services.xai_service import XAIService
from src.module8_database.connection import db_manager

# Singleton service instances
_inference_service = None
_graph_service = None
_analytics_service = None
_xai_service = None


def get_inference_service() -> InferenceService:
    """Provides singleton instance of InferenceService."""
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service


def get_graph_service() -> GraphService:
    """Provides singleton instance of GraphService."""
    global _graph_service
    if _graph_service is None:
        _graph_service = GraphService()
    return _graph_service


def get_xai_service(
    inference_service: InferenceService = Depends(get_inference_service),
) -> XAIService:
    """Provides instance of XAIService wired to the active inference model."""
    global _xai_service
    if _xai_service is None:
        _xai_service = XAIService(inference_service=inference_service)
    return _xai_service


def get_analytics_service() -> AnalyticsService:
    """Provides singleton instance of AnalyticsService."""
    global _analytics_service
    if _analytics_service is None:
        _analytics_service = AnalyticsService()
    return _analytics_service


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provides asynchronous SQLAlchemy session for route handlers."""
    async with db_manager.session() as session:
        yield session
