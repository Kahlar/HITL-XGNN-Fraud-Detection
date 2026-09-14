"""Analytics and dashboard telemetry routes."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.logger import get_logger
from src.module6_api.dependencies import get_analytics_service, get_db
from src.module6_api.schemas.analytics import AnalyticsMetricsResponse
from src.module6_api.services.analytics_service import AnalyticsService

logger = get_logger("Module6.Router.Analytics")
router = APIRouter(prefix="/analytics", tags=["Analytics & Telemetry"])


@router.get(
    "/metrics",
    response_model=AnalyticsMetricsResponse,
    summary="Get aggregated dataset statistics, model metrics, temporal drift, and feedback analytics",
)
async def get_dashboard_metrics(
    db: AsyncSession = Depends(get_db),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> AnalyticsMetricsResponse:
    """
    Consolidates dataset summaries, active model benchmark results, darknet shock temporal metrics,
    and reviewer feedback counts for interactive frontend dashboards.
    """
    return await analytics_service.get_dashboard_metrics(db)
