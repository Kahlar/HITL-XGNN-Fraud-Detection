from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.logger import get_logger
from src.module6_api.dependencies import (
    get_analytics_service,
    get_db,
    get_graph_service,
    get_inference_service,
)
from src.module6_api.schemas.analytics import (
    AnalyticsMetricsResponse,
    GroupEvaluationRequest,
    GroupEvaluationResponse,
)
from src.module6_api.services.analytics_service import AnalyticsService
from src.module6_api.services.graph_service import GraphService
from src.module6_api.services.inference_service import InferenceService

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


@router.post(
    "/evaluation-metrics",
    response_model=GroupEvaluationResponse,
    summary="Compute group-level F1, PR-AUC, and Confusion Matrix across a defined cohort",
)
async def evaluate_group_metrics_post(
    request: GroupEvaluationRequest,
    db: AsyncSession = Depends(get_db),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    graph_service: GraphService = Depends(get_graph_service),
    inference_service: InferenceService = Depends(get_inference_service),
) -> GroupEvaluationResponse:
    """
    Computes minority Illicit F1, PR-AUC, Precision, Recall, and Confusion Matrix for a specified cohort:
    scope can be 'timestep', 'filtered', or 'reviewed'. Masking unlabelled transactions (y == -1).
    """
    return await analytics_service.compute_group_evaluation(
        session=db,
        scope=request.scope,
        timestep=request.timestep,
        tx_ids=request.tx_ids,
        graph_service=graph_service,
        inference_service=inference_service,
    )


@router.get(
    "/evaluation-metrics",
    response_model=GroupEvaluationResponse,
    summary="Compute group-level F1, PR-AUC, and Confusion Matrix via query parameters",
)
async def evaluate_group_metrics_get(
    scope: str = Query(default="timestep", description="Scope: 'timestep', 'filtered', or 'reviewed'"),
    timestep: Optional[int] = Query(default=None, ge=1, le=49, description="Target timestep"),
    tx_ids: Optional[str] = Query(default=None, description="Comma-separated transaction IDs"),
    db: AsyncSession = Depends(get_db),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    graph_service: GraphService = Depends(get_graph_service),
    inference_service: InferenceService = Depends(get_inference_service),
) -> GroupEvaluationResponse:
    """
    GET version of group-level metric evaluation.
    """
    id_list = [t.strip() for t in tx_ids.split(",") if t.strip()] if tx_ids else None
    return await analytics_service.compute_group_evaluation(
        session=db,
        scope=scope,
        timestep=timestep,
        tx_ids=id_list,
        graph_service=graph_service,
        inference_service=inference_service,
    )
