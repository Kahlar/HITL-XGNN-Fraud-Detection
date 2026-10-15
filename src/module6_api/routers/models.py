"""Model registry, threshold transparency, and verification endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.logger import get_logger
from src.module6_api.dependencies import get_analytics_service, get_db
from src.module6_api.schemas.analytics import ModelValidationResponse, ThresholdPolicyResponse
from src.module6_api.services.analytics_service import AnalyticsService

logger = get_logger("Module6.Router.Models")
router = APIRouter(prefix="/models", tags=["Model Telemetry & Validation"])


@router.get(
    "/threshold",
    response_model=ThresholdPolicyResponse,
    summary="Get threshold selection methodology and decision policy",
)
async def get_threshold_policy(
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> ThresholdPolicyResponse:
    """
    Exposes the validation minority-F1 calibration methodology (tau* = 0.5517 on val t in [35, 39]),
    grid search specifications, and baseline comparison thresholds (e.g. XGBoost tau* = 0.9014).
    """
    return analytics_service.get_threshold_policy()


@router.get(
    "/validation",
    response_model=ModelValidationResponse,
    summary="Get model validation protocol, benchmark comparison, and software verification state",
)
async def get_model_validation_methodology(
    db: AsyncSession = Depends(get_db),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> ModelValidationResponse:
    """
    Exposes chronological split boundaries (train 1..34, val 35..39, test 40..49),
    the full 7-model benchmark comparison from EXP-01/EXP-02, and software verification evidence.
    """
    return analytics_service.get_model_validation_summary(db)
