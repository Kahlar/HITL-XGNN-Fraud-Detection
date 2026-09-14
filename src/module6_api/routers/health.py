"""Health and readiness probe endpoint."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.logger import get_logger
from src.module6_api.dependencies import get_db, get_inference_service
from src.module6_api.schemas.common import HealthResponse
from src.module6_api.services.inference_service import InferenceService

logger = get_logger("Module6.Router.Health")
router = APIRouter(tags=["System Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health and model readiness probe",
)
async def health_check(
    db: AsyncSession = Depends(get_db),
    inference_service: InferenceService = Depends(get_inference_service),
) -> HealthResponse:
    """
    Returns API status, database connectivity, and active GNN model availability.
    """
    # Check DB connectivity
    db_ok = False
    try:
        result = await db.execute(text("SELECT 1"))
        db_ok = (result.scalar() == 1)
    except Exception as e:
        logger.warning(f"Health check database ping failed: {e}")
        db_ok = False

    model_available = inference_service.is_available
    active_version = inference_service.model_version if model_available else None
    architecture = "GraphSAGE" if model_available else None

    return HealthResponse(
        status="healthy" if (db_ok or model_available) else "degraded",
        database_connected=db_ok,
        model_available=model_available,
        active_model_version=active_version,
        architecture=architecture,
    )
