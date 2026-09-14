"""Explainable AI (XAI) attribution routes using GNNExplainer."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from src.common.logger import get_logger
from src.module3_graph_builder import get_graph_tx_ids
from src.module6_api.dependencies import get_graph_service, get_xai_service
from src.module6_api.schemas.explanation import ExplanationResponse
from src.module6_api.services.graph_service import GraphService
from src.module6_api.services.xai_service import XAIService

logger = get_logger("Module6.Router.Explain")
router = APIRouter(prefix="/explain", tags=["Explainability Engine"])


@router.get(
    "/{tx_id}",
    response_model=ExplanationResponse,
    summary="Get GNNExplainer attribution and fidelity metrics for transaction",
)
async def get_transaction_explanation(
    tx_id: str,
    timestep: Optional[int] = Query(default=None, ge=1, le=49, description="Discrete timestep"),
    xai_service: XAIService = Depends(get_xai_service),
    graph_service: GraphService = Depends(get_graph_service),
) -> ExplanationResponse:
    """
    Returns feature importance rankings, edge attributions, and quantitative faithfulness metrics.
    """
    target_ts = timestep
    if target_ts is None:
        for ts in range(1, 50):
            try:
                g = graph_service.loader.load_graph(ts)
                tx_ids = get_graph_tx_ids(g)
                if tx_id in tx_ids:
                    target_ts = ts
                    break
            except Exception:
                continue

    if target_ts is None:
        raise HTTPException(status_code=404, detail=f"Transaction '{tx_id}' not found in any timestep graph.")

    try:
        explanation = xai_service.get_explanation(tx_id=tx_id, timestep=target_ts)
        return explanation
    except Exception as e:
        logger.error(f"Error generating explanation for {tx_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")
