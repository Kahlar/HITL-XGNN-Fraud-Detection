"""Graph visualization routes for 2-hop local subgraphs."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from src.common.logger import get_logger
from src.module3_graph_builder import get_graph_tx_ids
from src.module6_api.dependencies import get_graph_service, get_inference_service
from src.module6_api.schemas.graph import SubgraphResponse
from src.module6_api.services.graph_service import GraphService
from src.module6_api.services.inference_service import InferenceService

logger = get_logger("Module6.Router.Graph")
router = APIRouter(prefix="/graph", tags=["Graph Visualizer"])


@router.get(
    "/{tx_id}/subgraph",
    response_model=SubgraphResponse,
    summary="Get 2-hop induced computational subgraph for interactive visualization",
)
async def get_transaction_subgraph(
    tx_id: str,
    hops: int = Query(default=2, ge=1, le=3, description="Number of computational graph hops"),
    timestep: Optional[int] = Query(default=None, ge=1, le=49, description="Discrete timestep"),
    graph_service: GraphService = Depends(get_graph_service),
    inference_service: InferenceService = Depends(get_inference_service),
) -> SubgraphResponse:
    """
    Extracts the local directed transaction payment subgraph for interactive rendering.
    """
    target_ts = timestep
    target_graph = None
    target_tx_ids = None

    if target_ts is not None:
        try:
            target_graph = graph_service.loader.load_graph(target_ts)
            target_tx_ids = get_graph_tx_ids(target_graph)
            if tx_id not in target_tx_ids:
                target_ts = None
                target_graph = None
                target_tx_ids = None
        except Exception:
            target_ts = None
            target_graph = None
            target_tx_ids = None

    if target_ts is None:
        # Resolve timestep for tx_id across all timesteps
        for ts in range(1, 50):
            try:
                g = graph_service.loader.load_graph(ts)
                tx_ids = get_graph_tx_ids(g)
                if tx_id in tx_ids:
                    target_ts = ts
                    target_graph = g
                    target_tx_ids = tx_ids
                    break
            except Exception:
                continue

    if target_ts is None or target_graph is None or target_tx_ids is None:
        raise HTTPException(status_code=404, detail=f"Transaction '{tx_id}' not found in any timestep graph.")

    # Compute predictions for nodes in the target timestep graph using active InferenceService
    predictions_map = None
    if inference_service.is_available:
        try:
            x = graph_service.loader.get_feature_matrix(target_graph, config="original_all")
            probs = inference_service.score_timestep_nodes(x, target_graph.edge_index)
            predictions_map = {tid: float(probs[i]) for i, tid in enumerate(target_tx_ids)}
        except Exception as inf_err:
            logger.warning(f"Could not compute batch predictions for timestep {target_ts}: {inf_err}")

    try:
        subgraph = graph_service.get_subgraph(
            tx_id=tx_id,
            timestep=target_ts,
            k_hops=hops,
            predictions_map=predictions_map,
        )
        return subgraph
    except Exception as e:
        logger.error(f"Failed to extract subgraph for {tx_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract subgraph: {str(e)}")

