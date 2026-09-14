"""Transaction list, detail, and filtering endpoints."""

from math import ceil
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.logger import get_logger
from src.module3_graph_builder import get_graph_tx_ids
from src.module6_api.dependencies import get_db, get_graph_service, get_inference_service
from src.module6_api.schemas.common import PaginatedResponse
from src.module6_api.schemas.transaction import TransactionDetailResponse, TransactionResponse
from src.module6_api.services.graph_service import GraphService
from src.module6_api.services.inference_service import InferenceService
from src.module8_database.models.transaction import Transaction
from src.module8_database.repositories.transaction_repo import TransactionRepository

logger = get_logger("Module6.Router.Transactions")
router = APIRouter(prefix="/transactions", tags=["Transactions"])
tx_repo = TransactionRepository()


@router.get(
    "",
    response_model=PaginatedResponse[TransactionResponse],
    summary="List transactions with filtering and pagination",
)
async def list_transactions(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=50, ge=1, le=200, description="Items per page"),
    timestep: Optional[int] = Query(default=None, ge=1, le=49, description="Filter by discrete timestep"),
    risk_level: Optional[str] = Query(default=None, description="Filter by risk tier ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')"),
    triage_status: Optional[str] = Query(default=None, description="Filter by status ('PENDING', 'QUEUED', 'REVIEWED', 'ESCALATED')"),
    min_priority: float = Query(default=0.0, ge=0.0, le=1.0, description="Minimum composite priority score"),
    search: Optional[str] = Query(default=None, description="Filter/search by transaction ID"),
    db: AsyncSession = Depends(get_db),
    graph_service: GraphService = Depends(get_graph_service),
    inference_service: InferenceService = Depends(get_inference_service),
) -> PaginatedResponse[TransactionResponse]:
    """
    Retrieves a paginated list of transactions filtered by timestep, risk level, search query, and review state.
    """
    offset = (page - 1) * page_size
    risk_filters = [risk_level] if risk_level else None

    # Check if database has records
    db_count = await tx_repo.count(db)
    if db_count > 0:
        total = await tx_repo.count_triage_queue(
            session=db,
            timestep=timestep,
            risk_levels=risk_filters,
            min_priority=min_priority,
            triage_status=triage_status,
            search=search,
        )
        db_items = await tx_repo.get_triage_queue(
            session=db,
            timestep=timestep,
            risk_levels=risk_filters,
            min_priority=min_priority,
            triage_status=triage_status,
            search=search,
            limit=page_size,
            offset=offset,
        )
        total_pages = ceil(total / page_size) if total > 0 else 0
        items = [TransactionResponse.model_validate(tx) for tx in db_items]
        return PaginatedResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    # Fallback to loading from graph artifacts if database is empty
    target_ts = timestep if timestep is not None else 40
    try:
        data = graph_service.loader.load_graph(target_ts)
        x = graph_service.loader.get_feature_matrix(data, config="original_all")
        probs = inference_service.score_timestep_nodes(x, data.edge_index) if inference_service.is_available else [0.0] * data.num_nodes
        y = data.y.numpy()
        node_tx_ids = get_graph_tx_ids(data)

        fallback_items = []
        for i in range(len(node_tx_ids)):
            tx_id_str = str(node_tx_ids[i])
            if search and search.strip().lower() not in tx_id_str.lower():
                continue

            p = float(probs[i])
            item = inference_service.triage_router.triage_candidate(
                tx_id=node_tx_ids[i],
                timestep=target_ts,
                global_node_index=i,
                predicted_prob=p,
            )
            if risk_level and item.risk_level != risk_level:
                continue
            if item.priority_score < min_priority:
                continue

            from datetime import datetime, timezone
            fallback_items.append(
                TransactionResponse(
                    tx_id=item.tx_id,
                    timestep=item.timestep,
                    ground_truth_label=int(y[i]),
                    predicted_prob=item.predicted_prob,
                    predicted_class=item.predicted_class,
                    risk_level=item.risk_level,
                    uncertainty_score=item.uncertainty_score,
                    entropy=item.entropy,
                    priority_score=item.priority_score,
                    is_triaged=False,
                    triage_status="QUEUED",
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
            )

        fallback_items.sort(key=lambda x: (x.priority_score or 0.0), reverse=True)
        total = len(fallback_items)
        paginated = fallback_items[offset : offset + page_size]
        total_pages = ceil(total / page_size) if total > 0 else 0

        return PaginatedResponse(
            items=paginated,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )
    except Exception as e:
        logger.error(f"Error reading transactions from graph artifacts: {e}")
        return PaginatedResponse(items=[], total=0, page=page, page_size=page_size, total_pages=0)


@router.get(
    "/{tx_id}",
    response_model=TransactionDetailResponse,
    summary="Get single transaction detail and audit history",
)
async def get_transaction_detail(
    tx_id: str,
    db: AsyncSession = Depends(get_db),
    graph_service: GraphService = Depends(get_graph_service),
    inference_service: InferenceService = Depends(get_inference_service),
) -> TransactionDetailResponse:
    """
    Retrieves full transaction metadata, risk score, and connected feedback records.
    """
    tx = await tx_repo.get_by_tx_id(db, tx_id, load_details=True)
    if tx:
        return TransactionDetailResponse(
            tx_id=tx.tx_id,
            timestep=tx.timestep,
            ground_truth_label=tx.ground_truth_label,
            predicted_prob=tx.predicted_prob,
            predicted_class=tx.predicted_class,
            risk_level=tx.risk_level,
            uncertainty_score=tx.uncertainty_score,
            entropy=tx.entropy,
            priority_score=tx.priority_score,
            is_triaged=tx.is_triaged,
            triage_status=tx.triage_status,
            created_at=tx.created_at,
            updated_at=tx.updated_at,
            explanation_count=len(tx.explanations),
            feedback_count=len(tx.feedbacks),
            outgoing_edge_count=len(tx.outgoing_edges),
            incoming_edge_count=len(tx.incoming_edges),
        )

    # Search in timestep graphs
    for ts in range(1, 50):
        try:
            data = graph_service.loader.load_graph(ts)
            tx_ids = get_graph_tx_ids(data)
            if tx_id in tx_ids:
                loc_idx = tx_ids.index(tx_id)
                x = graph_service.loader.get_feature_matrix(data, config="original_all")
                triaged = inference_service.score_transaction(x, data.edge_index, loc_idx, tx_id, ts)
                y_val = int(data.y[loc_idx].item()) if hasattr(data, "y") else -1

                from datetime import datetime, timezone
                return TransactionDetailResponse(
                    tx_id=tx_id,
                    timestep=ts,
                    ground_truth_label=y_val,
                    predicted_prob=triaged.predicted_prob,
                    predicted_class=triaged.predicted_class,
                    risk_level=triaged.risk_level,
                    uncertainty_score=triaged.uncertainty_score,
                    entropy=triaged.entropy,
                    priority_score=triaged.priority_score,
                    is_triaged=False,
                    triage_status="QUEUED",
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                    explanation_count=1 if (default_config.paths.processed_data_dir / "experiments" / "exp05_explanations" / f"tx_{tx_id}.json").exists() else 0,
                    feedback_count=0,
                    outgoing_edge_count=0,
                    incoming_edge_count=0,
                )
        except Exception:
            continue

    raise HTTPException(status_code=404, detail=f"Transaction '{tx_id}' not found.")
