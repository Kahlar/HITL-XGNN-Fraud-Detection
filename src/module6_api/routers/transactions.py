"""Transaction list, detail, and filtering endpoints."""

from datetime import datetime, timezone
from math import ceil
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.config import default_config
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
    Scans the graph transaction dataset and overlays database reviewer audit records.
    """
    offset = (page - 1) * page_size
    now_utc = datetime.now(timezone.utc)

    try:
        # Load all DB transaction records to overlay triage status and reviewer metadata
        db_tx_map: Dict[str, Transaction] = {}
        try:
            db_transactions = (await db.execute(select(Transaction))).scalars().all()
            db_tx_map = {tx.tx_id: tx for tx in db_transactions}
        except Exception as db_err:
            logger.warning(f"Could not load DB transactions for overlay: {db_err}")

        matched_items: List[TransactionResponse] = []
        seen_tx_ids: set = set()

        # 1. Include DB transactions that match the filters
        for tx_id_key, db_tx in db_tx_map.items():
            if timestep is not None and db_tx.timestep != timestep:
                continue
            if search and search.strip() and search.strip().lower() not in tx_id_key.lower():
                continue
            if risk_level and db_tx.risk_level != risk_level:
                continue
            if triage_status and db_tx.triage_status != triage_status:
                continue
            if min_priority > 0.0 and (db_tx.priority_score or 0.0) < min_priority:
                continue

            matched_items.append(
                TransactionResponse(
                    tx_id=db_tx.tx_id,
                    timestep=db_tx.timestep or 40,
                    ground_truth_label=db_tx.ground_truth_label,
                    predicted_prob=db_tx.predicted_prob,
                    predicted_class=db_tx.predicted_class,
                    risk_level=db_tx.risk_level,
                    uncertainty_score=db_tx.uncertainty_score,
                    entropy=db_tx.entropy,
                    priority_score=db_tx.priority_score,
                    is_triaged=db_tx.is_triaged,
                    triage_status=db_tx.triage_status or "QUEUED",
                    created_at=db_tx.created_at or now_utc,
                    updated_at=db_tx.updated_at or now_utc,
                )
            )
            seen_tx_ids.add(tx_id_key)

        # 2. Determine target graph timesteps to scan
        if timestep is not None:
            target_timesteps = [timestep]
        elif search and search.strip():
            # When searching across all timesteps, check primary test timestep 40 first, then remaining timesteps
            target_timesteps = [40] + [ts for ts in range(1, 50) if ts != 40]
        elif triage_status in ("REVIEWED", "ESCALATED"):
            # Only scan timesteps containing DB reviewed transactions if filtering strictly for reviewed status
            reviewed_timesteps = [tx.timestep for tx in db_tx_map.values() if tx.timestep]
            target_timesteps = sorted(list(set(reviewed_timesteps))) if reviewed_timesteps else []
        else:
            target_timesteps = [40]

        for ts in target_timesteps:
            try:
                data = graph_service.loader.load_graph(ts)
                node_tx_ids = get_graph_tx_ids(data)

                # Filter matching node indices in this timestep
                matching_indices = []
                for i in range(len(node_tx_ids)):
                    tx_id_str = str(node_tx_ids[i])
                    if search and search.strip().lower() not in tx_id_str.lower():
                        continue
                    matching_indices.append(i)

                if not matching_indices:
                    continue

                x = graph_service.loader.get_feature_matrix(data, config="original_all")
                probs = inference_service.score_timestep_nodes(x, data.edge_index) if inference_service.is_available else [0.0] * data.num_nodes
                y = data.y.numpy() if hasattr(data, "y") and data.y is not None else [-1] * data.num_nodes

                for i in matching_indices:
                    tx_id_str = str(node_tx_ids[i])
                    p = float(probs[i])
                    item = inference_service.triage_router.triage_candidate(
                        tx_id=node_tx_ids[i],
                        timestep=ts,
                        global_node_index=i,
                        predicted_prob=p,
                    )

                    if tx_id_str in seen_tx_ids:
                        # If already added from DB, update any null fields with live GNN inference metrics
                        for existing in matched_items:
                            if existing.tx_id == tx_id_str:
                                if existing.predicted_prob is None:
                                    existing.predicted_prob = item.predicted_prob
                                    existing.predicted_class = item.predicted_class
                                    existing.risk_level = item.risk_level
                                    existing.uncertainty_score = item.uncertainty_score
                                    existing.entropy = item.entropy
                                    existing.priority_score = item.priority_score
                                break
                        continue

                    db_tx = db_tx_map.get(tx_id_str)
                    is_reviewed = db_tx.is_triaged if db_tx else False
                    current_triage_status = db_tx.triage_status if db_tx and db_tx.triage_status else "QUEUED"

                    if triage_status and current_triage_status != triage_status:
                        continue
                    if risk_level and item.risk_level != risk_level:
                        continue
                    if min_priority > 0.0 and item.priority_score < min_priority:
                        continue

                    matched_items.append(
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
                            is_triaged=is_reviewed,
                            triage_status=current_triage_status,
                            created_at=db_tx.created_at if db_tx and db_tx.created_at else now_utc,
                            updated_at=db_tx.updated_at if db_tx and db_tx.updated_at else now_utc,
                        )
                    )
                    seen_tx_ids.add(tx_id_str)

                # If searching across timesteps and we collected sufficient items for the page, break early
                if search and search.strip() and len(matched_items) >= (offset + page_size * 2):
                    break
            except Exception as ts_err:
                logger.debug(f"Could not scan timestep {ts}: {ts_err}")
                continue

        matched_items.sort(key=lambda x: (x.priority_score or 0.0), reverse=True)
        total = len(matched_items)
        paginated = matched_items[offset : offset + page_size]
        total_pages = ceil(total / page_size) if total > 0 else 0

        return PaginatedResponse(
            items=paginated,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )
    except Exception as e:
        logger.error(f"Error reading transactions: {e}")
        return PaginatedResponse(items=[], total=0, page=page, page_size=page_size, total_pages=0)


def _determine_outcome_category(ground_truth_label: int, predicted_class: Optional[int]) -> str:
    """Classifies transaction into standard binary outcome matrix (or NOT_EVALUABLE if unlabeled)."""
    if ground_truth_label not in (0, 1) or predicted_class is None:
        return "NOT_EVALUABLE"
    if ground_truth_label == 1 and predicted_class == 1:
        return "TRUE_POSITIVE"
    if ground_truth_label == 0 and predicted_class == 1:
        return "FALSE_POSITIVE"
    if ground_truth_label == 0 and predicted_class == 0:
        return "TRUE_NEGATIVE"
    if ground_truth_label == 1 and predicted_class == 0:
        return "FALSE_NEGATIVE"
    return "NOT_EVALUABLE"


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
    Retrieves full transaction metadata, risk score, decision telemetry, and connected feedback records.
    """
    thresh = inference_service.decision_threshold
    decision_rule_str = f"p >= {thresh:.4f} -> Illicit (1); otherwise Licit (0)"

    tx = await tx_repo.get_by_tx_id(db, tx_id, load_details=True)
    if tx:
        gt_lbl = tx.ground_truth_label
        pred_cls = tx.predicted_class
        expl_cnt = len(tx.explanations)
        return TransactionDetailResponse(
            tx_id=tx.tx_id,
            timestep=tx.timestep,
            ground_truth_label=gt_lbl,
            predicted_prob=tx.predicted_prob,
            predicted_class=pred_cls,
            risk_level=tx.risk_level,
            uncertainty_score=tx.uncertainty_score,
            entropy=tx.entropy,
            priority_score=tx.priority_score,
            is_triaged=tx.is_triaged,
            triage_status=tx.triage_status,
            created_at=tx.created_at,
            updated_at=tx.updated_at,
            explanation_count=expl_cnt,
            feedback_count=len(tx.feedbacks),
            outgoing_edge_count=len(tx.outgoing_edges),
            incoming_edge_count=len(tx.incoming_edge_count) if hasattr(tx, "incoming_edge_count") else len(tx.incoming_edges),
            model_version=inference_service.model_version,
            architecture="GraphSAGE",
            decision_threshold=thresh,
            decision_rule=decision_rule_str,
            ground_truth_status="known" if gt_lbl in (0, 1) else "unavailable",
            outcome_category=_determine_outcome_category(gt_lbl, pred_cls),
            explanation_status="AVAILABLE" if expl_cnt > 0 else "NOT_GENERATED",
            explanation_method="GNNExplainer",
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
                expl_path = default_config.paths.processed_data_dir / "experiments" / "exp05_explanations" / f"tx_{tx_id}.json"
                expl_cnt = 1 if expl_path.exists() else 0

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
                    explanation_count=expl_cnt,
                    feedback_count=0,
                    outgoing_edge_count=0,
                    incoming_edge_count=0,
                    model_version=inference_service.model_version,
                    architecture="GraphSAGE",
                    decision_threshold=thresh,
                    decision_rule=decision_rule_str,
                    ground_truth_status="known" if y_val in (0, 1) else "unavailable",
                    outcome_category=_determine_outcome_category(y_val, triaged.predicted_class),
                    explanation_status="AVAILABLE" if expl_cnt > 0 else "NOT_GENERATED",
                    explanation_method="GNNExplainer",
                )
        except Exception:
            continue

    raise HTTPException(status_code=404, detail=f"Transaction '{tx_id}' not found.")
