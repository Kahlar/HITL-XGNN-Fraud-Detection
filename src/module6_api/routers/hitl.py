"""Human-in-the-Loop triage queue and reviewer feedback submission routes."""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.logger import get_logger
from src.module3_graph_builder import get_graph_tx_ids
from src.module6_api.dependencies import get_db, get_graph_service, get_inference_service
from src.module6_api.schemas.feedback import (
    FeedbackResponse,
    FeedbackSubmissionRequest,
    TriageQueueItem,
    TriageQueueResponse,
)
from src.module6_api.services.graph_service import GraphService
from src.module6_api.services.inference_service import InferenceService
from src.module7_hitl.review_protocol import ReviewRecord, ReviewValidator
from src.module8_database.models.feedback import AnalystFeedback
from src.module8_database.models.transaction import Transaction
from src.module8_database.repositories.feedback_repo import FeedbackRepository
from src.module8_database.repositories.transaction_repo import TransactionRepository

logger = get_logger("Module6.Router.HITL")
router = APIRouter(prefix="/hitl", tags=["Human-in-the-Loop"])
feedback_repo = FeedbackRepository()
tx_repo = TransactionRepository()


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=201,
    summary="Submit reviewer verdict and case investigation feedback",
)
async def submit_analyst_feedback(
    submission: FeedbackSubmissionRequest,
    db: AsyncSession = Depends(get_db),
) -> FeedbackResponse:
    """
    Records an analyst verdict, audit justification, and advances transaction triage state.
    """
    # Schema validation
    test_record = ReviewRecord(
        analyst_id=submission.analyst_id,
        tx_id=submission.tx_id,
        timestep=0,
        global_node_index=0,
        verdict=submission.verdict,
        confidence=submission.confidence,
        rationale=submission.rationale,
        model_predicted_score=0.0,
        explanation_viewed=submission.explanation_viewed,
        reviewed_at=datetime.now(timezone.utc).isoformat(),
        feedback_source=submission.feedback_source,
    )
    try:
        ReviewValidator.validate(test_record)
    except ValueError as val_err:
        raise HTTPException(status_code=422, detail=str(val_err))

    # Ensure transaction exists in DB
    tx = await tx_repo.get_by_tx_id(db, submission.tx_id, load_details=False)
    if not tx:
        # Create minimal transaction entry if not yet in DB
        tx = Transaction(
            tx_id=submission.tx_id,
            timestep=40,
            ground_truth_label=-1,
            triage_status="REVIEWED",
            is_triaged=True,
        )
        await tx_repo.create(db, tx)

    now_utc = datetime.now(timezone.utc)
    # Create feedback model
    feedback_entry = AnalystFeedback(
        tx_id=submission.tx_id,
        analyst_id=submission.analyst_id,
        verdict=submission.verdict,
        confidence=submission.confidence,
        rationale=submission.rationale,
        model_predicted_score=tx.predicted_prob,
        explanation_viewed=submission.explanation_viewed,
        feedback_source=submission.feedback_source,
        reviewed_at=now_utc,
        created_at=now_utc,
    )

    created = await feedback_repo.submit_feedback(db, feedback_entry)
    await db.commit()

    logger.info(f"Recorded reviewer feedback for {submission.tx_id}: verdict={submission.verdict} by {submission.analyst_id}")
    return FeedbackResponse(
        id=created.id,
        tx_id=created.tx_id,
        analyst_id=created.analyst_id,
        verdict=created.verdict,
        confidence=created.confidence,
        rationale=created.rationale,
        model_predicted_score=created.model_predicted_score,
        explanation_viewed=created.explanation_viewed,
        feedback_source=created.feedback_source,
        reviewed_at=created.reviewed_at,
        created_at=created.created_at,
    )


@router.get(
    "/queue",
    response_model=TriageQueueResponse,
    summary="Get prioritized human reviewer triage queue",
)
async def get_triage_queue(
    timestep: Optional[int] = Query(default=None, ge=1, le=49, description="Discrete timestep"),
    min_priority: float = Query(default=0.40, ge=0.0, le=1.0, description="Minimum priority score threshold"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum items to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    db: AsyncSession = Depends(get_db),
    graph_service: GraphService = Depends(get_graph_service),
    inference_service: InferenceService = Depends(get_inference_service),
) -> TriageQueueResponse:
    """
    Returns prioritized review queue ranked by composite uncertainty, fraud risk, and graph diversity.
    """
    # 1. Check database first
    db_items = await tx_repo.get_triage_queue(
        session=db,
        timestep=timestep,
        min_priority=min_priority,
        triage_status="QUEUED",
        limit=limit,
        offset=offset,
    )

    if db_items:
        queue_items = [
            TriageQueueItem(
                tx_id=tx.tx_id,
                timestep=tx.timestep,
                predicted_prob=tx.predicted_prob or 0.0,
                predicted_class=tx.predicted_class or 0,
                risk_level=tx.risk_level or "LOW",
                uncertainty_score=tx.uncertainty_score or 0.0,
                entropy=tx.entropy or 0.0,
                priority_score=tx.priority_score or 0.0,
                reason="Prioritized in database triage queue",
                triage_status=tx.triage_status,
            )
            for tx in db_items
        ]
        risk_dist = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for item in queue_items:
            risk_dist[item.risk_level] = risk_dist.get(item.risk_level, 0) + 1

        return TriageQueueResponse(
            items=queue_items,
            total=len(queue_items),
            min_priority=min_priority,
            risk_distribution=risk_dist,
        )

    # 2. On-demand triage scoring from graph artifacts
    target_ts = timestep if timestep is not None else 40
    data = graph_service.loader.load_graph(target_ts)
    x = graph_service.loader.get_feature_matrix(data, config="original_all")
    probs = inference_service.score_timestep_nodes(x, data.edge_index) if inference_service.is_available else [0.0] * data.num_nodes
    node_tx_ids = get_graph_tx_ids(data)

    triage_candidates = []
    risk_dist = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}

    for i in range(len(node_tx_ids)):
        p = float(probs[i])
        triaged = inference_service.triage_router.triage_candidate(
            tx_id=node_tx_ids[i],
            timestep=target_ts,
            global_node_index=i,
            predicted_prob=p,
        )
        if triaged.priority_score >= min_priority:
            triage_candidates.append(
                TriageQueueItem(
                    tx_id=triaged.tx_id,
                    timestep=triaged.timestep,
                    predicted_prob=triaged.predicted_prob,
                    predicted_class=triaged.predicted_class,
                    risk_level=triaged.risk_level,
                    uncertainty_score=triaged.uncertainty_score,
                    entropy=triaged.entropy,
                    priority_score=triaged.priority_score,
                    reason=triaged.reason,
                    triage_status="QUEUED",
                )
            )
            risk_dist[triaged.risk_level] = risk_dist.get(triaged.risk_level, 0) + 1

    triage_candidates.sort(key=lambda x: x.priority_score, reverse=True)
    paginated = triage_candidates[offset : offset + limit]

    return TriageQueueResponse(
        items=paginated,
        total=len(triage_candidates),
        min_priority=min_priority,
        risk_distribution=risk_dist,
    )
