"""Pydantic models for transactions and detailed case views."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class TransactionResponse(BaseModel):
    """Transaction summary response item."""
    tx_id: str
    timestep: int
    ground_truth_label: int
    predicted_prob: Optional[float] = None
    predicted_class: Optional[int] = None
    risk_level: Optional[str] = None
    uncertainty_score: Optional[float] = None
    entropy: Optional[float] = None
    priority_score: Optional[float] = None
    is_triaged: bool = False
    triage_status: str = "PENDING"
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TransactionDetailResponse(TransactionResponse):
    """Detailed transaction response with explanation metadata and reviewer feedback."""
    explanation_count: int = 0
    feedback_count: int = 0
    outgoing_edge_count: int = 0
    incoming_edge_count: int = 0
    model_version: str = "graphsage_hitl_uncertainty_20"
    architecture: str = "GraphSAGE"
    decision_threshold: float = 0.5517
    decision_rule: str = "p >= 0.5517 -> Illicit (1); otherwise Licit (0)"
    ground_truth_status: str = "known"
    outcome_category: str = "NOT_EVALUABLE"
    explanation_status: str = "NOT_GENERATED"
    explanation_method: str = "GNNExplainer"
