"""Pydantic schemas for reviewer feedback submissions and triage queues."""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class FeedbackSubmissionRequest(BaseModel):
    """Schema for submitting analyst / reviewer verdicts."""
    tx_id: str = Field(..., description="Transaction string identifier")
    analyst_id: str = Field(..., description="Reviewer or investigator identifier")
    verdict: str = Field(..., description="Must be 'ILLICIT', 'LICIT', 'ESCALATED', or 'INCONCLUSIVE'")
    confidence: int = Field(..., ge=1, le=5, description="Confidence integer between 1 and 5")
    rationale: str = Field(..., min_length=3, description="Investigation case notes and justification")
    explanation_viewed: bool = Field(default=False, description="Whether analyst inspected GNNExplainer graph")
    feedback_source: str = Field(default="HUMAN_ANALYST", description="'HUMAN_ANALYST' or 'SIMULATED_ORACLE'")


class FeedbackResponse(BaseModel):
    """Reviewer feedback response payload."""
    id: int
    tx_id: str
    analyst_id: str
    verdict: str
    confidence: int
    rationale: str
    model_predicted_score: Optional[float] = None
    explanation_viewed: bool
    feedback_source: str
    reviewed_at: datetime
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TriageQueueItem(BaseModel):
    """Prioritized transaction in analyst triage review queue."""
    tx_id: str
    timestep: int
    predicted_prob: float
    predicted_class: int
    risk_level: str
    uncertainty_score: float
    entropy: float
    priority_score: float
    reason: str
    triage_status: str = "PENDING"

    model_config = ConfigDict(from_attributes=True)


class TriageQueueResponse(BaseModel):
    """Paginated triage review queue with risk distribution analytics."""
    items: List[TriageQueueItem]
    total: int
    min_priority: float
    risk_distribution: Dict[str, int]
