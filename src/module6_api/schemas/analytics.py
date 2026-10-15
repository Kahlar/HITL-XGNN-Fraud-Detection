"""Pydantic schemas for dashboard performance analytics, temporal drift, and active learning metrics."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class TemporalMetricItem(BaseModel):
    """Temporal metric evaluation for a discrete timestep."""
    timestep: int
    period: str
    num_labeled: int
    num_illicit: int
    illicit_prevalence_pct: float
    f1_score: float
    pr_auc: float
    precision: float
    recall: float


class ModelPerformanceSummary(BaseModel):
    """Model performance benchmark summary."""
    model_version: str
    architecture: str
    decision_threshold: float
    test_f1: float
    test_pr_auc: float
    test_precision: float
    test_recall: float
    test_roc_auc: float


class AnalyticsMetricsResponse(BaseModel):
    """Master analytics response for high-level dashboard charts."""
    dataset_summary: Dict[str, Any]
    active_model: ModelPerformanceSummary
    temporal_drift: List[TemporalMetricItem]
    active_learning_benchmarks: Dict[str, Any]
    feedback_stats: Dict[str, int]


class GroupEvaluationRequest(BaseModel):
    """Request payload for group-level metric evaluation."""
    scope: str = "timestep"  # 'timestep', 'filtered', or 'reviewed'
    timestep: Optional[int] = None
    tx_ids: Optional[List[str]] = None


class GroupEvaluationResponse(BaseModel):
    """Aggregated evaluation metrics for a defined cohort of transactions."""
    scope: str
    timestep: Optional[int] = None
    total_transactions: int
    labeled_count: int
    unlabeled_count: int
    illicit_count: int
    licit_count: int
    decision_threshold: float
    f1_score: Optional[float] = None
    pr_auc: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    status: str = "VALID"
    notes: str = ""


class ThresholdPolicyResponse(BaseModel):
    """Threshold selection methodology and decision policy response."""
    active_model: str
    architecture: str
    calibrated_threshold: float
    calibration_method: str
    calibration_split: str
    grid_specification: Dict[str, Any]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    comparison_thresholds: Dict[str, float]
    decision_rules: Dict[str, str]


class ModelValidationResponse(BaseModel):
    """Model inventory, chronological split boundaries, and verification evidence."""
    timestamp_utc: str
    train_timesteps: List[int]
    validation_timesteps: List[int]
    test_timesteps: List[int]
    zero_leakage_policy: str
    models: List[Dict[str, Any]]
    verification_evidence: Dict[str, Any]
