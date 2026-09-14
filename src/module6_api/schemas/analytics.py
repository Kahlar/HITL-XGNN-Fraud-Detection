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
