"""Pydantic schemas for GNNExplainer attribution and fidelity metrics."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class FeatureAttributionItem(BaseModel):
    """Ranked feature attribution item."""
    feature_index: int
    feature_name: str
    feature_type: str          # 'local' or 'aggregated'
    raw_importance: float
    normalized_importance: float
    rank: int


class EdgeAttributionItem(BaseModel):
    """Ranked directed edge attribution item."""
    source_tx_id: str
    target_tx_id: str
    source_global_idx: int
    target_global_idx: int
    edge_index_in_subgraph: int
    raw_importance: float
    normalized_importance: float
    rank: int


class FidelityBenchmarkMethod(BaseModel):
    """Aggregate fidelity benchmark metrics for an explainer method."""
    mean_fidelity_plus: float
    mean_fidelity_minus: float
    median_fidelity_plus: Optional[float] = None
    median_fidelity_minus: Optional[float] = None
    std_fidelity_plus: Optional[float] = None
    std_fidelity_minus: Optional[float] = None


class LatencyProfile(BaseModel):
    """Attribution latency performance profile."""
    num_samples: int
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    total_duration_seconds: float


class FidelityBenchmarkSummaryResponse(BaseModel):
    """Canonical EXP-05 comparative benchmark results."""
    overall_fidelity: Dict[str, Dict[str, float]]
    category_fidelity: Optional[Dict[str, Dict[str, Any]]] = None
    regime_fidelity: Optional[Dict[str, Dict[str, float]]] = None
    latency_profile: Optional[Dict[str, Any]] = None


class ExplanationResponse(BaseModel):
    """Structured XAI explanation response payload."""
    tx_id: str
    timestep: int
    model_version: str
    prediction_probability: float
    predicted_class: int
    risk_level: str
    category: str              # 'TP', 'FN', 'FP', 'TN', 'unlabeled'
    period: str                # 'pre_shock', 'shock_period', 'post_shock'
    subgraph_num_nodes: int
    subgraph_num_edges: int
    fidelity_plus: Optional[float] = None
    fidelity_minus: Optional[float] = None
    edge_sparsity: Optional[float] = None
    feature_sparsity: Optional[float] = None
    generation_latency_ms: float = 0.0
    top_features: List[FeatureAttributionItem]
    top_edges: List[EdgeAttributionItem]
    feature_summary: Dict[str, Any]
    benchmark_summary: Optional[Dict[str, Any]] = None
