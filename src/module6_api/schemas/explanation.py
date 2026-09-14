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
