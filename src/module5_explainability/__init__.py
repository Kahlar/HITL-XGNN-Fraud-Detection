"""Module 5 — Explainability Engine (GNNExplainer, GAT Attention, Fidelity & Stability)."""

from src.module5_explainability.attention_extractor import GATAttentionEdge, GATAttentionExtractor
from src.module5_explainability.feature_ranker import FeatureRanker, RankedFeature
from src.module5_explainability.fidelity import FidelityEvaluator, FidelityResult
from src.module5_explainability.gnn_explainer import (
    GNNExplainerEngine,
    RankedEdge,
    TransactionExplanation,
)
from src.module5_explainability.latency import LatencyProfiler, LatencyProfileSummary
from src.module5_explainability.stability import ExplanationStabilityEvaluator, StabilityMetric

__all__ = [
    "FeatureRanker",
    "RankedFeature",
    "GNNExplainerEngine",
    "RankedEdge",
    "TransactionExplanation",
    "GATAttentionExtractor",
    "GATAttentionEdge",
    "FidelityEvaluator",
    "FidelityResult",
    "ExplanationStabilityEvaluator",
    "StabilityMetric",
    "LatencyProfiler",
    "LatencyProfileSummary",
]
