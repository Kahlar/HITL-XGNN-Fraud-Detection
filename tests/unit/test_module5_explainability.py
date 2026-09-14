"""Unit tests for Module 5 — Explainability Engine (GNNExplainer, Fidelity, Sparsity, Stability)."""

from pathlib import Path
import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.module3_graph_builder.subgraph_extractor import SubgraphData
from src.module4_gnn.models.gat import GATNet
from src.module4_gnn.models.graphsage import GraphSAGENet
from src.module5_explainability.attention_extractor import GATAttentionExtractor
from src.module5_explainability.feature_ranker import FeatureRanker
from src.module5_explainability.fidelity import FidelityEvaluator
from src.module5_explainability.gnn_explainer import GNNExplainerEngine
from src.module5_explainability.latency import LatencyProfiler
from src.module5_explainability.stability import ExplanationStabilityEvaluator


@pytest.fixture
def synthetic_subgraph():
    """Creates a synthetic 2-hop computational subgraph with 6 nodes, 8 edges, 165 features."""
    torch.manual_seed(42)
    x = torch.randn(6, 165)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 1, 0, 5],
        [1, 2, 0, 4, 0, 5, 3, 2],
    ], dtype=torch.long)
    y = torch.tensor([1, 0, 1, 0, 0, 1], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=6)

    subgraph = SubgraphData(
        subgraph=data,
        target_tx_id="tx_target_123",
        target_local_idx=0,
        target_global_idx=100,
        k_hops=2,
        num_nodes=6,
        num_edges=8,
        node_tx_ids=["tx_target_123", "tx_101", "tx_102", "tx_103", "tx_104", "tx_105"],
        local_to_global_indices=[100, 101, 102, 103, 104, 105],
        global_to_local_indices={100: 0, 101: 1, 102: 2, 103: 3, 104: 4, 105: 5},
    )
    return subgraph


@pytest.fixture
def mock_graphsage():
    """Returns a GraphSAGENet instance."""
    torch.manual_seed(42)
    return GraphSAGENet(in_features=165, hidden_dim=32, num_classes=2, dropout=0.0)


def test_feature_ranker_partitioning():
    """Verifies that features are correctly classified as local vs. aggregated."""
    assert FeatureRanker.get_feature_type(0) == "local"
    assert FeatureRanker.get_feature_type(92) == "local"
    assert FeatureRanker.get_feature_type(93) == "aggregated"
    assert FeatureRanker.get_feature_type(164) == "aggregated"
    assert FeatureRanker.get_feature_name(42) == "feature_042"

    mask = np.zeros(165)
    mask[10] = 5.0  # local
    mask[100] = 2.5 # agg
    ranked, summary = FeatureRanker.rank_features(mask, top_k=5)

    assert ranked[0].feature_index == 10
    assert ranked[0].feature_type == "local"
    assert ranked[1].feature_index == 100
    assert ranked[1].feature_type == "aggregated"
    assert summary["dominant_feature_type"] == "local"
    assert summary["local_importance_ratio"] > summary["aggregated_importance_ratio"]


def test_gnn_explainer_execution(mock_graphsage, synthetic_subgraph):
    """Verifies that GNNExplainer produces valid edge and feature masks."""
    engine = GNNExplainerEngine(model=mock_graphsage, epochs=10, random_seed=42)
    explanation = engine.explain_subgraph(
        subgraph=synthetic_subgraph,
        target_local_idx=0,
        timestep=40,
        top_k_features=10,
        top_k_edges=5,
    )

    assert explanation.tx_id == "tx_target_123"
    assert explanation.timestep == 40
    assert explanation.global_node_index == 100
    assert explanation.subgraph_num_nodes == 6
    assert explanation.subgraph_num_edges == 8
    assert len(explanation.edge_mask) == 8
    assert len(explanation.feature_mask) == 165
    assert len(explanation.top_features) == 10
    assert len(explanation.top_edges) == 5
    assert explanation.generation_latency_ms > 0


def test_fidelity_and_sparsity_calculations(mock_graphsage, synthetic_subgraph):
    """Verifies Fidelity+, Fidelity-, and Sparsity calculations."""
    evaluator = FidelityEvaluator(model=mock_graphsage)
    edge_mask = np.array([0.9, 0.1, 0.8, 0.2, 0.05, 0.7, 0.3, 0.15])
    feat_mask = np.random.uniform(0.0, 1.0, 165)

    results = evaluator.evaluate_explanation_faithfulness(
        data=synthetic_subgraph.subgraph,
        target_local_idx=0,
        edge_mask=edge_mask,
        feature_mask=feat_mask,
        method_name="gnn_explainer",
        budget_ratios=[0.10, 0.25],
    )

    assert len(results) == 2
    r_10 = results[0]
    assert r_10.method == "gnn_explainer"
    assert r_10.budget_ratio == 0.10
    assert 0.0 <= r_10.edge_sparsity <= 1.0
    assert 0.0 <= r_10.feature_sparsity <= 1.0
    # Fid+ = p_orig - p_removed
    assert np.isclose(r_10.fidelity_plus, r_10.original_prob - r_10.prob_removed, atol=1e-4)
    # Fid- = p_orig - p_expl
    assert np.isclose(r_10.fidelity_minus, r_10.original_prob - r_10.prob_expl_only, atol=1e-4)


def test_random_baseline_reproducibility(mock_graphsage, synthetic_subgraph):
    """Verifies that random baseline explanations produce deterministic results under fixed seed."""
    evaluator = FidelityEvaluator(model=mock_graphsage)
    r1 = evaluator.evaluate_random_baseline(synthetic_subgraph.subgraph, target_local_idx=0, random_seed=42)
    r2 = evaluator.evaluate_random_baseline(synthetic_subgraph.subgraph, target_local_idx=0, random_seed=42)

    assert r1[0].fidelity_plus == r2[0].fidelity_plus
    assert r1[0].fidelity_minus == r2[0].fidelity_minus


def test_explanation_stability_jaccard(mock_graphsage, synthetic_subgraph):
    """Verifies Jaccard similarity across random initialization seeds."""
    evaluator = ExplanationStabilityEvaluator(model=mock_graphsage)
    metric = evaluator.evaluate_node_stability(
        subgraph=synthetic_subgraph,
        target_local_idx=0,
        timestep=40,
        seeds=[42, 123],
        epochs=10,
    )

    assert metric.tx_id == "tx_target_123"
    assert len(metric.seeds_evaluated) == 2
    assert 0.0 <= metric.mean_edge_jaccard <= 1.0
    assert 0.0 <= metric.mean_feature_cosine_similarity <= 1.0


def test_latency_profiler():
    """Verifies latency collection and percentile statistics."""
    profiler = LatencyProfiler()
    profiler.record(10.0, subgraph_nodes=5)
    profiler.record(20.0, subgraph_nodes=10)
    profiler.record(30.0, subgraph_nodes=15)

    summary = profiler.summarize()
    assert summary.num_samples == 3
    assert summary.mean_latency_ms == 20.0
    assert summary.median_latency_ms == 20.0
    assert summary.min_latency_ms == 10.0
    assert summary.max_latency_ms == 30.0
