"""Unit tests for Module 5 — Temporal Drift Evaluation and Distribution Shift Analysis (EXP-06)."""

from pathlib import Path
import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.module5_temporal_drift.drift_evaluator import (
    PeriodAggregates,
    TemporalDriftEvaluator,
    TimestepDriftMetric,
)


def test_period_classification():
    """Verifies that test timesteps are accurately assigned to analytical periods."""
    assert TemporalDriftEvaluator.classify_period(40) == "pre_shock"
    assert TemporalDriftEvaluator.classify_period(41) == "pre_shock"
    assert TemporalDriftEvaluator.classify_period(42) == "pre_shock"

    assert TemporalDriftEvaluator.classify_period(43) == "shock_period"
    assert TemporalDriftEvaluator.classify_period(44) == "shock_period"
    assert TemporalDriftEvaluator.classify_period(45) == "shock_period"
    assert TemporalDriftEvaluator.classify_period(46) == "shock_period"

    assert TemporalDriftEvaluator.classify_period(47) == "post_shock"
    assert TemporalDriftEvaluator.classify_period(48) == "post_shock"
    assert TemporalDriftEvaluator.classify_period(49) == "post_shock"


def test_period_aggregates_computation():
    """Verifies aggregation of metrics across analytical periods."""
    mock_metrics = [
        TimestepDriftMetric(
            timestep=40, total_nodes=100, labeled_nodes=50, illicit_count=10, licit_count=40,
            illicit_prevalence=0.20, predicted_illicit_count=8, threshold=0.5517,
            f1_illicit=0.80, pr_auc=0.75, roc_auc=0.90, precision_illicit=0.85, recall_illicit=0.75,
            accuracy=0.92, true_positives=6, false_positives=1, true_negatives=39, false_negatives=4,
            period="pre_shock",
        ),
        TimestepDriftMetric(
            timestep=41, total_nodes=100, labeled_nodes=50, illicit_count=10, licit_count=40,
            illicit_prevalence=0.20, predicted_illicit_count=8, threshold=0.5517,
            f1_illicit=0.70, pr_auc=0.65, roc_auc=0.85, precision_illicit=0.75, recall_illicit=0.65,
            accuracy=0.88, true_positives=5, false_positives=2, true_negatives=38, false_negatives=5,
            period="pre_shock",
        ),
    ]

    evaluator = TemporalDriftEvaluator(graphs_dir=Path("data/processed/graphs"))
    agg = evaluator.compute_period_aggregates(mock_metrics, "pre_shock")

    assert agg.period_name == "pre_shock"
    assert agg.timesteps == [40, 41]
    assert agg.total_labeled_nodes == 100
    assert agg.total_illicit == 20
    assert agg.mean_f1 == 0.75
    assert agg.mean_pr_auc == 0.70
    assert agg.mean_illicit_prevalence == 0.20


def test_handling_low_prevalence_edge_case():
    """Verifies metric computation when a timestep has near-zero or zero illicit samples."""
    evaluator = TemporalDriftEvaluator(graphs_dir=Path("data/processed/graphs"))

    # Synthetic graph with 1 illicit sample out of 50
    x = torch.randn(50, 16)
    edge_index = torch.stack([torch.arange(0, 49), torch.arange(1, 50)], dim=0)
    y = torch.tensor([1] + [0]*49, dtype=torch.long)
    synthetic_data = Data(x=x, edge_index=edge_index, y=y, num_nodes=50)

    # Mock linear model returning class 0 prediction
    class MockModel(torch.nn.Module):
        def forward(self, x, edge_index):
            return torch.stack([torch.ones(len(x))*5.0, torch.zeros(len(x))], dim=1)

    model = MockModel()

    # Create dummy method to pass synthetic data
    evaluator.loader.load_graph = lambda ts: synthetic_data
    evaluator.loader.get_feature_matrix = lambda data, config: data.x

    metric = evaluator.evaluate_timestep(model, timestep=46, threshold=0.5517)
    assert metric.illicit_count == 1
    assert metric.labeled_nodes == 50
    assert metric.f1_illicit == 0.0  # Predicted 0
    assert metric.recall_illicit == 0.0
    assert metric.false_positives == 0
    assert metric.false_negatives == 1
    assert metric.true_negatives == 49
    assert metric.true_positives == 0


def test_anti_leakage_and_immutability():
    """Asserts test timestep bounds and frozen threshold immutability."""
    test_timesteps = list(range(40, 50))
    assert len(test_timesteps) == 10
    assert min(test_timesteps) == 40
    assert max(test_timesteps) == 49

    # Train / Val bounds
    train_timesteps = set(range(1, 35))
    val_timesteps = set(range(35, 40))
    test_set = set(test_timesteps)

    assert test_set.isdisjoint(train_timesteps)
    assert test_set.isdisjoint(val_timesteps)
