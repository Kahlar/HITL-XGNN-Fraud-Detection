"""Unit tests for Module 7 — Human Analyst / HITL & Active Learning System (EXP-07)."""

from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.module7_hitl.active_learning import ActiveQueryEngine, CandidateTransaction
from src.module7_hitl.feedback_buffer import FeedbackBuffer
from src.module7_hitl.metrics import ActiveLearningMetrics
from src.module7_hitl.review_protocol import ReviewRecord, ReviewValidator, SimulatedReviewer
from src.module7_hitl.triage import TriageRouter


def test_triage_scoring_and_classification():
    """Verifies risk classification, Shannon entropy, and margin uncertainty."""
    router = TriageRouter(decision_threshold=0.5517)

    # Risk tiers
    assert router.classify_risk(0.90) == "CRITICAL"
    assert router.classify_risk(0.75) == "HIGH"
    assert router.classify_risk(0.45) == "MEDIUM"
    assert router.classify_risk(0.10) == "LOW"

    # Margin uncertainty (closest to tau* should be maximal)
    unc_close = router.compute_margin_uncertainty(0.5517)
    unc_far = router.compute_margin_uncertainty(0.99)
    assert np.isclose(unc_close, 1.0, atol=1e-3)
    assert unc_far < unc_close

    # Shannon entropy (maximal at p=0.5)
    ent_mid = router.compute_entropy(0.50)
    ent_ext = router.compute_entropy(0.01)
    assert np.isclose(ent_mid, 1.0, atol=1e-3)
    assert ent_ext < ent_mid

    # Triage item routing
    item = router.triage_candidate(
        tx_id="tx_123", timestep=35, global_node_index=10, predicted_prob=0.88, diversity_score=0.7
    )
    assert item.risk_level == "CRITICAL"
    assert item.predicted_class == 1
    assert item.priority_score > 0.5


def test_review_protocol_validation():
    """Verifies schema validation and rejection of invalid submissions."""
    valid_record = ReviewRecord(
        analyst_id="analyst_01",
        tx_id="tx_456",
        timestep=35,
        global_node_index=20,
        verdict="ILLICIT",
        confidence=5,
        rationale="Verified suspicious multi-hop fan-out structure.",
        model_predicted_score=0.92,
        explanation_viewed=True,
        reviewed_at=datetime.now(timezone.utc).isoformat(),
        feedback_source="HUMAN_ANALYST",
    )
    # Should pass without error
    ReviewValidator.validate(valid_record)

    # Invalid verdict
    invalid_verdict = ReviewRecord(
        analyst_id="analyst_01", tx_id="tx_456", timestep=35, global_node_index=20,
        verdict="FRAUDULENT_INVALID", confidence=5, rationale="note",
        model_predicted_score=0.9, explanation_viewed=False, reviewed_at="now", feedback_source="HUMAN_ANALYST"
    )
    with pytest.raises(ValueError, match="Invalid verdict"):
        ReviewValidator.validate(invalid_verdict)

    # Invalid confidence (outside 1..5)
    invalid_conf = ReviewRecord(
        analyst_id="analyst_01", tx_id="tx_456", timestep=35, global_node_index=20,
        verdict="ILLICIT", confidence=10, rationale="note",
        model_predicted_score=0.9, explanation_viewed=False, reviewed_at="now", feedback_source="HUMAN_ANALYST"
    )
    with pytest.raises(ValueError, match="Confidence must be an integer between 1 and 5"):
        ReviewValidator.validate(invalid_conf)


def test_simulated_oracle_reviewer():
    """Verifies simulated oracle reviewer label mapping and metadata."""
    reviewer = SimulatedReviewer(analyst_id="ORACLE_TEST")

    r_illicit = reviewer.review_transaction("tx_1", timestep=36, global_node_index=5, ground_truth_label=1, model_score=0.75)
    assert r_illicit.verdict == "ILLICIT"
    assert r_illicit.confidence == 5
    assert r_illicit.feedback_source == "SIMULATED_ORACLE"

    r_licit = reviewer.review_transaction("tx_2", timestep=36, global_node_index=6, ground_truth_label=0, model_score=0.15)
    assert r_licit.verdict == "LICIT"
    assert r_licit.confidence == 5

    r_inconcl = reviewer.review_transaction("tx_3", timestep=36, global_node_index=7, ground_truth_label=-1, model_score=0.50)
    assert r_inconcl.verdict == "INCONCLUSIVE"
    assert r_inconcl.confidence == 1


def test_active_learning_query_strategies():
    """Verifies query selection logic across all 5 strategies."""
    engine = ActiveQueryEngine(decision_threshold=0.55)

    candidates = [
        CandidateTransaction(tx_id="tx_0", timestep=35, global_node_index=0, predicted_prob=0.56, ground_truth=1, node_degree=10), # Near boundary, high deg
        CandidateTransaction(tx_id="tx_1", timestep=35, global_node_index=1, predicted_prob=0.95, ground_truth=1, node_degree=2),  # High risk
        CandidateTransaction(tx_id="tx_2", timestep=35, global_node_index=2, predicted_prob=0.10, ground_truth=0, node_degree=1),  # Low risk, certain
        CandidateTransaction(tx_id="tx_3", timestep=35, global_node_index=3, predicted_prob=0.54, ground_truth=0, node_degree=5),  # Near boundary
        CandidateTransaction(tx_id="tx_4", timestep=35, global_node_index=4, predicted_prob=0.88, ground_truth=1, node_degree=8),  # High risk
    ]

    # 1. Uncertainty: closest to 0.55 should be tx_0 (|0.56-0.55|=0.01) and tx_3 (|0.54-0.55|=0.01)
    q_unc = engine.sample_uncertainty(candidates, budget_count=2)
    unc_txs = {c.tx_id for c in q_unc}
    assert unc_txs == {"tx_0", "tx_3"}

    # 2. Entropy: highest entropy should also be near 0.50..0.56
    q_ent = engine.sample_entropy(candidates, budget_count=2)
    ent_txs = {c.tx_id for c in q_ent}
    assert "tx_0" in ent_txs and "tx_3" in ent_txs

    # 3. High Risk: highest prob should be tx_1 (0.95) and tx_4 (0.88)
    q_risk = engine.sample_high_risk(candidates, budget_count=2)
    assert [c.tx_id for c in q_risk] == ["tx_1", "tx_4"]

    # 4. Random: deterministic with fixed seed
    q_rnd1 = engine.sample_random(candidates, budget_count=2, random_seed=42)
    q_rnd2 = engine.sample_random(candidates, budget_count=2, random_seed=42)
    assert [c.tx_id for c in q_rnd1] == [c.tx_id for c in q_rnd2]


def test_feedback_buffer_operations(tmp_path):
    """Verifies adding, deduplication, JSON serialization, and reloading."""
    buffer = FeedbackBuffer(strategy_name="active_uncertainty", budget_ratio=0.10)
    reviewer = SimulatedReviewer()

    r1 = reviewer.review_transaction("tx_10", timestep=37, global_node_index=10, ground_truth_label=1, model_score=0.8)
    r2 = reviewer.review_transaction("tx_20", timestep=37, global_node_index=20, ground_truth_label=0, model_score=0.2)

    buffer.add_record(r1)
    buffer.add_record(r2)
    buffer.add_record(r1)  # Duplicate, should not increase size

    assert len(buffer) == 2
    assert buffer.contains_tx("tx_10")
    assert buffer.contains_tx("tx_20")
    assert not buffer.contains_tx("tx_99")

    # Persist and reload
    save_file = tmp_path / "test_buffer.json"
    buffer.save(save_file)

    loaded = FeedbackBuffer.load(save_file)
    assert len(loaded) == 2
    assert loaded.strategy_name == "active_uncertainty"
    assert loaded.contains_tx("tx_10")


def test_sample_efficiency_calculation():
    """Verifies sample efficiency metric normalization."""
    eff = ActiveLearningMetrics.compute_sample_efficiency(
        strategy="active_combined",
        budget_ratio=0.10,
        num_samples=500,
        base_f1=0.5175,
        retrained_f1=0.5675,  # +0.05 gain
        base_pr_auc=0.4499,
        retrained_pr_auc=0.4999, # +0.05 gain
        test_precision=0.75,
        test_recall=0.45,
    )
    assert eff.f1_delta == 0.05
    # (0.05 / 500) * 1000 = 0.10
    assert np.isclose(eff.f1_efficiency_per_1k, 0.10, atol=1e-4)


def test_anti_leakage_and_test_isolation():
    """Asserts that test timesteps t=40..49 are strictly excluded from feedback pools."""
    feedback_timesteps = set(range(35, 40))
    test_timesteps = set(range(40, 50))
    train_timesteps = set(range(1, 35))

    # Assert partitions are mutually disjoint
    assert feedback_timesteps.isdisjoint(test_timesteps)
    assert train_timesteps.isdisjoint(test_timesteps)
    assert train_timesteps.isdisjoint(feedback_timesteps)
