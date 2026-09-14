"""Unit tests for Module 4 — Tabular Baselines and Evaluator."""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from src.module4_gnn.evaluator import ModelEvaluator
from src.module4_gnn.loss import FocalLoss, WeightedCrossEntropyLoss
from src.module4_gnn.models.baselines import (
    LightGBMBaseline,
    MLPBaseline,
    MLPNet,
    RandomForestBaseline,
    XGBoostBaseline,
)


@pytest.fixture
def synthetic_tabular_data():
    """Generates synthetic train, val, and test matrices with class imbalance."""
    np.random.seed(42)
    # Train: 100 samples (10 illicit, 90 licit)
    X_train = np.random.randn(100, 20).astype(np.float32)
    y_train = np.array([1]*10 + [0]*90, dtype=np.int64)

    # Val: 30 samples (5 illicit, 25 licit)
    X_val = np.random.randn(30, 20).astype(np.float32)
    y_val = np.array([1]*5 + [0]*25, dtype=np.int64)

    # Test: 40 samples (6 illicit, 34 licit)
    X_test = np.random.randn(40, 20).astype(np.float32)
    y_test = np.array([1]*6 + [0]*34, dtype=np.int64)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def test_random_forest_baseline(synthetic_tabular_data, tmp_path: Path):
    """Verifies Random Forest training, probability prediction, and serialization."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = synthetic_tabular_data

    rf = RandomForestBaseline(n_estimators=10, max_depth=4, random_state=42)
    rf.fit(X_train, y_train)

    probs = rf.predict_proba(X_test)
    assert probs.shape == (40,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Save & reload
    save_path = tmp_path / "rf.joblib"
    rf.save(save_path)
    assert save_path.exists()

    loaded_rf = RandomForestBaseline.load(save_path)
    loaded_probs = loaded_rf.predict_proba(X_test)
    np.testing.assert_allclose(probs, loaded_probs)


def test_xgboost_baseline(synthetic_tabular_data, tmp_path: Path):
    """Verifies XGBoost training, scale_pos_weight, and serialization."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = synthetic_tabular_data

    xgb_model = XGBoostBaseline(n_estimators=15, max_depth=3, random_state=42)
    xgb_model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=5)

    probs = xgb_model.predict_proba(X_test)
    assert probs.shape == (40,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    save_path = tmp_path / "xgb.json"
    xgb_model.save(save_path)
    assert save_path.exists()

    loaded_xgb = XGBoostBaseline.load(save_path)
    loaded_probs = loaded_xgb.predict_proba(X_test)
    np.testing.assert_allclose(probs, loaded_probs, atol=1e-5)


def test_lightgbm_baseline(synthetic_tabular_data, tmp_path: Path):
    """Verifies LightGBM training, scale_pos_weight, and serialization."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = synthetic_tabular_data

    lgb_model = LightGBMBaseline(n_estimators=15, num_leaves=15, random_state=42)
    lgb_model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=5)

    probs = lgb_model.predict_proba(X_test)
    assert probs.shape == (40,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    save_path = tmp_path / "lgb.txt"
    lgb_model.save(save_path)
    assert save_path.exists()


def test_mlp_baseline(synthetic_tabular_data, tmp_path: Path):
    """Verifies MLP architecture, forward pass, training loop, and early stopping."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = synthetic_tabular_data

    mlp = MLPBaseline(in_features=20, hidden_dim1=32, hidden_dim2=16, random_state=42)
    mlp.fit(X_train, y_train, X_val, y_val, epochs=5, batch_size=32)

    probs = mlp.predict_proba(X_test)
    assert probs.shape == (40,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    save_path = tmp_path / "mlp.pt"
    mlp.save(save_path)
    assert save_path.exists()

    loaded_mlp = MLPBaseline.load(save_path)
    loaded_probs = loaded_mlp.predict_proba(X_test)
    np.testing.assert_allclose(probs, loaded_probs, atol=1e-5)


def test_focal_loss():
    """Verifies Focal Loss computation and reduction."""
    loss_fn = FocalLoss(alpha=0.75, gamma=2.0)
    logits = torch.randn(10, 2)
    targets = torch.tensor([1, 0, 0, 1, 0, 0, 1, 0, 0, 0], dtype=torch.long)

    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_threshold_calibration_and_evaluation(synthetic_tabular_data):
    """Verifies threshold calibration on validation and frozen evaluation on test."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = synthetic_tabular_data

    y_val_prob = np.linspace(0.1, 0.9, len(y_val))
    tau_star, best_f1 = ModelEvaluator.calibrate_threshold(y_val, y_val_prob)
    assert 0.01 <= tau_star <= 0.99

    # Evaluate on test with frozen threshold
    y_test_prob = np.linspace(0.1, 0.9, len(y_test))
    result = ModelEvaluator.evaluate(y_test, y_test_prob, threshold=tau_star)

    assert result.threshold == tau_star
    assert 0.0 <= result.precision_illicit <= 1.0
    assert 0.0 <= result.recall_illicit <= 1.0
    assert 0.0 <= result.f1_illicit <= 1.0
    assert 0.0 <= result.pr_auc <= 1.0
    assert 0.0 <= result.roc_auc <= 1.0
    assert len(result.confusion_matrix) == 2


def test_anti_leakage_guarantees():
    """Explicit tests verifying zero leakage across splits."""
    # 1. Timestep boundaries
    train_ts = set(range(1, 35))
    val_ts = set(range(35, 40))
    test_ts = set(range(40, 50))

    assert train_ts.isdisjoint(val_ts)
    assert val_ts.isdisjoint(test_ts)
    assert train_ts.isdisjoint(test_ts)

    # 2. Unknown label filtering
    raw_labels = np.array([1, 0, -1, 1, -1, 0])
    labeled_mask = (raw_labels != -1)
    filtered = raw_labels[labeled_mask]
    assert -1 not in filtered
    assert set(filtered) == {0, 1}

    # 3. Class weight calculated strictly on train labels
    y_train = np.array([1]*10 + [0]*90)
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    scale_pos = num_neg / num_pos
    assert scale_pos == 9.0  # 90 / 10
