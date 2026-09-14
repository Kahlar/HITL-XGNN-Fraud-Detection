"""Unit tests for Module 2 — Preprocessing & Feature Engineering."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.module2_preprocessing.cleaner import LabelCleaner
from src.module2_preprocessing.feature_engineering import GraphFeatureEngineer
from src.module2_preprocessing.scaler import TrainOnlyScaler
from src.module2_preprocessing.splitter import TemporalSplitter


def test_label_cleaner_mappings():
    """Verifies standard 1/2/unknown mappings and error handling."""
    cleaner = LabelCleaner()
    assert cleaner.map_label("1") == 1
    assert cleaner.map_label("2") == 0
    assert cleaner.map_label("unknown") == -1

    # Test DataFrame cleaning
    df_raw = pd.DataFrame({
        "txId": ["t1", "t2", "t3"],
        "class": ["1", "2", "unknown"]
    })
    df_cleaned = cleaner.clean_labels(df_raw)
    assert list(df_cleaned["label"]) == [1, 0, -1]
    assert list(df_cleaned["is_labeled"]) == [True, True, False]


def test_label_cleaner_invalid_label():
    """Verifies exception on invalid label."""
    cleaner = LabelCleaner()
    with pytest.raises(ValueError):
        cleaner.map_label("invalid_class_99")


def test_temporal_splitter_disjointness(tmp_path: Path):
    """Verifies train/val/test partitions are strictly disjoint."""
    splitter = TemporalSplitter(
        train_timesteps=(1, 34),
        val_timesteps=(35, 39),
        test_timesteps=(40, 49),
        output_dir=tmp_path
    )
    assert splitter.verify_disjoint_partitions() is True

    # Test split assignment
    assert splitter.assign_split(1) == "train"
    assert splitter.assign_split(34) == "train"
    assert splitter.assign_split(35) == "val"
    assert splitter.assign_split(39) == "val"
    assert splitter.assign_split(40) == "test"
    assert splitter.assign_split(49) == "test"


def test_temporal_splitter_statistics(tmp_path: Path):
    """Verifies split statistics and metadata generation on sample data."""
    splitter = TemporalSplitter(
        train_timesteps=(1, 2),
        val_timesteps=(3, 3),
        test_timesteps=(4, 4),
        output_dir=tmp_path
    )
    df_txs = pd.DataFrame({
        "txId": ["n1", "n2", "n3", "n4", "n5", "n6"],
        "time_step": [1, 2, 3, 4, 4, 4],
        "label": [1, 0, 0, 1, 0, -1]
    })
    df_split, metadata = splitter.split_and_generate_metadata(df_txs)

    assert len(df_split) == 6
    assert metadata.train.total_nodes == 2
    assert metadata.train.illicit_count == 1
    assert metadata.train.licit_count == 1
    assert metadata.validation.total_nodes == 1
    assert metadata.test.total_nodes == 3
    assert metadata.test.unknown_count == 1
    assert (tmp_path / "split_metadata.json").exists()


def test_graph_feature_engineer():
    """Verifies topological feature calculation from graph edges."""
    engineer = GraphFeatureEngineer(epsilon=1e-5)

    # 3 nodes: A -> B -> C, A -> C
    # A: out=2, in=0
    # B: in=1, out=1
    # C: in=2, out=0
    df_txs = pd.DataFrame({"txId": ["A", "B", "C"]})
    df_edges = pd.DataFrame({
        "txId1": ["A", "B", "A"],
        "txId2": ["B", "C", "C"]
    })

    df_feats = engineer.compute_graph_features(df_txs, df_edges)

    # Check A
    feat_a = df_feats[df_feats["txId"] == "A"].iloc[0]
    assert feat_a["in_degree"] == 0.0
    assert feat_a["out_degree"] == 2.0
    assert feat_a["total_degree"] == 2.0
    assert feat_a["flow_balance"] < 0  # Net outflow

    # Check C
    feat_c = df_feats[df_feats["txId"] == "C"].iloc[0]
    assert feat_c["in_degree"] == 2.0
    assert feat_c["out_degree"] == 0.0
    assert feat_c["flow_balance"] > 0  # Net inflow


def test_feature_configurations():
    """Verifies all 4 feature configurations produce expected matrix dimensions."""
    engineer = GraphFeatureEngineer()

    # Synthetic features dataframe with 165 cols
    feat_cols = [f"feat_{i}" for i in range(165)]
    df_features = pd.DataFrame(np.random.randn(10, 165), columns=feat_cols)
    df_features.insert(0, "time_step", 1)
    df_features.insert(0, "txId", [f"tx_{i}" for i in range(10)])

    # Synthetic engineered dataframe with 5 cols
    df_eng = pd.DataFrame({
        "txId": [f"tx_{i}" for i in range(10)],
        "in_degree": np.ones(10),
        "out_degree": np.ones(10),
        "total_degree": 2 * np.ones(10),
        "in_out_ratio": np.ones(10),
        "flow_balance": np.zeros(10),
    })

    # 1. original_all (165)
    mat_all, names_all = engineer.get_feature_matrix(df_features, config="original_all")
    assert mat_all.shape == (10, 165)
    assert len(names_all) == 165

    # 2. original_local (93)
    mat_loc, names_loc = engineer.get_feature_matrix(df_features, config="original_local")
    assert mat_loc.shape == (10, 93)
    assert len(names_loc) == 93

    # 3. engineered (5)
    mat_eng, names_eng = engineer.get_feature_matrix(df_features, df_engineered=df_eng, config="engineered")
    assert mat_eng.shape == (10, 5)
    assert len(names_eng) == 5

    # 4. combined (170)
    mat_comb, names_comb = engineer.get_feature_matrix(df_features, df_engineered=df_eng, config="combined")
    assert mat_comb.shape == (10, 170)
    assert len(names_comb) == 170


def test_train_only_scaler(tmp_path: Path):
    """Verifies scaler is fitted strictly on training data and applied to test."""
    scaler = TrainOnlyScaler(scaler_type="robust", output_dir=tmp_path)

    # Train feature matrix
    X_train = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]], dtype=np.float32)
    # Val/Test feature matrix
    X_test = np.array([[5.0, 50.0], [6.0, 60.0]], dtype=np.float32)

    X_train_scaled = scaler.fit_transform(X_train, train_timesteps=[1, 2])
    assert scaler.is_fitted is True

    # Transform test without re-fitting
    X_test_scaled = scaler.transform(X_test)
    assert X_test_scaled.shape == (2, 2)

    # Persist and reload
    save_path = scaler.save("test_scaler.joblib")
    assert save_path.exists()

    loaded_scaler = TrainOnlyScaler.load(save_path)
    assert loaded_scaler.is_fitted is True
    np.testing.assert_allclose(loaded_scaler.transform(X_test), X_test_scaled)
