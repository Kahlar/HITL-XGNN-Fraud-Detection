"""Unit tests for Module 1 — Dataset (Loader and Validator)."""

import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.module1_dataset.loader import EllipticDataLoader
from src.module1_dataset.validator import DatasetValidator


@pytest.fixture
def synthetic_raw_dataset(tmp_path: Path):
    """Creates a small synthetic Elliptic-like raw dataset directory."""
    raw_dir = tmp_path / "raw_elliptic"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Synthetic classes: 6 nodes (2 illicit, 2 licit, 2 unknown)
    classes_data = [
        ["txId", "class"],
        ["1001", "1"],
        ["1002", "2"],
        ["1003", "unknown"],
        ["1004", "1"],
        ["1005", "2"],
        ["1006", "unknown"],
    ]
    classes_file = raw_dir / "elliptic_txs_classes.csv"
    with open(classes_file, "w", encoding="utf-8") as f:
        for row in classes_data:
            f.write(",".join(row) + "\n")

    # 2. Synthetic edges: 5 directed edges, all intra-timestep
    # Timestep 1: 1001 -> 1002, 1002 -> 1003
    # Timestep 2: 1004 -> 1005, 1005 -> 1006
    edges_data = [
        ["txId1", "txId2"],
        ["1001", "1002"],
        ["1002", "1003"],
        ["1004", "1005"],
        ["1005", "1006"],
    ]
    edgelist_file = raw_dir / "elliptic_txs_edgelist.csv"
    with open(edgelist_file, "w", encoding="utf-8") as f:
        for row in edges_data:
            f.write(",".join(row) + "\n")

    # 3. Synthetic features: 6 rows, 167 columns (no header)
    # txId, time_step, feat_0 .. feat_164
    features_file = raw_dir / "elliptic_txs_features.csv"
    with open(features_file, "w", encoding="utf-8") as f:
        # TS 1 nodes: 1001, 1002, 1003
        for tx_id in ["1001", "1002", "1003"]:
            feats = [str(round(np.random.randn(), 4)) for _ in range(165)]
            row = [tx_id, "1"] + feats
            f.write(",".join(row) + "\n")
        # TS 2 nodes: 1004, 1005, 1006
        for tx_id in ["1004", "1005", "1006"]:
            feats = [str(round(np.random.randn(), 4)) for _ in range(165)]
            row = [tx_id, "2"] + feats
            f.write(",".join(row) + "\n")

    return raw_dir


def test_loader_file_discovery(synthetic_raw_dataset: Path):
    """Verifies loader discovers all 3 raw CSV files."""
    loader = EllipticDataLoader(synthetic_raw_dataset)
    assert loader.check_files_exist() is True


def test_loader_classes(synthetic_raw_dataset: Path):
    """Verifies class loading and schema."""
    loader = EllipticDataLoader(synthetic_raw_dataset)
    df_classes = loader.load_classes()
    assert len(df_classes) == 6
    assert list(df_classes.columns) == ["txId", "class"]
    assert set(df_classes["class"].unique()) == {"1", "2", "unknown"}


def test_loader_edges(synthetic_raw_dataset: Path):
    """Verifies edgelist loading."""
    loader = EllipticDataLoader(synthetic_raw_dataset)
    df_edges = loader.load_edges()
    assert len(df_edges) == 4
    assert list(df_edges.columns) == ["txId1", "txId2"]


def test_loader_features(synthetic_raw_dataset: Path):
    """Verifies features loading, column naming, and float32 dtype."""
    loader = EllipticDataLoader(synthetic_raw_dataset)
    df_features = loader.load_features(use_float32=True)
    assert df_features.shape == (6, 167)
    assert df_features.columns[0] == "txId"
    assert df_features.columns[1] == "time_step"
    assert df_features.columns[2] == "feat_0"
    assert df_features.columns[-1] == "feat_164"
    assert df_features["feat_0"].dtype == np.float32


def test_validator_on_synthetic(synthetic_raw_dataset: Path, tmp_path: Path):
    """Runs validator on synthetic data and checks JSON report."""
    out_dir = tmp_path / "processed"
    validator = DatasetValidator(synthetic_raw_dataset, output_dir=out_dir)
    report = validator.validate_and_generate_report()

    assert report.validation_passed is True
    assert report.total_unique_transactions == 6
    assert report.duplicate_tx_ids_found == 0
    assert report.missing_or_nan_values_found == 0
    assert report.edge_statistics.total_edges == 4
    assert report.edge_statistics.intra_timestep_edges == 4
    assert report.edge_statistics.pct_intra_timestep == 100.0
    assert report.class_distribution.illicit_count == 2
    assert report.class_distribution.licit_count == 2
    assert report.class_distribution.unknown_count == 2
    assert report.class_distribution.imbalance_ratio == 1.0

    # Verify report files saved
    assert (out_dir / "dataset_validation_report.json").exists()
    assert (out_dir / "dataset_validation_report.txt").exists()


def test_validator_detects_self_loops(tmp_path: Path):
    """Verifies validator detects and logs self-loops in edgelist."""
    raw_dir = tmp_path / "raw_self_loop"
    raw_dir.mkdir()

    # Classes
    pd.DataFrame({"txId": ["1", "2"], "class": ["1", "2"]}).to_csv(raw_dir / "elliptic_txs_classes.csv", index=False)
    # Edges with self-loop 1 -> 1
    pd.DataFrame({"txId1": ["1", "1"], "txId2": ["1", "2"]}).to_csv(raw_dir / "elliptic_txs_edgelist.csv", index=False)
    # Features
    feats = pd.DataFrame([["1", 1] + [0.0]*165, ["2", 1] + [0.0]*165])
    feats.to_csv(raw_dir / "elliptic_txs_features.csv", header=False, index=False)

    validator = DatasetValidator(raw_dir, output_dir=tmp_path / "out")
    report = validator.validate_and_generate_report()
    assert report.edge_statistics.self_loops == 1
