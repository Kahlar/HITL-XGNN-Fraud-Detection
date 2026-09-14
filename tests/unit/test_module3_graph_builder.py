"""Unit tests for Module 3 — Graph Builder, Subgraph Extractor, and Graph Statistics."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from src.module3_graph_builder.graph_stats import GraphStatisticsCalculator
from src.module3_graph_builder.pyg_builder import EllipticGraphBuilder, TimestepGraphLoader
from src.module3_graph_builder.subgraph_extractor import SubgraphExtractor


@pytest.fixture
def synthetic_graph_components():
    """Generates synthetic nodes, edges, features, and engineered data for testing."""
    # 5 nodes in timestep 1
    # 0 -> 1 -> 2, 0 -> 2, 3 -> 4, 4 is target, no isolated
    tx_ids = [f"tx_{i}" for i in range(5)]
    
    df_nodes = pd.DataFrame({
        "txId": tx_ids,
        "time_step": [1] * 5,
        "label": [1, 0, -1, 1, 0],
        "split": ["train"] * 5,
        "is_labeled": [True, True, False, True, True],
    })

    df_edges = pd.DataFrame({
        "txId1": ["tx_0", "tx_1", "tx_0", "tx_3"],
        "txId2": ["tx_1", "tx_2", "tx_2", "tx_4"],
    })

    # 165 features
    feat_cols = [f"feat_{i}" for i in range(165)]
    df_features = pd.DataFrame(np.random.randn(5, 165), columns=feat_cols)
    df_features.insert(0, "time_step", 1)
    df_features.insert(0, "txId", tx_ids)

    df_engineered = pd.DataFrame({
        "txId": tx_ids,
        "in_degree": [0.0, 1.0, 2.0, 0.0, 1.0],
        "out_degree": [2.0, 1.0, 0.0, 1.0, 0.0],
        "total_degree": [2.0, 2.0, 2.0, 1.0, 1.0],
        "in_out_ratio": [0.0, 1.0, 2.0, 0.0, 1.0],
        "flow_balance": [-1.0, 0.0, 1.0, -1.0, 1.0],
    })

    return df_nodes, df_edges, df_features, df_engineered


def test_build_timestep_graph(synthetic_graph_components):
    """Verifies single timestep PyG Data graph construction and attribute shapes."""
    df_nodes, df_edges, df_features, df_engineered = synthetic_graph_components
    builder = EllipticGraphBuilder()

    data = builder.build_timestep_graph(
        time_step=1,
        df_nodes=df_nodes,
        df_edges=df_edges,
        df_features=df_features,
        df_engineered=df_engineered,
        scaler=None
    )

    assert isinstance(data, Data)
    assert data.num_nodes == 5
    assert data.edge_index.shape == (2, 4)
    assert data.x.shape == (5, 170)
    assert data.y.shape == (5,)
    assert list(data.y.numpy()) == [1, 0, -1, 1, 0]
    assert data.time_step == 1
    assert data.split == "train"
    assert data.tx_id == [f"tx_{i}" for i in range(5)]
    assert (data.labeled_mask == torch.tensor([True, True, False, True, True])).all()


def test_feature_slicing(synthetic_graph_components):
    """Verifies feature configuration slices from TimestepGraphLoader."""
    df_nodes, df_edges, df_features, df_engineered = synthetic_graph_components
    builder = EllipticGraphBuilder()
    data = builder.build_timestep_graph(1, df_nodes, df_edges, df_features, df_engineered)

    # 1. original_all (165)
    x_all = TimestepGraphLoader.get_feature_matrix(data, "original_all")
    assert x_all.shape == (5, 165)

    # 2. original_local (93)
    x_loc = TimestepGraphLoader.get_feature_matrix(data, "original_local")
    assert x_loc.shape == (5, 93)

    # 3. engineered (5)
    x_eng = TimestepGraphLoader.get_feature_matrix(data, "engineered")
    assert x_eng.shape == (5, 5)

    # 4. combined (170)
    x_comb = TimestepGraphLoader.get_feature_matrix(data, "combined")
    assert x_comb.shape == (5, 170)


def test_2hop_subgraph_extraction(synthetic_graph_components):
    """Verifies 2-hop computational subgraph extraction preserving node IDs and mappings."""
    df_nodes, df_edges, df_features, df_engineered = synthetic_graph_components
    builder = EllipticGraphBuilder()
    data = builder.build_timestep_graph(1, df_nodes, df_edges, df_features, df_engineered)

    extractor = SubgraphExtractor(k_default=2)

    # Target node: tx_2 (connected to tx_1 and tx_0)
    sub_res = extractor.extract_k_hop_subgraph(data, target="tx_2", k=2, flow="both")

    assert sub_res.target_tx_id == "tx_2"
    assert sub_res.k_hops == 2
    assert "tx_2" in sub_res.node_tx_ids
    assert "tx_1" in sub_res.node_tx_ids
    assert "tx_0" in sub_res.node_tx_ids
    assert "tx_3" not in sub_res.node_tx_ids  # Disconnected component
    assert sub_res.num_nodes == 3
    assert sub_res.num_edges == 3  # (0->1, 1->2, 0->2)

    # Target local index correctly points to tx_2
    assert sub_res.subgraph.tx_id[sub_res.target_local_idx] == "tx_2"


def test_subgraph_isolated_node():
    """Verifies subgraph extractor handles isolated node with 0 edges gracefully."""
    # Graph with 2 isolated nodes
    data = Data(
        x=torch.randn(2, 170),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        y=torch.tensor([1, 0]),
        tx_id=["iso_1", "iso_2"],
        time_step=1,
        num_nodes=2,
    )
    extractor = SubgraphExtractor(k_default=2)
    sub_res = extractor.extract_k_hop_subgraph(data, target="iso_1", k=2)

    assert sub_res.num_nodes == 1
    assert sub_res.num_edges == 0
    assert sub_res.target_tx_id == "iso_1"
    assert sub_res.target_local_idx == 0


def test_graph_statistics(synthetic_graph_components, tmp_path: Path):
    """Verifies graph statistics calculation for timestep graphs."""
    df_nodes, df_edges, df_features, df_engineered = synthetic_graph_components
    builder = EllipticGraphBuilder()
    data = builder.build_timestep_graph(1, df_nodes, df_edges, df_features, df_engineered)

    calc = GraphStatisticsCalculator(output_dir=tmp_path)
    stats = calc.calculate_timestep_statistics(data)

    assert stats["time_step"] == 1
    assert stats["num_nodes"] == 5
    assert stats["num_edges"] == 4
    assert stats["illicit_count"] == 2
    assert stats["licit_count"] == 2
    assert stats["unknown_count"] == 1
    assert stats["weakly_connected_components"] == 2  # {0,1,2} and {3,4}


def test_real_timestep_graph_builder_integration(tmp_path: Path):
    """Integration test checking construction and serialization of a real timestep if available."""
    raw_dir = Path("data/raw/elliptic_bitcoin_dataset")
    if not (raw_dir / "elliptic_txs_classes.csv").exists():
        pytest.skip("Raw dataset not found at data/raw/elliptic_bitcoin_dataset")

    builder = EllipticGraphBuilder(raw_data_dir=raw_dir, processed_data_dir=tmp_path)
    df_classes = builder.loader.load_classes()
    df_edges = builder.loader.load_edges()
    df_timesteps = builder.loader.load_transaction_timesteps()
    df_features = builder.loader.load_features(use_float32=True)

    from src.module2_preprocessing.cleaner import LabelCleaner
    from src.module2_preprocessing.feature_engineering import GraphFeatureEngineer
    from src.module2_preprocessing.splitter import TemporalSplitter

    df_nodes = pd.merge(df_timesteps, LabelCleaner().clean_labels(df_classes), on="txId")
    df_nodes, _ = TemporalSplitter(output_dir=tmp_path).split_and_generate_metadata(df_nodes)
    df_eng = GraphFeatureEngineer().compute_graph_features(df_nodes, df_edges)

    # Build TS 1 graph (7,880 nodes)
    g1 = builder.build_timestep_graph(
        time_step=1,
        df_nodes=df_nodes,
        df_edges=df_edges,
        df_features=df_features,
        df_engineered=df_eng,
        scaler=None,
    )

    assert g1.num_nodes == 7880
    assert g1.x.shape == (7880, 170)
    assert g1.time_step == 1
    assert g1.split == "train"
    assert (g1.y == 1).sum().item() == 17
    assert (g1.y == 0).sum().item() == 2130
    assert (g1.y == -1).sum().item() == 5733
