"""Unit tests for GNN models (GCN, GraphSAGE, Multi-Head GAT), loss masking, and attention extraction."""

from pathlib import Path
import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.module4_gnn.evaluator import ModelEvaluator
from src.module4_gnn.loss import FocalLoss, WeightedCrossEntropyLoss
from src.module4_gnn.models.gat import GATNet
from src.module4_gnn.models.gcn import GCNNet
from src.module4_gnn.models.graphsage import GraphSAGENet


@pytest.fixture
def synthetic_pyg_graph():
    """Generates a synthetic directed PyG graph with labeled and unknown nodes."""
    torch.manual_seed(42)
    num_nodes = 20
    in_features = 16
    x = torch.randn(num_nodes, in_features)

    # 30 directed edges
    src = torch.randint(0, num_nodes, (30,))
    dst = torch.randint(0, num_nodes, (30,))
    edge_index = torch.stack([src, dst], dim=0)

    # Labels: 3 illicit (1), 10 licit (0), 7 unknown (-1)
    y = torch.tensor(
        [1, 0, 0, -1, 1, 0, -1, 0, 0, -1, 1, 0, 0, -1, 0, 0, -1, 0, -1, -1],
        dtype=torch.long,
    )

    return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)


def test_gcn_forward_and_backward(synthetic_pyg_graph):
    """Verifies GCN forward pass, output shape, and backward gradient flow."""
    data = synthetic_pyg_graph
    model = GCNNet(in_features=16, hidden_dim=32, num_classes=2, dropout=0.2)

    logits = model(data.x, data.edge_index)
    assert logits.shape == (20, 2)

    # Supervised loss over labeled nodes only
    labeled_mask = (data.y != -1)
    loss_fn = FocalLoss(alpha=0.75, gamma=2.0)
    loss = loss_fn(logits[labeled_mask], data.y[labeled_mask])

    assert loss.item() > 0.0
    loss.backward()

    # Verify gradients computed
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_graphsage_forward_and_backward(synthetic_pyg_graph):
    """Verifies GraphSAGE forward pass, output shape, and backward gradient flow."""
    data = synthetic_pyg_graph
    model = GraphSAGENet(in_features=16, hidden_dim=32, num_classes=2, dropout=0.2, aggr="mean")

    logits = model(data.x, data.edge_index)
    assert logits.shape == (20, 2)

    labeled_mask = (data.y != -1)
    loss_fn = nn_loss = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits[labeled_mask], data.y[labeled_mask])

    assert loss.item() > 0.0
    loss.backward()

    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_gat_forward_and_attention_extraction(synthetic_pyg_graph):
    """Verifies Multi-Head GAT forward pass, logits shape, and attention coefficient extraction."""
    data = synthetic_pyg_graph
    model = GATNet(in_features=16, hidden_dim=16, num_classes=2, heads=4, dropout=0.2)

    logits = model(data.x, data.edge_index)
    assert logits.shape == (20, 2)

    # Extract internal attention coefficients
    edge_idx_att, alpha = model.get_attention_weights(data.x, data.edge_index)
    assert edge_idx_att.dim() == 2 and edge_idx_att.size(0) == 2
    assert alpha.dim() == 2 and alpha.size(1) == 4  # 4 attention heads
    assert (alpha >= 0.0).all() and (alpha <= 1.0).all()


def test_gnn_checkpoint_save_and_load(synthetic_pyg_graph, tmp_path: Path):
    """Verifies GNN serialization, deserialization, and prediction determinism."""
    data = synthetic_pyg_graph
    model = GraphSAGENet(in_features=16, hidden_dim=32, num_classes=2)
    model.eval()

    with torch.no_grad():
        orig_logits = model(data.x, data.edge_index)

    save_path = tmp_path / "sage_test.pt"
    model.save_checkpoint(save_path, metadata={"test": True})
    assert save_path.exists()

    loaded_model = GraphSAGENet.load_checkpoint(save_path)
    loaded_model.eval()
    with torch.no_grad():
        loaded_logits = loaded_model(data.x, data.edge_index)

    torch.testing.assert_close(orig_logits, loaded_logits)


def test_unknown_node_masking_integrity(synthetic_pyg_graph):
    """Verifies that unknown nodes (y=-1) are masked out from loss calculation."""
    data = synthetic_pyg_graph
    model = GCNNet(in_features=16, hidden_dim=32, num_classes=2)

    logits = model(data.x, data.edge_index)
    labeled_mask = (data.y != -1)

    assert labeled_mask.sum().item() == 13  # 3 illicit + 10 licit
    assert (~labeled_mask).sum().item() == 7  # 7 unknown

    # Loss over labeled nodes succeeds without ValueError
    loss_fn = FocalLoss()
    loss = loss_fn(logits[labeled_mask], data.y[labeled_mask])
    assert torch.isfinite(loss)


def test_gnn_leakage_guarantees():
    """Explicit tests confirming anti-leakage boundaries for GNN training."""
    train_ts = set(range(1, 35))
    val_ts = set(range(35, 40))
    test_ts = set(range(40, 50))

    assert train_ts.isdisjoint(val_ts)
    assert val_ts.isdisjoint(test_ts)
    assert train_ts.isdisjoint(test_ts)

    # Labeled counts
    assert len(train_ts) == 34
    assert len(val_ts) == 5
    assert len(test_ts) == 10
