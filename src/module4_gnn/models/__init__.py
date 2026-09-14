"""Model architectures package containing non-graph baselines and GNN variants."""

from src.module4_gnn.models.baselines import (
    LightGBMBaseline,
    MLPBaseline,
    MLPNet,
    RandomForestBaseline,
    XGBoostBaseline,
)
from src.module4_gnn.models.gat import GATNet
from src.module4_gnn.models.gcn import GCNNet
from src.module4_gnn.models.graphsage import GraphSAGENet

__all__ = [
    "RandomForestBaseline",
    "XGBoostBaseline",
    "LightGBMBaseline",
    "MLPBaseline",
    "MLPNet",
    "GCNNet",
    "GraphSAGENet",
    "GATNet",
]
