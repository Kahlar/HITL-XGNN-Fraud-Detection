"""Model architectures package containing non-graph baselines and GNN variants."""

try:
    from src.module4_gnn.models.baselines import (
        LightGBMBaseline,
        MLPBaseline,
        MLPNet,
        RandomForestBaseline,
        XGBoostBaseline,
    )
except ImportError:
    LightGBMBaseline = None  # type: ignore
    MLPBaseline = None  # type: ignore
    MLPNet = None  # type: ignore
    RandomForestBaseline = None  # type: ignore
    XGBoostBaseline = None  # type: ignore
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
