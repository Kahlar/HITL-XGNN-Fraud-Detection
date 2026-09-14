"""Graph Convolutional Network (GCN) architecture for transaction fraud detection."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from src.common.logger import get_logger

logger = get_logger("Module4.GCN")


class GCNNet(nn.Module):
    """2-Layer Graph Convolutional Network."""

    def __init__(
        self,
        in_features: int = 165,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        num_layers: int = 2,
        add_self_loops: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.num_layers = num_layers
        self.add_self_loops = add_self_loops

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_features, hidden_dim, add_self_loops=add_self_loops))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # Intermediate layers if num_layers > 2
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=add_self_loops))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Output layer
        self.convs.append(GCNConv(hidden_dim, num_classes, add_self_loops=add_self_loops))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning classification logits of shape [N, num_classes].
        """
        h = x
        for i in range(len(self.convs) - 1):
            h = self.convs[i](h, edge_index)
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        logits = self.convs[-1](h, edge_index)
        return logits

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path: Union[str, Path], metadata: Optional[Dict] = None) -> None:
        """Saves model weights and hyperparameters."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "in_features": self.in_features,
                "hidden_dim": self.hidden_dim,
                "num_classes": self.num_classes,
                "dropout": self.dropout,
                "num_layers": self.num_layers,
                "add_self_loops": self.add_self_loops,
                "metadata": metadata or {},
            },
            p,
        )
        logger.info(f"Saved GCN checkpoint to {p}")

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path], device: Optional[str] = None) -> "GCNNet":
        """Loads model checkpoint from disk."""
        p = Path(path)
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        model = cls(
            in_features=ckpt["in_features"],
            hidden_dim=ckpt["hidden_dim"],
            num_classes=ckpt["num_classes"],
            dropout=ckpt["dropout"],
            num_layers=ckpt["num_layers"],
            add_self_loops=ckpt.get("add_self_loops", True),
        )
        model.load_state_dict(ckpt["state_dict"])
        if device:
            model = model.to(device)
        return model
