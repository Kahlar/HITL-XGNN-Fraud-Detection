"""Multi-Head Graph Attention Network (GAT) with internal attention coefficient extraction."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from src.common.logger import get_logger

logger = get_logger("Module4.GAT")


class GATNet(nn.Module):
    """
    2-Layer Multi-Head Graph Attention Network.
    
    NOTE: GAT attention coefficients are model-internal signals and are not assumed
    to be faithful post-hoc explanations. They are exposed here as model-internal weights
    for comparative benchmarking against post-hoc explainers (such as GNNExplainer in Phase 6).
    """

    def __init__(
        self,
        in_features: int = 165,
        hidden_dim: int = 32,
        num_classes: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        add_self_loops: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.heads = heads
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # Layer 1: multi-head attention
        self.conv1 = GATConv(
            in_channels=in_features,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=add_self_loops,
        )
        self.norm1 = nn.LayerNorm(hidden_dim * heads)

        # Layer 2: final aggregation to num_classes
        self.conv2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=num_classes,
            heads=1,
            concat=False,
            dropout=dropout,
            add_self_loops=add_self_loops,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.
        If return_attention_weights is True, returns (logits, (edge_index_attended, alpha_coefficients)).
        """
        if return_attention_weights:
            h, (edge_idx_1, alpha_1) = self.conv1(x, edge_index, return_attention_weights=True)
            h = self.norm1(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            logits, (edge_idx_2, alpha_2) = self.conv2(h, edge_idx_1, return_attention_weights=True)
            return logits, (edge_idx_1, alpha_1)
        else:
            h = self.conv1(x, edge_index)
            h = self.norm1(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            logits = self.conv2(h, edge_index)
            return logits

    def get_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts Layer 1 multi-head attention coefficients for downstream XAI analysis.
        
        Returns:
            edge_index_attended: torch.LongTensor of shape [2, E_with_loops]
            alpha_weights: torch.FloatTensor of shape [E_with_loops, heads]
        """
        self.eval()
        with torch.no_grad():
            _, (edge_idx, alpha) = self.conv1(x, edge_index, return_attention_weights=True)
        return edge_idx, alpha

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
                "heads": self.heads,
                "dropout": self.dropout,
                "add_self_loops": self.add_self_loops,
                "metadata": metadata or {},
            },
            p,
        )
        logger.info(f"Saved GAT checkpoint to {p}")

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path], device: Optional[str] = None) -> "GATNet":
        """Loads model checkpoint from disk."""
        p = Path(path)
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        model = cls(
            in_features=ckpt["in_features"],
            hidden_dim=ckpt["hidden_dim"],
            num_classes=ckpt["num_classes"],
            heads=ckpt["heads"],
            dropout=ckpt["dropout"],
            add_self_loops=ckpt.get("add_self_loops", True),
        )
        model.load_state_dict(ckpt["state_dict"])
        if device:
            model = model.to(device)
        return model
