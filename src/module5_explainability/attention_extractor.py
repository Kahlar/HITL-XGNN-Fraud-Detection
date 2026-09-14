"""Extraction and comparative rank correlation for GAT model-internal attention coefficients."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.stats import spearmanr
import torch
from torch_geometric.data import Data

from src.common.logger import get_logger
from src.module3_graph_builder.subgraph_extractor import SubgraphData
from src.module4_gnn.models.gat import GATNet

logger = get_logger("Module5.AttentionExtractor")


@dataclass
class GATAttentionEdge:
    """Individual edge attention weight from GAT layer 1."""
    source_local_idx: int
    target_local_idx: int
    mean_attention: float
    head_attentions: List[float]
    rank: int

    def to_dict(self) -> Dict:
        return asdict(self)


class GATAttentionExtractor:
    """Extracts model-internal attention coefficients from frozen GAT models and benchmarks against GNNExplainer."""

    def __init__(
        self,
        gat_checkpoint: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gat_checkpoint = Path(gat_checkpoint) if gat_checkpoint else Path("models/gnn/gat_best.pt")
        self.gat_model: Optional[GATNet] = None

        if self.gat_checkpoint.exists():
            logger.info(f"Loading frozen GAT checkpoint from {self.gat_checkpoint}...")
            self.gat_model = GATNet.load_checkpoint(self.gat_checkpoint, device=str(self.device))
            self.gat_model.eval()
        else:
            logger.warning(f"GAT checkpoint not found at {self.gat_checkpoint}. GAT comparison will be unavailable.")

    @property
    def is_available(self) -> bool:
        return self.gat_model is not None

    def extract_subgraph_attention(
        self,
        subgraph: SubgraphData,
    ) -> List[GATAttentionEdge]:
        """
        Extracts layer 1 multi-head attention coefficients for edges in a 2-hop computational subgraph.

        Note: GAT attention coefficients are model-internal signals and are not assumed to be faithful post-hoc explanations.
        """
        if not self.is_available:
            raise RuntimeError("GAT model is not available for attention extraction.")

        data = subgraph.subgraph
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)

        with torch.no_grad():
            edge_index_att, alpha = self.gat_model.get_attention_weights(x, edge_index)

        # alpha shape: [E_loops, num_heads]
        alpha_np = alpha.detach().cpu().numpy()
        edge_idx_np = edge_index_att.detach().cpu().numpy()

        # Average across the 4 attention heads
        mean_alpha = np.mean(alpha_np, axis=1)

        # Sort descending
        sorted_indices = np.argsort(-mean_alpha)

        ranked_attention_edges: List[GATAttentionEdge] = []
        for rank, idx in enumerate(sorted_indices, start=1):
            src = int(edge_idx_np[0, idx])
            dst = int(edge_idx_np[1, idx])
            head_vals = [round(float(v), 6) for v in alpha_np[idx]]

            ranked_attention_edges.append(
                GATAttentionEdge(
                    source_local_idx=src,
                    target_local_idx=dst,
                    mean_attention=round(float(mean_alpha[idx]), 6),
                    head_attentions=head_vals,
                    rank=rank,
                )
            )

        return ranked_attention_edges

    @staticmethod
    def compute_agreement(
        gnn_edge_mask: List[float],
        gat_attention_edges: List[GATAttentionEdge],
        subgraph_edge_index: torch.Tensor,
        top_k: int = 10,
    ) -> Dict[str, float]:
        """
        Computes rank correlation and top-k edge overlap (Jaccard) between GNNExplainer and GAT attention.
        """
        num_edges = len(gnn_edge_mask)
        if num_edges == 0 or len(gat_attention_edges) == 0:
            return {
                "spearman_rho": 0.0,
                "spearman_pvalue": 1.0,
                "top_k_jaccard_overlap": 0.0,
            }

        # Align GAT attention to the subgraph edge ordering
        edge_index_np = subgraph_edge_index.cpu().numpy()
        gat_dict = {(e.source_local_idx, e.target_local_idx): e.mean_attention for e in gat_attention_edges}

        aligned_gat_weights = []
        for i in range(num_edges):
            src, dst = int(edge_index_np[0, i]), int(edge_index_np[1, i])
            aligned_gat_weights.append(gat_dict.get((src, dst), 0.0))

        # Spearman rank correlation
        rho, p_val = spearmanr(gnn_edge_mask, aligned_gat_weights)
        rho_val = float(rho) if not np.isnan(rho) else 0.0
        pval_val = float(p_val) if not np.isnan(p_val) else 1.0

        # Top-k Jaccard similarity
        gnn_top_k = set(np.argsort(-np.array(gnn_edge_mask))[:top_k])
        gat_top_k = set(np.argsort(-np.array(aligned_gat_weights))[:top_k])

        intersection = len(gnn_top_k.intersection(gat_top_k))
        union = len(gnn_top_k.union(gat_top_k))
        jaccard = float(intersection / union) if union > 0 else 1.0

        return {
            "spearman_rho": round(rho_val, 4),
            "spearman_pvalue": round(pval_val, 4),
            "top_k_jaccard_overlap": round(jaccard, 4),
        }
