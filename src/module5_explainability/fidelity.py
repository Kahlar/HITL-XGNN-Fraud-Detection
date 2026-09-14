"""Quantitative fidelity (Fidelity+, Fidelity-) and sparsity benchmark engine for graph explanations."""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch_geometric.data import Data

from src.common.logger import get_logger

logger = get_logger("Module5.Fidelity")


@dataclass
class FidelityResult:
    """Quantitative fidelity and sparsity metrics for an explanation method at a given budget."""
    method: str  # 'gnn_explainer', 'gat_attention', 'random'
    budget_ratio: float  # e.g., 0.10 for top 10%
    num_selected_edges: int
    num_selected_features: int
    edge_sparsity: float
    feature_sparsity: float
    fidelity_plus: float   # p_orig - p_removed
    fidelity_minus: float  # p_orig - p_expl
    original_prob: float
    prob_removed: float
    prob_expl_only: float

    def to_dict(self) -> Dict:
        return asdict(self)


class FidelityEvaluator:
    """Evaluates the faithfulness and sparsity of graph explanations via systematic subgraph perturbation."""

    def __init__(self, model: torch.nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()

    def evaluate_explanation_faithfulness(
        self,
        data: Data,
        target_local_idx: int,
        edge_mask: Union[torch.Tensor, np.ndarray, List[float]],
        feature_mask: Union[torch.Tensor, np.ndarray, List[float]],
        method_name: str = "gnn_explainer",
        budget_ratios: Optional[List[float]] = None,
    ) -> List[FidelityResult]:
        """
        Calculates Fidelity+, Fidelity-, Edge Sparsity, and Feature Sparsity across multiple explanation budgets.

        Operational Definitions:
        - Fidelity+ (Sufficiency): Fid+ = p_orig - p_removed
          (Measures how much model confidence drops when top explanation edges/features are removed).
        - Fidelity- (Necessity):   Fid- = p_orig - p_expl
          (Measures prediction deviation when ONLY top explanation edges/features are retained).
        - Edge Sparsity: 1 - (|E_expl| / |E_subgraph|)
        - Feature Sparsity: 1 - (|F_expl| / |F_total|)
        """
        if budget_ratios is None:
            budget_ratios = [0.05, 0.10, 0.20, 0.30]

        x_orig = data.x.to(self.device)
        edge_index_orig = data.edge_index.to(self.device)

        # Baseline original prediction
        with torch.no_grad():
            logits_orig = self.model(x_orig, edge_index_orig)
            probs_orig = torch.softmax(logits_orig, dim=1)[:, 1].cpu().numpy()
            p_orig = float(probs_orig[target_local_idx])

        # Prepare masks
        e_mask = np.array(edge_mask, dtype=float)
        f_mask = np.array(feature_mask, dtype=float)

        num_edges = edge_index_orig.shape[1]
        num_features = x_orig.shape[1]

        # Rank indices
        sorted_edge_idx = np.argsort(-np.abs(e_mask)) if num_edges > 0 else np.array([], dtype=int)
        sorted_feat_idx = np.argsort(-np.abs(f_mask)) if num_features > 0 else np.array([], dtype=int)

        results: List[FidelityResult] = []

        for b in budget_ratios:
            k_edges = max(1, int(np.ceil(num_edges * b))) if num_edges > 0 else 0
            k_feats = max(1, int(np.ceil(num_features * b))) if num_features > 0 else 0

            top_edge_set = set(sorted_edge_idx[:k_edges])
            top_feat_set = set(sorted_feat_idx[:k_feats])

            edge_sparsity = float(1.0 - (k_edges / num_edges)) if num_edges > 0 else 1.0
            feat_sparsity = float(1.0 - (k_feats / num_features)) if num_features > 0 else 1.0

            # -------------------------------------------------------------
            # 1. PERTURBATION FOR FIDELITY+ (G_removed: Remove Top-k)
            # -------------------------------------------------------------
            if num_edges > 0:
                keep_edge_mask = torch.tensor([i not in top_edge_set for i in range(num_edges)], dtype=torch.bool, device=self.device)
                edge_index_removed = edge_index_orig[:, keep_edge_mask]
            else:
                edge_index_removed = edge_index_orig

            x_removed = x_orig.clone()
            for f_idx in top_feat_set:
                x_removed[:, f_idx] = 0.0  # Zero-out top features

            with torch.no_grad():
                logits_removed = self.model(x_removed, edge_index_removed)
                probs_removed = torch.softmax(logits_removed, dim=1)[:, 1].cpu().numpy()
                p_removed = float(probs_removed[target_local_idx])

            fid_plus = p_orig - p_removed

            # -------------------------------------------------------------
            # 2. PERTURBATION FOR FIDELITY- (G_expl: Retain ONLY Top-k)
            # -------------------------------------------------------------
            if num_edges > 0:
                keep_expl_mask = torch.tensor([i in top_edge_set for i in range(num_edges)], dtype=torch.bool, device=self.device)
                edge_index_expl = edge_index_orig[:, keep_expl_mask]
            else:
                edge_index_expl = edge_index_orig

            x_expl = torch.zeros_like(x_orig)
            for f_idx in top_feat_set:
                x_expl[:, f_idx] = x_orig[:, f_idx]  # Keep only top features

            with torch.no_grad():
                logits_expl = self.model(x_expl, edge_index_expl)
                probs_expl = torch.softmax(logits_expl, dim=1)[:, 1].cpu().numpy()
                p_expl = float(probs_expl[target_local_idx])

            fid_minus = p_orig - p_expl

            results.append(
                FidelityResult(
                    method=method_name,
                    budget_ratio=b,
                    num_selected_edges=k_edges,
                    num_selected_features=k_feats,
                    edge_sparsity=round(edge_sparsity, 4),
                    feature_sparsity=round(feat_sparsity, 4),
                    fidelity_plus=round(float(fid_plus), 4),
                    fidelity_minus=round(float(fid_minus), 4),
                    original_prob=round(p_orig, 4),
                    prob_removed=round(p_removed, 4),
                    prob_expl_only=round(p_expl, 4),
                )
            )

        return results

    def evaluate_random_baseline(
        self,
        data: Data,
        target_local_idx: int,
        budget_ratios: Optional[List[float]] = None,
        random_seed: int = 42,
    ) -> List[FidelityResult]:
        """Generates random edge and feature attribution masks to serve as a baseline for faithfulness benchmarking."""
        np.random.seed(random_seed)
        num_edges = data.edge_index.shape[1]
        num_features = data.x.shape[1]

        rand_edge_mask = np.random.uniform(0.0, 1.0, size=num_edges)
        rand_feat_mask = np.random.uniform(0.0, 1.0, size=num_features)

        return self.evaluate_explanation_faithfulness(
            data=data,
            target_local_idx=target_local_idx,
            edge_mask=rand_edge_mask,
            feature_mask=rand_feat_mask,
            method_name="random",
            budget_ratios=budget_ratios,
        )
