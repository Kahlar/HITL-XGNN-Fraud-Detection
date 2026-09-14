"""Local GNNExplainer implementation for GraphSAGE on 2-hop computational transaction subgraphs."""

from dataclasses import asdict, dataclass
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

from src.common.config import default_config
from src.common.logger import get_logger
from src.module3_graph_builder.subgraph_extractor import SubgraphData, SubgraphExtractor
from src.module5_explainability.feature_ranker import FeatureRanker, RankedFeature

logger = get_logger("Module5.GNNExplainer")


@dataclass
class RankedEdge:
    """Individual edge attribution metadata in directed computational subgraph."""
    source_tx_id: str
    target_tx_id: str
    source_global_idx: int
    target_global_idx: int
    source_local_idx: int
    target_local_idx: int
    edge_index_in_subgraph: int
    raw_importance: float
    normalized_importance: float
    rank: int

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TransactionExplanation:
    """Structured explanation artifact for a single target transaction."""
    tx_id: str
    timestep: int
    global_node_index: int
    local_node_index: int
    prediction_probability: float
    predicted_class: int
    ground_truth: int
    category: str  # 'TP', 'FN', 'FP', 'TN', 'unlabeled'
    period: str    # 'pre_shock', 'shock_period', 'post_shock'
    subgraph_num_nodes: int
    subgraph_num_edges: int
    edge_mask: List[float]
    feature_mask: List[float]
    top_features: List[Dict]
    top_edges: List[Dict]
    feature_summary: Dict
    fidelity_plus: Optional[float] = None
    fidelity_minus: Optional[float] = None
    edge_sparsity: Optional[float] = None
    feature_sparsity: Optional[float] = None
    generation_latency_ms: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


class GNNExplainerEngine:
    """Explains predictions of frozen GNN models using GNNExplainer on 2-hop local subgraphs."""

    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int = 100,
        lr: float = 0.01,
        device: Optional[str] = None,
        random_seed: int = 42,
    ):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=self.epochs, lr=self.lr),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=ModelConfig(
                mode="multiclass_classification",
                task_level="node",
                return_type="raw",
            ),
        )

    def explain_subgraph(
        self,
        subgraph: SubgraphData,
        target_local_idx: int,
        timestep: int,
        threshold: float = 0.5517,
        top_k_features: int = 15,
        top_k_edges: int = 15,
    ) -> TransactionExplanation:
        """
        Generates a local explanation for a target node within its 2-hop computational subgraph.
        """
        start_time = time.perf_counter()

        data = subgraph.subgraph
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        target_idx_tensor = torch.tensor(target_local_idx, dtype=torch.long, device=self.device)

        # Baseline original prediction
        with torch.no_grad():
            logits = self.model(x, edge_index)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            orig_prob = float(probs[target_local_idx])
            pred_class = int(orig_prob >= threshold)

        # Ground truth label
        y_val = int(data.y[target_local_idx].item()) if hasattr(data, "y") and data.y is not None else -1

        # Categorize TP, FN, FP, TN
        if y_val == 1:
            category = "TP" if pred_class == 1 else "FN"
        elif y_val == 0:
            category = "FP" if pred_class == 1 else "TN"
        else:
            category = "unlabeled"

        # Analytical period
        if 40 <= timestep <= 42:
            period = "pre_shock"
        elif 43 <= timestep <= 46:
            period = "shock_period"
        elif 47 <= timestep <= 49:
            period = "post_shock"
        else:
            period = "other"

        # Generate GNNExplainer attribution masks
        explanation = self.explainer(x, edge_index, index=target_idx_tensor)

        # Edge mask: shape [E]
        edge_mask_raw = explanation.edge_mask.detach().cpu().numpy() if explanation.edge_mask is not None else np.zeros(edge_index.shape[1])
        # Feature mask for target node: shape [165]
        if explanation.node_mask is not None:
            feature_mask_raw = explanation.node_mask[target_local_idx].detach().cpu().numpy()
        else:
            feature_mask_raw = np.zeros(x.shape[1])

        # Rank features
        ranked_features, feat_summary = FeatureRanker.rank_features(feature_mask_raw, top_k=top_k_features)

        # Rank edges
        num_edges = edge_index.shape[1]
        edge_sum = float(np.sum(np.abs(edge_mask_raw)))
        norm_edge_mask = np.abs(edge_mask_raw) / edge_sum if edge_sum > 0 else np.zeros_like(edge_mask_raw)
        sorted_edge_indices = np.argsort(-norm_edge_mask) if num_edges > 0 else []

        ranked_edges: List[RankedEdge] = []
        for rank, e_idx in enumerate(sorted_edge_indices[:top_k_edges], start=1):
            src_loc = int(edge_index[0, e_idx].item())
            dst_loc = int(edge_index[1, e_idx].item())
            src_glob = int(subgraph.local_to_global_indices[src_loc])
            dst_glob = int(subgraph.local_to_global_indices[dst_loc])
            src_tx = subgraph.node_tx_ids[src_loc] if subgraph.node_tx_ids else str(src_glob)
            dst_tx = subgraph.node_tx_ids[dst_loc] if subgraph.node_tx_ids else str(dst_glob)

            ranked_edges.append(
                RankedEdge(
                    source_tx_id=src_tx,
                    target_tx_id=dst_tx,
                    source_global_idx=src_glob,
                    target_global_idx=dst_glob,
                    source_local_idx=src_loc,
                    target_local_idx=dst_loc,
                    edge_index_in_subgraph=int(e_idx),
                    raw_importance=round(float(edge_mask_raw[e_idx]), 6),
                    normalized_importance=round(float(norm_edge_mask[e_idx]), 6),
                    rank=rank,
                )
            )

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        return TransactionExplanation(
            tx_id=subgraph.target_tx_id,
            timestep=timestep,
            global_node_index=subgraph.target_global_idx,
            local_node_index=target_local_idx,
            prediction_probability=round(orig_prob, 4),
            predicted_class=pred_class,
            ground_truth=y_val,
            category=category,
            period=period,
            subgraph_num_nodes=data.num_nodes,
            subgraph_num_edges=num_edges,
            edge_mask=[round(float(v), 6) for v in edge_mask_raw],
            feature_mask=[round(float(v), 6) for v in feature_mask_raw],
            top_features=[f.to_dict() for f in ranked_features],
            top_edges=[e.to_dict() for e in ranked_edges],
            feature_summary=feat_summary,
            generation_latency_ms=latency_ms,
        )
