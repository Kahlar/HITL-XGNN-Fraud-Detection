"""Explanation stability and reproducibility benchmark across random initialization seeds."""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial.distance import cosine
import torch

from src.common.logger import get_logger
from src.module3_graph_builder.subgraph_extractor import SubgraphData
from src.module5_explainability.gnn_explainer import GNNExplainerEngine

logger = get_logger("Module5.Stability")


@dataclass
class StabilityMetric:
    """Quantitative stability metrics for a target explanation across multiple seeds."""
    tx_id: str
    target_local_idx: int
    seeds_evaluated: List[int]
    mean_edge_jaccard: float
    std_edge_jaccard: float
    mean_feature_cosine_similarity: float
    std_feature_cosine_similarity: float
    top_k_edges: int

    def to_dict(self) -> Dict:
        return asdict(self)


class ExplanationStabilityEvaluator:
    """Measures the reproducibility and perturbation stability of GNNExplainer across multiple seeds."""

    def __init__(self, model: torch.nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device

    def evaluate_node_stability(
        self,
        subgraph: SubgraphData,
        target_local_idx: int,
        timestep: int,
        seeds: Optional[List[int]] = None,
        top_k_edges: int = 10,
        epochs: int = 60,
    ) -> StabilityMetric:
        """
        Evaluates explanation stability for a single target node across multiple random seeds.
        """
        if seeds is None:
            seeds = [42, 123, 999]

        edge_sets: List[set] = []
        feature_vectors: List[np.ndarray] = []

        target_tx_id = subgraph.node_tx_ids[target_local_idx] if subgraph.node_tx_ids else str(target_local_idx)

        for seed in seeds:
            engine = GNNExplainerEngine(
                model=self.model,
                epochs=epochs,
                device=self.device,
                random_seed=seed,
            )
            expl = engine.explain_subgraph(
                subgraph=subgraph,
                target_local_idx=target_local_idx,
                timestep=timestep,
                top_k_edges=top_k_edges,
            )

            # Top-k edge indices
            e_mask = np.array(expl.edge_mask)
            sorted_e = np.argsort(-np.abs(e_mask))[:top_k_edges] if len(e_mask) > 0 else np.array([])
            edge_sets.append(set(sorted_e))

            # Feature vector
            f_vec = np.array(expl.feature_mask)
            feature_vectors.append(f_vec)

        # Pairwise Jaccard similarities
        jaccard_scores: List[float] = []
        cosine_sims: List[float] = []

        num_runs = len(seeds)
        for i in range(num_runs):
            for j in range(i + 1, num_runs):
                set_i = edge_sets[i]
                set_j = edge_sets[j]
                union = len(set_i.union(set_j))
                if union > 0:
                    jaccard = float(len(set_i.intersection(set_j)) / union)
                else:
                    jaccard = 1.0
                jaccard_scores.append(jaccard)

                # Feature cosine similarity
                v_i = feature_vectors[i]
                v_j = feature_vectors[j]
                norm_i = np.linalg.norm(v_i)
                norm_j = np.linalg.norm(v_j)
                if norm_i > 0 and norm_j > 0:
                    cos_sim = float(1.0 - cosine(v_i, v_j))
                else:
                    cos_sim = 1.0
                cosine_sims.append(cos_sim)

        return StabilityMetric(
            tx_id=target_tx_id,
            target_local_idx=target_local_idx,
            seeds_evaluated=seeds,
            mean_edge_jaccard=round(float(np.mean(jaccard_scores)), 4) if jaccard_scores else 1.0,
            std_edge_jaccard=round(float(np.std(jaccard_scores)), 4) if jaccard_scores else 0.0,
            mean_feature_cosine_similarity=round(float(np.mean(cosine_sims)), 4) if cosine_sims else 1.0,
            std_feature_cosine_similarity=round(float(np.std(cosine_sims)), 4) if cosine_sims else 0.0,
            top_k_edges=top_k_edges,
        )
