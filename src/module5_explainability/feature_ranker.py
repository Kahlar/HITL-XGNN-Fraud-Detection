"""Feature attribution ranker and dimension partitioner for 165 Elliptic features."""

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple, Union
import numpy as np
import torch


@dataclass
class RankedFeature:
    """Individual feature attribution metadata."""
    feature_index: int
    feature_name: str
    feature_type: str  # 'local' (0..92) or 'aggregated' (93..164)
    raw_importance: float
    normalized_importance: float
    rank: int

    def to_dict(self) -> Dict:
        return asdict(self)


class FeatureRanker:
    """Ranks and partitions feature importance vectors from GNNExplainer."""

    LOCAL_FEATURE_COUNT = 93
    AGGREGATED_FEATURE_COUNT = 72
    TOTAL_FEATURES = 165

    @classmethod
    def get_feature_name(cls, feature_idx: int) -> str:
        """Returns standard positional feature identifier without inventing semantic meanings."""
        return f"feature_{feature_idx:03d}"

    @classmethod
    def get_feature_type(cls, feature_idx: int) -> str:
        """Determines whether a feature is a local transaction metric or 1-hop aggregation."""
        if 0 <= feature_idx < cls.LOCAL_FEATURE_COUNT:
            return "local"
        elif cls.LOCAL_FEATURE_COUNT <= feature_idx < cls.TOTAL_FEATURES:
            return "aggregated"
        else:
            return "unknown"

    @classmethod
    def rank_features(
        cls,
        feature_mask: Union[torch.Tensor, np.ndarray],
        top_k: int = 15,
    ) -> Tuple[List[RankedFeature], Dict[str, float]]:
        """
        Ranks features by importance attribution and computes local vs aggregated contribution ratios.

        Args:
            feature_mask: 1D array of shape [165] with raw attribution weights.
            top_k: Number of top features to return in detailed rank list.

        Returns:
            Tuple of (top_k_ranked_features, summary_stats)
        """
        if isinstance(feature_mask, torch.Tensor):
            weights = feature_mask.detach().cpu().numpy().flatten()
        else:
            weights = np.array(feature_mask).flatten()

        total_sum = float(np.sum(np.abs(weights)))
        if total_sum > 0:
            norm_weights = np.abs(weights) / total_sum
        else:
            norm_weights = np.zeros_like(weights)

        # Sort descending
        sorted_indices = np.argsort(-norm_weights)

        ranked_list: List[RankedFeature] = []
        for rank, idx in enumerate(sorted_indices[:top_k], start=1):
            ranked_list.append(
                RankedFeature(
                    feature_index=int(idx),
                    feature_name=cls.get_feature_name(idx),
                    feature_type=cls.get_feature_type(idx),
                    raw_importance=round(float(weights[idx]), 6),
                    normalized_importance=round(float(norm_weights[idx]), 6),
                    rank=rank,
                )
            )

        # Local vs Aggregated total contributions
        local_mask = norm_weights[:cls.LOCAL_FEATURE_COUNT]
        agg_mask = norm_weights[cls.LOCAL_FEATURE_COUNT:]
        local_ratio = float(np.sum(local_mask))
        agg_ratio = float(np.sum(agg_mask))

        summary = {
            "local_importance_ratio": round(local_ratio, 4),
            "aggregated_importance_ratio": round(agg_ratio, 4),
            "dominant_feature_type": "local" if local_ratio >= agg_ratio else "aggregated",
            "top_1_feature_index": int(sorted_indices[0]),
            "top_1_feature_name": cls.get_feature_name(sorted_indices[0]),
            "top_1_importance": round(float(norm_weights[sorted_indices[0]]), 6),
        }

        return ranked_list, summary
