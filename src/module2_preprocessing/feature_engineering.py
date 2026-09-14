"""System-engineered graph topological features computed from transaction flow edges."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.common.logger import get_logger

logger = get_logger("Module2.FeatureEngineer")

ENGINEERED_FEATURE_COLUMNS = [
    "in_degree",
    "out_degree",
    "total_degree",
    "in_out_ratio",
    "flow_balance",
]


class GraphFeatureEngineer:
    """Computes topological transaction flow features per node."""

    def __init__(self, epsilon: float = 1e-5):
        self.epsilon = epsilon

    def compute_graph_features(
        self,
        df_transactions: pd.DataFrame,
        df_edges: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Computes 5 topological features from the payment flow graph:
        1. in_degree: Number of incoming transactions
        2. out_degree: Number of outgoing transactions
        3. total_degree: in_degree + out_degree
        4. in_out_ratio: (in_degree + eps) / (out_degree + eps)
        5. flow_balance: (in_degree - out_degree) / (in_degree + out_degree + eps)
        
        Guarantees that every txId in df_transactions is present in the output.
        """
        if "txId" not in df_transactions.columns:
            raise ValueError("df_transactions must contain 'txId'.")
        if "txId1" not in df_edges.columns or "txId2" not in df_edges.columns:
            raise ValueError("df_edges must contain 'txId1' and 'txId2'.")

        logger.info(f"Computing graph features for {len(df_transactions)} transactions from {len(df_edges)} edges...")

        # Compute out-degree (sources) and in-degree (targets)
        out_degrees = df_edges["txId1"].value_counts()
        in_degrees = df_edges["txId2"].value_counts()

        # Build feature DataFrame aligned with df_transactions
        df_feat = pd.DataFrame({"txId": df_transactions["txId"]})
        df_feat["in_degree"] = df_feat["txId"].map(in_degrees).fillna(0).astype(np.float32)
        df_feat["out_degree"] = df_feat["txId"].map(out_degrees).fillna(0).astype(np.float32)
        df_feat["total_degree"] = df_feat["in_degree"] + df_feat["out_degree"]
        
        # Non-linear ratio and flow balance
        df_feat["in_out_ratio"] = (df_feat["in_degree"] + self.epsilon) / (df_feat["out_degree"] + self.epsilon)
        df_feat["flow_balance"] = (df_feat["in_degree"] - df_feat["out_degree"]) / (df_feat["total_degree"] + self.epsilon)

        logger.info(
            f"Computed {len(ENGINEERED_FEATURE_COLUMNS)} engineered features. "
            f"Mean total degree: {df_feat['total_degree'].mean():.2f}, "
            f"Max total degree: {df_feat['total_degree'].max():.2f}"
        )
        return df_feat

    @staticmethod
    def get_feature_matrix(
        df_features: pd.DataFrame,
        df_engineered: Optional[pd.DataFrame] = None,
        config: str = "original_all"
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extracts feature matrix for one of the four supported configurations:
        1. 'original_all'   : all 165 raw features (feat_0..feat_164)
        2. 'original_local' : 93 local raw features (feat_0..feat_92)
        3. 'engineered'     : 5 system-engineered graph features
        4. 'combined'       : 165 original + 5 engineered = 170 features
        
        Returns (feature_array: np.ndarray, feature_names: List[str]).
        """
        if config == "original_all":
            cols = [f"feat_{i}" for i in range(165)]
            return df_features[cols].to_numpy(dtype=np.float32), cols

        elif config == "original_local":
            cols = [f"feat_{i}" for i in range(93)]
            return df_features[cols].to_numpy(dtype=np.float32), cols

        elif config == "engineered":
            if df_engineered is None:
                raise ValueError("df_engineered must be provided for 'engineered' config.")
            return df_engineered[ENGINEERED_FEATURE_COLUMNS].to_numpy(dtype=np.float32), ENGINEERED_FEATURE_COLUMNS

        elif config == "combined":
            if df_engineered is None:
                raise ValueError("df_engineered must be provided for 'combined' config.")
            orig_cols = [f"feat_{i}" for i in range(165)]
            orig_mat = df_features[orig_cols].to_numpy(dtype=np.float32)
            eng_mat = df_engineered[ENGINEERED_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            combined_mat = np.hstack([orig_mat, eng_mat])
            return combined_mat, orig_cols + ENGINEERED_FEATURE_COLUMNS

        else:
            raise ValueError(f"Unknown feature config '{config}'. Expected 'original_all', 'original_local', 'engineered', or 'combined'.")
