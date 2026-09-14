"""PyTorch Geometric temporal graph builder for Elliptic Bitcoin transactions."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.common.config import default_config
from src.common.logger import get_logger
from src.module1_dataset.loader import EllipticDataLoader
from src.module2_preprocessing.cleaner import LabelCleaner
from src.module2_preprocessing.feature_engineering import (
    ENGINEERED_FEATURE_COLUMNS,
    GraphFeatureEngineer,
)
from src.module2_preprocessing.scaler import TrainOnlyScaler
from src.module2_preprocessing.splitter import TemporalSplitter

logger = get_logger("Module3.PyGBuilder")


class EllipticGraphBuilder:
    """Constructs independent PyTorch Geometric Data graphs for each discrete timestep."""

    def __init__(
        self,
        raw_data_dir: Optional[Union[str, Path]] = None,
        processed_data_dir: Optional[Union[str, Path]] = None,
    ):
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else default_config.paths.raw_data_dir
        self.processed_data_dir = Path(processed_data_dir) if processed_data_dir else default_config.paths.processed_data_dir
        self.graphs_dir = self.processed_data_dir / "graphs"
        self.loader = EllipticDataLoader(self.raw_data_dir)
        self.splitter = TemporalSplitter()

    def build_timestep_graph(
        self,
        time_step: int,
        df_nodes: pd.DataFrame,
        df_edges: pd.DataFrame,
        df_features: pd.DataFrame,
        df_engineered: pd.DataFrame,
        scaler: Optional[TrainOnlyScaler] = None,
    ) -> Data:
        """
        Constructs a single PyG Data graph for timestep t.
        
        Node Features:
        - Columns 0..92   (93): Original local features
        - Columns 93..164 (72): Original aggregated features
        - Columns 0..164 (165): Original all features
        - Columns 165..169 (5): Engineered graph flow features
        Total width: 170 columns (combined)
        """
        # Filter for current timestep
        ts_nodes = df_nodes[df_nodes["time_step"] == time_step].copy().reset_index(drop=True)
        num_nodes = len(ts_nodes)
        if num_nodes == 0:
            raise ValueError(f"No nodes found for timestep {time_step}.")

        tx_ids = ts_nodes["txId"].tolist()
        tx_to_idx = {tx_id: i for i, tx_id in enumerate(tx_ids)}
        idx_to_tx = tx_ids

        # Build feature matrices
        ts_features = df_features[df_features["time_step"] == time_step].copy()
        # Ensure row alignment with ts_nodes
        ts_features = ts_features.set_index("txId").loc[tx_ids].reset_index()
        
        ts_engineered = df_engineered.set_index("txId").loc[tx_ids].reset_index()

        orig_cols = [f"feat_{i}" for i in range(165)]
        orig_mat = ts_features[orig_cols].to_numpy(dtype=np.float32)
        eng_mat = ts_engineered[ENGINEERED_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        
        # Raw combined matrix: 165 + 5 = 170
        raw_combined = np.hstack([orig_mat, eng_mat])

        # Apply train-fitted scaler if provided
        if scaler is not None and scaler.is_fitted:
            x_scaled = scaler.transform(raw_combined)
        else:
            x_scaled = raw_combined

        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        x_raw_tensor = torch.tensor(raw_combined, dtype=torch.float32)

        # Labels: 1 = illicit, 0 = licit, -1 = unknown
        labels = ts_nodes["label"].to_numpy(dtype=np.int64)
        y_tensor = torch.tensor(labels, dtype=torch.long)

        # Edge Index construction
        # Filter edges where both source and target are in this timestep
        ts_node_set = set(tx_ids)
        # Fast filter on string txIds
        valid_edges_mask = df_edges["txId1"].isin(ts_node_set) & df_edges["txId2"].isin(ts_node_set)
        ts_edges = df_edges[valid_edges_mask]

        if len(ts_edges) > 0:
            src_indices = [tx_to_idx[u] for u in ts_edges["txId1"]]
            dst_indices = [tx_to_idx[v] for v in ts_edges["txId2"]]
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Validate self-loops and edge ranges
        if edge_index.numel() > 0:
            self_loops = (edge_index[0] == edge_index[1]).sum().item()
            if self_loops > 0:
                logger.warning(f"Timestep {time_step}: Found {self_loops} self-loops in edge_index.")
            if (edge_index < 0).any() or (edge_index >= num_nodes).any():
                raise ValueError(f"Timestep {time_step}: Edge indices exceed node bounds [0, {num_nodes-1}]!")

        split_name = self.splitter.assign_split(time_step)
        
        # Boolean masks
        train_mask = torch.tensor(ts_nodes["split"] == "train", dtype=torch.bool)
        val_mask = torch.tensor(ts_nodes["split"] == "val", dtype=torch.bool)
        test_mask = torch.tensor(ts_nodes["split"] == "test", dtype=torch.bool)
        labeled_mask = torch.tensor(ts_nodes["is_labeled"].to_numpy(), dtype=torch.bool)

        # PyG Data object
        data = Data(
            x=x_tensor,
            x_raw=x_raw_tensor,
            edge_index=edge_index,
            y=y_tensor,
            tx_id=tx_ids,
            time_step=time_step,
            split=split_name,
            num_nodes=num_nodes,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            labeled_mask=labeled_mask,
        )

        return data

    def build_and_save_all_graphs(self) -> Dict[int, Path]:
        """
        Builds all 49 timestep graphs and persists them to data/processed/graphs/timestep_XX.pt.
        Returns dictionary mapping timestep -> saved file path.
        """
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Building PyG graphs for all 49 timesteps -> {self.graphs_dir}...")

        # 1. Load dataset components
        df_classes = self.loader.load_classes()
        df_edges = self.loader.load_edges()
        df_timesteps = self.loader.load_transaction_timesteps()
        df_features = self.loader.load_features(use_float32=True)

        # 2. Preprocessing & Label Cleaning
        cleaner = LabelCleaner()
        df_cleaned_labels = cleaner.clean_labels(df_classes)
        df_nodes = pd.merge(df_timesteps, df_cleaned_labels, on="txId", how="inner")

        # 3. Split Assignment
        df_nodes, split_meta = self.splitter.split_and_generate_metadata(df_nodes)

        # 4. Graph Feature Engineering
        engineer = GraphFeatureEngineer()
        df_engineered = engineer.compute_graph_features(df_nodes, df_edges)

        # 5. Fit Scaler strictly on train split (t <= 34)
        mat_combined, combined_cols = engineer.get_feature_matrix(
            df_features=df_features,
            df_engineered=df_engineered,
            config="combined"
        )
        train_mask = (df_nodes["split"] == "train").to_numpy()
        X_train = mat_combined[train_mask]

        scaler = TrainOnlyScaler(scaler_type="robust", output_dir=self.processed_data_dir)
        scaler.fit(X_train, train_timesteps=default_config.splits.train_range)
        scaler.save("feature_scaler.joblib")

        # 6. Construct and save graphs for t = 1..49
        saved_paths: Dict[int, Path] = {}
        total_nodes_built = 0
        total_edges_built = 0

        for ts in range(1, 50):
            data = self.build_timestep_graph(
                time_step=ts,
                df_nodes=df_nodes,
                df_edges=df_edges,
                df_features=df_features,
                df_engineered=df_engineered,
                scaler=scaler,
            )
            
            save_path = self.graphs_dir / f"timestep_{ts:02d}.pt"
            torch.save(data, save_path)
            saved_paths[ts] = save_path
            
            total_nodes_built += data.num_nodes
            total_edges_built += data.edge_index.size(1)

        logger.info(
            f"Successfully built and saved {len(saved_paths)} graphs: "
            f"Total Nodes={total_nodes_built:,}, Total Edges={total_edges_built:,}"
        )

        return saved_paths


class TimestepGraphLoader:
    """Loads saved PyG graph objects and provides helper accessors."""

    def __init__(self, graphs_dir: Optional[Union[str, Path]] = None):
        self.graphs_dir = Path(graphs_dir) if graphs_dir else default_config.paths.processed_data_dir / "graphs"

    def load_graph(self, time_step: int) -> Data:
        """Loads a single timestep PyG Data object."""
        path = self.graphs_dir / f"timestep_{time_step:02d}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Graph for timestep {time_step} not found at {path}")
        return torch.load(path, weights_only=False)

    @staticmethod
    def get_tx_to_idx(data: Data) -> Dict[str, int]:
        """Returns mapping from string txId to node index."""
        return {tx_id: i for i, tx_id in enumerate(data.tx_id)}

    @staticmethod
    def get_idx_to_tx(data: Data) -> List[str]:
        """Returns list of txIds in node index order."""
        return data.tx_id

    @staticmethod
    def get_feature_matrix(data: Data, config: str = "combined", use_raw: bool = False) -> torch.Tensor:
        """
        Extracts feature tensor according to configuration:
        - 'original_all'   : x[:, :165] (165 features)
        - 'original_local' : x[:, :93]  (93 features)
        - 'engineered'     : x[:, 165:] (5 features)
        - 'combined'       : x[:, :]    (170 features)
        """
        tensor = data.x_raw if use_raw else data.x
        if config == "original_all":
            return tensor[:, :165]
        elif config == "original_local":
            return tensor[:, :93]
        elif config == "engineered":
            return tensor[:, 165:]
        elif config == "combined":
            return tensor[:, :170]
        else:
            raise ValueError(f"Unknown feature config '{config}'. Expected 'original_all', 'original_local', 'engineered', or 'combined'.")


def get_graph_tx_ids(data: Data) -> List[str]:
    """
    Safely resolves canonical transaction IDs from a PyG Data object.
    Prioritizes data.tx_id, with backwards-compatible fallback to data.node_tx_ids.
    Raises ValueError if neither exists.
    """
    if hasattr(data, "tx_id") and data.tx_id is not None:
        return data.tx_id
    if hasattr(data, "node_tx_ids") and data.node_tx_ids is not None:
        return data.node_tx_ids
    raise ValueError("Graph Data object does not contain canonical 'tx_id' or 'node_tx_ids'.")

