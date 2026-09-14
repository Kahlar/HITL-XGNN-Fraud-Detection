"""Memory-efficient loader for the raw Elliptic Bitcoin Dataset files."""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.common.config import default_config
from src.common.logger import get_logger

logger = get_logger("Module1.DataLoader")


class EllipticDataLoader:
    """Memory-efficient loader for Elliptic Bitcoin transaction graph files."""

    def __init__(self, raw_data_dir: Optional[Union[str, Path]] = None):
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else default_config.paths.raw_data_dir
        self.classes_file = self.raw_data_dir / "elliptic_txs_classes.csv"
        self.edgelist_file = self.raw_data_dir / "elliptic_txs_edgelist.csv"
        self.features_file = self.raw_data_dir / "elliptic_txs_features.csv"

    def check_files_exist(self) -> bool:
        """Verifies that all three required raw CSV files exist."""
        for name, p in [
            ("classes", self.classes_file),
            ("edgelist", self.edgelist_file),
            ("features", self.features_file),
        ]:
            if not p.exists():
                logger.error(f"Missing required file: {p} ({name})")
                return False
        return True

    def load_classes(self) -> pd.DataFrame:
        """Loads transaction classes DataFrame with columns ['txId', 'class']."""
        if not self.classes_file.exists():
            raise FileNotFoundError(f"Classes file not found at {self.classes_file}")
        
        logger.info(f"Loading classes from {self.classes_file}...")
        df = pd.read_csv(
            self.classes_file,
            dtype={"txId": "string", "class": "string"},
            engine="c"
        )
        return df

    def load_edges(self) -> pd.DataFrame:
        """Loads payment flow edgelist DataFrame with columns ['txId1', 'txId2']."""
        if not self.edgelist_file.exists():
            raise FileNotFoundError(f"Edgelist file not found at {self.edgelist_file}")
        
        logger.info(f"Loading edgelist from {self.edgelist_file}...")
        df = pd.read_csv(
            self.edgelist_file,
            dtype={"txId1": "string", "txId2": "string"},
            engine="c"
        )
        return df

    def load_features(self, use_float32: bool = True) -> pd.DataFrame:
        """
        Loads transaction features DataFrame.
        Column 0: txId (string)
        Column 1: time_step (int32)
        Columns 2..166: feat_0..feat_164 (float32 if use_float32 else float64)
        """
        if not self.features_file.exists():
            raise FileNotFoundError(f"Features file not found at {self.features_file}")
        
        logger.info(f"Loading features from {self.features_file} (use_float32={use_float32})...")
        
        # Build column names: txId, time_step, feat_0 .. feat_164
        col_names = ["txId", "time_step"] + [f"feat_{i}" for i in range(165)]
        
        float_type = np.float32 if use_float32 else np.float64
        dtype_dict = {"txId": "string", "time_step": np.int32}
        for i in range(165):
            dtype_dict[f"feat_{i}"] = float_type

        df = pd.read_csv(
            self.features_file,
            header=None,
            names=col_names,
            dtype=dtype_dict,
            engine="c"
        )
        logger.info(f"Loaded features: shape={df.shape}, memory_usage={df.memory_usage().sum() / (1024*1024):.1f} MB")
        return df

    def load_transaction_timesteps(self) -> pd.DataFrame:
        """Loads only txId and time_step without loading all 165 feature columns into memory."""
        if not self.features_file.exists():
            raise FileNotFoundError(f"Features file not found at {self.features_file}")
        
        logger.info(f"Loading transaction timesteps only from {self.features_file}...")
        df = pd.read_csv(
            self.features_file,
            header=None,
            usecols=[0, 1],
            names=["txId", "time_step"],
            dtype={"txId": "string", "time_step": np.int32},
            engine="c"
        )
        return df

    def load_all(self, load_features_matrix: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """Loads classes, edges, and optionally features."""
        if not self.check_files_exist():
            raise FileNotFoundError("One or more raw dataset files are missing.")
        
        df_classes = self.load_classes()
        df_edges = self.load_edges()
        df_features = self.load_features() if load_features_matrix else None
        return df_classes, df_edges, df_features
