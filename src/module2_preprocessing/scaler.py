"""Train-only feature scaler ensuring zero leakage across temporal splits."""

from pathlib import Path
from typing import List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.common.config import default_config
from src.common.logger import get_logger

logger = get_logger("Module2.TrainOnlyScaler")


class TrainOnlyScaler:
    """
    Fits feature scalers strictly on training timesteps (e.g. t <= 34)
    and applies the learned transform inductively to validation and test partitions.
    """

    def __init__(self, scaler_type: str = "robust", output_dir: Optional[Union[str, Path]] = None):
        self.scaler_type = scaler_type.lower()
        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
        elif self.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler type '{scaler_type}'. Expected 'robust' or 'standard'.")
        
        self.output_dir = Path(output_dir) if output_dir else default_config.paths.processed_data_dir
        self.is_fitted = False
        self.fitted_timesteps: List[int] = []

    def fit(self, X_train: np.ndarray, train_timesteps: Optional[List[int]] = None) -> "TrainOnlyScaler":
        """Fits the scaler strictly on the training feature matrix."""
        logger.info(f"Fitting {self.scaler_type} scaler on training data with shape {X_train.shape}...")
        self.scaler.fit(X_train)
        self.is_fitted = True
        self.fitted_timesteps = train_timesteps or default_config.splits.train_range
        logger.info(f"Scaler successfully fitted on timesteps {min(self.fitted_timesteps)}..{max(self.fitted_timesteps)}.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms a feature matrix using the fitted parameters."""
        if not self.is_fitted:
            raise RuntimeError("Cannot transform before fitting the scaler on training data.")
        return self.scaler.transform(X).astype(np.float32)

    def fit_transform(self, X_train: np.ndarray, train_timesteps: Optional[List[int]] = None) -> np.ndarray:
        """Fits strictly on train and returns transformed train array."""
        self.fit(X_train, train_timesteps)
        return self.transform(X_train)

    def save(self, filename: str = "feature_scaler.joblib") -> Path:
        """Persists fitted scaler artifact to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted scaler.")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.output_dir / filename
        joblib.dump(
            {
                "scaler": self.scaler,
                "scaler_type": self.scaler_type,
                "fitted_timesteps": self.fitted_timesteps,
                "is_fitted": self.is_fitted,
            },
            save_path
        )
        logger.info(f"Saved fitted scaler to {save_path}")
        return save_path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TrainOnlyScaler":
        """Loads a persisted scaler from disk."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Scaler file not found at {p}")
        
        data = joblib.load(p)
        instance = cls(scaler_type=data["scaler_type"])
        instance.scaler = data["scaler"]
        instance.fitted_timesteps = data["fitted_timesteps"]
        instance.is_fitted = data["is_fitted"]
        logger.info(f"Loaded scaler from {p} (fitted on timesteps {min(instance.fitted_timesteps)}..{max(instance.fitted_timesteps)})")
        return instance
