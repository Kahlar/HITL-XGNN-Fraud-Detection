"""Global configuration module for paths, parameters, and environment settings."""

from pathlib import Path
from typing import List, Tuple
from pydantic import BaseModel, Field


class DatasetPaths(BaseModel):
    """File paths for raw and processed datasets."""
    root_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    raw_data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "raw" / "elliptic_bitcoin_dataset")
    processed_data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "processed")
    classes_file: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "raw" / "elliptic_bitcoin_dataset" / "elliptic_txs_classes.csv")
    edgelist_file: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "raw" / "elliptic_bitcoin_dataset" / "elliptic_txs_edgelist.csv")
    features_file: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "raw" / "elliptic_bitcoin_dataset" / "elliptic_txs_features.csv")


class SplitConfig(BaseModel):
    """Temporal split definitions for train, validation, and test sets."""
    train_timesteps: Tuple[int, int] = (1, 34)
    val_timesteps: Tuple[int, int] = (35, 39)
    test_timesteps: Tuple[int, int] = (40, 49)

    @property
    def train_range(self) -> List[int]:
        return list(range(self.train_timesteps[0], self.train_timesteps[1] + 1))

    @property
    def val_range(self) -> List[int]:
        return list(range(self.val_timesteps[0], self.val_timesteps[1] + 1))

    @property
    def test_range(self) -> List[int]:
        return list(range(self.test_timesteps[0], self.test_timesteps[1] + 1))


class ProjectConfig(BaseModel):
    """Master project configuration."""
    paths: DatasetPaths = Field(default_factory=DatasetPaths)
    splits: SplitConfig = Field(default_factory=SplitConfig)
    random_seed: int = 42
    
    # Feature counts
    num_original_local_features: int = 93
    num_original_agg_features: int = 72
    num_original_all_features: int = 165
    num_engineered_features: int = 5


default_config = ProjectConfig()
