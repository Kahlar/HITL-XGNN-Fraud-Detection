"""Domain schemas and Pydantic models for validation, splits, and metadata."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DatasetFileInfo(BaseModel):
    """File metadata for a single dataset file."""
    filename: str
    exists: bool
    size_bytes: int
    size_mb: float
    num_rows: int
    num_columns: int
    header_present: bool
    sample_first_row: List[str]


class ClassDistribution(BaseModel):
    """Class frequency distribution summary."""
    illicit_count: int = Field(..., description="Class 1 count (fraud)")
    licit_count: int = Field(..., description="Class 2 count (legitimate)")
    unknown_count: int = Field(..., description="Unknown class count (unlabeled)")
    total_count: int
    labeled_count: int
    pct_labeled: float
    pct_illicit_in_labeled: float
    imbalance_ratio: float = Field(..., description="Ratio of licit to illicit labeled nodes")


class EdgeStatistics(BaseModel):
    """Graph edge summary."""
    total_edges: int
    unique_sources: int
    unique_targets: int
    unique_nodes: int
    self_loops: int
    intra_timestep_edges: int
    inter_timestep_edges: int
    pct_intra_timestep: float


class DatasetValidationReport(BaseModel):
    """Full dataset validation report artifact schema."""
    validation_passed: bool
    timestamp_utc: str
    files: Dict[str, DatasetFileInfo]
    cross_file_node_count_match: bool
    total_unique_transactions: int
    duplicate_tx_ids_found: int
    missing_or_nan_values_found: int
    non_finite_numeric_values_found: int
    num_timesteps: int
    timestep_range: List[int]
    class_distribution: ClassDistribution
    edge_statistics: EdgeStatistics
    timesteps_node_counts: Dict[int, int]
    timesteps_class_distribution: Dict[int, Dict[str, int]]
    validation_notes: List[str]


class PartitionStats(BaseModel):
    """Statistics for a single temporal partition (train, val, or test)."""
    partition_name: str
    timestep_min: int
    timestep_max: int
    timesteps: List[int]
    total_nodes: int
    labeled_nodes: int
    illicit_count: int
    licit_count: int
    unknown_count: int
    pct_labeled: float
    pct_illicit_in_labeled: float
    imbalance_ratio: float


class SplitMetadata(BaseModel):
    """Full temporal split metadata schema."""
    train: PartitionStats
    validation: PartitionStats
    test: PartitionStats
    disjoint_timesteps_verified: bool
    disjoint_nodes_verified: bool
    total_nodes_across_splits: int
    per_timestep_stats: Dict[int, PartitionStats]


class PreprocessingMetadata(BaseModel):
    """Metadata describing the preprocessing pipeline outputs."""
    num_total_transactions: int
    label_mapping: Dict[str, int]
    original_all_feature_dim: int
    original_local_feature_dim: int
    engineered_feature_dim: int
    combined_feature_dim: int
    scaler_type: str
    scaler_fit_timesteps: List[int]
    saved_artifacts: Dict[str, str]
