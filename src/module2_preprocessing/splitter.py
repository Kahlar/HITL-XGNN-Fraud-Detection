"""Temporal dataset partitioner and split metadata generator."""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from collections import Counter

import pandas as pd

from src.common.config import default_config
from src.common.logger import get_logger
from src.common.schemas import PartitionStats, SplitMetadata

logger = get_logger("Module2.TemporalSplitter")


class TemporalSplitter:
    """Partitions transactions into chronological train, validation, and test splits."""

    def __init__(
        self,
        train_timesteps: Optional[Tuple[int, int]] = None,
        val_timesteps: Optional[Tuple[int, int]] = None,
        test_timesteps: Optional[Tuple[int, int]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.train_range = list(range(
            (train_timesteps or default_config.splits.train_timesteps)[0],
            (train_timesteps or default_config.splits.train_timesteps)[1] + 1
        ))
        self.val_range = list(range(
            (val_timesteps or default_config.splits.val_timesteps)[0],
            (val_timesteps or default_config.splits.val_timesteps)[1] + 1
        ))
        self.test_range = list(range(
            (test_timesteps or default_config.splits.test_timesteps)[0],
            (test_timesteps or default_config.splits.test_timesteps)[1] + 1
        ))
        self.output_dir = Path(output_dir) if output_dir else default_config.paths.processed_data_dir

    def verify_disjoint_partitions(self) -> bool:
        """Verifies that train, validation, and test timesteps are strictly disjoint."""
        set_train = set(self.train_range)
        set_val = set(self.val_range)
        set_test = set(self.test_range)

        overlap_tv = set_train.intersection(set_val)
        overlap_vt = set_val.intersection(set_test)
        overlap_tt = set_train.intersection(set_test)

        if overlap_tv or overlap_vt or overlap_tt:
            raise ValueError(
                f"Temporal split overlap detected! Train-Val: {overlap_tv}, Val-Test: {overlap_vt}, Train-Test: {overlap_tt}"
            )
        return True

    def assign_split(self, timestep: int) -> str:
        """Maps a single timestep integer to its split partition name."""
        if timestep in self.train_range:
            return "train"
        elif timestep in self.val_range:
            return "val"
        elif timestep in self.test_range:
            return "test"
        else:
            raise ValueError(f"Timestep {timestep} does not belong to any defined split partition.")

    def _compute_partition_stats(
        self,
        name: str,
        df_subset: pd.DataFrame,
        timesteps: List[int]
    ) -> PartitionStats:
        """Computes statistical breakdown for a partition."""
        total_nodes = len(df_subset)
        if total_nodes == 0:
            return PartitionStats(
                partition_name=name,
                timestep_min=min(timesteps) if timesteps else 0,
                timestep_max=max(timesteps) if timesteps else 0,
                timesteps=timesteps,
                total_nodes=0,
                labeled_nodes=0,
                illicit_count=0,
                licit_count=0,
                unknown_count=0,
                pct_labeled=0.0,
                pct_illicit_in_labeled=0.0,
                imbalance_ratio=0.0,
            )

        illicit_cnt = int((df_subset["label"] == 1).sum())
        licit_cnt = int((df_subset["label"] == 0).sum())
        unknown_cnt = int((df_subset["label"] == -1).sum())
        labeled_cnt = illicit_cnt + licit_cnt

        pct_labeled = round((labeled_cnt / total_nodes) * 100, 2)
        pct_illicit_in_labeled = round((illicit_cnt / labeled_cnt) * 100, 2) if labeled_cnt > 0 else 0.0
        imbalance_ratio = round(licit_cnt / illicit_cnt, 2) if illicit_cnt > 0 else 0.0

        return PartitionStats(
            partition_name=name,
            timestep_min=min(timesteps),
            timestep_max=max(timesteps),
            timesteps=timesteps,
            total_nodes=total_nodes,
            labeled_nodes=labeled_cnt,
            illicit_count=illicit_cnt,
            licit_count=licit_cnt,
            unknown_count=unknown_cnt,
            pct_labeled=pct_labeled,
            pct_illicit_in_labeled=pct_illicit_in_labeled,
            imbalance_ratio=imbalance_ratio,
        )

    def split_and_generate_metadata(
        self,
        df_transactions: pd.DataFrame
    ) -> Tuple[pd.DataFrame, SplitMetadata]:
        """
        Takes DataFrame with ['txId', 'time_step', 'label'] and:
        1. Verifies disjointness.
        2. Assigns 'split' column ('train', 'val', 'test').
        3. Computes partition statistics and per-timestep statistics.
        4. Saves data/processed/split_metadata.json.
        """
        self.verify_disjoint_partitions()

        if "txId" not in df_transactions.columns or "time_step" not in df_transactions.columns or "label" not in df_transactions.columns:
            raise ValueError("Input DataFrame must contain 'txId', 'time_step', and 'label'.")

        logger.info("Computing temporal train/val/test splits...")
        df = df_transactions.copy()
        df["split"] = df["time_step"].map(self.assign_split)

        # Check unique node assignment
        tx_counts = df["txId"].value_counts()
        if (tx_counts > 1).any():
            raise ValueError("Duplicate transactions detected in split assignment!")

        # Partition subsets
        df_train = df[df["split"] == "train"]
        df_val = df[df["split"] == "val"]
        df_test = df[df["split"] == "test"]

        train_stats = self._compute_partition_stats("train", df_train, self.train_range)
        val_stats = self._compute_partition_stats("val", df_val, self.val_range)
        test_stats = self._compute_partition_stats("test", df_test, self.test_range)

        # Per-timestep stats
        per_timestep_stats: Dict[int, PartitionStats] = {}
        all_timesteps = sorted(df["time_step"].unique().tolist())
        for ts in all_timesteps:
            df_ts = df[df["time_step"] == ts]
            per_timestep_stats[int(ts)] = self._compute_partition_stats(f"timestep_{ts}", df_ts, [int(ts)])

        metadata = SplitMetadata(
            train=train_stats,
            validation=val_stats,
            test=test_stats,
            disjoint_timesteps_verified=True,
            disjoint_nodes_verified=True,
            total_nodes_across_splits=len(df),
            per_timestep_stats=per_timestep_stats,
        )

        # Save split_metadata.json
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / "split_metadata.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(metadata.model_dump_json(indent=2))
        logger.info(f"Saved split metadata to {json_path}")

        return df, metadata
