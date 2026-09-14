"""Feedback buffer persistence, indexing, and PyG graph integration for active learning retraining."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch
from torch_geometric.data import Data

from src.common.logger import get_logger
from src.module7_hitl.review_protocol import ReviewRecord, ReviewValidator

logger = get_logger("Module7.FeedbackBuffer")


class FeedbackBuffer:
    """Stores, persists, and exposes verified reviewer feedback for retraining."""

    def __init__(self, strategy_name: str = "active", budget_ratio: float = 0.10):
        self.strategy_name = strategy_name
        self.budget_ratio = budget_ratio
        self.records: List[ReviewRecord] = []
        self._tx_id_index: Dict[str, ReviewRecord] = {}

    def __len__(self) -> int:
        return len(self.records)

    def add_record(self, record: ReviewRecord) -> None:
        """Validates and appends a review record to the buffer."""
        ReviewValidator.validate(record)
        if record.tx_id not in self._tx_id_index:
            self.records.append(record)
            self._tx_id_index[record.tx_id] = record

    def add_records(self, records: List[ReviewRecord]) -> None:
        """Batch appends review records."""
        for r in records:
            self.add_record(r)

    def contains_tx(self, tx_id: str) -> bool:
        """Checks if a transaction is in the feedback buffer."""
        return tx_id in self._tx_id_index

    def get_record(self, tx_id: str) -> Optional[ReviewRecord]:
        """Retrieves a review record by tx_id."""
        return self._tx_id_index.get(tx_id)

    def get_tx_ids(self) -> Set[str]:
        """Returns the set of all transaction IDs in the buffer."""
        return set(self._tx_id_index.keys())

    def get_summary(self) -> Dict:
        """Computes summary statistics of feedback items."""
        total = len(self.records)
        illicit_cnt = sum(1 for r in self.records if r.verdict == "ILLICIT")
        licit_cnt = sum(1 for r in self.records if r.verdict == "LICIT")
        inconcl_cnt = sum(1 for r in self.records if r.verdict == "INCONCLUSIVE")
        escal_cnt = sum(1 for r in self.records if r.verdict == "ESCALATED")

        by_timestep = {}
        for r in self.records:
            by_timestep[r.timestep] = by_timestep.get(r.timestep, 0) + 1

        return {
            "strategy": self.strategy_name,
            "budget_ratio": self.budget_ratio,
            "total_feedback_count": total,
            "illicit_count": illicit_cnt,
            "licit_count": licit_cnt,
            "inconclusive_count": inconcl_cnt,
            "escalated_count": escal_cnt,
            "illicit_ratio": round(float(illicit_cnt / total), 4) if total > 0 else 0.0,
            "timestep_distribution": by_timestep,
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Persists feedback buffer to JSON."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "strategy": self.strategy_name,
            "budget_ratio": self.budget_ratio,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "records": [r.to_dict() for r in self.records],
            "summary": self.get_summary(),
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved feedback buffer ({len(self.records)} records) to {p}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "FeedbackBuffer":
        """Loads feedback buffer from JSON."""
        p = Path(filepath)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        buffer = cls(
            strategy_name=data.get("strategy", "unknown"),
            budget_ratio=data.get("budget_ratio", 0.0),
        )
        for r_dict in data["records"]:
            record = ReviewRecord(**r_dict)
            buffer.add_record(record)
        return buffer
