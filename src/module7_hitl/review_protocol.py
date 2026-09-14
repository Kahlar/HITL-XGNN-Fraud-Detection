"""Reviewer submission protocol, schema validation, and simulated oracle review engine."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

from src.common.logger import get_logger

logger = get_logger("Module7.ReviewProtocol")


@dataclass
class ReviewRecord:
    """Standardized schema for human/simulated reviewer feedback submissions."""
    analyst_id: str
    tx_id: str
    timestep: int
    global_node_index: int
    verdict: str               # 'ILLICIT', 'LICIT', 'ESCALATED', 'INCONCLUSIVE'
    confidence: int            # Integer between 1 and 5 (1=Very Uncertain, 5=Absolute Certainty)
    rationale: str             # Case notes or rule trigger justification
    model_predicted_score: float
    explanation_viewed: bool
    reviewed_at: str           # ISO 8601 UTC timestamp
    feedback_source: str       # 'SIMULATED_ORACLE' or 'HUMAN_ANALYST'

    def to_dict(self) -> Dict:
        return asdict(self)


class ReviewValidator:
    """Validates submitted feedback records against compliance and integrity rules."""

    VALID_VERDICTS = {"ILLICIT", "LICIT", "ESCALATED", "INCONCLUSIVE"}
    VALID_SOURCES = {"SIMULATED_ORACLE", "HUMAN_ANALYST"}

    @classmethod
    def validate(cls, record: ReviewRecord) -> None:
        """Validates a review record, raising ValueError if invalid."""
        if not record.analyst_id or not isinstance(record.analyst_id, str):
            raise ValueError("analyst_id is required and must be a non-empty string.")

        if not record.tx_id or not isinstance(record.tx_id, str):
            raise ValueError("tx_id is required and must be a non-empty string.")

        if record.verdict not in cls.VALID_VERDICTS:
            raise ValueError(f"Invalid verdict '{record.verdict}'. Must be one of: {cls.VALID_VERDICTS}")

        if not isinstance(record.confidence, int) or not (1 <= record.confidence <= 5):
            raise ValueError(f"Confidence must be an integer between 1 and 5 (got {record.confidence}).")

        if record.feedback_source not in cls.VALID_SOURCES:
            raise ValueError(f"Invalid feedback_source '{record.feedback_source}'. Must be in: {cls.VALID_SOURCES}")


class SimulatedReviewer:
    """
    Simulates qualified financial-fraud investigator verdicts using ground-truth dataset labels as an oracle.

    Disclaimer:
    Ground-truth labels are used as an oracle solely for offline HITL simulation and active-learning
    reproducibility experiments. No real analyst feedback is claimed for offline benchmarks.
    """

    def __init__(self, analyst_id: str = "SIMULATED_ORACLE_01"):
        self.analyst_id = analyst_id

    def review_transaction(
        self,
        tx_id: str,
        timestep: int,
        global_node_index: int,
        ground_truth_label: int,
        model_score: float,
        explanation_viewed: bool = True,
    ) -> ReviewRecord:
        """
        Simulates an expert review verdict based on oracle ground truth.

        Mapping:
        - Class 1 -> 'ILLICIT' (Confidence 5)
        - Class 0 -> 'LICIT'    (Confidence 5)
        - Class -1 -> 'INCONCLUSIVE' (Confidence 1)
        """
        if ground_truth_label == 1:
            verdict = "ILLICIT"
            confidence = 5
            rationale = "Oracle ground truth: confirmed illicit financial laundering transaction."
        elif ground_truth_label == 0:
            verdict = "LICIT"
            confidence = 5
            rationale = "Oracle ground truth: verified legitimate commercial transaction."
        else:
            verdict = "INCONCLUSIVE"
            confidence = 1
            rationale = "Unlabeled transaction in raw dataset; inconclusive evidence."

        record = ReviewRecord(
            analyst_id=self.analyst_id,
            tx_id=tx_id,
            timestep=timestep,
            global_node_index=global_node_index,
            verdict=verdict,
            confidence=confidence,
            rationale=rationale,
            model_predicted_score=round(float(model_score), 4),
            explanation_viewed=explanation_viewed,
            reviewed_at=datetime.now(timezone.utc).isoformat(),
            feedback_source="SIMULATED_ORACLE",
        )

        ReviewValidator.validate(record)
        return record
