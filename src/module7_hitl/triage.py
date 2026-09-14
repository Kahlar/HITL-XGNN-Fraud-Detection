"""Triage engine for routing uncertain and high-risk transactions to analyst review queues."""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Union, Tuple
import numpy as np

from src.common.logger import get_logger

logger = get_logger("Module7.Triage")


@dataclass
class TriageItem:
    """Represents an individual transaction evaluated and prioritized for review."""
    tx_id: str
    timestep: int
    global_node_index: int
    predicted_prob: float
    predicted_class: int
    risk_level: str          # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    uncertainty_score: float  # Normalized [0, 1] where 1.0 is maximal uncertainty around decision threshold
    entropy: float            # Binary Shannon entropy in bits
    priority_score: float     # Composite priority score [0, 1]
    reason: str               # Human-readable triage rationale

    def to_dict(self) -> Dict:
        return asdict(self)


class TriageRouter:
    """Routes model predictions into risk tiers and prioritized reviewer triage queues."""

    def __init__(
        self,
        decision_threshold: float = 0.5517,
        critical_risk_thresh: float = 0.85,
        high_risk_thresh: float = 0.70,
        medium_risk_thresh: float = 0.40,
        uncertainty_weight: float = 0.45,
        fraud_risk_weight: float = 0.35,
        diversity_weight: float = 0.20,
    ):
        self.decision_threshold = decision_threshold
        self.critical_risk_thresh = critical_risk_thresh
        self.high_risk_thresh = high_risk_thresh
        self.medium_risk_thresh = medium_risk_thresh

        self.w_uncertainty = uncertainty_weight
        self.w_risk = fraud_risk_weight
        self.w_diversity = diversity_weight

    def classify_risk(self, prob: float) -> str:
        """Categorizes transaction fraud probability into risk tiers."""
        if prob >= self.critical_risk_thresh:
            return "CRITICAL"
        elif prob >= self.high_risk_thresh:
            return "HIGH"
        elif prob >= self.medium_risk_thresh:
            return "MEDIUM"
        else:
            return "LOW"

    @staticmethod
    def compute_entropy(prob: float) -> float:
        """Calculates binary Shannon entropy in bits H(p) = -p log2(p) - (1-p) log2(1-p)."""
        eps = 1e-12
        p = np.clip(prob, eps, 1.0 - eps)
        h = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
        return float(h)

    def compute_margin_uncertainty(self, prob: float) -> float:
        """
        Calculates normalized margin uncertainty relative to the decision threshold tau*.
        Uncertainty = 1.0 - (|p - tau*| / max_dist), bounded in [0, 1].
        """
        max_dist = max(self.decision_threshold, 1.0 - self.decision_threshold)
        dist = abs(prob - self.decision_threshold)
        unc = 1.0 - (dist / max(max_dist, 1e-6))
        return float(np.clip(unc, 0.0, 1.0))

    def compute_priority_score(
        self,
        prob: float,
        uncertainty_score: float,
        diversity_score: float = 0.5,
    ) -> Tuple[float, str]:
        """
        Calculates a composite triage priority score:
        Priority = w1 * Uncertainty + w2 * FraudRisk + w3 * Diversity
        """
        # Risk score: raw probability of fraud
        risk_score = float(np.clip(prob, 0.0, 1.0))
        priority = (
            self.w_uncertainty * uncertainty_score
            + self.w_risk * risk_score
            + self.w_diversity * diversity_score
        )
        priority = float(np.clip(priority, 0.0, 1.0))

        # Generate interpretability reason string
        if prob >= self.critical_risk_thresh:
            reason = f"Critical illicit risk (P={prob:.3f} >= {self.critical_risk_thresh:.2f})"
        elif uncertainty_score >= 0.80:
            reason = f"High model decision boundary uncertainty (|P - tau*| = {abs(prob - self.decision_threshold):.3f})"
        elif prob >= self.high_risk_thresh:
            reason = f"High illicit risk tier (P={prob:.3f})"
        elif prob >= self.medium_risk_thresh:
            reason = f"Moderate suspicion tier (P={prob:.3f})"
        else:
            reason = "Standard baseline transaction"

        return round(priority, 4), reason

    def triage_candidate(
        self,
        tx_id: str,
        timestep: int,
        global_node_index: int,
        predicted_prob: float,
        diversity_score: float = 0.5,
    ) -> TriageItem:
        """Triages a single candidate transaction."""
        risk_level = self.classify_risk(predicted_prob)
        unc_score = self.compute_margin_uncertainty(predicted_prob)
        entropy_val = self.compute_entropy(predicted_prob)
        priority_val, reason_str = self.compute_priority_score(predicted_prob, unc_score, diversity_score)
        pred_class = int(predicted_prob >= self.decision_threshold)

        return TriageItem(
            tx_id=tx_id,
            timestep=timestep,
            global_node_index=global_node_index,
            predicted_prob=round(float(predicted_prob), 4),
            predicted_class=pred_class,
            risk_level=risk_level,
            uncertainty_score=round(float(unc_score), 4),
            entropy=round(float(entropy_val), 4),
            priority_score=priority_val,
            reason=reason_str,
        )

    def route_queue(self, candidates: List[Dict]) -> List[TriageItem]:
        """
        Processes and ranks a list of candidate dictionaries into a sorted triage queue.
        Sorted in descending order of composite priority score.
        """
        items: List[TriageItem] = []
        for c in candidates:
            item = self.triage_candidate(
                tx_id=c["tx_id"],
                timestep=c["timestep"],
                global_node_index=c.get("global_node_index", 0),
                predicted_prob=c["predicted_prob"],
                diversity_score=c.get("diversity_score", 0.5),
            )
            items.append(item)

        items.sort(key=lambda x: x.priority_score, reverse=True)
        return items
