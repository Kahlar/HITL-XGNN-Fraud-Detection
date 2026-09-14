"""Active learning query strategies: Random, Margin Uncertainty, Shannon Entropy, High Risk, and Combined Diversity."""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.common.logger import get_logger
from src.module7_hitl.triage import TriageRouter

logger = get_logger("Module7.ActiveLearning")


@dataclass
class CandidateTransaction:
    """Candidate transaction in the feedback pool available for active querying."""
    tx_id: str
    timestep: int
    global_node_index: int
    predicted_prob: float
    ground_truth: int
    node_degree: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


class ActiveQueryEngine:
    """Executes active learning query strategies on candidate pools."""

    def __init__(self, decision_threshold: float = 0.5517):
        self.decision_threshold = decision_threshold
        self.triage_router = TriageRouter(decision_threshold=decision_threshold)

    def sample_random(
        self,
        candidates: List[CandidateTransaction],
        budget_count: int,
        random_seed: int = 42,
    ) -> List[CandidateTransaction]:
        """Baseline strategy: Selects candidate transactions uniformly at random."""
        np.random.seed(random_seed)
        n = min(budget_count, len(candidates))
        selected_indices = np.random.choice(len(candidates), size=n, replace=False)
        return [candidates[i] for i in selected_indices]

    def sample_uncertainty(
        self,
        candidates: List[CandidateTransaction],
        budget_count: int,
    ) -> List[CandidateTransaction]:
        """
        Margin Uncertainty strategy: Prioritizes transactions closest to the decision threshold tau*.
        Ranked by ascending |p - tau*|.
        """
        sorted_candidates = sorted(
            candidates,
            key=lambda c: abs(c.predicted_prob - self.decision_threshold),
        )
        return sorted_candidates[:budget_count]

    def sample_entropy(
        self,
        candidates: List[CandidateTransaction],
        budget_count: int,
    ) -> List[CandidateTransaction]:
        """
        Shannon Entropy strategy: Prioritizes transactions with maximal prediction entropy.
        Ranked by descending H(p) = -p log2(p) - (1-p) log2(1-p).
        """
        sorted_candidates = sorted(
            candidates,
            key=lambda c: self.triage_router.compute_entropy(c.predicted_prob),
            reverse=True,
        )
        return sorted_candidates[:budget_count]

    def sample_high_risk(
        self,
        candidates: List[CandidateTransaction],
        budget_count: int,
    ) -> List[CandidateTransaction]:
        """
        High Risk strategy: Prioritizes transactions with the highest predicted probability of fraud.
        Ranked by descending P(illicit).
        """
        sorted_candidates = sorted(
            candidates,
            key=lambda c: c.predicted_prob,
            reverse=True,
        )
        return sorted_candidates[:budget_count]

    def sample_combined_active(
        self,
        candidates: List[CandidateTransaction],
        budget_count: int,
        w_uncertainty: float = 0.50,
        w_risk: float = 0.30,
        w_diversity: float = 0.20,
    ) -> List[CandidateTransaction]:
        """
        Combined Active strategy: Balances decision boundary uncertainty, fraud risk tier,
        and graph structural diversity (degree diversity across timesteps).

        Algorithm:
        1. Compute normalized uncertainty: 1 - (|p - tau*| / max_dist)
        2. Compute normalized fraud risk: p
        3. Compute graph diversity score: log(degree + 1) normalized
        4. Composite Score = w_unc * unc + w_risk * risk + w_div * div
        5. Select top-k highest composite score candidates.
        """
        max_dist = max(self.decision_threshold, 1.0 - self.decision_threshold)
        max_deg = max((c.node_degree for c in candidates), default=1)
        max_log_deg = np.log1p(max(max_deg, 1))

        scored_candidates: List[Tuple[float, CandidateTransaction]] = []
        for c in candidates:
            # 1. Uncertainty
            dist = abs(c.predicted_prob - self.decision_threshold)
            unc = 1.0 - (dist / max(max_dist, 1e-6))
            unc = float(np.clip(unc, 0.0, 1.0))

            # 2. Risk
            risk = float(np.clip(c.predicted_prob, 0.0, 1.0))

            # 3. Diversity
            div = float(np.log1p(c.node_degree) / max_log_deg) if max_log_deg > 0 else 0.5

            # Composite Score
            comp_score = w_uncertainty * unc + w_risk * risk + w_diversity * div
            scored_candidates.append((comp_score, c))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored_candidates[:budget_count]]
