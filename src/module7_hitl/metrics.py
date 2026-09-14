"""Metrics for evaluating active learning sample efficiency and feedback query quality."""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Union
import numpy as np

from src.common.logger import get_logger
from src.module7_hitl.active_learning import CandidateTransaction

logger = get_logger("Module7.Metrics")


@dataclass
class SampleEfficiencyMetric:
    """Quantitative performance gains normalized per reviewed sample."""
    strategy: str
    budget_ratio: float
    num_samples_reviewed: int
    base_f1: float
    retrained_f1: float
    f1_delta: float
    f1_efficiency_per_1k: float  # (f1_delta / num_samples) * 1000
    base_pr_auc: float
    retrained_pr_auc: float
    pr_auc_delta: float
    pr_auc_efficiency_per_1k: float
    test_precision: float
    test_recall: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FeedbackQueryStats:
    """Statistical properties of transactions selected by an active learning query strategy."""
    strategy: str
    budget_ratio: float
    total_selected: int
    illicit_count: int
    licit_count: int
    illicit_percentage: float
    mean_fraud_prob: float
    mean_uncertainty_margin: float
    mean_entropy_bits: float
    std_entropy_bits: float

    def to_dict(self) -> Dict:
        return asdict(self)


class ActiveLearningMetrics:
    """Calculates sample efficiency and query distribution metrics."""

    @staticmethod
    def compute_sample_efficiency(
        strategy: str,
        budget_ratio: float,
        num_samples: int,
        base_f1: float,
        retrained_f1: float,
        base_pr_auc: float,
        retrained_pr_auc: float,
        test_precision: float,
        test_recall: float,
    ) -> SampleEfficiencyMetric:
        """Calculates performance delta and efficiency normalized per 1,000 feedback transactions."""
        f1_delta = retrained_f1 - base_f1
        pr_auc_delta = retrained_pr_auc - base_pr_auc

        f1_eff = float((f1_delta / max(num_samples, 1)) * 1000.0) if num_samples > 0 else 0.0
        prauc_eff = float((pr_auc_delta / max(num_samples, 1)) * 1000.0) if num_samples > 0 else 0.0

        return SampleEfficiencyMetric(
            strategy=strategy,
            budget_ratio=budget_ratio,
            num_samples_reviewed=num_samples,
            base_f1=round(float(base_f1), 4),
            retrained_f1=round(float(retrained_f1), 4),
            f1_delta=round(float(f1_delta), 4),
            f1_efficiency_per_1k=round(f1_eff, 4),
            base_pr_auc=round(float(base_pr_auc), 4),
            retrained_pr_auc=round(float(retrained_pr_auc), 4),
            pr_auc_delta=round(float(pr_auc_delta), 4),
            pr_auc_efficiency_per_1k=round(prauc_eff, 4),
            test_precision=round(float(test_precision), 4),
            test_recall=round(float(test_recall), 4),
        )

    @staticmethod
    def compute_query_stats(
        strategy: str,
        budget_ratio: float,
        selected_candidates: List[CandidateTransaction],
        decision_threshold: float = 0.5517,
    ) -> FeedbackQueryStats:
        """Computes statistical distribution metrics of query selections."""
        total = len(selected_candidates)
        if total == 0:
            return FeedbackQueryStats(
                strategy=strategy,
                budget_ratio=budget_ratio,
                total_selected=0,
                illicit_count=0,
                licit_count=0,
                illicit_percentage=0.0,
                mean_fraud_prob=0.0,
                mean_uncertainty_margin=0.0,
                mean_entropy_bits=0.0,
                std_entropy_bits=0.0,
            )

        illicit_cnt = sum(1 for c in selected_candidates if c.ground_truth == 1)
        licit_cnt = sum(1 for c in selected_candidates if c.ground_truth == 0)
        illicit_pct = round(float(illicit_cnt / total) * 100.0, 2)

        probs = np.array([c.predicted_prob for c in selected_candidates])
        margins = np.abs(probs - decision_threshold)

        eps = 1e-12
        p_clipped = np.clip(probs, eps, 1.0 - eps)
        entropies = -p_clipped * np.log2(p_clipped) - (1.0 - p_clipped) * np.log2(1.0 - p_clipped)

        return FeedbackQueryStats(
            strategy=strategy,
            budget_ratio=budget_ratio,
            total_selected=total,
            illicit_count=illicit_cnt,
            licit_count=licit_cnt,
            illicit_percentage=illicit_pct,
            mean_fraud_prob=round(float(np.mean(probs)), 4),
            mean_uncertainty_margin=round(float(np.mean(margins)), 4),
            mean_entropy_bits=round(float(np.mean(entropies)), 4),
            std_entropy_bits=round(float(np.std(entropies)), 4),
        )
